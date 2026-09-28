"""Optional BF16 LoRA backend for a bounded, restartable functional test.

Any annotations are confined to the optional tensor/tokenizer library boundary;
the lightweight test environment does not install these libraries.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from foco_llm.core.prompts import build_prompt
from foco_llm.core.training_data import IGNORE_INDEX, mask_completion
from foco_llm.models.experiment import Problem
from foco_llm.models.training import SmokeConfig, StepResult, TrainingExample


class TrainingError(RuntimeError):
    """A numerical or parameter-isolation invariant failed."""


class TrainingBackend:
    """Train only query/value LoRA adapters on one CUDA device."""

    def __init__(self, config: SmokeConfig) -> None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.torch = importlib.import_module("torch")
        self.transformers = importlib.import_module("transformers")
        self.peft = importlib.import_module("peft")
        self.transformers.set_seed(config.seed)
        self.torch.use_deterministic_algorithms(True)
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False
        )
        self.model = self.peft.get_peft_model(
            self._base(),
            self.peft.LoraConfig(
                task_type="CAUSAL_LM",
                r=config.rank,
                lora_alpha=config.rank * 2,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.model.enable_input_require_grads()
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        if not parameters or any(
            "lora_" not in n for n, p in self.model.named_parameters() if p.requires_grad
        ):
            raise TrainingError("Only LoRA parameters may be trainable")
        self.optimizer = self.torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=0.0
        )
        self.torch.cuda.reset_peak_memory_stats()

    def _base(self) -> Any:
        model = self.transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=self.torch.bfloat16,
            attn_implementation="eager",
        ).to("cuda")
        model.config.use_cache = False
        return model

    def metadata(self) -> dict[str, str | int]:
        """Record effective software, device, precision, and trainable parameter count."""
        return {
            "torch": str(self.torch.__version__),
            "transformers": self.transformers.__version__,
            "peft": self.peft.__version__,
            "device": self.torch.cuda.get_device_name(0),
            "cuda": str(self.torch.version.cuda),
            "base_dtype": str(self.model.dtype),
            "attention": "eager",
            "gradient_checkpointing": "non-reentrant",
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
        }

    def encode(self, problem: Problem) -> TrainingExample:
        """Keep reference labels only in the assistant completion, never its prompt."""
        messages = [{"role": "user", "content": build_prompt(problem).text}]
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        target = json.dumps({"answer": problem.answer, "evidence": list(problem.evidence)})
        full = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            tokenize=True,
            add_generation_prompt=False,
        )
        return mask_completion(problem.id, tuple(prefix), tuple(full), self.config.max_tokens)

    def _inputs(self, example: TrainingExample) -> dict[str, Any]:
        ids = self.torch.tensor([example.input_ids], device="cuda")
        return {
            "input_ids": ids,
            "attention_mask": self.torch.ones_like(ids),
            "labels": self.torch.tensor([example.labels], device="cuda"),
        }

    def probe(self, example: TrainingExample) -> tuple[float, Any]:
        """Measure training-example loss and last-position logits in evaluation mode."""
        self.model.eval()
        with self.torch.no_grad():
            output = self.model(**self._inputs(example))
        if not self.torch.isfinite(output.loss) or not self.torch.isfinite(output.logits).all():
            raise TrainingError("Non-finite probe loss or logits")
        return float(output.loss), output.logits[0, -1].detach().float().cpu()

    def step(self, example: TrainingExample, index: int) -> StepResult:
        """Perform one finite update, rejecting invalid losses or gradients."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.torch.cuda.synchronize()
        started = perf_counter()
        loss = self.model(**self._inputs(example)).loss
        if not self.torch.isfinite(loss):
            raise TrainingError("Non-finite training loss")
        loss.backward()
        gradients = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
        if not gradients:
            raise TrainingError("No adapter gradients")
        self.torch.nn.utils.clip_grad_norm_(gradients, max_norm=1.0, error_if_nonfinite=True)
        if any(p.grad is not None for p in self.model.parameters() if not p.requires_grad):
            raise TrainingError("Frozen base model received gradients")
        self.optimizer.step()
        if any(not self.torch.isfinite(p).all() for p in gradients):
            raise TrainingError("Non-finite adapter weights after optimizer update")
        self.torch.cuda.synchronize()
        return StepResult(
            step=index,
            example_id=example.id,
            loss=float(loss.detach()),
            elapsed_seconds=perf_counter() - started,
            supervised_tokens=sum(value != IGNORE_INDEX for value in example.labels),
            peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
        )

    def digest(self, *, adapters: bool) -> str:
        """Hash actual parameter bytes to verify base freezing and adapter updates."""
        digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            if ("lora_" in name) == adapters:
                digest.update(name.encode("utf-8"))
                raw = parameter.detach().contiguous().view(self.torch.uint8).cpu()
                digest.update(raw.numpy().tobytes())
        return digest.hexdigest()

    def save(self, path: Path) -> None:
        """Save adapter plus optimizer and generator states inside an unpublished directory."""
        self.model.save_pretrained(path, safe_serialization=True)
        self.torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "cpu_rng": self.torch.get_rng_state(),
                "cuda_rng": self.torch.cuda.get_rng_state_all(),
            },
            path / "state.pt",
        )

    def restore(self, path: Path) -> None:
        """Restore only locally produced, integrity-checked tensor state."""
        tensors = importlib.import_module("safetensors.torch").load_file(
            path / "adapter_model.safetensors"
        )
        self.peft.set_peft_model_state_dict(self.model, tensors)
        state = self.torch.load(path / "state.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.torch.set_rng_state(state["cpu_rng"])
        self.torch.cuda.set_rng_state_all(state["cuda_rng"])

    def reload(self, path: Path, example: TrainingExample, expected: Any) -> float:
        """Free the training model, load a fresh base and adapter, and compare logits."""
        import gc

        del self.optimizer
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
        self.model = self.peft.PeftModel.from_pretrained(self._base(), path, is_trainable=False)
        _, actual = self.probe(example)
        difference = float((actual - expected).abs().max())
        if not self.torch.allclose(actual, expected, rtol=0.0, atol=0.0001):
            raise TrainingError(f"Reload changed probe logits: max_abs_difference={difference}")
        return difference
