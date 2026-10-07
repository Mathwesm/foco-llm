"""Optional Transformers adapter; imports heavyweight libraries only on use."""

from __future__ import annotations

import importlib
import os
from time import perf_counter
from typing import Any

from foco_llm.models.experiment import Prompt
from foco_llm.models.inference import Generation, InferenceConfig


class NumericalInferenceError(RuntimeError):
    """Generation cannot choose a token from invalid numerical scores."""


def check_logits(_input_ids: object, scores: Any) -> Any:
    """Reject NaN, positive infinity, or rows with no finite candidate token.

    The optional tensor-library boundary uses Any to avoid requiring PyTorch in
    the lightweight CI environment. Negative infinity is a valid token mask.
    """
    if scores.isnan().any() or scores.isposinf().any() or not scores.isfinite().any(dim=-1).all():
        raise NumericalInferenceError(
            "Invalid logits detected; aborting instead of emitting tokens"
        )
    return scores


class TransformersBackend:
    """Run a pinned Safetensors model locally without remote Python code."""

    def __init__(self, config: InferenceConfig) -> None:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.config = config
        self.torch = importlib.import_module("torch")
        self.transformers = importlib.import_module("transformers")
        if config.device == "cuda" and not self.torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install the inference group or select CPU")
        self.transformers.set_seed(config.seed)
        self.torch.use_deterministic_algorithms(True)
        self.torch.backends.cuda.matmul.allow_tf32 = False
        self.torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False
        )
        self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=getattr(self.torch, config.precision)
            if config.device == "cuda"
            else self.torch.float32,
            attn_implementation="eager",
        ).to(config.device)
        self.model.eval()

    def metadata(self) -> dict[str, str]:
        """Return software and accelerator provenance for resume compatibility."""
        return {
            "torch": str(self.torch.__version__),
            "transformers": str(self.transformers.__version__),
            "cuda": str(self.torch.version.cuda),
            "device": str(self.torch.cuda.get_device_name(0))
            if self.config.device == "cuda"
            else "cpu",
            "dtype": str(self.model.dtype),
            "attention": "eager",
        }

    def generate(self, prompt: Prompt) -> Generation:
        """Generate one bounded greedy continuation and measure CUDA allocations."""
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt.text}], tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        input_count = int(inputs.input_ids.shape[-1])
        if input_count > self.config.max_input_tokens:
            raise ValueError("Input exceeds configured context limit; truncation is forbidden")
        is_cuda = self.config.device == "cuda"
        if is_cuda:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                logits_processor=[check_logits],
                generation_config=self.transformers.GenerationConfig(
                    do_sample=False,
                    max_new_tokens=self.config.max_new_tokens,
                    max_time=self.config.max_seconds,
                    eos_token_id=self.model.generation_config.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True,
                ),
            )
        if is_cuda:
            self.torch.cuda.synchronize()
        elapsed = perf_counter() - start
        tokens = outputs[0, input_count:]
        count = int(tokens.shape[0])
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        reason = "eos" if count and int(tokens[-1]) in eos_ids else "time_limit"
        if reason != "eos" and count >= self.config.max_new_tokens:
            reason = "token_limit"
        return Generation(
            text=str(self.tokenizer.decode(tokens, skip_special_tokens=True)),
            input_tokens=input_count,
            output_tokens=count,
            elapsed_seconds=elapsed,
            peak_allocated_bytes=int(self.torch.cuda.max_memory_allocated()) if is_cuda else 0,
            peak_reserved_bytes=int(self.torch.cuda.max_memory_reserved()) if is_cuda else 0,
            stop_reason=reason,
        )
