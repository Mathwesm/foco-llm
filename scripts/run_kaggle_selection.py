"""Train or evaluate pinned Qwen arithmetic selectors on one Kaggle T4."""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import EXPECTED_CASES, FORMS, checkpoint, score_selection, summarize

from foco_llm.core.arithmetic_selection import selection_prompt_v2
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.models.experiment import GenerationConfig, Problem
from foco_llm.models.inference import InferenceConfig
from foco_llm.models.training import StepResult
from foco_llm.services.adapter_backend import verify_adapter
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.selection_training import SelectionTrainingBackend
from foco_llm.services.training_checkpoints import save_checkpoint, verify_checkpoint
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
REVISION = (
    "aa8e72537993ba99e69dfaafa59ed015b17504d1"  # pragma: allowlist secret -- public model revision
)
MODELS = {
    "3b": (MODEL_ID, REVISION),
    "7b": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",  # pragma: allowlist secret -- public revision
    ),
}
# This is the public SHA-256 of the original generated training pool.
TRAIN_SHA256 = (
    "b142fe60ba9cd6a5a6f95f1a1ac7adfc7fff8717d1ec5789639397966b7d789e"  # pragma: allowlist secret
)
DIAGNOSTIC = Path("reports/2026-10-01/arithmetic-diagnostic/dataset.json")
MIN_STEPS = 2
MAX_STEPS = 1024


def quantization(transformers: Any, torch: Any) -> Any:
    """Use NF4 QLoRA with FP16 compute, supported by the T4."""
    return transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


def require_t4() -> None:
    """Stop before downloading weights when CUDA or VRAM is inadequate."""
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU accelerator")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RuntimeError("The first CUDA device needs at least 14 GiB of VRAM")


def prepare_dataset(destination: Path) -> Path:
    """Recreate the original training pool and reject generator drift."""
    problems = generate_v21(GenerationConfig())
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if digest != TRAIN_SHA256:
        raise ValueError(f"Training dataset changed: expected={TRAIN_SHA256}, actual={digest}")
    path = destination / "training-dataset.json"
    publish_text(path, payload)
    return path


class KaggleSelectionBackend(SelectionTrainingBackend):
    """Reuse the local selection objective with an NF4 frozen backbone."""

    def _base(self) -> Any:
        model = self.transformers.AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            revision=self.config.revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=self.torch.float16,
            quantization_config=quantization(self.transformers, self.torch),
            device_map={"": 0},
            attn_implementation="eager",
        )
        model.config.use_cache = False
        return self.peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    def metadata(self) -> dict[str, str | int]:
        """Identify quantized training separately from the local BF16 pilot."""
        return {
            **super().metadata(),
            "quantization": "nf4-double-quant",
            "compute_dtype": "float16",
            "bitsandbytes": importlib.import_module("bitsandbytes").__version__,
        }


class KaggleInferenceBackend(TransformersBackend):
    """Run baseline or a verified adapter on the same quantized base."""

    def __init__(
        self,
        config: InferenceConfig,
        adapter: Path | None,
        evaluation_dataset: Path,
        training_dataset: Path,
    ) -> None:
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        self.config = config
        self.torch = importlib.import_module("torch")
        self.transformers = importlib.import_module("transformers")
        self.transformers.set_seed(config.seed)
        self.tokenizer = self.transformers.AutoTokenizer.from_pretrained(
            config.model_id, revision=config.revision, trust_remote_code=False
        )
        self.model = self.transformers.AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            use_safetensors=True,
            dtype=self.torch.float16,
            quantization_config=quantization(self.transformers, self.torch),
            device_map={"": 0},
            attn_implementation="eager",
        )
        self.adapter_identity = (
            verify_adapter(adapter, config, evaluation_dataset, training_dataset)
            if adapter is not None
            else {}
        )
        if adapter is not None:
            peft = importlib.import_module("peft")
            self.model = peft.PeftModel.from_pretrained(self.model, adapter, is_trainable=False)
        self.model.eval()

    def metadata(self) -> dict[str, str]:
        """Bind responses to the model, 4-bit mode, and adapter bytes."""
        return {
            **super().metadata(),
            **self.adapter_identity,
            "quantization": "nf4-double-quant",
            "bitsandbytes": str(importlib.import_module("bitsandbytes").__version__),
        }


def latest_sparse_checkpoint(destination: Path, interval: int) -> Path | None:
    """Find the highest complete checkpoint without requiring every update on disk."""
    paths = sorted(destination.glob("step-*"))
    previous = 0
    for path in paths:
        state = verify_checkpoint(path)
        step = len(state.history)
        if path.name != f"step-{step:04d}" or step <= previous:
            raise ValueError("Sparse training checkpoint sequence is inconsistent")
        if step - previous > interval:
            raise ValueError("Sparse training checkpoint interval was exceeded")
        previous = step
    return paths[-1] if paths else None


def train(dataset: Path, config: PilotConfig, output: Path, interval: int, model_size: str) -> Path:
    """Train the evidence selector, saving resumable immutable checkpoints."""
    selected = select_pilot(load_dataset(dataset), config)
    backend = KaggleSelectionBackend(config)
    examples = tuple(backend.encode(problem) for problem in selected)
    manifest = {
        "protocol": f"arithmetic-selection-qlora-{model_size}-v1",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "diagnostic_sha256": hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "example_ids": [problem.id for problem in selected],
        "runtime": backend.metadata(),
        "checkpoint_interval": interval,
        "schedule": "fixed seeded base order, repeated cyclically; batch size one",
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    if (destination / "summary.json").exists():
        logger.info("Completed Kaggle training already exists: {}", destination)
        return destination
    latest = latest_sparse_checkpoint(destination, interval)
    history: list[StepResult] = list(verify_checkpoint(latest).history) if latest else []
    if latest:
        backend.restore(latest)
    initial_loss, _ = backend.probe(examples[0])
    for index in range(len(history), config.steps):
        history.append(backend.step(examples[index % len(examples)], index + 1))
        if (index + 1) % interval == 0 or index + 1 == config.steps:
            save_checkpoint(backend, destination, history)
            logger.info("Kaggle checkpoint: step={}, loss={:.4f}", index + 1, history[-1].loss)
    final_loss, _ = backend.probe(examples[0])
    summary = {
        "experiment_type": "exploratory_larger_model_not_confirmatory_evidence",
        "completed_steps": len(history),
        "unique_training_examples": len(examples),
        "initial_train_probe_loss": initial_loss if not latest else None,
        "final_train_probe_loss": final_loss,
        "training_seconds": sum(result.elapsed_seconds for result in history),
        "peak_reserved_bytes": max(result.peak_reserved_bytes for result in history),
    }
    publish_text(
        destination / "history.json", json.dumps([r.model_dump() for r in history], indent=2)
    )
    publish_text(destination / "summary.json", json.dumps(summary, indent=2))
    return destination


def evaluate(
    dataset: Path,
    training_dataset: Path,
    output: Path,
    adapter: Path | None,
    model_size: str,
) -> Path:
    """Score the frozen 40 examples, retaining unedited model generations."""
    all_cases = TypeAdapter(tuple[Problem, ...]).validate_json(dataset.read_text(encoding="utf-8"))
    cases = tuple(problem for problem in all_cases if problem.id.rsplit(":", 1)[-1] in FORMS)
    if len(cases) != EXPECTED_CASES or len({problem.id for problem in cases}) != EXPECTED_CASES:
        raise ValueError("Expected 40 unique held-out arithmetic selection cases")
    model_id, revision = MODELS[model_size]
    config = InferenceConfig(model_id=model_id, revision=revision, precision="float16")
    backend = KaggleInferenceBackend(config, adapter, dataset, training_dataset)
    manifest = {
        "protocol": f"arithmetic-selection-qlora-{model_size}-evaluation-v1",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "runtime": backend.metadata(),
        "cases": len(cases),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    rows = []
    for index, problem in enumerate(cases, 1):
        prompt = selection_prompt_v2(problem)
        response = checkpoint(prompt, destination, backend).generation
        rows.append({**score_selection(problem, response.text), "raw_response": response.text})
        if index % 10 == 0:
            logger.info("Kaggle evaluation: completed={}, total={}", index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))
    return destination


@logger.catch(reraise=True)
def main() -> None:
    """Run one requested stage without mixing training and evaluation artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "baseline", "train", "adapted"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/foco-output"))
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--checkpoint-every", type=int, default=40)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-size", choices=tuple(MODELS), default="3b")
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    dataset = args.output / "training-dataset.json"
    if args.mode == "prepare":
        logger.info("Training dataset: {}", prepare_dataset(args.output))
        return
    if not dataset.exists():
        raise FileNotFoundError("Run prepare before baseline, train, or adapted")
    require_t4()
    model_id, revision = MODELS[args.model_size]
    if args.mode == "train":
        if not MIN_STEPS <= args.steps <= MAX_STEPS or args.checkpoint_every < 1:
            raise ValueError("Steps must be 2..1024 and checkpoint interval must be positive")
        config_path = Path("configs/arithmetic-selection-2026-10-01.json")
        config = PilotConfig.model_validate_json(config_path.read_text(encoding="utf-8"))
        config = PilotConfig.model_validate(
            {**config.model_dump(), "model_id": model_id, "revision": revision, "steps": args.steps}
        )
        logger.info(
            "Training output: {}",
            train(
                dataset, config, args.output / "training", args.checkpoint_every, args.model_size
            ),
        )
    elif args.mode == "adapted":
        if args.adapter is None:
            raise ValueError("Adapted evaluation requires --adapter")
        logger.info(
            "Adapted evaluation: {}",
            evaluate(DIAGNOSTIC, dataset, args.output / "adapted", args.adapter, args.model_size),
        )
    else:
        logger.info(
            "Baseline evaluation: {}",
            evaluate(DIAGNOSTIC, dataset, args.output / "baseline", None, args.model_size),
        )


if __name__ == "__main__":
    main()
