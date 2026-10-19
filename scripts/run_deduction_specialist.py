"""Train a deduction-only 7B selector and evaluate a disjoint frozen holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter
from run_cross_task_transfer import CONDITIONS, TASKS, selection_prompt, validate
from run_kaggle_selection import KaggleSelectionBackend, latest_sparse_checkpoint, require_t4

from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.core.training_data import mask_completion
from foco_llm.models.experiment import GenerationConfig, Problem, Split
from foco_llm.models.training import StepResult, TrainingExample
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.selection_training import selection_target
from foco_llm.services.training_checkpoints import save_checkpoint, verify_checkpoint
from foco_llm.utils.logger import setup_logging

SEED = 20261007
BASES_PER_TASK = 30
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])
CONFIG = Path("configs/deduction-specialist-2026-10-07.json")


class DeductionTrainingBackend(KaggleSelectionBackend):
    """Supervise deduction evidence with the exact transfer evaluation prompt."""

    def encode(self, problem: Problem) -> TrainingExample:
        """Mask the task prompt and train only the JSON evidence completion."""
        messages = [{"role": "user", "content": selection_prompt(problem).text}]
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": selection_target(problem)}],
            tokenize=True,
            add_generation_prompt=False,
        )
        return mask_completion(problem.id, tuple(prefix), tuple(full), self.config.max_tokens)


def prepare(output: Path) -> tuple[Path, Path]:
    """Freeze one generated pool and its disjoint test-split transfer cases."""
    pool = generate_v21(GenerationConfig(seed=SEED, problems_per_task=300))
    training_path = output / "training-dataset.json"
    holdout_path = output / "holdout-dataset.json"
    holdout = tuple(
        case
        for case in pool
        if case.task in TASKS and case.split == Split.TEST and case.condition in CONDITIONS
    )
    validate(holdout)
    config = PilotConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    selected = select_pilot(pool, config)
    if len(selected) != config.examples or {p.base_id for p in selected} & {
        p.base_id for p in holdout
    }:
        raise ValueError("Training and holdout bases overlap or training budget differs")
    training_payload = DATASET_ADAPTER.dump_json(selected, indent=2).decode("utf-8")
    holdout_payload = DATASET_ADAPTER.dump_json(holdout, indent=2).decode("utf-8")
    publish_text(training_path, training_payload)
    publish_text(holdout_path, holdout_payload)
    manifest = {
        "protocol": "deduction-specialist-7b-v1",
        "seed": SEED,
        "training_sha256": hashlib.sha256(training_payload.encode()).hexdigest(),
        "holdout_sha256": hashlib.sha256(holdout_payload.encode()).hexdigest(),
        "training_example_ids": [case.id for case in selected],
        "holdout_cases": len(holdout),
        "bases_per_task": BASES_PER_TASK,
        "primary_metric": "deduction_selection_exact_by_condition",
        "secondary_metric": "tracking_selection_exact_by_condition",
        "note": "New post-hoc experiment; template family is shared across train and test.",
    }
    publish_text(output / "dataset-manifest.json", json.dumps(manifest, indent=2))
    return training_path, holdout_path


def train(dataset: Path, output: Path, interval: int) -> Path:
    """Fit a single QLoRA adapter with integrity-checked sparse checkpoints."""
    config = PilotConfig.model_validate_json(CONFIG.read_text(encoding="utf-8"))
    selected = select_pilot(DATASET_ADAPTER.validate_json(dataset.read_bytes()), config)
    backend = DeductionTrainingBackend(config)
    examples = tuple(backend.encode(problem) for problem in selected)
    manifest = {
        "protocol": "deduction-specialist-7b-qlora-v1",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
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
        logger.info("Completed training already exists: {}", destination)
        return destination
    latest = latest_sparse_checkpoint(destination, interval)
    history: list[StepResult] = list(verify_checkpoint(latest).history) if latest else []
    if latest:
        backend.restore(latest)
    initial_loss = backend.probe(examples[0])[0] if not latest else None
    for index in range(len(history), config.steps):
        history.append(backend.step(examples[index % len(examples)], index + 1))
        if (index + 1) % interval == 0 or index + 1 == config.steps:
            save_checkpoint(backend, destination, history)
            logger.info("Deduction checkpoint: step={}, loss={:.4f}", index + 1, history[-1].loss)
    summary = {
        "experiment_type": "post_hoc_deduction_specialist",
        "completed_steps": len(history),
        "unique_training_examples": len(examples),
        "initial_train_probe_loss": initial_loss,
        "final_train_probe_loss": backend.probe(examples[0])[0],
        "training_seconds": sum(result.elapsed_seconds for result in history),
        "peak_reserved_bytes": max(result.peak_reserved_bytes for result in history),
    }
    publish_text(
        destination / "history.json", json.dumps([r.model_dump() for r in history], indent=2)
    )
    publish_text(destination / "summary.json", json.dumps(summary, indent=2))
    return destination


@logger.catch(reraise=True)
def main() -> None:
    """Prepare CPU-only data or fit the specialist on a Kaggle T4."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/deduction-7b"))
    parser.add_argument("--checkpoint-every", type=int, default=80)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.mode == "prepare":
        logger.info("Frozen datasets: {}", prepare(args.output))
        return
    if args.checkpoint_every < 1:
        parser.error("Checkpoint interval must be positive")
    dataset = args.output / "training-dataset.json"
    if not dataset.exists():
        raise FileNotFoundError("Run prepare before train")
    require_t4()
    logger.info(
        "Deduction adapter: {}", train(dataset, args.output / "training", args.checkpoint_every)
    )


if __name__ == "__main__":
    main()
