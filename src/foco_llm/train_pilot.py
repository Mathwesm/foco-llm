"""Run a bounded source-task LoRA pilot, preserving the original smoke executor."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger

from foco_llm.core.arithmetic_curriculum import with_arithmetic_prefixes
from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.models.training import StepResult, TrainingExample
from foco_llm.services.answer_training import AnswerTrainingBackend
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.training_backend import TrainingBackend, TrainingError
from foco_llm.services.training_checkpoints import (
    latest_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)
from foco_llm.utils.logger import setup_logging


def run_pilot(dataset: Path, config: PilotConfig, output: Path) -> Path:
    """Train only source examples, checkpoint each update, and verify adapter reload."""
    selected = select_pilot(load_dataset(dataset), config)
    if config.arithmetic_prefix_curriculum:
        selected = with_arithmetic_prefixes(selected)
    backend = (
        AnswerTrainingBackend(config) if config.supervision == "answer" else TrainingBackend(config)
    )
    examples = tuple(backend.encode(p) for p in selected)
    manifest = {
        "protocol": (
            "arithmetic-prefix-curriculum-v1"
            if config.arithmetic_prefix_curriculum
            else "balanced-task-pilot-v1"
            if config.tasks
            else "source-task-pilot-v1"
        ),
        "config": config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "source_sha256": _source_fingerprint(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "example_ids": [p.id for p in selected],
        "supervision": config.supervision,
        "schedule": "fixed seeded base order, repeated cyclically; batch size one",
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    publish_text(
        destination / "examples.json", json.dumps([e.model_dump() for e in examples], indent=2)
    )
    if (destination / "summary.json").exists():
        logger.info("Completed pilot already exists: {}", destination)
        return destination
    initial_base = backend.digest(adapters=False)
    initial_adapter = backend.digest(adapters=True)
    initial_loss, _ = backend.probe(examples[0])
    latest = latest_checkpoint(destination)
    history: list[StepResult] = list(verify_checkpoint(latest).history) if latest else []
    if latest:
        backend.restore(latest)
    for index in range(len(history), config.steps):
        example = examples[index % len(examples)]
        history.append(backend.step(example, index + 1))
        save_checkpoint(backend, destination, history)
        logger.info("Pilot update: step={}, loss={:.4f}", index + 1, history[-1].loss)
    _complete(backend, destination, examples, history, initial_base, initial_adapter, initial_loss)
    return destination


def _complete(
    backend: TrainingBackend,
    destination: Path,
    examples: tuple[TrainingExample, ...],
    history: list[StepResult],
    initial_base: str,
    initial_adapter: str,
    initial_loss: float,
) -> None:
    if backend.digest(adapters=False) != initial_base:
        raise TrainingError("Pilot changed frozen base weights")
    if backend.digest(adapters=True) == initial_adapter:
        raise TrainingError("Pilot failed to update adapters")
    final_loss, logits = backend.probe(examples[0])
    difference = backend.reload(destination / f"step-{len(history):04d}", examples[0], logits)
    summary = {
        "experiment_type": "source_task_development_pilot_not_confirmatory_evidence",
        "completed_steps": len(history),
        "unique_training_examples": len(examples),
        "initial_train_probe_loss": initial_loss,
        "final_train_probe_loss": final_loss,
        "base_weights_unchanged": True,
        "adapters_changed": True,
        "reload_max_abs_logit_difference": difference,
        "training_seconds": sum(r.elapsed_seconds for r in history),
        "supervised_tokens": sum(r.supervised_tokens for r in history),
        "peak_allocated_bytes": max(r.peak_allocated_bytes for r in history),
        "peak_reserved_bytes": max(r.peak_reserved_bytes for r in history),
    }
    publish_text(
        destination / "history.json", json.dumps([r.model_dump() for r in history], indent=2)
    )
    publish_text(destination / "summary.json", json.dumps(summary, indent=2))
    logger.info("Pilot completed and adapter reload verified: {}", destination)


@logger.catch(reraise=True)
def main() -> None:
    """Read validated JSON configuration and start the manually requested local pilot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    config = PilotConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    run_pilot(args.dataset, config, args.output)


if __name__ == "__main__":
    main()
