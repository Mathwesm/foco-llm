"""Bounded LoRA smoke test with restart, immutable artifacts, and reload validation."""

import hashlib
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure

from foco_llm.core.training_data import select_smoke_examples
from foco_llm.models.training import SmokeConfig, StepResult, TrainingExample
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.training_backend import TrainingBackend, TrainingError
from foco_llm.services.training_checkpoints import (
    latest_checkpoint,
    save_checkpoint,
    verify_checkpoint,
)


def _plot(history: list[StepResult]) -> Figure:
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    axis.plot([r.step for r in history], [r.loss for r in history], marker="o", color="#0072B2")
    axis.set(
        xlabel="Optimizer update (count)",
        ylabel="Completion loss (nats/token)",
        title="Functional LoRA test: different training examples, not validation",
    )
    axis.set_ylim(bottom=0)
    return figure


def _finish(
    backend: TrainingBackend,
    destination: Path,
    history: list[StepResult],
    initial_base: str,
    initial_adapter: str,
    initial_loss: float,
) -> None:
    adapter = backend.digest(adapters=True)
    if initial_base != backend.digest(adapters=False) or initial_adapter == adapter:
        raise TrainingError("Base weights changed or adapters failed to update")
    examples = json.loads((destination / "examples.json").read_text(encoding="utf-8"))
    probe = TrainingExample.model_validate(examples[0])
    loss, logits = backend.probe(probe)
    path = destination / f"step-{len(history):04d}"
    difference = backend.reload(path, probe, logits)
    summary = {
        "experiment_type": "functional_training_test_not_effectiveness_evidence",
        "completed_steps": len(history),
        "base_weights_unchanged": True,
        "adapters_changed": True,
        "reload_max_abs_logit_difference": difference,
        "initial_train_probe_loss": initial_loss,
        "final_train_probe_loss": loss,
        "training_seconds": sum(r.elapsed_seconds for r in history),
        "supervised_tokens": sum(r.supervised_tokens for r in history),
        "peak_allocated_bytes": max(r.peak_allocated_bytes for r in history),
        "peak_reserved_bytes": max(r.peak_reserved_bytes for r in history),
    }
    publish_text(
        destination / "history.json", json.dumps([r.model_dump() for r in history], indent=2)
    )
    if not (destination / "training-loss.png").exists():
        _plot(history).savefig(destination / "training-loss.png", dpi=180, bbox_inches="tight")
    publish_text(destination / "summary.json", json.dumps(summary, indent=2))


def run_smoke(
    dataset: Path, config: SmokeConfig, output: Path, stop_after: int | None = None
) -> Path:
    """Train a bounded adapter, optionally stopping after a checkpoint to test restart."""
    if stop_after is not None and not 1 <= stop_after < config.steps:
        raise ValueError("Pause step must be positive and less than total steps")
    selected = select_smoke_examples(load_dataset(dataset))
    backend = TrainingBackend(config)
    examples = tuple(backend.encode(p) for p in selected)
    manifest = {
        "config": config.model_dump(),
        "runtime": backend.metadata(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "example_ids": [p.id for p in selected],
        "supervision": "answer_and_evidence; prompt masked; assistant EOS supervised",
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    publish_text(
        destination / "examples.json", json.dumps([e.model_dump() for e in examples], indent=2)
    )
    if (destination / "summary.json").exists():
        logger.info("Completed smoke test already exists: {}", destination)
        return destination
    base, adapter = backend.digest(adapters=False), backend.digest(adapters=True)
    initial_loss, _ = backend.probe(examples[0])
    latest = latest_checkpoint(destination)
    history = list(verify_checkpoint(latest).history) if latest else []
    if latest:
        backend.restore(latest)
        logger.info("Resumed training checkpoint: completed_steps={}", len(history))
    if stop_after is not None and len(history) >= stop_after:
        logger.info("Requested pause already reached: {}", destination)
        return destination
    for index in range(len(history), config.steps):
        history.append(backend.step(examples[index % len(examples)], index + 1))
        save_checkpoint(backend, destination, history)
        logger.info("Training checkpoint: step={}, loss={:.4f}", index + 1, history[-1].loss)
        if stop_after is not None and index + 1 >= stop_after and index + 1 < config.steps:
            logger.info("Requested functional pause after checkpoint: {}", destination)
            return destination
    _finish(backend, destination, history, base, adapter, initial_loss)
    logger.info("Functional training and reload verified: {}", destination)
    return destination
