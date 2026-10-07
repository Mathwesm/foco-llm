"""Resume immutable model responses, preserve failures, and publish paired metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import ValidationError

from foco_llm.core.evaluation import evaluate
from foco_llm.core.plots import plot_paired_accuracy
from foco_llm.core.prompts import PROMPT_VERSION, build_prompt
from foco_llm.models.experiment import ModelRun, Prediction, Problem, Prompt
from foco_llm.models.inference import Checkpoint, Generation, InferenceConfig, Response
from foco_llm.services.artifacts import load_dataset, publish_text


class Backend(Protocol):
    """Minimal adapter interface; backends never receive reference labels."""

    def generate(self, prompt: Prompt) -> Generation:
        """Return the raw continuation and observed resource use."""
        ...

    def metadata(self) -> dict[str, str]:
        """Return the runtime identity needed to prevent mixed checkpoints."""
        ...


def parse_response(problem: Problem, text: str) -> Prediction:
    """Parse strict JSON without recovering explanations or repairing answers.

    Raises:
        ValueError: JSON is malformed or evidence is duplicate or nonexistent.
    """
    response = Response.model_validate_json(text)
    if len(set(response.evidence)) != len(response.evidence):
        raise ValueError("Duplicate evidence identifiers")
    if not set(response.evidence).issubset({fact.id for fact in problem.facts}):
        raise ValueError("Unknown evidence identifier")
    return Prediction(id=problem.id, answer=response.answer, evidence=response.evidence)


def _checkpoint(problem: Problem, directory: Path, backend: Backend) -> Checkpoint:
    prompt = build_prompt(problem)
    fingerprint = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    filename = hashlib.sha256(problem.id.encode("utf-8")).hexdigest()
    path = directory / "responses" / f"{filename}.json"
    if path.exists():
        saved = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if saved.id != problem.id or saved.prompt_sha256 != fingerprint:
            raise ValueError("Checkpoint does not match the current prompt")
        return saved
    generation = backend.generate(prompt)
    record = Checkpoint(
        id=problem.id,
        prompt_sha256=fingerprint,
        finished_at=datetime.now(UTC),
        generation=generation,
    )
    publish_text(path, record.model_dump_json(indent=2))
    return record


def _collect(
    problems: tuple[Problem, ...], directory: Path, backend: Backend
) -> tuple[list[Checkpoint], list[Prediction], dict[str, str]]:
    records: list[Checkpoint] = []
    predictions: list[Prediction] = []
    rejected: dict[str, str] = {}
    for index, problem in enumerate(problems, start=1):
        record = _checkpoint(problem, directory, backend)
        records.append(record)
        try:
            predictions.append(parse_response(problem, record.generation.text))
        except (ValidationError, ValueError):
            rejected[problem.id] = "invalid_json_or_evidence"
        logger.info("Inference checkpoint: completed={}, total={}", index, len(problems))
    return records, predictions, rejected


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_text(encoding="utf-8").encode("utf-8"))
    return digest.hexdigest()


def _publish_results(
    directory: Path,
    problems: tuple[Problem, ...],
    config: InferenceConfig,
    collected: tuple[list[Checkpoint], list[Prediction], dict[str, str]],
) -> None:
    records, predictions, rejected = collected
    run = ModelRun(
        model_id=config.model_id,
        model_revision=config.revision,
        prompt_version=PROMPT_VERSION,
        seed=config.seed,
        decoding={"do_sample": False, "max_new_tokens": config.max_new_tokens},
        predictions=tuple(predictions),
    )
    report = evaluate(problems, run)
    publish_text(directory / "predictions.json", run.model_dump_json(indent=2))
    publish_text(directory / "report.json", report.model_dump_json(indent=2))
    summary = {
        "attempted": len(records),
        "valid": len(predictions),
        "rejected": rejected,
        "generation_seconds": sum(r.generation.elapsed_seconds for r in records),
        "output_tokens": sum(r.generation.output_tokens for r in records),
        "peak_allocated_bytes": max(r.generation.peak_allocated_bytes for r in records),
        "peak_reserved_bytes": max(r.generation.peak_reserved_bytes for r in records),
        "stop_counts": {
            reason: sum(r.generation.stop_reason == reason for r in records)
            for reason in ("eos", "token_limit", "time_limit")
        },
    }
    publish_text(directory / "summary.json", json.dumps(summary, indent=2))
    chart = directory / "paired-accuracy.png"
    if not chart.exists():
        figure, _ = plot_paired_accuracy(report)
        figure.savefig(chart, dpi=180, bbox_inches="tight")


def run_inference(dataset: Path, output: Path, config: InferenceConfig, backend: Backend) -> Path:
    """Run one split, resuming only exactly matching data, code, and environment.

    Args:
        dataset: Independently validated reference dataset.
        output: Date-partitioned run directory chosen by the caller.
        config: Immutable model and generation configuration.
        backend: Initialized local generation adapter.

    Returns:
        Immutable result directory, with manifest and per-example checkpoints.
    """
    problems = tuple(p for p in load_dataset(dataset) if p.split == config.split)
    if not problems:
        raise ValueError("Selected split contains no examples")
    manifest = {
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "prompt_version": PROMPT_VERSION,
        "runtime": backend.metadata(),
        "examples": len(problems),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    directory = output / fingerprint
    publish_text(directory / "manifest.json", payload)
    collected = _collect(problems, directory, backend)
    _publish_results(directory, problems, config, collected)
    logger.info("Inference complete: run={}", directory)
    return directory
