"""Evaluate model-selected arithmetic evidence before applying a calculator."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.arithmetic_selection import (
    parse_selected_ids_v2,
    selected_calculation,
    selection_prompt_v2,
)
from foco_llm.models.experiment import Problem, Prompt
from foco_llm.models.inference import Checkpoint, InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

FORMS = ("text-1", "text-2", "text-4", "similar-4")
EXPECTED_CASES = 40


def checkpoint(prompt: Prompt, directory: Path, backend: TransformersBackend) -> Checkpoint:
    """Resume only if the saved raw generation matches this exact selector prompt."""
    fingerprint = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    path = directory / "responses" / f"{hashlib.sha256(prompt.id.encode()).hexdigest()}.json"
    if path.exists():
        saved = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if saved.id != prompt.id or saved.prompt_sha256 != fingerprint:
            raise ValueError("Selection checkpoint differs from current prompt")
        return saved
    record = Checkpoint(
        id=prompt.id,
        prompt_sha256=fingerprint,
        finished_at=datetime.now(UTC),
        generation=backend.generate(prompt),
    )
    publish_text(path, record.model_dump_json(indent=2))
    return record


def score_selection(problem: Problem, raw: str) -> dict[str, object]:
    """Score relevance and calculated answer without repairing model output."""
    row: dict[str, object] = {
        "id": problem.id,
        "form": problem.id.rsplit(":", 1)[-1],
        "expected_ids": list(problem.evidence),
        "expected_answer": problem.answer,
        "selected_ids": [],
        "selection_valid": False,
        "selection_exact": False,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": len(problem.evidence),
        "calculation_valid": False,
        "calculated_answer": "",
        "answer_correct": False,
        "error": "",
    }
    try:
        selected = parse_selected_ids_v2(raw, problem.facts)
        row["selection_valid"] = True
        row["selected_ids"] = list(selected)
        relevant = set(problem.evidence)
        chosen = set(selected)
        row["selection_exact"] = chosen == relevant
        row["true_positive"] = len(chosen & relevant)
        row["false_positive"] = len(chosen - relevant)
        row["false_negative"] = len(relevant - chosen)
        expression, calculated = selected_calculation(problem.facts, selected)
        row["calculation_valid"] = True
        row["expression"] = expression
        row["calculated_answer"] = calculated
        row["answer_correct"] = calculated == problem.answer
    except ValueError as error:
        row["error"] = str(error)
    return row


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate exact selection, arithmetic, and fact-level error counts."""
    groups = {}
    for form in FORMS:
        selected = [row for row in rows if row["form"] == form]
        groups[form] = {
            "cases": len(selected),
            "selection_valid": sum(bool(row["selection_valid"]) for row in selected),
            "selection_exact": sum(bool(row["selection_exact"]) for row in selected),
            "calculation_valid": sum(bool(row["calculation_valid"]) for row in selected),
            "answer_correct": sum(bool(row["answer_correct"]) for row in selected),
            "true_positive": sum(int(row["true_positive"]) for row in selected),
            "false_positive": sum(int(row["false_positive"]) for row in selected),
            "false_negative": sum(int(row["false_negative"]) for row in selected),
        }
    return {"cases": len(rows), "groups": groups}


@logger.catch(reraise=True)
def main() -> None:
    """Run a pinned selector and publish immutable raw and scored results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.adapter is not None and args.training_dataset is None:
        raise ValueError("Adapter evaluation requires the original training dataset")
    all_cases = TypeAdapter(tuple[Problem, ...]).validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    cases = tuple(p for p in all_cases if p.id.rsplit(":", 1)[-1] in FORMS)
    if len(cases) != EXPECTED_CASES or len({p.id for p in cases}) != EXPECTED_CASES:
        raise ValueError("Expected 40 unique held-out arithmetic selection cases")
    config = InferenceConfig(model_id=args.model_id, revision=args.revision, precision="bfloat16")
    backend = (
        AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
        if args.adapter is not None
        else TransformersBackend(config)
    )
    manifest = {
        "protocol": "arithmetic-selection-v2",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "runtime": backend.metadata(),
        "cases": len(cases),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    output = args.output / hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    publish_text(output / "manifest.json", payload)
    rows = []
    for index, problem in enumerate(cases, 1):
        raw = checkpoint(selection_prompt_v2(problem), output, backend).generation.text
        rows.append(score_selection(problem, raw))
        if index % 10 == 0:
            logger.info("Arithmetic selection progress: completed={}, total={}", index, len(cases))
    publish_text(output / "scores.json", json.dumps(rows, indent=2))
    publish_text(output / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Arithmetic selection complete: path={}", output)


if __name__ == "__main__":
    main()
