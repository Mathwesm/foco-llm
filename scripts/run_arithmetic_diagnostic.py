"""Run resumable arithmetic probes without weakening benchmark validation."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import ContentError, decode_content
from foco_llm.models.experiment import Problem
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _checkpoint, _source_fingerprint, parse_response
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

EXPECTED_PROBES = 70


def load_probes(path: Path) -> tuple[Problem, ...]:
    """Validate the diagnostic contract without applying the original sentence solver."""
    probes = TypeAdapter(tuple[Problem, ...]).validate_json(path.read_text(encoding="utf-8"))
    if len(probes) != EXPECTED_PROBES or len({p.id for p in probes}) != len(probes):
        raise ValueError("Expected 70 unique arithmetic diagnostic probes")
    return probes


def score_probe(problem: Problem, raw: str) -> dict[str, str | bool]:
    """Score the unchanged continuation under strict and answer-only contracts."""
    strict_valid = False
    answer = ""
    try:
        parse_response(problem, raw)
        strict_valid = True
    except ValueError:
        pass
    try:
        answer = decode_content(raw).answer
    except ContentError:
        answer = ""
    return {
        "id": problem.id,
        "form": problem.id.rsplit(":", 1)[-1],
        "strict_valid": strict_valid,
        "answer_readable": bool(answer),
        "answer_correct": answer.strip() == problem.answer,
        "expected": problem.answer,
        "observed": answer,
    }


def summarize(rows: list[dict[str, str | bool]]) -> dict[str, object]:
    """Aggregate answer accuracy by predeclared probe form and depth."""
    forms = sorted({str(row["form"]) for row in rows})
    grouped = {}
    for form in forms:
        selected = [row for row in rows if row["form"] == form]
        grouped[form] = {
            "cases": len(selected),
            "strict_valid": sum(bool(row["strict_valid"]) for row in selected),
            "answer_readable": sum(bool(row["answer_readable"]) for row in selected),
            "answer_correct": sum(bool(row["answer_correct"]) for row in selected),
        }
    return {"cases": len(rows), "groups": grouped}


@logger.catch(reraise=True)
def main() -> None:
    """Load a pinned base or adapter, checkpoint every probe, then publish scores."""
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
    probes = load_probes(args.dataset)
    config = InferenceConfig(
        model_id=args.model_id,
        revision=args.revision,
        split="validation",
        precision="bfloat16",
        max_new_tokens=128,
    )
    backend = (
        AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
        if args.adapter is not None
        else TransformersBackend(config)
    )
    manifest = {
        "protocol": "arithmetic-diagnostic-v1",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "runtime": backend.metadata(),
        "cases": len(probes),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    output = args.output / hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    publish_text(output / "manifest.json", payload)
    rows = []
    for index, problem in enumerate(probes, 1):
        checkpoint = _checkpoint(problem, output, backend)
        rows.append(score_probe(problem, checkpoint.generation.text))
        if index % 10 == 0:
            logger.info(
                "Arithmetic diagnostic progress: completed={}, total={}", index, len(probes)
            )
    publish_text(output / "scores.json", json.dumps(rows, indent=2))
    publish_text(output / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Arithmetic diagnostic complete: path={}", output)


if __name__ == "__main__":
    main()
