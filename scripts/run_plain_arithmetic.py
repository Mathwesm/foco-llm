"""Probe the same expressions with an integer-only prompt and no evidence contract."""

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.models.experiment import Problem, Prompt
from foco_llm.models.inference import Checkpoint, InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

EXPRESSION_PATTERN = re.compile(r"To answer the question, calculate ([0-9+ -]+)\.")
EXPECTED_EXPRESSIONS = 30


def plain_prompt(problem: Problem, boxed: bool = False) -> Prompt:
    """Use the frozen expression without reference answer or evidence labels."""
    match = EXPRESSION_PATTERN.fullmatch(problem.facts[0].text)
    if match is None:
        raise ValueError("Expected a predeclared arithmetic expression")
    instruction = (
        f"Please reason step by step and put your final answer within \\boxed{{}}. "
        f"Calculate {match[1]}."
        if boxed
        else f"Calculate {match[1]}. Reply with only the integer."
    )
    return Prompt(id=problem.id, text=instruction)


def score_integer(raw: str, expected: str) -> tuple[bool, bool]:
    """Separate integer-only compliance from arithmetic correctness."""
    value = raw.strip()
    is_integer = re.fullmatch(r"-?\d+", value) is not None
    return is_integer, is_integer and value == expected


def score_boxed(raw: str, expected: str) -> tuple[bool, bool]:
    """Accept one unambiguous boxed integer in the model's native response."""
    boxed = re.findall(r"\\boxed\{\s*(-?\d+)\s*\}", raw)
    return len(boxed) == 1, len(boxed) == 1 and boxed[0] == expected


def checkpoint(
    problem: Problem, directory: Path, backend: TransformersBackend, boxed: bool = False
) -> Checkpoint:
    """Resume a raw generation only when its exact plain prompt matches."""
    prompt = plain_prompt(problem, boxed=boxed)
    fingerprint = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    filename = hashlib.sha256(problem.id.encode("utf-8")).hexdigest()
    path = directory / "responses" / f"{filename}.json"
    if path.exists():
        saved = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if saved.id != problem.id or saved.prompt_sha256 != fingerprint:
            raise ValueError("Plain arithmetic checkpoint differs from current prompt")
        return saved
    record = Checkpoint(
        id=problem.id,
        prompt_sha256=fingerprint,
        finished_at=datetime.now(UTC),
        generation=backend.generate(prompt),
    )
    publish_text(path, record.model_dump_json(indent=2))
    return record


@logger.catch(reraise=True)
def main() -> None:
    """Generate each expression once and publish strict integer scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    parser.add_argument("--boxed", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.adapter is not None and args.training_dataset is None:
        raise ValueError("Adapter evaluation requires the original training dataset")
    all_cases = TypeAdapter(tuple[Problem, ...]).validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    cases = tuple(p for p in all_cases if ":expression-" in p.id)
    if len(cases) != EXPECTED_EXPRESSIONS:
        raise ValueError("Expected 30 frozen arithmetic expressions")
    config = InferenceConfig(
        model_id=args.model_id,
        revision=args.revision,
        precision="bfloat16",
        max_new_tokens=512 if args.boxed else 128,
    )
    backend = (
        AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
        if args.adapter is not None
        else TransformersBackend(config)
    )
    manifest = {
        "protocol": "boxed-arithmetic-v1" if args.boxed else "plain-arithmetic-v1",
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
    format_key = "boxed_valid" if args.boxed else "integer_only"
    score = score_boxed if args.boxed else score_integer
    for problem in cases:
        raw = checkpoint(problem, output, backend, boxed=args.boxed).generation.text
        valid, correct = score(raw, problem.answer)
        rows.append(
            {
                "id": problem.id,
                "form": problem.id.rsplit(":", 1)[-1],
                "expected": problem.answer,
                "observed": raw,
                format_key: valid,
                "answer_correct": correct,
            }
        )
    publish_text(output / "scores.json", json.dumps(rows, indent=2))
    groups = {
        form: {
            "cases": sum(row["form"] == form for row in rows),
            format_key: sum(row["form"] == form and row[format_key] for row in rows),
            "answer_correct": sum(row["form"] == form and row["answer_correct"] for row in rows),
        }
        for form in ("expression-1", "expression-2", "expression-4")
    }
    publish_text(
        output / "summary.json", json.dumps({"cases": len(rows), "groups": groups}, indent=2)
    )
    logger.info("Plain arithmetic complete: path={}", output)


if __name__ == "__main__":
    main()
