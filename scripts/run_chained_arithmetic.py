"""Chain four single-operation model calls on preselected arithmetic expressions."""

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

EXPRESSION_PATTERN = re.compile(r"To answer the question, calculate (\d+)((?:\s*[+-]\s*\d+){4})\.")
EXPECTED_CASES = 10
UPDATES_PER_CASE = 4


def operations(problem: Problem) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Read the explicitly supplied four-step expression without evaluating it."""
    match = EXPRESSION_PATTERN.fullmatch(problem.facts[0].text)
    if match is None:
        raise ValueError("Expected one four-operation diagnostic expression")
    updates = tuple((sign, int(value)) for sign, value in re.findall(r"([+-])\s*(\d+)", match[2]))
    if len(updates) != UPDATES_PER_CASE:
        raise ValueError("Expected four arithmetic updates")
    return int(match[1]), updates


def _checkpoint(prompt: Prompt, directory: Path, backend: TransformersBackend) -> Checkpoint:
    fingerprint = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    path = directory / "responses" / f"{hashlib.sha256(prompt.id.encode()).hexdigest()}.json"
    if path.exists():
        saved = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if saved.id != prompt.id or saved.prompt_sha256 != fingerprint:
            raise ValueError("Chained arithmetic checkpoint differs from current prompt")
        return saved
    record = Checkpoint(
        id=prompt.id,
        prompt_sha256=fingerprint,
        finished_at=datetime.now(UTC),
        generation=backend.generate(prompt),
    )
    publish_text(path, record.model_dump_json(indent=2))
    return record


def run_case(problem: Problem, directory: Path, backend: TransformersBackend) -> dict[str, object]:
    """Feed each model subtotal forward and record the first failed operation."""
    current, updates = operations(problem)
    reference = current + sum(value if sign == "+" else -value for sign, value in updates)
    if str(reference) != problem.answer:
        raise ValueError("Reference arithmetic expression disagrees with diagnostic answer")
    expected = current
    steps = []
    for index, (sign, value) in enumerate(updates, 1):
        local_expected = current + (value if sign == "+" else -value)
        prompt = Prompt(
            id=f"{problem.id}:step-{index}",
            text=f"Calculate {current} {sign} {value}. Reply with only the integer.",
        )
        raw = _checkpoint(prompt, directory, backend).generation.text.strip()
        expected += -value if sign == "-" else value
        valid = re.fullmatch(r"-?\d+", raw) is not None
        steps.append(
            {
                "step": index,
                "prompt": prompt.text,
                "observed": raw,
                "expected_local": str(local_expected),
                "expected_oracle": str(expected),
                "integer_only": valid,
                "operation_correct": valid and int(raw) == local_expected,
                "subtotal_correct": valid and int(raw) == expected,
            }
        )
        if not valid:
            break
        current = int(raw)
    return {
        "id": problem.id,
        "expected": problem.answer,
        "final_observed": str(current) if len(steps) == len(updates) else "",
        "final_correct": len(steps) == len(updates) and str(current) == problem.answer,
        "all_steps_correct": len(steps) == len(updates)
        and all(s["operation_correct"] for s in steps),
        "steps": steps,
    }


@logger.catch(reraise=True)
def main() -> None:
    """Run a pinned model, preserving resumable raw calls and per-case scores."""
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
    cases = tuple(p for p in all_cases if p.id.endswith(":expression-4"))
    if len(cases) != EXPECTED_CASES:
        raise ValueError("Expected ten four-update arithmetic expressions")
    config = InferenceConfig(model_id=args.model_id, revision=args.revision, precision="bfloat16")
    backend = (
        AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
        if args.adapter is not None
        else TransformersBackend(config)
    )
    manifest = {
        "protocol": "chained-arithmetic-v2",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "runtime": backend.metadata(),
        "cases": len(cases),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    directory = args.output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(directory / "manifest.json", payload)
    results = [run_case(problem, directory, backend) for problem in cases]
    publish_text(directory / "scores.json", json.dumps(results, indent=2))
    summary = {
        "cases": len(results),
        "final_correct": sum(bool(r["final_correct"]) for r in results),
        "all_steps_correct": sum(bool(r["all_steps_correct"]) for r in results),
        "generated_steps": sum(len(r["steps"]) for r in results),
    }
    publish_text(directory / "summary.json", json.dumps(summary, indent=2))
    logger.info("Chained arithmetic complete: path={}", directory)


if __name__ == "__main__":
    main()
