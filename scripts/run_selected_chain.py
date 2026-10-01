"""Calculate model-selected facts with the existing four-step model chain."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter
from run_chained_arithmetic import EXPECTED_CASES, UPDATES_PER_CASE, run_case

from foco_llm.core.arithmetic_selection import selected_calculation
from foco_llm.models.experiment import Fact, Problem
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.utils.logger import setup_logging


def score_case(
    problem: Problem, row: dict[str, object], output: Path, backend: AdapterBackend
) -> dict[str, object]:
    """Evaluate one selected expression without reading oracle IDs during inference."""
    result: dict[str, object] = {
        "id": problem.id,
        "expected": problem.answer,
        "selection_exact": row["selection_exact"],
        "selected_ids": row["selected_ids"],
        "model_answer": "",
        "answer_correct": False,
        "generated_steps": 0,
        "reason": "",
    }
    if not row["selection_valid"]:
        result["reason"] = "invalid_selection"
        return result
    try:
        expression, selected_answer = selected_calculation(
            problem.facts, tuple(row["selected_ids"])
        )
        if expression != row.get("expression") or selected_answer != row.get("calculated_answer"):
            raise RuntimeError("Selection score expression differs from selected facts")
        if expression.count("+") + expression.count("-") != UPDATES_PER_CASE:
            result["reason"] = "incomplete_expression"
            return result
        derived = problem.model_copy(
            update={
                "facts": (Fact(id="F1", text=f"To answer the question, calculate {expression}."),),
                "answer": selected_answer,
                "evidence": ("F1",),
            }
        )
        chain = run_case(derived, output, backend)
        result["expression"] = expression
        result["model_answer"] = chain["final_observed"]
        result["answer_correct"] = chain["final_observed"] == problem.answer
        result["generated_steps"] = len(chain["steps"])
        result["steps"] = chain["steps"]
    except ValueError as error:
        result["reason"] = str(error)
    return result


@logger.catch(reraise=True)
def main() -> None:
    """Score end-to-end selection and model arithmetic without oracle repairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("selection_scores", type=Path)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--training-dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    all_cases = TypeAdapter(tuple[Problem, ...]).validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    cases = tuple(p for p in all_cases if p.id.endswith(":similar-4"))
    if len(cases) != EXPECTED_CASES:
        raise ValueError("Expected ten noisy arithmetic diagnostic cases")
    dataset_digest = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    selection_manifest = json.loads(
        (args.selection_scores.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        selection_manifest["protocol"] != "arithmetic-selection-v2"
        or selection_manifest["dataset_sha256"] != dataset_digest
    ):
        raise ValueError("Selection manifest does not match the evaluated dataset")
    scores = json.loads(args.selection_scores.read_text(encoding="utf-8"))
    selected = {row["id"]: row for row in scores if row["form"] == "similar-4"}
    if len(selected) != EXPECTED_CASES or set(selected) != {p.id for p in cases}:
        raise ValueError("Selection scores do not cover the ten paired cases")
    config = InferenceConfig(model_id=args.model_id, revision=args.revision, precision="bfloat16")
    backend = AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
    manifest = {
        "protocol": "selected-chain-v1",
        "config": config.model_dump(mode="json"),
        "dataset_sha256": dataset_digest,
        "selection_scores_sha256": hashlib.sha256(args.selection_scores.read_bytes()).hexdigest(),
        "source_sha256": _source_fingerprint(),
        "runtime": backend.metadata(),
        "cases": len(cases),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    output = args.output / hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    publish_text(output / "manifest.json", payload)
    results = [score_case(problem, selected[problem.id], output, backend) for problem in cases]
    publish_text(output / "scores.json", json.dumps(results, indent=2))
    summary = {
        "cases": len(results),
        "selection_exact": sum(bool(row["selection_exact"]) for row in results),
        "eligible": sum(row["generated_steps"] == UPDATES_PER_CASE for row in results),
        "generated_steps": sum(int(row["generated_steps"]) for row in results),
        "answer_correct": sum(bool(row["answer_correct"]) for row in results),
    }
    publish_text(output / "summary.json", json.dumps(summary, indent=2))
    logger.info("Selected chain complete: path={}", output)


if __name__ == "__main__":
    main()
