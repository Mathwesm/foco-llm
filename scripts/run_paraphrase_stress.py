"""Probe arithmetic evidence selection under controlled local paraphrases."""

import argparse
import gc
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint

from foco_llm.core.arithmetic_selection import (
    lexical_select,
    parse_selected_ids_v2,
    selection_prompt_v2,
)
from foco_llm.models.experiment import Fact, Problem
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pragma: allowlist secret -- public revision
DIAGNOSTIC = Path("reports/2026-10-01/arithmetic-diagnostic/dataset.json")
TRAINING = Path("data/benchmark-v21/2026-09-28/b142fe60ba9cd6a5/dataset.json")
ADAPTER = Path("data/arithmetic-selection-training/2026-10-01/78ce609fc85f07f4/step-0320")
INITIAL = re.compile(r"The initial marble count of (\w+) is (\d+)\.")
UPDATE = re.compile(
    r"Update ([A-Z]): For (\w+), (\d+) marbles? (?:are|is) (added to|removed from) it\."
)
TARGET = re.compile(r"remain in (\w+)\?")
EXPECTED_CASES = 10


def paraphrase_fact(fact: Fact, target: str, evidence: set[str]) -> Fact:
    """Change surface wording while preserving the original arithmetic event."""
    initial = INITIAL.fullmatch(fact.text)
    if initial is not None:
        container, count = initial.groups()
        text = f"At the start, {container} held {count} marbles."
    else:
        update = UPDATE.fullmatch(fact.text)
        if update is None:
            raise ValueError(f"Unrecognized arithmetic template: {fact.id}")
        label, container, count, operation = update.groups()
        if fact.id in evidence:
            if container != target:
                raise ValueError("Reference evidence points to another container")
            subject = "the container named in the question"
        else:
            subject = container
        verb = "received" if operation == "added to" else "lost"
        text = f"Update {label}: {subject} {verb} {count} marbles."
    return Fact(id=fact.id, text=text)


def paraphrase_case(problem: Problem) -> Problem:
    """Add a target-name decoy and remove literal target mentions from updates."""
    match = TARGET.search(problem.question)
    if match is None or not problem.id.endswith(":similar-4"):
        raise ValueError("Expected a four-update similar-condition arithmetic case")
    target = match[1]
    evidence = set(problem.evidence)
    facts = tuple(paraphrase_fact(fact, target, evidence) for fact in problem.facts)
    decoy_id = f"F{max(int(fact.id[1:]) for fact in facts) + 1}"
    decoy = Fact(id=decoy_id, text=f"The painted label on {target} shows the number 27.")
    return Problem.model_validate(
        {
            **problem.model_dump(),
            "id": f"{problem.base_id}:paraphrase-4",
            "facts": [*(fact.model_dump() for fact in facts), decoy.model_dump()],
        }
    )


def make_cases(path: Path) -> tuple[Problem, ...]:
    """Pair ten previously explored validation bases with controlled rewrites."""
    source = TypeAdapter(tuple[Problem, ...]).validate_json(path.read_text(encoding="utf-8"))
    cases = tuple(
        paraphrase_case(problem) for problem in source if problem.id.endswith(":similar-4")
    )
    if len(cases) != EXPECTED_CASES or len({case.base_id for case in cases}) != EXPECTED_CASES:
        raise ValueError("Expected ten distinct paraphrased validation bases")
    return cases


def score(problem: Problem, raw: str) -> dict[str, object]:
    """Count exact selection and fact-level errors without parsing the arithmetic."""
    row: dict[str, object] = {
        "id": problem.id,
        "expected_ids": list(problem.evidence),
        "raw_response": raw,
        "selected_ids": [],
        "valid": False,
        "exact": False,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": len(problem.evidence),
        "error": "",
    }
    try:
        chosen = parse_selected_ids_v2(raw, problem.facts)
    except ValueError as error:
        row["error"] = str(error)
        return row
    relevant = set(problem.evidence)
    selected = set(chosen)
    row.update(
        selected_ids=list(chosen),
        valid=True,
        exact=selected == relevant,
        true_positive=len(selected & relevant),
        false_positive=len(selected - relevant),
        false_negative=len(relevant - selected),
    )
    return row


def summary(rows: list[dict[str, object]]) -> dict[str, int]:
    """Aggregate selection outcomes over the fixed ten-case denominator."""
    return {
        "cases": len(rows),
        "valid": sum(bool(row["valid"]) for row in rows),
        "exact": sum(bool(row["exact"]) for row in rows),
        "true_positive": sum(int(row["true_positive"]) for row in rows),
        "false_positive": sum(int(row["false_positive"]) for row in rows),
        "false_negative": sum(int(row["false_negative"]) for row in rows),
    }


def run_model(
    cases: tuple[Problem, ...],
    dataset: Path,
    work: Path,
    config: InferenceConfig,
    *,
    adapted: bool,
) -> tuple[dict[str, str], list[dict[str, object]]]:
    """Generate or resume raw responses for one base or adapter arm."""
    backend = (
        AdapterBackend(config, ADAPTER, dataset, TRAINING)
        if adapted
        else TransformersBackend(config)
    )
    metadata = backend.metadata()
    arm = "adapted" if adapted else "baseline"
    rows = []
    for index, problem in enumerate(cases, start=1):
        raw = checkpoint(selection_prompt_v2(problem), work / arm, backend).generation.text
        rows.append(score(problem, raw))
        logger.info("Paraphrase selection progress: arm={}, completed={}", arm, index)
    del backend
    gc.collect()
    torch = __import__("torch")
    torch.cuda.empty_cache()
    return metadata, rows


@logger.catch(reraise=True)
def main() -> None:
    """Run frozen local controls and publish a single auditable study artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("data/paraphrase-stress/2026-10-01"))
    parser.add_argument(
        "--report", type=Path, default=Path("reports/2026-10-01/paraphrase-stress/study.json")
    )
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    cases = make_cases(DIAGNOSTIC)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(cases, indent=2).decode("utf-8")
    dataset = args.work / "dataset.json"
    publish_text(dataset, payload)
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if args.report.exists():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if previous.get("dataset_sha256") != dataset_digest:
            raise ValueError("Existing report belongs to a different paraphrase dataset")
        logger.info("Completed paraphrase stress report already exists: {}", args.report)
        return
    lexical = [
        score(problem, json.dumps({"evidence": lexical_select(problem)})) for problem in cases
    ]
    config = InferenceConfig(model_id=MODEL_ID, revision=REVISION, precision="bfloat16")
    baseline_metadata, baseline = run_model(cases, dataset, args.work, config, adapted=False)
    adapted_metadata, adapted = run_model(cases, dataset, args.work, config, adapted=True)
    report = {
        "protocol": "posthoc-paraphrase-selection-stress-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "source_dataset_sha256": hashlib.sha256(DIAGNOSTIC.read_bytes()).hexdigest(),
        "dataset_sha256": dataset_digest,
        "source_sha256": _source_fingerprint(),
        "training_dataset_sha256": hashlib.sha256(TRAINING.read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "cases": [case.model_dump(mode="json") for case in cases],
        "arms": {
            "lexical": {"summary": summary(lexical), "scores": lexical},
            "baseline": {
                "runtime": baseline_metadata,
                "summary": summary(baseline),
                "scores": baseline,
            },
            "adapted": {
                "runtime": adapted_metadata,
                "summary": summary(adapted),
                "scores": adapted,
            },
        },
        "limitations": [
            "Post-hoc rewrites of ten previously explored validation bases",
            "Explicit reference to the question makes the rewrite artificial",
            "Lexical decoy is deliberately adversarial to the literal-match rule",
            "Selection only; paraphrased arithmetic is not passed to the template calculator",
        ],
    }
    publish_text(args.report, json.dumps(report, indent=2))
    logger.info("Paraphrase stress report: path={}", args.report)


if __name__ == "__main__":
    main()
