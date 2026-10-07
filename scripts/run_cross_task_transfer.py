"""Freeze and evaluate held-out evidence selection in deduction and tracking."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint
from run_blind_selection import (
    CLOUD_ID,
    CLOUD_REVISION,
    LOCAL_ID,
    LOCAL_REVISION,
    inference_config,
)

from foco_llm.core.arithmetic_selection import parse_selected_ids_v2
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.models.experiment import Condition, GenerationConfig, Problem, Prompt, Split, Task
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

SEED = 20261004
BASES_PER_TASK = 30
CHAIN_FACTS = 6
CONDITIONS = (Condition.UNRELATED, Condition.SIMILAR)
TASKS = (Task.DEDUCTION, Task.TRACKING)
PROTOCOL = "cross-task-evidence-transfer-v1"
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])


def generate() -> tuple[Problem, ...]:
    """Select fresh test-split chains from both non-arithmetic task families."""
    pool = generate_v21(GenerationConfig(seed=SEED, problems_per_task=300))
    cases: list[Problem] = []
    for task in TASKS:
        task_cases = [
            case
            for case in pool
            if case.task == task and case.split == Split.TEST and case.condition in CONDITIONS
        ]
        if len(task_cases) != BASES_PER_TASK * len(CONDITIONS):
            raise ValueError("Unexpected held-out task count")
        cases.extend(task_cases)
    return tuple(cases)


def validate(cases: tuple[Problem, ...]) -> None:
    """Reject missing pairs, malformed evidence, and any arithmetic items."""
    if len(cases) != len(TASKS) * BASES_PER_TASK * len(CONDITIONS):
        raise ValueError("Transfer dataset size differs from protocol")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Transfer dataset contains duplicate IDs")
    pairs: dict[str, list[Problem]] = {}
    for case in cases:
        if case.task not in TASKS or case.split != Split.TEST:
            raise ValueError("Transfer item has an unexpected task or split")
        pairs.setdefault(case.base_id, []).append(case)
        if len(case.evidence) != CHAIN_FACTS:
            raise ValueError("Transfer chain must have six relevant facts")
    if len(pairs) != len(TASKS) * BASES_PER_TASK:
        raise ValueError("Transfer dataset has missing bases")
    for variants in pairs.values():
        if {case.condition for case in variants} != set(CONDITIONS):
            raise ValueError("Transfer base has missing conditions")
        if len({case.answer for case in variants}) != 1:
            raise ValueError("Transfer pair has different answers")


def selection_prompt(problem: Problem) -> Prompt:
    """Request only evidence IDs using a fixed format example and no gold labels."""
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in problem.facts)
    task_instruction = {
        Task.DEDUCTION: (
            "Select the starting property of the queried object and the entire "
            "implication chain needed to reach its terminal property."
        ),
        Task.TRACKING: (
            "Select the initial location of the queried object and every timed "
            "transfer on its path to the final box."
        ),
    }[problem.task]
    example = (
        "Example: [F1] The initial marble count of red is 5. "
        "[F2] The initial marble count of blue is 8. "
        "[F3] For red, 2 marbles are added to it. "
        "Question: How many marbles remain in red? "
        'Response: {"evidence":["F1","F3"]}.\n\n'
    )
    instruction = (
        f"{task_instruction} Ignore facts unrelated to the queried object or chain. "
        'Return only a JSON object with one key, "evidence", containing fact ID '
        'strings such as "F1". Do not return the answer. '
        "Keep the facts in their original order."
    )
    return Prompt(
        id=problem.id, text=f"{example}{instruction}\n\n{facts}\n\nQuestion: {problem.question}"
    )


def symbolic_select(problem: Problem) -> tuple[str, ...]:
    """Follow visible code links as a task-specific solvability control."""
    target_pattern = r"for (\w+)\?" if problem.task == Task.DEDUCTION else r"contains (\w+) after"
    target = re.search(target_pattern, problem.question)
    if target is None:
        raise ValueError("Question does not identify an object")
    if problem.task == Task.DEDUCTION:
        initial_pattern = rf"Property (P\d+) holds for {re.escape(target[1])}\."
        edge_pattern = r"Anything with property (P\d+) also has property (P\d+)\."
    else:
        initial_pattern = rf"The location of {re.escape(target[1])} at time 0 is box (B\d+)\."
        edge_pattern = r"Time \d+: move all items from box (B\d+) into box (B\d+)\."
    initial = next(
        (fact for fact in problem.facts if re.fullmatch(initial_pattern, fact.text)), None
    )
    if initial is None:
        raise ValueError("Target has no starting fact")
    start = re.fullmatch(initial_pattern, initial.text)
    if start is None:
        raise ValueError("Starting fact is malformed")
    current = start[1]
    selected = {initial.id}
    while True:
        edges = [
            (fact, match)
            for fact in problem.facts
            if (match := re.fullmatch(edge_pattern, fact.text)) and match[1] == current
        ]
        if not edges:
            break
        if len(edges) != 1 or edges[0][0].id in selected:
            raise ValueError("Visible code chain is ambiguous or cyclic")
        fact, match = edges[0]
        selected.add(fact.id)
        current = match[2]
    return tuple(fact.id for fact in problem.facts if fact.id in selected)


def score(problem: Problem, raw: str) -> dict[str, object]:
    """Score exact evidence selection without repairing model continuations."""
    row: dict[str, object] = {
        "id": problem.id,
        "base_id": problem.base_id,
        "task": problem.task.value,
        "condition": problem.condition.value,
        "expected_ids": list(problem.evidence),
        "selected_ids": [],
        "selection_valid": False,
        "selection_exact": False,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": len(problem.evidence),
        "error": "",
    }
    try:
        chosen = set(parse_selected_ids_v2(raw, problem.facts))
        relevant = set(problem.evidence)
        row.update(
            selected_ids=sorted(chosen, key=lambda item: int(item[1:])),
            selection_valid=True,
            selection_exact=chosen == relevant,
            true_positive=len(chosen & relevant),
            false_positive=len(chosen - relevant),
            false_negative=len(relevant - chosen),
        )
    except ValueError as error:
        row["error"] = str(error)
    return row


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate by task and distractor condition."""
    groups: dict[str, dict[str, int]] = {}
    for task in TASKS:
        for condition in CONDITIONS:
            subset = [
                row
                for row in rows
                if row["task"] == task.value and row["condition"] == condition.value
            ]
            groups[f"{task.value}:{condition.value}"] = {
                "cases": len(subset),
                "selection_valid": sum(bool(row["selection_valid"]) for row in subset),
                "selection_exact": sum(bool(row["selection_exact"]) for row in subset),
                "false_positive": sum(cast(int, row["false_positive"]) for row in subset),
                "false_negative": sum(cast(int, row["false_negative"]) for row in subset),
            }
    return {"cases": len(rows), "base_problems": len(rows) // 2, "groups": groups}


def prepare(dataset: Path) -> None:
    """Freeze the dataset and cryptographic identity before model evaluation."""
    cases = generate()
    validate(cases)
    payload = DATASET_ADAPTER.dump_json(cases, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    publish_text(dataset, payload)
    publish_text(
        dataset.with_name("manifest.json"),
        json.dumps(
            {
                "protocol": PROTOCOL,
                "seed": SEED,
                "tasks": [task.value for task in TASKS],
                "conditions": [condition.value for condition in CONDITIONS],
                "bases_per_task": BASES_PER_TASK,
                "cases": len(cases),
                "dataset_sha256": digest,
                "prompt_version": "cross_task_selection_v1",
                "primary_metric": "selection_exact_by_task_and_condition",
                "note": "Freeze before inference; no prompt or adapter tuning on these items.",
            },
            indent=2,
        ),
    )
    logger.info("Frozen transfer cases={}, sha256={}", len(cases), digest)


def evaluate(args: argparse.Namespace) -> None:
    """Run a frozen checkpoint and preserve every raw generation and score."""
    cases = DATASET_ADAPTER.validate_json(args.dataset.read_text(encoding="utf-8"))
    validate(cases)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    backend = None
    config = None
    model_id = args.backend
    revision = "frozen-v1"
    if args.backend in ("local", "cloud"):
        model_id, revision = (
            (LOCAL_ID, LOCAL_REVISION) if args.backend == "local" else (CLOUD_ID, CLOUD_REVISION)
        )
        config = inference_config(model_id, revision)
        if args.backend == "cloud":
            from run_kaggle_selection import KaggleInferenceBackend

            backend = KaggleInferenceBackend(
                config, args.adapter, args.dataset, args.training_dataset
            )
        elif args.adapter is not None:
            backend = AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
        else:
            backend = TransformersBackend(config)
    manifest = {
        "protocol": PROTOCOL,
        "dataset_sha256": dataset_hash,
        "prompt_version": "cross_task_selection_v1",
        "model_id": model_id,
        "revision": revision,
        "config": config.model_dump(mode="json") if config is not None else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "runtime": backend.metadata() if backend is not None else {},
        "cases": len(cases),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    destination = args.output / digest[:16]
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows = []
    for index, case in enumerate(cases, 1):
        raw = (
            json.dumps({"evidence": symbolic_select(case)})
            if backend is None
            else checkpoint(selection_prompt(case), destination, backend).generation.text
        )
        rows.append({**score(case, raw), "raw_response": raw})
        if index % 10 == 0:
            logger.info("Transfer: arm={}, completed={}/{}", args.backend, index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Transfer complete: path={}", destination)


@logger.catch(reraise=True)
def main() -> None:
    """Prepare or evaluate the held-out cross-task transfer dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("symbolic", "local", "cloud"))
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.mode == "prepare":
        prepare(args.dataset)
        return
    if args.output is None or args.backend is None:
        parser.error("Evaluation requires --output and --backend")
    if args.adapter is not None and args.training_dataset is None:
        parser.error("Adapter evaluation requires --training-dataset")
    evaluate(args)


if __name__ == "__main__":
    main()
