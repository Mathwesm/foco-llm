"""Audit frozen test labels from visible sentences using a separate parser."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from loguru import logger

from foco_llm.models.experiment import Problem, Task

INITIAL_MARBLES = re.compile(r"Initially, there are (\d+) marbles inside (\w+)\.")
UPDATE_MARBLES = re.compile(
    r"An update to (\w+): (\d+) marble(?:s are| is) (added to|removed from) it\."
)
PROPERTY_ROOT = re.compile(r"Property (P\d+) holds for (\w+)\.")
PROPERTY_EDGE = re.compile(r"Anything with property (P\d+) also has property (P\d+)\.")
LOCATION_ROOT = re.compile(r"The location of (\w+) at time 0 is box (B\d+)\.")
LOCATION_EDGE = re.compile(r"Time (\d+): move all items from box (B\d+) into box (B\d+)\.")
TEST_DEPTH = 5
EXPECTED_EVIDENCE_COUNT = TEST_DEPTH + 1


def _question_target(problem: Problem) -> str:
    """Extract the queried entity from the visible question."""
    patterns = {
        Task.ARITHMETIC: r"remain in (\w+)\?",
        Task.DEDUCTION: r"follows for (\w+)\?",
        Task.TRACKING: r"contains (\w+) after",
    }
    match = re.search(patterns[problem.task], problem.question)
    if match is None:
        raise ValueError(f"Question target is missing: {problem.id}")
    return match.group(1)


def _audit_arithmetic(problem: Problem, target: str) -> tuple[str, set[str]]:
    """Read the initial count and signed updates from test-split sentences."""
    starts: list[tuple[str, int]] = []
    updates: list[tuple[str, int]] = []
    for fact in problem.facts:
        initial = INITIAL_MARBLES.fullmatch(fact.text)
        update = UPDATE_MARBLES.search(fact.text)
        if initial is not None and initial.group(2) == target:
            starts.append((fact.id, int(initial.group(1))))
        elif update is not None and update.group(1) == target:
            sign = 1 if update.group(3) == "added to" else -1
            updates.append((fact.id, sign * int(update.group(2))))
    if len(starts) != 1 or len(updates) != TEST_DEPTH:
        raise ValueError(f"Unexpected arithmetic chain: {problem.id}")
    answer = starts[0][1] + sum(value for _, value in updates)
    return str(answer), {starts[0][0], *(fact_id for fact_id, _ in updates)}


def _audit_deduction(problem: Problem, target: str) -> tuple[str, set[str]]:
    """Follow the unique property chain from the visible premise."""
    roots: list[tuple[str, str]] = []
    edges: dict[str, list[tuple[str, str]]] = {}
    for fact in problem.facts:
        root = PROPERTY_ROOT.fullmatch(fact.text)
        edge = PROPERTY_EDGE.fullmatch(fact.text)
        if root is not None and root.group(2) == target:
            roots.append((fact.id, root.group(1)))
        elif edge is not None:
            edges.setdefault(edge.group(1), []).append((fact.id, edge.group(2)))
    if len(roots) != 1:
        raise ValueError(f"Unexpected deduction premise: {problem.id}")
    identifiers = {roots[0][0]}
    property_code = roots[0][1]
    for _ in range(TEST_DEPTH):
        outgoing = edges.get(property_code, [])
        if len(outgoing) != 1:
            raise ValueError(f"Ambiguous deduction edge: {problem.id}")
        fact_id, property_code = outgoing[0]
        identifiers.add(fact_id)
    if property_code in edges:
        raise ValueError(f"Deduction chain continues: {problem.id}")
    return property_code, identifiers


def _audit_tracking(problem: Problem, target: str) -> tuple[str, set[str]]:
    """Replay visible timed transfers for the queried object."""
    roots: list[tuple[str, str]] = []
    events: list[tuple[int, str, str, str]] = []
    for fact in problem.facts:
        root = LOCATION_ROOT.fullmatch(fact.text)
        edge = LOCATION_EDGE.fullmatch(fact.text)
        if root is not None and root.group(1) == target:
            roots.append((fact.id, root.group(2)))
        elif edge is not None:
            events.append((int(edge.group(1)), fact.id, edge.group(2), edge.group(3)))
    if len(roots) != 1:
        raise ValueError(f"Unexpected tracking premise: {problem.id}")
    identifiers = {roots[0][0]}
    location = roots[0][1]
    for _, fact_id, source, destination in sorted(events):
        if source == location:
            identifiers.add(fact_id)
            location = destination
    if len(identifiers) != EXPECTED_EVIDENCE_COUNT:
        raise ValueError(f"Unexpected tracking chain: {problem.id}")
    return location, identifiers


def audit_problem(problem: Problem) -> dict[str, object]:
    """Compare a separately parsed answer and evidence with private labels.

    Args:
        problem: A validated problem whose question and facts are visible.

    Returns:
        An audit record with the parsed values and match flags.

    Raises:
        ValueError: If the visible chain is ambiguous or malformed.
    """
    target = _question_target(problem)
    solvers = {
        Task.ARITHMETIC: _audit_arithmetic,
        Task.DEDUCTION: _audit_deduction,
        Task.TRACKING: _audit_tracking,
    }
    answer, evidence = solvers[problem.task](problem, target)
    return {
        "id": problem.id,
        "task": problem.task.value,
        "condition": problem.condition.value,
        "answer_matches": answer == problem.answer,
        "evidence_matches": evidence == set(problem.evidence),
        "parsed_answer": answer,
        "parsed_evidence": sorted(evidence),
    }


def audit_dataset(path: Path) -> dict[str, object]:
    """Audit all test records without importing the generator or reference solver.

    Args:
        path: Frozen JSON dataset path.

    Returns:
        Summary counts and every discrepant case.

    Raises:
        ValueError: If the dataset is malformed or contains non-test cases.
    """
    raw = path.read_bytes()
    decoded = json.loads(raw)
    if not isinstance(decoded, list):
        raise ValueError("Frozen dataset must contain a list")
    problems = [Problem.model_validate(item) for item in decoded]
    if any(problem.split.value != "test" for problem in problems):
        raise ValueError("Frozen dataset contains a non-test problem")
    results = [audit_problem(problem) for problem in problems]
    counts = Counter(str(result["task"]) for result in results)
    failures = [
        result
        for result in results
        if not result["answer_matches"] or not result["evidence_matches"]
    ]
    return {
        "source": path.as_posix(),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "method": "Independent exact-template parser of visible test-split text",
        "total": len(results),
        "by_task": dict(sorted(counts.items())),
        "answer_failures": sum(not bool(result["answer_matches"]) for result in results),
        "evidence_failures": sum(not bool(result["evidence_matches"]) for result in results),
        "discrepancies": failures,
    }


def main() -> None:
    """Audit the frozen dataset and create a new report without overwriting one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    logger.info(
        "Audited records={} discrepancies={} output={}",
        report["total"],
        len(report["discrepancies"]),
        args.output,
    )


if __name__ == "__main__":
    main()
