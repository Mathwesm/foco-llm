"""Independent reference solvers for the controlled pilot grammar."""

from __future__ import annotations

import re

from foco_llm.models.experiment import Condition, Problem, Task

ARITHMETIC_FACT_COUNT = 3


class DatasetError(ValueError):
    """Invalid or inconsistent experimental data."""


def _arithmetic(problem: Problem) -> str:
    quantities = []
    for fact in problem.facts:
        if "container A" in fact.text or "Container A" in fact.text:
            number = re.search(r"\b\d+\b", fact.text)
            if number is None:
                raise DatasetError("Arithmetic fact is missing a quantity")
            quantity = int(number.group())
            quantities.append(-quantity if "removed" in fact.text else quantity)
    if len(quantities) != ARITHMETIC_FACT_COUNT:
        raise DatasetError("Arithmetic problem requires three target facts")
    return str(sum(quantities))


def _deduction(problem: Problem) -> str:
    premises = "\n".join(fact.text for fact in problem.facts)
    root = re.search(r"object-\d+ has property (P\d+)\.", premises)
    if root is None:
        raise DatasetError("Deduction problem is missing its premise")
    known = {root.group(1)}
    rules = re.findall(
        r"Any object with property (P\d+) has (?:property|final category) ([PC]\d+)\.", premises
    )
    for _ in range(len(rules)):
        known.update(target for source, target in rules if source in known)
    conclusions = [value for value in known if value.startswith("C")]
    if len(conclusions) != 1:
        raise DatasetError("Deduction problem requires one reachable category")
    return conclusions[0]


def _tracking(problem: Problem) -> str:
    position = ""
    for fact in problem.facts:
        initial = re.search(r"starts in box (B\d+)\.", fact.text)
        transfer = re.search(
            r"contents of box (B\d+) are (?:then )?moved into box (B\d+)\.", fact.text
        )
        if initial:
            position = initial.group(1)
        if transfer and transfer.group(1) == position:
            position = transfer.group(2)
    if not position:
        raise DatasetError("Tracking problem is missing its initial state")
    return position


def solve(problem: Problem) -> str:
    """Solve the pilot grammar without consulting its stored answer.

    Args:
        problem: Generated problem containing all presented facts.

    Returns:
        Answer computed from the controlled sentences.

    Raises:
        DatasetError: Required facts are missing or ambiguous.
    """
    solvers = {Task.ARITHMETIC: _arithmetic, Task.DEDUCTION: _deduction, Task.TRACKING: _tracking}
    return solvers[problem.task](problem)


def _necessary_evidence(problem: Problem) -> set[str]:
    necessary = set()
    for fact in problem.facts:
        reduced = problem.model_copy(
            update={"facts": tuple(f for f in problem.facts if f.id != fact.id)}
        )
        try:
            answer = solve(reduced)
        except DatasetError:
            necessary.add(fact.id)
        else:
            if answer != problem.answer:
                necessary.add(fact.id)
    return necessary


def validate_dataset(problems: tuple[Problem, ...]) -> None:
    """Reject duplicate IDs, label errors, leakage, and unpaired variants.

    Args:
        problems: Complete generated dataset, not an inference-only view.

    Raises:
        DatasetError: Any scientific invariant fails.
    """
    if not problems:
        raise DatasetError("Dataset must not be empty")
    if len({problem.id for problem in problems}) != len(problems):
        raise DatasetError("Problem identifiers must be unique")
    groups: dict[str, list[Problem]] = {}
    for problem in problems:
        if solve(problem) != problem.answer:
            raise DatasetError(f"Reference answer mismatch: {problem.id}")
        if _necessary_evidence(problem) != set(problem.evidence):
            raise DatasetError(f"Evidence annotations do not match necessary facts: {problem.id}")
        groups.setdefault(problem.base_id, []).append(problem)
    for variants in groups.values():
        if len({problem.split for problem in variants}) != 1:
            raise DatasetError("Variants of one problem must share a split")
        if len({problem.answer for problem in variants}) != 1:
            raise DatasetError("Distractors must preserve the answer")
        if len(variants) != len(Condition) or {p.condition for p in variants} != set(Condition):
            raise DatasetError("Each base problem requires all conditions exactly once")
