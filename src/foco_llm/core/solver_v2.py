"""Independent parsers and reference solvers for benchmark-v2 sentences."""

from __future__ import annotations

import re

from foco_llm.models.experiment import Problem, Task

EDGE_ENDPOINTS = 2


def _target(problem: Problem) -> str:
    patterns = {
        Task.ARITHMETIC: r"remain in (\w+)\?",
        Task.DEDUCTION: r"follows for (\w+)\?",
        Task.TRACKING: r"contains (\w+) after",
    }
    match = re.search(patterns[problem.task], problem.question)
    if match is None:
        raise ValueError("Missing target in v2 question")
    return match[1]


def _arithmetic(problem: Problem, target: str) -> str:
    total = 0
    initial = 0
    for fact in problem.facts:
        if not re.search(rf"\b{re.escape(target)}\b", fact.text):
            continue
        number = re.search(r"\b\d+\b", fact.text)
        if number is None:
            raise ValueError("Missing arithmetic quantity")
        value = int(number[0])
        if "removed from" in fact.text:
            total -= value
        else:
            total += value
        initial += "initial" in fact.text.lower()
    if initial != 1:
        raise ValueError("Expected one initial arithmetic fact")
    return str(total)


def _deduction(problem: Problem, target: str) -> str:
    roots = []
    rules: list[tuple[str, str]] = []
    for fact in problem.facts:
        codes = re.findall(r"\bP\d+\b", fact.text)
        if re.search(rf"\b{re.escape(target)}\b", fact.text) and len(codes) == 1:
            roots.append(codes[0])
        elif len(codes) == EDGE_ENDPOINTS:
            rules.append((codes[0], codes[1]))
    if len(roots) != 1:
        raise ValueError("Expected one deduction premise")
    known = set(roots)
    for _ in rules:
        known.update(b for a, b in rules if a in known)
    leaves = known - {a for a, _ in rules}
    if len(leaves) != 1:
        raise ValueError("Expected one reachable terminal property")
    return str(next(iter(leaves)))


def _tracking(problem: Problem, target: str) -> str:
    position = ""
    events: list[tuple[int, str, str]] = []
    for fact in problem.facts:
        boxes = re.findall(r"\bB\d+\b", fact.text)
        time = re.search(r"[Tt]ime (\d+)", fact.text)
        if not boxes or time is None:
            continue
        if int(time[1]) == 0 and re.search(rf"\b{re.escape(target)}\b", fact.text):
            if position:
                raise ValueError("Multiple initial tracking locations")
            position = boxes[0]
        elif len(boxes) == EDGE_ENDPOINTS:
            events.append((int(time[1]), boxes[0], boxes[1]))
    if not position:
        raise ValueError("Missing initial tracking location")
    for _, source, destination in sorted(events):
        if source == position:
            position = destination
    return position


def solve_v2(problem: Problem) -> str:
    """Derive an answer from the question and facts, without reading annotations."""
    solver = {Task.ARITHMETIC: _arithmetic, Task.DEDUCTION: _deduction, Task.TRACKING: _tracking}
    return solver[problem.task](problem, _target(problem))
