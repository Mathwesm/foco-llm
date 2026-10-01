"""Build calculation, text, and distraction probes from held-out arithmetic cases."""

import re

from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task

PROBE_DEPTHS = (1, 2, 4)


def _quantity(fact: Fact) -> int:
    match = re.search(r"\b\d+\b", fact.text)
    if match is None:
        raise ValueError("Arithmetic fact has no quantity")
    return int(match[0])


def _parts(clean: Problem) -> tuple[Fact, tuple[Fact, ...]]:
    relevant = tuple(f for f in clean.facts if f.id in clean.evidence)
    initial = tuple(f for f in relevant if "initial" in f.text.lower())
    updates = tuple(f for f in relevant if f not in initial)
    if len(initial) != 1 or len(updates) != PROBE_DEPTHS[-1]:
        raise ValueError("Diagnostic requires one initial count and four updates")
    return initial[0], updates


def _expected(initial: Fact, updates: tuple[Fact, ...]) -> tuple[str, str]:
    total = _quantity(initial)
    expression = str(total)
    for fact in updates:
        sign = "-" if "removed from" in fact.text else "+"
        value = _quantity(fact)
        total += -value if sign == "-" else value
        expression += f" {sign} {value}"
    return str(total), expression


def _probe(clean: Problem, name: str, facts: tuple[Fact, ...], answer: str) -> Problem:
    base_id = clean.base_id if name in ("text-4", "similar-4") else f"{clean.base_id}:{name}"
    condition = Condition.SIMILAR if name == "similar-4" else Condition.CLEAN
    return Problem(
        id=f"{clean.base_id}:{name}",
        base_id=base_id,
        task=Task.ARITHMETIC,
        split=Split.VALIDATION,
        condition=condition,
        facts=facts,
        question=clean.question,
        answer=answer,
        evidence=tuple(f.id for f in facts),
    )


def arithmetic_probes(clean: Problem, similar: Problem) -> tuple[Problem, ...]:
    """Create seven diagnosis cases without exposing the reference answer in prompts.

    Args:
        clean: Original held-out clean arithmetic problem.
        similar: Paired similar-distractor problem.

    Returns:
        Three expression cases, three relevant-text cases, and one noisy case.

    Raises:
        ValueError: Source cases are unpaired or violate expected depth.
    """
    if (
        clean.task != Task.ARITHMETIC
        or clean.split != Split.VALIDATION
        or clean.condition != Condition.CLEAN
        or similar.base_id != clean.base_id
        or similar.condition != Condition.SIMILAR
    ):
        raise ValueError("Diagnostic requires paired validation arithmetic cases")
    initial, updates = _parts(clean)
    cases: list[Problem] = []
    for depth in PROBE_DEPTHS:
        selected = updates[:depth]
        answer, expression = _expected(initial, selected)
        expression_fact = Fact(id="F1", text=f"To answer the question, calculate {expression}.")
        cases.append(_probe(clean, f"expression-{depth}", (expression_fact,), answer))
        text_facts = tuple(f for f in clean.facts if f == initial or f in selected)
        text_case = _probe(clean, f"text-{depth}", text_facts, answer)
        if solve_v2(text_case) != answer:
            raise ValueError("Text probe disagrees with independent arithmetic solver")
        cases.append(text_case)
    noisy = similar.model_copy(update={"id": f"{clean.base_id}:similar-4"})
    if solve_v2(noisy) != clean.answer:
        raise ValueError("Similar distractor changes the reference answer")
    cases.append(noisy)
    return tuple(cases)
