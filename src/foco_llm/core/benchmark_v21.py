"""Audited v2.1 revision preserving relative evidence order across paired contexts."""

from __future__ import annotations

import re
from collections import defaultdict
from string import ascii_uppercase

from foco_llm.core.benchmark_v2 import _rng, generate_v2
from foco_llm.core.validation import DatasetError, validate_dataset
from foco_llm.models.experiment import Condition, Fact, GenerationConfig, Problem, Task

UNRELATED = (
    "A distant library has wooden doors.",
    "The window in another building is closed.",
    "A painter in another town likes quiet streets.",
    "The ceiling of a distant office is white.",
    "A remote garden has a narrow path.",
    "A distant room has a round table.",
)
NUMERIC = (
    ("A notebook in another room has", "page", "pages"),
    ("A distant library has", "shelf", "shelves"),
    ("A remote garden has", "tree", "trees"),
    ("A street in another town has", "building", "buildings"),
    ("A distant office has", "window", "windows"),
    ("A drawer in another room has", "pencil", "pencils"),
)


def _sentences(facts: tuple[Fact, ...], task: Task, seed: int, key: str) -> list[str]:
    texts = []
    update = 0
    labels = _rng(seed, f"v21-events:{key}").sample(ascii_uppercase, len(ascii_uppercase))
    for fact in facts:
        text = re.sub(r"\b1 marbles are\b", "1 marble is", fact.text)
        if task == Task.ARITHMETIC and ("added to" in text or "removed from" in text):
            # Distinct event labels disambiguate repeated equal additions/removals.
            text = f"Update {labels[update]}: {text}"
            update += 1
        texts.append(text)
    return texts


def _noise(problem: Problem, count: int, seed: int) -> list[str]:
    if problem.condition == Condition.SIMILAR:
        return _sentences(
            tuple(f for f in problem.facts if f.id not in problem.evidence),
            problem.task,
            seed,
            f"{problem.base_id}:other",
        )
    if problem.condition == Condition.UNRELATED:
        return list(UNRELATED[:count])
    if problem.condition == Condition.NUMERIC:
        rng = _rng(seed, f"v21-numeric:{problem.base_id}")
        output = []
        for prefix, singular, plural in NUMERIC[:count]:
            value = rng.randint(1, 99)
            output.append(f"{prefix} {value} {singular if value == 1 else plural}.")
        return output
    return []


def _variant(problem: Problem, relevant: list[str], seed: int) -> Problem:
    noise = _noise(problem, len(relevant), seed)
    rng = _rng(seed, f"v21-layout:{problem.base_id}")
    rng.shuffle(noise)
    # Sample positions, never permute the already ordered relevant sequence.
    slots = set(rng.sample(range(len(relevant) + len(noise)), len(relevant)))
    clean, extra = iter(relevant), iter(noise)
    tagged = [
        (next(clean), True) if i in slots else (next(extra), False)
        for i in range(len(relevant) + len(noise))
    ]
    facts = tuple(Fact(id=f"F{i}", text=text) for i, (text, _) in enumerate(tagged, 1))
    return Problem(
        **problem.model_dump(exclude={"facts", "evidence"}),
        facts=facts,
        evidence=tuple(f.id for f, (_, needed) in zip(facts, tagged, strict=True) if needed),
    )


def validate_v21(problems: tuple[Problem, ...]) -> None:
    """Check reference answers plus paired order and equal distractor sentence counts."""
    validate_dataset(problems)
    groups: dict[str, list[Problem]] = defaultdict(list)
    for problem in problems:
        groups[problem.base_id].append(problem)
        if any(re.search(r"\b1 (?:marbles|pages)\b", f.text) for f in problem.facts):
            raise DatasetError("Invalid singular agreement in v2.1")
    for variants in groups.values():
        clean = next(p for p in variants if p.condition == Condition.CLEAN)
        order = tuple(f.text for f in clean.facts)
        noisy_positions = {p.evidence for p in variants if p.condition != Condition.CLEAN}
        if len(noisy_positions) != 1:
            raise DatasetError("Evidence positions differ across distracted conditions")
        for variant in variants:
            actual = tuple(f.text for f in variant.facts if f.id in variant.evidence)
            if actual != order:
                raise DatasetError("Paired relevant fact order differs")
            if variant.condition != Condition.CLEAN and len(variant.facts) != len(order) * 2:
                raise DatasetError("Distractor sentence count differs from relevant fact count")


def generate_v21(config: GenerationConfig) -> tuple[Problem, ...]:
    """Revise v2 without modifying its generator or any historical dataset file."""
    original = generate_v2(config)
    clean = {
        p.base_id: _sentences(p.facts, p.task, config.seed, f"{p.base_id}:target")
        for p in original
        if p.condition == Condition.CLEAN
    }
    revised = tuple(_variant(problem, clean[problem.base_id], config.seed) for problem in original)
    validate_v21(revised)
    return revised
