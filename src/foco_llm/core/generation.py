"""Deterministic task construction with independent distractor branches."""

from __future__ import annotations

import hashlib
import random

from foco_llm.models.experiment import (
    Condition,
    Fact,
    GenerationConfig,
    Problem,
    Split,
    Task,
)

SPLIT_PERIOD = 10
VALIDATION_SLOT = 8
TEST_SLOT = 9


def _random(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))  # noqa: S311 -- seeded experiment RNG.


def _arithmetic(index: int, rng: random.Random) -> tuple[list[str], str, str, str]:
    start, received, given = (rng.randint(1, 50) for _ in range(3))
    given = min(given, start + received - 1)
    item = f"token-{index}"
    facts = [
        f"Container A initially holds {start} {item} tokens.",
        f"{received} more {item} tokens are added to container A.",
        f"{given} {item} tokens are removed from container A.",
    ]
    return (
        facts,
        f"How many {item} tokens remain in container A?",
        str(start + received - given),
        f"Container B initially holds {rng.randint(1, 99)} different tokens and gains 7.",
    )


def _deduction(index: int, rng: random.Random) -> tuple[list[str], str, str, str]:
    first, second, final = rng.sample(range(1000), 3)
    subject = f"object-{index}"
    facts = [
        f"{subject} has property P{first}.",
        f"Any object with property P{first} has property P{second}.",
        f"Any object with property P{second} has final category C{final}.",
    ]
    return (
        facts,
        f"What final category is implied for {subject}? Give the category code.",
        f"C{final}",
        f"A different object has property Q{first}. Property Q{first} implies category D{final}.",
    )


def _tracking(index: int, rng: random.Random) -> tuple[list[str], str, str, str]:
    first, second, final, unrelated = rng.sample(range(1000), 4)
    facts = [
        f"A marker named marker-{index} starts in box B{first}. All other boxes are empty.",
        f"The contents of box B{first} are moved into box B{second}.",
        f"The contents of box B{second} are then moved into box B{final}.",
    ]
    return (
        facts,
        f"Which box contains marker-{index} after these events? Give the box code.",
        f"B{final}",
        f"After these events, a separate bead is placed in box B{unrelated}.",
    )


def _split(index: int) -> Split:
    partition = index % SPLIT_PERIOD
    if partition == VALIDATION_SLOT:
        return Split.VALIDATION
    if partition == TEST_SLOT:
        return Split.TEST
    return Split.TRAIN


def _problem_variants(task: Task, index: int, seed: int) -> list[Problem]:
    base_id = f"{task.value}-{index:06d}"
    rng = _random(seed, base_id)
    builders = {Task.ARITHMETIC: _arithmetic, Task.DEDUCTION: _deduction, Task.TRACKING: _tracking}
    statements, question, answer, similar = builders[task](index, rng)
    distractors = {
        Condition.UNRELATED: "The outside of the room's door is green.",
        Condition.NUMERIC: f"A closed notebook in another room has {rng.randint(1, 99)} pages.",
        Condition.SIMILAR: similar,
    }
    variants = []
    for condition in Condition:
        tagged = [(statement, True) for statement in statements]
        if condition != Condition.CLEAN:
            tagged.insert(
                _random(seed, f"{base_id}:{condition}").randrange(4),
                (distractors[condition], False),
            )
        facts = tuple(
            Fact(id=f"F{position + 1}", text=text) for position, (text, _) in enumerate(tagged)
        )
        evidence = tuple(
            fact.id for fact, (_, relevant) in zip(facts, tagged, strict=True) if relevant
        )
        variants.append(
            Problem(
                id=f"{base_id}:{condition}",
                base_id=base_id,
                task=task,
                split=_split(index),
                condition=condition,
                facts=facts,
                question=question,
                answer=answer,
                evidence=evidence,
            )
        )
    return variants


def generate_problems(config: GenerationConfig) -> tuple[Problem, ...]:
    """Build grouped clean/distracted examples from a fixed seed.

    Args:
        config: Seed and number of underlying problems per task.

    Returns:
        All variants in stable task, problem, and condition order.
    """
    return tuple(
        problem
        for task in Task
        for index in range(config.problems_per_task)
        for problem in _problem_variants(task, index, config.seed)
    )
