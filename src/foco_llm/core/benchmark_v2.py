"""Versioned synthetic benchmark with split-specific language and chain lengths."""

from __future__ import annotations

import hashlib
import random
from itertools import pairwise

from foco_llm.models.experiment import Condition, Fact, GenerationConfig, Problem, Split, Task

# Each split has a disjoint surface template. Depth also changes across splits;
# this is a joint language/depth shift, not an isolated measure of either factor.
TEMPLATES = {
    Split.TRAIN: (
        "{name} initially contains {value} marbles.",
        "{value} marbles are {verb} {name}.",
        "{name} has property {value}.",
        "Property {source} implies property {target}.",
        "At time 0, {name} starts in box {value}.",
        "At time {time}, the contents of box {source} move to box {target}.",
    ),
    Split.VALIDATION: (
        "The initial marble count of {name} is {value}.",
        "For {name}, {value} marbles are {verb} it.",
        "The object {name} possesses property {value}.",
        "Having property {source} entails property {target}.",
        "At time 0, box {value} contains {name}.",
        "At time {time}, transfer everything from box {source} to box {target}.",
    ),
    Split.TEST: (
        "Initially, there are {value} marbles inside {name}.",
        "An update to {name}: {value} marbles are {verb} it.",
        "Property {value} holds for {name}.",
        "Anything with property {source} also has property {target}.",
        "The location of {name} at time 0 is box {value}.",
        "Time {time}: move all items from box {source} into box {target}.",
    ),
}


def _rng(seed: int, key: str) -> random.Random:
    digest = hashlib.sha256(f"v2:{seed}:{key}".encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))  # noqa: S311 -- reproducible data.


def _partition(index: int) -> Split:
    return {8: Split.VALIDATION, 9: Split.TEST}.get(index % 10, Split.TRAIN)


def _arithmetic(
    name: str, depth: int, rng: random.Random, style: tuple[str, ...]
) -> tuple[list[str], str]:
    balance = rng.randint(50, 80)
    facts = [style[0].format(name=name, value=balance)]
    for _ in range(depth):
        change = rng.choice((-1, 1)) * rng.randint(1, 9)
        facts.append(
            style[1].format(
                name=name, value=abs(change), verb="added to" if change > 0 else "removed from"
            )
        )
        balance += change
    return facts, str(balance)


def _deduction(
    name: str, depth: int, rng: random.Random, style: tuple[str, ...]
) -> tuple[list[str], str]:
    codes = [f"P{number}" for number in rng.sample(range(10000), depth + 1)]
    facts = [style[2].format(name=name, value=codes[0])]
    facts.extend(style[3].format(source=a, target=b) for a, b in pairwise(codes))
    return facts, codes[-1]


def _tracking(
    name: str, depth: int, rng: random.Random, style: tuple[str, ...]
) -> tuple[list[str], str]:
    codes = [f"B{number}" for number in rng.sample(range(10000), depth + 1)]
    facts = [style[4].format(name=name, value=codes[0])]
    facts.extend(
        style[5].format(time=time, source=a, target=b)
        for time, (a, b) in enumerate(pairwise(codes), start=1)
    )
    return facts, codes[-1]


def _variants(task: Task, index: int, seed: int) -> list[Problem]:
    base_id = f"v2-{task}-{index:06d}"
    rng = _rng(seed, base_id)
    split = _partition(index)
    depth = rng.choice((2, 3)) if split == Split.TRAIN else 4 if split == Split.VALIDATION else 5
    target, other = rng.sample(("amber", "violet", "silver", "copper", "azure", "ivory"), 2)
    builder = {Task.ARITHMETIC: _arithmetic, Task.DEDUCTION: _deduction, Task.TRACKING: _tracking}[
        task
    ]
    relevant, answer = builder(target, depth, rng, TEMPLATES[split])
    similar, _ = builder(other, depth, rng, TEMPLATES[split])
    # Reject symbol collisions so independent distractor chains stay disconnected.
    while task != Task.ARITHMETIC and _codes(relevant) & _codes(similar):
        similar, _ = builder(other, depth, rng, TEMPLATES[split])
    questions = {
        Task.ARITHMETIC: f"How many marbles remain in {target}?",
        Task.DEDUCTION: f"What terminal property follows for {target}? Return its code.",
        Task.TRACKING: f"Which box contains {target} after all timed events? Return its code.",
    }
    distractors = {
        Condition.CLEAN: [],
        Condition.UNRELATED: ["The room has a green door."],
        Condition.NUMERIC: [f"A notebook has {rng.randint(1, 99)} pages."],
        Condition.SIMILAR: similar,
    }
    output = []
    for condition in Condition:
        tagged = [(text, True) for text in relevant] + [
            (text, False) for text in distractors[condition]
        ]
        _rng(seed, f"{base_id}:{condition}").shuffle(tagged)
        facts = tuple(Fact(id=f"F{i}", text=text) for i, (text, _) in enumerate(tagged, 1))
        output.append(
            Problem(
                id=f"{base_id}:{condition}",
                base_id=base_id,
                task=task,
                split=split,
                condition=condition,
                facts=facts,
                question=questions[task],
                answer=answer,
                evidence=tuple(
                    f.id for f, (_, needed) in zip(facts, tagged, strict=True) if needed
                ),
            )
        )
    return output


def _codes(sentences: list[str]) -> set[str]:
    import re

    return set(re.findall(r"\b[PB]\d+\b", " ".join(sentences)))


def generate_v2(config: GenerationConfig) -> tuple[Problem, ...]:
    """Build an immutable candidate benchmark with paired variants and shuffled facts."""
    return tuple(
        problem
        for task in Task
        for index in range(config.problems_per_task)
        for problem in _variants(task, index, config.seed)
    )
