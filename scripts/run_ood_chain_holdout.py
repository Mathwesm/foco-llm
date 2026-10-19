"""Freeze an out-of-template chain-selection holdout before model inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from random import Random

from loguru import logger
from pydantic import TypeAdapter
from run_cross_task_transfer import validate

from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task
from foco_llm.services.artifacts import publish_text
from foco_llm.utils.logger import setup_logging

SEED = 20261008
BASES_PER_TASK = 30
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])
UNRELATED = (
    "A clock in the hallway stopped yesterday.",
    "The kitchen shelf is made of oak.",
    "A museum ticket was printed on Tuesday.",
    "The distant market closes at sunset.",
    "A photograph hangs beside the entrance.",
    "The orchard has a stone footpath.",
)


def _chain(task: Task, index: int) -> tuple[list[str], list[str], str, str]:
    """Create two disjoint chains with shuffled edge and sentence order."""
    offset = 0 if task == Task.DEDUCTION else 10_000
    rng = Random(SEED + offset + index)  # noqa: S311 -- reproducible data, not secrets.
    codes = rng.sample(range(10_000, 99_999), 12)
    target = f"S{index + 1}" if task == Task.DEDUCTION else f"token-{index + 1}"
    rival = f"other-{index + 1}"
    if task == Task.DEDUCTION:
        nodes = [f"T{code}" for code in codes]
        relevant = [f"The specimen {target} is marked with trait {nodes[0]}."]
        similar = [f"The specimen {rival} is marked with trait {nodes[6]}."]
        for chain, start in ((relevant, 0), (similar, 6)):
            for step in range(5):
                source, destination = nodes[start + step : start + step + 2]
                sentence = (
                    f"Trait {source} invariably implies trait {destination}."
                    if step % 2 == 0
                    else f"Possessing trait {source} guarantees trait {destination}."
                )
                chain.append(sentence)
        question = f"Which trait is ultimately implied for the specimen {target}?"
    else:
        nodes = [f"C{code}" for code in codes]
        relevant = [f"Before the transfers, the {target} rests in crate {nodes[0]}."]
        similar = [f"Before the transfers, the {rival} rests in crate {nodes[6]}."]
        for chain, start in ((relevant, 0), (similar, 6)):
            for step in range(5):
                source, destination = nodes[start + step : start + step + 2]
                sentence = (
                    f"At minute {step + 1}, everything in crate {source} "
                    f"was transferred into crate {destination}."
                )
                chain.append(sentence)
        question = f"Which crate contains the {target} after the fifth transfer?"
    rng.shuffle(relevant)
    rng.shuffle(similar)
    return relevant, similar, question, nodes[5]


def _variant(task: Task, index: int, condition: Condition) -> Problem:
    """Pair identical relevant facts with unrelated or competing-chain noise."""
    relevant, similar, question, answer = _chain(task, index)
    noise = list(UNRELATED) if condition == Condition.UNRELATED else similar
    rng = Random(  # noqa: S311 -- reproducible paired layouts, not secrets.
        SEED + (0 if task == Task.DEDUCTION else 10_000) + index + 100_000
    )
    rng.shuffle(noise)
    slots = set(rng.sample(range(12), 6))
    needed, distractors = iter(relevant), iter(noise)
    tagged = [
        (next(needed), True) if position in slots else (next(distractors), False)
        for position in range(12)
    ]
    facts = tuple(
        Fact(id=f"F{position + 1}", text=text) for position, (text, _) in enumerate(tagged)
    )
    base_id = f"ood-{task.value}-{index:04d}"
    return Problem(
        id=f"{base_id}:{condition.value}",
        base_id=base_id,
        task=task,
        split=Split.TEST,
        condition=condition,
        facts=facts,
        question=question,
        answer=answer,
        evidence=tuple(
            fact.id for fact, (_, is_needed) in zip(facts, tagged, strict=True) if is_needed
        ),
    )


def _resolve(problem: Problem) -> tuple[str, tuple[str, ...]]:
    """Parse the novel grammar and follow the target chain independently."""
    target = re.search(r"(?:specimen|the) (S\d+|token-\d+)", problem.question)
    if target is None:
        raise ValueError("Question has no target")
    if problem.task == Task.DEDUCTION:
        initial = rf"The specimen {re.escape(target[1])} is marked with trait (T\d+)\."
        edges = (
            r"Trait (T\d+) invariably implies trait (T\d+)\.",
            r"Possessing trait (T\d+) guarantees trait (T\d+)\.",
        )
    else:
        initial = rf"Before the transfers, the {re.escape(target[1])} rests in crate (C\d+)\."
        edges = (r"At minute \d+, everything in crate (C\d+) was transferred into crate (C\d+)\.",)
    starts = [
        (fact.id, match[1]) for fact in problem.facts if (match := re.fullmatch(initial, fact.text))
    ]
    if len(starts) != 1:
        raise ValueError("Target must have exactly one start")
    selected = {starts[0][0]}
    current = starts[0][1]
    for _ in range(5):
        matches = [
            (fact.id, match[2])
            for fact in problem.facts
            for pattern in edges
            if (match := re.fullmatch(pattern, fact.text)) and match[1] == current
        ]
        if len(matches) != 1:
            raise ValueError("Chain is missing or ambiguous")
        fact_id, current = matches[0]
        selected.add(fact_id)
    return current, tuple(fact.id for fact in problem.facts if fact.id in selected)


def generate() -> tuple[Problem, ...]:
    """Generate 30 paired bases per task and verify every gold label."""
    cases = tuple(
        _variant(task, index, condition)
        for task in (Task.DEDUCTION, Task.TRACKING)
        for index in range(BASES_PER_TASK)
        for condition in (Condition.UNRELATED, Condition.SIMILAR)
    )
    validate(cases)
    for case in cases:
        if _resolve(case) != (case.answer, case.evidence):
            raise ValueError(f"Symbolic oracle disagrees with gold for {case.id}")
    return cases


def prepare(output: Path) -> Path:
    """Persist the frozen dataset and a hash before GPU evaluation."""
    dataset = output / "dataset.json"
    payload = DATASET_ADAPTER.dump_json(generate(), indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    publish_text(dataset, payload)
    publish_text(
        output / "manifest.json",
        json.dumps(
            {
                "protocol": "out-of-template-chain-v1",
                "seed": SEED,
                "cases": 120,
                "bases_per_task": BASES_PER_TASK,
                "dataset_sha256": digest,
                "oracle_exact": 120,
                "primary_metric": "deduction_selection_exact_by_condition",
                "note": (
                    "Post-hoc new sentence templates and shuffled chain order; "
                    "no model labels used."
                ),
            },
            indent=2,
        ),
    )
    logger.info("Frozen out-of-template cases={}, sha256={}", 120, digest)
    return dataset


@logger.catch(reraise=True)
def main() -> None:
    """Generate the fixed out-of-template test without loading a model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    prepare(args.output)


if __name__ == "__main__":
    main()
