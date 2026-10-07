"""Freeze and evaluate fresh paired arithmetic evidence-selection problems."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint, score_selection

from foco_llm.core.arithmetic_selection import lexical_select, selection_prompt_v2
from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

SEED = 20261003
BASES = 20
RELEVANT_FACTS = 5
FORMS = ("explicit", "alias")
COLORS = ("blue", "red", "green", "yellow", "purple", "orange")
ALIASES = ("primary jar", "main box", "first basket", "chosen bin", "marked crate")
LOCAL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_REVISION = (
    "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pragma: allowlist secret -- public revision
)
CLOUD_ID = "Qwen/Qwen2.5-7B-Instruct"
CLOUD_REVISION = (
    "a09a35458c702b33eeacc393d103063234e8bc28"  # pragma: allowlist secret -- public revision
)


def _update(name: str, amount: int, is_addition: bool) -> str:
    """Keep arithmetic phrasing compatible with the fixed executor."""
    operation = "added to" if is_addition else "removed from"
    return f"For {name}, {amount} marbles are {operation} it."


def _problem(index: int, form: str, rng: random.Random) -> Problem:
    """Build one item with private evidence labels and visible facts."""
    target = COLORS[index % len(COLORS)]
    other = COLORS[(index + 1) % len(COLORS)]
    alias = ALIASES[index % len(ALIASES)]
    initial = rng.randint(45, 85)
    other_initial = rng.randint(25, 80)
    changes = [(rng.randint(2, 12), rng.choice((True, False))) for _ in range(4)]
    other_changes = [(rng.randint(2, 12), rng.choice((True, False))) for _ in range(4)]
    if form == "explicit":
        target_initial = f"The initial marble count of {target} is {initial}."
        other_initial_text = f"The initial marble count of {other} is {other_initial}."
        update_name = target
    else:
        target_initial = (
            f"The initial marble count of {target} (also called the {alias}) is {initial}."
        )
        other_initial_text = (
            f"The initial marble count of {other} (the jar with a {target} lid) "
            f"is {other_initial}."
        )
        update_name = f"the {alias}"
    updates = [(_update(update_name, amount, sign), True) for amount, sign in changes]
    distractions = [(_update(other, amount, sign), False) for amount, sign in other_changes]
    # Relevant operations retain their original order, with matched distracting operations.
    ordered = [(target_initial, True), (other_initial_text, False)]
    for relevant, irrelevant in zip(updates, distractions, strict=True):
        if rng.choice((True, False)):
            ordered.extend((relevant, irrelevant))
        else:
            ordered.extend((irrelevant, relevant))
    facts = tuple(
        Fact(id=f"F{position}", text=text) for position, (text, _) in enumerate(ordered, 1)
    )
    evidence = tuple(fact.id for fact, (_, needed) in zip(facts, ordered, strict=True) if needed)
    answer = initial + sum(amount if sign else -amount for amount, sign in changes)
    base_id = f"blind-arithmetic-{index:04d}"
    return Problem(
        id=f"{base_id}:{form}",
        base_id=base_id,
        task=Task.ARITHMETIC,
        split=Split.TEST,
        condition=Condition.CLEAN if form == "explicit" else Condition.SIMILAR,
        facts=facts,
        question=f"How many marbles remain in {target}?",
        answer=str(answer),
        evidence=evidence,
    )


def generate() -> tuple[Problem, ...]:
    """Create twenty paired bases with a fixed seed and no model feedback."""
    # Deterministic synthetic data generation is not a cryptographic use of randomness.
    rng = random.Random(SEED)  # noqa: S311
    pairs: list[Problem] = []
    for index in range(BASES):
        # Reuse numeric values within the pair while changing only linguistic cues.
        state = rng.getstate()
        explicit = _problem(index, "explicit", rng)
        rng.setstate(state)
        alias = _problem(index, "alias", rng)
        pairs.extend((explicit, alias))
    return tuple(pairs)


def _validate(problems: tuple[Problem, ...]) -> None:
    """Reject malformed pairs and labels before opening the test to models."""
    if len(problems) != BASES * len(FORMS):
        raise ValueError("Blind dataset size differs from the frozen design")
    for explicit, alias in zip(problems[::2], problems[1::2], strict=True):
        if explicit.base_id != alias.base_id or explicit.answer != alias.answer:
            raise ValueError("Paired questions or answers differ")
        if len(explicit.evidence) != RELEVANT_FACTS or len(alias.evidence) != RELEVANT_FACTS:
            raise ValueError("Every pair must have one initial fact and four updates")
        if (
            score_selection(explicit, json.dumps({"evidence": explicit.evidence}))["answer_correct"]
            is not True
        ):
            raise ValueError("Explicit reference cannot be executed")
        if (
            score_selection(alias, json.dumps({"evidence": alias.evidence}))["answer_correct"]
            is not True
        ):
            raise ValueError("Alias reference cannot be executed")


def prepare(dataset: Path) -> None:
    """Publish one immutable test set and its cryptographic identity."""
    problems = generate()
    _validate(problems)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    publish_text(dataset, payload)
    publish_text(
        dataset.with_name("manifest.json"),
        json.dumps(
            {
                "protocol": "blind-arithmetic-selection-v1",
                "seed": SEED,
                "base_problems": BASES,
                "forms": list(FORMS),
                "dataset_sha256": digest,
                "note": (
                    "Freeze before any model inference; no hyperparameter selection " "on this set."
                ),
            },
            indent=2,
        ),
    )
    logger.info("Frozen blind dataset: cases={}, sha256={}", len(problems), digest)


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    """Report dependent paired cases with explicit denominators."""
    groups = {}
    for form in FORMS:
        selected = [row for row in rows if row["form"] == form]
        groups[form] = {
            "cases": len(selected),
            "selection_exact": sum(bool(row["selection_exact"]) for row in selected),
            "answer_correct": sum(bool(row["answer_correct"]) for row in selected),
            "selection_valid": sum(bool(row["selection_valid"]) for row in selected),
            "false_positive": sum(cast(int, row["false_positive"]) for row in selected),
            "false_negative": sum(cast(int, row["false_negative"]) for row in selected),
        }
    return {"base_problems": BASES, "cases": len(rows), "groups": groups}


def inference_config(model_id: str, revision: str) -> InferenceConfig:
    """Record that fresh paired cases belong to the held-out test split."""
    return InferenceConfig(
        model_id=model_id,
        revision=revision,
        precision="float32" if model_id == CLOUD_ID else "bfloat16",
        split=Split.TEST,
    )


def evaluate(args: argparse.Namespace) -> None:
    """Run fixed prompts and keep raw generations plus scored outputs."""
    payload = args.dataset.read_text(encoding="utf-8")
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(payload)
    _validate(problems)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    config = None
    if args.backend == "lexical":
        backend = None
        model_id = "literal-target-rule"
        revision = "frozen-v1"
    else:
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
        "protocol": "blind-arithmetic-selection-v1",
        "dataset_sha256": dataset_hash,
        "model_id": model_id,
        "revision": revision,
        "config": config.model_dump(mode="json") if config is not None else None,
        "adapter": str(args.adapter) if args.adapter else None,
        "runtime": backend.metadata() if backend is not None else {},
        "cases": len(problems),
    }
    destination = (
        args.output
        / hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    )
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows = []
    for index, problem in enumerate(problems, 1):
        if backend is None:
            raw = json.dumps({"evidence": lexical_select(problem)})
        else:
            raw = checkpoint(selection_prompt_v2(problem), destination, backend).generation.text
        rows.append({**score_selection(problem, raw), "raw_response": raw})
        if index % 10 == 0:
            logger.info(
                "Blind evaluation: model={}, completed={}/{}", model_id, index, len(problems)
            )
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(_summary(rows), indent=2))
    logger.info("Blind evaluation complete: {}", destination)


@logger.catch(reraise=True)
def main() -> None:
    """Prepare the blind set or evaluate one frozen model arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("lexical", "local", "cloud"))
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
