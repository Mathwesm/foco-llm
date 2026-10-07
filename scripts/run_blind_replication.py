"""Freeze and evaluate a larger paired replication of arithmetic selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint, score_selection
from run_blind_selection import (
    CLOUD_ID,
    CLOUD_REVISION,
    LOCAL_ID,
    LOCAL_REVISION,
    _problem,
    inference_config,
)

from foco_llm.core.arithmetic_selection import selection_prompt_v2
from foco_llm.models.experiment import Problem
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

SEED = 20261004
BASES = 60
RELEVANT_FACTS = 5
FORMS = ("explicit", "alias")
PROTOCOL = "blind-arithmetic-replication-v2"
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])


def generate() -> tuple[Problem, ...]:
    """Create unseen numeric instances with paired surface forms."""
    rng = random.Random(SEED)  # noqa: S311 -- deterministic experimental seed
    cases: list[Problem] = []
    for index in range(BASES):
        state = rng.getstate()
        for form in FORMS:
            rng.setstate(state)
            problem = _problem(index + 10000, form, rng)
            cases.append(
                problem.model_copy(
                    update={
                        "id": f"replication-arithmetic-{index:04d}:{form}",
                        "base_id": f"replication-arithmetic-{index:04d}",
                    }
                )
            )
    return tuple(cases)


def validate(cases: tuple[Problem, ...]) -> None:
    """Check pairing and that the gold evidence can be executed."""
    if len(cases) != BASES * len(FORMS):
        raise ValueError("Replication dataset size differs from the protocol")
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("Replication dataset contains duplicate IDs")
    for explicit, alias in zip(cases[::2], cases[1::2], strict=True):
        if explicit.base_id != alias.base_id or explicit.answer != alias.answer:
            raise ValueError("Paired references differ")
        for case in (explicit, alias):
            if len(case.evidence) != RELEVANT_FACTS:
                raise ValueError("Reference must contain five relevant facts")
            raw = json.dumps({"evidence": case.evidence})
            if score_selection(case, raw)["answer_correct"] is not True:
                raise ValueError("Reference cannot be executed")


def symbolic_select(problem: Problem) -> tuple[str, ...]:
    """Resolve a declared alias using visible text and select that entity's updates."""
    target_match = re.search(r"remain in (\w+)\?", problem.question)
    if target_match is None:
        raise ValueError("Question has no target")
    target = target_match[1]
    initial = next(
        (
            fact
            for fact in problem.facts
            if fact.text.startswith(f"The initial marble count of {target} ")
        ),
        None,
    )
    if initial is None:
        raise ValueError("Target has no initial count")
    alias_match = re.search(r"\(also called the ([^)]+)\)", initial.text)
    name = f"the {alias_match[1]}" if alias_match else target
    return tuple(
        fact.id
        for fact in problem.facts
        if fact.id == initial.id or fact.text.startswith(f"For {name}, ")
    )


def prepare(dataset: Path) -> None:
    """Publish the fixed labels and cryptographic identity before inference."""
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
                "base_problems": BASES,
                "forms": FORMS,
                "dataset_sha256": digest,
                "model_prompt": "selection_prompt_v2",
                "primary_metric": "selection_exact_by_form",
                "secondary_metrics": ["answer_correct", "false_positive", "false_negative"],
                "note": "Freeze before inference; do not tune prompts or adapters on this set.",
            },
            indent=2,
        ),
    )
    logger.info("Frozen replication: cases={}, sha256={}", len(cases), digest)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate by form while retaining each paired base in raw scores."""
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


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate one frozen arm and preserve every raw continuation."""
    cases = DATASET_ADAPTER.validate_json(args.dataset.read_text(encoding="utf-8"))
    validate(cases)
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    backend = None
    config = None
    model_id = args.backend
    revision = "frozen-v2"
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
            else checkpoint(selection_prompt_v2(case), destination, backend).generation.text
        )
        rows.append({**score_selection(case, raw), "raw_response": raw})
        if index % 10 == 0:
            logger.info("Replication: arm={}, completed={}/{}", args.backend, index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Replication complete: path={}", destination)


@logger.catch(reraise=True)
def main() -> None:
    """Freeze or evaluate the replication dataset."""
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
