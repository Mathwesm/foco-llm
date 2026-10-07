"""Measure selector sensitivity to removing one relevant or irrelevant fact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint, score_selection
from run_blind_replication import generate
from run_blind_selection import (
    CLOUD_ID,
    CLOUD_REVISION,
    LOCAL_ID,
    LOCAL_REVISION,
    inference_config,
)

from foco_llm.core.arithmetic_selection import selected_calculation, selection_prompt_v2
from foco_llm.models.experiment import Problem
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

BASES = 20
FORMS = ("full", "remove_relevant", "remove_irrelevant")
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])


def variants() -> tuple[Problem, ...]:
    """Freeze 20 explicit bases under three controlled visible contexts."""
    cases: list[Problem] = []
    for original in generate()[::2][:BASES]:
        relevant_to_remove = original.evidence[1]
        irrelevant_to_remove = next(
            fact.id for fact in original.facts if fact.id not in original.evidence
        )
        for form, removed in (
            ("full", None),
            ("remove_relevant", relevant_to_remove),
            ("remove_irrelevant", irrelevant_to_remove),
        ):
            facts = tuple(fact for fact in original.facts if fact.id != removed)
            evidence = tuple(
                identifier for identifier in original.evidence if identifier != removed
            )
            answer = original.answer
            if form == "remove_relevant":
                answer = selected_calculation(facts, evidence)[1]
            cases.append(
                Problem.model_validate(
                    {
                        **original.model_dump(),
                        "id": f"{original.base_id}:{form}",
                        "facts": facts,
                        "evidence": evidence,
                        "answer": answer,
                    }
                )
            )
    return tuple(cases)


def prepare(path: Path) -> None:
    """Publish the frozen counterfactual cases and their digest."""
    cases = variants()
    if len(cases) != BASES * len(FORMS):
        raise ValueError("Ablation dataset size differs from protocol")
    payload = DATASET_ADAPTER.dump_json(cases, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    publish_text(path, payload)
    publish_text(
        path.with_name("manifest.json"),
        json.dumps(
            {
                "protocol": "fact-removal-ablation-v1",
                "source": "blind-arithmetic-replication-v2",
                "base_problems": BASES,
                "forms": FORMS,
                "dataset_sha256": digest,
                "note": "The removed relevant update changes the context-derived answer.",
            },
            indent=2,
        ),
    )
    logger.info("Frozen fact ablation: cases={}, sha256={}", len(cases), digest)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate exact selection and answer counts by intervention."""
    groups: dict[str, object] = {}
    for form in FORMS:
        selected = [row for row in rows if row["form"] == form]
        groups[form] = {
            "cases": len(selected),
            "selection_valid": sum(bool(row["selection_valid"]) for row in selected),
            "selection_exact": sum(bool(row["selection_exact"]) for row in selected),
            "answer_correct": sum(bool(row["answer_correct"]) for row in selected),
            "false_positive": sum(cast(int, row["false_positive"]) for row in selected),
            "false_negative": sum(cast(int, row["false_negative"]) for row in selected),
        }
    return {"base_problems": BASES, "cases": len(rows), "groups": groups}


def evaluate(args: argparse.Namespace) -> None:
    """Run one model arm and save each raw response for paired analysis."""
    cases = DATASET_ADAPTER.validate_json(args.dataset.read_text(encoding="utf-8"))
    if len(cases) != BASES * len(FORMS):
        raise ValueError("Ablation dataset size differs from protocol")
    model_id, revision = (
        (CLOUD_ID, CLOUD_REVISION) if args.backend == "cloud" else (LOCAL_ID, LOCAL_REVISION)
    )
    config = inference_config(model_id, revision)
    if args.backend == "cloud":
        from run_kaggle_selection import KaggleInferenceBackend

        backend = KaggleInferenceBackend(config, args.adapter, args.dataset, args.training_dataset)
    elif args.adapter is not None:
        backend = AdapterBackend(config, args.adapter, args.dataset, args.training_dataset)
    else:
        backend = TransformersBackend(config)
    manifest = {
        "protocol": "fact-removal-ablation-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "model_id": model_id,
        "revision": revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "runtime": backend.metadata(),
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    destination = args.output / digest[:16]
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, 1):
        raw = checkpoint(selection_prompt_v2(case), destination, backend).generation.text
        rows.append({**score_selection(case, raw), "raw_response": raw})
        if index % 10 == 0:
            logger.info("Ablation: completed={}/{}", index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Ablation complete: path={}", destination)


@logger.catch(reraise=True)
def main() -> None:
    """Prepare the frozen cases or evaluate one local arm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backend", choices=("local", "cloud"), default="local")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.mode == "prepare":
        prepare(args.dataset)
        return
    if args.output is None:
        parser.error("Evaluation requires --output")
    if args.adapter is not None and args.training_dataset is None:
        parser.error("Adapter evaluation requires --training-dataset")
    evaluate(args)


if __name__ == "__main__":
    main()
