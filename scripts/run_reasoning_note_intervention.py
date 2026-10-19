"""Test whether model-written evidence notes affect later fact selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint
from run_blind_selection import CLOUD_ID, CLOUD_REVISION, LOCAL_ID, LOCAL_REVISION
from run_cross_task_transfer import score

from foco_llm.models.experiment import Condition, Problem, Prompt, Task
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.artifacts import publish_text
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging

CASES_PER_TASK = 10
# Public dataset digest, not a credential.
EXPECTED_DATASET_SHA256 = (
    "5a1ee80d4a2af8c27b50fc5e6fbde6eb2369388e01a21c298093447181d64951"  # pragma: allowlist secret
)
ARMS = ("no_note", "full_note", "remove_relevant", "remove_irrelevant")
LINE = re.compile(r"^\s*(F\d+)\s*:\s*(USE|SKIP)\b.+", re.IGNORECASE)
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])


def select_cases(cases: tuple[Problem, ...]) -> tuple[Problem, ...]:
    """Choose a fixed set of similar-chain distractor cases without model feedback."""
    selected = tuple(
        case
        for task in (Task.DEDUCTION, Task.TRACKING)
        for case in cases
        if case.task == task and case.condition == Condition.SIMILAR
    )
    expected = 2 * 30
    if len(selected) != expected:
        raise ValueError("Expected thirty similar-chain cases per task")
    return tuple(
        case
        for task in (Task.DEDUCTION, Task.TRACKING)
        for case in selected
        if case.task == task and int(case.base_id.rsplit("-", 1)[-1]) < CASES_PER_TASK
    )


def trace_prompt(case: Problem) -> Prompt:
    """Ask for a model-written note with one independently editable fact line."""
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in case.facts)
    return Prompt(
        id=f"{case.id}:note",
        text=(
            "For each fact below, write exactly one line in the format "
            "F1: USE or F1: SKIP, followed by a short reason. "
            "Use one fact ID per line, in the presented order. "
            "After the lines, write one final JSON object with an evidence array. "
            "Do not invent fact IDs.\n\n"
            f"{facts}\n\nQuestion: {case.question}"
        ),
    )


def parse_note(raw: str, case: Problem) -> tuple[str, ...]:
    """Accept only a complete, unambiguous model-written note."""
    lines = tuple(line.strip() for line in raw.splitlines() if LINE.fullmatch(line.strip()))
    ids = [LINE.fullmatch(line)[1].upper() for line in lines if LINE.fullmatch(line)]
    expected = [fact.id for fact in case.facts]
    if ids != expected:
        raise ValueError("Note does not contain one ordered line for every visible fact")
    if any(len(re.findall(r"\bF\d+\b", line, flags=re.IGNORECASE)) != 1 for line in lines):
        raise ValueError("Note line refers to multiple fact IDs")
    return lines


def note_variants(lines: tuple[str, ...], case: Problem) -> dict[str, str]:
    """Remove one relevant or irrelevant note line while retaining all input facts."""
    if len(lines) != len(case.facts):
        raise ValueError("Note and case have different fact counts")
    relevant = set(case.evidence)
    relevant_index = next(index for index, fact in enumerate(case.facts) if fact.id in relevant)
    irrelevant_index = next(
        index for index, fact in enumerate(case.facts) if fact.id not in relevant
    )
    return {
        "no_note": "",
        "full_note": "\n".join(lines),
        "remove_relevant": "\n".join(
            line for index, line in enumerate(lines) if index != relevant_index
        ),
        "remove_irrelevant": "\n".join(
            line for index, line in enumerate(lines) if index != irrelevant_index
        ),
    }


def decision_prompt(case: Problem, arm: str, note: str) -> Prompt:
    """Hold the original facts fixed and request a fresh final selection."""
    if arm not in ARMS:
        raise ValueError("Unknown intervention arm")
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in case.facts)
    draft = f"\n\nEarlier draft note (it may be incomplete or mistaken):\n{note}" if note else ""
    return Prompt(
        id=f"{case.id}:{arm}",
        text=(
            "Select all and only the facts needed to answer the question. "
            "Reconsider the original facts; do not simply copy a draft note. "
            'Return only JSON in the form {"evidence":["F1"]}. '
            "Do not include the answer.\n\n"
            f"{facts}\n\nQuestion: {case.question}{draft}"
        ),
    )


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate exact selections and paired changes on eligible cases."""
    groups: dict[str, dict[str, int]] = {}
    for task in (Task.DEDUCTION, Task.TRACKING):
        for arm in ARMS:
            subset = [row for row in rows if row["task"] == task.value and row["arm"] == arm]
            groups[f"{task.value}:{arm}"] = {
                "cases": len(subset),
                "valid": sum(bool(row["selection_valid"]) for row in subset),
                "exact": sum(bool(row["selection_exact"]) for row in subset),
            }
    by_case: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        by_case.setdefault(str(row["id"]), {})[str(row["arm"])] = row
    for task in (Task.DEDUCTION, Task.TRACKING):
        for arm in ARMS:
            if arm == "full_note":
                continue
            pairs = [
                variants
                for variants in by_case.values()
                if variants.get("full_note", {}).get("task") == task.value
                and arm in variants
                and bool(variants["full_note"]["selection_valid"])
                and bool(variants[arm]["selection_valid"])
            ]
            groups[f"{task.value}:{arm}"]["valid_pairs_vs_full"] = len(pairs)
            groups[f"{task.value}:{arm}"]["changed_vs_full"] = sum(
                variants[arm]["selected_ids"] != variants["full_note"]["selected_ids"]
                for variants in pairs
            )
    return {"eligible_cases": len(rows) // len(ARMS), "groups": groups}


def build_backend(
    backend_name: str, adapter: Path | None, dataset: Path, training_dataset: Path | None
) -> TransformersBackend:
    """Load one pinned local or cloud model, optionally with a verified adapter."""
    model_id, revision = (
        (LOCAL_ID, LOCAL_REVISION) if backend_name == "local" else (CLOUD_ID, CLOUD_REVISION)
    )
    config = InferenceConfig(
        model_id=model_id,
        revision=revision,
        max_input_tokens=2048,
        max_new_tokens=256,
        max_seconds=120,
        precision="bfloat16" if backend_name == "local" else "float32",
    )
    if backend_name == "cloud":
        from run_kaggle_selection import KaggleInferenceBackend

        return KaggleInferenceBackend(config, adapter, dataset, training_dataset)
    if adapter is not None:
        return AdapterBackend(config, adapter, dataset, training_dataset)
    return TransformersBackend(config)


def evaluate(args: argparse.Namespace) -> None:
    """Generate notes, intervene on them, and preserve all raw responses."""
    cases = select_cases(DATASET_ADAPTER.validate_json(args.dataset.read_bytes()))
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError("Dataset hash differs from the frozen protocol")
    backend = build_backend(args.backend, args.adapter, args.dataset, args.training_dataset)
    manifest = {
        "protocol": "model-written-note-intervention-v1",
        "dataset_sha256": dataset_hash,
        "selected_ids": [case.id for case in cases],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": backend.config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "adapter": str(args.adapter) if args.adapter else None,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
    destination = args.output / digest[:16]
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    for index, case in enumerate(cases, 1):
        raw_note = checkpoint(trace_prompt(case), destination, backend).generation.text
        try:
            lines = parse_note(raw_note, case)
        except ValueError as error:
            rejected.append({"id": case.id, "reason": str(error), "raw_note": raw_note})
            continue
        for arm, note in note_variants(lines, case).items():
            raw = checkpoint(decision_prompt(case, arm, note), destination, backend).generation.text
            rows.append({**score(case, raw), "arm": arm, "raw_note": raw_note, "raw_response": raw})
        logger.info("Note intervention completed={}/{}", index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "rejected.json", json.dumps(rejected, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))
    logger.info("Note intervention finished: eligible={}, path={}", len(rows) // 4, destination)


@logger.catch(reraise=True)
def main() -> None:
    """Run a frozen local or Kaggle note intervention."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("local", "cloud"), required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    args = parser.parse_args()
    if args.adapter is not None and args.training_dataset is None:
        parser.error("Adapter evaluation requires --training-dataset")
    setup_logging(log_dir=Path("logs"), serialize=True)
    evaluate(args)


if __name__ == "__main__":
    main()
