"""Audit paired evidence selection in a saved Kaggle output archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

EXPECTED_CASES = 160
CASES_PER_GROUP = 20


def _read_json(archive: ZipFile, suffix: str) -> Any:
    """Read the unique archive member ending with the requested suffix."""
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {suffix} member, found {len(matches)}")
    return json.loads(archive.read(matches[0]))


def _index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index rows by case ID and reject duplicates."""
    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate case IDs in Kaggle output")
    return indexed


def _paired_counts(
    base: dict[str, dict[str, Any]],
    adapted: dict[str, dict[str, Any]],
    case_ids: list[str],
) -> dict[str, int]:
    """Count shared success, gains, losses and shared failure."""
    pairs = Counter(
        (bool(base[case_id]["selection_exact"]), bool(adapted[case_id]["selection_exact"]))
        for case_id in case_ids
    )
    return {
        "both_correct": pairs[True, True],
        "adapter_gain": pairs[False, True],
        "adapter_loss": pairs[True, False],
        "both_wrong": pairs[False, False],
    }


def _bootstrap_interval(
    base: dict[str, dict[str, Any]],
    adapted: dict[str, dict[str, Any]],
    dataset: dict[str, dict[str, Any]],
    tasks: tuple[str, ...],
    iterations: int = 20_000,
) -> list[float]:
    """Resample independent bases within tasks for an exploratory interval."""
    rng = np.random.default_rng(42)
    task_differences: list[np.ndarray] = []
    for task in tasks:
        ids = sorted({row["base_id"] for row in dataset.values() if row["task"] == task})
        differences = np.asarray(
            [
                np.mean(
                    [
                        int(adapted[case_id]["selection_exact"])
                        - int(base[case_id]["selection_exact"])
                        for case_id in dataset
                        if dataset[case_id]["base_id"] == base_id
                    ]
                )
                for base_id in ids
            ]
        )
        sampled = rng.choice(differences, size=(iterations, len(ids)), replace=True)
        task_differences.append(sampled.mean(axis=1))
    combined = np.mean(task_differences, axis=0)
    return [round(float(value), 4) for value in np.quantile(combined, [0.025, 0.975])]


def analyze(archive_path: Path) -> dict[str, Any]:
    """Validate source records and summarize paired behavior."""
    with ZipFile(archive_path) as archive:
        dataset_bytes = archive.read("chain-divergence-7b/dataset.json")
        manifest = _read_json(archive, "chain-divergence-7b/manifest.json")
        if hashlib.sha256(dataset_bytes).hexdigest() != manifest["dataset_sha256"]:
            raise ValueError("Dataset SHA-256 does not match the frozen manifest")
        dataset = _index(json.loads(dataset_bytes))
        base = _index(_read_json(archive, "/base/e7a0793750e0ba6f/scores.json"))
        adapted = _index(_read_json(archive, "/specialist/fbc21b93667f0fe3/scores.json"))
    if not (set(dataset) == set(base) == set(adapted)) or len(dataset) != EXPECTED_CASES:
        raise ValueError("Dataset and model score IDs must match across 160 cases")
    if any(not row["selection_valid"] for row in [*base.values(), *adapted.values()]):
        raise ValueError("At least one response is invalid")
    groups = {}
    for task in ("deduction", "tracking"):
        for arm in ("unrelated", "early", "middle", "late"):
            ids = [
                case_id
                for case_id, row in dataset.items()
                if row["task"] == task and case_id.endswith(f":{arm}")
            ]
            if len(ids) != CASES_PER_GROUP:
                raise ValueError(f"Expected 20 cases in {task}:{arm}")
            groups[f"{task}:{arm}"] = _paired_counts(base, adapted, ids)
    all_ids = list(dataset)
    by_task = {}
    for task in ("deduction", "tracking"):
        ids = [case_id for case_id in all_ids if dataset[case_id]["task"] == task]
        by_task[task] = {
            "paired": _paired_counts(base, adapted, ids),
            "difference_interval_95": _bootstrap_interval(base, adapted, dataset, (task,)),
        }
    return {
        "source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "dataset_sha256": manifest["dataset_sha256"],
        "cases": len(dataset),
        "independent_bases": 40,
        "groups": groups,
        "by_task": by_task,
        "overall": {
            "paired": _paired_counts(base, adapted, all_ids),
            "difference_interval_95": _bootstrap_interval(
                base, adapted, dataset, ("deduction", "tracking")
            ),
        },
        "interval_note": (
            "Exploratory stratified percentile bootstrap, 20,000 resamples of 20 bases "
            "within each task; the four arms of each base remain together."
        ),
    }


def main() -> None:
    """Write a new audit JSON file without replacing previous results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
