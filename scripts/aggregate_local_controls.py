"""Audit the nine local controls and summarize paired outcomes by problem base."""

import argparse
import json
import random
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.models.experiment import Problem, Split, Task
from foco_llm.services.artifacts import publish_text

ARMS = ("C1", "C2", "C3")
SEEDS = (42, 43, 44)
BOOTSTRAP_SAMPLES = 5000
BASES_PER_TASK = 10
VARIANTS_PER_BASE = 4
VALIDATION_EXAMPLES = len(Task) * BASES_PER_TASK * VARIANTS_PER_BASE


def clustered_interval(values: list[float], seed: int = 42) -> tuple[float, float]:
    """Resample whole problem bases, not their four correlated variants."""
    if not values:
        raise ValueError("Cannot bootstrap an empty set of problem bases")
    rng = random.Random(seed)  # noqa: S311 -- reproducible non-cryptographic resampling.
    n = len(values)
    samples = sorted(sum(rng.choices(values, k=n)) / n for _ in range(BOOTSTRAP_SAMPLES))
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def load_rows(root: Path, expected_ids: set[str]) -> dict[str, dict[str, bool]]:
    """Load every arm/seed and ensure none selected a different validation set."""
    results = {}
    for seed in SEEDS:
        for arm in ARMS:
            name = f"{arm}-seed-{seed}"
            directory = root / name
            if not (directory / "complete.json").exists():
                raise ValueError(f"Incomplete control: {name}")
            payload = json.loads(
                (directory / "comparison/transitions.json").read_text(encoding="utf-8")
            )
            entries = payload["examples"]
            if len(entries) != len(expected_ids) or {e["id"] for e in entries} != expected_ids:
                raise ValueError(f"Validation IDs differ: {name}")
            results[name] = {e["id"]: bool(e["after"]) for e in entries}
    return results


def summaries(
    problems: tuple[Problem, ...], outcomes: dict[str, dict[str, bool]]
) -> dict[str, object]:
    """Report seed-wise counts and paired base-level bootstrap contrasts."""
    groups = {}
    contrasts = {}
    for task in Task:
        task_cases = [p for p in problems if p.task == task]
        base_ids = sorted({p.base_id for p in task_cases})
        if len(base_ids) != BASES_PER_TASK or len(task_cases) != BASES_PER_TASK * VARIANTS_PER_BASE:
            raise ValueError("Expected ten paired bases and forty variants per task")
        for arm in ARMS:
            scores = [
                sum(outcomes[f"{arm}-seed-{seed}"][p.id] for p in task_cases) for seed in SEEDS
            ]
            groups[f"{task}:{arm}"] = {
                "scores": scores,
                "mean_correct": sum(scores) / 3,
                "denominator_per_seed": len(task_cases),
            }
        for left, right in (("C1", "C2"), ("C2", "C3")):
            values = []
            for base in base_ids:
                cases = [p for p in task_cases if p.base_id == base]
                values.append(
                    sum(
                        outcomes[f"{right}-seed-{seed}"][p.id]
                        - outcomes[f"{left}-seed-{seed}"][p.id]
                        for seed in SEEDS
                        for p in cases
                    )
                    / (len(SEEDS) * len(cases))
                )
            low, high = clustered_interval(values)
            contrasts[f"{task}:{right}-{left}"] = {
                "mean_percentage_points": 100 * sum(values) / len(values),
                "base_bootstrap_95_percentage_points": [100 * low, 100 * high],
                "problem_bases": len(values),
            }
    totals = {arm: [sum(outcomes[f"{arm}-seed-{seed}"].values()) for seed in SEEDS] for arm in ARMS}
    return {
        "protocol": "local-controls-matrix-v1",
        "seeds": list(SEEDS),
        "total_correct_by_seed": totals,
        "denominator_per_seed": len(problems),
        "groups": groups,
        "contrasts": contrasts,
        "bootstrap_scope": "problem bases conditional on three fixed training seeds",
    }


def plot_totals(report: dict[str, object]) -> Figure:
    """Return a zero-based chart with one point per seed and honest uncertainty."""
    totals = report["total_correct_by_seed"]
    if not isinstance(totals, dict):
        raise ValueError("Missing totals")
    figure = Figure(figsize=(7, 4), layout="constrained")
    axis = figure.subplots()
    colors = ("#0072B2", "#E69F00", "#009E73")
    for position, (arm, color) in enumerate(zip(ARMS, colors, strict=True)):
        values = totals[arm]
        axis.bar(position, sum(values) / len(values), color=color, width=0.6)
        axis.scatter(
            [position - 0.14, position, position + 0.14], values, color="black", s=18, zorder=3
        )
    axis.set(
        xticks=range(3),
        xticklabels=list(ARMS),
        ylim=(0, 120),
        xlabel="Training control (three seeds each)",
        ylabel="Correct answers (out of 120)",
        title="Local controls on frozen validation set",
    )
    return figure


def main() -> None:
    """Consolidate only fully verified controls without opening the final test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (args.baseline / "validation.json").read_text(encoding="utf-8")
    )
    if len(problems) != VALIDATION_EXAMPLES or any(p.split != Split.VALIDATION for p in problems):
        raise ValueError("Expected the frozen 120-case validation set")
    outcomes = load_rows(args.root, {p.id for p in problems})
    report = summaries(problems, outcomes)
    publish_text(args.output / "report.json", json.dumps(report, indent=2))
    chart = args.output / "accuracy-by-control.png"
    if not chart.exists():
        plot_totals(report).savefig(chart, dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    main()
