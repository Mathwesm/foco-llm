"""Plot exact evidence-selection counts from a saved chain-divergence summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict, Field

from foco_llm.utils.logger import setup_logging

TASKS = ("deduction", "tracking")
ARMS = ("unrelated", "early", "middle", "late")
COLORS = {"base": "#0072B2", "deduction_specialist": "#E69F00"}
CASES_PER_GROUP = 20


class GroupScore(BaseModel):
    """Validate one reported task-arm group before plotting."""

    model_config = ConfigDict(extra="forbid")
    cases: int = Field(ge=1)
    valid: int = Field(ge=0)
    exact: int = Field(ge=0)
    false_positive: int = Field(ge=0)
    false_negative: int = Field(ge=0)


class Summary(BaseModel):
    """Validate the transcribed saved-run summary."""

    model_config = ConfigDict(extra="forbid")
    protocol: str
    source_url: str
    source_commit: str
    dataset_sha256: str
    cases_per_model: int = Field(ge=1)
    bases: int = Field(ge=1)
    source_note: str
    groups: dict[str, dict[str, GroupScore]]


def plot(summary: Summary) -> Figure:
    """Draw paired model bars without truncating the accuracy scale.

    Args:
        summary: Validated per-task and per-arm score counts.

    Returns:
        Figure for caller-controlled export.

    Raises:
        ValueError: If a required group is missing or has an invalid count.
    """
    if set(summary.groups) != set(COLORS):
        raise ValueError("Expected base and deduction_specialist groups")
    figure = Figure(figsize=(12, 6), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    labels = ("Unrelated", "Early", "Middle", "Late")
    for axis, task in zip(axes, TASKS, strict=True):
        for model_index, (model, color) in enumerate(COLORS.items()):
            scores = []
            for arm in ARMS:
                group = summary.groups[model].get(f"{task}:{arm}")
                if group is None or group.cases != CASES_PER_GROUP or group.exact > group.cases:
                    raise ValueError(f"Invalid or missing score for {model}:{task}:{arm}")
                scores.append(group.exact)
            positions = [index + (model_index - 0.5) * 0.36 for index in range(4)]
            bars = axis.bar(
                positions,
                scores,
                width=0.35,
                color=color,
                label="7B base" if model == "base" else "7B + adapter",
            )
            axis.bar_label(bars, padding=2, fontsize=9)
        axis.set(
            title="Deduction" if task == "deduction" else "Tracking",
            xlabel="Distractor divergence point",
            xticks=list(range(4)),
            xticklabels=labels,
            ylim=(0, 20),
        )
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Exact evidence selections (out of 20)")
    axes[1].legend(loc="upper right")
    figure.suptitle("Chain-divergence holdout: exact evidence selection")
    return figure


@logger.catch(reraise=True)
def main() -> None:
    """Validate the saved summary and export its figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"))
    summary = Summary.model_validate(json.loads(args.input.read_text(encoding="utf-8")))
    figure = plot(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing figure: {args.output}")
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    logger.info("Saved chain divergence figure to {}", args.output)


if __name__ == "__main__":
    main()
