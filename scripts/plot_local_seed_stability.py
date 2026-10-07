"""Plot exact selection across three local training seeds and frozen datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

plt.switch_backend("Agg")
REPLICATION_BASES = 60
ABLATION_BASES = 20


class GroupSummary(BaseModel):
    """Exact-selection count for one frozen condition."""

    model_config = ConfigDict(extra="ignore")
    cases: int = Field(gt=0)
    selection_exact: int = Field(ge=0)


def read_counts(path: Path) -> tuple[list[int], list[list[int]], list[list[int]]]:
    """Validate and return the three-seed exact-selection matrices.

    Args:
        path: Combined local replication and ablation artifact.

    Returns:
        Seed labels, two replication series, and three ablation series.

    Raises:
        ValueError: The bundle lacks a required seed, form, or denominator.
    """
    bundle = json.loads(path.read_text(encoding="utf-8"))
    arms = sorted(bundle["arms"], key=lambda arm: arm["seed"])
    seeds = [arm["seed"] for arm in arms]
    if seeds != [42, 43, 44]:
        raise ValueError("Expected exactly the three local seeds 42, 43, and 44")
    replication = [[], []]
    ablation = [[], [], []]
    for arm in arms:
        for index, form in enumerate(("explicit", "alias")):
            data = arm["experiments"]["replication"]["summary"]["groups"][form]
            group = GroupSummary.model_validate(data)
            if group.cases != REPLICATION_BASES:
                raise ValueError("Replication denominator differs from 60")
            replication[index].append(group.selection_exact)
        for index, form in enumerate(("full", "remove_relevant", "remove_irrelevant")):
            data = arm["experiments"]["ablation"]["summary"]["groups"][form]
            group = GroupSummary.model_validate(data)
            if group.cases != ABLATION_BASES:
                raise ValueError("Ablation denominator differs from 20")
            ablation[index].append(group.selection_exact)
    return seeds, replication, ablation


def plot_counts(
    seeds: list[int], replication: list[list[int]], ablation: list[list[int]]
) -> plt.Figure:
    """Return a two-panel count plot with honest zero-based axes.

    Args:
        seeds: Three training seeds.
        replication: Exact selections in explicit and alias conditions.
        ablation: Exact selections under the three fact-removal conditions.

    Returns:
        Matplotlib figure ready to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = list(range(len(seeds)))
    colors = ["#327AA5", "#D89032", "#629C79"]
    for index, (values, label) in enumerate(zip(replication, ("Explicit", "Alias"), strict=True)):
        positions = [item + (index - 0.5) * 0.32 for item in x]
        bars = axes[0].bar(positions, values, width=0.29, color=colors[index], label=label)
        axes[0].bar_label(bars, padding=2, fontsize=8)
    axes[0].set(
        title="Blind same-domain replication",
        ylabel="Exact selections (out of 60)",
        ylim=(0, 65),
        xticks=x,
        xticklabels=[str(seed) for seed in seeds],
    )
    axes[0].set_yticks(range(0, 61, 10))
    axes[0].legend(frameon=False)
    for index, (values, label) in enumerate(
        zip(ablation, ("Full", "Relevant removed", "Irrelevant removed"), strict=True)
    ):
        positions = [item + (index - 1) * 0.25 for item in x]
        bars = axes[1].bar(positions, values, width=0.23, color=colors[index], label=label)
        axes[1].bar_label(bars, padding=2, fontsize=8)
    axes[1].set(
        title="Fact-removal ablation",
        ylabel="Exact selections (out of 20)",
        ylim=(0, 21),
        xticks=x,
        xticklabels=[str(seed) for seed in seeds],
    )
    axes[1].set_yticks(range(0, 21, 5))
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel("Training seed")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.2, top=0.86, wspace=0.32)
    return fig


def main() -> None:
    """Read the preserved scores and save one high-resolution figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Figure already exists; choose a new output path")
    seeds, replication, ablation = read_counts(args.bundle)
    figure = plot_counts(seeds, replication, ablation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
