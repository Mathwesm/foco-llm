"""Plot exact fact-selection accuracy for the arithmetic diagnostic arms."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

ARMS = (
    ("base-15b-v2", "Original", "#555555"),
    ("mixed-15b-v2", "Misto", "#0072B2"),
    ("math-prefix-15b-v2", "Subtotais", "#E69F00"),
    ("evidence-only-15b-v2", "Seleção treinada", "#009E73"),
    ("lexical-control", "Regra lexical", "#CC79A7"),
)
FORMS = ("text-1", "text-2", "text-4", "similar-4")


def plot(root: Path) -> Figure:
    """Return a grouped chart with a common zero-based count scale."""
    figure = Figure(figsize=(10, 5), layout="constrained")
    axis = figure.subplots()
    positions = np.arange(len(FORMS))
    width = 0.16
    for index, (arm, label, color) in enumerate(ARMS):
        summary = json.loads((root / arm / "summary.json").read_text(encoding="utf-8"))
        values = [summary["groups"][form]["selection_exact"] for form in FORMS]
        axis.bar(positions + (index - 2) * width, values, width, label=label, color=color)
    axis.set(
        title="Seleção exata de fatos por condição (10 casos cada)",
        xlabel="Condição do problema",
        ylabel="Seleções exatas / 10",
        xticks=positions,
        xticklabels=("Texto, 1 op.", "Texto, 2 op.", "Texto, 4 op.", "Distrator, 4 op."),
        ylim=(0, 10.6),
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.root).savefig(args.output, dpi=180, bbox_inches="tight")
