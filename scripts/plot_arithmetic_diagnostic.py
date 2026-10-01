"""Plot arithmetic answer accuracy by expression depth and context form."""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

ARMS = (
    ("base-15b", "Original", "#555555"),
    ("mixed-15b", "Misto", "#0072B2"),
    ("arithmetic-15b", "Aritmética", "#E69F00"),
    ("math-prefix-15b", "Subtotais", "#009E73"),
)
DEPTHS = (1, 2, 4)


def plot(root: Path) -> Figure:
    """Return a color-blind-safe comparison of ten cases per plotted point."""
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2, sharey=True)
    for arm, label, color in ARMS:
        summary = json.loads((root / arm / "summary.json").read_text(encoding="utf-8"))
        for axis, form in zip(axes, ("expression", "text"), strict=True):
            values = [summary["groups"][f"{form}-{depth}"]["answer_correct"] for depth in DEPTHS]
            axis.plot(DEPTHS, values, marker="o", label=label, color=color)
    for axis, title in zip(axes, ("Expressão direta", "Fatos em texto"), strict=True):
        axis.set(
            title=title,
            xlabel="Atualizações aritméticas (contagem)",
            ylabel="Respostas corretas / 10" if axis == axes[0] else "",
            xticks=DEPTHS,
            ylim=(0, 10),
            xlim=(0.7, 4.3),
        )
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")
    figure.suptitle("Diagnóstico local: acurácia cai com o número de operações")
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.root).savefig(args.output, dpi=180, bbox_inches="tight")
