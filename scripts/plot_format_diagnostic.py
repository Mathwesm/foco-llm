"""Plot published format diagnostics without changing scores or denominators."""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import BaseModel, Field, model_validator

from foco_llm.models.experiment import Task


class Group(BaseModel):
    """Counted diagnostic outcomes for one arm and task."""

    arm: str
    task: Task
    count: int = Field(gt=0)
    correct: int = Field(ge=0)
    readable: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> "Group":
        """Reject impossible counts before displaying percentages."""
        if not self.correct <= self.readable <= self.count:
            raise ValueError("Invalid format diagnostic counts")
        return self


def build_chart(groups: list[Group]) -> Figure:
    """Return a labeled figure with common zero-based percentage scales."""
    figure = Figure(figsize=(10, 4), layout="constrained")
    axes = figure.subplots(1, 2)
    styles = (
        ("original", "Original", -0.25, "#0072B2"),
        ("explicit_json", "Explicit JSON", 0, "#E69F00"),
        ("plain_answer", "Plain answer", 0.25, "#009E73"),
    )
    for axis, metric, title in zip(
        axes, ("correct", "readable"), ("Answer accuracy", "Contract readability"), strict=True
    ):
        for arm, label, offset, color in styles:
            selected = [next(g for g in groups if g.arm == arm and g.task == t) for t in Task]
            values = [getattr(g, metric) / g.count * 100 for g in selected]
            axis.bar([i + offset for i in range(3)], values, width=0.25, label=label, color=color)
        axis.set(
            xticks=range(3),
            xticklabels=[t.value for t in Task],
            ylim=(0, 100),
            xlabel="Task (10 clean cases each)",
            ylabel="Percent (%)",
            title=title,
        )
    axes[0].legend(fontsize=8)
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite figure")
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    build_chart([Group.model_validate(g) for g in payload["groups"]]).savefig(
        args.output, dpi=180, bbox_inches="tight"
    )
