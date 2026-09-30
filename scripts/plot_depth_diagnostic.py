"""Plot clean diagnostic accuracy with no overwrite or outcome filtering."""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import BaseModel, Field, model_validator

from foco_llm.models.experiment import Task


class Group(BaseModel):
    """Validated results for a task and chain depth."""

    task: Task
    depth: int = Field(ge=1, le=4)
    count: int = Field(gt=0)
    correct: int = Field(ge=0)
    readable: int = Field(ge=0)

    @model_validator(mode="after")
    def check_counts(self) -> "Group":
        """Reject impossible scores."""
        if not self.correct <= self.readable <= self.count:
            raise ValueError("Invalid diagnostic counts")
        return self


def build_chart(groups: list[Group]) -> Figure:
    """Return a common-scale chart of all prespecified task/depth groups."""
    indexed = {(g.task, g.depth): g for g in groups}
    expected = {(task, depth) for task in Task for depth in range(1, 5)}
    if len(indexed) != len(groups) or set(indexed) != expected:
        raise ValueError("Expected exactly one group per task and depth")
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    for task, color in zip(Task, ("#0072B2", "#E69F00", "#009E73"), strict=True):
        rows = [indexed[task, depth] for depth in range(1, 5)]
        axis.plot(
            [g.depth for g in rows],
            [100 * g.correct / g.count for g in rows],
            marker="o",
            color=color,
            label=task.value,
        )
    axis.set(
        xlabel="Chain depth (operations)",
        ylabel="Answer accuracy (%)",
        ylim=(0, 100),
        xticks=range(1, 5),
        title="Clean development probes: 10 paired chains per task",
    )
    axis.legend()
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
