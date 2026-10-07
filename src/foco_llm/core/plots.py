"""Scientific figures derived only from supplied data and evaluation records."""

from collections import Counter

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from foco_llm.core.evaluation import EvaluationReport
from foco_llm.models.experiment import Problem


def plot_dataset(problems: tuple[Problem, ...]) -> tuple[Figure, Axes]:
    """Plot actual generated input counts, not model performance.

    Args:
        problems: Generated experimental records.

    Returns:
        Figure and axes for caller-controlled exporting.
    """
    counts = Counter(f"{p.task}\n{p.split}" for p in problems)
    figure = Figure(figsize=(12, 5), layout="constrained")
    axis = figure.subplots()
    axis.bar(list(counts), list(counts.values()), color="#0072B2")
    axis.set(
        title="Synthetic pilot composition (not model results)",
        xlabel="Task / split",
        ylabel="Examples (count)",
    )
    axis.tick_params(axis="x", labelsize=8)
    return figure, axis


def plot_paired_accuracy(report: EvaluationReport) -> tuple[Figure, Axes]:
    """Compare clean and distracted accuracies on matched examples.

    Args:
        report: Metrics produced from recorded model predictions.

    Returns:
        Scatter plot on shared zero-to-one scales, with an identity reference.
    """
    figure = Figure(figsize=(8, 7), layout="constrained")
    axis = figure.subplots()
    colors = {"unrelated": "#0072B2", "numeric": "#E69F00", "similar": "#009E73"}
    for condition, color in colors.items():
        points = [point for point in report.paired if point.condition == condition]
        axis.scatter(
            [p.clean_accuracy for p in points],
            [p.distracted_accuracy for p in points],
            color=color,
            label=condition,
            s=70,
        )
        for point in points:
            axis.annotate(
                point.task.value,
                (point.clean_accuracy, point.distracted_accuracy),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Equal accuracy")
    axis.set(
        xlim=(0, 1.03),
        ylim=(0, 1.03),
        xlabel="Clean accuracy (fraction)",
        ylabel="Distracted accuracy (fraction)",
        title=f"Paired accuracy: {report.model_id}",
    )
    axis.legend()
    return figure, axis
