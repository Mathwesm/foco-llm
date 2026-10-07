"""Publish auditable synthetic validation results without modifying original runs."""

import argparse
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.core.evaluation import EvaluationReport, evaluate
from foco_llm.core.plots import plot_paired_accuracy
from foco_llm.models.experiment import ModelRun, Problem, Split
from foco_llm.models.inference import Checkpoint
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.inference import parse_response


def build_chart(report: EvaluationReport) -> Figure:
    """Return a grouped figure with visible zero scores for the recorded model."""
    figure = Figure(figsize=(11, 5), layout="constrained")
    axis = figure.subplots()
    conditions = ("clean", "unrelated", "numeric", "similar")
    colors = ("#555555", "#0072B2", "#E69F00", "#009E73")
    tasks = sorted({group.task.value for group in report.groups})
    width = 0.19
    for index, (condition, color) in enumerate(zip(conditions, colors, strict=True)):
        values = [
            next(
                g.accuracy * 100 for g in report.groups if g.task == t and g.condition == condition
            )
            for t in tasks
        ]
        bars = axis.bar(
            [position + index * width for position in range(len(tasks))],
            values,
            width=width,
            label=condition,
            color=color,
        )
        axis.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9)
    axis.set_xticks([p + 1.5 * width for p in range(len(tasks))], tasks)
    axis.set(
        ylim=(0, 110),
        ylabel="Strict answer accuracy (%)",
        xlabel="Task",
        title=f"Validation pilot: {report.model_id}",
    )
    axis.legend(loc="upper right")
    return figure


def export(dataset: Path, run_dir: Path, output: Path, split: Split = Split.VALIDATION) -> None:
    """Verify strict metrics against raw responses and export one requested split."""
    problems = tuple(p for p in load_dataset(dataset) if p.split == split)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config"]["split"] != split:
        raise ValueError("Export split differs from inference manifest")
    run = ModelRun.model_validate_json((run_dir / "predictions.json").read_text(encoding="utf-8"))
    records = sorted(
        (
            Checkpoint.model_validate_json(p.read_text(encoding="utf-8"))
            for p in (run_dir / "responses").glob("*.json")
        ),
        key=lambda r: r.id,
    )
    by_id = {p.id: p for p in problems}
    if len(records) != len(problems) or {r.id for r in records} != set(by_id):
        raise ValueError("Export requires exactly one response for every validation example")
    parsed = []
    for record in records:
        try:
            parsed.append(parse_response(by_id[record.id], record.generation.text))
        except ValueError:
            continue
    if sorted(parsed, key=lambda p: p.id) != sorted(run.predictions, key=lambda p: p.id):
        raise ValueError("Predictions do not match raw responses")
    report = evaluate(problems, run)
    stored = EvaluationReport.model_validate_json(
        (run_dir / "report.json").read_text(encoding="utf-8")
    )
    if report != stored:
        raise ValueError("Stored metrics do not match recomputed metrics")
    for name in ("manifest.json", "predictions.json", "report.json", "summary.json"):
        publish_text(output / name, (run_dir / name).read_text(encoding="utf-8"))
    publish_text(
        output / "raw-responses.json",
        TypeAdapter(list[Checkpoint]).dump_json(records, indent=2).decode("utf-8"),
    )
    publish_text(
        output / f"{split.value}.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8"),
    )
    format_counts = {
        "examples": len(records),
        "strict_valid": len(parsed),
        "starts_with_markdown_fence": sum(
            r.generation.text.lstrip().startswith("```") for r in records
        ),
        "note": "Descriptive format audit only; no repaired answers are scored.",
    }
    publish_text(output / "format-audit.json", json.dumps(format_counts, indent=2))
    chart = output / "accuracy-by-condition.png"
    if not chart.exists():
        build_chart(report).savefig(chart, dpi=180, bbox_inches="tight")
    scatter = output / "paired-accuracy.png"
    if not scatter.exists():
        figure, _ = plot_paired_accuracy(report)
        figure.savefig(scatter, dpi=180, bbox_inches="tight")
    logger.info("Validation export verified and saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=[s.value for s in Split], default="validation")
    args = parser.parse_args()
    export(args.dataset, args.run_dir, args.output, Split(args.split))
