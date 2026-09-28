"""Re-score saved validation responses with a versioned content-envelope diagnostic."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import PROTOCOL_VERSION, ContentEvaluation, compare_protocols
from foco_llm.core.evaluation import EvaluationReport
from foco_llm.core.plots import plot_paired_accuracy
from foco_llm.core.validation import validate_dataset
from foco_llm.models.experiment import ModelRun, Problem, Split
from foco_llm.models.inference import Checkpoint
from foco_llm.services.artifacts import publish_text


def build_chart(result: ContentEvaluation) -> Figure:
    """Plot independent answer scores and validated evidence scores as post-hoc."""
    figure = Figure(figsize=(12, 7), layout="constrained")
    axes = figure.subplots(2, 1)
    groups = result.content.groups
    labels = [f"{g.task}\n{g.condition}" for g in groups]
    for axis, values, title in zip(
        axes,
        [
            [g.accuracy * 100 for g in result.answer_only.groups],
            [g.evidence_exact * 100 for g in groups],
        ],
        ["Answer-only accuracy (%)", "Exact evidence match (%)"],
        strict=True,
    ):
        bars = axis.bar(labels, values, color="#0072B2")
        axis.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=8)
        axis.set(ylim=(0, 115), ylabel=title, xlabel="Task / condition")
        axis.tick_params(axis="x", labelsize=7)
    figure.suptitle(
        f"Post-hoc validation diagnostic: {PROTOCOL_VERSION}\nSame raw outputs; no training"
    )
    return figure


def analyze(source: Path, output: Path) -> ContentEvaluation:
    """Verify raw checkpoints and original scores before publishing a separate analysis."""
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (source / "validation.json").read_text(encoding="utf-8")
    )
    validate_dataset(problems)
    if any(p.split != Split.VALIDATION for p in problems):
        raise ValueError("This diagnostic is restricted to validation examples")
    records = TypeAdapter(tuple[Checkpoint, ...]).validate_json(
        (source / "raw-responses.json").read_text(encoding="utf-8")
    )
    provenance = ModelRun.model_validate_json(
        (source / "predictions.json").read_text(encoding="utf-8")
    )
    result = compare_protocols(problems, records, provenance)
    original = EvaluationReport.model_validate_json(
        (source / "report.json").read_text(encoding="utf-8")
    )
    if result.strict != original:
        raise ValueError("Replayed strict scores differ from the original report")
    inputs = [
        {"file": name, "source_sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
        for name in ("validation.json", "raw-responses.json", "predictions.json", "report.json")
    ]
    publish_text(
        output / "manifest.json",
        json.dumps(
            {
                "protocol": PROTOCOL_VERSION,
                "inputs": inputs,
                "analysis_type": "post_hoc_validation_diagnostic",
            },
            indent=2,
        ),
    )
    publish_text(output / "comparison.json", result.model_dump_json(indent=2))
    chart = output / "content-metrics.png"
    if not chart.exists():
        build_chart(result).savefig(chart, dpi=180, bbox_inches="tight")
    scatter = output / "answer-only-paired-accuracy.png"
    if not scatter.exists():
        figure, axis = plot_paired_accuracy(result.answer_only)
        axis.set_title(f"Answer-only paired accuracy: {result.answer_only.model_id}")
        figure.savefig(scatter, dpi=180, bbox_inches="tight")
    logger.info(
        "Diagnostic saved: strict={}, content={}, examples={}, path={}",
        result.strict_accepted,
        result.content_accepted,
        result.examples,
        output,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.source, args.output)
