"""Compare two audited runs under identical data, decoding, and runtime controls."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import BaseModel

from foco_llm.core.content_evaluation import PROTOCOL_VERSION, ContentEvaluation
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.artifacts import publish_text


class Manifest(BaseModel):
    """Required run controls for a fair descriptive comparison."""

    config: InferenceConfig
    dataset_sha256: str
    prompt_version: str
    runtime: dict[str, str]
    examples: int


class InputDigest(BaseModel):
    """A named input and its immutable content digest."""

    file: str
    source_sha256: str


class AnalysisManifest(BaseModel):
    """Binding between a diagnostic and its original run artifacts."""

    protocol: str
    inputs: list[InputDigest]


def verify_controls(left: Manifest, right: Manifest) -> None:
    """Reject comparisons that vary controls other than the model checkpoint."""
    if left.config.model_dump(exclude={"model_id", "revision"}) != right.config.model_dump(
        exclude={"model_id", "revision"}
    ):
        raise ValueError("Generation configuration differs between runs")
    if (left.dataset_sha256, left.prompt_version, left.runtime, left.examples) != (
        right.dataset_sha256,
        right.prompt_version,
        right.runtime,
        right.examples,
    ):
        raise ValueError("Dataset, prompt, runtime, or example count differs between runs")


def load_analysis(source: Path) -> tuple[Manifest, ContentEvaluation]:
    """Verify every analysis input digest before reading its scores."""
    run = Manifest.model_validate_json((source / "manifest.json").read_text(encoding="utf-8"))
    analysis = source.with_name(source.name + "-content-v2")
    metadata = AnalysisManifest.model_validate_json(
        (analysis / "manifest.json").read_text(encoding="utf-8")
    )
    required = {"validation.json", "raw-responses.json", "predictions.json", "report.json"}
    if len(metadata.inputs) != len(required) or {item.file for item in metadata.inputs} != required:
        raise ValueError("Analysis input manifest is incomplete or contains unexpected paths")
    for item in metadata.inputs:
        if hashlib.sha256((source / item.file).read_bytes()).hexdigest() != item.source_sha256:
            raise ValueError("Analysis does not match its original input artifacts")
    result = ContentEvaluation.model_validate_json(
        (analysis / "comparison.json").read_text(encoding="utf-8")
    )
    if metadata.protocol != PROTOCOL_VERSION or result.protocol != PROTOCOL_VERSION:
        raise ValueError("Analysis protocol differs from the comparison protocol")
    if (
        result.content.model_id != run.config.model_id
        or result.content.model_revision != run.config.revision
    ):
        raise ValueError("Analysis model does not match the run manifest")
    return run, result


def build_chart(left: ContentEvaluation, right: ContentEvaluation) -> Figure:
    """Return paired model bars on common scales, without inferring causality."""
    figure = Figure(figsize=(13, 5), layout="constrained")
    axis = figure.subplots()
    groups = left.answer_only.groups
    keys = [(g.task, g.condition) for g in groups]
    for offset, result, color in [(-0.2, left, "#0072B2"), (0.2, right, "#E69F00")]:
        values = {(g.task, g.condition): g.accuracy * 100 for g in result.answer_only.groups}
        if set(values) != set(keys):
            raise ValueError("Analysis strata differ between models")
        bars = axis.bar(
            [i + offset for i in range(len(keys))],
            [values[key] for key in keys],
            width=0.4,
            label=result.content.model_id,
            color=color,
        )
        axis.bar_label(bars, fmt="%.0f", padding=2, fontsize=7)
    axis.set_xticks(range(len(keys)), [f"{task}\n{condition}" for task, condition in keys])
    axis.tick_params(axis="x", labelsize=7)
    axis.set(
        ylim=(0, 125),
        ylabel="Answer-only accuracy (%)",
        xlabel="Task / condition",
        title="Exploratory validation comparison: same 120 examples, no training",
    )
    axis.legend(loc="upper right", fontsize=8)
    return figure


def compare(left: Path, right: Path, output: Path) -> None:
    """Publish a reproducible comparison after checking shared experimental controls."""
    left_manifest, left_result = load_analysis(left)
    right_manifest, right_result = load_analysis(right)
    verify_controls(left_manifest, right_manifest)
    if (left / "validation.json").read_bytes() != (right / "validation.json").read_bytes():
        raise ValueError("Selected validation examples differ between runs")
    payload = {
        "protocol": PROTOCOL_VERSION,
        "comparison_type": "exploratory_not_causal",
        "runs": [left.name, right.name],
        "models": [left_result.model_dump(), right_result.model_dump()],
    }
    publish_text(output / "comparison.json", json.dumps(payload, indent=2))
    chart = output / "model-comparison.png"
    if not chart.exists():
        build_chart(left_result, right_result).savefig(chart, dpi=180, bbox_inches="tight")
    logger.info("Controlled descriptive comparison saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.left, args.right, args.output)
