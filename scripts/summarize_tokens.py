"""Audit actual prompt lengths after inference without equating sentences and tokens."""

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean

from loguru import logger
from matplotlib.figure import Figure
from pydantic import BaseModel, TypeAdapter

from foco_llm.core.content_evaluation import _verify_records
from foco_llm.models.experiment import Condition, Problem, Task
from foco_llm.models.inference import Checkpoint
from foco_llm.services.artifacts import publish_text


class TokenGroup(BaseModel):
    """Observed tokenizer lengths, including the model's conversation template."""

    task: Task
    condition: Condition
    examples: int
    input_min: int
    input_mean: float
    input_max: int
    output_mean: float


def aggregate(problems: tuple[Problem, ...], records: tuple[Checkpoint, ...]) -> list[TokenGroup]:
    """Verify exact prompt binding before grouping measured token counts."""
    _verify_records(problems, records)
    indexed = {record.id: record for record in records}
    groups = []
    for task in Task:
        for condition in Condition:
            selected = [
                indexed[p.id].generation
                for p in problems
                if p.task == task and p.condition == condition
            ]
            if not selected:
                raise ValueError("Token report requires all task/condition strata")
            sizes = [r.input_tokens for r in selected]
            groups.append(
                TokenGroup(
                    task=task,
                    condition=condition,
                    examples=len(selected),
                    input_min=min(sizes),
                    input_mean=fmean(sizes),
                    input_max=max(sizes),
                    output_mean=fmean(r.output_tokens for r in selected),
                )
            )
    return groups


def plot_tokens(groups: list[TokenGroup]) -> Figure:
    """Show observed mean/range; error bars are not confidence intervals."""
    figure = Figure(figsize=(12, 4), layout="constrained")
    axes = figure.subplots(1, 3, sharey=True)
    for task, axis in zip(Task, axes, strict=True):
        selected = [g for g in groups if g.task == task]
        axis.bar(
            [g.condition.value for g in selected],
            [g.input_mean for g in selected],
            color="#0072B2",
            yerr=[
                [g.input_mean - g.input_min for g in selected],
                [g.input_max - g.input_mean for g in selected],
            ],
            capsize=4,
        )
        axis.set(title=task.value, xlabel="Context condition", ylabel="Input length (tokens)")
        axis.tick_params(axis="x", labelsize=8, rotation=20)
    figure.suptitle("Observed prompt lengths: mean and min-max range, not confidence intervals")
    return figure


@logger.catch(reraise=True)
def summarize(source: Path, output: Path) -> None:
    """Publish immutable length diagnostics from complete, verified raw responses."""
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (source / "validation.json").read_text(encoding="utf-8")
    )
    records = TypeAdapter(tuple[Checkpoint, ...]).validate_json(
        (source / "raw-responses.json").read_text(encoding="utf-8")
    )
    groups = aggregate(problems, records)
    inputs = [
        {"file": name, "source_sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
        for name in ("validation.json", "raw-responses.json")
    ]
    payload = {
        "inputs": inputs,
        "groups": [g.model_dump() for g in groups],
        "interpretation": "Sentence counts are controlled; token lengths still differ",
    }
    publish_text(output / "token-statistics.json", json.dumps(payload, indent=2))
    if not (output / "input-tokens.png").exists():
        plot_tokens(groups).savefig(output / "input-tokens.png", dpi=180, bbox_inches="tight")
    logger.info("Verified token-length report saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.output)
