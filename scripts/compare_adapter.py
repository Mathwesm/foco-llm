"""Publish paired before/after answer transitions for one local adapter."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import ContentError, _verify_records, decode_content
from foco_llm.core.evaluation import _correct
from foco_llm.models.experiment import Condition, Prediction, Problem, Task
from foco_llm.models.inference import Checkpoint
from foco_llm.services.artifacts import publish_text


def outcomes(source: Path, problems: tuple[Problem, ...]) -> dict[str, bool]:
    """Keep unreadable answers as failures and apply the existing scoring rule."""
    records = TypeAdapter(tuple[Checkpoint, ...]).validate_json(
        (source / "raw-responses.json").read_text(encoding="utf-8")
    )
    _verify_records(problems, records)
    indexed = {p.id: p for p in problems}
    result = {}
    for record in records:
        try:
            answer = decode_content(record.generation.text).answer
            prediction = Prediction(id=record.id, answer=answer, evidence=())
            result[record.id] = _correct(indexed[record.id], prediction)
        except ContentError:
            result[record.id] = False
    return result


def verify_controls(before: Path, after: Path) -> None:
    """Reject mismatched validation data, generation settings, or base runtimes."""
    left, right = [
        json.loads((p / "manifest.json").read_text(encoding="utf-8")) for p in (before, after)
    ]
    for key in ("config", "dataset_sha256", "examples", "prompt_version"):
        if left[key] != right[key]:
            raise ValueError(f"Baseline and adapter controls differ: {key}")
    if "adapter_sha256" in left["runtime"] or "adapter_sha256" not in right["runtime"]:
        raise ValueError("Expected an original baseline followed by an identified adapter")
    for key, value in left["runtime"].items():
        if right["runtime"].get(key) != value:
            raise ValueError(f"Base runtime differs: {key}")
    if (before / "validation.json").read_bytes() != (after / "validation.json").read_bytes():
        raise ValueError("Validation examples differ")


@logger.catch(reraise=True)
def compare(before: Path, after: Path, output: Path) -> None:
    """Export per-example outcomes and grouped transitions without excluding failures."""
    verify_controls(before, after)
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (before / "validation.json").read_text(encoding="utf-8")
    )
    left, right = outcomes(before, problems), outcomes(after, problems)
    rows = [{"id": p.id, "before": left[p.id], "after": right[p.id]} for p in problems]
    groups = []
    for task in Task:
        for condition in Condition:
            ids = [p.id for p in problems if p.task == task and p.condition == condition]
            groups.append(
                {
                    "task": task,
                    "condition": condition,
                    "count": len(ids),
                    "before_correct": sum(left[i] for i in ids),
                    "after_correct": sum(right[i] for i in ids),
                    "correct_to_wrong": sum(left[i] and not right[i] for i in ids),
                    "wrong_to_correct": sum(not left[i] and right[i] for i in ids),
                }
            )
    payload = {"protocol": "content-envelope-v2", "groups": groups, "examples": rows}
    publish_text(output / "transitions.json", json.dumps(payload, indent=2))
    inputs = [
        {"arm": arm, "file": name, "sha256": hashlib.sha256((p / name).read_bytes()).hexdigest()}
        for arm, p in (("before", before), ("after", after))
        for name in ("manifest.json", "raw-responses.json", "validation.json")
    ]
    publish_text(output / "inputs.json", json.dumps(inputs, indent=2))
    figure = Figure(figsize=(11, 4), layout="constrained")
    axis = figure.subplots()
    ids = [p.id for p in problems]
    for label, values, color, offset in (
        ("Original", left, "#0072B2", -0.2),
        ("LoRA adapter", right, "#D55E00", 0.2),
    ):
        totals = [sum(values[i] for i in ids if i.startswith(f"v2-{t}-")) for t in Task]
        axis.bar([i + offset for i in range(3)], totals, width=0.4, label=label, color=color)
    axis.set(
        xticks=range(3),
        xticklabels=[t.value for t in Task],
        ylim=(0, 40),
        xlabel="Task",
        ylabel="Correct answers (out of 40)",
        title="Adapter validation: all context conditions combined",
    )
    axis.legend()
    if not (output / "before-after.png").exists():
        figure.savefig(output / "before-after.png", dpi=180, bbox_inches="tight")
    logger.info("Verified paired adapter comparison saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    compare(args.before, args.after, args.output)
