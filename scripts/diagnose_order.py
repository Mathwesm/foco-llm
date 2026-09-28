"""Run a bounded paired reversal diagnostic on previously evaluated clean validation cases."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import ContentError, _verify_records, decode_content
from foco_llm.core.prompts import PROMPT_VERSION
from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, Problem, Split, Task
from foco_llm.models.inference import Checkpoint, InferenceConfig
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _checkpoint, _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend

CASES_PER_TASK = 10


def reverse_facts(problem: Problem) -> Problem:
    """Reverse only presentation order and reject inconsistent reference answers."""
    if problem.split != Split.VALIDATION or problem.condition != Condition.CLEAN:
        raise ValueError("Diagnostic requires clean validation problems")
    changed = problem.model_copy(update={"facts": tuple(reversed(problem.facts))})
    if solve_v2(problem) != problem.answer or solve_v2(changed) != problem.answer:
        raise ValueError("Reference answer is not invariant to fact reversal")
    return changed


def correct(problem: Problem, record: Checkpoint) -> bool:
    """Apply the existing answer-only rule, counting rejected content as incorrect."""
    try:
        return (
            decode_content(record.generation.text).answer.strip().casefold()
            == problem.answer.casefold()
        )
    except ContentError:
        return False


def load_inputs(source: Path) -> tuple[tuple[Problem, ...], dict[str, Checkpoint]]:
    """Verify complete baseline prompt binding before selecting all clean cases."""
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (source / "validation.json").read_text(encoding="utf-8")
    )
    records = TypeAdapter(tuple[Checkpoint, ...]).validate_json(
        (source / "raw-responses.json").read_text(encoding="utf-8")
    )
    _verify_records(problems, records)
    clean = tuple(p for p in problems if p.condition == Condition.CLEAN)
    if any(sum(p.task == task for p in clean) != CASES_PER_TASK for task in Task):
        raise ValueError("Expected ten clean problems per task")
    for problem in clean:
        reverse_facts(problem)
    return clean, {r.id: r for r in records}


def publish_report(
    output: Path,
    problems: tuple[Problem, ...],
    before: dict[str, Checkpoint],
    after: list[Checkpoint],
) -> None:
    """Save all paired decisions, including failures, with a descriptive chart."""
    rows = [
        {
            "id": p.id,
            "task": p.task,
            "original_correct": correct(p, before[p.id]),
            "reversed_correct": correct(p, r),
        }
        for p, r in zip(problems, after, strict=True)
    ]
    groups = []
    for task in Task:
        selected = [r for r in rows if r["task"] == task]
        groups.append(
            {
                "task": task,
                "count": len(selected),
                "original_correct": sum(bool(r["original_correct"]) for r in selected),
                "reversed_correct": sum(bool(r["reversed_correct"]) for r in selected),
                "gained": sum(
                    not r["original_correct"] and bool(r["reversed_correct"]) for r in selected
                ),
                "lost": sum(
                    bool(r["original_correct"]) and not r["reversed_correct"] for r in selected
                ),
            }
        )
    publish_text(output / "report.json", json.dumps({"groups": groups, "pairs": rows}, indent=2))
    publish_text(
        output / "raw-responses.json",
        TypeAdapter(list[Checkpoint]).dump_json(after, indent=2).decode(),
    )
    figure = Figure(figsize=(8, 4), layout="constrained")
    axis = figure.subplots()
    for label, key, offset, color in (
        ("Original", "original_correct", -0.2, "#0072B2"),
        ("Reversed", "reversed_correct", 0.2, "#D55E00"),
    ):
        axis.bar(
            [i + offset for i in range(3)],
            [g[key] for g in groups],
            width=0.4,
            label=label,
            color=color,
        )
    axis.set(
        xticks=range(3),
        xticklabels=[t.value for t in Task],
        ylim=(0, 10),
        xlabel="Task",
        ylabel="Correct answers (out of 10)",
        title="Clean validation: fact-order reversal",
    )
    axis.legend()
    if not (output / "order-accuracy.png").exists():
        figure.savefig(output / "order-accuracy.png", dpi=180, bbox_inches="tight")


@logger.catch(reraise=True)
def run(source: Path, output: Path) -> None:
    """Reuse the original baseline and generate only the thirty reversed prompts."""
    problems, baseline = load_inputs(source)
    original = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if original["prompt_version"] != PROMPT_VERSION or "adapter_sha256" in original["runtime"]:
        raise ValueError("Expected unchanged baseline prompt and original model")
    config = InferenceConfig.model_validate(original["config"])
    backend = TransformersBackend(config)
    if backend.metadata() != original["runtime"]:
        raise ValueError("Diagnostic runtime differs from baseline")
    manifest = {
        "protocol": "order-reversal-v1",
        "config": config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "source_sha256": _source_fingerprint(),
        "script": {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "inputs": [
            {"file": name, "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()}
            for name in ("manifest.json", "validation.json", "raw-responses.json")
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    reversed_problems = tuple(reverse_facts(p) for p in problems)
    publish_text(
        destination / "problems.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(reversed_problems, indent=2).decode(),
    )
    records = []
    for problem in reversed_problems:
        records.append(_checkpoint(problem, destination, backend))
    publish_report(destination, problems, baseline, records)
    logger.info("Order diagnostic complete: {}", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
