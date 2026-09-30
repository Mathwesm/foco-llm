"""Evaluate two fixed format contrasts on all thirty clean validation problems."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import _verify_records
from foco_llm.core.format_diagnostic import FormatArm, diagnostic_prompt, format_answer
from foco_llm.models.experiment import Condition, Problem, Split, Task
from foco_llm.models.inference import Checkpoint, InferenceConfig
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend

CASES_PER_TASK = 10
ARMS: tuple[FormatArm, ...] = ("explicit_json", "plain_answer")


def attempt(
    problem: Problem, arm: FormatArm, directory: Path, backend: TransformersBackend
) -> Checkpoint:
    """Resume only an exact prompt match, preserving raw outputs including failures."""
    prompt = diagnostic_prompt(problem, arm)
    digest = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
    path = directory / arm / f"{hashlib.sha256(problem.id.encode()).hexdigest()}.json"
    if path.exists():
        record = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        if record.id != problem.id or record.prompt_sha256 != digest:
            raise ValueError("Format checkpoint does not match prompt")
        return record
    record = Checkpoint(
        id=problem.id,
        prompt_sha256=digest,
        finished_at=datetime.now(UTC),
        generation=backend.generate(prompt),
    )
    publish_text(path, record.model_dump_json(indent=2))
    return record


def load_inputs(source: Path) -> tuple[tuple[Problem, ...], dict[str, Checkpoint]]:
    """Check baseline binding and select all clean validation cases without outcome selection."""
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(
        (source / "validation.json").read_text(encoding="utf-8")
    )
    records = TypeAdapter(tuple[Checkpoint, ...]).validate_json(
        (source / "raw-responses.json").read_text(encoding="utf-8")
    )
    _verify_records(problems, records)
    clean = tuple(p for p in problems if p.condition == Condition.CLEAN)
    if any(p.split != Split.VALIDATION for p in clean) or any(
        sum(p.task == t for p in clean) != CASES_PER_TASK for t in Task
    ):
        raise ValueError("Expected ten clean validation cases per task")
    return clean, {r.id: r for r in records}


def summarize(
    problems: tuple[Problem, ...], records: dict[str, list[Checkpoint]]
) -> dict[str, object]:
    """Report readability and correctness independently with every attempt in the denominator."""
    rows = []
    for arm, outputs in records.items():
        contract: FormatArm = "plain_answer" if arm == "plain_answer" else "explicit_json"
        indexed = {r.id: r for r in outputs}
        if len(indexed) != len(outputs) or set(indexed) != {p.id for p in problems}:
            raise ValueError("Expected exactly one result per problem and arm")
        for p in problems:
            answer = format_answer(indexed[p.id].generation.text, contract)
            rows.append(
                {
                    "arm": arm,
                    "id": p.id,
                    "task": p.task,
                    "readable": answer is not None,
                    "correct": answer is not None
                    and answer.strip().casefold() == p.answer.casefold(),
                }
            )
    groups = []
    for arm in records:
        for task in Task:
            selected = [r for r in rows if r["arm"] == arm and r["task"] == task]
            groups.append(
                {
                    "arm": arm,
                    "task": task,
                    "count": len(selected),
                    "correct": sum(bool(r["correct"]) for r in selected),
                    "readable": sum(bool(r["readable"]) for r in selected),
                }
            )
    return {"groups": groups, "decisions": rows}


@logger.catch(reraise=True)
def run(source: Path, output: Path) -> None:
    """Run a bounded format diagnostic with manifest-bound resumable responses."""
    problems, baseline = load_inputs(source)
    previous = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if previous["prompt_version"] != "evidence-json-v1" or "adapter_sha256" in previous["runtime"]:
        raise ValueError("Expected original baseline and evidence-json-v1")
    config = InferenceConfig.model_validate(previous["config"])
    backend = TransformersBackend(config)
    if backend.metadata() != previous["runtime"]:
        raise ValueError("Runtime differs from baseline")
    manifest = {
        "protocol": "format-diagnostic-v1",
        "config": config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "source_sha256": _source_fingerprint(),
        "script": {"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "inputs": [
            {"file": n, "sha256": hashlib.sha256((source / n).read_bytes()).hexdigest()}
            for n in ("manifest.json", "validation.json", "raw-responses.json")
        ],
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    publish_text(
        destination / "problems.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode(),
    )
    records = {"original": [baseline[p.id] for p in problems]}
    for arm in ARMS:
        records[arm] = [attempt(p, arm, destination, backend) for p in problems]
        publish_text(
            destination / f"{arm}-responses.json",
            TypeAdapter(list[Checkpoint]).dump_json(records[arm], indent=2).decode(),
        )
        logger.info("Format arm complete: {}", arm)
    publish_text(destination / "report.json", json.dumps(summarize(problems, records), indent=2))
    logger.info("Format diagnostic complete: {}", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
