"""Run 120 clean development probes with fixed language and paired chain depth."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.content_evaluation import ContentError, decode_content
from foco_llm.core.depth_diagnostic import depth_problems
from foco_llm.models.experiment import Problem, Task
from foco_llm.models.inference import Checkpoint, InferenceConfig
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _checkpoint, _source_fingerprint
from foco_llm.services.transformers_backend import TransformersBackend


def decision(problem: Problem, record: Checkpoint) -> dict[str, object]:
    """Score full content envelopes without extracting answers from explanations."""
    try:
        answer = decode_content(record.generation.text).answer
    except ContentError:
        answer = None
    return {
        "id": problem.id,
        "task": problem.task,
        "depth": len(problem.facts) - 1,
        "readable": answer is not None,
        "correct": answer == problem.answer,
    }


@logger.catch(reraise=True)
def run(output: Path) -> None:
    """Preserve immutable inputs, runtime identity, and per-example checkpoints."""
    config = InferenceConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        # Public Hugging Face commit SHA, not a credential.
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",  # pragma: allowlist secret
        precision="bfloat16",
    )
    problems = depth_problems()
    backend = TransformersBackend(config)
    manifest = {
        "protocol": "depth-diagnostic-v1",
        "config": config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "source_sha256": _source_fingerprint(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    destination = output / hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    publish_text(destination / "manifest.json", payload)
    publish_text(
        destination / "problems.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode(),
    )
    records = [_checkpoint(p, destination, backend) for p in problems]
    rows = [decision(p, r) for p, r in zip(problems, records, strict=True)]
    groups = []
    for task in Task:
        for depth in range(1, 5):
            selected = [r for r in rows if r["task"] == task and r["depth"] == depth]
            groups.append(
                {
                    "task": task,
                    "depth": depth,
                    "count": len(selected),
                    "correct": sum(bool(r["correct"]) for r in selected),
                    "readable": sum(bool(r["readable"]) for r in selected),
                }
            )
    publish_text(
        destination / "raw-responses.json",
        TypeAdapter(list[Checkpoint]).dump_json(records, indent=2).decode(),
    )
    publish_text(
        destination / "report.json", json.dumps({"groups": groups, "decisions": rows}, indent=2)
    )
    logger.info("Depth diagnostic complete: {}", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
