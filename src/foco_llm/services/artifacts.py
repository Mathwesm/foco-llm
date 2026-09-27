"""Atomic, content-addressed local artifacts with validated loading."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import TypeAdapter

from foco_llm.core.generation import generate_problems
from foco_llm.core.prompts import PROMPT_VERSION, build_prompt
from foco_llm.core.validation import DatasetError, validate_dataset
from foco_llm.models.experiment import GenerationConfig, Problem, Split

GENERATOR_VERSION = "pilot-v1"


def publish_text(path: Path, content: str, *, pointer: bool = False) -> None:
    """Publish text atomically, refusing changes to existing result files.

    Args:
        path: Destination within the caller's output directory.
        content: UTF-8 content to publish.
        pointer: Allow replacing a small latest pointer after reading it.

    Raises:
        DatasetError: A different artifact already occupies the destination.
    """
    if path.exists():
        previous = path.read_text(encoding="utf-8")
        if previous == content:
            return
        if not pointer:
            raise DatasetError(f"Refusing to overwrite an existing artifact: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_dataset(config: GenerationConfig, output: Path) -> Path:
    """Generate and validate data, publish it, and update the latest pointer.

    Args:
        config: Reproducible generation parameters.
        output: Root for date-partitioned processed outputs.

    Returns:
        Directory containing the manifest, labels, and model-visible prompts.
    """
    problems = generate_problems(config)
    validate_dataset(problems)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8")
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    destination = output / datetime.now(UTC).date().isoformat() / fingerprint[:16]
    publish_text(destination / "dataset.json", payload)
    for split in Split:
        prompts = [
            build_prompt(problem).model_dump() for problem in problems if problem.split == split
        ]
        publish_text(destination / f"prompts-{split}.json", json.dumps(prompts, indent=2))
    manifest = {
        "schema_version": "1",
        "generator_version": GENERATOR_VERSION,
        "prompt_version": PROMPT_VERSION,
        "sha256": fingerprint,
        "config": config.model_dump(),
        "examples": len(problems),
        "base_problems": len({problem.base_id for problem in problems}),
        "split_counts": {split.value: sum(p.split == split for p in problems) for split in Split},
        "limitations": [
            "Shared linguistic templates across splits",
            "English synthetic pilot only",
        ],
    }
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    publish_text(
        output / "latest.json",
        json.dumps({"path": destination.relative_to(output).as_posix(), "sha256": fingerprint}),
        pointer=True,
    )
    return destination


def load_dataset(path: Path) -> tuple[Problem, ...]:
    """Load UTF-8 labels and verify paired scientific invariants.

    Args:
        path: A generated dataset JSON file.

    Returns:
        Validated experimental records.
    """
    problems = TypeAdapter(tuple[Problem, ...]).validate_json(path.read_text(encoding="utf-8"))
    validate_dataset(problems)
    return problems
