"""Publish benchmark-v2 separately from historical pilot data."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.benchmark_v2 import generate_v2
from foco_llm.core.validation import validate_dataset
from foco_llm.models.experiment import GenerationConfig, Problem, Split
from foco_llm.services.artifacts import publish_text


@logger.catch(reraise=True)
def main() -> None:
    """Generate, independently validate, and preserve a candidate dataset."""
    config = GenerationConfig()
    problems = generate_v2(config)
    validate_dataset(problems)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output = Path("data/benchmark-v2") / datetime.now(UTC).date().isoformat() / digest[:16]
    publish_text(output / "dataset.json", payload)
    publish_text(
        output / "manifest.json",
        json.dumps(
            {
                "generator_version": "benchmark-v2-candidate",
                "dataset_sha256": digest,
                "config": config.model_dump(),
                "examples": len(problems),
                "split_counts": {s.value: sum(p.split == s for p in problems) for s in Split},
                "shift": "Disjoint language templates and depths: train 2/3, validation 4, test 5",
                "status": (
                    "Candidate pending baseline and broader manual audit; not a final benchmark"
                ),
            },
            indent=2,
        ),
    )
    logger.info("Validated v2 dataset: examples={}, path={}", len(problems), output)


if __name__ == "__main__":
    main()
