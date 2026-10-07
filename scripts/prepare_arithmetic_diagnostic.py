"""Publish a frozen seven-form arithmetic diagnostic on new validation bases."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.arithmetic_diagnostic import arithmetic_probes
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.models.experiment import Condition, GenerationConfig, Problem, Split, Task
from foco_llm.services.artifacts import publish_text
from foco_llm.utils.logger import setup_logging

EXPECTED_BASE_CASES = 10
PROBES_PER_BASE = 7


@logger.catch(reraise=True)
def main() -> None:
    """Generate deterministic diagnostic cases and record their provenance."""
    setup_logging(log_dir=Path("logs"), serialize=True)
    config = GenerationConfig(seed=20261002, problems_per_task=100)
    source = generate_v21(config)
    clean = tuple(
        p
        for p in source
        if p.task == Task.ARITHMETIC
        and p.split == Split.VALIDATION
        and p.condition == Condition.CLEAN
    )
    similar = {
        p.base_id: p
        for p in source
        if p.task == Task.ARITHMETIC
        and p.split == Split.VALIDATION
        and p.condition == Condition.SIMILAR
    }
    probes = tuple(probe for p in clean for probe in arithmetic_probes(p, similar[p.base_id]))
    if len(clean) != EXPECTED_BASE_CASES or len(probes) != EXPECTED_BASE_CASES * PROBES_PER_BASE:
        raise ValueError("Unexpected diagnostic case count")
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(probes, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output = Path("data/arithmetic-diagnostic") / datetime.now(UTC).date().isoformat() / digest[:16]
    publish_text(output / "dataset.json", payload)
    publish_text(
        output / "manifest.json",
        json.dumps(
            {
                "protocol": "arithmetic-diagnostic-v1",
                "generation_config": config.model_dump(),
                "dataset_sha256": digest,
                "base_cases": len(clean),
                "probes": len(probes),
            },
            indent=2,
        ),
    )
    logger.info("Arithmetic diagnostic prepared: cases={}, path={}", len(probes), output)


if __name__ == "__main__":
    main()
