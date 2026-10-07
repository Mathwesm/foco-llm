"""Publish candidate v2.1 data with paired-context controls and an audit summary."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.models.experiment import Condition, GenerationConfig, Problem, Split, Task
from foco_llm.services.artifacts import publish_text
from foco_llm.utils.logger import setup_logging


@logger.catch(reraise=True)
def main() -> None:
    """Preserve a deterministic dataset and machine-readable invariant checks."""
    setup_logging(log_dir=Path("logs"), serialize=True)
    config = GenerationConfig()
    problems = generate_v21(config)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(problems, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    output = Path("data/benchmark-v21") / datetime.now(UTC).date().isoformat() / digest[:16]
    manifest = {
        "generator_version": "benchmark-v2.1-candidate",
        "dataset_sha256": digest,
        "config": config.model_dump(),
        "examples": len(problems),
        "split_counts": {s.value: sum(p.split == s for p in problems) for s in Split},
        "checked_invariants": [
            "reference_answers",
            "necessary_evidence",
            "paired_splits",
            "paired_relevant_order",
            "matched_distracted_evidence_positions",
            "matched_distractor_sentence_counts",
        ],
        "limitations": [
            "Token lengths not exactly matched",
            "Joint template/depth shift",
            "One template per split and task",
            "Small validation pilot",
        ],
    }
    publish_text(output / "dataset.json", payload)
    publish_text(output / "manifest.json", json.dumps(manifest, indent=2))
    # Only train/validation examples are selected for manual review, not final test.
    sample = tuple(
        p
        for p in problems
        if p.split in (Split.TRAIN, Split.VALIDATION) and p.base_id.endswith(("000000", "000008"))
    )
    publish_text(
        output / "audit-sample.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(sample, indent=2).decode("utf-8"),
    )
    logger.info(
        "Validated v2.1: examples={}, audit_sample={}, tasks={}, conditions={}, path={}",
        len(problems),
        len(sample),
        len(Task),
        len(Condition),
        output,
    )


if __name__ == "__main__":
    main()
