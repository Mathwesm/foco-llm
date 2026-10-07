"""Publish compact, auditable arithmetic results without local model weights."""

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from foco_llm.models.inference import Checkpoint
from foco_llm.services.artifacts import publish_text


def export(source: Path, output: Path) -> None:
    """Verify one completed run and aggregate its immutable raw checkpoints.

    Args:
        source: Completed local run directory.
        output: Public report directory for this arm.

    Raises:
        ValueError: Case counts or identifiers disagree.
    """
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    scores = json.loads((source / "scores.json").read_text(encoding="utf-8"))
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    records = tuple(
        Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted((source / "responses").glob("*.json"))
    )
    record_ids = {record.id for record in records}
    is_chained = (
        manifest["protocol"].startswith("chained-arithmetic-v")
        or manifest["protocol"] == "selected-chain-v1"
    )
    score_ids = (
        {f"{row['id']}:step-{step['step']}" for row in scores for step in row.get("steps", [])}
        if is_chained
        else {row["id"] for row in scores}
    )
    expected_records = summary["generated_steps"] if is_chained else manifest["cases"]
    if (
        len(records) != expected_records
        or len(scores) != manifest["cases"]
        or len(record_ids) != len(records)
        or record_ids != score_ids
        or summary["cases"] != manifest["cases"]
    ):
        raise ValueError("Arithmetic report records disagree with run manifest")
    for filename in ("manifest.json", "scores.json", "summary.json"):
        publish_text(output / filename, (source / filename).read_text(encoding="utf-8"))
    raw = TypeAdapter(tuple[Checkpoint, ...]).dump_json(records, indent=2).decode("utf-8")
    publish_text(output / "raw-responses.json", raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
