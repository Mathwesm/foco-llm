"""Evaluate frozen original and seed-42 controls once on the untouched test split."""

import argparse
import json
from pathlib import Path

from loguru import logger
from run_local_controls import execute, only_run

from foco_llm.services.artifacts import publish_text

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pragma: allowlist secret -- public revision
ARMS = ("C0", "C1", "C2", "C3")


def run_one(dataset: Path, matrix: Path, root: Path, arm: str) -> Path:
    """Keep one exact test run per arm and export raw responses and scores."""
    location = root / arm
    location.mkdir(parents=True, exist_ok=True)
    if arm == "C0":
        arguments = [
            "-m",
            "foco_llm.inference",
            str(dataset),
            "--model-id",
            MODEL,
            "--revision",
            REVISION,
            "--precision",
            "bfloat16",
            "--split",
            "test",
            "--output",
            str(location / "inference"),
        ]
    else:
        trained = only_run(matrix / f"{arm}-seed-42" / "training")
        arguments = [
            "-m",
            "foco_llm.evaluate_adapter",
            str(dataset),
            str(trained / "step-0064"),
            "--model-id",
            MODEL,
            "--revision",
            REVISION,
            "--split",
            "test",
            "--output",
            str(location / "inference"),
        ]
    execute(arguments, location / "inference.log")
    inferred = only_run(location / "inference")
    commands = [
        [
            "scripts/export_baseline.py",
            str(dataset),
            str(inferred),
            str(location / "export"),
            "--split",
            "test",
        ],
        ["scripts/evaluate_content.py", str(location / "export"), str(location / "content")],
    ]
    for command in commands:
        execute(command, location / "export.log")
    return location / "export"


@logger.catch(reraise=True)
def main() -> None:
    """Run the prespecified four models; preserve partial progress on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.matrix / "completed-matrix.json").exists():
        raise ValueError("Local control matrix must complete before final test")
    outputs = {arm: run_one(args.dataset, args.matrix, args.output, arm) for arm in ARMS}
    for arm in ARMS[1:]:
        execute(
            [
                "scripts/compare_adapter.py",
                str(outputs["C0"]),
                str(outputs[arm]),
                str(args.output / arm / "comparison"),
            ],
            args.output / arm / "comparison.log",
        )
    publish_text(
        args.output / "completed-final.json",
        json.dumps({key: str(value) for key, value in outputs.items()}, indent=2),
    )
    logger.info("Frozen four-arm final test completed: {}", args.output)


if __name__ == "__main__":
    main()
