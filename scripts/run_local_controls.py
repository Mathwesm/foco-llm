"""Run the frozen nine-arm local matrix sequentially with isolated GPU processes."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

from foco_llm.core.pilot_configuration import PilotConfig
from foco_llm.models.experiment import Condition
from foco_llm.services.artifacts import publish_text
from foco_llm.services.inference import _source_fingerprint


def execute(arguments: list[str], log: Path) -> None:
    """Run one bounded child process, retaining its diagnostics on failure."""
    environment = {**os.environ, "PYTHONUTF8": "1"}
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(  # noqa: S603 -- fixed Python entrypoints, argument list, no shell.
            [sys.executable, *arguments],
            check=True,
            timeout=1800,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )


def only_run(root: Path) -> Path:
    """Reject ambiguous output instead of silently choosing the newest run."""
    paths = list(root.glob("*/manifest.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one manifest under {root}")
    return paths[0].parent


def run_arm(dataset: Path, baseline: Path, root: Path, config: PilotConfig) -> dict[str, str]:
    """Resume one exact configuration and export its paired validation results."""
    root.mkdir(parents=True, exist_ok=True)
    binding = {
        "source_sha256": _source_fingerprint(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
    }
    publish_text(root / "binding.json", json.dumps(binding, sort_keys=True, indent=2))
    publish_text(root / "config.json", config.model_dump_json(indent=2))
    if (root / "complete.json").exists():
        return json.loads((root / "complete.json").read_text(encoding="utf-8"))
    execute(
        [
            "-m",
            "foco_llm.train_pilot",
            str(dataset),
            str(root / "config.json"),
            str(root / "training"),
        ],
        root / "training.log",
    )
    trained = only_run(root / "training")
    execute(
        [
            "-m",
            "foco_llm.evaluate_adapter",
            str(dataset),
            str(trained / f"step-{config.steps:04d}"),
            "--revision",
            config.revision,
            "--output",
            str(root / "validation"),
        ],
        root / "validation.log",
    )
    evaluated = only_run(root / "validation")
    commands = [
        ["scripts/export_pilot.py", str(dataset), str(trained), str(root / "export-training")],
        [
            "scripts/export_baseline.py",
            str(dataset),
            str(evaluated),
            str(root / "export-validation"),
        ],
        ["scripts/evaluate_content.py", str(root / "export-validation"), str(root / "content")],
        [
            "scripts/compare_adapter.py",
            str(baseline),
            str(root / "export-validation"),
            str(root / "comparison"),
        ],
    ]
    for command in commands:
        execute(command, root / "export.log")
    result = {"training": str(trained), "validation": str(evaluated), "root": str(root)}
    publish_text(root / "complete.json", json.dumps(result, indent=2))
    logger.info("Local control completed: {}", root.name)
    return result


@logger.catch(reraise=True)
def main() -> None:
    """Execute all prespecified arms; errors stop the batch with nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    original = PilotConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    results = {}
    for seed in (42, 43, 44):
        for arm in ("C1", "C2", "C3"):
            settings = original.model_dump(mode="json")
            settings.update(
                seed=seed,
                selection_seed=42,
                condition=Condition.CLEAN if arm == "C1" else Condition.SIMILAR,
                supervision="answer_and_evidence" if arm == "C3" else "answer",
            )
            config = PilotConfig.model_validate(settings)
            name = f"{arm}-seed-{seed}"
            results[name] = run_arm(args.dataset, args.baseline, args.output / name, config)
    publish_text(args.output / "completed-matrix.json", json.dumps(results, indent=2))
    logger.info("All nine local training and validation runs completed: {}", args.output)


if __name__ == "__main__":
    main()
