"""Execute one named follow-up training arm with the existing resumable runner."""

import argparse
import hashlib
import json
from pathlib import Path

from loguru import logger
from run_local_controls import run_arm

from foco_llm.core.pilot_configuration import PilotConfig


@logger.catch(reraise=True)
def main() -> None:
    """Validate a frozen arm before starting its sequential GPU work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("plan", type=Path)
    parser.add_argument("arm")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.arm not in plan:
        raise ValueError(f"Unknown training arm: {args.arm}")
    config = PilotConfig.model_validate(plan[args.arm])
    if not (args.baseline / "manifest.json").exists():
        raise ValueError("Baseline export is missing")
    baseline = json.loads((args.baseline / "manifest.json").read_text(encoding="utf-8"))
    if (baseline["config"]["model_id"], baseline["config"]["revision"]) != (
        config.model_id,
        config.revision,
    ):
        raise ValueError("Baseline model or revision differs from training config")
    if baseline["config"]["split"] != "validation":
        raise ValueError("Targeted training requires a validation baseline")
    if baseline["dataset_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
        raise ValueError("Baseline and training dataset differ")
    run_arm(args.dataset, args.baseline, args.output / args.arm, config)


if __name__ == "__main__":
    main()
