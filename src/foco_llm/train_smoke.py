"""Run a short local adapter test; never select validation or test examples for training."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from foco_llm.models.training import SmokeConfig
from foco_llm.services.training import run_smoke
from foco_llm.utils.logger import setup_logging


@logger.catch(reraise=True)
def main() -> None:
    """Validate configuration and run the manually initiated functional test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--stop-after", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path("data/training") / datetime.now(UTC).date().isoformat()
    )
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    run_smoke(
        args.dataset,
        SmokeConfig(revision=args.revision, steps=args.steps),
        args.output,
        args.stop_after,
    )


if __name__ == "__main__":
    main()
