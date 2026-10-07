"""Entrypoint. Run with ``python -m foco_llm``."""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from foco_llm.config import Settings
from foco_llm.core.evaluation import evaluate
from foco_llm.core.plots import plot_dataset, plot_paired_accuracy
from foco_llm.models.experiment import GenerationConfig, ModelRun, Split
from foco_llm.services.artifacts import load_dataset, prepare_dataset, publish_text
from foco_llm.utils.logger import setup_logging


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate controlled relevance experiments."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate and validate the synthetic pilot")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--problems-per-task", type=int, default=100)
    prepare.add_argument("--output", type=Path, default=Path("data/processed"))
    validate = commands.add_parser("validate", help="Independently validate generated labels")
    validate.add_argument("dataset", type=Path)
    score = commands.add_parser("score", help="Score recorded predictions without running a model")
    score.add_argument("dataset", type=Path)
    score.add_argument("predictions", type=Path)
    score.add_argument("--split", choices=[split.value for split in Split], default="test")
    score.add_argument("--output", type=Path, default=Path("data/evaluations"))
    return parser


def _score(args: argparse.Namespace) -> None:
    problems = tuple(
        problem for problem in load_dataset(args.dataset) if problem.split == args.split
    )
    run = ModelRun.model_validate_json(args.predictions.read_text(encoding="utf-8"))
    report = evaluate(problems, run)
    content = report.model_dump_json(indent=2)
    fingerprint = hashlib.sha256(
        args.dataset.read_bytes() + args.predictions.read_bytes() + args.split.encode("utf-8")
    ).hexdigest()[:16]
    destination = args.output / datetime.now(UTC).date().isoformat() / fingerprint
    publish_text(destination / "report.json", content)
    figure, _ = plot_paired_accuracy(report)
    chart = destination / "paired-accuracy.png"
    if not chart.exists():
        figure.savefig(chart, dpi=180, bbox_inches="tight")
    logger.info("Evaluation saved: examples={}, report={}", len(problems), destination)


@logger.catch(reraise=True)
def main(argv: list[str] | None = None) -> int:
    """Run commands and propagate errors as nonzero process exits.

    Args:
        argv: Optional arguments supplied by integration tests.

    Returns:
        Zero after successful completion; exceptions propagate on failure.
    """
    args = _parser().parse_args(argv)
    settings = Settings()
    setup_logging(
        log_dir=settings.log_dir,
        level=settings.log_level,
        serialize=settings.log_serialize,
    )
    if args.command == "prepare":
        config = GenerationConfig(seed=args.seed, problems_per_task=args.problems_per_task)
        destination = prepare_dataset(config, args.output)
        problems = load_dataset(destination / "dataset.json")
        chart = destination / "dataset-profile.png"
        if not chart.exists():
            figure, _ = plot_dataset(problems)
            figure.savefig(chart, dpi=180, bbox_inches="tight")
        logger.info("Validated dataset saved: examples={}, path={}", len(problems), destination)
    elif args.command == "validate":
        logger.info("Dataset validated: examples={}", len(load_dataset(args.dataset)))
    else:
        _score(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
