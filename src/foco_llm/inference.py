"""Local inference CLI, separate from lightweight data and scoring commands."""

import argparse
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from foco_llm.config import Settings
from foco_llm.models.experiment import Split
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.inference import run_inference
from foco_llm.services.transformers_backend import TransformersBackend
from foco_llm.utils.logger import setup_logging


@logger.catch(reraise=True)
def main() -> None:
    """Run a validation baseline by default, leaving the test split untouched."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--split", choices=[s.value for s in Split], default="validation")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--precision", choices=["float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-seconds", type=float, default=60)
    parser.add_argument(
        "--output", type=Path, default=Path("data/inference") / datetime.now(UTC).date().isoformat()
    )
    args = parser.parse_args()
    settings = Settings()
    setup_logging(log_dir=settings.log_dir, serialize=True)
    config = InferenceConfig(
        revision=args.revision,
        model_id=args.model_id,
        split=args.split,
        device=args.device,
        precision=args.precision,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        max_seconds=args.max_seconds,
    )
    run_inference(args.dataset, args.output, config, TransformersBackend(config))


if __name__ == "__main__":
    main()
