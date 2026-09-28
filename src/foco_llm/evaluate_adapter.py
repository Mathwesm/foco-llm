"""Evaluate one local adapter on validation using unchanged baseline decoding."""

import argparse
from pathlib import Path

from loguru import logger

from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import AdapterBackend
from foco_llm.services.inference import run_inference
from foco_llm.utils.logger import setup_logging


@logger.catch(reraise=True)
def main() -> None:
    """Load a verified adapter and save immutable validation responses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("adapter", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    config = InferenceConfig(model_id=args.model_id, revision=args.revision, precision="bfloat16")
    backend = AdapterBackend(config, args.adapter, args.dataset)
    run_inference(args.dataset, args.output, config, backend)


if __name__ == "__main__":
    main()
