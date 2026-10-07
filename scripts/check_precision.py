"""Reproduce the FP16 numerical failure and verify bounded BF16 generation."""

import argparse
import json
from pathlib import Path

from loguru import logger

from foco_llm.core.prompts import build_prompt
from foco_llm.models.inference import InferenceConfig
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.transformers_backend import NumericalInferenceError, TransformersBackend


def diagnose(dataset: Path, output: Path) -> None:
    """Inspect first-step logits and test the numerical guard on one validation prompt."""
    config = InferenceConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        # Public, pinned Hugging Face commit hash; not a credential.
        revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306",  # pragma: allowlist secret
        max_new_tokens=8,
    )
    engine = TransformersBackend(config)
    problem = next(p for p in load_dataset(dataset) if p.split == "validation")
    prompt = build_prompt(problem)
    text = engine.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = engine.tokenizer(text, return_tensors="pt").to("cuda")
    results = []
    for dtype in [engine.torch.float16, engine.torch.bfloat16]:
        engine.model.to(dtype=dtype)
        with engine.torch.inference_mode():
            scores = engine.model(**inputs).logits[0, -1]
        guard_raised = False
        continuation = ""
        try:
            continuation = engine.generate(prompt).text
        except NumericalInferenceError:
            guard_raised = True
        results.append(
            {
                "dtype": str(dtype),
                "finite": bool(scores.isfinite().all()),
                "nan_count": int(scores.isnan().sum()),
                "vocabulary_size": scores.numel(),
                "guard_raised": guard_raised,
                "eight_token_preview": continuation,
            }
        )
    payload = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "example": problem.id,
        "results": results,
        "scope": "One-prompt numerical diagnostic; BF16 converted from FP16 in memory.",
    }
    publish_text(output, json.dumps(payload, indent=2))
    logger.info("Precision diagnostic saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    diagnose(args.dataset, args.output)
