"""Answer-prefix supervision with the same complete teacher-forced JSON target."""

import json

from foco_llm.core.prompts import build_prompt
from foco_llm.core.training_data import IGNORE_INDEX
from foco_llm.models.experiment import Problem
from foco_llm.models.training import TrainingExample
from foco_llm.services.training_backend import TrainingBackend


def mask_answer(
    example: TrainingExample, offsets: tuple[tuple[int, int], ...], start: int, end: int
) -> TrainingExample:
    """Mask evidence and suffix tokens using verified character offsets.

    Args:
        example: Existing completion-masked record with unchanged input tokens.
        offsets: Character spans aligned one-to-one with the input token IDs.
        start: Beginning of the assistant JSON answer prefix.
        end: Exclusive end of that prefix, before the evidence field.

    Returns:
        Record with loss only on tokens fully contained in the answer prefix.

    Raises:
        ValueError: Offsets or boundaries are invalid, or no target survives.
    """
    if len(offsets) != len(example.input_ids) or not 0 <= start < end:
        raise ValueError("Invalid answer mask boundaries or offset count")
    labels = tuple(
        label if start <= left < right <= end else IGNORE_INDEX
        for label, (left, right) in zip(example.labels, offsets, strict=True)
    )
    if all(label == IGNORE_INDEX for label in labels):
        raise ValueError("Answer mask has no supervised tokens")
    return example.model_copy(update={"labels": labels})


class AnswerTrainingBackend(TrainingBackend):
    """Ignore evidence losses without changing facts, prompts, or target serialization."""

    def encode(self, problem: Problem) -> TrainingExample:
        """Check token alignment before masking the assistant evidence suffix."""
        example = super().encode(problem)
        messages = [{"role": "user", "content": build_prompt(problem).text}]
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        target = json.dumps({"answer": problem.answer, "evidence": list(problem.evidence)})
        text = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": target}],
            tokenize=False,
            add_generation_prompt=False,
        )
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if tuple(encoded["input_ids"]) != example.input_ids or not text.startswith(prefix + target):
            raise ValueError("Answer mask tokenization differs from training input")
        # Exclude the closing object brace: the complete target continues with evidence.
        end = len(prefix) + len(json.dumps({"answer": problem.answer})) - 1
        return mask_answer(example, tuple(encoded["offset_mapping"]), len(prefix), end)
