"""Training-only selection and explicit completion masking without truncation."""

from foco_llm.models.experiment import Condition, Problem, Split, Task
from foco_llm.models.training import TrainingExample

IGNORE_INDEX = -100


def select_smoke_examples(problems: tuple[Problem, ...]) -> tuple[Problem, ...]:
    """Select one clean/similar training pair per task without exposing evaluation data."""
    output = []
    for task in Task:
        candidates = [p for p in problems if p.task == task and p.split == Split.TRAIN]
        if not candidates:
            raise ValueError(f"No training examples for task: {task}")
        base = min(p.base_id for p in candidates)
        for condition in (Condition.CLEAN, Condition.SIMILAR):
            matches = [p for p in candidates if p.base_id == base and p.condition == condition]
            if len(matches) != 1:
                raise ValueError("Expected one paired training example per selected condition")
            output.extend(matches)
    return tuple(output)


def mask_completion(
    identifier: str, prefix: tuple[int, ...], full: tuple[int, ...], max_tokens: int
) -> TrainingExample:
    """Mask only the verified prompt prefix and reject empty or truncated targets."""
    if not prefix or len(full) <= len(prefix) or full[: len(prefix)] != prefix:
        raise ValueError("Completion tokenization must preserve a nonempty prompt prefix")
    if len(full) > max_tokens:
        raise ValueError("Training sequence exceeds limit; truncation is forbidden")
    return TrainingExample(
        id=identifier,
        input_ids=full,
        labels=(IGNORE_INDEX,) * len(prefix) + full[len(prefix) :],
        prompt_tokens=len(prefix),
    )
