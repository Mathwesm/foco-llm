"""Bounded source-task pilot configuration and training-only selection."""

import hashlib
from typing import Literal

from pydantic import Field

from foco_llm.models.experiment import Condition, Problem, Split, Task
from foco_llm.models.training import SmokeConfig


class PilotConfig(SmokeConfig):
    """Development pilot using answer-and-evidence supervision, not a full sweep."""

    steps: int = Field(default=64, ge=2, le=256)
    source_task: Task = Task.DEDUCTION
    condition: Condition = Condition.SIMILAR
    examples: int = Field(default=32, ge=1, le=80)
    selection_seed: int = Field(default=42, ge=0)
    supervision: Literal["answer", "answer_and_evidence"] = "answer_and_evidence"


def select_pilot(problems: tuple[Problem, ...], config: PilotConfig) -> tuple[Problem, ...]:
    """Select source-task training bases without inspecting answers or model success.

    Args:
        problems: Frozen dataset containing all splits.
        config: Source task, condition, seed, and bounded example budget.

    Returns:
        Deterministically ordered source examples, independent of input file order.

    Raises:
        ValueError: Too few examples or duplicate source bases are present.
    """
    selected = [
        p
        for p in problems
        if p.split == Split.TRAIN
        and p.task == config.source_task
        and p.condition == config.condition
    ]
    if len({p.base_id for p in selected}) != len(selected):
        raise ValueError("Duplicate source training bases")
    if len(selected) < config.examples:
        raise ValueError("Insufficient source training examples")
    selected.sort(
        key=lambda p: hashlib.sha256(f"{config.selection_seed}:{p.base_id}".encode()).digest()
    )
    return tuple(selected[: config.examples])
