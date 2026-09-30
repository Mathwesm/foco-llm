"""Bounded source-task pilot configuration and training-only selection."""

import hashlib
from typing import Literal, Self

from pydantic import Field, model_validator

from foco_llm.models.experiment import Condition, Problem, Split, Task
from foco_llm.models.training import SmokeConfig


class PilotConfig(SmokeConfig):
    """Development pilot using answer-and-evidence supervision, not a full sweep."""

    steps: int = Field(default=64, ge=2, le=256)
    source_task: Task | None = Task.DEDUCTION
    condition: Condition = Condition.SIMILAR
    examples: int = Field(default=32, ge=1, le=240)
    selection_seed: int = Field(default=42, ge=0)
    supervision: Literal["answer", "answer_and_evidence"] = "answer_and_evidence"
    tasks: tuple[Task, ...] | None = None

    @model_validator(mode="after")
    def validate_tasks(self) -> Self:
        """Require an even training budget for every explicitly selected task."""
        if self.tasks is None and self.source_task is None:
            raise ValueError("A source task or explicit task list is required")
        if self.tasks is not None and (
            not self.tasks
            or len(set(self.tasks)) != len(self.tasks)
            or self.examples % len(self.tasks) != 0
        ):
            raise ValueError("Tasks must be unique and divide the example budget evenly")
        return self


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
    tasks = config.tasks if config.tasks is not None else (config.source_task,)
    if any(task is None for task in tasks):
        raise ValueError("A source task or explicit task list is required")
    per_task = config.examples // len(tasks)
    groups: list[list[Problem]] = []
    for task in tasks:
        selected = [
            p
            for p in problems
            if p.split == Split.TRAIN and p.task == task and p.condition == config.condition
        ]
        if len({p.base_id for p in selected}) != len(selected):
            raise ValueError("Duplicate source training bases")
        if len(selected) < per_task:
            raise ValueError("Insufficient source training examples")
        selected.sort(
            key=lambda p: hashlib.sha256(f"{config.selection_seed}:{p.base_id}".encode()).digest()
        )
        groups.append(selected[:per_task])
    return tuple(
        groups[task_index][index] for index in range(per_task) for task_index in range(len(tasks))
    )
