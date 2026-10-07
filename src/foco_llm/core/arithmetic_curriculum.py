"""Create shorter arithmetic training questions from training-only facts."""

from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, Problem, Split, Task

MIN_UPDATES = 2


def arithmetic_prefixes(problem: Problem) -> tuple[Problem, ...]:
    """Derive verified partial sums without changing the original training case.

    Args:
        problem: One arithmetic example from the training split.

    Returns:
        Increasing clean prefixes before the full operation sequence.

    Raises:
        ValueError: The source has no unambiguous initial count and updates.
    """
    if problem.split != Split.TRAIN or problem.task != Task.ARITHMETIC:
        raise ValueError("Arithmetic prefixes require a training arithmetic problem")
    relevant = tuple(f for f in problem.facts if f.id in problem.evidence)
    starts = tuple(f for f in relevant if "initial" in f.text.lower())
    updates = tuple(f for f in relevant if f not in starts)
    if len(starts) != 1 or len(updates) < MIN_UPDATES:
        raise ValueError("Arithmetic source requires one initial count and two updates")
    output: list[Problem] = []
    for count in range(1, len(updates)):
        facts = (starts[0], *updates[:count])
        partial = Problem(
            id=f"{problem.id}:prefix-{count}",
            base_id=f"{problem.base_id}:prefix-{count}",
            task=Task.ARITHMETIC,
            split=Split.TRAIN,
            condition=Condition.CLEAN,
            facts=facts,
            question=problem.question,
            answer="0",
            evidence=tuple(f.id for f in facts),
        )
        output.append(partial.model_copy(update={"answer": solve_v2(partial)}))
    return tuple(output)


def with_arithmetic_prefixes(selected: tuple[Problem, ...]) -> tuple[Problem, ...]:
    """Interleave partial sums immediately before each full arithmetic source."""
    output: list[Problem] = []
    for problem in selected:
        if problem.task == Task.ARITHMETIC:
            output.extend(arithmetic_prefixes(problem))
        output.append(problem)
    return tuple(output)
