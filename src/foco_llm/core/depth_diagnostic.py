"""Controlled clean chains, separate from the frozen benchmark and its test set."""

from foco_llm.core.benchmark_v2 import TEMPLATES, _arithmetic, _deduction, _rng, _tracking
from foco_llm.core.benchmark_v21 import _sentences
from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task


def depth_problems(seed: int = 42) -> tuple[Problem, ...]:
    """Build ten paired chains per task at depths one through four.

    Args:
        seed: Explicit generation seed in a diagnostic-only random namespace.

    Returns:
        Clean development examples with fixed validation language and relative order.
    """
    output = []
    builders = {Task.ARITHMETIC: _arithmetic, Task.DEDUCTION: _deduction, Task.TRACKING: _tracking}
    questions = {
        Task.ARITHMETIC: "How many marbles remain in amber?",
        Task.DEDUCTION: "What terminal property follows for amber? Return its code.",
        Task.TRACKING: "Which box contains amber after all timed events? Return its code.",
    }
    for task in Task:
        for index in range(10):
            base = f"depth-diagnostic-{task}-{index:03d}"
            texts, _ = builders[task]("amber", 4, _rng(seed, base), TEMPLATES[Split.VALIDATION])
            facts = tuple(Fact(id=f"F{i}", text=t) for i, t in enumerate(texts, 1))
            revised = _sentences(facts, task, seed, base)
            order = list(range(len(facts)))
            _rng(seed, f"{base}:order").shuffle(order)
            for depth in range(1, 5):
                selected = tuple(Fact(id=f"F{i + 1}", text=revised[i]) for i in order if i <= depth)
                problem = Problem(
                    id=f"{base}:depth-{depth}",
                    base_id=base,
                    task=task,
                    split=Split.VALIDATION,
                    condition=Condition.CLEAN,
                    facts=selected,
                    question=questions[task],
                    answer="pending",
                    evidence=tuple(f.id for f in selected),
                )
                output.append(problem.model_copy(update={"answer": solve_v2(problem)}))
    return tuple(output)
