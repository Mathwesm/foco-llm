from itertools import pairwise

from foco_llm.core.depth_diagnostic import depth_problems
from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, Split, Task


def test_depth_pairs_preserve_shared_facts_and_order():
    problems = depth_problems()
    assert len(problems) == 120
    assert len({p.id for p in problems}) == 120
    for base in {p.base_id for p in problems}:
        chain = [p for p in problems if p.base_id == base]
        assert [len(p.facts) for p in chain] == [2, 3, 4, 5]
        for smaller, larger in pairwise(chain):
            ids = {f.id for f in smaller.facts}
            assert tuple(f for f in larger.facts if f.id in ids) == smaller.facts
            assert smaller.question == larger.question


def test_diagnostic_does_not_use_final_test_or_noise():
    problems = depth_problems()
    assert all(p.split == Split.VALIDATION and p.condition == Condition.CLEAN for p in problems)
    assert all(p.answer != "pending" and p.answer == solve_v2(p) for p in problems)
    assert all(sum(p.task == task for p in problems) == 40 for task in Task)
    assert problems == depth_problems()
    assert problems != depth_problems(seed=43)
