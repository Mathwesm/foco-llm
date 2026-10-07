import runpy
from pathlib import Path

import pytest

from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.models.experiment import Condition, GenerationConfig, Split


def test_reversal_preserves_semantics_and_identifiers():
    reverse = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "diagnose_order.py"))[
        "reverse_facts"
    ]
    problems = generate_v21(GenerationConfig(problems_per_task=10))
    for p in problems:
        if p.split == Split.VALIDATION and p.condition == Condition.CLEAN:
            changed = reverse(p)
            assert changed.facts == p.facts[::-1]
            assert changed.facts != p.facts
            assert changed.model_dump(exclude={"facts"}) == p.model_dump(exclude={"facts"})
            with pytest.raises(ValueError, match="not invariant"):
                reverse(p.model_copy(update={"answer": "wrong"}))
        else:
            with pytest.raises(ValueError, match="clean validation"):
                reverse(p)
