import hashlib
import re

import pytest
from pydantic import TypeAdapter

from foco_llm.core.benchmark_v2 import generate_v2
from foco_llm.core.benchmark_v21 import generate_v21, validate_v21
from foco_llm.core.validation import DatasetError
from foco_llm.models.experiment import Condition, GenerationConfig, Problem


def test_revision_preserves_legacy_data_and_answers():
    config = GenerationConfig()
    old = generate_v2(config)
    payload = TypeAdapter(tuple[Problem, ...]).dump_json(old, indent=2)
    # Audited public dataset digest, not a credential.
    expected_prefix = "f6dd4f65804eb724"  # pragma: allowlist secret
    assert hashlib.sha256(payload).hexdigest().startswith(expected_prefix)
    revised = generate_v21(config)
    assert [(p.id, p.answer, p.split) for p in old] == [(p.id, p.answer, p.split) for p in revised]
    assert revised == generate_v21(config)


def test_old_candidate_exposes_pairing_confound():
    with pytest.raises(DatasetError, match="singular|order|sentence count"):
        validate_v21(generate_v2(GenerationConfig(problems_per_task=10)))


def test_singular_events_are_correct_and_distinct():
    problems = generate_v21(GenerationConfig(problems_per_task=10))
    sentences = [f.text for p in problems for f in p.facts]
    assert any("1 marble is" in text for text in sentences)
    assert not any(re.search(r"\b1 marbles\b", text) for text in sentences)
    for problem in problems:
        relevant = [f.text for f in problem.facts if f.id in problem.evidence]
        assert len(relevant) == len(set(relevant))


@pytest.mark.parametrize("damage", ["order", "count"])
def test_audit_rejects_order_changes_and_extra_distractors(damage):
    problems = list(generate_v21(GenerationConfig(problems_per_task=10)))
    index = next(i for i, p in enumerate(problems) if p.condition == Condition.UNRELATED)
    problem = problems[index]
    facts = list(problem.facts)
    if damage == "order":
        positions = [i for i, f in enumerate(facts) if f.id in problem.evidence]
        a, b = positions[:2]
        facts[a], facts[b] = facts[b], facts[a]
    else:
        facts.append(facts[-1].model_copy(update={"id": "F999", "text": "A distant door is open."}))
    problems[index] = problem.model_copy(update={"facts": tuple(facts)})
    with pytest.raises(DatasetError, match="order|sentence count|positions"):
        validate_v21(tuple(problems))
