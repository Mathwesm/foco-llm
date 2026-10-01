"""Behavioral checks for independent relevance selection and execution."""

import runpy
from pathlib import Path

import pytest

from foco_llm.core.arithmetic_diagnostic import arithmetic_probes
from foco_llm.core.arithmetic_selection import (
    lexical_select,
    parse_selected_ids,
    parse_selected_ids_v2,
    selected_calculation,
    selection_prompt_v2,
)
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.pilot_configuration import PilotConfig
from foco_llm.models.experiment import Condition, GenerationConfig, Split, Task
from foco_llm.services.selection_training import selection_target


def _similar_case():
    source = generate_v21(GenerationConfig(seed=20261002, problems_per_task=10))
    clean = next(
        p
        for p in source
        if p.task == Task.ARITHMETIC
        and p.split == Split.VALIDATION
        and p.condition == Condition.CLEAN
    )
    similar = next(
        p for p in source if p.base_id == clean.base_id and p.condition == Condition.SIMILAR
    )
    return arithmetic_probes(clean, similar)[-1]


def test_prompt_does_not_reveal_reference_evidence_or_answer():
    problem = _similar_case()
    prompt = selection_prompt_v2(problem)
    assert problem.question in prompt.text
    assert all(f"[{fact.id}] {fact.text}" in prompt.text for fact in problem.facts)
    assert str(list(problem.evidence)) not in prompt.text
    assert f'"answer": "{problem.answer}"' not in prompt.text


@pytest.mark.parametrize("raw", ['["F1", "F1"]', '["F999"]', "[]", '{"facts":["F1"]}', "[1]"])
def test_selection_rejects_malformed_or_invalid_ids(raw):
    with pytest.raises(ValueError, match="Selection"):
        parse_selected_ids(raw, _similar_case().facts)


def test_v2_parser_requires_evidence_object():
    facts = _similar_case().facts
    assert parse_selected_ids_v2('{"evidence":["F1"]}', facts) == ("F1",)
    with pytest.raises(ValueError, match="evidence field"):
        parse_selected_ids_v2('["F1"]', facts)


def test_calculation_uses_only_selected_facts():
    problem = _similar_case()
    expression, result = selected_calculation(problem.facts, problem.evidence)
    assert result == problem.answer
    assert expression.count("+") + expression.count("-") == 4
    distractor = next(
        f.id for f in problem.facts if f.id not in problem.evidence and "initial" in f.text
    )
    selected = (*problem.evidence, distractor)
    with pytest.raises(ValueError, match="exactly one initial count"):
        selected_calculation(problem.facts, selected)


def test_scoring_separates_selection_failure_from_arithmetic():
    problem = _similar_case()
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_arithmetic_selection.py"
    score = runpy.run_path(str(script))["score_selection"]
    correct = score(
        problem,
        '{"evidence":[' + ",".join(f'"{fact}"' for fact in problem.evidence) + "]}",
    )
    invalid = score(problem, '{"evidence":["F999"]}')
    assert correct["selection_exact"] and correct["answer_correct"]
    assert not invalid["selection_valid"] and not invalid["answer_correct"]


def test_evidence_only_training_target_excludes_answer():
    problem = _similar_case()
    target = selection_target(problem)
    assert target.startswith('{"evidence":[')
    assert '"answer"' not in target
    assert target != problem.answer
    config = PilotConfig(
        revision="a" * 40,
        source_task=Task.ARITHMETIC,
        supervision="evidence_only",
        examples=8,
    )
    assert config.supervision == "evidence_only"


def test_lexical_control_uses_visible_target_only():
    problem = _similar_case()
    assert set(lexical_select(problem)) == set(problem.evidence)
    changed = problem.model_copy(update={"question": "How many marbles remain in ivory?"})
    assert set(lexical_select(changed)) != set(problem.evidence)
