import pytest

from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.format_diagnostic import diagnostic_prompt, format_answer
from foco_llm.models.experiment import GenerationConfig


@pytest.mark.parametrize(
    "text,expected",
    [
        (" 47\n", "47"),
        ("B7771", "B7771"),
        ('"47"', None),
        ('{"answer":"47"}', None),
        ("Answer: 47", None),
        ("47.0", None),
        ("F4", None),
        ("", None),
        ("47\n48", None),
    ],
)
def test_plain_answer_never_repairs_or_extracts(text, expected):
    assert format_answer(text, "plain_answer") == expected


def test_json_contract_preserves_prior_schema_and_envelope_rules():
    assert format_answer('```json\n{"answer":"47","evidence":[]}\n```', "explicit_json") == "47"
    assert format_answer('{"answer":47,"evidence":[]}', "explicit_json") is None


@pytest.mark.parametrize("arm", ["plain_answer", "explicit_json"])
def test_prompt_does_not_depend_on_privileged_annotations(arm):
    problem = generate_v21(GenerationConfig(problems_per_task=10))[0]
    changed = problem.model_copy(
        update={"answer": "PRIVATE_REFERENCE", "evidence": ("PRIVATE_FACT",)}
    )
    assert diagnostic_prompt(problem, arm) == diagnostic_prompt(changed, arm)
    text = diagnostic_prompt(problem, arm).text
    assert problem.question in text
    for fact in problem.facts:
        assert f"[{fact.id}] {fact.text}" in text
