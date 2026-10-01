"""Behavioral regression checks for the arithmetic calculation probes."""

import runpy
from pathlib import Path

import pytest

from foco_llm.core.arithmetic_diagnostic import arithmetic_probes
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import Condition, GenerationConfig, Split, Task
from foco_llm.models.inference import Generation


def _source_cases():
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
    return clean, similar


def test_probe_depths_and_answers_are_independently_reproducible():
    clean, similar = _source_cases()
    probes = arithmetic_probes(clean, similar)
    assert len(probes) == 7
    assert len({p.id for p in probes}) == 7
    assert all(p.split == Split.VALIDATION for p in probes)
    assert [p.answer for p in probes if ":text-" in p.id] == [
        solve_v2(p) for p in probes if ":text-" in p.id
    ]
    assert next(p for p in probes if p.id.endswith("similar-4")).answer == clean.answer
    assert [p.answer for p in probes if ":expression-" in p.id] == [
        p.answer for p in probes if ":text-" in p.id
    ]


def test_probe_builder_rejects_unpaired_source():
    clean, similar = _source_cases()
    with pytest.raises(ValueError, match="paired validation"):
        arithmetic_probes(clean, similar.model_copy(update={"base_id": "other"}))


def test_scoring_separates_wrong_math_from_unreadable_format():
    clean, similar = _source_cases()
    expression = arithmetic_probes(clean, similar)[0]
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_arithmetic_diagnostic.py"
    score = runpy.run_path(str(script))["score_probe"]
    correct = score(expression, f'{{"answer":"{expression.answer}","evidence":["F1"]}}')
    wrong = score(expression, '{"answer":"999","evidence":["F1"]}')
    malformed = score(expression, "I think 999")
    assert correct["strict_valid"] and correct["answer_correct"]
    assert wrong["strict_valid"] and not wrong["answer_correct"]
    assert not malformed["strict_valid"] and not malformed["answer_readable"]


def test_plain_expression_prompt_and_integer_scoring():
    clean, similar = _source_cases()
    expression = arithmetic_probes(clean, similar)[0]
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_plain_arithmetic.py"
    namespace = runpy.run_path(str(script))
    prompt = namespace["plain_prompt"](expression)
    score = namespace["score_integer"]
    assert expression.answer not in prompt.text
    assert "evidence" not in prompt.text
    assert score(expression.answer, expression.answer) == (True, True)
    assert score("The answer is " + expression.answer, expression.answer) == (False, False)
    assert score("999", expression.answer) == (True, False)


def test_boxed_scoring_rejects_missing_or_multiple_final_answers():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_plain_arithmetic.py"
    score = runpy.run_path(str(script))["score_boxed"]
    assert score("The result is \\boxed{41}.", "41") == (True, True)
    assert score("The result is \\boxed{40}.", "41") == (True, False)
    assert score("41", "41") == (False, False)
    assert score("\\boxed{40}, then \\boxed{41}", "41") == (False, False)


def test_chained_calls_feed_model_output_forward(tmp_path):
    clean, similar = _source_cases()
    problem = next(p for p in arithmetic_probes(clean, similar) if p.id.endswith("expression-4"))
    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "run_chained_arithmetic.py")
    )
    initial, updates = namespace["operations"](problem)

    class WrongFirstBackend:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt):
            self.prompts.append(prompt.text)
            return Generation(
                text="0",
                input_tokens=1,
                output_tokens=1,
                elapsed_seconds=0,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
                stop_reason="eos",
            )

    backend = WrongFirstBackend()
    result = namespace["run_case"](problem, tmp_path, backend)
    assert len(backend.prompts) == 4
    assert backend.prompts[0] == (
        f"Calculate {initial} {updates[0][0]} {updates[0][1]}. Reply with only the integer."
    )
    assert backend.prompts[1] == (
        f"Calculate 0 {updates[1][0]} {updates[1][1]}. Reply with only the integer."
    )
    assert not result["all_steps_correct"]
    assert not result["final_correct"]
