"""Behavioral checks for the frozen single-fact intervention."""

import json
import runpy
from pathlib import Path

import pytest

from foco_llm.core.arithmetic_selection import selection_prompt_v2


@pytest.fixture
def ablation_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CLI with its sibling scripts available for imports."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_fact_ablation.py"))


def test_removal_changes_only_intended_fact_and_reference(
    ablation_script: dict[str, object],
) -> None:
    """Catch accidental changes to the question, other facts, or paired answer."""
    cases = ablation_script["variants"]()
    assert len(cases) == ablation_script["BASES"] * len(ablation_script["FORMS"])
    for full, remove_relevant, remove_irrelevant in zip(
        cases[::3], cases[1::3], cases[2::3], strict=True
    ):
        assert full.question == remove_relevant.question == remove_irrelevant.question
        full_ids = {fact.id for fact in full.facts}
        assert len(full_ids - {fact.id for fact in remove_relevant.facts}) == 1
        assert len(full_ids - {fact.id for fact in remove_irrelevant.facts}) == 1
        assert len(remove_relevant.evidence) == len(full.evidence) - 1
        assert remove_irrelevant.evidence == full.evidence
        assert remove_relevant.answer != full.answer
        assert remove_irrelevant.answer == full.answer


def test_reference_selection_remains_executable_after_ablation(
    ablation_script: dict[str, object],
) -> None:
    """Catch counterfactual labels that cannot be scored or omit visible facts."""
    score_selection = ablation_script["score_selection"]
    for case in ablation_script["variants"]():
        raw = json.dumps({"evidence": case.evidence})
        assert score_selection(case, raw)["answer_correct"] is True
        assert case.question in selection_prompt_v2(case).text
