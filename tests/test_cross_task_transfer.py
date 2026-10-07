"""Behavior checks for frozen cross-task evidence selection."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def transfer_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CLI with the production scripts available for imports."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_cross_task_transfer.py"))


def test_frozen_pairs_are_solvable_from_visible_facts(
    transfer_script: dict[str, object],
) -> None:
    """Catch missing chains, incorrect labels, and unsolvable distractors."""
    cases = transfer_script["generate"]()
    transfer_script["validate"](cases)
    assert len(cases) == 120
    for case in cases:
        selected = transfer_script["symbolic_select"](case)
        assert set(selected) == set(case.evidence)
    for first, second in zip(cases[::2], cases[1::2], strict=True):
        assert first.base_id == second.base_id
        assert first.answer == second.answer


def test_scoring_rejects_missing_and_extra_evidence(
    transfer_script: dict[str, object],
) -> None:
    """Catch a selector that accepts incomplete or contaminated chains."""
    case = transfer_script["generate"]()[1]
    expected = list(case.evidence)
    extra = next(fact.id for fact in case.facts if fact.id not in case.evidence)
    score = transfer_script["score"]
    exact = score(case, json.dumps({"evidence": expected}))
    missing = score(case, json.dumps({"evidence": expected[:-1]}))
    contaminated = score(case, json.dumps({"evidence": [*expected, extra]}))
    assert exact["selection_exact"] is True
    assert missing["selection_exact"] is False
    assert missing["false_negative"] == 1
    assert contaminated["selection_exact"] is False
    assert contaminated["false_positive"] == 1


def test_symbolic_control_ignores_hidden_labels(
    transfer_script: dict[str, object],
) -> None:
    """Catch accidental access to the private evidence field."""
    case = transfer_script["generate"]()[0]
    corrupted = case.model_copy(update={"evidence": ("F1",)})
    assert set(transfer_script["symbolic_select"](corrupted)) == set(case.evidence)
