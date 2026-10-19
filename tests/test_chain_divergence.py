"""Behavioral checks for the controlled chain-divergence holdout."""

import json
import runpy
from pathlib import Path

import pytest

from foco_llm.models.experiment import Fact, Problem


@pytest.fixture
def protocol(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CPU-only generator from the installed project."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_chain_divergence.py"))


def test_four_arms_keep_gold_and_positions_frozen(
    protocol: dict[str, object], tmp_path: Path
) -> None:
    """Catch answer drift, missing pairs and distractor-position confounding."""
    dataset = protocol["prepare"](tmp_path)
    cases: tuple[Problem, ...] = protocol["ADAPTER"].validate_json(dataset.read_bytes())
    assert len(cases) == 160
    assert len({case.base_id for case in cases}) == 40
    for offset in range(0, len(cases), 4):
        variants = cases[offset : offset + 4]
        assert [case.id.rsplit(":", 1)[1] for case in variants] == [
            "unrelated",
            "early",
            "middle",
            "late",
        ]
        assert len({case.answer for case in variants}) == 1
        assert len({case.evidence for case in variants}) == 1
        assert all(len(case.facts) == 8 for case in variants)
        assert all(
            protocol["symbolic_oracle"](case) == (case.answer, case.evidence) for case in variants
        )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["oracle_exact"] == 160
    assert protocol["prepare"](tmp_path) == dataset


@pytest.mark.parametrize("task_offset", [0, 80])
def test_oracle_rejects_missing_target_edge(protocol: dict[str, object], task_offset: int) -> None:
    """Reject a broken deduction or tracking chain instead of scoring it."""
    case: Problem = protocol["generate"]()[task_offset]
    fact_id = case.evidence[1]
    corrupted = case.model_copy(
        update={
            "facts": tuple(
                Fact(id=fact.id, text="This edge is missing.") if fact.id == fact_id else fact
                for fact in case.facts
            )
        }
    )
    with pytest.raises(ValueError, match="missing or ambiguous"):
        protocol["symbolic_oracle"](corrupted)


def test_pair_summary_counts_control_failures(protocol: dict[str, object]) -> None:
    """Keep the control denominator when all answers are wrong (floor effect)."""
    cases: tuple[Problem, ...] = protocol["generate"]()
    rows = [
        {
            "id": case.id,
            "task": case.task.value,
            "selection_valid": False,
            "selection_exact": False,
            "false_positive": 0,
            "false_negative": 4,
        }
        for case in cases
    ]
    summary = protocol["summarize"](rows)
    assert summary["paired"]["deduction:early"] == {
        "bases": 20,
        "control_exact": 0,
        "both_exact": 0,
        "control_only": 0,
        "rival_only": 0,
    }
