"""Regression checks for the out-of-template chain holdout."""

import json
import runpy
from pathlib import Path

import pytest

from foco_llm.models.experiment import Fact, Problem


@pytest.fixture
def holdout(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CPU-only holdout script from the installed project."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_ood_chain_holdout.py"))


def test_holdout_freezes_new_grammar_and_paired_gold(
    holdout: dict[str, object], tmp_path: Path
) -> None:
    """Catch label drift, pair mismatch, or accidental old-template reuse."""
    dataset = holdout["prepare"](tmp_path)
    cases: tuple[Problem, ...] = holdout["DATASET_ADAPTER"].validate_json(dataset.read_bytes())
    assert len(cases) == 120
    assert len({case.base_id for case in cases}) == 60
    for unrelated, similar in zip(cases[::2], cases[1::2], strict=True):
        assert unrelated.base_id == similar.base_id
        assert unrelated.answer == similar.answer
        assert unrelated.evidence == similar.evidence
        assert [unrelated.facts[int(key[1:]) - 1].text for key in unrelated.evidence] == [
            similar.facts[int(key[1:]) - 1].text for key in similar.evidence
        ]
        assert all("Property P" not in fact.text for fact in unrelated.facts)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["oracle_exact"] == 120
    assert holdout["prepare"](tmp_path) == dataset


def test_oracle_rejects_broken_chain(holdout: dict[str, object]) -> None:
    """Catch a gapped gold chain that would make model scores meaningless."""
    case: Problem = holdout["generate"]()[0]
    facts = tuple(
        Fact(id=fact.id, text="A broken edge.") if fact.id == case.evidence[0] else fact
        for fact in case.facts
    )
    corrupted = case.model_copy(update={"facts": facts})
    with pytest.raises(ValueError, match="Chain is missing or ambiguous|exactly one start"):
        holdout["_resolve"](corrupted)
