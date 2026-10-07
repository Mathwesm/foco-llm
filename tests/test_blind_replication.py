"""Behavior checks for the frozen blind replication."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def replication_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CLI with its script dependencies on the production path."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_blind_replication.py"))


def test_replication_pairs_and_symbolic_control(
    replication_script: dict[str, object],
) -> None:
    """Catch broken pairing, labels, or visible alias resolution."""
    cases = replication_script["generate"]()
    replication_script["validate"](cases)
    assert len(cases) == 120
    for explicit, alias in zip(cases[::2], cases[1::2], strict=True):
        assert explicit.answer == alias.answer
        for case in (explicit, alias):
            assert set(replication_script["symbolic_select"](case)) == set(case.evidence)


def test_symbolic_control_does_not_trust_private_evidence(
    replication_script: dict[str, object],
) -> None:
    """Catch accidental use of hidden labels by the symbolic baseline."""
    case = replication_script["generate"]()[1]
    corrupted = case.model_copy(update={"evidence": ("F2",)})
    assert replication_script["symbolic_select"](corrupted) != corrupted.evidence
    assert set(replication_script["symbolic_select"](corrupted)) == set(case.evidence)
