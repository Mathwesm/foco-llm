"""Behavioral checks for paired model-written note interventions."""

import runpy
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, object]]:
    """Load CPU-only experiment helpers from the repository scripts directory."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return (
        runpy.run_path(str(scripts / "run_ood_chain_holdout.py")),
        runpy.run_path(str(scripts / "run_reasoning_note_intervention.py")),
    )


def test_interventions_change_only_one_note_line(
    modules: tuple[dict[str, object], dict[str, object]],
) -> None:
    """Catch removal of input facts or the wrong note category."""
    holdout, intervention = modules
    selected = intervention["select_cases"](holdout["generate"]())
    assert len(selected) == 20
    case = selected[0]
    raw = "\n".join(f"{fact.id}: USE because I inspected it" for fact in case.facts)
    lines = intervention["parse_note"](raw, case)
    variants = intervention["note_variants"](lines, case)
    assert len(variants["full_note"].splitlines()) == len(case.facts)
    for arm in ("remove_relevant", "remove_irrelevant"):
        assert len(variants[arm].splitlines()) == len(case.facts) - 1
        prompt = intervention["decision_prompt"](case, arm, variants[arm]).text
        assert all(f"[{fact.id}] {fact.text}" in prompt for fact in case.facts)
    removed_relevant = set(lines) - set(variants["remove_relevant"].splitlines())
    removed_irrelevant = set(lines) - set(variants["remove_irrelevant"].splitlines())
    assert next(iter(removed_relevant)).split(":", 1)[0] in case.evidence
    assert next(iter(removed_irrelevant)).split(":", 1)[0] not in case.evidence


@pytest.mark.parametrize(
    "raw", ["F1: USE because needed", "F1: USE because needed\nF1: SKIP because not needed"]
)
def test_incomplete_or_duplicate_note_is_rejected(
    raw: str, modules: tuple[dict[str, object], dict[str, object]]
) -> None:
    """Prevent scoring an unpaired intervention when its source note is malformed."""
    holdout, intervention = modules
    case = intervention["select_cases"](holdout["generate"]())[0]
    with pytest.raises(ValueError, match="one ordered line"):
        intervention["parse_note"](raw, case)


def test_summary_counts_paired_changes_only_when_both_outputs_are_valid(
    modules: tuple[dict[str, object], dict[str, object]],
) -> None:
    """Catch a false causal-change count caused by malformed JSON responses."""
    _, intervention = modules
    rows = [
        {
            "id": "a",
            "task": "deduction",
            "arm": arm,
            "selection_valid": valid,
            "selection_exact": exact,
            "selected_ids": selected,
        }
        for arm, valid, exact, selected in (
            ("full_note", True, True, ["F1"]),
            ("no_note", True, False, ["F2"]),
            ("remove_relevant", False, False, []),
            ("remove_irrelevant", True, True, ["F1"]),
        )
    ]
    groups = intervention["summarize"](rows)["groups"]
    assert groups["deduction:no_note"]["changed_vs_full"] == 1
    assert groups["deduction:remove_relevant"]["valid_pairs_vs_full"] == 0
    assert groups["deduction:remove_irrelevant"]["changed_vs_full"] == 0
