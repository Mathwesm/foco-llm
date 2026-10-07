"""Regression checks for the imported Kaggle evidence totals."""

import runpy
from pathlib import Path

import pytest

module = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "import_kaggle_seed_replication.py")
)
Score = module["Score"]
verify_scores = module["verify_scores"]


def test_verify_scores_accepts_independent_group_totals() -> None:
    """A complete two-form score set must match its saved summary."""
    scores = [
        Score(id="base-1:explicit", form="explicit", selection_exact=True, answer_correct=True),
        Score(id="base-1:alias", form="alias", selection_exact=False, answer_correct=False),
    ]
    summary = {
        "cases": 2,
        "groups": {
            "explicit": {"cases": 1, "selection_exact": 1, "answer_correct": 1},
            "alias": {"cases": 1, "selection_exact": 0, "answer_correct": 0},
        },
    }
    verify_scores(scores, summary, ("explicit", "alias"))


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (
            [
                Score(id="same", form="explicit", selection_exact=True, answer_correct=True),
                Score(id="same", form="alias", selection_exact=False, answer_correct=False),
            ],
            "Duplicate case identifiers",
        ),
        (
            [
                Score(id="a", form="explicit", selection_exact=False, answer_correct=True),
                Score(id="b", form="alias", selection_exact=False, answer_correct=False),
            ],
            "scores disagree with summary for explicit",
        ),
    ],
)
def test_verify_scores_rejects_duplicate_or_inflated_totals(
    scores: list[Score], message: str
) -> None:
    """Duplicate IDs and reported exact counts without backing scores fail."""
    summary = {
        "cases": 2,
        "groups": {
            "explicit": {"cases": 1, "selection_exact": 1, "answer_correct": 1},
            "alias": {"cases": 1, "selection_exact": 0, "answer_correct": 0},
        },
    }
    with pytest.raises(ValueError, match=message):
        verify_scores(scores, summary, ("explicit", "alias"))
