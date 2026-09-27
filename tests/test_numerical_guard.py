import math

import pytest

from foco_llm.services.transformers_backend import NumericalInferenceError, check_logits


class Flags:
    """Minimal tensor-shaped boolean reductions without GPU dependencies."""

    def __init__(self, rows):
        self.rows = rows

    def any(self, dim=None):
        """Reduce rows or all values with tensor-compatible argument names."""
        if dim == -1:
            return Flags([[any(row) for row in self.rows]])
        return any(value for row in self.rows for value in row)

    def all(self):
        """Require every boolean value."""
        return all(value for row in self.rows for value in row)


class Scores:
    """Numeric fixture exposing the score operations used by the generation hook."""

    def __init__(self, rows):
        self.rows = rows

    def isnan(self):
        """Identify invalid NaN values."""
        return Flags([[math.isnan(value) for value in row] for row in self.rows])

    def isposinf(self):
        """Identify positive infinities, excluding intentional negative masks."""
        return Flags([[value == math.inf for value in row] for row in self.rows])

    def isfinite(self):
        """Identify finite token candidates."""
        return Flags([[math.isfinite(value) for value in row] for row in self.rows])


@pytest.mark.parametrize(
    "rows",
    [
        [[math.nan, math.nan]],
        [[1.0, math.nan]],
        [[1.0, math.inf]],
        [[-math.inf, -math.inf]],
        [[1.0, 2.0], [-math.inf, -math.inf]],
    ],
)
def test_invalid_scores_raise_instead_of_generating_token_zero(rows):
    with pytest.raises(NumericalInferenceError, match="Invalid logits"):
        check_logits(None, Scores(rows))


def test_finite_scores_and_negative_token_masks_remain_unchanged():
    scores = Scores([[1.0, -math.inf], [-2.0, 3.0]])
    assert check_logits(None, scores) is scores
