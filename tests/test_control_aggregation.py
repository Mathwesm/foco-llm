import runpy
from pathlib import Path

import pytest


def test_bootstrap_uses_bases_as_units_and_is_reproducible():
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts" / "aggregate_local_controls.py")
    )
    interval = module["clustered_interval"]([0.0, 1.0], seed=42)
    assert interval == (0.0, 1.0)
    assert module["clustered_interval"]([0.0, 1.0], seed=42) == interval


def test_bootstrap_rejects_missing_bases():
    module = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts" / "aggregate_local_controls.py")
    )
    with pytest.raises(ValueError, match="empty"):
        module["clustered_interval"]([])
