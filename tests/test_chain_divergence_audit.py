"""Regression checks for paired auditing of saved chain-divergence outputs."""

import json
import runpy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest


@pytest.fixture
def audit(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the CPU-only analysis script."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "analyze_chain_divergence.py"))


def test_saved_output_counts_matched_gains_and_losses(audit: dict[str, object]) -> None:
    """Catch accidental independent-case aggregation or wrong model pairing."""
    archive = (
        Path(__file__).resolve().parents[1]
        / "reports/2026-10-08/chain-divergence-kaggle-output.zip"
    )
    result = audit["analyze"](archive)
    assert result["cases"] == 160
    assert result["independent_bases"] == 40
    assert result["overall"]["paired"] == {
        "both_correct": 24,
        "adapter_gain": 59,
        "adapter_loss": 7,
        "both_wrong": 70,
    }
    assert result["by_task"]["tracking"]["paired"]["adapter_loss"] == 0


def test_rejects_archive_with_modified_frozen_dataset(
    audit: dict[str, object], tmp_path: Path
) -> None:
    """Catch a case archive whose dataset no longer matches its manifest."""
    archive = tmp_path / "corrupted.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as destination:
        destination.writestr("chain-divergence-7b/dataset.json", json.dumps([]))
        destination.writestr(
            "chain-divergence-7b/manifest.json", json.dumps({"dataset_sha256": "0" * 64})
        )
    with pytest.raises(ValueError, match="Dataset SHA-256"):
        audit["analyze"](archive)
