"""Regression checks for deduction-specialist data isolation and supervision."""

import json
import runpy
from pathlib import Path

import pytest

from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.models.experiment import Problem, Split, Task


@pytest.fixture
def specialist(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the experiment entrypoint without importing CUDA dependencies."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_deduction_specialist.py"))


def test_prepare_freezes_disjoint_training_and_holdout(
    specialist: dict[str, object], tmp_path: Path
) -> None:
    """Catch target leakage, missing paired cases, or a changed training budget."""
    training, holdout = specialist["prepare"](tmp_path)
    config = PilotConfig.model_validate_json(specialist["CONFIG"].read_text(encoding="utf-8"))
    selected = select_pilot(
        specialist["DATASET_ADAPTER"].validate_json(training.read_bytes()), config
    )
    cases = specialist["DATASET_ADAPTER"].validate_json(holdout.read_text(encoding="utf-8"))
    assert len(selected) == 80
    assert all(case.task == Task.DEDUCTION and case.split == Split.TRAIN for case in selected)
    assert len(cases) == 120
    assert {case.base_id for case in selected}.isdisjoint({case.base_id for case in cases})
    manifest = json.loads((tmp_path / "dataset-manifest.json").read_text(encoding="utf-8"))
    assert manifest["holdout_cases"] == 120


def test_deduction_prompt_omits_private_evidence(
    specialist: dict[str, object], tmp_path: Path
) -> None:
    """Catch accidentally putting gold evidence IDs in the model-visible prompt."""
    training, _ = specialist["prepare"](tmp_path)
    config = PilotConfig.model_validate_json(specialist["CONFIG"].read_text(encoding="utf-8"))
    case: Problem = select_pilot(
        specialist["DATASET_ADAPTER"].validate_json(training.read_bytes()), config
    )[0]
    prompt = specialist["selection_prompt"](case).text
    assert '"evidence":' in prompt  # Format example is allowed.
    assert json.dumps({"evidence": list(case.evidence)}, separators=(",", ":")) not in prompt
