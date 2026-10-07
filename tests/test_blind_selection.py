"""Behavioral checks for the frozen blind evidence-selection protocol."""

import json
import runpy
from pathlib import Path

import pytest

from foco_llm.core.arithmetic_selection import lexical_select
from foco_llm.models.experiment import Split


@pytest.fixture
def blind_script(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Load the command module with the same script import path as production."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / "run_blind_selection.py"))


def test_pairs_preserve_answer_and_executable_evidence(blind_script: dict[str, object]) -> None:
    """Catch a cue change that accidentally changes arithmetic or labels."""
    problems = blind_script["generate"]()
    assert len(problems) == blind_script["BASES"] * len(blind_script["FORMS"])
    for explicit, alias in zip(problems[::2], problems[1::2], strict=True):
        assert explicit.base_id == alias.base_id
        assert explicit.answer == alias.answer
        for problem in (explicit, alias):
            scored = blind_script["score_selection"](
                problem, json.dumps({"evidence": problem.evidence})
            )
            assert scored["selection_exact"] is True
            assert scored["answer_correct"] is True


def test_alias_condition_defeats_literal_target_rule(blind_script: dict[str, object]) -> None:
    """Catch accidental target-name leakage into alias updates or missed decoys."""
    problems = blind_script["generate"]()
    for explicit, alias in zip(problems[::2], problems[1::2], strict=True):
        assert lexical_select(explicit) == explicit.evidence
        assert lexical_select(alias) != alias.evidence


def test_inference_manifest_identifies_held_out_split(blind_script: dict[str, object]) -> None:
    """Catch a misleading validation label in fresh-test inference records."""
    config = blind_script["inference_config"](
        blind_script["LOCAL_ID"], blind_script["LOCAL_REVISION"]
    )
    assert config.split == Split.TEST
    cloud_config = blind_script["inference_config"](
        blind_script["CLOUD_ID"], blind_script["CLOUD_REVISION"]
    )
    assert cloud_config.split == Split.TEST
    assert cloud_config.precision == "float32"
