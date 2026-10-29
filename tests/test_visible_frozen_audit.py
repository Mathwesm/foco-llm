"""Regression checks for the visible-text audit of frozen labels."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task


@pytest.mark.parametrize("expected_answer,answer_matches", [("59", True), ("60", False)])
def test_arithmetic_audit_reads_singular_update_and_detects_wrong_label(
    tmp_path: Path, expected_answer: str, answer_matches: bool
) -> None:
    """A one-marble update must affect the independently reconstructed answer."""
    sentences = (
        "Initially, there are 61 marbles inside amber.",
        "Update A: An update to amber: 4 marbles are added to it.",
        "Update B: An update to amber: 6 marbles are removed from it.",
        "Update C: An update to amber: 3 marbles are removed from it.",
        "Update D: An update to amber: 1 marble is removed from it.",
        "Update E: An update to amber: 4 marbles are added to it.",
    )
    problem = Problem(
        id="audit-arithmetic-singular:clean",
        base_id="audit-arithmetic-singular",
        task=Task.ARITHMETIC,
        split=Split.TEST,
        condition=Condition.CLEAN,
        facts=tuple(Fact(id=f"F{index}", text=text) for index, text in enumerate(sentences, 1)),
        question="How many marbles remain in amber?",
        answer=expected_answer,
        evidence=tuple(f"F{index}" for index in range(1, 7)),
    )
    dataset = tmp_path / "dataset.json"
    output = tmp_path / "audit.json"
    dataset.write_text(json.dumps([problem.model_dump(mode="json")]), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_visible_frozen_test.py"

    completed = subprocess.run(  # noqa: S603 -- fixed local audit script and test paths.
        [sys.executable, str(script), "--dataset", str(dataset), "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["answer_failures"] == (0 if answer_matches else 1)
    assert report["evidence_failures"] == 0
    if not answer_matches:
        assert report["discrepancies"][0]["parsed_answer"] == "59"
