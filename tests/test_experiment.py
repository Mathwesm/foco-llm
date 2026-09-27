from pathlib import Path

import pytest
from pydantic import ValidationError

from foco_llm.__main__ import main
from foco_llm.core.evaluation import evaluate
from foco_llm.core.generation import generate_problems
from foco_llm.core.prompts import build_prompt
from foco_llm.core.validation import DatasetError, solve, validate_dataset
from foco_llm.models.experiment import (
    Condition,
    GenerationConfig,
    ModelRun,
    Prediction,
    Split,
    Task,
)
from foco_llm.services.artifacts import load_dataset, prepare_dataset, publish_text


@pytest.fixture
def problems():
    return generate_problems(GenerationConfig(problems_per_task=10))


def make_run(predictions: tuple[Prediction, ...]) -> ModelRun:
    return ModelRun(
        model_id="test-double",
        model_revision="fixture-v1",
        seed=42,
        prompt_version="evidence-json-v1",
        decoding={"sample": False},
        predictions=predictions,
    )


def test_generation_is_reproducible_and_seed_sensitive(problems):
    assert problems == generate_problems(GenerationConfig(problems_per_task=10))
    assert problems != generate_problems(GenerationConfig(problems_per_task=10, seed=43))


def test_variants_share_answers_and_splits(problems):
    validate_dataset(problems)
    assert len(problems) == 120
    for task in Task:
        assert sum(p.task == task and p.split == Split.TRAIN for p in problems) == 32
        assert sum(p.task == task and p.split == Split.TEST for p in problems) == 4
    for problem in problems:
        assert solve(problem) == problem.answer


def test_prompt_does_not_read_private_answer_or_evidence(problems):
    original = problems[0]
    altered = original.model_copy(update={"answer": "DO-NOT-EXPOSE", "evidence": ("F999",)})
    assert build_prompt(original) == build_prompt(altered)


def test_relevance_ids_are_not_always_first_three(problems):
    noisy = [p for p in problems if p.condition != Condition.CLEAN]
    assert any("F4" in p.evidence for p in noisy)
    assert any("F1" not in p.evidence for p in noisy)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("answer", "incorrect", "Reference answer mismatch"),
        ("split", Split.TEST, "must share a split"),
        ("evidence", ("F1",), "Evidence annotations do not match"),
    ],
)
def test_validator_rejects_corrupted_ground_truth(problems, field, value, message):
    corrupted = (problems[0].model_copy(update={field: value}), *problems[1:])
    with pytest.raises(DatasetError, match=message):
        validate_dataset(corrupted)


@pytest.mark.parametrize("count", [0, 1, 9, -1])
def test_too_small_datasets_cannot_silently_omit_test_split(count):
    with pytest.raises(ValidationError, match="greater than or equal to 10"):
        GenerationConfig(problems_per_task=count)


def test_missing_predictions_count_as_errors_and_undefined_conditional(problems):
    report = evaluate(problems, make_run(()))
    assert all(group.accuracy == 0 and group.answered == 0 for group in report.groups)
    assert all(pair.conditional_failure_rate is None for pair in report.paired)


def test_paired_failure_metrics_and_evidence_scores(problems):
    predictions = tuple(
        Prediction(
            id=p.id,
            answer=p.answer if p.condition == Condition.CLEAN else "wrong",
            evidence=p.evidence if p.condition == Condition.CLEAN else (),
        )
        for p in problems
    )
    report = evaluate(problems, make_run(predictions))
    for pair in report.paired:
        assert pair.penalty_percentage_points == 100
        assert pair.correct_to_wrong == 10
        assert pair.wrong_to_correct == 0
        assert pair.conditional_failure_rate == 1
    assert all(
        group.evidence_exact == (1 if group.condition == Condition.CLEAN else 0)
        for group in report.groups
    )


def test_unknown_and_duplicate_prediction_ids_are_rejected(problems):
    prediction = Prediction(id="not-a-problem", answer="0")
    with pytest.raises(DatasetError, match="outside the evaluation dataset"):
        evaluate(problems, make_run((prediction,)))
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        make_run((prediction, prediction))


def test_artifacts_are_idempotent_and_refuse_overwrite(tmp_path: Path):
    config = GenerationConfig(problems_per_task=10)
    first = prepare_dataset(config, tmp_path)
    dataset = first / "dataset.json"
    timestamp = dataset.stat().st_mtime_ns
    assert prepare_dataset(config, tmp_path) == first
    assert dataset.stat().st_mtime_ns == timestamp
    assert len(load_dataset(dataset)) == 120
    with pytest.raises(DatasetError, match="Refusing to overwrite"):
        publish_text(dataset, "corrupt")


def test_missing_and_malformed_dataset_errors_are_visible(tmp_path: Path):
    path = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        load_dataset(path)
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValidationError, match="Invalid JSON"):
        load_dataset(path)


def test_cli_prepares_validates_and_scores_only_fixture_predictions(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    output = tmp_path / "processed"
    assert main(["prepare", "--problems-per-task", "10", "--output", str(output)]) == 0
    dataset = next(output.glob("*/*/dataset.json"))
    assert main(["validate", str(dataset)]) == 0
    test_problems = tuple(p for p in load_dataset(dataset) if p.split == Split.TEST)
    run = make_run(
        tuple(Prediction(id=p.id, answer=p.answer, evidence=p.evidence) for p in test_problems)
    )
    predictions = tmp_path / "fixture-predictions.json"
    predictions.write_text(run.model_dump_json(), encoding="utf-8")
    reports = tmp_path / "reports"
    assert main(["score", str(dataset), str(predictions), "--output", str(reports)]) == 0
    assert len(list(reports.glob("*/*/report.json"))) == 1
    assert len(list(reports.glob("*/*/paired-accuracy.png"))) == 1
