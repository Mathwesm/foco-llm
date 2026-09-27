"""Paired accuracy and evidence metrics with explicit missing responses."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from foco_llm.core.validation import DatasetError
from foco_llm.models.experiment import Condition, ModelRun, Prediction, Problem, Task


class GroupMetrics(BaseModel):
    """Metrics for one task and context condition."""

    task: Task
    condition: Condition
    count: int
    answered: int
    accuracy: float
    evidence_precision: float
    evidence_recall: float
    evidence_exact: float


class PairedMetrics(BaseModel):
    """Transitions between matched clean and distracted responses."""

    task: Task
    condition: Condition
    pairs: int
    clean_accuracy: float
    distracted_accuracy: float
    penalty_percentage_points: float
    correct_to_wrong: int
    wrong_to_correct: int
    conditional_failure_rate: float | None


class EvaluationReport(BaseModel):
    """Versioned report containing model provenance and evaluation results."""

    schema_version: str = "1"
    model_id: str
    model_revision: str
    prompt_version: str
    seed: int
    decoding: dict[str, str | int | float | bool]
    groups: list[GroupMetrics]
    paired: list[PairedMetrics]


def _correct(problem: Problem, prediction: Prediction | None) -> bool:
    return (
        prediction is not None and prediction.answer.strip().casefold() == problem.answer.casefold()
    )


def _group(problems: list[Problem], responses: dict[str, Prediction]) -> GroupMetrics:
    correct = answered = exact = 0
    precision = recall = 0.0
    for problem in problems:
        prediction = responses.get(problem.id)
        correct += _correct(problem, prediction)
        answered += prediction is not None
        selected = set(prediction.evidence) if prediction else set()
        expected = set(problem.evidence)
        overlap = len(selected & expected)
        precision += overlap / len(selected) if selected else 0.0
        recall += overlap / len(expected)
        exact += selected == expected
    count = len(problems)
    return GroupMetrics(
        task=problems[0].task,
        condition=problems[0].condition,
        count=count,
        answered=answered,
        accuracy=correct / count,
        evidence_precision=precision / count,
        evidence_recall=recall / count,
        evidence_exact=exact / count,
    )


def _paired(problems: tuple[Problem, ...], responses: dict[str, Prediction]) -> list[PairedMetrics]:
    clean = {p.base_id: p for p in problems if p.condition == Condition.CLEAN}
    grouped: dict[tuple[Task, Condition], list[tuple[bool, bool]]] = defaultdict(list)
    for problem in problems:
        if problem.condition == Condition.CLEAN:
            continue
        if problem.base_id not in clean:
            raise DatasetError("Paired evaluation requires each clean counterpart")
        original = clean[problem.base_id]
        grouped[problem.task, problem.condition].append(
            (
                _correct(original, responses.get(original.id)),
                _correct(problem, responses.get(problem.id)),
            )
        )
    output = []
    for (task, condition), pairs in grouped.items():
        clean_correct = sum(first for first, _ in pairs)
        distracted_correct = sum(second for _, second in pairs)
        lost = sum(first and not second for first, second in pairs)
        gained = sum(not first and second for first, second in pairs)
        output.append(
            PairedMetrics(
                task=task,
                condition=condition,
                pairs=len(pairs),
                clean_accuracy=clean_correct / len(pairs),
                distracted_accuracy=distracted_correct / len(pairs),
                penalty_percentage_points=100 * (clean_correct - distracted_correct) / len(pairs),
                correct_to_wrong=lost,
                wrong_to_correct=gained,
                conditional_failure_rate=lost / clean_correct if clean_correct else None,
            )
        )
    return output


def evaluate(problems: tuple[Problem, ...], run: ModelRun) -> EvaluationReport:
    """Score all requested examples, counting omitted predictions as failures.

    Args:
        problems: A nonempty complete set of clean/distracted groups to evaluate.
        run: Predictions and the model/decoding provenance.

    Returns:
        Per-task metrics and paired transitions. No invented model results.

    Raises:
        DatasetError: Inputs contain unknown IDs, duplicates, or missing pairs.
    """
    identifiers = {p.id for p in problems}
    if not problems or len(identifiers) != len(problems):
        raise DatasetError("Evaluation requires nonempty unique problem identifiers")
    responses = {prediction.id: prediction for prediction in run.predictions}
    if set(responses) - identifiers:
        raise DatasetError("Predictions contain identifiers outside the evaluation dataset")
    grouped: dict[tuple[Task, Condition], list[Problem]] = defaultdict(list)
    for problem in problems:
        grouped[problem.task, problem.condition].append(problem)
    return EvaluationReport(
        model_id=run.model_id,
        model_revision=run.model_revision,
        prompt_version=run.prompt_version,
        seed=run.seed,
        decoding=run.decoding,
        groups=[_group(group, responses) for group in grouped.values()],
        paired=_paired(problems, responses),
    )
