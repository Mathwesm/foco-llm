"""Versioned post-hoc content scoring, independent of strict format adherence."""

from __future__ import annotations

import hashlib
import json
import re

from pydantic import BaseModel

from foco_llm.core.evaluation import EvaluationReport, evaluate
from foco_llm.core.prompts import build_prompt
from foco_llm.models.experiment import ModelRun, Prediction, Problem
from foco_llm.models.inference import Checkpoint, Response
from foco_llm.services.inference import parse_response

PROTOCOL_VERSION = "content-envelope-v2"


class ContentError(ValueError):
    """An output violates the explicitly bounded content contract."""


class Decision(BaseModel):
    """Acceptance decisions per example; missing or rejected output remains an error."""

    id: str
    strict_accepted: bool
    content_accepted: bool
    answer_readable: bool = False
    content_error: str | None = None


class ContentEvaluation(BaseModel):
    """Separate reports over the same denominator and unchanged raw continuations."""

    protocol: str = PROTOCOL_VERSION
    analysis_type: str = "post_hoc_validation_diagnostic"
    examples: int
    strict_accepted: int
    content_accepted: int
    answer_readable: int
    decisions: list[Decision]
    strict: EvaluationReport
    content: EvaluationReport
    answer_only: EvaluationReport


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ContentError("duplicate_json_key")
        result[key] = value
    return result


def decode_content(text: str) -> Response:
    """Decode a complete schema-valid object without validating evidence references."""
    body = text.strip()
    if body.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\r?\n(.*?)\r?\n```", body, flags=re.DOTALL)
        if match is None or "```" in match[1]:
            raise ContentError("invalid_markdown_envelope")
        body = match[1]
    try:
        decoded = json.loads(body, object_pairs_hook=_unique_object)
        return Response.model_validate(decoded)
    except ContentError:
        raise
    except ValueError as error:
        raise ContentError("invalid_json_or_schema") from error


def parse_content(problem: Problem, text: str) -> Prediction:
    """Accept a complete JSON object or one fence, with valid evidence identifiers.

    Args:
        problem: Provides the response ID and allowed fact identifiers only.
        text: Unchanged raw model output.

    Returns:
        Prediction without numerical, semantic, or evidence repair.

    Raises:
        ContentError: Envelope, JSON schema, or evidence identifiers are invalid.
    """
    response = decode_content(text)
    if len(set(response.evidence)) != len(response.evidence):
        raise ContentError("duplicate_evidence")
    if not set(response.evidence).issubset({fact.id for fact in problem.facts}):
        raise ContentError("unknown_evidence")
    return Prediction(id=problem.id, answer=response.answer, evidence=response.evidence)


def _verify_records(problems: tuple[Problem, ...], records: tuple[Checkpoint, ...]) -> None:
    expected = {p.id: p for p in problems}
    identifiers = [record.id for record in records]
    if len(identifiers) != len(set(identifiers)) or set(identifiers) != set(expected):
        raise ValueError("Expected exactly one checkpoint for every selected example")
    for record in records:
        prompt = build_prompt(expected[record.id])
        digest = hashlib.sha256(prompt.text.encode("utf-8")).hexdigest()
        if digest != record.prompt_sha256:
            raise ValueError("Checkpoint prompt hash does not match selected data")


def compare_protocols(
    problems: tuple[Problem, ...], records: tuple[Checkpoint, ...], provenance: ModelRun
) -> ContentEvaluation:
    """Score raw outputs twice, preserving the original strict parser and denominator.

    Args:
        problems: Paired examples selected before inspecting predictions.
        records: Exactly one prompt-matched checkpoint per selected example.
        provenance: Original model settings; predictions are reconstructed from raw output.

    Returns:
        Strict and content scores plus a complete rejection audit.
    """
    _verify_records(problems, records)
    by_id = {p.id: p for p in problems}
    strict: list[Prediction] = []
    content: list[Prediction] = []
    answers: list[Prediction] = []
    decisions = []
    for record in records:
        decision = Decision(id=record.id, strict_accepted=False, content_accepted=False)
        try:
            strict.append(parse_response(by_id[record.id], record.generation.text))
            decision.strict_accepted = True
        except ValueError:
            decision.strict_accepted = False
        try:
            response = decode_content(record.generation.text)
            answers.append(Prediction(id=record.id, answer=response.answer, evidence=()))
            decision.answer_readable = True
            content.append(parse_content(by_id[record.id], record.generation.text))
            decision.content_accepted = True
        except ContentError as error:
            decision.content_error = str(error)
        decisions.append(decision)
    return ContentEvaluation(
        examples=len(problems),
        strict_accepted=len(strict),
        content_accepted=len(content),
        answer_readable=len(answers),
        decisions=decisions,
        strict=evaluate(problems, provenance.model_copy(update={"predictions": tuple(strict)})),
        content=evaluate(problems, provenance.model_copy(update={"predictions": tuple(content)})),
        answer_only=evaluate(
            problems, provenance.model_copy(update={"predictions": tuple(answers)})
        ),
    )
