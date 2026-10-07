import hashlib
import json
from datetime import UTC, datetime

import pytest

from foco_llm.core.content_evaluation import ContentError, compare_protocols, parse_content
from foco_llm.core.generation import generate_problems
from foco_llm.core.prompts import build_prompt
from foco_llm.models.experiment import GenerationConfig, ModelRun, Split
from foco_llm.models.inference import Checkpoint, Generation
from foco_llm.services.inference import parse_response


@pytest.mark.parametrize(
    "wrapper", ["{}", "```json\n{}\n```", "```\n{}\n```", " ```json\r\n{}\r\n``` "]
)
def test_envelopes_do_not_change_answer_or_evidence(wrapper):
    problem = generate_problems(GenerationConfig(problems_per_task=10))[0]
    payload = json.dumps({"answer": "wrong", "evidence": ["F1"]})
    prediction = parse_content(problem, wrapper.format(payload))
    assert prediction.answer == "wrong"
    assert prediction.evidence == ("F1",)


@pytest.mark.parametrize(
    "text",
    [
        'Explanation\n```json\n{"answer":"1","evidence":[]}\n```',
        '```json\n{"answer":"1","evidence":[]}\n```\nExplanation',
        '```json\n{"answer":"1","evidence":[]}',
        '```python\n{"answer":"1","evidence":[]}\n```',
        "```json\n{}\n```\n```json\n{}\n```",
        '{"answer":"1","answer":"2","evidence":[]}',
        '{"answer":1,"evidence":[]}',
        '{"answer":"1","evidence":["F1","F1"]}',
        '{"answer":"1","evidence":["F999"]}',
        '{"answer":"1","evidence":[],"other":1}',
        "[]",
        "",
    ],
)
def test_ambiguous_or_malformed_outputs_are_rejected(text):
    problem = generate_problems(GenerationConfig(problems_per_task=10))[0]
    with pytest.raises(ContentError):
        parse_content(problem, text)


def test_strict_parser_still_rejects_markdown():
    problem = generate_problems(GenerationConfig(problems_per_task=10))[0]
    text = '```json\n{"answer":"1","evidence":[]}\n```'
    with pytest.raises(ValueError):
        parse_response(problem, text)


def test_plain_json_remains_accepted_by_both_protocols():
    problems, records, provenance = _fixture()
    payload = json.dumps({"answer": problems[0].answer, "evidence": problems[0].evidence})
    altered = records[0].model_copy(
        update={"generation": records[0].generation.model_copy(update={"text": payload})}
    )
    result = compare_protocols(problems, (altered, *records[1:]), provenance)
    assert result.strict_accepted == 1
    assert result.content_accepted == 12
    assert sum(g.count * g.accuracy for g in result.strict.groups) == 1


def _fixture():
    problems = tuple(
        p
        for p in generate_problems(GenerationConfig(problems_per_task=10))
        if p.split == Split.VALIDATION
    )
    records = tuple(
        Checkpoint(
            id=p.id,
            prompt_sha256=hashlib.sha256(build_prompt(p).text.encode()).hexdigest(),
            finished_at=datetime.now(UTC),
            generation=Generation(
                text="```json\n"
                + json.dumps({"answer": p.answer, "evidence": p.evidence})
                + "\n```",
                input_tokens=1,
                output_tokens=1,
                elapsed_seconds=1,
                peak_allocated_bytes=0,
                peak_reserved_bytes=0,
                stop_reason="eos",
            ),
        )
        for p in problems
    )
    provenance = ModelRun(
        model_id="fixture",
        model_revision="fixture",
        prompt_version="evidence-json-v1",
        seed=42,
        decoding={},
        predictions=(),
    )
    return problems, records, provenance


def test_reports_keep_denominator_and_strict_scores():
    problems, records, provenance = _fixture()
    # One malformed output must remain an error instead of shrinking the denominator.
    bad = records[0].model_copy(
        update={"generation": records[0].generation.model_copy(update={"text": "invalid"})}
    )
    result = compare_protocols(problems, (bad, *records[1:]), provenance)
    assert result.examples == 12
    assert result.strict_accepted == 0
    assert result.content_accepted == 11
    assert result.answer_readable == 11
    assert sum(g.count * g.accuracy for g in result.strict.groups) == 0
    assert sum(g.count * g.accuracy for g in result.content.groups) == 11
    assert sum(g.count for g in result.content.groups) == 12


def test_correct_answer_with_invalid_evidence_is_scored_independently():
    problems, records, provenance = _fixture()
    payload = json.dumps({"answer": problems[0].answer, "evidence": ["F999"]})
    altered = records[0].model_copy(
        update={"generation": records[0].generation.model_copy(update={"text": payload})}
    )
    result = compare_protocols(problems, (altered, *records[1:]), provenance)
    assert result.answer_readable == 12
    assert result.content_accepted == 11
    assert sum(g.count * g.accuracy for g in result.answer_only.groups) == 12
    assert sum(g.count * g.accuracy for g in result.content.groups) == 11
    assert result.decisions[0].content_error == "unknown_evidence"


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "hash"])
def test_mismatched_checkpoint_sets_are_rejected(corruption):
    problems, records, provenance = _fixture()
    if corruption == "missing":
        records = records[:-1]
    elif corruption == "duplicate":
        records = (*records[:-1], records[0])
    else:
        records = (records[0].model_copy(update={"prompt_sha256": "wrong"}), *records[1:])
    with pytest.raises(ValueError, match="checkpoint|Checkpoint"):
        compare_protocols(problems, records, provenance)
