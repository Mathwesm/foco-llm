"""Predefined prompt contrasts without answer repair or reference leakage."""

import re
from typing import Literal

from foco_llm.core.content_evaluation import ContentError, decode_content
from foco_llm.models.experiment import Problem, Prompt

FormatArm = Literal["explicit_json", "plain_answer"]


def diagnostic_prompt(problem: Problem, arm: FormatArm) -> Prompt:
    """Render fixed format instructions with unchanged facts and question.

    Args:
        problem: Private record whose annotations must not enter the prompt.
        arm: Prespecified response contract.

    Returns:
        Public prompt containing neither reference answers nor evidence labels.
    """
    if arm == "explicit_json":
        instruction = (
            'Return only one JSON object with "answer" and "evidence". '
            'The "answer" value must be a JSON string in double quotes: '
            "the numerical count without units, or the property/box code. "
            'The "evidence" value must be a list of fact identifiers needed to answer. '
            "Do not use Markdown fences or additional text."
        )
    elif arm == "plain_answer":
        instruction = (
            "Return only the numerical count without units, or the property/box code. "
            "Do not return fact identifiers, evidence, JSON, quotes, or explanation."
        )
    else:
        raise ValueError("Unknown diagnostic format arm")
    facts = "\n".join(f"[{f.id}] {f.text}" for f in problem.facts)
    return Prompt(id=problem.id, text=f"{instruction}\n\n{facts}\n\nQuestion: {problem.question}")


def format_answer(text: str, arm: FormatArm) -> str | None:
    """Read a complete contracted response; never extract or repair an answer."""
    if arm == "explicit_json":
        try:
            return decode_content(text).answer
        except ContentError:
            return None
    if arm != "plain_answer":
        raise ValueError("Unknown diagnostic format arm")
    body = text.strip()
    return body if re.fullmatch(r"(?:-?\d+|[PB]\d+)", body) else None
