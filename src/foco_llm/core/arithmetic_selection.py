"""Select arithmetic facts and calculate solely from model-selected inputs."""

import json
import re

from foco_llm.models.experiment import Fact, Problem, Prompt

TARGET_PATTERN = re.compile(r"remain in (\w+)\?")
QUANTITY_PATTERN = re.compile(r"\b\d+\b")
FACT_ID_PATTERN = re.compile(r"F[1-9][0-9]*")


def selection_prompt(problem: Problem) -> Prompt:
    """Ask for fact IDs using only the user-visible question and sentences.

    Args:
        problem: Evaluation item; private labels are never included in the prompt.

    Returns:
        Prompt requesting a JSON array of selected fact IDs.

    Raises:
        ValueError: The question does not identify an arithmetic target.
    """
    match = TARGET_PATTERN.search(problem.question)
    if match is None:
        raise ValueError("Question does not identify a target container")
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in problem.facts)
    instruction = (
        f"The question asks about {match[1]}. Select the initial count and every "
        "update about this same container. Ignore facts about other containers. "
        "Return only a JSON array of the selected fact IDs, in their original order."
    )
    return Prompt(id=problem.id, text=f"{instruction}\n\n{facts}\n\nQuestion: {problem.question}")


def selection_prompt_v2(problem: Problem) -> Prompt:
    """Show a format example before requesting IDs from the visible facts."""
    match = TARGET_PATTERN.search(problem.question)
    if match is None:
        raise ValueError("Question does not identify a target container")
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in problem.facts)
    instruction = (
        "Example: [F1] The initial marble count of red is 5. "
        "[F2] The initial marble count of blue is 8. "
        "[F3] For red, 2 marbles are added to it. "
        "Question: How many marbles remain in red? "
        'Response: {"evidence":["F1","F3"]}.\n\n'
        f"Now select the facts about {match[1]}: its initial count and every "
        "update about this same container. Ignore other containers. "
        'Return only a JSON object with one key, "evidence", containing fact '
        'ID strings such as "F1". Do not return quantities or the answer. '
        "Keep the facts in their original order."
    )
    return Prompt(id=problem.id, text=f"{instruction}\n\n{facts}\n\nQuestion: {problem.question}")


def parse_selected_ids(raw: str, facts: tuple[Fact, ...]) -> tuple[str, ...]:
    """Reject malformed, duplicate, or nonexistent model-selected fact IDs.

    Args:
        raw: Unmodified model continuation.
        facts: Facts visible in its prompt.

    Returns:
        Selected IDs, preserving the model's list order.

    Raises:
        ValueError: Selection is not a complete valid JSON ID array.
    """
    try:
        decoded = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise ValueError("Selection is not a JSON array") from error
    allowed = {fact.id for fact in facts}
    if (
        not isinstance(decoded, list)
        or not decoded
        or any(
            not isinstance(value, str) or FACT_ID_PATTERN.fullmatch(value) is None
            for value in decoded
        )
        or len(decoded) != len(set(decoded))
        or not set(decoded).issubset(allowed)
    ):
        raise ValueError("Selection contains missing, duplicate, or unknown fact IDs")
    return tuple(decoded)


def parse_selected_ids_v2(raw: str, facts: tuple[Fact, ...]) -> tuple[str, ...]:
    """Accept only a complete evidence object with known unique IDs."""
    try:
        decoded = json.loads(raw.strip())
    except json.JSONDecodeError as error:
        raise ValueError("Selection is not a JSON object") from error
    if not isinstance(decoded, dict) or set(decoded) != {"evidence"}:
        raise ValueError("Selection requires only an evidence field")
    return parse_selected_ids(json.dumps(decoded["evidence"]), facts)


def selected_calculation(facts: tuple[Fact, ...], selected_ids: tuple[str, ...]) -> tuple[str, str]:
    """Compute from selected sentences alone, without reference labels or answer.

    Args:
        facts: User-visible arithmetic sentences.
        selected_ids: IDs produced by the selector.

    Returns:
        Expression and integer result as strings.

    Raises:
        ValueError: Selected facts cannot form one unambiguous calculation.
    """
    selected = [fact for fact in facts if fact.id in selected_ids]
    initial = [fact for fact in selected if "initial" in fact.text.lower()]
    if len(initial) != 1:
        raise ValueError("Selected facts require exactly one initial count")
    initial_number = QUANTITY_PATTERN.search(initial[0].text)
    if initial_number is None:
        raise ValueError("Selected initial fact has no quantity")
    total = int(initial_number[0])
    expression = str(total)
    for fact in selected:
        if fact == initial[0]:
            continue
        quantity = QUANTITY_PATTERN.search(fact.text)
        if quantity is None:
            raise ValueError("Selected update has no quantity")
        if "removed from" in fact.text:
            sign = "-"
            total -= int(quantity[0])
        elif "added to" in fact.text:
            sign = "+"
            total += int(quantity[0])
        else:
            raise ValueError("Selected fact has no arithmetic operation")
        expression += f" {sign} {quantity[0]}"
    return expression, str(total)
