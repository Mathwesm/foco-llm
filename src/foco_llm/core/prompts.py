"""Separate model-visible inputs from privileged evaluation targets."""

from foco_llm.models.experiment import Problem, Prompt

PROMPT_VERSION = "evidence-json-v1"


def build_prompt(problem: Problem) -> Prompt:
    """Render facts and the question, never answers or relevance annotations.

    Args:
        problem: Private dataset record.

    Returns:
        Safe public prompt with explicit structured-output instructions.
    """
    facts = "\n".join(f"[{fact.id}] {fact.text}" for fact in problem.facts)
    instruction = (
        'Return only a JSON object with "answer" (a string) and "evidence" '
        "(a list of fact identifiers needed to answer the question). "
        "For counts, return an integer without units. For codes, return the code only."
    )
    return Prompt(id=problem.id, text=f"{instruction}\n\n{facts}\n\nQuestion: {problem.question}")
