"""Evidence-only LoRA supervision for arithmetic fact selection."""

import json

from foco_llm.core.arithmetic_selection import selection_prompt_v2
from foco_llm.core.training_data import mask_completion
from foco_llm.models.experiment import Problem
from foco_llm.models.training import TrainingExample
from foco_llm.services.training_backend import TrainingBackend


def selection_target(problem: Problem) -> str:
    """Serialize training evidence without including the arithmetic answer."""
    return json.dumps({"evidence": list(problem.evidence)}, separators=(",", ":"))


class SelectionTrainingBackend(TrainingBackend):
    """Train the same LoRA backbone on evidence IDs rather than final answers."""

    def encode(self, problem: Problem) -> TrainingExample:
        """Mask the selector prompt and supervise only its JSON completion."""
        messages = [{"role": "user", "content": selection_prompt_v2(problem).text}]
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        full = self.tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": selection_target(problem)}],
            tokenize=True,
            add_generation_prompt=False,
        )
        return mask_completion(problem.id, tuple(prefix), tuple(full), self.config.max_tokens)
