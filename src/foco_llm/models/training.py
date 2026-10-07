"""Bounded configuration for a functional local LoRA training test."""

from pydantic import Field

from foco_llm.models.experiment import Record


class SmokeConfig(Record):
    """Fixed small experiment; not a full training sweep."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    seed: int = Field(default=42, ge=0)
    steps: int = Field(default=6, ge=2, le=12)
    max_tokens: int = Field(default=512, ge=64, le=512)
    learning_rate: float = Field(default=0.0001, gt=0, le=0.01)
    rank: int = Field(default=4, ge=1, le=8)


class TrainingExample(Record):
    """Validated tokenized record with prompt-only loss masking."""

    id: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int


class StepResult(Record):
    """Finite training loss and duration for one optimizer update."""

    step: int
    example_id: str
    loss: float = Field(allow_inf_nan=False)
    elapsed_seconds: float = Field(ge=0)
    supervised_tokens: int = Field(gt=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
