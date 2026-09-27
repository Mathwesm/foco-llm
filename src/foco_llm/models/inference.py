"""Contracts for resumable local inference and raw responses."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from foco_llm.models.experiment import Record, Split


class InferenceConfig(Record):
    """Pinned model and bounded deterministic generation settings."""

    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    split: Split = Split.VALIDATION
    seed: int = Field(default=42, ge=0)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    max_input_tokens: int = Field(default=1024, ge=1, le=4096)
    max_seconds: float = Field(default=60, gt=0, le=600)
    device: Literal["cuda", "cpu"] = "cuda"


class Generation(Record):
    """Unmodified continuation and measured per-example resource use."""

    text: str
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    stop_reason: Literal["eos", "token_limit", "time_limit"]


class Response(Record):
    """Model-visible schema, with no privileged answer repair."""

    answer: str
    evidence: tuple[str, ...]


class Checkpoint(Record):
    """One immutable completed attempt, including invalid model output."""

    id: str
    prompt_sha256: str
    finished_at: datetime
    generation: Generation
