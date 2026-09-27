"""Validated experimental records and generation settings."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Task(StrEnum):
    """Supported synthetic reasoning families."""

    ARITHMETIC = "arithmetic"
    DEDUCTION = "deduction"
    TRACKING = "tracking"


class Split(StrEnum):
    """Disjoint problem partitions."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Condition(StrEnum):
    """Answer-preserving context conditions."""

    CLEAN = "clean"
    UNRELATED = "unrelated"
    NUMERIC = "numeric"
    SIMILAR = "similar"


class Record(BaseModel):
    """Immutable records that reject unexpected input fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GenerationConfig(Record):
    """Validated reproducible dataset dimensions."""

    seed: int = Field(default=42, ge=0)
    problems_per_task: int = Field(default=100, ge=10, le=100_000)


class Fact(Record):
    """One identified sentence presented to a model."""

    id: str = Field(pattern=r"^F[1-9][0-9]*$")
    text: str = Field(min_length=1)


class Problem(Record):
    """Private evaluation record, including reference answers and evidence."""

    id: str = Field(min_length=1)
    base_id: str = Field(min_length=1)
    task: Task
    split: Split
    condition: Condition
    facts: tuple[Fact, ...] = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Reject duplicate fact identifiers and missing evidence targets."""
        fact_ids = [fact.id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("Fact identifiers must be unique")
        if len(self.evidence) != len(set(self.evidence)):
            raise ValueError("Evidence identifiers must be unique")
        if not set(self.evidence).issubset(fact_ids):
            raise ValueError("Evidence must refer to existing facts")
        return self


class Prompt(Record):
    """Public inference input with no answers or evidence labels."""

    id: str
    text: str


class Prediction(Record):
    """One model response; empty answers remain scoreable failures."""

    id: str = Field(min_length=1)
    answer: str
    evidence: tuple[str, ...] = ()


class ModelRun(Record):
    """Provenance required before scoring model-generated predictions."""

    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    seed: int = Field(ge=0)
    decoding: dict[str, str | int | float | bool]
    predictions: tuple[Prediction, ...]

    @model_validator(mode="after")
    def validate_ids(self) -> Self:
        """Reject duplicate response identifiers instead of silently overwriting."""
        identifiers = [prediction.id for prediction in self.predictions]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Prediction identifiers must be unique")
        return self
