"""Atomic local checkpoints with integrity checks before tensor deserialization."""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

from foco_llm.models.experiment import Record
from foco_llm.models.training import StepResult
from foco_llm.services.artifacts import publish_text
from foco_llm.services.training_backend import TrainingBackend

STATE_FILES = ("adapter_model.safetensors", "adapter_config.json", "state.pt")


class CheckpointState(Record):
    """Completed update history bound to adapter and optimizer file digests."""

    history: tuple[StepResult, ...]
    digests: dict[str, str]


def verify_checkpoint(path: Path) -> CheckpointState:
    """Reject incomplete, modified, or out-of-order checkpoints before loading."""
    state = CheckpointState.model_validate_json(
        (path / "complete.json").read_text(encoding="utf-8")
    )
    if not state.history or [r.step for r in state.history] != list(
        range(1, len(state.history) + 1)
    ):
        raise ValueError("Checkpoint history must contain consecutive updates")
    if set(state.digests) != set(STATE_FILES):
        raise ValueError("Checkpoint integrity manifest has unexpected files")
    for name, digest in state.digests.items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    return state


def save_checkpoint(backend: TrainingBackend, destination: Path, history: list[StepResult]) -> Path:
    """Publish a complete checkpoint atomically without overwriting an earlier step."""
    path = destination / f"step-{len(history):04d}"
    if path.exists():
        raise ValueError("Refusing to overwrite a published training checkpoint")
    with TemporaryDirectory(prefix="partial-", dir=destination) as temporary:
        staging = Path(temporary)
        backend.save(staging)
        state = CheckpointState(
            history=tuple(history),
            digests={
                name: hashlib.sha256((staging / name).read_bytes()).hexdigest()
                for name in STATE_FILES
            },
        )
        publish_text(staging / "complete.json", state.model_dump_json(indent=2))
        staging.rename(path)
    return path


def latest_checkpoint(destination: Path) -> Path | None:
    """Locate the highest completed step and reject gaps or mismatched folder numbers."""
    paths = sorted(destination.glob("step-*"))
    for index, path in enumerate(paths, start=1):
        if path.name != f"step-{index:04d}" or len(verify_checkpoint(path).history) != index:
            raise ValueError("Training checkpoint sequence is inconsistent")
    return paths[-1] if paths else None
