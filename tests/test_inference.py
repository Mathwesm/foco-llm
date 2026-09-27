import json
from pathlib import Path

import pytest

from foco_llm.core.generation import generate_problems
from foco_llm.models.experiment import GenerationConfig, Prompt
from foco_llm.models.inference import Generation, InferenceConfig
from foco_llm.services.artifacts import prepare_dataset
from foco_llm.services.inference import parse_response, run_inference


class FakeBackend:
    """Count generation calls and simulate an interruption without model downloads."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.calls = 0
        self.fail_after = fail_after

    def metadata(self) -> dict[str, str]:
        """Identify the fake runtime."""
        return {"engine": "test-only"}

    def generate(self, prompt: Prompt) -> Generation:
        """Return invalid output or interrupt at the requested checkpoint."""
        assert prompt.text
        if self.calls == self.fail_after:
            raise RuntimeError("simulated interruption")
        self.calls += 1
        return Generation(
            text="not JSON",
            input_tokens=20,
            output_tokens=4,
            elapsed_seconds=1.0,
            peak_allocated_bytes=0,
            peak_reserved_bytes=0,
            stop_reason="eos",
        )


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": 12, "evidence": ["F1"]}',
        '{"answer": "12", "evidence": ["F99"]}',
        '{"answer": "12", "evidence": ["F1", "F1"]}',
        'Here is the answer: {"answer": "12", "evidence": []}',
        '{"answer": "12", "evidence": [], "extra": true}',
    ],
)
def test_invalid_outputs_are_not_repaired(text: str) -> None:
    problem = generate_problems(GenerationConfig(problems_per_task=10))[0]
    with pytest.raises(ValueError):
        parse_response(problem, text)


def test_parser_preserves_wrong_answer_for_scoring() -> None:
    problem = generate_problems(GenerationConfig(problems_per_task=10))[0]
    result = parse_response(problem, '{"answer": "wrong", "evidence": []}')
    assert result.answer == "wrong"
    assert result.id == problem.id


def test_interrupted_run_resumes_and_completed_run_is_idempotent(tmp_path: Path) -> None:
    dataset = prepare_dataset(GenerationConfig(problems_per_task=10), tmp_path / "dataset")
    config = InferenceConfig(revision="a" * 40)
    output = tmp_path / "inference"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_inference(dataset / "dataset.json", output, config, FakeBackend(fail_after=2))
    backend = FakeBackend()
    directory = run_inference(dataset / "dataset.json", output, config, backend)
    assert backend.calls == 10
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["attempted"] == 12
    assert summary["valid"] == 0
    assert len(summary["rejected"]) == 12
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert all(group["accuracy"] == 0 for group in report["groups"])
    previous = (directory / "summary.json").stat().st_mtime_ns
    run_inference(dataset / "dataset.json", output, config, backend)
    assert backend.calls == 10
    assert (directory / "summary.json").stat().st_mtime_ns == previous


def test_changed_generation_settings_cannot_reuse_checkpoints(tmp_path: Path) -> None:
    dataset = prepare_dataset(GenerationConfig(problems_per_task=10), tmp_path / "dataset")
    backend = FakeBackend()
    config = InferenceConfig(revision="a" * 40)
    first = run_inference(dataset / "dataset.json", tmp_path / "runs", config, backend)
    second = run_inference(
        dataset / "dataset.json",
        tmp_path / "runs",
        config.model_copy(update={"max_new_tokens": 64}),
        backend,
    )
    assert first != second
    assert backend.calls == 24


def test_mismatched_checkpoint_fails_visibly(tmp_path: Path) -> None:
    dataset = prepare_dataset(GenerationConfig(problems_per_task=10), tmp_path / "dataset")
    config = InferenceConfig(revision="a" * 40)
    output = tmp_path / "runs"
    directory = run_inference(dataset / "dataset.json", output, config, FakeBackend())
    checkpoint = next((directory / "responses").glob("*.json"))
    record = json.loads(checkpoint.read_text(encoding="utf-8"))
    record["prompt_sha256"] = "corrupted"
    checkpoint.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint does not match"):
        run_inference(dataset / "dataset.json", output, config, FakeBackend())
