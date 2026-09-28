import hashlib
import json

import pytest

from foco_llm.core.benchmark_v2 import generate_v2
from foco_llm.core.training_data import mask_completion, select_smoke_examples
from foco_llm.models.experiment import GenerationConfig, Split, Task
from foco_llm.models.training import StepResult
from foco_llm.services.training_checkpoints import STATE_FILES, verify_checkpoint


def test_mask_preserves_every_completion_token_and_masks_entire_prompt():
    example = mask_completion("sample", (1, 2, 3), (1, 2, 3, 4, 5, 6), 6)
    assert example.labels == (-100, -100, -100, 4, 5, 6)
    assert example.input_ids == (1, 2, 3, 4, 5, 6)


@pytest.mark.parametrize(
    "prefix,full,limit,message",
    [
        ((1, 2), (1, 9, 3), 9, "preserve"),
        ((1, 2), (1, 2), 9, "preserve"),
        ((), (1,), 9, "preserve"),
        ((1,), (1, 2, 3), 2, "truncation"),
    ],
)
def test_mask_refuses_boundary_mismatch_empty_target_and_truncation(prefix, full, limit, message):
    with pytest.raises(ValueError, match=message):
        mask_completion("sample", prefix, full, limit)


def test_selection_excludes_evaluation_and_covers_each_task():
    problems = generate_v2(GenerationConfig(problems_per_task=10))
    selected = select_smoke_examples(problems)
    assert len(selected) == 6
    assert {p.task for p in selected} == set(Task)
    assert all(p.split == Split.TRAIN for p in selected)
    with pytest.raises(ValueError, match="No training examples"):
        select_smoke_examples(tuple(p for p in problems if p.split != Split.TRAIN))


def test_checkpoint_detects_modified_weights_before_loading(tmp_path):
    for name in STATE_FILES:
        (tmp_path / name).write_bytes(b"local-test-payload")
    record = StepResult(
        step=1,
        example_id="sample",
        loss=1.0,
        elapsed_seconds=0.1,
        supervised_tokens=2,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
    )
    metadata = {
        "history": [record.model_dump()],
        "digests": {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in STATE_FILES
        },
    }
    (tmp_path / "complete.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert verify_checkpoint(tmp_path).history[0].step == 1
    (tmp_path / "adapter_model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="integrity failure"):
        verify_checkpoint(tmp_path)
