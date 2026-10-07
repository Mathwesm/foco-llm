import pytest

from foco_llm.core.training_data import mask_completion
from foco_llm.services.answer_training import mask_answer


def test_answer_mask_excludes_prompt_evidence_and_boundary_spanning_tokens():
    example = mask_completion("sample", (1, 2), (1, 2, 3, 4, 5, 6), 6)
    masked = mask_answer(example, ((0, 2), (2, 4), (4, 6), (6, 8), (8, 11), (11, 15)), 4, 10)
    assert masked.labels == (-100, -100, 3, 4, -100, -100)
    assert masked.input_ids == example.input_ids


@pytest.mark.parametrize("offsets,start,end", [(((0, 1),), 0, 1), (((0, 1), (1, 2)), 2, 1)])
def test_invalid_alignment_is_rejected(offsets, start, end):
    example = mask_completion("sample", (1,), (1, 2), 2)
    with pytest.raises(ValueError, match="Invalid answer mask"):
        mask_answer(example, offsets, start, end)


def test_empty_answer_mask_cannot_silently_skip_training():
    example = mask_completion("sample", (1,), (1, 2), 2)
    with pytest.raises(ValueError, match="no supervised tokens"):
        mask_answer(example, ((0, 1), (1, 2)), 3, 4)
