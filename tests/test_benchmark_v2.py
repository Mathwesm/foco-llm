import pytest

from foco_llm.core.benchmark_v2 import generate_v2
from foco_llm.core.validation import DatasetError, solve, validate_dataset
from foco_llm.models.experiment import Condition, Fact, GenerationConfig, Problem, Split, Task


@pytest.fixture
def problems():
    return generate_v2(GenerationConfig(problems_per_task=10))


def test_v2_is_paired_reproducible_and_has_disjoint_depths(problems):
    validate_dataset(problems)
    assert problems == generate_v2(GenerationConfig(problems_per_task=10))
    assert problems != generate_v2(GenerationConfig(problems_per_task=10, seed=43))
    lengths = {split: {len(p.evidence) for p in problems if p.split == split} for split in Split}
    assert lengths == {Split.TRAIN: {3, 4}, Split.VALIDATION: {5}, Split.TEST: {6}}
    assert {p.split for p in problems if p.base_id == "v2-tracking-000008"} == {Split.VALIDATION}


@pytest.mark.parametrize(
    "task,facts,question,answer",
    [
        (
            Task.ARITHMETIC,
            [
                "7 marbles are removed from amber.",
                "amber initially contains 20 marbles.",
                "5 marbles are added to amber.",
                "violet initially contains 99 marbles.",
            ],
            "How many marbles remain in amber?",
            "18",
        ),
        (
            Task.DEDUCTION,
            [
                "Property P2 implies property P3.",
                "violet has property P8.",
                "amber has property P1.",
                "Property P1 implies property P2.",
            ],
            "What terminal property follows for amber? Return its code.",
            "P3",
        ),
        (
            Task.TRACKING,
            [
                "At time 2, the contents of box B2 move to box B3.",
                "At time 0, amber starts in box B1.",
                "At time 1, the contents of box B1 move to box B2.",
                "At time 0, violet starts in box B9.",
            ],
            "Which box contains amber after all timed events? Return its code.",
            "B3",
        ),
    ],
)
def test_independent_v2_solver_handles_shuffled_facts(task, facts, question, answer):
    problem = Problem(
        id="fixture",
        base_id="v2-fixture",
        task=task,
        split=Split.TRAIN,
        condition=Condition.SIMILAR,
        facts=tuple(Fact(id=f"F{i}", text=t) for i, t in enumerate(facts, 1)),
        question=question,
        answer="deliberately-wrong",
        evidence=("F1",),
    )
    assert solve(problem) == answer


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("answer", "wrong", "Reference answer mismatch"),
        ("evidence", ("F1",), "Evidence annotations do not match"),
    ],
)
def test_v2_rejects_corrupt_annotations(problems, field, value, message):
    modified = (problems[0].model_copy(update={field: value}), *problems[1:])
    with pytest.raises(DatasetError, match=message):
        validate_dataset(modified)


def test_similar_distractors_use_same_symbol_family_and_variable_fact_positions(problems):
    noisy = [p for p in problems if p.condition == Condition.SIMILAR]
    assert any(p.evidence != tuple(f"F{i}" for i in range(1, len(p.evidence) + 1)) for p in noisy)
    for problem in noisy:
        irrelevant = [f.text for f in problem.facts if f.id not in problem.evidence]
        assert len(irrelevant) == len(problem.evidence)
        if problem.task == Task.DEDUCTION:
            assert all("property p" in text.lower() for text in irrelevant)
