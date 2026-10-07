import pytest

from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.models.experiment import Condition, GenerationConfig, Split, Task


def test_source_selection_excludes_targets_and_is_stable_under_input_reordering():
    data = generate_v21(GenerationConfig(problems_per_task=10))
    config = PilotConfig(revision="a" * 40, examples=4)
    selected = select_pilot(data, config)
    assert len(selected) == 4
    assert all(p.task == Task.DEDUCTION and p.split == Split.TRAIN for p in selected)
    assert all(p.condition == Condition.SIMILAR for p in selected)
    assert selected == select_pilot(tuple(reversed(data)), config)
    clean = select_pilot(data, config.model_copy(update={"condition": Condition.CLEAN}))
    assert [p.base_id for p in selected] == [p.base_id for p in clean]
    repeated = select_pilot(data, config.model_copy(update={"seed": 43}))
    assert repeated == selected


def test_missing_or_duplicate_sources_fail_instead_of_using_evaluation():
    data = generate_v21(GenerationConfig(problems_per_task=10))
    config = PilotConfig(revision="a" * 40, examples=4)
    with pytest.raises(ValueError, match="Insufficient source"):
        select_pilot(tuple(p for p in data if p.split != Split.TRAIN), config)
    selected = select_pilot(data, config)
    with pytest.raises(ValueError, match="Duplicate source"):
        select_pilot((*data, selected[0]), config)


@pytest.mark.parametrize(
    ("tasks", "examples"),
    [((Task.ARITHMETIC, Task.DEDUCTION), 8), (tuple(Task), 12)],
)
def test_balanced_selection_uses_only_training_bases(tasks, examples):
    data = generate_v21(GenerationConfig(problems_per_task=20))
    config = PilotConfig(revision="a" * 40, tasks=tasks, examples=examples)
    selected = select_pilot(data, config)
    assert len(selected) == examples
    assert all(p.split == Split.TRAIN and p.condition == Condition.SIMILAR for p in selected)
    assert all(sum(p.task == task for p in selected) == examples // len(tasks) for task in tasks)
    assert selected == select_pilot(tuple(reversed(data)), config)


@pytest.mark.parametrize(
    "tasks,examples",
    [((), 8), ((Task.ARITHMETIC, Task.ARITHMETIC), 8), ((Task.ARITHMETIC, Task.DEDUCTION), 7)],
)
def test_invalid_balanced_selection_is_rejected(tasks, examples):
    with pytest.raises(ValueError, match="Tasks must be unique"):
        PilotConfig(revision="a" * 40, tasks=tasks, examples=examples)
