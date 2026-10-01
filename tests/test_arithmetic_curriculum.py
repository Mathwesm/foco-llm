"""Behavioral checks for training-only arithmetic prefix augmentation."""

import runpy
from pathlib import Path

import pytest

from foco_llm.core.arithmetic_curriculum import arithmetic_prefixes, with_arithmetic_prefixes
from foco_llm.core.benchmark_v21 import generate_v21
from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.core.solver_v2 import solve_v2
from foco_llm.models.experiment import GenerationConfig, Split, Task


def test_prefixes_compute_partial_answers_without_evaluation_examples():
    dataset = generate_v21(GenerationConfig(problems_per_task=20))
    source = next(p for p in dataset if p.split == Split.TRAIN and p.task == Task.ARITHMETIC)
    prefixes = arithmetic_prefixes(source)
    assert prefixes
    assert all(p.split == Split.TRAIN and p.condition == "clean" for p in prefixes)
    assert all(p.answer == solve_v2(p) for p in prefixes)
    assert all(len(p.evidence) == len(p.facts) for p in prefixes)
    assert all(len(p.facts) < len(source.evidence) for p in prefixes)
    assert len({p.id for p in prefixes}) == len(prefixes)


def test_mixed_curriculum_retains_originals_and_does_not_add_other_tasks():
    dataset = generate_v21(GenerationConfig(problems_per_task=20))
    config = PilotConfig(
        revision="a" * 40,
        source_task=None,
        tasks=(Task.ARITHMETIC, Task.DEDUCTION, Task.TRACKING),
        examples=12,
        arithmetic_prefix_curriculum=True,
    )
    selected = select_pilot(dataset, config)
    augmented = with_arithmetic_prefixes(selected)
    assert all(problem in augmented for problem in selected)
    assert len([p for p in augmented if p.task != Task.ARITHMETIC]) == 8
    assert len(augmented) > len(selected)
    assert len({p.id for p in augmented}) == len(augmented)


def test_prefixes_reject_validation_cases_and_irrelevant_config():
    dataset = generate_v21(GenerationConfig(problems_per_task=10))
    validation = next(
        p for p in dataset if p.split == Split.VALIDATION and p.task == Task.ARITHMETIC
    )
    with pytest.raises(ValueError, match="training arithmetic"):
        arithmetic_prefixes(validation)
    with pytest.raises(ValueError, match="requires arithmetic"):
        PilotConfig(revision="a" * 40, arithmetic_prefix_curriculum=True)


def test_export_reconstructs_augmented_training_ids():
    dataset = generate_v21(GenerationConfig(problems_per_task=20))
    config = PilotConfig(
        revision="a" * 40,
        source_task=Task.ARITHMETIC,
        examples=4,
        arithmetic_prefix_curriculum=True,
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "export_pilot.py"
    reconstructed = runpy.run_path(str(script))["training_selection"](dataset, config)
    assert len(reconstructed) > config.examples
    assert any(":prefix-" in problem.id for problem in reconstructed)
    assert all(problem.split == Split.TRAIN for problem in reconstructed)
