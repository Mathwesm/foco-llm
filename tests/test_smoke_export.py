import hashlib
import json
import runpy
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from foco_llm.core.benchmark_v2 import generate_v2
from foco_llm.core.training_data import select_smoke_examples
from foco_llm.models.experiment import GenerationConfig, Problem


@pytest.mark.parametrize("change", ["dataset", "selection"])
def test_export_rejects_mismatched_dataset_or_training_selection(tmp_path, change):
    verify_dataset = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "export_smoke.py"))[
        "verify_dataset"
    ]
    problems = generate_v2(GenerationConfig(problems_per_task=10))
    dataset = tmp_path / "dataset.json"
    dataset.write_bytes(TypeAdapter(tuple[Problem, ...]).dump_json(problems))
    manifest = tmp_path / "manifest.json"
    metadata = {
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "example_ids": [p.id for p in select_smoke_examples(problems)],
    }
    manifest.write_text(json.dumps(metadata), encoding="utf-8")
    verify_dataset(dataset, manifest)
    if change == "dataset":
        dataset.write_bytes(dataset.read_bytes() + b" ")
    else:
        metadata["example_ids"] = ["incorrect"]
        manifest.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="differs|differ"):
        verify_dataset(dataset, manifest)
