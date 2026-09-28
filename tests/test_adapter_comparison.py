import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("field", ["config", "dataset_sha256", "prompt_version", "runtime"])
def test_comparison_rejects_uncontrolled_changes(tmp_path, field):
    verify = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "compare_adapter.py"))[
        "verify_controls"
    ]
    before, after = tmp_path / "before", tmp_path / "after"
    original = {
        "config": {"seed": 42},
        "dataset_sha256": "data",
        "examples": 120,
        "prompt_version": "prompt",
        "runtime": {"dtype": "bfloat16"},
    }
    changed = {**original, "runtime": {"dtype": "bfloat16", "adapter_sha256": "adapter"}}
    if field == "runtime":
        changed[field]["dtype"] = "float16"
    else:
        changed[field] = "different"
    for path, manifest in ((before, original), (after, changed)):
        path.mkdir()
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "validation.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="differ"):
        verify(before, after)


def test_comparison_accepts_adapter_metadata_but_rejects_changed_examples(tmp_path):
    verify = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "compare_adapter.py"))[
        "verify_controls"
    ]
    paths = tmp_path / "before", tmp_path / "after"
    for index, path in enumerate(paths):
        path.mkdir()
        manifest = {
            "config": {},
            "dataset_sha256": "same",
            "examples": 120,
            "prompt_version": "same",
            "runtime": {"dtype": "bfloat16"},
        }
        if index:
            manifest["runtime"]["adapter_sha256"] = "weights"
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "validation.json").write_text("[]", encoding="utf-8")
    verify(*paths)
    (paths[1] / "validation.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="Validation examples differ"):
        verify(*paths)
