import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_sha256", "different"),
        ("prompt_version", "different"),
        ("runtime", {"dtype": "float32"}),
        ("examples", 119),
    ],
)
def test_incompatible_run_controls_are_rejected(field, value):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "compare_baselines.py"))
    manifest = module["Manifest"](
        config={"revision": "a" * 40},
        dataset_sha256="fixture",
        prompt_version="fixture",
        runtime={"dtype": "float16"},
        examples=120,
    )
    altered = manifest.model_copy(update={field: value})
    with pytest.raises(ValueError, match="differs between runs"):
        module["verify_controls"](manifest, altered)


def test_only_model_identity_may_change():
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "compare_baselines.py"))
    manifest = module["Manifest"](
        config={"revision": "a" * 40},
        dataset_sha256="fixture",
        prompt_version="fixture",
        runtime={"dtype": "float16"},
        examples=120,
    )
    config = manifest.config.model_copy(update={"model_id": "other", "revision": "b" * 40})
    assert (
        module["verify_controls"](manifest, manifest.model_copy(update={"config": config})) is None
    )
    changed_seed = config.model_copy(update={"seed": 43})
    with pytest.raises(ValueError, match="Generation configuration differs"):
        module["verify_controls"](manifest, manifest.model_copy(update={"config": changed_seed}))
