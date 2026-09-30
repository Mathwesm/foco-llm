import hashlib
import json

import pytest

from foco_llm.models.inference import InferenceConfig
from foco_llm.services.adapter_backend import verify_adapter
from foco_llm.services.training_checkpoints import STATE_FILES


@pytest.fixture
def adapter_files(tmp_path):
    dataset = tmp_path / "dataset.json"
    dataset.write_text("[]", encoding="utf-8")
    config = InferenceConfig(revision="a" * 40)
    checkpoint = tmp_path / "step-0001"
    checkpoint.mkdir()
    digests = {}
    for name in STATE_FILES:
        (checkpoint / name).write_bytes(b"fixture")
        digests[name] = hashlib.sha256(b"fixture").hexdigest()
    state = {
        "history": [
            {
                "step": 1,
                "example_id": "train",
                "loss": 1.0,
                "elapsed_seconds": 1.0,
                "supervised_tokens": 2,
                "peak_allocated_bytes": 0,
                "peak_reserved_bytes": 0,
            }
        ],
        "digests": digests,
    }
    (checkpoint / "complete.json").write_text(json.dumps(state), encoding="utf-8")
    manifest = {
        "config": {"revision": config.revision, "model_id": config.model_id},
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint, config, dataset


@pytest.mark.parametrize("change", ["weights", "dataset", "revision", "model"])
def test_adapter_rejects_wrong_provenance_before_loading(adapter_files, change):
    checkpoint, config, dataset = adapter_files
    if change == "weights":
        (checkpoint / "adapter_model.safetensors").write_bytes(b"damaged")
    elif change == "dataset":
        dataset.write_text("[1]", encoding="utf-8")
    elif change == "revision":
        config = config.model_copy(update={"revision": "b" * 40})
    else:
        config = config.model_copy(update={"model_id": "different"})
    with pytest.raises(ValueError, match="integrity|differs|does not match"):
        verify_adapter(checkpoint, config, dataset)


def test_adapter_identity_binds_weights_and_training_manifest(adapter_files):
    checkpoint, config, dataset = adapter_files
    identity = verify_adapter(checkpoint, config, dataset)
    assert identity["adapter_sha256"] == hashlib.sha256(b"fixture").hexdigest()
    manifest = checkpoint.parent / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    updated = verify_adapter(checkpoint, config, dataset)
    assert identity["training_manifest_sha256"] != updated["training_manifest_sha256"]
    assert identity["adapter_sha256"] == updated["adapter_sha256"]


def test_source_pilot_manifest_accepts_extended_budget_without_loosening_smoke(adapter_files):
    from foco_llm.core.pilot_configuration import PilotConfig
    from foco_llm.models.training import SmokeConfig

    checkpoint, config, dataset = adapter_files
    manifest_path = checkpoint.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"] = PilotConfig(
        revision=config.revision, model_id=config.model_id
    ).model_dump()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_adapter(checkpoint, config, dataset)["adapter_step"] == "1"
    with pytest.raises(ValueError, match="less than or equal to 12"):
        SmokeConfig(revision=config.revision, steps=64)
