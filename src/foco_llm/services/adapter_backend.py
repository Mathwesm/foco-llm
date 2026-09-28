"""Load a verified local smoke adapter without mixing baseline checkpoints."""

import hashlib
import importlib
from pathlib import Path

from pydantic import BaseModel

from foco_llm.models.inference import InferenceConfig
from foco_llm.models.training import SmokeConfig
from foco_llm.services.training_checkpoints import verify_checkpoint
from foco_llm.services.transformers_backend import TransformersBackend


class TrainingManifest(BaseModel):
    """Required training provenance; additional recorded metadata is retained on disk."""

    config: SmokeConfig
    dataset_sha256: str


def verify_adapter(path: Path, config: InferenceConfig, dataset: Path) -> dict[str, str]:
    """Reject damaged checkpoints, wrong base revisions, or a different dataset.

    Args:
        path: Local completed training checkpoint directory.
        config: Requested base model and generation settings.
        dataset: Frozen dataset used for paired validation.

    Returns:
        Content identities to include in the inference manifest.

    Raises:
        ValueError: Checkpoint integrity or training provenance does not match.
    """
    state = verify_checkpoint(path)
    manifest_path = path.parent / "manifest.json"
    manifest = TrainingManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if (config.model_id, config.revision) != (manifest.config.model_id, manifest.config.revision):
        raise ValueError("Adapter base model or revision does not match inference")
    if manifest.dataset_sha256 != hashlib.sha256(dataset.read_bytes()).hexdigest():
        raise ValueError("Adapter training dataset differs from evaluation dataset")
    return {
        "adapter_sha256": state.digests["adapter_model.safetensors"],
        "adapter_config_sha256": state.digests["adapter_config.json"],
        "training_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "adapter_step": str(len(state.history)),
    }


class AdapterBackend(TransformersBackend):
    """Use the baseline generation path with an integrity-checked local adapter."""

    def __init__(self, config: InferenceConfig, path: Path, dataset: Path) -> None:
        self.adapter_identity = verify_adapter(path, config, dataset)
        super().__init__(config)
        peft = importlib.import_module("peft")
        self.model = peft.PeftModel.from_pretrained(self.model, path, is_trainable=False)
        self.model.eval()
        self.peft_version = str(peft.__version__)

    def metadata(self) -> dict[str, str]:
        """Bind resumable responses to the exact adapter bytes and training manifest."""
        return {**super().metadata(), **self.adapter_identity, "peft": self.peft_version}
