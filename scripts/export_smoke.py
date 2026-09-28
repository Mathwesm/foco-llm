"""Audit resumed/uninterrupted local training and publish portable result records."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path

from loguru import logger
from pydantic import TypeAdapter

from foco_llm.core.training_data import select_smoke_examples
from foco_llm.models.experiment import Problem
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.training import _plot
from foco_llm.services.training_checkpoints import latest_checkpoint, verify_checkpoint


def verify_dataset(dataset: Path, manifest_path: Path) -> None:
    """Reject exports that attach examples from a different dataset to the run."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Export dataset differs from training manifest")
    selected = select_smoke_examples(load_dataset(dataset))
    if [p.id for p in selected] != manifest["example_ids"]:
        raise ValueError("Export examples differ from training selection")


def compare_states(left: Path, right: Path) -> dict[str, bool]:
    """Check exact adapter and optimizer tensor equality, without loading model weights."""
    torch = importlib.import_module("torch")
    tensors = importlib.import_module("safetensors.torch")
    first = tensors.load_file(left / "adapter_model.safetensors")
    second = tensors.load_file(right / "adapter_model.safetensors")
    adapters_equal = first.keys() == second.keys() and all(
        torch.equal(value, second[key]) for key, value in first.items()
    )
    first_state = torch.load(left / "state.pt", map_location="cpu", weights_only=True)
    second_state = torch.load(right / "state.pt", map_location="cpu", weights_only=True)
    a, b = first_state["optimizer"], second_state["optimizer"]
    optimizer_equal = (
        a["param_groups"] == b["param_groups"] and a["state"].keys() == b["state"].keys()
    )
    optimizer_equal = optimizer_equal and all(
        values.keys() == b["state"][key].keys()
        and all(torch.equal(value, b["state"][key][field]) for field, value in values.items())
        for key, values in a["state"].items()
    )
    return {"adapter_tensors_equal": adapters_equal, "optimizer_tensors_equal": optimizer_equal}


@logger.catch(reraise=True)
def export(dataset: Path, resumed: Path, uninterrupted: Path, output: Path) -> None:
    """Verify checkpoint integrity and export results without weights or final-test data."""
    left, right = latest_checkpoint(resumed), latest_checkpoint(uninterrupted)
    if left is None or right is None:
        raise ValueError("Both runs must contain completed checkpoints")
    if (resumed / "manifest.json").read_bytes() != (uninterrupted / "manifest.json").read_bytes():
        raise ValueError("Run controls differ")
    verify_dataset(dataset, resumed / "manifest.json")
    a, b = verify_checkpoint(left), verify_checkpoint(right)
    equality = compare_states(left, right)
    equality["losses_equal"] = [(r.step, r.example_id, r.loss) for r in a.history] == [
        (r.step, r.example_id, r.loss) for r in b.history
    ]
    if not all(equality.values()):
        raise ValueError("Resumed training differs from uninterrupted control")
    for run, name in ((resumed, "resumed"), (uninterrupted, "uninterrupted")):
        for filename in ("manifest.json", "summary.json", "history.json"):
            publish_text(output / name / filename, (run / filename).read_text(encoding="utf-8"))
    publish_text(output / "resume-verification.json", json.dumps(equality, indent=2))
    examples = select_smoke_examples(load_dataset(dataset))
    publish_text(
        output / "training-examples.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(examples, indent=2).decode("utf-8"),
    )
    publish_text(
        output / "dataset-manifest.json",
        (dataset.parent / "manifest.json").read_text(encoding="utf-8"),
    )
    if not (output / "training-loss.png").exists():
        _plot(list(a.history)).savefig(output / "training-loss.png", dpi=180, bbox_inches="tight")
    logger.info("Exact restart equivalence verified; training report saved: {}", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("uninterrupted", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.dataset, args.resumed, args.uninterrupted, args.output)
