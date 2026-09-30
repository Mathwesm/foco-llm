"""Export verified source-task pilot provenance and loss without publishing weights."""

import argparse
import hashlib
import json
from pathlib import Path

from matplotlib.figure import Figure
from pydantic import TypeAdapter

from foco_llm.core.pilot_configuration import PilotConfig, select_pilot
from foco_llm.models.experiment import Problem
from foco_llm.models.training import StepResult
from foco_llm.services.artifacts import load_dataset, publish_text
from foco_llm.services.training_checkpoints import verify_checkpoint


def plot_history(history: tuple[StepResult, ...]) -> Figure:
    """Plot observed training-example losses; these are not validation scores."""
    figure = Figure(figsize=(9, 4), layout="constrained")
    axis = figure.subplots()
    axis.plot([r.step for r in history], [r.loss for r in history], color="#0072B2")
    axis.set(
        xlabel="Optimizer update (count)",
        ylabel="Completion loss (nats/token)",
        title="Source-task pilot: training loss across different examples",
        ylim=(0, None),
    )
    return figure


def export(dataset: Path, source: Path, output: Path) -> None:
    """Verify source selection and final checkpoint before immutable publication."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    config = PilotConfig.model_validate(manifest["config"])
    if hashlib.sha256(dataset.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Pilot dataset differs from manifest")
    selected = select_pilot(load_dataset(dataset), config)
    if [p.id for p in selected] != manifest["example_ids"]:
        raise ValueError("Pilot selection differs from manifest")
    state = verify_checkpoint(source / f"step-{config.steps:04d}")
    history = TypeAdapter(tuple[StepResult, ...]).validate_json(
        (source / "history.json").read_text(encoding="utf-8")
    )
    if history != state.history or len(history) != config.steps:
        raise ValueError("Pilot history differs from final checkpoint")
    if any(r.example_id != selected[i % len(selected)].id for i, r in enumerate(history)):
        raise ValueError("Pilot history violates the declared training schedule")
    for name in ("manifest.json", "summary.json", "history.json"):
        publish_text(output / name, (source / name).read_text(encoding="utf-8"))
    publish_text(
        output / "training-examples.json",
        TypeAdapter(tuple[Problem, ...]).dump_json(selected, indent=2).decode(),
    )
    if not (output / "training-loss.png").exists():
        plot_history(history).savefig(output / "training-loss.png", dpi=180, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.dataset, args.source, args.output)
