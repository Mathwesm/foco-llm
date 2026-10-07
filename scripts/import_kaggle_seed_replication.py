"""Validate a Kaggle seed-replication archive and preserve compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

plt.switch_backend("Agg")
REPLICATION_FORMS = ("explicit", "alias")
ABLATION_FORMS = ("full", "remove_relevant", "remove_irrelevant")
PREVIOUS_COUNTS = {
    "replication": {"explicit": 60, "alias": 46},
    "ablation": {"full": 20, "remove_relevant": 18, "remove_irrelevant": 19},
}
KAGGLE_RUN = (
    "https://www.kaggle.com/code/matheusdsousaribeiro/"
    "foco-llm-qwen-7b-seed-43-replication?scriptVersionId=356099017"
)
TRAINING_SEED = 43
TRAINING_STEPS = 320


class Score(BaseModel):
    """Fields needed to independently check the published group counts."""

    model_config = ConfigDict(extra="allow")
    id: str
    form: str
    selection_exact: bool
    answer_correct: bool


def read_json(archive: zipfile.ZipFile, member: str) -> Any:
    """Read one JSON member from an archive.

    Args:
        archive: Downloaded Kaggle output archive.
        member: Exact member name inside the archive.

    Returns:
        Parsed JSON value.

    Raises:
        KeyError: The member is absent.
        ValueError: The member is not valid JSON.
    """
    return json.loads(archive.read(member))


def find_member(archive: zipfile.ZipFile, suffix: str) -> str:
    """Find exactly one archive member by a stable suffix.

    Args:
        archive: Downloaded Kaggle output archive.
        suffix: Required trailing path.

    Returns:
        Matching full archive path.

    Raises:
        ValueError: The suffix matches zero or multiple members.
    """
    matches = [name for name in archive.namelist() if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one archive member ending in {suffix}; found {len(matches)}")
    return matches[0]


def verify_scores(scores: list[Score], summary: dict[str, Any], forms: tuple[str, ...]) -> None:
    """Check score uniqueness and all exact-selection and answer totals.

    Args:
        scores: Per-case outputs including raw model responses.
        summary: Saved Kaggle evaluation summary.
        forms: Frozen evaluation forms.

    Raises:
        ValueError: Scores and summary disagree or cases are duplicated.
    """
    if len({score.id for score in scores}) != len(scores):
        raise ValueError("Duplicate case identifiers in Kaggle scores")
    if len(scores) != summary["cases"]:
        raise ValueError("Kaggle score count differs from summary")
    if {score.form for score in scores} != set(forms):
        raise ValueError("Kaggle scores contain unexpected forms")
    for form in forms:
        members = [score for score in scores if score.form == form]
        observed = summary["groups"][form]
        if (
            len(members) != observed["cases"]
            or sum(score.selection_exact for score in members) != observed["selection_exact"]
            or sum(score.answer_correct for score in members) != observed["answer_correct"]
        ):
            raise ValueError(f"Kaggle scores disagree with summary for {form}")


def load_experiment(
    archive: zipfile.ZipFile, kind: str, dataset: Path, forms: tuple[str, ...]
) -> dict[str, Any]:
    """Read and verify one frozen evaluation from the downloaded archive.

    Args:
        archive: Downloaded Kaggle output archive.
        kind: Replication or ablation directory name.
        dataset: Repository copy of the frozen evaluation dataset.
        forms: Frozen condition names.

    Returns:
        Verified manifest, summary, and full per-case scores.

    Raises:
        ValueError: Dataset, adapter, or score integrity check fails.
    """
    root = f"/{kind}/{_run_id(archive, kind)}/"
    manifest = read_json(archive, find_member(archive, root + "manifest.json"))
    summary = read_json(archive, find_member(archive, root + "summary.json"))
    raw_scores = read_json(archive, find_member(archive, root + "scores.json"))
    scores = [Score.model_validate(item) for item in raw_scores]
    verify_scores(scores, summary, forms)
    dataset_sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if manifest["dataset_sha256"] != dataset_sha:
        raise ValueError(f"Frozen {kind} dataset hash differs from Kaggle run")
    manifest.pop("adapter", None)
    return {"manifest": manifest, "summary": summary, "scores": raw_scores}


def _run_id(archive: zipfile.ZipFile, kind: str) -> str:
    """Get the sole run identifier for one evaluation directory."""
    run_ids = {
        parts[2]
        for name in archive.namelist()
        if (parts := name.split("/"))[1:2] == [kind] and parts[3:]
    }
    if len(run_ids) != 1:
        raise ValueError(f"Expected one {kind} run; found {len(run_ids)}")
    return run_ids.pop()


def plot_comparison(bundle: dict[str, Any]) -> Figure:
    """Plot both 7B training seeds on honest zero-based axes.

    Args:
        bundle: Verified imported evidence and the documented seed-42 counts.

    Returns:
        Matplotlib figure for the TCC.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for axis, kind, forms, maximum in zip(
        axes,
        ("replication", "ablation"),
        (REPLICATION_FORMS, ABLATION_FORMS),
        (60, 20),
        strict=True,
    ):
        for index, form in enumerate(forms):
            values = [
                PREVIOUS_COUNTS[kind][form],
                bundle[kind]["summary"]["groups"][form]["selection_exact"],
            ]
            bars = axis.bar(
                [seed + (index - (len(forms) - 1) / 2) * 0.24 for seed in (0, 1)],
                values,
                width=0.21,
                label=form.replace("_", " ").title(),
                color=("#327AA5", "#D89032", "#629C79")[index],
            )
            axis.bar_label(bars, padding=2, fontsize=8)
        axis.set(
            title="Same-domain replication" if kind == "replication" else "Fact-removal ablation",
            xlabel="Training seed",
            ylabel=f"Exact selections (out of {maximum})",
            xticks=(0, 1),
            xticklabels=("42", "43"),
            ylim=(0, maximum + 5),
        )
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
        axis.legend(
            frameon=False,
            fontsize=8,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.24),
            ncol=len(forms),
        )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.28, top=0.87, wspace=0.3)
    return fig


def main() -> None:
    """Import verified evidence and render a two-seed comparison figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--figure", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.figure.exists():
        raise ValueError("Output already exists; choose fresh paths")
    with zipfile.ZipFile(args.archive) as archive:
        training_root = f"/training/{_run_id(archive, 'training')}/"
        training_member = find_member(archive, training_root + "manifest.json")
        training_bytes = archive.read(training_member)
        training_manifest = json.loads(training_bytes)
        training_summary = read_json(archive, find_member(archive, training_root + "summary.json"))
        training_data = archive.read(find_member(archive, "/training-dataset.json"))
        training_sha = hashlib.sha256(training_data).hexdigest()
        adapter_member = find_member(archive, training_root + "step-0320/adapter_model.safetensors")
        adapter_sha = hashlib.sha256(archive.read(adapter_member)).hexdigest()
        if (
            training_manifest["config"]["seed"] != TRAINING_SEED
            or training_summary["completed_steps"] != TRAINING_STEPS
        ):
            raise ValueError("Kaggle training differs from the frozen seed-43 plan")
        if training_manifest["dataset_sha256"] != training_sha:
            raise ValueError("Kaggle training dataset hash mismatch")
        replication = load_experiment(
            archive,
            "replication",
            Path("reports/2026-10-04/blind-replication/dataset.json"),
            REPLICATION_FORMS,
        )
        ablation = load_experiment(
            archive,
            "ablation",
            Path("reports/2026-10-04/fact-ablation/dataset.json"),
            ABLATION_FORMS,
        )
    for result in (replication, ablation):
        if result["manifest"]["runtime"]["adapter_sha256"] != adapter_sha:
            raise ValueError("Evaluation does not use the preserved seed-43 adapter")
    expected = hashlib.sha256(training_bytes).hexdigest()
    if {
        replication["manifest"]["runtime"]["training_manifest_sha256"],
        ablation["manifest"]["runtime"]["training_manifest_sha256"],
    } != {expected}:
        raise ValueError("Evaluation manifest does not match training run")
    training_manifest.pop("diagnostic_sha256", None)
    bundle = {
        "protocol": "post-hoc-7b-seed-sensitivity-v1",
        "kaggle_run": KAGGLE_RUN,
        "seed_42_counts": PREVIOUS_COUNTS,
        "seed_42_sources": [
            "docs/blind-replication-protocol-2026-10-04.md",
            "docs/fact-ablation-2026-10-04.md",
        ],
        "training": {"manifest": training_manifest, "summary": training_summary},
        "replication": replication,
        "ablation": ablation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    figure = plot_comparison(bundle)
    figure.savefig(args.figure, dpi=180, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
