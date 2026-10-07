"""Audit paired evidence-selection errors in the frozen 7B fact ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from loguru import logger
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, TypeAdapter
from run_arithmetic_selection import score_selection

from foco_llm.models.experiment import Problem

plt.switch_backend("Agg")

FORMS = ("full", "remove_relevant", "remove_irrelevant")
ARM_NAMES = ("baseline", "adapted-0320")
BASES = 20
EXPECTED_CASES = BASES * len(FORMS)
DATASET_ADAPTER = TypeAdapter(tuple[Problem, ...])


class RawResponse(BaseModel):
    """One preserved selector response associated with its frozen case."""

    model_config = ConfigDict(extra="forbid")
    id: str
    raw_response: str


class RawArtifact(BaseModel):
    """Transcribed Kaggle response artifact with provenance metadata."""

    model_config = ConfigDict(extra="forbid")
    source: str
    notebook_url: str
    dataset_sha256: str
    model_revision: str
    adapter_sha256: str
    arms: dict[str, list[RawResponse]]


def validate_inputs(dataset_path: Path, raw_path: Path) -> tuple[dict[str, Problem], RawArtifact]:
    """Validate provenance, paired forms, and exactly one response per arm/case.

    Args:
        dataset_path: Frozen case definitions.
        raw_path: Preserved 7B responses.

    Returns:
        Case lookup and validated response artifact.

    Raises:
        ValueError: Inputs disagree or contain missing or duplicate records.
    """
    dataset_bytes = dataset_path.read_bytes()
    cases = DATASET_ADAPTER.validate_json(dataset_bytes)
    raw = RawArtifact.model_validate_json(raw_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(dataset_bytes).hexdigest()
    if digest != raw.dataset_sha256:
        raise ValueError("Raw response dataset digest differs from frozen dataset")
    case_map = {case.id: case for case in cases}
    if len(case_map) != EXPECTED_CASES or len(cases) != EXPECTED_CASES:
        raise ValueError("Frozen dataset must contain 60 unique cases")
    base_forms: dict[str, set[str]] = {}
    for case in cases:
        base_id, form = case.id.rsplit(":", 1)
        base_forms.setdefault(base_id, set()).add(form)
    if len(base_forms) != BASES or any(forms != set(FORMS) for forms in base_forms.values()):
        raise ValueError("Frozen dataset must contain 20 complete paired triplets")
    if set(raw.arms) != set(ARM_NAMES):
        raise ValueError("Response artifact must contain baseline and adapted-0320")
    for arm, rows in raw.arms.items():
        row_ids = [row.id for row in rows]
        if len(row_ids) != len(case_map) or set(row_ids) != set(case_map):
            raise ValueError(f"Arm {arm} has missing, extra, or duplicate case IDs")
    return case_map, raw


def score_arms(
    cases: dict[str, Problem], raw: RawArtifact
) -> dict[str, dict[str, dict[str, object]]]:
    """Rescore every preserved response using the project's production scorer.

    Args:
        cases: Frozen case lookup.
        raw: Validated response artifact.

    Returns:
        Scores keyed by arm and case ID.
    """
    return {
        arm: {row.id: score_selection(cases[row.id], row.raw_response) for row in rows}
        for arm, rows in raw.arms.items()
    }


def audit_arm(rows: dict[str, dict[str, object]]) -> dict[str, object]:
    """Summarize error mechanisms and within-base correctness transitions.

    Args:
        rows: Rescored cases for one model arm.

    Returns:
        Counts by form, individual failed cases, and paired transitions.
    """
    groups: dict[str, dict[str, int]] = {}
    errors: list[dict[str, object]] = []
    for form in FORMS:
        selected = [row for row in rows.values() if row["form"] == form]
        groups[form] = {
            "cases": len(selected),
            "exact": sum(bool(row["selection_exact"]) for row in selected),
            "malformed": sum(not bool(row["selection_valid"]) for row in selected),
            "cases_with_inclusion": sum(int(row["false_positive"]) > 0 for row in selected),
            "cases_with_omission": sum(int(row["false_negative"]) > 0 for row in selected),
            "included_facts": sum(int(row["false_positive"]) for row in selected),
            "omitted_facts": sum(int(row["false_negative"]) for row in selected),
        }
        for row in selected:
            if row["selection_exact"]:
                continue
            expected = set(row["expected_ids"])
            chosen = set(row["selected_ids"])
            errors.append(
                {
                    "id": row["id"],
                    "included_ids": sorted(chosen - expected),
                    "omitted_ids": sorted(expected - chosen),
                    "error": row["error"],
                }
            )
    transitions: dict[str, dict[str, int]] = {}
    for form in FORMS[1:]:
        counts: Counter[str] = Counter()
        for index in range(BASES):
            base = f"replication-arithmetic-{index:04d}"
            original = bool(rows[f"{base}:full"]["selection_exact"])
            changed = bool(rows[f"{base}:{form}"]["selection_exact"])
            counts[f"{int(original)}->{int(changed)}"] += 1
        transitions[form] = dict(sorted(counts.items()))
    return {"groups": groups, "errors": errors, "transitions": transitions}


def plot_errors(audits: dict[str, dict[str, object]]) -> plt.Figure:
    """Plot case-level error mechanisms without implying independent samples.

    Args:
        audits: Validated per-arm summaries.

    Returns:
        Matplotlib figure for the six arm-by-form conditions.
    """
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    labels = ["Full", "Relevant removed", "Irrelevant removed"]
    positions = list(range(6))
    omission = []
    inclusion = []
    malformed = []
    for arm in ARM_NAMES:
        groups = audits[arm]["groups"]
        for form in FORMS:
            group = groups[form]
            omission.append(group["cases_with_omission"])
            inclusion.append(group["cases_with_inclusion"])
            malformed.append(group["malformed"])
    ax.bar(positions, inclusion, color="#D89032", label="Included irrelevant fact")
    ax.bar(positions, omission, bottom=inclusion, color="#327AA5", label="Omitted relevant fact")
    ax.bar(
        positions,
        malformed,
        bottom=[a + b for a, b in zip(inclusion, omission, strict=True)],
        color="#8064A2",
        label="Malformed output",
    )
    ax.set_xticks(positions, labels * 2, rotation=20, ha="right")
    ax.set_ylim(0, 4)
    ax.set_yticks(range(5))
    ax.set_ylabel("Cases with error type (out of 20)")
    ax.set_title("7B fact-removal ablation: evidence-selection errors")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3)
    ax.axvline(2.5, color="#888888", linewidth=0.8)
    ax.text(1, 3.65, "Base 7B", ha="center", fontsize=10)
    ax.text(4, 3.65, "Adapted 7B", ha="center", fontsize=10)
    fig.subplots_adjust(bottom=0.32)
    return fig


def main() -> None:
    """Create a reproducible JSON audit and a publication-sized figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases, raw = validate_inputs(args.dataset, args.raw)
    scored = score_arms(cases, raw)
    audits = {arm: audit_arm(rows) for arm, rows in scored.items()}
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "audit.json"
    if report_path.exists() or (args.output / "error-types.png").exists():
        raise ValueError("Audit output already exists; choose a new run directory")
    report_path.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")
    figure = plot_errors(audits)
    figure.savefig(args.output / "error-types.png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    logger.info("Audited cases={} arms={} output={}", len(cases), len(audits), args.output)


if __name__ == "__main__":
    main()
