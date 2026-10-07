"""Score a transparent literal-target selector as a synthetic benchmark control."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import TypeAdapter
from run_arithmetic_selection import EXPECTED_CASES, FORMS, score_selection, summarize

from foco_llm.core.arithmetic_selection import lexical_select
from foco_llm.models.experiment import Problem
from foco_llm.services.artifacts import publish_text


def main() -> None:
    """Publish selection and calculation scores without model inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    all_cases = TypeAdapter(tuple[Problem, ...]).validate_json(
        args.dataset.read_text(encoding="utf-8")
    )
    cases = tuple(p for p in all_cases if p.id.rsplit(":", 1)[-1] in FORMS)
    if len(cases) != EXPECTED_CASES:
        raise ValueError("Expected 40 arithmetic selection cases")
    manifest = {
        "protocol": "lexical-selection-posthoc-v1",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "cases": len(cases),
        "rule": "literal target from question as a whole word in fact sentence",
    }
    payload = json.dumps(manifest, sort_keys=True, indent=2)
    output = args.output / hashlib.sha256(payload.encode()).hexdigest()[:16]
    publish_text(output / "manifest.json", payload)
    rows = [
        score_selection(problem, json.dumps({"evidence": lexical_select(problem)}))
        for problem in cases
    ]
    publish_text(output / "scores.json", json.dumps(rows, indent=2))
    publish_text(output / "summary.json", json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
