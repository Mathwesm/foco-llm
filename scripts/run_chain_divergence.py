"""Freeze and evaluate chain distractors that diverge at controlled hops."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from random import Random
from typing import cast

from loguru import logger
from pydantic import TypeAdapter
from run_arithmetic_selection import checkpoint
from run_blind_selection import CLOUD_ID, CLOUD_REVISION, inference_config
from run_cross_task_transfer import score, selection_prompt
from run_kaggle_selection import KaggleInferenceBackend

from foco_llm.models.experiment import Condition, Fact, Problem, Split, Task
from foco_llm.services.artifacts import publish_text
from foco_llm.utils.logger import setup_logging

SEED = 20261009
BASES_PER_TASK = 20
ARMS = ("unrelated", "early", "middle", "late")
DISTRACTORS = (
    "The museum ticket was printed on Tuesday.",
    "A photograph hangs beside the entrance.",
    "The distant market closes at sunset.",
    "The kitchen shelf is made of oak.",
)
ADAPTER = TypeAdapter(tuple[Problem, ...])


def _sentences(task: Task, index: int, arm: str) -> tuple[list[str], list[str], str, str]:
    """Build one three-hop target and a scope-separated competing chain."""
    rng = Random(SEED + index + (10_000 if task == Task.TRACKING else 0))  # noqa: S311
    codes = rng.sample(range(10_000, 99_999), 7)
    prefix = "T" if task == Task.DEDUCTION else "C"
    nodes = [f"{prefix}{code}" for code in codes]
    target = f"S{index + 1}" if task == Task.DEDUCTION else f"token-{index + 1}"
    rival = f"other-{index + 1}"
    if task == Task.DEDUCTION:
        question = f"Which trait follows after three implications for specimen {target}?"
    else:
        question = f"Which crate contains token {target} after minute three?"

    def start(entity: str, node: str) -> str:
        """Write one entity-scoped chain start."""
        if task == Task.DEDUCTION:
            return f"Specimen {entity} has trait {node}."
        return f"Token {entity} starts in crate {node}."

    def edge(entity: str, step: int, source: str, destination: str) -> str:
        """Write one entity-scoped chain edge."""
        if task == Task.DEDUCTION:
            return f"For specimen {entity}, trait {source} implies trait {destination}."
        return (
            f"For token {entity}, at minute {step}, it moves from crate {source} "
            f"to crate {destination}."
        )

    relevant = [start(target, nodes[0])]
    relevant.extend(edge(target, step, nodes[step - 1], nodes[step]) for step in range(1, 4))
    if arm == "unrelated":
        return relevant, list(DISTRACTORS), question, nodes[3]
    branch = {"early": 1, "middle": 2, "late": 3}[arm]
    rival_nodes = [*nodes[:branch], *nodes[4 : 8 - branch]]
    competing = [start(rival, rival_nodes[0])]
    competing.extend(
        edge(rival, step, rival_nodes[step - 1], rival_nodes[step]) for step in range(1, 4)
    )
    return relevant, competing, question, nodes[3]


def _case(task: Task, index: int, arm: str) -> Problem:
    """Keep relevant facts, IDs and layout identical across the four arms."""
    relevant, noise, question, answer = _sentences(task, index, arm)
    rng = Random(SEED + 100_000 + index + (10_000 if task == Task.TRACKING else 0))  # noqa: S311
    slots = set(rng.sample(range(8), 4))
    relevant_items, noise_items = iter(relevant), iter(noise)
    tagged = [
        (next(relevant_items), True) if position in slots else (next(noise_items), False)
        for position in range(8)
    ]
    facts = tuple(
        Fact(id=f"F{position + 1}", text=value) for position, (value, _) in enumerate(tagged)
    )
    base_id = f"divergence-{task.value}-{index:04d}"
    return Problem(
        id=f"{base_id}:{arm}",
        base_id=base_id,
        task=task,
        split=Split.TEST,
        condition=Condition.UNRELATED if arm == "unrelated" else Condition.SIMILAR,
        facts=facts,
        question=question,
        answer=answer,
        evidence=tuple(fact.id for fact, (_, needed) in zip(facts, tagged, strict=True) if needed),
    )


def symbolic_oracle(case: Problem) -> tuple[str, tuple[str, ...]]:
    """Resolve only relations scoped to the queried entity; reject ambiguity."""
    if case.task == Task.DEDUCTION:
        query = re.fullmatch(
            r"Which trait follows after three implications for specimen (S\d+)\?", case.question
        )
        start_pattern = rf"Specimen {re.escape(query[1])} has trait (T\d+)\." if query else ""
        edge_pattern = (
            rf"For specimen {re.escape(query[1])}, trait (T\d+) implies trait (T\d+)\."
            if query
            else ""
        )
    else:
        query = re.fullmatch(
            r"Which crate contains token (token-\d+) after minute three\?", case.question
        )
        start_pattern = rf"Token {re.escape(query[1])} starts in crate (C\d+)\." if query else ""
        edge_pattern = (
            rf"For token {re.escape(query[1])}, at minute (\d+), "
            r"it moves from crate (C\d+) to crate (C\d+)\."
            if query
            else ""
        )
    if query is None:
        raise ValueError("Question has no supported target")
    starts = [
        (fact.id, match[1])
        for fact in case.facts
        if (match := re.fullmatch(start_pattern, fact.text))
    ]
    if len(starts) != 1:
        raise ValueError("Target has no unique starting fact")
    selected = {starts[0][0]}
    current = starts[0][1]
    for step in range(1, 4):
        matches = []
        for fact in case.facts:
            match = re.fullmatch(edge_pattern, fact.text)
            if match is None:
                continue
            source, destination = (
                (match[1], match[2]) if case.task == Task.DEDUCTION else (match[2], match[3])
            )
            if source == current and (case.task == Task.DEDUCTION or int(match[1]) == step):
                matches.append((fact.id, destination))
        if len(matches) != 1:
            raise ValueError("Target chain is missing or ambiguous")
        fact_id, current = matches[0]
        selected.add(fact_id)
    return current, tuple(fact.id for fact in case.facts if fact.id in selected)


def generate() -> tuple[Problem, ...]:
    """Freeze paired cases and independently check every answer and path."""
    cases = tuple(
        _case(task, index, arm)
        for task in (Task.DEDUCTION, Task.TRACKING)
        for index in range(BASES_PER_TASK)
        for arm in ARMS
    )
    for offset in range(0, len(cases), len(ARMS)):
        variants = cases[offset : offset + len(ARMS)]
        reference = variants[0]
        if any(
            case.base_id != reference.base_id
            or case.question != reference.question
            or case.answer != reference.answer
            or case.evidence != reference.evidence
            or tuple(
                case.facts[index].text
                for index, fact in enumerate(case.facts)
                if fact.id in case.evidence
            )
            != tuple(fact.text for fact in reference.facts if fact.id in reference.evidence)
            for case in variants
        ):
            raise ValueError("Paired variants changed their gold chain")
        for case in variants:
            if symbolic_oracle(case) != (case.answer, case.evidence):
                raise ValueError(f"Symbolic oracle disagrees with gold for {case.id}")
    return cases


def prepare(output: Path) -> Path:
    """Write the frozen dataset and its identity without model inference."""
    cases = generate()
    payload = ADAPTER.dump_json(cases, indent=2).decode("utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    dataset = output / "dataset.json"
    publish_text(dataset, payload)
    publish_text(
        output / "manifest.json",
        json.dumps(
            {
                "protocol": "chain-divergence-v1",
                "seed": SEED,
                "cases": len(cases),
                "bases": len(cases) // len(ARMS),
                "arms": list(ARMS),
                "dataset_sha256": digest,
                "oracle_exact": len(cases),
                "primary_metric": "selection_exact_by_task_and_arm",
            },
            indent=2,
        ),
    )
    return dataset


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    """Report each arm and paired changes from the unrelated control."""
    groups: dict[str, dict[str, int]] = {}
    paired: dict[str, dict[str, int]] = {}
    for task in (Task.DEDUCTION, Task.TRACKING):
        task_rows = {cast(str, row["id"]): row for row in rows if row["task"] == task.value}
        for arm in ARMS:
            subset = [row for row in task_rows.values() if cast(str, row["id"]).endswith(f":{arm}")]
            groups[f"{task.value}:{arm}"] = {
                "cases": len(subset),
                "valid": sum(bool(row["selection_valid"]) for row in subset),
                "exact": sum(bool(row["selection_exact"]) for row in subset),
                "false_positive": sum(cast(int, row["false_positive"]) for row in subset),
                "false_negative": sum(cast(int, row["false_negative"]) for row in subset),
            }
            if arm == "unrelated":
                continue
            matched = [
                (task_rows[cast(str, row["id"]).removesuffix(f":{arm}") + ":unrelated"], row)
                for row in subset
            ]
            paired[f"{task.value}:{arm}"] = {
                "bases": len(matched),
                "control_exact": sum(bool(control["selection_exact"]) for control, _ in matched),
                "both_exact": sum(
                    bool(control["selection_exact"]) and bool(rival["selection_exact"])
                    for control, rival in matched
                ),
                "control_only": sum(
                    bool(control["selection_exact"]) and not bool(rival["selection_exact"])
                    for control, rival in matched
                ),
                "rival_only": sum(
                    not bool(control["selection_exact"]) and bool(rival["selection_exact"])
                    for control, rival in matched
                ),
            }
    return {"cases": len(rows), "groups": groups, "paired": paired}


def evaluate(
    dataset: Path, output: Path, adapter: Path | None, training_dataset: Path | None
) -> None:
    """Evaluate one frozen model arm and retain every unedited generation."""
    cases = ADAPTER.validate_json(dataset.read_text(encoding="utf-8"))
    if cases != generate():
        raise ValueError("Dataset differs from the frozen generator")
    if adapter is not None and training_dataset is None:
        raise ValueError("Adapter evaluation requires the original training dataset")
    config = inference_config(CLOUD_ID, CLOUD_REVISION)
    backend = KaggleInferenceBackend(config, adapter, dataset, training_dataset or dataset)
    manifest = {
        "protocol": "chain-divergence-v1",
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "model_id": CLOUD_ID,
        "revision": CLOUD_REVISION,
        "config": config.model_dump(mode="json"),
        "adapter": str(adapter) if adapter else None,
        "runtime": backend.metadata(),
    }
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    destination = output / identity
    publish_text(destination / "manifest.json", json.dumps(manifest, indent=2))
    rows = []
    for index, case in enumerate(cases, 1):
        raw = checkpoint(selection_prompt(case), destination, backend).generation.text
        rows.append({**score(case, raw), "arm": case.id.rsplit(":", 1)[1], "raw_response": raw})
        if index % 20 == 0:
            logger.info("Chain divergence: completed={}/{}", index, len(cases))
    publish_text(destination / "scores.json", json.dumps(rows, indent=2))
    publish_text(destination / "summary.json", json.dumps(summarize(rows), indent=2))


@logger.catch(reraise=True)
def main() -> None:
    """Prepare or evaluate the controlled chain-divergence holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--training-dataset", type=Path)
    args = parser.parse_args()
    setup_logging(log_dir=Path("logs"), serialize=True)
    if args.mode == "prepare":
        prepare(args.output)
    elif args.dataset is None:
        parser.error("Evaluation requires --dataset")
    else:
        evaluate(args.dataset, args.output, args.adapter, args.training_dataset)


if __name__ == "__main__":
    main()
