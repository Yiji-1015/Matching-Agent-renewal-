from __future__ import annotations

import hashlib
import json
import random
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .io import append_jsonl, read_jsonl


RUBRIC = """You are a blinded evaluator of two candidate matches for a user's post.
Judge interaction feasibility, complementary intent/roles, constraint compatibility,
and persona/context fit. Do not prefer wording similarity by itself. Select A, B, or
tie. Scores are 1 (poor) to 5 (excellent). Return concise evidence grounded only in
the supplied text."""


class PairwiseJudgment(BaseModel):
    choice: Literal["A", "B", "tie"]
    interaction_feasibility_A: int = Field(ge=1, le=5)
    interaction_feasibility_B: int = Field(ge=1, le=5)
    complementary_roles_A: int = Field(ge=1, le=5)
    complementary_roles_B: int = Field(ge=1, le=5)
    constraint_fit_A: int = Field(ge=1, le=5)
    constraint_fit_B: int = Field(ge=1, le=5)
    persona_context_fit_A: int = Field(ge=1, le=5)
    persona_context_fit_B: int = Field(ge=1, le=5)
    rationale: str


def _balanced_proposed_a(case_ids: list[str], seed: int) -> set[str]:
    shuffled = sorted(case_ids)
    random.Random(seed).shuffle(shuffled)
    return set(shuffled[: len(shuffled) // 2])


def judge_refactored_runs(
    cases_path: Path,
    runs_path: Path,
    output_path: Path,
    *,
    model: str,
    seed: int = 2025,
) -> list[dict]:
    """Blindly compare each completed refactored run with its legacy baseline."""
    cases = {case["case_id"]: case for case in read_jsonl(cases_path)}
    completed_attempts = [r for r in read_jsonl(runs_path) if r.get("status") == "completed"]
    runs = list({run["case_id"]: run for run in completed_attempts}.values())
    runs = [
        run
        for run in runs
        if cases[run["case_id"]].get("legacy_baseline_output", "").strip()
        and run["result"].get("matched_message", "").strip()
    ]
    run_experiments = {run.get("experiment_id") for run in runs}
    if len(run_experiments) != 1 or None in run_experiments:
        raise ValueError("Judge input must contain exactly one run experiment")
    judge_config = {"temperature": 0, "structured_output": "PairwiseJudgment"}
    judge = ChatOpenAI(model=model, temperature=0).with_structured_output(PairwiseJudgment)
    prompt_hash = hashlib.sha256(RUBRIC.encode()).hexdigest()
    run_experiment_id = next(iter(run_experiments))
    schema_hash = hashlib.sha256(
        json.dumps(PairwiseJudgment.model_json_schema(), sort_keys=True).encode()
    ).hexdigest()
    implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dependencies = {}
    for package in ("langchain-openai", "pydantic"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = "not-installed"
    cohort_sha256 = hashlib.sha256(
        "|".join(sorted(run["run_id"] for run in runs)).encode()
    ).hexdigest()
    judge_id = hashlib.sha256(
        f"{run_experiment_id}|{model}|{prompt_hash}|{schema_hash}|"
        f"{implementation_hash}|{dependencies}|{judge_config}|{seed}|{cohort_sha256}".encode()
    ).hexdigest()[:20]
    previous = read_jsonl(output_path, missing_ok=True)
    previous_judge_ids = {record.get("judge_experiment_id") for record in previous}
    if previous_judge_ids and previous_judge_ids != {judge_id}:
        raise ValueError(
            "Refusing incompatible judge resume: eligible run cohort or judge "
            "configuration changed. Use a new --output path."
        )
    done = {
        (record.get("judge_experiment_id"), record.get("run_id"))
        for record in previous
        if record.get("status") == "completed"
    }
    produced = []
    proposed_a = _balanced_proposed_a([run["case_id"] for run in runs], seed)
    for run in runs:
        case_id = run["case_id"]
        done_key = (judge_id, run["run_id"])
        if done_key in done:
            continue
        case = cases[case_id]
        baseline = ("baseline", case["legacy_baseline_output"])
        proposed = ("proposed", run["result"].get("matched_message", ""))
        variants = [proposed, baseline] if case_id in proposed_a else [baseline, proposed]
        payload = (
            f"{RUBRIC}\n\nUSER POST:\n{case['input_message']}\n\n"
            f"CANDIDATE A:\n{variants[0][1]}\n\nCANDIDATE B:\n{variants[1][1]}"
        )
        base = {
            "judge_experiment_id": judge_id,
            "run_experiment_id": run_experiment_id,
            "run_id": run["run_id"],
            "case_id": case_id,
            "judge_model": model,
            "prompt_sha256": prompt_hash,
            "schema_sha256": schema_hash,
            "implementation_sha256": implementation_hash,
            "dependencies": dependencies,
            "judge_config": judge_config,
            "eligible_cohort_sha256": cohort_sha256,
            "seed": seed,
            "position_key": {"A": variants[0][0], "B": variants[1][0]},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        started = time.perf_counter()
        try:
            response = judge.invoke(payload)
            decoded = "tie" if response.choice == "tie" else variants[0 if response.choice == "A" else 1][0]
            record = {
                **base,
                "status": "completed",
                "latency_seconds": time.perf_counter() - started,
                "decoded_choice": decoded,
                "judgment": response.model_dump(),
            }
        except Exception as exc:
            decoded = "error"
            record = {
                **base,
                "status": "error",
                "latency_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        append_jsonl(output_path, record)
        produced.append(record)
        if record["status"] == "completed":
            done.add(done_key)
        print(f"[{case_id}] judge={decoded}", flush=True)
    return produced
