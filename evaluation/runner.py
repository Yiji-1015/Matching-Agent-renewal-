from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from matching_agent import build_matching_graph, get_initial_state
from matching_agent.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_RECURSION_LIMIT,
    DEFAULT_RETRIEVAL_K,
    DEFAULT_VECTORSTORE_DIR,
)
from matching_agent.nodes import MatchingAgentNodes
from matching_agent.retriever import load_retriever

from .io import append_jsonl, read_jsonl, write_json
from .metrics import evaluate_run


def _run_id(case_id: str, model: str) -> str:
    stamp = datetime.now(timezone.utc).isoformat()
    return hashlib.sha256(f"{case_id}|{model}|{stamp}".encode()).hexdigest()[:16]


def _tree_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    resolved = {}
    for package in ("langchain", "langgraph", "langchain-openai", "faiss-cpu", "pydantic"):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def build_experiment_manifest(cases_path: Path, model: str) -> dict:
    project_root = Path(__file__).resolve().parents[1]
    index_files = [path for path in DEFAULT_VECTORSTORE_DIR.glob("*") if path.is_file()]
    inputs = {
        "cases_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "model": model,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
        "retrieval_k": DEFAULT_RETRIEVAL_K,
        "max_evaluations": DEFAULT_MAX_EVALUATIONS,
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "prompt_set_sha256": _tree_sha256(list((project_root / "prompts").glob("*.txt"))),
        "matching_agent_code_sha256": _tree_sha256(list((project_root / "matching_agent").glob("*.py"))),
        "evaluation_code_sha256": _tree_sha256(list((project_root / "evaluation").glob("*.py"))),
        "vectorstore_sha256": _tree_sha256(index_files),
        "requirements_sha256": hashlib.sha256((project_root / "requirements.txt").read_bytes()).hexdigest(),
        "package_versions": _package_versions(),
    }
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    return {
        "experiment_id": hashlib.sha256(canonical.encode()).hexdigest()[:20],
        "inputs": inputs,
    }


def _serializable_result(result: dict) -> dict:
    keys = (
        "reformed_queries",
        "candidate_hits",
        "selector_results",
        "matched_message",
        "matched_username",
        "certainty",
        "fail_or_not",
        "evaluation_count",
        "retry_count",
        "failure_log",
        "trace",
    )
    return {key: result.get(key) for key in keys}


def run_cases(
    cases_path: Path,
    output_path: Path,
    *,
    model: str,
    limit: int | None = None,
    resume: bool = True,
) -> list[dict]:
    cases = read_jsonl(cases_path)
    if limit is not None:
        cases = cases[:limit]
    manifest = build_experiment_manifest(cases_path, model)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    previous = read_jsonl(output_path, missing_ok=True) if resume else []
    if resume and previous:
        previous_ids = {record.get("experiment_id") for record in previous}
        if previous_ids != {manifest["experiment_id"]}:
            raise ValueError(
                "Refusing incompatible resume: model, data, prompts, code, "
                "retriever, or dependencies changed. Use a new --output path."
            )
    if not resume and output_path.exists():
        output_path.unlink()
    if not resume and manifest_path.exists():
        manifest_path.unlink()
    write_json(
        manifest_path,
        {
            **manifest,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    )
    done = {
        record["case_id"]
        for record in previous
        if record.get("status") == "completed"
    }
    graph = build_matching_graph(
        nodes=MatchingAgentNodes(load_retriever(), llm_model=model)
    )
    produced = []
    for case in cases:
        if case["case_id"] in done:
            continue
        run_id = _run_id(case["case_id"], model)
        base = {
            "run_id": run_id,
            "experiment_id": manifest["experiment_id"],
            "case_id": case["case_id"],
            "model": model,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": sum(record.get("case_id") == case["case_id"] for record in previous) + 1,
        }
        started = time.perf_counter()
        try:
            state = get_initial_state(case["input_message"], case["username"])
            result = graph.invoke(
                state,
                config=RunnableConfig(
                    recursion_limit=DEFAULT_RECURSION_LIMIT,
                    configurable={"thread_id": run_id},
                ),
            )
            record = {
                **base,
                "status": "completed",
                "latency_seconds": time.perf_counter() - started,
                "result": _serializable_result(result),
                "automatic_metrics": evaluate_run(case, result),
            }
        except Exception as exc:  # Preserve failures for audit/resume.
            record = {
                **base,
                "status": "error",
                "latency_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        append_jsonl(output_path, record)
        produced.append(record)
        print(f"[{case['case_id']}] {record['status']}", flush=True)
    return previous + produced
