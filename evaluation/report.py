from __future__ import annotations

from pathlib import Path

from .io import read_jsonl, write_json
from .metrics import summarize_human_ratings, summarize_runs


def _text(value: str) -> str:
    return (value or "(결측)").replace("\n", " ").strip()


def write_legacy_report(
    cases_path: Path,
    ratings_path: Path | None,
    output_dir: Path,
) -> dict:
    cases = read_jsonl(cases_path)
    summary = {
        "design": "baseline-failure challenge set; not an overall-accuracy sample",
        "challenge_cases": len(cases),
        "legacy_improved_cases": sum(c["legacy_improved_label"] for c in cases),
        "legacy_correction_rate": (
            sum(c["legacy_improved_label"] for c in cases) / len(cases)
        ),
        "missing_baseline_outputs": sum(not c["legacy_baseline_output"] for c in cases),
        "missing_agent_outputs": sum(not c["legacy_agent_output"] for c in cases),
    }
    if ratings_path:
        summary["human_evaluation"] = summarize_human_ratings(read_jsonl(ratings_path))
    write_json(output_dir / "legacy_summary.json", summary)

    lines = [
        "# Legacy challenge-set casebook",
        "",
        "> 이 자료는 baseline이 실패한 사례 80건을 모은 challenge set이다. "
        "39/80은 전체 정확도가 아니라 이 실패 집합에서의 correction rate이다.",
        "",
        f"- Cases: {len(cases)}",
        f"- Labeled corrections: {summary['legacy_improved_cases']} "
        f"({summary['legacy_correction_rate']:.1%})",
        "",
    ]
    for case in cases:
        label = "개선 사례" if case["legacy_improved_label"] else "미개선/미판정"
        lines.extend(
            [
                f"## {case['case_id']} — {label}",
                "",
                f"- 입력: {_text(case['input_message'])}",
                f"- Baseline: {_text(case['legacy_baseline_output'])}",
                f"- Proposed (2025): {_text(case['legacy_agent_output'])}",
                "",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "legacy_casebook.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def write_run_report(cases_path: Path, runs_path: Path, output_dir: Path) -> dict:
    cases = {case["case_id"]: case for case in read_jsonl(cases_path)}
    attempts = read_jsonl(runs_path)
    summary = summarize_runs(attempts)
    runs = list({run["case_id"]: run for run in attempts}.values())
    write_json(output_dir / "run_summary.json", summary)
    lines = ["# Refactored LangGraph evaluation run", ""]
    for run in runs:
        case = cases.get(run["case_id"], {})
        lines.extend([f"## {run['case_id']} — {run['status']}", ""])
        if run["status"] == "completed":
            result = run["result"]
            lines.extend(
                [
                    f"- 입력: {_text(case.get('input_message', ''))}",
                    f"- Legacy baseline: {_text(case.get('legacy_baseline_output', ''))}",
                    f"- Legacy proposed: {_text(case.get('legacy_agent_output', ''))}",
                    f"- Refactored: {_text(result.get('matched_message', ''))}",
                    f"- Pipeline valid: {run['automatic_metrics']['pipeline_valid']}",
                    f"- Queries: {result.get('reformed_queries', [])}",
                ]
            )
        else:
            lines.append(f"- Error: {run.get('error_type')}: {run.get('error_message')}")
        lines.append("")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_casebook.md").write_text("\n".join(lines), encoding="utf-8")
    return summary
