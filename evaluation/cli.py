from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from .dataset import prepare_legacy_cases, prepare_raw_ratings
from .human import export_blind_sheet, import_blind_responses, validate_complete_panel
from .io import read_jsonl, write_json
from .report import write_legacy_report, write_run_report


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "evaluation" / "data" / "challenge_80.jsonl"
DEFAULT_RATINGS = ROOT / "evaluation" / "data" / "human_ratings_40x3.jsonl"
DEFAULT_REPORTS = ROOT / "evaluation" / "reports"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Matching-Agent evaluation pipeline")
    sub = root.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Derive versionable JSONL from legacy workbooks")
    prepare.add_argument("--source-xlsx", type=Path, required=True)
    prepare.add_argument("--ratings-xlsx", type=Path)
    prepare.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    prepare.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)

    legacy = sub.add_parser("legacy-report", help="Build legacy summary and 80-case casebook")
    legacy.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    legacy.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    legacy.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)

    run = sub.add_parser("run", help="Run the refactored graph over challenge cases")
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    run.add_argument("--output", type=Path, default=ROOT / "evaluation" / "runs" / "latest.jsonl")
    run.add_argument("--model", default="gpt-4o")
    run.add_argument("--limit", type=int)
    run.add_argument("--no-resume", action="store_true")

    report = sub.add_parser("run-report", help="Summarize graph health and create a casebook")
    report.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS)

    blind = sub.add_parser("export-blind", help="Export randomized A/B human-rating CSV")
    blind.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    blind.add_argument("--runs", type=Path, help="Use completed refactored outputs as the proposed arm")
    blind.add_argument("--output", type=Path, required=True)
    blind.add_argument("--key-output", type=Path, required=True, help="Protected decoding key; do not send to raters")
    blind.add_argument("--seed", type=int, default=2025)

    score = sub.add_parser("score-human", help="Decode response CSVs and calculate raw agreement")
    score.add_argument("--key", type=Path)
    score.add_argument("--responses", type=Path, nargs="+")
    score.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    score.add_argument("--output", type=Path, default=DEFAULT_REPORTS / "human_summary.json")

    judge = sub.add_parser("judge", help="Optional blinded LLM judge: legacy baseline vs refactor")
    judge.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    judge.add_argument("--runs", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--model", default="gpt-4o-mini")
    judge.add_argument("--seed", type=int, default=2025)
    return root


def main() -> None:
    load_dotenv()
    args = parser().parse_args()
    if args.command == "prepare":
        cases = prepare_legacy_cases(args.source_xlsx, args.cases)
        ratings = prepare_raw_ratings(args.ratings_xlsx, args.ratings) if args.ratings_xlsx else []
        print(json.dumps({"cases": len(cases), "ratings": len(ratings)}, ensure_ascii=False))
    elif args.command == "legacy-report":
        print(json.dumps(write_legacy_report(args.cases, args.ratings, args.output_dir), ensure_ascii=False, indent=2))
    elif args.command == "run":
        from .runner import run_cases

        run_cases(args.cases, args.output, model=args.model, limit=args.limit, resume=not args.no_resume)
    elif args.command == "run-report":
        print(json.dumps(write_run_report(args.cases, args.runs, args.output_dir), ensure_ascii=False, indent=2))
    elif args.command == "export-blind":
        keys = export_blind_sheet(
            args.cases,
            args.output,
            key_output=args.key_output,
            runs_path=args.runs,
            seed=args.seed,
        )
        print(f"exported_items={len(keys)}")
        print(args.output)
    elif args.command == "score-human":
        if args.responses and not args.key:
            raise SystemExit("--key is required when --responses are supplied")
        ratings = (
            import_blind_responses(args.responses, args.key, args.ratings)
            if args.responses
            else read_jsonl(args.ratings)
        )
        summary = validate_complete_panel(ratings)
        write_json(args.output, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "judge":
        from .judge import judge_refactored_runs

        judge_refactored_runs(args.cases, args.runs, args.output, model=args.model, seed=args.seed)


if __name__ == "__main__":
    main()
