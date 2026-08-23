from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.human import export_blind_sheet, import_blind_responses
from evaluation.io import read_jsonl, write_jsonl
from evaluation.metrics import (
    evaluate_run,
    fleiss_kappa,
    summarize_human_ratings,
    summarize_runs,
)


class EvaluationTests(unittest.TestCase):
    def test_pipeline_health_is_not_semantic_accuracy(self):
        case = {"username": "me"}
        result = {
            "matched_message": "need a designer",
            "matched_username": "other",
            "reformed_queries": ["designer wanted", "design collaboration"],
            "candidate_hits": [
                {"username": "other", "message": "need a designer", "sources": [{}, {}]}
            ],
            "evaluation_count": 1,
            "retry_count": 0,
            "fail_or_not": "success",
        }
        metrics = evaluate_run(case, result)
        self.assertTrue(metrics["pipeline_valid"])
        self.assertNotIn("accuracy", metrics)

    def test_run_summary_counts_failures(self):
        valid = {
            "case_id": "CH-001",
            "status": "completed",
            "automatic_metrics": {
                "pipeline_valid": True,
                "checks": {"has_output": True},
                "candidate_count": 4,
                "retry_count": 0,
            },
        }
        summary = summarize_runs([valid, {"case_id": "CH-002", "status": "error"}])
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["errors"], 1)

    def test_human_summary_counts_unanimous_items(self):
        ratings = []
        for item in range(40):
            choices = ["proposed"] * 3
            if item < 6:
                choices[item % 3] = "baseline"
            for rater, choice in enumerate(choices):
                ratings.append({"item_id": str(item), "rater_id": str(rater), "choice": choice})
        summary = summarize_human_ratings(ratings)
        self.assertEqual(summary["proposed_votes"], 114)
        self.assertEqual(summary["unanimous_proposed_items"], 34)

    def test_checked_in_raw_derivative_reproduces_2025_evidence(self):
        path = Path(__file__).resolve().parents[1] / "evaluation" / "data" / "human_ratings_40x3.jsonl"
        if not path.exists():
            self.skipTest("private local evidence derivative is not present")
        ratings = read_jsonl(path)
        summary = summarize_human_ratings(ratings)
        self.assertEqual(len({(r["item_id"], r["rater_id"]) for r in ratings}), 120)
        self.assertEqual(summary["proposed_votes"], 112)
        self.assertEqual(summary["unanimous_proposed_items"], 34)

    def test_degenerate_kappa_is_undefined(self):
        ratings = [
            {"item_id": str(item), "choice": "proposed"}
            for item in range(2)
            for _ in range(3)
        ]
        self.assertIsNone(fleiss_kappa(ratings))

    def test_blind_export_is_reproducible_and_has_separate_key(self):
        case = {
            "case_id": "CH-001",
            "input_message": "input",
            "legacy_baseline_output": "base",
            "legacy_agent_output": "agent",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = root / "cases.jsonl"
            form = root / "form.csv"
            key = root / "private" / "key.jsonl"
            write_jsonl(cases, [case])
            first = export_blind_sheet(cases, form, key_output=key, seed=7)
            second = export_blind_sheet(cases, form, key_output=key, seed=7)
            self.assertEqual(first, second)
            self.assertTrue(key.exists())
            with form.with_name("form_F1.csv").open(encoding="utf-8-sig", newline="") as handle:
                row_f1 = next(csv.DictReader(handle))
            with form.with_name("form_F2.csv").open(encoding="utf-8-sig", newline="") as handle:
                row_f2 = next(csv.DictReader(handle))
            self.assertEqual(row_f1["candidate_A"], row_f2["candidate_B"])
            self.assertEqual(row_f1["candidate_B"], row_f2["candidate_A"])

    def test_blind_import_rejects_altered_candidate_text(self):
        case = {
            "case_id": "CH-001",
            "input_message": "input",
            "legacy_baseline_output": "base",
            "legacy_agent_output": "agent",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases, form, key = root / "cases.jsonl", root / "form.csv", root / "key.jsonl"
            write_jsonl(cases, [case])
            export_blind_sheet(cases, form, key_output=key)
            form_f1 = form.with_name("form_F1.csv")
            with form_f1.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["choice"] = "A"
            rows[0]["candidate_A"] = "tampered"
            response = root / "rater_1.csv"
            with response.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Altered text"):
                import_blind_responses([response], key, root / "ratings.jsonl")


if __name__ == "__main__":
    unittest.main()
