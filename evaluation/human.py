from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .metrics import summarize_human_ratings


def _stable_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{seed}|{case_id}".encode()).hexdigest()
    return int(digest[:16], 16)


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def export_blind_sheet(
    cases_path: Path,
    output_csv: Path,
    *,
    key_output: Path,
    runs_path: Path | None = None,
    seed: int = 2025,
) -> list[dict]:
    """Create counterbalanced forms and a separately located decoding key.

    Without ``runs_path`` this reproduces the 2025 legacy comparison. With it,
    the proposed arm is replaced by each completed refactored graph result.
    Incomplete pairs are excluded rather than showing a revealing blank arm.
    """
    cases = read_jsonl(cases_path)
    run_outputs = None
    run_metadata = {}
    if runs_path:
        completed = [
            record for record in read_jsonl(runs_path)
            if record.get("status") == "completed"
        ]
        experiments = {record.get("experiment_id") for record in completed}
        if len(experiments) != 1 or None in experiments:
            raise ValueError("Blind export requires exactly one run experiment")
        latest = {record["case_id"]: record for record in completed}
        run_outputs = {
            case_id: record["result"].get("matched_message", "")
            for case_id, record in latest.items()
        }
        run_metadata = {
            case_id: {
                "run_id": record["run_id"],
                "experiment_id": record["experiment_id"],
            }
            for case_id, record in latest.items()
        }
    forms = 2
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    key_records = []
    fields = ["form_id", "item_id", "input_message", "candidate_A", "candidate_B", "choice", "comment"]
    for form_number in range(1, 3):
        form_id = f"F{form_number}"
        form_path = output_csv.with_name(
            f"{output_csv.stem}_{form_id}{output_csv.suffix}"
        )
        with form_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for case in cases:
                proposed_output = (
                    run_outputs.get(case["case_id"], "")
                    if run_outputs is not None
                    else case["legacy_agent_output"]
                )
                if not case["legacy_baseline_output"] or not proposed_output:
                    continue
                variants = [
                    ("baseline", case["legacy_baseline_output"]),
                    ("proposed", proposed_output),
                ]
                random.Random(_stable_seed(seed, case["case_id"])).shuffle(variants)
                if form_number == 2:
                    variants.reverse()
                writer.writerow(
                    {
                        "form_id": form_id,
                        "item_id": case["case_id"],
                        "input_message": case["input_message"],
                        "candidate_A": variants[0][1],
                        "candidate_B": variants[1][1],
                        "choice": "",
                        "comment": "",
                    }
                )
                key_records.append(
                    {
                        "form_id": form_id,
                        "item_id": case["case_id"],
                        "A": variants[0][0],
                        "B": variants[1][0],
                        "seed": seed,
                        "proposed_source": "refactored_run" if runs_path else "legacy_2025",
                        **run_metadata.get(case["case_id"], {}),
                        "input_sha256": _text_sha256(case["input_message"]),
                        "candidate_A_sha256": _text_sha256(variants[0][1]),
                        "candidate_B_sha256": _text_sha256(variants[1][1]),
                    }
                )
    write_jsonl(key_output, key_records)
    return key_records


def import_blind_responses(
    response_csvs: list[Path], key_path: Path, output_jsonl: Path
) -> list[dict]:
    key = {
        (record["form_id"], record["item_id"]): record
        for record in read_jsonl(key_path)
    }
    ratings = []
    for response_path in response_csvs:
        with response_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
            actual_items = {(row.get("form_id", ""), row.get("item_id", "")) for row in rows}
            form_ids = {form_id for form_id, _ in actual_items}
            expected_items = {
                pair for pair in key if pair[0] in form_ids
            }
            if len(form_ids) != 1 or actual_items != expected_items or len(actual_items) != len(rows):
                raise ValueError(
                    f"Altered, missing, duplicate, or mixed-form rows in {response_path.name}"
                )
            for row in rows:
                choice = row.get("choice", "").strip().upper()
                if choice not in {"A", "B", "TIE"}:
                    raise ValueError(
                        f"Missing/invalid choice in {response_path.name}: {row.get('item_id')}"
                    )
                key_item = key.get((row.get("form_id", ""), row["item_id"]))
                if key_item is None:
                    raise ValueError(f"Unknown form/item in {response_path.name}")
                if (
                    _text_sha256(row.get("input_message", "")) != key_item["input_sha256"]
                    or _text_sha256(row.get("candidate_A", "")) != key_item["candidate_A_sha256"]
                    or _text_sha256(row.get("candidate_B", "")) != key_item["candidate_B_sha256"]
                ):
                    raise ValueError(f"Altered text in {response_path.name}: {row['item_id']}")
                decoded = "tie" if choice == "TIE" else key_item[choice]
                ratings.append(
                    {
                        "item_id": row["item_id"],
                        "rater_id": response_path.stem,
                        "form_id": row["form_id"],
                        "chosen_position": choice,
                        "choice": decoded,
                        "comment": row.get("comment", "").strip(),
                        "source_file": response_path.name,
                    }
                )
    write_jsonl(output_jsonl, ratings)
    return ratings


def validate_complete_panel(ratings: list[dict]) -> dict:
    if not ratings:
        raise ValueError("No human ratings")
    by_item = defaultdict(list)
    for rating in ratings:
        by_item[rating["item_id"]].append(rating["rater_id"])
    duplicate_count = len(ratings) - len(
        {(r["item_id"], r["rater_id"]) for r in ratings}
    )
    rater_sets = {frozenset(values) for values in by_item.values()}
    invalid_choices = {
        rating["choice"]
        for rating in ratings
        if rating["choice"] not in {"baseline", "proposed", "tie"}
    }
    form_ratings = [rating for rating in ratings if rating.get("form_id")]
    form_counts = defaultdict(int)
    if form_ratings:
        forms_by_rater = defaultdict(set)
        for rating in form_ratings:
            form_counts[rating["form_id"]] += 1
            forms_by_rater[rating["rater_id"]].add(rating["form_id"])
        rater_form_counts = defaultdict(int)
        for forms in forms_by_rater.values():
            if len(forms) == 1:
                rater_form_counts[next(iter(forms))] += 1
        form_invalid = (
            set(form_counts) != {"F1", "F2"}
            or any(len(forms) != 1 for forms in forms_by_rater.values())
            or abs(rater_form_counts["F1"] - rater_form_counts["F2"]) > 1
        )
    else:
        form_invalid = False
    if duplicate_count or len(rater_sets) != 1 or len(next(iter(rater_sets))) < 2 or invalid_choices or form_invalid:
        raise ValueError(
            "Incomplete/invalid panel: duplicate pairs, unequal rater sets, "
            "fewer than two raters, or unknown choices"
        )
    return {
        "items": len(by_item),
        "raters_per_item": sorted({len(values) for values in by_item.values()}),
        "duplicate_item_rater_pairs": duplicate_count,
        "form_rating_counts": dict(form_counts),
        "form_rater_counts": dict(rater_form_counts) if form_ratings else {},
        "summary": summarize_human_ratings(ratings),
    }
