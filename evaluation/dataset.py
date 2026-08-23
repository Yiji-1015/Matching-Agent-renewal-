from __future__ import annotations

import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .io import write_json, write_jsonl


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_legacy_cases(source_xlsx: Path, output_jsonl: Path) -> list[dict]:
    """Create a versionable derivative of the untouched 2025 evaluation workbook."""
    all_cases = pd.read_excel(source_xlsx, sheet_name="전체 데이터(80)")
    improved = pd.read_excel(source_xlsx, sheet_name="개선된 사례")
    required = {
        "User",
        "Input_Message",
        "Baseline_Matched_Message",
        "new_matched_message",
    }
    if not required.issubset(all_cases.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(all_cases.columns))}")

    improved_inputs = {normalize_text(value) for value in improved["Input_Message"]}
    records = []
    for index, row in all_cases.iterrows():
        input_message = normalize_text(row["Input_Message"])
        records.append(
            {
                "case_id": f"CH-{index + 1:03d}",
                "username": normalize_text(row["User"]),
                "input_message": input_message,
                "legacy_baseline_output": normalize_text(
                    row["Baseline_Matched_Message"]
                ),
                "legacy_agent_output": normalize_text(row["new_matched_message"]),
                "legacy_improved_label": input_message in improved_inputs,
                "dataset_role": "baseline_failure_challenge_set",
                "source": {
                    "workbook": source_xlsx.name,
                    "sheet": "전체 데이터(80)",
                    "row": index + 2,
                    "label_sheet": "개선된 사례",
                },
            }
        )

    if len(records) != 80:
        raise ValueError(f"Expected 80 challenge cases, found {len(records)}")
    if sum(case["legacy_improved_label"] for case in records) != 39:
        raise ValueError("Expected exactly 39 legacy improved cases")
    write_jsonl(output_jsonl, records)
    manifest_path = output_jsonl.parent / "provenance.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_workbook": source_xlsx.name,
            "source_sha256": file_sha256(source_xlsx),
            "source_sheets": ["전체 데이터(80)", "개선된 사례"],
            "case_count": len(records),
            "improved_label_count": 39,
            "design": "baseline-failure challenge set",
        }
    )
    write_json(manifest_path, manifest)
    return records


def prepare_raw_ratings(ratings_xlsx: Path, output_jsonl: Path) -> list[dict]:
    """Extract only rating2. rating3 is excluded because it is not raw response data."""
    frame = pd.read_excel(ratings_xlsx, sheet_name="rating2")
    columns = [column for column in frame.columns if column != "Evaluator"]
    records = []
    for row_position, (_, row) in enumerate(frame.iterrows(), start=2):
        item_id = f"HE-{int(row['Evaluator']):03d}"
        for rater in columns:
            choice = normalize_text(row[rater]).lower()
            if choice not in {"baseline", "proposed"}:
                raise ValueError(f"Unexpected rating {choice!r} for {item_id}/{rater}")
            records.append(
                {
                    "item_id": item_id,
                    "rater_id": str(rater).strip(),
                    "choice": choice,
                    "source": {
                        "workbook": ratings_xlsx.name,
                        "sheet": "rating2",
                        "row": row_position,
                    },
                }
            )
    item_ids = {record["item_id"] for record in records}
    pairs = {(record["item_id"], record["rater_id"]) for record in records}
    proposed = sum(record["choice"] == "proposed" for record in records)
    unanimous = sum(
        all(r["choice"] == "proposed" for r in records if r["item_id"] == item)
        for item in item_ids
    )
    if not (
        len(records) == 120
        and len(item_ids) == 40
        and len(pairs) == 120
        and all(sum(r["item_id"] == item for r in records) == 3 for item in item_ids)
        and proposed == 112
        and unanimous == 34
    ):
        raise ValueError("rating2 does not match the verified 40x3 raw evidence panel")
    write_jsonl(output_jsonl, records)
    manifest_path = output_jsonl.parent / "provenance.json"
    manifest = {}
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ratings_workbook"] = ratings_xlsx.name
    manifest["ratings_sha256"] = file_sha256(ratings_xlsx)
    manifest["ratings_sheet_used"] = "rating2"
    manifest["ratings_count"] = len(records)
    manifest["excluded_sheet"] = {
        "name": "rating3",
        "reason": "not raw responses; observed votes were redistributed",
    }
    write_json(manifest_path, manifest)
    return records
