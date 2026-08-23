from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean


def evaluate_run(case: dict, result: dict) -> dict:
    """Score graph invariants, not semantic correctness.

    Semantic quality requires human or blinded LLM judgments. Keeping these
    categories separate prevents pipeline-health numbers from being mislabeled
    as matching accuracy.
    """
    message = (result.get("matched_message") or "").strip()
    username = (result.get("matched_username") or "").strip()
    queries = [str(q).strip() for q in result.get("reformed_queries", [])]
    hits = result.get("candidate_hits", [])
    candidate_selected = any(
        hit.get("message") == message
        and hit.get("username") == username
        for hit in hits
    )
    self_match = bool(username and username == case.get("username"))
    checks = {
        "has_output": bool(message and username),
        "selected_from_retrieved_candidates": candidate_selected,
        "excluded_self_match": not self_match,
        "exactly_two_distinct_reformed_queries": (
            len(queries) == 2 and len({q.casefold() for q in queries}) == 2
        ),
        "evaluation_bound_respected": int(result.get("evaluation_count", 0)) <= 2,
        # A semantic evaluator failure can retry once; two missing-result retries
        # are possible because that defensive path has no evaluation_count.
        "retry_bound_respected": int(result.get("retry_count", 0)) <= 2,
    }
    return {
        "checks": checks,
        "pipeline_valid": all(checks.values()),
        "candidate_count": len(hits),
        "retrieval_source_count": sum(len(hit.get("sources", [])) for hit in hits),
        "evaluation_count": int(result.get("evaluation_count", 0)),
        "retry_count": int(result.get("retry_count", 0)),
        "final_status": result.get("fail_or_not", ""),
    }


def summarize_runs(records: list[dict]) -> dict:
    latest_by_case = {record["case_id"]: record for record in records}
    selected = list(latest_by_case.values())
    completed = [record for record in selected if record.get("status") == "completed"]
    failed = [record for record in selected if record.get("status") == "error"]
    if not completed:
        return {
            "attempts": len(records),
            "unique_cases": len(selected),
            "completed": 0,
            "errors": len(failed),
        }
    check_names = completed[0]["automatic_metrics"]["checks"].keys()
    return {
        "attempts": len(records),
        "unique_cases": len(selected),
        "completed": len(completed),
        "errors": len(failed),
        "pipeline_valid_rate": mean(
            record["automatic_metrics"]["pipeline_valid"] for record in completed
        ),
        "check_pass_rates": {
            name: mean(
                record["automatic_metrics"]["checks"][name]
                for record in completed
            )
            for name in check_names
        },
        "mean_candidate_count": mean(
            record["automatic_metrics"]["candidate_count"] for record in completed
        ),
        "retry_rate": mean(
            record["automatic_metrics"]["retry_count"] > 0 for record in completed
        ),
    }


def fleiss_kappa(ratings: list[dict]) -> float | None:
    """Fleiss' kappa for complete nominal ratings with two or more categories."""
    by_item: dict[str, list[str]] = defaultdict(list)
    for rating in ratings:
        by_item[rating["item_id"]].append(rating["choice"])
    counts = [Counter(values) for values in by_item.values()]
    if not counts:
        raise ValueError("No ratings")
    n_raters = len(next(iter(by_item.values())))
    if n_raters < 2 or any(len(values) != n_raters for values in by_item.values()):
        raise ValueError("Fleiss' kappa requires the same >=2 raters per item")
    categories = sorted({choice for values in by_item.values() for choice in values})
    if len(categories) < 2:
        return None
    observed = mean(
        (sum(count[category] ** 2 for category in categories) - n_raters)
        / (n_raters * (n_raters - 1))
        for count in counts
    )
    total = len(counts) * n_raters
    proportions = {
        category: sum(count[category] for count in counts) / total
        for category in categories
    }
    expected = sum(value**2 for value in proportions.values())
    return (observed - expected) / (1 - expected) if expected < 1 else None


def summarize_human_ratings(ratings: list[dict]) -> dict:
    by_item: dict[str, list[str]] = defaultdict(list)
    for rating in ratings:
        by_item[rating["item_id"]].append(rating["choice"])
    all_choices = [choice for values in by_item.values() for choice in values]
    unanimous_proposed = sum(
        bool(values) and all(value == "proposed" for value in values)
        for values in by_item.values()
    )
    return {
        "items": len(by_item),
        "ratings": len(all_choices),
        "proposed_votes": all_choices.count("proposed"),
        "proposed_vote_rate": all_choices.count("proposed") / len(all_choices),
        "unanimous_proposed_items": unanimous_proposed,
        "unanimous_proposed_rate": unanimous_proposed / len(by_item),
        "full_agreement_rate": mean(len(set(values)) == 1 for values in by_item.values()),
        "fleiss_kappa_raw": fleiss_kappa(ratings),
    }
