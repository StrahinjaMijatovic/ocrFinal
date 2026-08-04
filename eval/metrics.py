from typing import Any


def json_valid_rate(results: list[dict | None]) -> float:
    """Fraction of results that are valid dicts (non-None)."""
    if not results:
        return 0.0
    return sum(1 for r in results if isinstance(r, dict)) / len(results)


def field_completeness(results: list[dict], expected_fields: list[str]) -> float:
    """Average fraction of expected fields that are populated (non-null) across results."""
    if not results or not expected_fields:
        return 0.0
    scores = []
    for r in results:
        fields = r.get("fields", {})
        filled = sum(1 for f in expected_fields if fields.get(f) is not None)
        scores.append(filled / len(expected_fields))
    return sum(scores) / len(scores)


def doc_type_accuracy(results: list[dict], expected_types: list[str]) -> float:
    """Fraction of results where document_type matches expected."""
    if not results:
        return 0.0
    correct = sum(
        1 for r, exp in zip(results, expected_types)
        if isinstance(r, dict) and r.get("document_type") == exp
    )
    return correct / len(results)


def category_precision(assigned: list[str], expected: list[str]) -> float:
    """Fraction of assigned categories that are in the expected set."""
    if not assigned:
        return 0.0
    return sum(1 for c in assigned if c in expected) / len(assigned)


def summarize(variant: str, results: list[dict], expected_types: list[str],
              expected_fields: list[str], latencies: list[float]) -> dict:
    return {
        "variant": variant,
        "json_valid_rate": round(json_valid_rate(results), 3),
        "field_completeness": round(field_completeness(results, expected_fields), 3),
        "doc_type_accuracy": round(doc_type_accuracy(results, expected_types), 3),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "n_cases": len(results),
    }
