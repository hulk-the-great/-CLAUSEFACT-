"""Binary document/clause metrics and equal-width expected calibration error.

No external ML dependencies are needed for metrics/testing.
"""
from __future__ import annotations

import math


def binary_counts(y: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, int]:
    if len(y) != len(scores):
        raise ValueError("y and scores length mismatch")
    pred = [int(x >= threshold) for x in scores]
    return {
        "tp": sum(a == 1 and b == 1 for a, b in zip(y, pred)),
        "tn": sum(a == 0 and b == 0 for a, b in zip(y, pred)),
        "fp": sum(a == 0 and b == 1 for a, b in zip(y, pred)),
        "fn": sum(a == 1 and b == 0 for a, b in zip(y, pred)),
    }


def f1_score(y: list[int], scores: list[float], threshold: float = 0.5) -> float:
    c = binary_counts(y, scores, threshold)
    denom = 2 * c["tp"] + c["fp"] + c["fn"]
    return 2 * c["tp"] / denom if denom else 0.0


def auroc(y: list[int], scores: list[float]) -> float | None:
    """Rank-based ROC AUC; returns null for a single-class set."""
    if len(y) != len(scores):
        raise ValueError("y and scores length mismatch")
    positives = sum(y)
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    sorted_pairs = sorted(zip(scores, y), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(sorted_pairs):
        j = i + 1
        while j < len(sorted_pairs) and sorted_pairs[j][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in sorted_pairs[i:j])
        i = j
    return (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)


def ece(y: list[int], scores: list[float], n_bins: int = 10) -> float:
    """Confidence ECE: confidence in predicted class vs correctness."""
    if len(y) != len(scores) or not y or n_bins < 1:
        raise ValueError("invalid ECE input")
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for truth, p in zip(y, scores):
        if not (0 <= p <= 1) or not math.isfinite(p):
            raise ValueError("probabilities must be finite values in [0,1]")
        guess = int(p >= .5)
        confidence = max(p, 1 - p)
        index = min(int(confidence * n_bins), n_bins - 1)
        bins[index].append((confidence, int(guess == truth)))
    result = 0.0
    for bucket in bins:
        if bucket:
            avg_conf = sum(a for a, _ in bucket) / len(bucket)
            avg_acc = sum(b for _, b in bucket) / len(bucket)
            result += len(bucket) / len(y) * abs(avg_conf - avg_acc)
    return result


def classification_metrics(y: list[int], scores: list[float]) -> dict:
    counts = binary_counts(y, scores)
    precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.
    recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.
    return {
        "n": len(y), "accuracy": (counts["tp"] + counts["tn"]) / len(y) if y else None,
        "precision": precision, "recall": recall, "f1": f1_score(y, scores),
        "auroc": auroc(y, scores), "ece": ece(y, scores) if y else None,
        "confusion": counts,
    }
