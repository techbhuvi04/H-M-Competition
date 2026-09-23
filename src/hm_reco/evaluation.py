"""MAP@12 (Mean Average Precision at 12), the competition metric.

  MAP@12 = 1/U * sum_u  1/min(m_u, 12) * sum_{k<=12} P_u(k) * rel_u(k)

Only customers who bought something in the scored week count (U), and
duplicate predictions earn nothing.
"""
from __future__ import annotations

import pandas as pd


def average_precision_at_k(predicted: list, actual: set, k: int = 12) -> float:
    if not actual:
        return 0.0
    hits, score, seen = 0, 0.0, set()
    for i, item in enumerate(predicted[:k], start=1):
        if item in actual and item not in seen:
            hits += 1
            score += hits / i
        seen.add(item)
    return score / min(len(actual), k)


def mean_average_precision(predictions: dict, ground_truth: dict, k: int = 12) -> float:
    """predictions: {customer: ranked [article, ...]}, ground_truth: {customer: {article, ...}}."""
    scored = [c for c, items in ground_truth.items() if items]
    if not scored:
        return 0.0
    return sum(average_precision_at_k(predictions.get(c, []), ground_truth[c], k) for c in scored) / len(scored)


def predictions_from_scores(scored: pd.DataFrame, k: int = 12) -> dict:
    """scored: customer_id, article_id, score -> {customer_id: top-k article_ids by score}."""
    top = scored.sort_values(["customer_id", "score"], ascending=[True, False])
    top = top[top.groupby("customer_id").cumcount() < k]
    return top.groupby("customer_id")["article_id"].agg(list).to_dict()
