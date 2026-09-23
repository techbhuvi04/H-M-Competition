"""MAP@12 (Mean Average Precision at 12), the competition metric.

`mean_average_precision` takes a mapping of customer_id -> list of
predicted article_ids (already ranked, best first) and a mapping of
customer_id -> set of article_ids actually purchased, and returns the
competition score.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def average_precision_at_k(predicted: list, actual: set, k: int = 12) -> float:
    if not actual:
        return 0.0
    predicted = predicted[:k]

    hits = 0
    score = 0.0
    seen = set()
    for i, item in enumerate(predicted, start=1):
        if item in actual and item not in seen:
            hits += 1
            score += hits / i
        seen.add(item)

    return score / min(len(actual), k)


def mean_average_precision(
    predictions: Dict, ground_truth: Dict, k: int = 12
) -> float:
    """predictions: {customer_id: [article_id, ...]} (ranked)
    ground_truth: {customer_id: {article_id, ...}} (actual purchases)

    Customers with no ground-truth purchases are excluded, matching the
    competition's own scoring rule.
    """
    scored_customers = [c for c in ground_truth if len(ground_truth[c]) > 0]
    if not scored_customers:
        return 0.0

    total = 0.0
    for customer in scored_customers:
        predicted = predictions.get(customer, [])
        total += average_precision_at_k(predicted, ground_truth[customer], k)
    return total / len(scored_customers)


def predictions_from_scores(scored: "pd.DataFrame", k: int = 12) -> Dict:
    """scored: DataFrame with columns customer_id, article_id, score.
    Returns {customer_id: [top-k article_ids by score, descending]}.
    """
    import pandas as pd  # local import to keep module import-light

    ranked = scored.sort_values(["customer_id", "score"], ascending=[True, False])
    grouped = ranked.groupby("customer_id")["article_id"].apply(lambda s: s.head(k).tolist())
    return grouped.to_dict()


def ground_truth_from_transactions(transactions, target_week: int) -> Dict:
    target = transactions[transactions["week"] == target_week]
    return target.groupby("customer_id")["article_id"].apply(set).to_dict()
