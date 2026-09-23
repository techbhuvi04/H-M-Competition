"""Formatting the final submission.csv.

Two things the competition requires that are easy to get wrong:
  1. Every customer in customers.csv needs a row, even ones with no
     candidates/history (cold-start) -- fill those with popular items.
  2. article_id must be written as a zero-padded 10-digit string
     (the raw ids have a leading zero, e.g. "0706016001").
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _format_article_id(article_id: int) -> str:
    return str(int(article_id)).zfill(10)


def build_submission(
    predictions: Dict,
    all_customer_codes: pd.Series,
    customer_id_map: pd.Series,
    fallback_articles: list[int],
    k: int = 12,
) -> pd.DataFrame:
    """predictions: {customer_code: [article_id, ...]} from
    `evaluation.predictions_from_scores`.
    all_customer_codes: every customer_code that must appear in the output
      (i.e. customers["customer_id"] after `data.load_dataset` remapping).
    customer_id_map: code -> original hex customer_id string (Dataset.customer_id_map).
    fallback_articles: articles used to pad predictions with fewer than k
      items (e.g. this week's most popular articles) -- there's no MAP@12
      penalty for extra guesses, so always fill all k slots.
    """
    fallback = [_format_article_id(a) for a in fallback_articles]

    rows = []
    for code in all_customer_codes:
        preds = [_format_article_id(a) for a in predictions.get(code, [])][:k]
        if len(preds) < k:
            for a in fallback:
                if a not in preds:
                    preds.append(a)
                if len(preds) == k:
                    break
        rows.append((customer_id_map[code], " ".join(preds)))

    return pd.DataFrame(rows, columns=["customer_id", "prediction"])
