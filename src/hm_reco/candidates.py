"""Candidate generation (the "recall" stage).

Each function below returns a DataFrame of (customer_id, article_id) pairs:
plausible items a customer might buy next week. We deliberately keep recall
sources simple and cheap, then let the ranking model in `model.py` decide
which candidates actually look promising.

All functions take `transactions` already filtered to "history up to and
not including the target week" — see `features.build_training_frame`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def repurchase_candidates(transactions: pd.DataFrame, lookback_weeks: int, target_week: int) -> pd.DataFrame:
    """Items each customer already bought in the last `lookback_weeks`.

    Repurchase is one of the strongest signals in this dataset: customers
    frequently rebuy staples (basics, underwear, kids' clothing sizes).
    """
    recent = transactions[transactions["week"] >= target_week - lookback_weeks]
    cand = recent[["customer_id", "article_id"]].drop_duplicates()
    return cand


def popular_last_week_candidates(transactions: pd.DataFrame, target_week: int, top_n: int = 12) -> pd.DataFrame:
    """The N best-selling articles in the week right before `target_week`,
    offered to every customer as a fallback / trend signal.
    """
    last_week = transactions[transactions["week"] == target_week - 1]
    top_articles = last_week["article_id"].value_counts().head(top_n).index
    return pd.DataFrame({"article_id": top_articles})


def popular_by_age_bucket(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    target_week: int,
    lookback_weeks: int = 1,
    n_buckets: int = 4,
    top_n: int = 12,
) -> pd.DataFrame:
    """Best-selling articles within each customer age bucket.

    Fashion preference correlates with age (e.g. kidswear vs. menswear vs.
    womenswear basics), so age-bucket popularity is a cheap way to
    personalize the "popular items" fallback beyond a single global list.
    """
    recent = transactions[transactions["week"] >= target_week - lookback_weeks]
    recent = recent.merge(customers[["customer_id", "age"]], on="customer_id", how="left")
    recent = recent.dropna(subset=["age"])
    recent["age_bucket"] = pd.qcut(recent["age"], n_buckets, duplicates="drop")

    top_per_bucket = (
        recent.groupby("age_bucket", observed=True)["article_id"]
        .value_counts()
        .groupby(level=0, observed=True)
        .head(top_n)
        .reset_index(name="count")[["age_bucket", "article_id"]]
    )

    customers_with_bucket = customers[["customer_id", "age"]].dropna(subset=["age"]).copy()
    bucket_edges = recent["age_bucket"].cat.categories
    customers_with_bucket["age_bucket"] = pd.cut(customers_with_bucket["age"], bins=_edges_to_bins(bucket_edges))

    # Merge (not a Python-level cross-product loop) keeps this to one pass
    # over ~1.4M customers even though top_per_bucket only has a handful of
    # (bucket, article) rows per bucket.
    cand = customers_with_bucket.merge(top_per_bucket, on="age_bucket", how="inner")
    return cand[["customer_id", "article_id"]]


def _edges_to_bins(categories) -> list:
    edges = [categories[0].left] + [c.right for c in categories]
    edges[0] = -float("inf")
    edges[-1] = float("inf")
    return edges


def item_pair_candidates(transactions: pd.DataFrame, pairs: pd.DataFrame, lookback_weeks: int, target_week: int) -> pd.DataFrame:
    """Items frequently bought alongside a customer's recent purchases
    (item-to-item co-purchase, "customers who bought X also bought Y").

    `pairs` is the output of `build_item_pairs` computed on the *same*
    history window, so there is no leakage from the target week.
    """
    recent = transactions[transactions["week"] >= target_week - lookback_weeks]
    recent_articles = recent[["customer_id", "article_id"]].drop_duplicates()
    cand = recent_articles.merge(pairs, on="article_id", how="inner")
    cand = cand[["customer_id", "paired_article_id"]].rename(columns={"paired_article_id": "article_id"})
    return cand.drop_duplicates()


def build_item_pairs(transactions: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """For every article, the `top_k` articles most frequently bought by
    the same customer on the same day (a simple co-purchase signal).

    This is O(baskets), computed once per training/prediction window and
    reused by `item_pair_candidates`.
    """
    baskets = transactions[["customer_id", "article_id", "t_dat"]].drop_duplicates()
    basket_key = ["customer_id", "t_dat"]
    merged = baskets.merge(baskets, on=basket_key, suffixes=("", "_paired"))
    merged = merged[merged["article_id"] != merged["article_id_paired"]]

    pair_counts = (
        merged.groupby(["article_id", "article_id_paired"])
        .size()
        .reset_index(name="pair_count")
    )
    pair_counts = pair_counts.sort_values(["article_id", "pair_count"], ascending=[True, False])
    top_pairs = pair_counts.groupby("article_id").head(top_k)
    return top_pairs.rename(columns={"article_id_paired": "paired_article_id"})[
        ["article_id", "paired_article_id"]
    ]


def combine_candidates(*candidate_frames: pd.DataFrame, all_customers: pd.Series | None = None) -> pd.DataFrame:
    """Union several candidate frames into one deduplicated (customer_id,
    article_id) table. Frames without a customer_id column (e.g. a plain
    popular-items list) are broadcast to `all_customers`.
    """
    parts = []
    for frame in candidate_frames:
        if frame.empty:
            continue
        if "customer_id" not in frame.columns:
            if all_customers is None:
                raise ValueError("all_customers required to broadcast a customer-less candidate frame")
            n_customers = len(all_customers)
            n_articles = len(frame)
            frame = pd.DataFrame(
                {
                    "customer_id": np.repeat(all_customers.to_numpy(), n_articles),
                    "article_id": np.tile(frame["article_id"].to_numpy(), n_customers),
                }
            )
        parts.append(frame[["customer_id", "article_id"]])
    if not parts:
        return pd.DataFrame(columns=["customer_id", "article_id"])
    combined = pd.concat(parts, ignore_index=True)
    return combined.drop_duplicates().reset_index(drop=True)
