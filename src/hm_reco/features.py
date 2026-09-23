"""Feature engineering and training-frame assembly.

The core idea (standard for this kind of competition): pick a "target week"
whose purchases are the labels, build candidates + features from *only*
the weeks strictly before it, then label each candidate 1 if the customer
actually bought that article in the target week, else 0.

Stacking several (feature window, target week) pairs gives the ranker more
training data without needing extra raw history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import candidates as cand_mod


def article_popularity_features(transactions: pd.DataFrame, target_week: int, windows=(1, 2, 4)) -> pd.DataFrame:
    """Purchase counts per article over the last N weeks before target_week,
    for each window in `windows`. Captures short-term trend.
    """
    out = None
    for w in windows:
        recent = transactions[transactions["week"] >= target_week - w]
        counts = recent.groupby("article_id").size().rename(f"article_count_last_{w}w")
        out = counts.to_frame() if out is None else out.join(counts, how="outer")
    return out.fillna(0).reset_index()


def article_price_features(transactions: pd.DataFrame, target_week: int, window: int = 4) -> pd.DataFrame:
    """Average and most-recent price per article, from the last `window`
    weeks. Price signals markdowns/promotions, which drive short-term demand.
    """
    recent = transactions[transactions["week"] >= target_week - window]
    agg = recent.groupby("article_id")["price"].agg(article_avg_price="mean")
    last_price = (
        recent.sort_values("t_dat").groupby("article_id")["price"].last().rename("article_last_price")
    )
    return agg.join(last_price, how="outer").reset_index()


def customer_features(transactions: pd.DataFrame, customers: pd.DataFrame, target_week: int, window: int = 12) -> pd.DataFrame:
    """Per-customer summary of recent activity: purchase count, average
    spend, and days since last purchase (as of the start of target_week).
    """
    recent = transactions[transactions["week"] >= target_week - window]
    agg = recent.groupby("customer_id").agg(
        cust_purchase_count=("article_id", "size"),
        cust_avg_price=("price", "mean"),
        cust_last_week=("week", "max"),
    )
    agg["cust_weeks_since_purchase"] = target_week - agg["cust_last_week"] - 1
    agg = agg.drop(columns="cust_last_week").reset_index()

    out = customers[["customer_id", "age"]].merge(agg, on="customer_id", how="left")
    out["cust_purchase_count"] = out["cust_purchase_count"].fillna(0)
    out["cust_weeks_since_purchase"] = out["cust_weeks_since_purchase"].fillna(window)
    return out


def customer_article_features(transactions: pd.DataFrame, target_week: int) -> pd.DataFrame:
    """Per (customer, article) history: how many times bought before, and
    how many weeks ago the most recent purchase was. The strongest features
    for the repurchase signal.
    """
    hist = transactions[["customer_id", "article_id", "week"]]
    agg = hist.groupby(["customer_id", "article_id"]).agg(
        ca_bought_count=("week", "size"),
        ca_last_week=("week", "max"),
    )
    agg["ca_weeks_since_bought"] = target_week - agg["ca_last_week"] - 1
    return agg.drop(columns="ca_last_week").reset_index()


def attach_article_metadata(cand: pd.DataFrame, articles: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Join selected static article columns (category, department, ...)."""
    return cand.merge(articles[["article_id"] + columns], on="article_id", how="left")


def build_training_frame(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    target_week: int,
    *,
    history_weeks: int,
    repurchase_weeks: int = 3,
    pair_weeks: int = 2,
    pair_top_k: int = 5,
    age_bucket_weeks: int = 1,
    article_columns: tuple[str, ...] = ("index_code", "product_group_name", "department_no"),
    for_customers: pd.Series | None = None,
    labeled: bool = True,
):
    """Assemble one (candidates + features [+ label]) frame for `target_week`.

    `history` is transactions restricted to weeks strictly before
    `target_week` and no further back than `history_weeks` — this is the
    only data the candidate/feature functions may see, which is what keeps
    the split leakage-free.

    If `labeled` is True (training/validation), a `label` column is added:
    1 if the customer actually bought that article during `target_week`.
    If False (final submission), no label column is added and
    `for_customers` should list every customer_id we must predict for.
    """
    history = transactions[
        (transactions["week"] < target_week) & (transactions["week"] >= target_week - history_weeks)
    ]

    pairs = cand_mod.build_item_pairs(
        history[history["week"] >= target_week - pair_weeks], top_k=pair_top_k
    )

    customer_pool = for_customers if for_customers is not None else customers["customer_id"]

    repurchase = cand_mod.repurchase_candidates(history, repurchase_weeks, target_week)
    pair_cand = cand_mod.item_pair_candidates(history, pairs, pair_weeks, target_week)
    popular = cand_mod.popular_last_week_candidates(history, target_week)
    age_popular = cand_mod.popular_by_age_bucket(history, customers, target_week, age_bucket_weeks)

    cand = cand_mod.combine_candidates(
        repurchase, pair_cand, age_popular, popular, all_customers=customer_pool
    )
    # Restrict to the customers we actually need predictions/labels for.
    cand = cand[cand["customer_id"].isin(customer_pool)]

    if labeled:
        target = transactions[transactions["week"] == target_week][["customer_id", "article_id"]].drop_duplicates()
        target["label"] = 1
        cand = cand.merge(target, on=["customer_id", "article_id"], how="left")
        cand["label"] = cand["label"].fillna(0).astype("int8")

    cand = cand.merge(article_popularity_features(history, target_week), on="article_id", how="left")
    cand = cand.merge(article_price_features(history, target_week), on="article_id", how="left")
    cand = cand.merge(customer_features(history, customers, target_week), on="customer_id", how="left")
    cand = cand.merge(customer_article_features(history, target_week), on=["customer_id", "article_id"], how="left")
    cand = attach_article_metadata(cand, articles, list(article_columns))

    numeric_fill_cols = [c for c in cand.columns if c.startswith(("article_count", "article_avg", "article_last", "ca_"))]
    cand[numeric_fill_cols] = cand[numeric_fill_cols].fillna(0)
    cand["ca_weeks_since_bought"] = cand["ca_weeks_since_bought"].replace(0, history_weeks)

    return cand
