"""Feature engineering for (customer, article) candidate pairs.

Almost all features are user-item *interaction* features, which is where
the signal is in this competition. Following the categories that worked
best in practice:

  Count          user-item / user-category / item counts over the last week,
                 month (4w), season (12w), same week last year and all time,
                 plus a time-weighted count
  Time           days since first / last purchase
  Mean/Max/Min   aggregations of price, age and sales channel
  Difference /   customer age vs. the item's average buyer age, customer
  Ratio          spend vs. item price, share of a user's purchases in a category
  Retrieval      the scores each retrieval strategy produced (itemCF
                 similarity, popularity rank, repurchase recency, ...)

About half of all customers have no purchases in the last 3 months, so the
all-time cumulative features matter as much as the recent-window ones.

All features for target week W are computed from `hist` (weeks < W) only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ARTICLE_CATEGORICALS

KEYS = ["customer_id", "article_id"]
WINDOWS = {"1w": 7, "4w": 28, "12w": 84}


def _window_flags(hist: pd.DataFrame, end_day: int) -> pd.DataFrame:
    """Add 0/1 columns marking which recency windows each transaction falls in,
    plus a time-decay weight, so windowed counts are one groupby-sum."""
    rel = (hist["day"] - end_day).astype("int16")
    out = hist.assign(rel=rel)
    for name, days in WINDOWS.items():
        out[f"in_{name}"] = (rel < days).astype("int8")
    out["tw"] = (1.0 / (1.0 + rel / 7.0)).astype("float32")
    return out


def _f32(frame: pd.DataFrame) -> pd.DataFrame:
    for c in frame.columns:
        if frame[c].dtype == "float64":
            frame[c] = frame[c].astype("float32")
    return frame


def item_features(hist: pd.DataFrame, customers: pd.DataFrame, articles: pd.DataFrame, end_day: int) -> pd.DataFrame:
    h = _window_flags(hist, end_day)
    g = h.groupby("article_id")
    f = pd.DataFrame({
        "i_cnt_all": g.size(),
        "i_cnt_1w": g["in_1w"].sum(),
        "i_cnt_4w": g["in_4w"].sum(),
        "i_cnt_12w": g["in_12w"].sum(),
        "i_tw_cnt": g["tw"].sum(),
        "i_first_days": g["rel"].max(),
        "i_last_days": g["rel"].min(),
        "i_price_mean": g["price"].mean(),
        "i_price_max": g["price"].max(),
        "i_price_min": g["price"].min(),
        "i_channel_mean": g["sales_channel_id"].mean(),
    })
    # same week last year: seasonality signal
    ly = h[(h["rel"] >= 52 * 7 - 7) & (h["rel"] < 52 * 7)]
    f["i_cnt_ly"] = ly.groupby("article_id").size()

    recent = h[h["in_4w"] == 1]
    rg = recent.groupby("article_id")
    f["i_users_4w"] = rg["customer_id"].nunique()
    f["i_price_4w"] = rg["price"].mean()
    ages = customers.set_index("customer_id")["age"]
    recent_age = recent["customer_id"].map(ages)
    f["i_age_mean_4w"] = recent_age.groupby(recent["article_id"]).mean()
    f["i_age_std_4w"] = recent_age.groupby(recent["article_id"]).std()

    last_week = h[h["in_1w"] == 1]
    f["i_price_1w"] = last_week.groupby("article_id")["price"].mean()

    f["i_trend"] = f["i_cnt_1w"] / (f["i_cnt_4w"] / 4 + 1)
    f["i_price_drop"] = f["i_price_1w"] / f["i_price_4w"]
    f[["i_cnt_ly", "i_users_4w"]] = f[["i_cnt_ly", "i_users_4w"]].fillna(0)

    f = f.reset_index()
    pc = articles.set_index("article_id")["product_code"]
    f["product_code"] = f["article_id"].map(pc)
    pg = f.groupby("product_code")
    f["pc_cnt_1w"] = pg["i_cnt_1w"].transform("sum")
    f["pc_cnt_4w"] = pg["i_cnt_4w"].transform("sum")
    return _f32(f.drop(columns="product_code"))


def user_features(hist_c: pd.DataFrame, customers: pd.DataFrame, end_day: int) -> pd.DataFrame:
    h = _window_flags(hist_c, end_day)
    g = h.groupby("customer_id")
    f = pd.DataFrame({
        "u_cnt_all": g.size(),
        "u_cnt_1w": g["in_1w"].sum(),
        "u_cnt_4w": g["in_4w"].sum(),
        "u_cnt_12w": g["in_12w"].sum(),
        "u_first_days": g["rel"].max(),
        "u_last_days": g["rel"].min(),
        "u_active_days": g["day"].nunique(),
        "u_price_mean": g["price"].mean(),
        "u_price_max": g["price"].max(),
        "u_price_min": g["price"].min(),
        "u_channel_mean": g["sales_channel_id"].mean(),
        "u_n_items": g["article_id"].nunique(),
    }).reset_index()
    cols = ["customer_id", "age", "FN", "Active", "club_member_status", "fashion_news_frequency"]
    out = customers[cols].merge(f, on="customer_id", how="left")
    for c in ["u_cnt_all", "u_cnt_1w", "u_cnt_4w", "u_cnt_12w", "u_active_days", "u_n_items"]:
        out[c] = out[c].fillna(0)
    return _f32(out)


def user_item_features(hist_c: pd.DataFrame, end_day: int) -> pd.DataFrame:
    h = _window_flags(hist_c, end_day)
    g = h.groupby(KEYS)
    f = pd.DataFrame({
        "ui_cnt_all": g.size(),
        "ui_cnt_1w": g["in_1w"].sum(),
        "ui_cnt_4w": g["in_4w"].sum(),
        "ui_cnt_12w": g["in_12w"].sum(),
        "ui_tw_cnt": g["tw"].sum(),
        "ui_first_days": g["rel"].max(),
        "ui_last_days": g["rel"].min(),
    }).reset_index()
    return _f32(f)


def user_group_features(hist_c: pd.DataFrame, articles: pd.DataFrame, end_day: int, col: str, prefix: str) -> pd.DataFrame:
    """Counts of the customer's purchases within one article grouping
    (product_code / product_type / department / ...)."""
    h = _window_flags(hist_c, end_day)
    h[col] = h["article_id"].map(articles.set_index("article_id")[col])
    g = h.groupby(["customer_id", col])
    f = pd.DataFrame({
        f"{prefix}_cnt_all": g.size(),
        f"{prefix}_cnt_4w": g["in_4w"].sum(),
        f"{prefix}_tw_cnt": g["tw"].sum(),
        f"{prefix}_last_days": g["rel"].min(),
    }).reset_index()
    return _f32(f)


GROUP_FEATURES = {
    "product_code": "upc",
    "product_type_no": "upt",
    "department_no": "udp",
    "section_no": "usc",
    "garment_group_no": "ugg",
}


def build(
    cand: pd.DataFrame,
    hist: pd.DataFrame,
    customers: pd.DataFrame,
    articles: pd.DataFrame,
    end_day: int,
    item_feats: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach every feature to the candidate pairs in `cand`. `item_feats`
    (from `item_features`) can be passed in to avoid recomputing it per batch."""
    cust_ids = cand["customer_id"].unique()
    hist_c = hist[hist["customer_id"].isin(cust_ids)]

    df = cand.merge(articles, on="article_id", how="left")
    if item_feats is None:
        item_feats = item_features(hist, customers, articles, end_day)
    df = df.merge(item_feats, on="article_id", how="left")
    df = df.merge(user_features(hist_c, customers[customers["customer_id"].isin(cust_ids)], end_day),
                  on="customer_id", how="left")
    df = df.merge(user_item_features(hist_c, end_day), on=KEYS, how="left")
    for col, prefix in GROUP_FEATURES.items():
        df = df.merge(user_group_features(hist_c, articles, end_day, col, prefix),
                      on=["customer_id", col], how="left")

    count_cols = [c for c in df.columns if "_cnt" in c and c.startswith(("ui_", "up", "ud", "us", "ug"))]
    df[count_cols] = df[count_cols].fillna(0)

    # difference / ratio features
    df["age_diff"] = df["age"] - df["i_age_mean_4w"]
    df["age_z"] = df["age_diff"] / (df["i_age_std_4w"] + 1)
    df["price_ratio"] = df["i_price_1w"].fillna(df["i_price_mean"]) / df["u_price_mean"]
    df["ui_share_of_item"] = df["ui_cnt_all"] / (df["i_cnt_all"] + 1)
    df["ui_share_of_user"] = df["ui_cnt_all"] / (df["u_cnt_all"] + 1)
    for prefix in GROUP_FEATURES.values():
        df[f"{prefix}_share"] = df[f"{prefix}_cnt_all"] / (df["u_cnt_all"] + 1)
    df["n_sources"] = df[["rep_days_ago", "cf_score", "sib_sales", "pop_rank", "agepop_rank"]].notna().sum(axis=1)

    return _f32(df)


FEATURE_EXCLUDE = {"customer_id", "article_id", "label", "product_code"}
CATEGORICAL = [c for c in ARTICLE_CATEGORICALS] + ["club_member_status", "fashion_news_frequency"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]
