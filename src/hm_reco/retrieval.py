"""Candidate generation (retrieval / recall).

Every strategy looks only at `hist`, the transactions strictly before the
target week, and returns (customer_id, article_id, <score column>). The
score each strategy produces is kept and later used as a ranking feature,
so the model knows *why* an item was retrieved and how strongly.

Strategies:
  repurchase   items the customer bought before, most recent first
  itemcf       item-to-item collaborative filtering on recent co-purchases
  siblings     other colours/sizes (same product_code) of recent purchases
  popular      last week's best sellers, time-weighted
  age_popular  last week's best sellers within the customer's age bucket

Fashion moves fast, so every strategy is biased towards recent weeks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["customer_id", "article_id"]


def week_end_day(target_week: int, n_weeks: int) -> int:
    """`day` value of the last day before `target_week` starts."""
    return (n_weeks - target_week) * 7


def repurchase(hist: pd.DataFrame, end_day: int, max_weeks: int = 104, top_n: int = 30) -> pd.DataFrame:
    recent = hist[hist["day"] < end_day + max_weeks * 7]
    last = recent.groupby(KEYS, as_index=False)["day"].min()
    last["rep_days_ago"] = (last["day"] - end_day).astype("int16")
    last = last.sort_values(["customer_id", "rep_days_ago"])
    last = last.groupby("customer_id").head(top_n)
    return last[KEYS + ["rep_days_ago"]]


def item_similarity(hist: pd.DataFrame, end_day: int, sim_weeks: int = 4, neighbours: int = 20) -> pd.DataFrame:
    """Item-to-item similarity from co-purchases by the same customer within
    the last `sim_weeks`. Co-occurrences from customers who bought many items
    are down-weighted, and scores are normalised by item popularity (cosine)."""
    window = hist[hist["day"] < end_day + sim_weeks * 7][KEYS].drop_duplicates()
    basket = window.groupby("customer_id")["article_id"].transform("size")
    keep = (basket >= 2) & (basket <= 50)
    window = window[keep].assign(w=(1.0 / np.log1p(basket[keep])).astype("float32"))

    pairs = window.merge(window[KEYS], on="customer_id", suffixes=("", "_to"))
    pairs = pairs[pairs["article_id"] != pairs["article_id_to"]]
    sim = pairs.groupby(["article_id", "article_id_to"], as_index=False)["w"].sum()
    item_cnt = window.groupby("article_id").size()
    sim["w"] /= np.sqrt(sim["article_id"].map(item_cnt).to_numpy() * sim["article_id_to"].map(item_cnt).to_numpy())
    return sim.sort_values(["article_id", "w"], ascending=[True, False]).groupby("article_id").head(neighbours)


def itemcf(hist_c: pd.DataFrame, sim: pd.DataFrame, end_day: int, user_weeks: int = 12, top_n: int = 20) -> pd.DataFrame:
    """Items similar to what the customer bought recently (recency-weighted)."""
    user_items = hist_c[hist_c["day"] < end_day + user_weeks * 7]
    user_items = user_items.groupby(KEYS, as_index=False)["day"].min()
    user_items["decay"] = (1.0 / (1.0 + (user_items["day"] - end_day) / 7.0)).astype("float32")

    cand = user_items[KEYS + ["decay"]].merge(sim, on="article_id")
    cand["cf_score"] = cand["decay"] * cand["w"]
    cand = cand.groupby(["customer_id", "article_id_to"], as_index=False)["cf_score"].sum()
    cand = cand.rename(columns={"article_id_to": "article_id"})
    return cand.sort_values(["customer_id", "cf_score"], ascending=[True, False]).groupby("customer_id").head(top_n)


def sibling_sales(hist: pd.DataFrame, articles: pd.DataFrame, end_day: int, per_product: int = 3) -> pd.DataFrame:
    """Top-selling articles of each product_code during the last week."""
    code = articles.set_index("article_id")["product_code"]
    last_week = hist[hist["day"] < end_day + 7]
    sales = last_week.groupby("article_id").size().rename("sib_sales").reset_index()
    sales["product_code"] = sales["article_id"].map(code)
    sales = sales.sort_values(["product_code", "sib_sales"], ascending=[True, False])
    return sales.groupby("product_code").head(per_product)


def siblings(hist_c: pd.DataFrame, sales: pd.DataFrame, articles: pd.DataFrame, end_day: int, user_weeks: int = 4) -> pd.DataFrame:
    """Best-selling variants (other colours/sizes) of products the customer bought recently."""
    code = articles.set_index("article_id")["product_code"]
    user_codes = hist_c[hist_c["day"] < end_day + user_weeks * 7][KEYS].drop_duplicates()
    user_codes = user_codes.assign(product_code=user_codes["article_id"].map(code).to_numpy())
    user_codes = user_codes[["customer_id", "product_code"]].drop_duplicates()
    cand = user_codes.merge(sales, on="product_code")[KEYS + ["sib_sales"]]
    return cand.groupby(KEYS, as_index=False)["sib_sales"].max()


def popular(hist: pd.DataFrame, end_day: int, top_n: int = 50) -> pd.DataFrame:
    """Time-weighted best sellers of the last week (broadcast to all customers)."""
    last_week = hist[hist["day"] < end_day + 7]
    weight = 1.0 / (1.0 + (last_week["day"] - end_day))
    score = weight.groupby(last_week["article_id"]).sum().nlargest(top_n)
    out = score.rename("pop_score").reset_index()
    out["pop_rank"] = np.arange(1, len(out) + 1, dtype="int16")
    return out


def age_bucket(age: pd.Series) -> pd.Series:
    bins = [-1, 19, 24, 29, 34, 44, 54, 200]
    return pd.cut(age.fillna(30), bins=bins, labels=False).astype("int8")


def age_popular(hist: pd.DataFrame, cust: pd.DataFrame, end_day: int, top_n: int = 50) -> pd.DataFrame:
    """Last week's best sellers per age bucket, returned per bucket (merged on later)."""
    last_week = hist[hist["day"] < end_day + 7][["customer_id", "article_id"]]
    buckets = age_bucket(cust.set_index("customer_id")["age"])
    last_week = last_week.assign(age_bucket=last_week["customer_id"].map(buckets).to_numpy())
    counts = last_week.groupby(["age_bucket", "article_id"]).size().rename("agepop_cnt").reset_index()
    counts = counts.sort_values(["age_bucket", "agepop_cnt"], ascending=[True, False])
    counts = counts.groupby("age_bucket").head(top_n)
    counts["agepop_rank"] = counts.groupby("age_bucket").cumcount().astype("int16") + 1
    return counts


@dataclass
class WeekContext:
    """Everything retrieval needs for one target week that doesn't depend on
    which customers are being scored, computed once and reused per batch."""
    hist: pd.DataFrame
    end_day: int
    sim: pd.DataFrame
    sib_sales: pd.DataFrame
    pop: pd.DataFrame
    agepop: pd.DataFrame


def week_context(hist: pd.DataFrame, all_customers: pd.DataFrame, articles: pd.DataFrame, end_day: int) -> WeekContext:
    return WeekContext(
        hist=hist,
        end_day=end_day,
        sim=item_similarity(hist, end_day),
        sib_sales=sibling_sales(hist, articles, end_day),
        pop=popular(hist, end_day),
        agepop=age_popular(hist, all_customers, end_day),
    )


def generate(ctx: WeekContext, customers: pd.DataFrame, articles: pd.DataFrame) -> pd.DataFrame:
    """Union of all strategies for `customers`, with each strategy's score as
    a column (NaN where that strategy didn't retrieve the item)."""
    cust_ids = customers["customer_id"].to_numpy()
    hist_c = ctx.hist[ctx.hist["customer_id"].isin(cust_ids)]

    scored = [
        repurchase(hist_c, ctx.end_day),
        itemcf(hist_c, ctx.sim, ctx.end_day),
        siblings(hist_c, ctx.sib_sales, articles, ctx.end_day),
    ]
    pop_cand = pd.DataFrame({
        "customer_id": np.repeat(cust_ids, len(ctx.pop)),
        "article_id": np.tile(ctx.pop["article_id"].to_numpy(), len(cust_ids)),
    })
    ages = customers[["customer_id"]].assign(age_bucket=age_bucket(customers["age"]).to_numpy())
    agepop_cand = ages.merge(ctx.agepop[["age_bucket", "article_id"]], on="age_bucket")[KEYS]

    cand = pd.concat([p[KEYS] for p in scored] + [pop_cand, agepop_cand], ignore_index=True)
    cand = cand.drop_duplicates(ignore_index=True)
    for p in scored:
        cand = cand.merge(p, on=KEYS, how="left")
    cand = cand.merge(ctx.pop[["article_id", "pop_score", "pop_rank"]], on="article_id", how="left")
    cand = cand.merge(ages, on="customer_id", how="left")
    cand = cand.merge(ctx.agepop[["age_bucket", "article_id", "agepop_rank"]], on=["age_bucket", "article_id"], how="left")
    return cand.drop(columns="age_bucket")
