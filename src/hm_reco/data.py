"""Loading the raw H&M CSVs into compact, memory-friendly DataFrames.

The first load parses the CSVs (~1 min) and caches them as parquet in the
data directory; later loads read the parquet files in a few seconds.

Conventions used everywhere else in the package:
  - customer_id is a compact int32 code (Dataset.customer_ids maps it back
    to the original hex string for the submission)
  - article_id is int32 (zero-padded back to 10 characters on output)
  - `day` counts days back from the last transaction date (0 = 2020-09-22)
  - `week` counts 7-day blocks forward in time, aligned so the last week of
    the data (2020-09-16 .. 2020-09-22) is a full week. The test week
    (2020-09-23 .. 2020-09-29) is therefore `n_weeks`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ARTICLE_CATEGORICALS = [
    "product_type_no", "product_group_name", "graphical_appearance_no",
    "colour_group_code", "perceived_colour_value_id", "department_no",
    "index_code", "index_group_no", "section_no", "garment_group_no",
]


@dataclass
class Dataset:
    transactions: pd.DataFrame
    customers: pd.DataFrame
    articles: pd.DataFrame
    customer_ids: np.ndarray  # code -> original hex customer_id
    n_weeks: int


def _encode(series: pd.Series) -> pd.Series:
    return series.astype("category").cat.codes.astype("int16")


def _parse_csvs(data_dir: str):
    articles = pd.read_csv(os.path.join(data_dir, "articles.csv"), dtype={"article_id": "int32"})
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    transactions = pd.read_csv(
        os.path.join(data_dir, "transactions_train.csv"),
        dtype={"article_id": "int32", "price": "float32", "sales_channel_id": "int8"},
        parse_dates=["t_dat"],
    )

    customer_ids = customers["customer_id"].to_numpy()
    code_of = pd.Series(np.arange(len(customer_ids), dtype="int32"), index=customer_ids)
    customers["customer_id"] = np.arange(len(customer_ids), dtype="int32")
    transactions["customer_id"] = transactions["customer_id"].map(code_of).astype("int32")

    customers["age"] = customers["age"].astype("float32")
    for col in ["FN", "Active"]:
        customers[col] = customers[col].fillna(0).astype("int8")
    for col in ["club_member_status", "fashion_news_frequency", "postal_code"]:
        customers[col] = _encode(customers[col].fillna("NONE")) if col != "postal_code" else (
            customers[col].astype("category").cat.codes.astype("int32")
        )

    articles["product_code"] = articles["product_code"].astype("int32")
    for col in ARTICLE_CATEGORICALS:
        if articles[col].dtype == object:
            articles[col] = _encode(articles[col])
        else:
            articles[col] = articles[col].astype("int32")
    articles = articles[["article_id", "product_code"] + ARTICLE_CATEGORICALS]

    last_date = transactions["t_dat"].max()
    transactions["day"] = (last_date - transactions["t_dat"]).dt.days.astype("int16")
    transactions = transactions.drop(columns="t_dat")
    transactions = transactions.sort_values(["day", "customer_id"], ascending=[False, True], ignore_index=True)

    return transactions, customers, articles, pd.Series(customer_ids, name="customer_id")


def load_dataset(data_dir: str) -> Dataset:
    cache = {name: os.path.join(data_dir, f"{name}.parquet")
             for name in ["transactions", "customers", "articles", "customer_ids"]}

    if all(os.path.exists(p) for p in cache.values()):
        transactions = pd.read_parquet(cache["transactions"])
        customers = pd.read_parquet(cache["customers"])
        articles = pd.read_parquet(cache["articles"])
        customer_ids = pd.read_parquet(cache["customer_ids"])["customer_id"]
    else:
        transactions, customers, articles, customer_ids = _parse_csvs(data_dir)
        transactions.to_parquet(cache["transactions"], index=False)
        customers.to_parquet(cache["customers"], index=False)
        articles.to_parquet(cache["articles"], index=False)
        customer_ids.to_frame().to_parquet(cache["customer_ids"], index=False)

    # consecutive 0..n-1 codes, which LightGBM's categorical handling expects
    for col in ARTICLE_CATEGORICALS:
        articles[col] = articles[col].astype("category").cat.codes.astype("int16")

    max_day = int(transactions["day"].max())
    n_weeks = max_day // 7 + 1
    transactions["week"] = (n_weeks - 1 - transactions["day"] // 7).astype("int16")

    return Dataset(
        transactions=transactions,
        customers=customers,
        articles=articles,
        customer_ids=customer_ids.to_numpy(),
        n_weeks=n_weeks,
    )
