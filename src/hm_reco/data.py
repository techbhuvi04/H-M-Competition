"""Loading and lightly compressing the raw H&M CSVs.

The raw files are large (transactions alone is ~31.8M rows), so we downcast
dtypes on load to keep everything in memory on a normal laptop:
  - customer_id / article_id -> smaller integer codes instead of long strings
  - price -> float32
  - t_dat -> datetime64, plus an integer "week" column counting back from
    the most recent Sunday-aligned week in the data
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Dataset:
    transactions: pd.DataFrame
    customers: pd.DataFrame
    articles: pd.DataFrame
    customer_id_map: pd.Series  # original hex string -> compact int code
    n_weeks: int  # total number of weeks in `transactions.week`


def _week_index(dates: pd.Series) -> pd.Series:
    """Map each date to an integer week number, increasing over time.

    Week boundaries follow the transaction data's own cadence: every 7 days
    starting from the earliest date in the dataset, so week 0 is the first
    week and later weeks are simply offset by whole weeks from it.
    """
    start = dates.min().normalize()
    days_since_start = (dates - start).dt.days
    return (days_since_start // 7).astype("int16")


def load_dataset(data_dir: str) -> Dataset:
    """Load articles.csv, customers.csv and transactions_train.csv from
    `data_dir`, downcast dtypes, and add a `week` column to transactions.
    """
    articles = pd.read_csv(
        os.path.join(data_dir, "articles.csv"),
        dtype={"article_id": "int32"},
    )
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    transactions = pd.read_csv(
        os.path.join(data_dir, "transactions_train.csv"),
        dtype={
            "article_id": "int32",
            "price": "float32",
            "sales_channel_id": "int8",
        },
        parse_dates=["t_dat"],
    )

    # Map long hex customer_id strings to compact int32 codes. This is the
    # single biggest memory win: the raw id is a 64-char hex string, an
    # int32 code is 4 bytes.
    all_ids = pd.Index(customers["customer_id"].unique())
    customer_id_map = pd.Series(np.arange(len(all_ids), dtype="int32"), index=all_ids)

    customers = customers.copy()
    customers["customer_id"] = customers["customer_id"].map(customer_id_map).astype("int32")
    transactions = transactions.copy()
    transactions["customer_id"] = (
        transactions["customer_id"].map(customer_id_map).astype("int32")
    )

    transactions["week"] = _week_index(transactions["t_dat"])
    n_weeks = int(transactions["week"].max()) + 1

    # customer_id_map is stored id -> code; keep the inverse (code -> id)
    # for writing the submission back out with real customer ids.
    inverse_map = pd.Series(all_ids.values, index=customer_id_map.values)

    return Dataset(
        transactions=transactions,
        customers=customers,
        articles=articles,
        customer_id_map=inverse_map,
        n_weeks=n_weeks,
    )
