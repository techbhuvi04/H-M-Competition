"""End-to-end training / validation / prediction.

Training data: for each of several consecutive target weeks, retrieve
candidates for the customers who bought something that week, label each
candidate 1 if it was bought, downsample the negatives, then build features.
Negatives are downsampled *before* feature building, which is what keeps
memory manageable on a laptop.

Validation: the last week of the data (2020-09-16 .. 09-22) is held out;
candidates are retrieved for every customer who bought that week, scored in
batches, and the top 12 per customer are compared with what they bought.

Submission: the same model setup is retrained with every target week shifted
one week later, then scores every customer for the test week.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import evaluation, features, retrieval
from .data import Dataset

KEYS = ["customer_id", "article_id"]


@dataclass
class Config:
    n_train_weeks: int = 6
    neg_per_week: int = 1_000_000
    predict_batch: int = 25_000
    seed: int = 42
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=100,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
    ))
    num_boost_round: int = 1000
    early_stopping: int = 50


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _history(ds: Dataset, target_week: int) -> pd.DataFrame:
    return ds.transactions[ds.transactions["week"] < target_week]


def _labels(ds: Dataset, target_week: int) -> pd.DataFrame:
    t = ds.transactions[ds.transactions["week"] == target_week][KEYS].drop_duplicates()
    return t.assign(label=np.int8(1))


@dataclass
class Week:
    ctx: retrieval.WeekContext
    item_feats: pd.DataFrame


def prepare_week(ds: Dataset, target_week: int) -> Week:
    hist = _history(ds, target_week)
    end_day = retrieval.week_end_day(target_week, ds.n_weeks)
    ctx = retrieval.week_context(hist, ds.customers, ds.articles, end_day)
    return Week(ctx, features.item_features(hist, ds.customers, ds.articles, end_day))


def featurise(ds: Dataset, week: Week, cand: pd.DataFrame) -> pd.DataFrame:
    return features.build(cand, week.ctx.hist, ds.customers, ds.articles, week.ctx.end_day, week.item_feats)


def training_week(ds: Dataset, target_week: int, cfg: Config) -> pd.DataFrame:
    week = prepare_week(ds, target_week)
    labels = _labels(ds, target_week)
    buyers = ds.customers[ds.customers["customer_id"].isin(labels["customer_id"].unique())]

    cand = retrieval.generate(week.ctx, buyers, ds.articles)
    cand = cand.merge(labels, on=KEYS, how="left")
    cand["label"] = cand["label"].fillna(0).astype("int8")
    hits = int(cand["label"].sum())

    pos = cand[cand["label"] == 1]
    neg = cand[cand["label"] == 0]
    neg = neg.sample(min(cfg.neg_per_week, len(neg)), random_state=cfg.seed)
    cand = pd.concat([pos, neg], ignore_index=True)

    df = featurise(ds, week, cand)
    df["week"] = np.int16(target_week)
    _log(f"week {target_week}: {len(buyers):,} buyers, hits {hits:,}/{len(labels):,} "
         f"({hits / len(labels):.1%} recall), train rows {len(df):,}")
    return df


def training_set(ds: Dataset, last_target_week: int, cfg: Config) -> pd.DataFrame:
    weeks = range(last_target_week - cfg.n_train_weeks + 1, last_target_week + 1)
    frames = []
    for w in weeks:
        frames.append(training_week(ds, w, cfg))
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def _dataset(df: pd.DataFrame, cols: list[str]) -> lgb.Dataset:
    cats = [c for c in features.CATEGORICAL if c in cols]
    return lgb.Dataset(df[cols], label=df["label"], categorical_feature=cats, free_raw_data=True)


def score_customers(
    ds: Dataset,
    model: lgb.Booster,
    target_week: int,
    customers: pd.DataFrame,
    cfg: Config,
    k: int = 12,
) -> dict:
    """Retrieve + featurise + score `customers` in batches; return top-k per customer."""
    week = prepare_week(ds, target_week)
    cols = model.feature_name()
    preds: dict = {}
    for start in range(0, len(customers), cfg.predict_batch):
        batch = customers.iloc[start:start + cfg.predict_batch]
        cand = retrieval.generate(week.ctx, batch, ds.articles)
        df = featurise(ds, week, cand)
        df["score"] = model.predict(df[cols], num_threads=0)
        preds.update(evaluation.predictions_from_scores(df[["customer_id", "article_id", "score"]], k))
        _log(f"  scored {min(start + cfg.predict_batch, len(customers)):,}/{len(customers):,} customers")
        del cand, df
        gc.collect()
    return preds


def popular_fallback(ds: Dataset, target_week: int, k: int = 12) -> list[int]:
    end_day = retrieval.week_end_day(target_week, ds.n_weeks)
    return retrieval.popular(_history(ds, target_week), end_day, top_n=k)["article_id"].tolist()


def fill(preds: dict, customers, fallback: list[int], k: int = 12) -> dict:
    out = {}
    for c in customers:
        p = list(preds.get(c, []))
        for a in fallback:
            if len(p) >= k:
                break
            if a not in p:
                p.append(a)
        out[c] = p[:k]
    return out


def run_cv(ds: Dataset, cfg: Config, sample: int | None = None) -> tuple[lgb.Booster, float]:
    """Train on the `n_train_weeks` before the last week and score the last week.
    `sample` scores only a random subset of validation buyers (for quick runs)."""
    val_week = ds.n_weeks - 1
    _log(f"building training set: target weeks {val_week - cfg.n_train_weeks}..{val_week - 1}")
    train = training_set(ds, val_week - 1, cfg)
    _log("building validation set (downsampled, for early stopping)")
    valid = training_week(ds, val_week, cfg)

    cols = features.feature_columns(train.drop(columns="week"))
    _log(f"training LightGBM on {len(train):,} rows x {len(cols)} features")
    model = lgb.train(
        cfg.lgb_params,
        _dataset(train, cols),
        num_boost_round=cfg.num_boost_round,
        valid_sets=[_dataset(valid, cols)],
        callbacks=[lgb.early_stopping(cfg.early_stopping), lgb.log_evaluation(50)],
    )
    del train, valid
    gc.collect()

    _log("scoring every validation-week buyer")
    labels = _labels(ds, val_week)
    buyers = ds.customers[ds.customers["customer_id"].isin(labels["customer_id"].unique())]
    if sample:
        buyers = buyers.sample(min(sample, len(buyers)), random_state=cfg.seed)
        labels = labels[labels["customer_id"].isin(buyers["customer_id"])]
    preds = score_customers(ds, model, val_week, buyers, cfg)
    preds = fill(preds, buyers["customer_id"], popular_fallback(ds, val_week))
    truth = labels.groupby("customer_id")["article_id"].apply(set).to_dict()
    score = evaluation.mean_average_precision(preds, truth)

    baseline = fill({}, buyers["customer_id"], popular_fallback(ds, val_week))
    _log(f"validation MAP@12: {score:.5f}  (popular-items baseline: "
         f"{evaluation.mean_average_precision(baseline, truth):.5f})")
    return model, score


def run_submission(ds: Dataset, cfg: Config, num_boost_round: int) -> dict:
    """Retrain on the most recent `n_train_weeks` and predict the test week for all customers."""
    test_week = ds.n_weeks
    _log(f"building training set: target weeks {test_week - cfg.n_train_weeks}..{test_week - 1}")
    train = training_set(ds, test_week - 1, cfg)
    cols = features.feature_columns(train.drop(columns="week"))
    _log(f"training final model for {num_boost_round} rounds on {len(train):,} rows")
    model = lgb.train(cfg.lgb_params, _dataset(train, cols), num_boost_round=num_boost_round)
    del train
    gc.collect()

    _log("scoring all customers for the test week")
    preds = score_customers(ds, model, test_week, ds.customers, cfg)
    return fill(preds, ds.customers["customer_id"], popular_fallback(ds, test_week))
