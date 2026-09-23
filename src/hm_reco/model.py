"""Training and inference for the LightGBM ranking model.

We use LGBMRanker with the LambdaRank objective: for each customer, the
model learns to order their candidates so that actual purchases sort above
non-purchases, which is a closer match to MAP@12 than plain binary
classification.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

ID_COLUMNS = ("customer_id", "article_id", "label")
CATEGORICAL_COLUMNS = ("index_code", "product_group_name", "department_no")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ID_COLUMNS]


def _group_sizes(frame: pd.DataFrame) -> np.ndarray:
    """LGBMRanker needs each customer's candidates grouped contiguously,
    with the group sizes passed alongside the data (not a group column).
    """
    return frame.groupby("customer_id", sort=False).size().to_numpy()


def train_ranker(
    train_frame: pd.DataFrame,
    *,
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
    n_estimators: int = 200,
    num_leaves: int = 20,
    eval_frame: pd.DataFrame | None = None,
    early_stopping_rounds: int | None = 20,
) -> LGBMRanker:
    """Train an LGBMRanker on `train_frame` (must have customer_id sorted
    contiguously, e.g. via `.sort_values("customer_id")`, and a `label`
    column). Optionally validate on `eval_frame` of the same shape.
    """
    train_frame = train_frame.sort_values("customer_id").reset_index(drop=True)
    features = feature_columns(train_frame)
    cat_cols = [c for c in categorical_columns if c in features]
    for c in cat_cols:
        train_frame[c] = train_frame[c].astype("category")

    model = LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        num_leaves=num_leaves,
    )

    fit_kwargs = dict(
        X=train_frame[features],
        y=train_frame["label"],
        group=_group_sizes(train_frame),
        categorical_feature=cat_cols or "auto",
    )

    if eval_frame is not None:
        eval_frame = eval_frame.sort_values("customer_id").reset_index(drop=True)
        for c in cat_cols:
            eval_frame[c] = eval_frame[c].astype("category")
        fit_kwargs["eval_set"] = [(eval_frame[features], eval_frame["label"])]
        fit_kwargs["eval_group"] = [_group_sizes(eval_frame)]
        fit_kwargs["eval_at"] = [12]
        if early_stopping_rounds:
            from lightgbm import early_stopping, log_evaluation

            fit_kwargs["callbacks"] = [
                early_stopping(early_stopping_rounds),
                log_evaluation(10),
            ]

    model.fit(**fit_kwargs)
    return model


def score_candidates(model: LGBMRanker, frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame`'s (customer_id, article_id) with an added `score`
    column from the trained model, ready for `evaluation.predictions_from_scores`.
    """
    features = feature_columns(frame)
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in features]
    scored = frame.copy()
    for c in cat_cols:
        scored[c] = scored[c].astype("category")
    scored["score"] = model.predict(scored[features])
    return scored[["customer_id", "article_id", "score"]]
