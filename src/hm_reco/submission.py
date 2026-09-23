"""Writing submission.csv.

The competition needs a row for every customer in customers.csv, with
article ids written as zero-padded 10-character strings ("0706016001").
Predictions are expected to be already padded to 12 items by `pipeline.fill`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_submission(predictions: dict, customer_codes: pd.Series, customer_ids: np.ndarray) -> pd.DataFrame:
    """predictions: {customer_code: [article_id, ...]}
    customer_codes: every customer code to write (customers["customer_id"])
    customer_ids: code -> original hex id (Dataset.customer_ids)
    """
    codes = customer_codes.to_numpy()
    rows = [" ".join(f"{a:010d}" for a in predictions.get(c, [])) for c in codes]
    return pd.DataFrame({"customer_id": customer_ids[codes], "prediction": rows})
