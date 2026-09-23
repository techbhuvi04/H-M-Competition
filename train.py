"""Run the H&M recommender.

    python train.py cv                   # validate on the last week, print MAP@12
    python train.py submit --rounds 300  # retrain on the latest weeks, write outputs/submission.csv
    python train.py cv --weeks 2 --neg 300000 --sample 20000   # quick run

`--rounds` for `submit` should be the best iteration reported by `cv`.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from hm_reco import data, pipeline, submission  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["cv", "submit"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--weeks", type=int, default=6, help="number of training target weeks")
    parser.add_argument("--neg", type=int, default=1_000_000, help="negative samples kept per training week")
    parser.add_argument("--rounds", type=int, default=None, help="boosting rounds for submit")
    parser.add_argument("--sample", type=int, default=None, help="cv only: score a random sample of validation buyers")
    args = parser.parse_args()

    cfg = pipeline.Config(n_train_weeks=args.weeks, neg_per_week=args.neg)
    ds = data.load_dataset(args.data_dir)
    pipeline._log(f"loaded {len(ds.transactions):,} transactions, {ds.n_weeks} weeks")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "cv":
        model, score = pipeline.run_cv(ds, cfg, sample=args.sample)
        model.save_model(os.path.join(args.out_dir, "model_cv.txt"))
        pipeline._log(f"best iteration: {model.best_iteration} -> use `--rounds {model.best_iteration}` for submit")
    else:
        if args.rounds is None:
            parser.error("submit needs --rounds (the best iteration from `cv`)")
        preds = pipeline.run_submission(ds, cfg, args.rounds)
        sub = submission.build_submission(preds, ds.customers["customer_id"], ds.customer_ids)
        path = os.path.join(args.out_dir, "submission.csv")
        sub.to_csv(path, index=False)
        pipeline._log(f"wrote {path} ({len(sub):,} rows)")


if __name__ == "__main__":
    main()
