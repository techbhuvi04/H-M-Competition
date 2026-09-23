# H&M Personalized Fashion Recommendations

A two-stage recommender system (candidate retrieval → LightGBM LambdaRank ranking) for the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle competition. The task: for every customer, predict the 12 articles they are most likely
to buy in the 7 days after the training data ends.

Built with plain **pandas, NumPy, scikit-learn and LightGBM** — no GPU or external helper library required.

## Problem

- **Data:** ~31.8M transactions (2018-09-20 to 2020-09-22), ~1.37M customers, ~105K articles with metadata.
- **Goal:** for every customer, output a ranked list of 12 `article_id`s for the following week.
- **Metric:** Mean Average Precision @ 12 (MAP@12). Ranking order matters, and only customers who
  actually purchase in the test week are scored — but there's no penalty for guessing wrong, so
  every customer should get a full 12 predictions.

## Approach

```
transactions ──► candidate generation ──► feature engineering ──► LightGBM LambdaRank ──► top-12 per customer
```

### 1. Candidate generation (recall) — [`src/hm_reco/candidates.py`](src/hm_reco/candidates.py)
For each customer, candidates are the union of:

| Source | Idea |
|---|---|
| Repurchase | Items the customer bought in the last 3 weeks |
| Item pairs | Items frequently co-purchased (same customer, same day) with the customer's recent items |
| Popular last week | Global best-sellers from the week before the target week (cold-start fallback) |
| Age-bucket popularity | Best-sellers within the customer's age quartile |

### 2. Features — [`src/hm_reco/features.py`](src/hm_reco/features.py)
- **Article:** purchase counts over 1/2/4-week windows (trend), average and most-recent price
- **Customer:** recent purchase count, average spend, weeks since last purchase, age
- **Customer × article:** times bought before, weeks since last bought (the strongest repurchase signal)
- **Article metadata:** category/department/product-group joined from `articles.csv`

All features for a given target week are computed strictly from transactions *before* that week —
no leakage from the label week.

### 3. Ranking — [`src/hm_reco/model.py`](src/hm_reco/model.py)
- `LGBMRanker` with the LambdaRank objective (directly optimizes ranking quality, closer to MAP@12
  than plain binary classification).
- **Validation:** a time-based split — the last available week is held out, and the model is trained
  on the week before it. Random splits would leak future information.
- **Submission:** the final model retrains on the most recent labeled week and predicts one week ahead;
  customers with fewer than 12 candidates are padded with the fallback popular-items list
  ([`src/hm_reco/submission.py`](src/hm_reco/submission.py)).

### Evaluation — [`src/hm_reco/evaluation.py`](src/hm_reco/evaluation.py)
A from-scratch MAP@12 implementation matching the competition's own scoring rule (customers with no
purchases in the scored week are excluded, predictions are deduplicated and capped at 12).

## Repository structure

```
.
├── hm-recommendations.ipynb   # end-to-end driver notebook: load → split → candidates → features → train → validate → predict
├── src/hm_reco/
│   ├── data.py          # CSV loading, dtype downcasting, week indexing
│   ├── candidates.py    # candidate generation (recall)
│   ├── features.py      # feature engineering + training-frame assembly
│   ├── model.py          # LGBMRanker training / scoring
│   ├── evaluation.py    # MAP@12 implementation
│   └── submission.py    # zero-padded, fallback-filled submission.csv writer
├── data/                 # Kaggle CSVs go here (git-ignored, see data/README.md)
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   ├── sample_submission.csv
│   └── README.md
├── outputs/              # submission.csv lands here (git-ignored)
├── requirements.txt
└── README.md
```

## Getting started

```bash
git clone <this-repo-url>
cd <repo>

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download the data into data/ (see data/README.md)
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data
unzip data/h-and-m-personalized-fashion-recommendations.zip -d data

jupyter notebook hm-recommendations.ipynb
```

Runs on a normal laptop (no GPU needed) — full data load is ~31.8M transaction rows and takes about
a minute; candidate/feature generation on the full customer base takes a few minutes per week.
The submission is written to `outputs/submission.csv`.

## Possible improvements
- More retrieval sources: item2vec/ALS embeddings, same-`product_code` colour/size siblings, text or
  image similarity from the product descriptions and garment photos.
- Cross-validate over multiple recent weeks instead of a single holdout, for a more stable score estimate.
- Blend with a second ranker (CatBoost/XGBoost) or tune LightGBM hyperparameters.

## License
The competition data belongs to H&M Group and is covered by the
[competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules);
it is not redistributed here.
