# H&M Personalized Fashion Recommendations

A two-stage recommender system (candidate retrieval → LightGBM LambdaRank) for the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle competition. The task is to predict the 12 articles each customer will buy in the 7 days after the training data ends.

| Metric | Value |
|---|---|
| Local CV MAP@12 (validation week 104) | **0.03498** |
| Final competition standing of the original solution | **52nd place** |

> **Credit.** This solution was written by **Jacob CP** and shared publicly on Kaggle
> ([notebook discussion](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324076/),
> [code-development notes](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324078)).
> The helper library is his repo [JacobCP/kaggle-handm-helpers](https://github.com/JacobCP/kaggle-handm-helpers),
> included here as a git submodule. It is not copied into this repository.
> This repo reorganises the notebook so it can be studied and run outside Kaggle.

---

## Problem

- **Data:** about 31.8M transactions (Sep 2018 to Sep 2020), about 1.37M customers, and about 105K articles with metadata.
- **Goal:** for every customer, output a ranked list of 12 `article_id`s for the following week.
- **Metric:** Mean Average Precision @ 12. Ranking order matters, and only customers who purchase in the test week are scored.

## Approach

```
transactions ──► candidate generation ──► feature engineering ──► LightGBM LambdaRank ──► top-12 per customer
```

### 1. Candidate generation (recall)
For each customer, candidates are the union of:

| Source | Idea |
|---|---|
| Recent customer purchases | Items the customer bought in the last 3 weeks (repurchase) |
| Customer's last active weeks | Items from the customer's last 12 weeks of activity |
| Item pairs | Items frequently co-purchased with the customer's recent items (5 pairs per item, rebuilt each week to avoid leakage) |
| Popular articles | Last week's top sellers within the customer's preferred department |
| Age-bucket popularity | Top sellers in the customer's age group (4 buckets) |

Candidates are then filtered to articles that sold recently, since fashion is highly seasonal.

### 2. Features (about 40 in total)
- **Customer × article:** last purchase date, rebuy count and ratio, co-purchase pair strength
- **Customer × hierarchy:** the customer's share of purchases in each department, section, index, and product type
- **Article:** popularity lags over 1, 3, 14 and 30 days, recent popularity, sales channel, price and last-week price ratio
- **Customer:** age, sales channel, number of unique purchase days

### 3. Ranking
- `LGBMRanker` (LambdaRank objective), 20 leaves, 150–200 trees, evaluated on MAP@12 and NDCG@12.
- Training data is 5 consecutive weeks stacked together. Each week's features use only data from before that week.
- **Validation:** a time-based split, with week 104 (the last training week, 2020-09-16 to 09-22) held out.
- **Submission:** two models (trained with label weeks 104 and 105) are ensembled. Prediction runs in 2 customer batches to fit in GPU memory.

### Candidate stage diagnostics (validation week)
| | Value |
|---|---|
| Candidate recall | 7.98% |
| Candidate precision | 0.82% |
| Candidate pairs | about 2.1M |

Recall at the candidate stage caps the final score. Adding more retrieval sources is the most direct way to improve.

---

## Repository structure

```
.
├── notebooks/
│   └── hm-recommendations-lgbm-ranker.ipynb   # full pipeline: load → pairs → candidates → features → CV → submission
├── handmhelpers/                              # git submodule: JacobCP/kaggle-handm-helpers @ 86c412e
│   ├── io.py            # data loading and dtypes
│   ├── pairs.py         # item co-purchase pairs
│   ├── candidates.py    # retrieval strategies
│   ├── fe.py            # feature engineering
│   ├── cv.py            # time split, MAP@12, recall reports
│   ├── modeling.py      # LGBMRanker training and prediction
│   └── sub.py           # submission formatting
├── data/                # Kaggle CSVs go here (git-ignored, see data/README.md)
│   ├── articles.csv            # 105K products, metadata
│   ├── customers.csv           # 1.37M customers, metadata
│   ├── transactions_train.csv  # 31.8M purchases, 2018-09-20 → 2020-09-22
│   ├── sample_submission.csv   # submission format example
│   └── README.md
├── outputs/             # submission.csv and saved models (git-ignored)
├── requirements.txt
└── README.md
```

## Getting started

### Option A: Kaggle (easiest)
1. Create a new Kaggle notebook and attach the competition data and the
   [`handmhelpers`](https://www.kaggle.com/datasets/jacob34/handmhelpers) dataset.
2. Turn on a **GPU** accelerator.
3. Upload `notebooks/hm-recommendations-lgbm-ranker.ipynb` and run all cells. The notebook detects Kaggle paths automatically.

### Option B: Local
Requires an **NVIDIA GPU** because the pipeline uses [RAPIDS cuDF](https://docs.rapids.ai/install).

```bash
git clone --recurse-submodules <this-repo-url>
cd <repo>

# Install cuDF by following https://docs.rapids.ai/install (conda recommended), then:
pip install -r requirements.txt

# Download the data into data/ (see data/README.md)
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data
unzip data/h-and-m-personalized-fashion-recommendations.zip -d data

jupyter notebook notebooks/hm-recommendations-lgbm-ranker.ipynb
```

The submission is written to `outputs/submission.csv`.

> The submodule is pinned to commit `86c412e`, the version this notebook was built against.
> Later commits of the helper repo may break the notebook.

### Runtime (Kaggle P100 GPU)
| Step | Time |
|---|---|
| Load data | ~47 s |
| Build item pairs (9 weeks) | ~1 min |
| CV (1 fold, 5 training weeks) | ~3.5 min |
| Full train and predict for submission | ~12 min |

## Possible improvements
- More retrieval sources, such as ALS or item2vec embeddings, same-`product_code` siblings, and text or image similarity. These would raise the 8% recall ceiling.
- Cross-validate over more folds (weeks 101–104) to get a more stable estimate.
- Blend with other rankers (CatBoost, XGBoost).

## License and attribution
The original solution and the `handmhelpers` library belong to Jacob CP. See his Kaggle posts and
[GitHub repo](https://github.com/JacobCP/kaggle-handm-helpers) for terms. The competition data belongs to H&M Group and is
covered by the [competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
It is not redistributed here.
