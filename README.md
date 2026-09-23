# H&M Personalized Fashion Recommendations

A recommender for the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
Kaggle competition. For each of 1.37M customers it predicts the 12 articles they are most likely to buy in the
week after the training data ends.

The pipeline has three stages: **multi-strategy retrieval**, then **user-item interaction features**, then a
**LightGBM binary classifier**. It is written in pandas and LightGBM and runs on a laptop with 8 GB of RAM and no GPU.

## Results

Validation uses the last week of the data (2020-09-16 to 09-22), the week right before the test week. Every customer
who bought something that week (68,984 customers) is scored.

| Model | Validation MAP@12 |
|---|---|
| Top-12 popular items of the previous week | 0.00959 |
| **Retrieval + LightGBM (this repo)** | **0.03626** |

| Candidate stage (validation week) | Value |
|---|---|
| Candidates per customer | ~104 |
| Purchases retrieved | 29,114 of 213,728 (13.6%) |

Retrieval recall sets the upper limit: the ranker can only reorder what retrieval found.

## Approach

```
transactions ──► retrieval (~100 candidates / customer) ──► ~90 features ──► LightGBM ──► top 12
```

The approach follows the ideas in the
[1st place write-up](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/writeups/senkin13-30crmnsia-1st-place-solution)
(senkin13 and 30CrMnSiA): build the training data yourself from past weeks, use several recall strategies
biased towards recent popularity, compute mostly interaction features, downsample negatives, and train a GBDT.
It is scaled down to fit in 8 GB of RAM.

### 1. Retrieval: [`src/hm_reco/retrieval.py`](src/hm_reco/retrieval.py)

| Strategy | Candidates / customer | Idea |
|---|---|---|
| Repurchase | up to 30 | Items the customer bought in the last 2 years, most recent first |
| ItemCF | up to 20 | Items co-purchased with the customer's recent purchases (cosine-normalised, recency-weighted) |
| Siblings | ~5 | Best-selling colour and size variants (same `product_code`) of recent purchases |
| Popular | 50 | Time-weighted best sellers of the previous week |
| Age popular | 50 | Best sellers of the previous week within the customer's age bucket |

Fashion is fast-moving and seasonal, so recent popularity is the strongest single source. On the validation week,
the top 100 popular items alone recover more purchases than all the personalised sources combined. Each
strategy's score (repurchase recency, itemCF score, popularity rank, ...) is kept as a feature.

### 2. Features: [`src/hm_reco/features.py`](src/hm_reco/features.py)

| Type | Examples |
|---|---|
| Count | user-item, user-product-code, user-department/section/product-type/garment-group and item counts over the last week, 4 weeks, 12 weeks, the same week last year and all time; time-weighted counts |
| Time | days since first and last purchase (user, item, user-item, user-category) |
| Mean / max / min | price, buyer age, sales channel |
| Difference / ratio | customer age minus the item's average buyer age, item price vs. customer's average spend, user-item count as a share of the user's and the item's totals, category shares |
| Retrieval | itemCF score, popularity ranks, repurchase recency, number of sources that retrieved the item |

About half of all customers have no purchases in the last 3 months, so all-time cumulative features are included
alongside the recent windows. Every feature for target week *W* is computed only from weeks before *W*.

Features with the most gain: time-weighted count of the customer's purchases of the same product (`upc_tw_cnt`),
item sales last week (`i_cnt_1w`), department, and days since the customer last bought the item (`ui_last_days`).

### 3. Training: [`src/hm_reco/pipeline.py`](src/hm_reco/pipeline.py)

- **Training data:** 6 consecutive target weeks. For each week, candidates are retrieved for that week's buyers
  and labelled 1 if bought.
- **Negative downsampling:** 600k negatives are kept per week, before features are built. This keeps memory
  manageable (about 3.8M training rows).
- **Model:** LightGBM binary classifier (127 leaves, learning rate 0.05), with early stopping on the validation week.
- **Submission:** the target weeks are shifted forward one week and the model is retrained for the best iteration
  found in validation. All customers are then scored in batches. Any customer with fewer than 12 candidates is
  padded with popular items.

## Repository structure

```
.
├── train.py                  # command line: `cv` (validate) and `submit` (write submission.csv)
├── hm-recommendations.ipynb  # step-by-step walkthrough of the same pipeline
├── src/hm_reco/
│   ├── data.py         # CSV loading, parquet cache, compact ids, week indexing
│   ├── retrieval.py    # candidate generation strategies
│   ├── features.py     # interaction feature engineering
│   ├── pipeline.py     # training set assembly, negative sampling, training, batched scoring
│   ├── evaluation.py   # MAP@12
│   └── submission.py   # submission.csv writer (zero-padded article ids)
├── data/               # Kaggle CSVs go here (git-ignored, see data/README.md)
├── outputs/            # logs, saved model, submission.csv (git-ignored)
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
# put the competition CSVs in data/ (see data/README.md)

python train.py cv --weeks 6 --neg 600000                  # ~15 min, prints validation MAP@12 and best iteration
python train.py submit --weeks 6 --neg 600000 --rounds 434 # ~2 h, writes outputs/submission.csv
python train.py cv --weeks 1 --neg 200000 --sample 5000    # ~3 min smoke test
```

The first run parses the CSVs (about 1 minute) and caches them as parquet in `data/`. Timings are from an
8-core laptop with 8 GB of RAM.

## Possible improvements

- **Recall.** 13.6% of purchases are retrieved at about 100 candidates per customer. The 1st-place team retrieved
  roughly 18% at the same budget. Embedding-based retrieval (word2vec item embeddings, graph user embeddings) and
  more tuning of the source mix are the most direct next steps.
- **More data.** 1–2M negatives per week instead of 600k, and more training weeks, given more memory.
- **Ensembling.** Blend several LightGBM seeds with CatBoost.

## Data

The competition data belongs to H&M Group and is covered by the
[competition rules](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/rules).
It is not included in this repository.
