# Data

The dataset is not included in this repo because of its size (about 30 GB with images) and the competition rules.

Download it from the [competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
or with the Kaggle CLI:

```bash
kaggle competitions download -c h-and-m-personalized-fashion-recommendations -p data
unzip data/h-and-m-personalized-fashion-recommendations.zip -d data
```

Expected layout:

```
data/
├── articles.csv
├── customers.csv
├── transactions_train.csv
└── sample_submission.csv   # format example only, not used by the pipeline
```

The pipeline reads these three:

| File | Rows | Description |
|---|---|---|
| `transactions_train.csv` | ~31.8M | `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id` |
| `customers.csv` | ~1.37M | customer metadata (age, club status, news frequency, …) |
| `articles.csv` | ~105K | product metadata (type, colour, department, section, description, …) |

Note: `article_id` has a leading zero (e.g. `0706016001`). Keep it as a string when writing submissions.
