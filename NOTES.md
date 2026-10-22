# ShopRank notes

## Phase 1 — Data

Amazon Reviews 2023, Video Games: 4,624,615 reviews, 2,766,656 users, 137,249 products (1998-11 → 2023-09).
Saved to `data/interim/` as Parquet, plus a 10% user sample (276,666 users, 460,834 reviews, `random_state=42`).

## Phase 2 — EDA key findings (10% user sample, `notebooks/02_eda.ipynb`)

- **Ratings are very positive.** 61.2% of reviews are 5★ and 74.6% are ≥4★. "Liked" = rating ≥ 4. A model that always says "liked" is already 74.6% accurate, so accuracy is useless — use ranking metrics (Recall@K, NDCG@K).
- **Strong popularity bias.** The top 1% of products receive 28.8% of reviews, while 43.4% of products have a single review. A most-popular baseline will be hard to beat, and tail items need content features.
- **Most users are one-off.** 72.2% of users wrote one review and only 4.2% wrote ≥5. Only 21.0% of test-period users appear in train, so cold start is the main problem, not a corner case.
- **The user × product table is 99.9973% empty** (density 0.0027%). Dense storage would need 67 GB, so use `scipy.sparse`. 6,822 duplicate (user, product) pairs must be deduplicated.
- **Time split (before 5-core): train < 2022-07-01, val 2022-07-01 → 2022-12-31, test ≥ 2023-01-01** (91.6% / 4.4% / 3.9%). Activity peaks in 2020; December + January hold 23.3% of reviews; 2023 tails off because collection ended in Sept 2023.
- **Product metadata has gaps.** Price is missing for 46.9% of products, description is empty for 35.7%, features for 22.3%, categories for 6.7%, store for 1.8%. Empty lists hide from `isna()`. Plan: impute price by platform median + a `price_missing` flag; fall back to title for text.
- **Outliers exist.** Prices run up to $3,499.99 with 32 items at ≤ $0.01; one user posted 78 reviews in a single day and 435 user-days have ≥10 reviews (bot-like). Plan: clip/log price, flag burst users. Price is nearly uncorrelated with everything (Spearman ≤ 0.17), while review count and helpful votes overlap (0.61–0.65) — keep one popularity count.
- **Category and verified status matter, but weakly.** Platform vs liked: χ² p < 0.001 but Cramér's V = 0.087 (Switch 81.0% liked vs PC 70.2%). Verified buyers (86.3% of reviews) like 75.9% vs 66.0% for unverified (mean 4.10 vs 3.73). Keep both as features; don't filter out unverified reviews.

## Phase 3 — Clean + time split (`notebooks/03_clean_split.ipynb`, full data)

**Split dates: T1 = 2022-07-01 (val starts), T2 = 2023-01-01 (test starts).**
The guide's T1 = 2023-06-01 / T2 = 2023-08-01 leaves only 82 test interactions after the 5-core filter (10 scorable users), because the last months of 2023 are nearly empty.

| Step | Interactions |
|---|---|
| Raw | 4,624,615 |
| Exact duplicates removed | 4,570,966 |
| Keep latest per (user, product) | 4,555,500 |
| From 2018-01-01 | 2,374,766 |
| 5-core | **184,567** (24,243 users, 9,999 products, 83.3% liked) |

| Split | Interactions | Users | Products | Dates | Users to score (warm / cold) |
|---|---|---|---|---|---|
| Train | 149,524 (81.0%) | 22,324 | 8,519 | 2018-01-01 → 2022-06-30 | — |
| Val | 21,327 (11.6%) | 6,939 | 4,696 | 2022-07-01 → 2022-12-31 | 4,549 / 1,495 |
| Test | 13,716 (7.4%) | 4,686 | 3,917 | 2023-01-01 → 2023-09-02 | 2,686 / 1,428 |

- warm = in train and ≥1 liked product in the window (main metric); cold = liked something in the window but has no train history.
- Items: `price` gaps filled with median $27.97 + `price_missing` flag (24.3% of products); `store` gaps → "unknown" (0.2%); `item_text` = title + features + description, lowercased, HTML stripped.
- Metadata `average_rating` / `rating_number` dropped; replaced by `train_rating_count`, `train_avg_rating`, `train_liked_rate` from train reviews only. 14.8% of products have no train reviews (they first appear in val/test).
- The 5-core filter is applied before splitting, so it uses future activity to decide which users/products to keep — standard practice, but note it when comparing to production.

**Leakage rules**
- Every feature uses only data from before the cutoff.
- Fit scalers, encoders, TF-IDF and K-Means on train only; `.transform()` val and test.
- Never use metadata `average_rating` / `rating_number`; use the `train_*` columns.
- Never shuffle rows across time.

Outputs in `data/processed/`: `train.parquet`, `val.parquet`, `test.parquet`, `items.parquet`, `eval_users.parquet`.

## Phase 4 — Evaluation harness + baselines (`notebooks/04_baselines.ipynb`)

- Metrics in `src/shoprank/evaluation/metrics.py`; tests in `tests/test_metrics.py` (`python -m pytest`).
- Hand check: recommended = [A, B, C], liked = {B} → Recall@3 = 1.0, NDCG@3 = 0.6309 = 1/log2(3). ✔
- `evaluate()` removes products the user already bought in train before taking the top K.
- Scored on the validation window (2022-07-01 → 2022-12-31). Coverage = share of the 9,999-product catalog shown to anyone.
- Warm users: 4,549 (median 1 liked val product each, mean 2.37). Cold users: 1,495.

## Results (validation, warm users, K = 10)

| Model | Recall@10 | NDCG@10 | HitRate@10 | Coverage |
|---|---|---|---|---|
| Random | 0.0005 | 0.0004 | 0.0018 | 98.78% |
| Popular (all time) | 0.0127 | 0.0071 | 0.0286 | 0.16% |
| Trending (last 30 days) | 0.0251 | 0.0164 | 0.0517 | 0.13% |

Cold users (no train history), same window: Popular Recall@10 0.0110 / NDCG@10 0.0092; Trending 0.0142 / 0.0112.
Trending (1,454 likes on 1,023 products in June 2022, ties broken by all-time popularity) is the baseline to beat.

## Phase 5 — Retrieval (`notebooks/05_retrieval.ipynb`, code in `src/shoprank/retrieval/`)

- Sparse user × product matrix from train: 22,324 × 9,999, 149,524 filled (0.067%), 1.3 MB vs 0.89 GB dense. Columns cover all 9,999 products so text sources can reach products first seen after T1.
- Sources, all fit on train only: item-KNN (cosine, 50 neighbours), TF-IDF (1–2-grams, min_df 3, 50k features, fit on the 8,519 train products), `all-MiniLM-L6-v2` embeddings (cached in `data/processed/item_embeddings.npy`, 26 s on MPS), trending popularity.
- User profile = average vector of liked train products (464 users with no liked product fall back to all their products).
- Equal quotas waste the strongest source: 25 from each (~91 products) gets 0.1007, *below* trending@100 alone. Giving popularity a bigger share fixes it.

| Retrieval (validation, 4,549 warm users) | Recall | Avg candidates |
|---|---|---|
| Item-KNN @100 | 0.0575 | 100 |
| TF-IDF @100 | 0.0770 | 100 |
| Embeddings @100 | 0.0626 | 100 |
| Trending @100 | 0.1028 | 100 |
| Union 25/25/25/25 | 0.1007 | 91 |
| Union 20/20/10/50 (cf/tfidf/emb/pop) | 0.1112 | 95 |
| **Union 30/30/30/75 — saved** | **0.1477** | 153 |
| Union 50/50/50/50 | 0.1484 | 179 |

- 10% user sample run first: same ordering (trending 0.1270 > TF-IDF 0.0913 > embeddings 0.0644 > item-KNN 0.0519; saved union 0.1767).
- TF-IDF beats embeddings (0.0770 vs 0.0626), likely because product titles carry exact platform and franchise words that bag-of-words matches directly.
- Every source adds products no other source finds (in the saved candidates): pop 744, cf 169, tfidf 157, emb 102.
- `data/processed/candidates_val.parquet`: 696,065 rows, 4,549 users, 124–165 candidates each, columns `cf_score`, `tfidf_score`, `emb_score`, `pop_rank`, `label` (liked in val; 0.20% positive).

## Phase 6 — Supporting models

### A. Sentiment (`notebooks/06a_sentiment.ipynb`)
4–5★ = positive, 1–2★ = negative, 3★ dropped. TF-IDF (1–2-grams, min_df 5, 183k terms, **no stop-word list** — it would delete "not"), fit on 139,115 train reviews; scored on 19,792 validation reviews (90.8% positive).

| Model | Accuracy | F1 pos | F1 neg | Macro F1 | PR-AUC neg |
|---|---|---|---|---|---|
| Always positive | 0.9081 | 0.9518 | 0.0000 | 0.4759 | 0.0919 |
| Multinomial NB | 0.9572 | 0.9767 | 0.7349 | 0.8558 | 0.8305 |
| Logistic Regression (balanced) | 0.9569 | 0.9760 | 0.7892 | 0.8826 | 0.8912 |
| LinearSVC | 0.9683 | 0.9827 | 0.8122 | 0.8975 | 0.8996 |

- LinearSVC is best, but `avg_sentiment` uses Logistic Regression because it gives probabilities.
- Train reviews get **out-of-fold** probabilities (5 folds) so `avg_sentiment` isn't inflated by in-sample fit; 3★ reviews average 0.491 positive.
- `data/processed/item_sentiment.parquet`: `avg_sentiment` for all 9,999 products (1,480 without train reviews → 0.822). Correlation with `train_avg_rating` is 0.873, so the ranker may find it largely redundant.
- Caveat: Amazon auto-titles like "Five Stars" / "Two Stars" leak the label into the text and inflate the scores somewhat.
- Stretch goal (DistilBERT) not done.

### B. Rating regression from scratch (`notebooks/06b_rating.ipynb`)
Features: user avg rating, product avg rating, price, `avg_sentiment`. Train rows use **leave-one-out** averages (a review's own rating/sentiment never feeds its features); validation rows use train averages. Standardized with a train-fit `StandardScaler` (on raw features lr=0.01 diverges to 1e220 because of price).

- Learning rates (1,000 steps): 0.001 still descending (MSE 1.417), 0.01 converges (1.064), 0.1 converges in ~20 steps (1.064), 1.0 diverges. Plot: `reports/figures/rating_gd_lr.png`.

| Model (validation, clipped to 1–5) | RMSE | MAE |
|---|---|---|
| Global mean | 1.0907 | 0.8390 |
| **GD from scratch (lr=0.1)** | **1.0696** | **0.7877** |
| sklearn LinearRegression | 1.0696 | 0.7877 |
| Ridge | 1.0696 | 0.7877 |
| GradientBoostingRegressor | 1.0671 | 0.7904 |

- GD weights equal sklearn's to 4 decimals. Gains over the global mean are small: user average dominates (weight 0.45), price barely matters (0.01).
- Saved to `models/rating_gd.joblib` (scaler + weights) for the dashboard's "expected rating".

### C. User segments (`notebooks/06c_segments.ipynb`)
Features per train user as of T1: purchases, days since last purchase, avg known price paid, avg rating given, verified share, distinct stores. `log1p` on the long-tailed four, then `StandardScaler`, K-Means k = 2…10.
- Silhouette ≈ 0.18–0.22 for every k (overlapping groups, no clear elbow). Best in 4–6 is **k = 6** (0.207).

| Segment | Users | Purchases | Days since last | Avg price | Avg rating | Verified |
|---|---|---|---|---|---|---|
| Budget buyers, lapsed | 8,061 | 5.8 | 661 | $30 | 4.62 | 97% |
| Premium buyers, lapsed | 4,935 | 5.9 | 635 | $108 | 4.60 | 96% |
| Frequent multi-store buyers | 3,039 | 13.6 | 375 | $52 | 4.46 | 94% |
| Critical raters | 2,925 | 5.3 | 361 | $61 | 2.90 | 90% |
| Light buyers | 1,849 | 2.4 | 413 | $39 | 4.54 | 95% |
| Unverified reviewers | 1,515 | 8.1 | 376 | $61 | 4.16 | 15% |

- PCA: PC1 (32%) = activity (purchases, stores), PC2 (21%) = rating + verified + recency. Plot: `reports/figures/segments_pca.png`.
- `data/processed/user_segments.parquet`: `segment_id`, `segment_name` and the six features for 22,324 train users. Users not in train → `segment_id = -1` in Phase 7.

## Phase 7 — Ranker data (`notebooks/07_ranker_data.ipynb`, code in `src/shoprank/ranking/`)

Both copies come from one function, `build_snapshot(interactions, cutoff, label_end)`, which refits item-KNN, TF-IDF, embedding profiles and trending on the history before its cutoff and recomputes every feature from that history. Phase 6 models fit on train only (sentiment classifier → `models/sentiment_lr.joblib`, segment scaler + K-Means → `models/segments.joblib`) are reused with `.predict()`.

| Copy | Features from | Labels from | Rows | Users | Positive rate |
|---|---|---|---|---|---|
| `ranker_train.parquet` | before 2022-07-01 | 2022-07-01 → 2022-12-31 | 174,135 | 1,141 | **0.81%** |
| `ranker_test.parquet` | before 2023-01-01 | 2023-01-01 → end | 571,098 | 3,719 | **0.22%** |

- Users scored = seen before the cutoff and liked ≥1 product in the label window. For the test copy "seen" means train **or** validation, so it has 3,719 users (vs 2,686 when counting train only in Phase 3).
- Candidate recall: train copy 0.1477 (matches Phase 5 exactly), test copy 0.1417.
- Only 25% (train) / 26% (test) of users have any liked product among their candidates. The training copy drops the other 3,408 users (696,065 → 174,135 rows); the test copy keeps them, so test Recall@K can never exceed ~0.14.
- Val/test review sentiment comes from the train-fit classifier; train reviews keep out-of-fold scores.
- **Leakage test** (`tests/test_leakage.py`): rebuilds the T1 copy with every row on/after T1 scrambled (ratings, sentiment, verified flag, plus 2,000 fake future purchases). All 18 features are identical for matching (user, product) rows; labels change. Sanity check: deliberately computing product stats from all rows makes the test fail on `item_review_count`.

**Feature selection (training copy)**
- Correlation heatmap (`reports/figures/ranker_feature_corr.png`): no pair above |r| 0.9. Strongest: item_avg_rating vs item_avg_sentiment 0.857, price vs price_minus_user_avg 0.784, tfidf vs emb 0.735.
- Pearson with label: all |r| < 0.05 (0.8% positives); strongest bought_store_before 0.045, item_review_count 0.036, tfidf/cf/emb ≈ 0.023.
- Chi-square: 26 of 51 store columns and 2 of 6 segment columns significant at p < 0.05 (PlayStation store χ² = 144).

**Final features (17):** n_purchases, avg_rating_given, days_since_last, segment_id · item_review_count, item_avg_rating, item_avg_sentiment, price, price_missing, store, item_days_since_first · cf_score, tfidf_score, emb_score, pop_rank, price_minus_user_avg, bought_store_before.
- Dropped `avg_price_paid`: exactly price − price_minus_user_avg (the three columns have rank 2) and the weakest label correlation (−0.0001).
- Weak-screen features (item_avg_rating, item_avg_sentiment, days_since_last) are kept for the Phase 8 importance/SHAP recheck.
- Preprocessing: `shoprank.ranking.preprocess.make_preprocessor()` — StandardScaler on numeric, one-hot segment_id and the top 50 stores (rest → "infrequent"); fit on the training copy only. 1.8% of test rows have a store never seen in train.
