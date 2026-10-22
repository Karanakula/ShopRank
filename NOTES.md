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
