# ShopRank notes

## Phase 1 — Data

Amazon Reviews 2023, Video Games: 4,624,615 reviews, 2,766,656 users, 137,249 products (1998-11 → 2023-09).
Saved to `data/interim/` as Parquet, plus a 10% user sample (276,666 users, 460,834 reviews, `random_state=42`).

## Phase 2 — EDA key findings (10% user sample, `notebooks/02_eda.ipynb`)

- **Ratings are very positive.** 61.2% of reviews are 5★ and 74.6% are ≥4★. "Liked" = rating ≥ 4. A model that always says "liked" is already 74.6% accurate, so accuracy is useless — use ranking metrics (Recall@K, NDCG@K).
- **Strong popularity bias.** The top 1% of products receive 28.8% of reviews, while 43.4% of products have a single review. A most-popular baseline will be hard to beat, and tail items need content features.
- **Most users are one-off.** 72.2% of users wrote one review and only 4.2% wrote ≥5. Only 21.0% of test-period users appear in train, so cold start is the main problem, not a corner case.
- **The user × product table is 99.9973% empty** (density 0.0027%). Dense storage would need 67 GB, so use `scipy.sparse`. 6,822 duplicate (user, product) pairs must be deduplicated.
- **Time split: train < 2022-07-01, val 2022-07-01 → 2022-12-31, test ≥ 2023-01-01** (91.6% / 4.4% / 3.9%). Activity peaks in 2020; December + January hold 23.3% of reviews; 2023 tails off because collection ended in Sept 2023.
- **Product metadata has gaps.** Price is missing for 46.9% of products, description is empty for 35.7%, features for 22.3%, categories for 6.7%, store for 1.8%. Empty lists hide from `isna()`. Plan: impute price by platform median + a `price_missing` flag; fall back to title for text.
- **Outliers exist.** Prices run up to $3,499.99 with 32 items at ≤ $0.01; one user posted 78 reviews in a single day and 435 user-days have ≥10 reviews (bot-like). Plan: clip/log price, flag burst users. Price is nearly uncorrelated with everything (Spearman ≤ 0.17), while review count and helpful votes overlap (0.61–0.65) — keep one popularity count.
- **Category and verified status matter, but weakly.** Platform vs liked: χ² p < 0.001 but Cramér's V = 0.087 (Switch 81.0% liked vs PC 70.2%). Verified buyers (86.3% of reviews) like 75.9% vs 66.0% for unverified (mean 4.10 vs 3.73). Keep both as features; don't filter out unverified reviews.
