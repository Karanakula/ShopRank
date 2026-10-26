"""Pick ~20 demo users for the dashboard and save them to app/demo_users.json.

From every segment, among users with a readable history (5-40 purchases), keep those whose top 10
differs most from the popularity baseline, so the side-by-side comparison is always visible.

    python scripts/pick_demo_users.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.service import ShopRankService  # noqa: E402

PER_SEGMENT = {"Budget buyers, lapsed": 4, "Premium buyers, lapsed": 4, "Frequent multi-store buyers": 4,
               "Unverified reviewers": 3, "Critical raters": 3, "Light buyers": 2}
POOL = 60


def main() -> None:
    service = ShopRankService(ROOT / "models")
    users = pd.read_parquet(ROOT / "models" / "segments.parquet")
    readable = users[users["n_purchases"].between(5, 40)]

    picked = []
    for segment, n in PER_SEGMENT.items():
        pool = readable[readable["segment_name"] == segment]
        if len(pool) < n:
            pool = users[users["segment_name"] == segment]
        pool = pool.sample(min(POOL, len(pool)), random_state=42)
        scored = []
        for user in pool.itertuples():
            recs = {i["parent_asin"] for i in service.recommend(user.user_id, 10)["items"]}
            popular = {i["parent_asin"] for i in service.popular(user.user_id, 10)["items"]}
            top = service.recommend(user.user_id, 1)["items"][0]
            scored.append((len(recs & popular), -top["score"], user))
        for overlap, _, user in sorted(scored, key=lambda s: s[:2])[:n]:
            history = service.history(user.user_id)["items"]
            stores = pd.Series([h["store"] for h in history if h["store"] != "unknown"]).value_counts()
            label = f"{segment} · {user.n_purchases} purchases"
            if len(stores):
                label += f" · mostly {stores.index[0]}"
            picked.append({"user_id": user.user_id, "segment": segment, "n_purchases": int(user.n_purchases),
                           "overlap_with_popular": int(overlap), "label": label})

    out = ROOT / "app" / "demo_users.json"
    out.write_text(json.dumps(picked, indent=2) + "\n")
    print(f"wrote {len(picked)} demo users to {out}")
    print(pd.DataFrame(picked)[["segment", "n_purchases", "overlap_with_popular", "label"]].to_string(index=False))


if __name__ == "__main__":
    main()
