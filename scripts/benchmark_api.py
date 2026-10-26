"""Send N requests to a running API and print latency percentiles.

    python scripts/benchmark_api.py --url http://localhost:8000 --n 100
"""

import argparse
import random
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    users = pd.read_parquet(ROOT / "models/segments.parquet", columns=["user_id"])["user_id"].tolist()
    rng = random.Random(args.seed)
    items = httpx.get(f"{args.url}/recommend/warmup", params={"k": 50}).json()["items"]
    item_ids = [i["parent_asin"] for i in items]

    requests = {
        "recommend (known user)": lambda c: c.get(f"/recommend/{rng.choice(users)}", params={"k": args.k}),
        "recommend (new user)": lambda c: c.get(f"/recommend/new-{rng.random()}", params={"k": args.k}),
        "similar": lambda c: c.get(f"/similar/{rng.choice(item_ids)}", params={"k": args.k}),
        "sentiment": lambda c: c.post("/sentiment", json={"text": "Fun game but the controls are clunky"}),
        "predict-rating": lambda c: c.post("/predict-rating",
                                           json={"user_id": rng.choice(users), "parent_asin": rng.choice(item_ids)}),
    }
    with httpx.Client(base_url=args.url, timeout=10) as client:
        for _ in range(5):
            requests["recommend (known user)"](client)
        print(f"{'endpoint':24} {'n':>4} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
        for name, send in requests.items():
            times = []
            for _ in range(args.n):
                t = time.perf_counter()
                send(client).raise_for_status()
                times.append((time.perf_counter() - t) * 1000)
            p50, p95 = np.percentile(times, [50, 95])
            print(f"{name:24} {args.n:>4} {p50:>8.1f} {p95:>8.1f} {max(times):>8.1f}")


if __name__ == "__main__":
    main()
