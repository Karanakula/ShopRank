"""Model quality gate: the checks a trained ranker must pass before it is allowed to ship."""


def quality_gate(metadata: dict, min_ndcg: float, min_roc_auc: float, k: int = 10) -> list[str]:
    """Reasons to reject the model described by `metadata.json`; an empty list means it passes.

    It must reach `min_ndcg` and `min_roc_auc` on the test window and beat every baseline
    (trending popularity and each retrieval score alone) on NDCG@k.
    """
    ndcg = f"ndcg@{k}"
    test = metadata.get("test_metrics") or {}
    if ndcg not in test:
        return [f"metadata has no test {ndcg}"]
    failures = []
    if test[ndcg] < min_ndcg:
        failures.append(f"test {ndcg} {test[ndcg]:.4f} < {min_ndcg}")
    if test.get("roc_auc", 0) < min_roc_auc:
        failures.append(f"test roc_auc {test.get('roc_auc', 0):.4f} < {min_roc_auc}")
    for name, base in (metadata.get("baselines") or {}).items():
        if test[ndcg] <= base[ndcg]:
            failures.append(f"test {ndcg} {test[ndcg]:.4f} does not beat {name} ({base[ndcg]:.4f})")
    if not metadata.get("baselines"):
        failures.append("metadata has no baselines to compare against")
    return failures
