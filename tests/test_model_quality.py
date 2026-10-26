"""Model quality gate: the saved model in `models/metadata.json` is still good enough to ship."""

import json

import pytest

from conftest import REAL_MODELS, ROOT
from shoprank.config import load_config
from shoprank.evaluation.gate import quality_gate

GATE = load_config(ROOT / "config.yaml")["quality_gate"]
GOOD = {"test_metrics": {"ndcg@10": 0.032, "roc_auc": 0.78},
        "baselines": {"trending": {"ndcg@10": 0.017}, "sorted_by_cf_score": {"ndcg@10": 0.011}}}


def test_gate_passes_a_good_model():
    assert quality_gate(GOOD, GATE["min_ndcg"], GATE["min_roc_auc"]) == []


@pytest.mark.parametrize("change, reason", [
    ({"test_metrics": {"ndcg@10": 0.016, "roc_auc": 0.78}}, "does not beat trending"),
    ({"test_metrics": {"ndcg@10": 0.020, "roc_auc": 0.78}, "baselines": {}}, "<"),
    ({"test_metrics": {"ndcg@10": 0.032, "roc_auc": 0.60}}, "roc_auc"),
    ({"test_metrics": {}}, "no test ndcg@10"),
])
def test_gate_rejects_a_bad_model(change, reason):
    failures = quality_gate(GOOD | change, GATE["min_ndcg"], GATE["min_roc_auc"])
    assert any(reason in f for f in failures), failures


@pytest.mark.skipif(not (REAL_MODELS / "metadata.json").exists(), reason="run `make train` first")
def test_saved_model_beats_popularity_baseline():
    meta = json.loads((REAL_MODELS / "metadata.json").read_text())
    assert meta["test_metrics"]["ndcg@10"] > meta["baselines"]["trending"]["ndcg@10"]
    assert quality_gate(meta, GATE["min_ndcg"], GATE["min_roc_auc"]) == []
