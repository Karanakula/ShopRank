import altair as alt
import pandas as pd
import streamlit as st
from common import REPORTS, metadata, model_comparison, sidebar_status

st.set_page_config(page_title="Model report · ShopRank", page_icon="📊", layout="wide")
sidebar_status()
st.title("📊 Model report")

meta = metadata()
test = meta["test_metrics"]
c = st.columns(5)
c[0].metric("NDCG@10", f"{test['ndcg@10']:.4f}")
c[1].metric("Recall@10", f"{test['recall@10']:.4f}")
c[2].metric("Hit rate@10", f"{test['hit_rate@10']:.1%}")
c[3].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
c[4].metric("Coverage@10", f"{test['coverage@10']:.0%}")
st.caption(f"Shipped model: **LightGBM (tuned)**, chosen by cross-validated NDCG@10 on the training copy, "
           f"then scored once on the test window ({meta['split']['t2']} onward).")

results = model_comparison()
baselines = pd.DataFrame([
    {"model": {"trending": "Trending (popularity baseline)"}.get(name, name.replace("sorted_by_", "Retrieval only: ")),
     "test_ndcg@10": m["ndcg@10"]}
    for name, m in meta["baselines"].items() if name != "sorted_by_pop_rank"
])
chart_data = pd.concat([
    results[["model", "test_ndcg@10", "winner"]].assign(kind=lambda d: d["winner"].map({True: "shipped",
                                                                                        False: "ranker"})),
    baselines.assign(kind="baseline"),
])

st.markdown("#### Test NDCG@10 for every model")
st.altair_chart(
    alt.Chart(chart_data).mark_bar().encode(
        x=alt.X("test_ndcg@10:Q", title="Test NDCG@10"),
        y=alt.Y("model:N", sort="-x", title=None),
        color=alt.Color("kind:N", title=None, scale=alt.Scale(domain=["shipped", "ranker", "baseline"],
                                                              range=["#e4572e", "#4c78a8", "#bab0ac"])),
        tooltip=["model", alt.Tooltip("test_ndcg@10:Q", format=".4f")],
    ).properties(height=480),
    width="stretch",
)

st.markdown("#### All results")
st.dataframe(results.drop(columns="winner"), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.4f") for c in results.columns
                            if c not in ("model", "winner")})
st.caption("CV = 5-fold cross-validation grouped by user on the training copy (labels from Jul–Dec 2022). "
           "Test = the 2023 window. Logistic regression's log loss is high because balanced class weights shift "
           "its probabilities; its ranking is unaffected.")

st.markdown("#### What drives the ranker (SHAP, 5,000 test rows)")
shap_png = REPORTS / "figures" / "shap_summary.png"
if shap_png.exists():
    st.image(str(shap_png), width=760)
st.caption("Newer products, trending rank, review count and text similarity push scores up the most; "
           "user-only features barely matter because they are the same for every product in a user's list.")
