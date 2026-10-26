import json

import altair as alt
import streamlit as st
from common import PREDICTION_LOG, REPORTS, health, prediction_log, sidebar_status

st.set_page_config(page_title="Monitoring · ShopRank", page_icon="📈", layout="wide")
sidebar_status()
st.title("📈 Monitoring")

if st.button("Refresh"):
    prediction_log.clear()
    health.clear()

st.markdown("#### Data drift")
drift_file = REPORTS / "drift_summary.json"
if drift_file.exists():
    st.json(json.loads(drift_file.read_text()))
else:
    st.info("Drift reports arrive in Phase 17. This page will show the latest summary from "
            "`reports/drift_summary.json` once it exists.")

st.markdown("#### API traffic")
log = prediction_log()
if log.empty:
    st.info(f"No requests logged yet ({PREDICTION_LOG}). Use the other pages, then refresh.")
    st.stop()

h = health()
c = st.columns(4)
c[0].metric("Requests logged", f"{len(log):,}")
c[1].metric("p50 latency", f"{log['latency_ms'].quantile(0.5):.1f} ms")
c[2].metric("p95 latency", f"{log['latency_ms'].quantile(0.95):.1f} ms")
c[3].metric("Model serving now", h["model_version"] if h else "API offline")

by_endpoint = log.groupby("endpoint")["latency_ms"].agg(
    requests="size", p50=lambda s: s.quantile(0.5), p95=lambda s: s.quantile(0.95), max="max").round(2)
st.dataframe(by_endpoint, width="stretch")

per_minute = log.set_index("time").groupby("endpoint").resample("1min").size().rename("requests").reset_index()
left, right = st.columns(2)
left.altair_chart(
    alt.Chart(per_minute).mark_bar().encode(x=alt.X("time:T", title=None), y=alt.Y("requests:Q", stack=True),
                                            color="endpoint:N").properties(title="Requests per minute", height=300),
    width="stretch",
)
recent = log.tail(2000)
right.altair_chart(
    alt.Chart(recent).mark_circle(size=14, opacity=0.5).encode(
        x=alt.X("time:T", title=None), y=alt.Y("latency_ms:Q", title="Latency (ms)", scale=alt.Scale(type="log")),
        color="endpoint:N", tooltip=["endpoint", "latency_ms", "user_id"],
    ).properties(title="Latency per request (last 2,000)", height=300),
    width="stretch",
)

if "strategy" in log:
    rec = log[log["endpoint"] == "recommend"]
    if len(rec):
        share = (rec["strategy"] == "popular").mean()
        st.caption(f"{share:.1%} of /recommend calls fell back to popular products (unknown users).")
