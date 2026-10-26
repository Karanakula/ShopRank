import pandas as pd
import streamlit as st
from common import APIError, demo_users, get, price, short, show_api_error, sidebar_status, stars

st.set_page_config(page_title="Recommend · ShopRank", page_icon="🛒", layout="wide")
sidebar_status()
st.title("🛒 Recommend")

users = demo_users()
labels = {u["user_id"]: u["label"] for u in users}
top = st.columns([3, 2, 1.4])
user_id = top[0].selectbox("Pick a shopper", list(labels), format_func=lambda u: f"{labels[u]}  ({u[:10]}…)")
custom = top[1].text_input("…or paste any user ID", placeholder="e.g. a brand-new user gets best-sellers")
user_id = custom.strip() or user_id
k = top[2].slider("How many", 5, 20, 10)
compare = st.toggle("Compare with the popularity baseline (best-sellers)", value=True)

try:
    recs = get(f"/recommend/{user_id}", k=k)
    popular = get("/popular", k=k, user_id=user_id) if compare else None
    try:
        history = get(f"/users/{user_id}/history")
    except APIError:
        history = {"segment": None, "items": []}
except APIError as err:
    show_api_error(err)

if recs["strategy"] == "popular":
    st.info("This user has no purchase history, so ShopRank falls back to trending products "
            "(`strategy: popular`).")
elif history["segment"]:
    st.caption(f"Segment: **{history['segment']}** · {len(history['items'])} purchases")


def rec_table(items: list[dict], with_reason: bool) -> pd.DataFrame:
    rows = []
    for i in items:
        row = {"#": i["rank"], "Product": short(i["title"]), "Price": price(i["price"]), "Store": i["store"],
               "Expected": stars(i.get("expected_rating"))}
        if with_reason:
            row["Score"] = i["score"]
            row["Because you bought"] = short(i["reason"]["title"], 40) if i["reason"] else "—"
        rows.append(row)
    return pd.DataFrame(rows)


score_column = st.column_config.ProgressColumn("Score", help="ranker probability of a like", format="%.3f",
                                               min_value=0.0, max_value=max([i["score"] or 0 for i in recs["items"]]
                                                                            + [0.01]))
columns = {"#": st.column_config.NumberColumn("#", width="small"),
           "Product": st.column_config.TextColumn("Product", width="medium"),
           "Because you bought": st.column_config.TextColumn("Because you bought", width="medium"),
           "Score": score_column}
height = 38 + 35 * max(k, min(len(history["items"]), 15))
cols = st.columns([1, 1.7, 1.1] if compare else [1, 2])

with cols[0]:
    st.markdown("#### Purchase history")
    if history["items"]:
        st.dataframe(pd.DataFrame([{"Product": short(h["title"], 45), "Stars": "★" * h["rating"], "Date": h["date"]}
                                   for h in history["items"]]),
                     hide_index=True, width="stretch", height=height, column_config=columns)
    else:
        st.write("No history.")

with cols[1]:
    st.markdown(f"#### ShopRank top {k}" if recs["strategy"] == "personalized" else f"#### Trending top {k}")
    st.dataframe(rec_table(recs["items"], recs["strategy"] == "personalized"), hide_index=True,
                 width="stretch", height=height, column_config=columns)

if compare:
    with cols[2]:
        st.markdown(f"#### Best-sellers top {k}")
        st.dataframe(rec_table(popular["items"], False).drop(columns="Store"), hide_index=True,
                     width="stretch", height=height, column_config=columns)

    mine = {i["parent_asin"] for i in recs["items"]}
    base = {i["parent_asin"] for i in popular["items"]}
    stores_mine = {i["store"] for i in recs["items"]}
    stores_base = {i["store"] for i in popular["items"]}
    m1, m2, m3 = st.columns(3)
    m1.metric("Products in both lists", f"{len(mine & base)} of {k}")
    m2.metric("Different stores (ShopRank vs best-sellers)", f"{len(stores_mine)} vs {len(stores_base)}")
    hist_stores = {h["store"] for h in history["items"]}
    m3.metric("Recommendations from stores this shopper already bought from",
              f"{sum(i['store'] in hist_stores for i in recs['items'])} vs "
              f"{sum(i['store'] in hist_stores for i in popular['items'])}")
    st.caption("Best-sellers are the same for everyone (minus what this shopper already owns); "
               "ShopRank's list follows the shopper's own history.")
