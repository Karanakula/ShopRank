import pandas as pd
import streamlit as st
from common import APIError, get, price, short, show_api_error, sidebar_status

st.set_page_config(page_title="Similar products · ShopRank", page_icon="🔎", layout="wide")
sidebar_status()
st.title("🔎 Similar products")
st.caption("“Bought together” neighbours come from item-KNN on shared buyers; when a product has fewer than "
           "k of those, the list is filled with products whose descriptions are closest (sentence embeddings).")

c1, c2 = st.columns([3, 1])
query = c1.text_input("Search a product title", value="zelda")
k = c2.slider("How many", 5, 20, 10)
if len(query.strip()) < 2:
    st.stop()

try:
    results = get("/search", q=query, limit=30)["items"]
except APIError as err:
    show_api_error(err)
if not results:
    st.warning("No product title contains all of those words.")
    st.stop()

options = {r["parent_asin"]: r for r in results}
asin = st.selectbox("Product", list(options),
                    format_func=lambda a: f"{options[a]['title'][:110]} · {options[a]['review_count']} reviews")

try:
    similar = get(f"/similar/{asin}", k=k)
except APIError as err:
    show_api_error(err)

item = similar["item"]
st.markdown(f"**{item['title']}** · {price(item['price'])} · {item['store']}")
st.dataframe(
    pd.DataFrame([{"Product": short(s["title"], 80), "Price": price(s["price"]), "Store": s["store"],
                   "Similarity": s["similarity"],
                   "Why": "bought by the same people" if s["source"] == "co_purchase" else "similar description"}
                  for s in similar["items"]]),
    hide_index=True, width="stretch",
    column_config={"Similarity": st.column_config.ProgressColumn("Similarity", format="%.3f", min_value=0.0,
                                                                 max_value=1.0)},
)
