"""ShopRank dashboard. Run with `streamlit run app/Home.py` (the API must be running: `make api`)."""

import streamlit as st
from common import metadata, sidebar_status

st.set_page_config(page_title="ShopRank", page_icon="🎮", layout="wide")
sidebar_status()

st.title("🎮 ShopRank")
st.subheader("Personalized video-game recommendations from 4.6 million Amazon reviews")

meta = metadata()
test, trending = meta["test_metrics"], meta["baselines"]["trending"]
lift = test["ndcg@10"] / trending["ndcg@10"] - 1

c1, c2, c3, c4 = st.columns(4)
c1.metric("NDCG@10", f"{test['ndcg@10']:.4f}", f"{lift:+.0%}",
          help=f"Ranking quality on unseen 2023 purchases; best-sellers score {trending['ndcg@10']:.4f}")
c2.metric("Coverage", f"{test['coverage@10']:.0%}", f"{test['coverage@10'] / trending['coverage@10']:.0f}×",
          help=f"Share of the catalogue in someone's top 10; best-sellers reach {trending['coverage@10']:.2%}")
c3.metric("Shoppers", f"{meta['data']['users']:,}")
c4.metric("Products", f"{meta['data']['products']:,}")
st.caption(f"Measured on purchases from {meta['split']['t2']} onward, which the model never saw. "
           "A best-seller list is the baseline every recommender has to beat.")

st.markdown("### How it works")
a, b, c = st.columns(3)
a.markdown("**1 · Find candidates**  \nFour quick searches pull ~150 products per shopper: "
           "*people who bought this also bought*, similar descriptions (TF-IDF and sentence embeddings), "
           "and what is trending.")
b.markdown("**2 · Rank them**  \nA LightGBM model scores each candidate from 18 features: the shopper's habits, "
           "the product's rating, review sentiment and age, and how well they match.")
c.markdown("**3 · Explain and serve**  \nA FastAPI service returns the top 10 in a few milliseconds, each with a "
           "*because you bought …* reason. This dashboard calls that API.")

st.markdown("### Try it")
st.page_link("pages/1_Recommend.py", label="**Recommend** – pick a shopper, see their top 10 next to best-sellers",
             icon="🛒")
st.page_link("pages/2_Similar_products.py", label="**Similar products** – search a game or accessory", icon="🔎")
st.page_link("pages/3_Review_analyzer.py", label="**Review analyzer** – paste a review, get sentiment and stars",
             icon="💬")
st.page_link("pages/4_Customer_segments.py", label="**Customer segments** – six shopper types found by K-Means",
             icon="👥")
st.page_link("pages/5_Model_report.py", label="**Model report** – 13 models compared, SHAP explanations", icon="📊")
st.page_link("pages/6_Monitoring.py", label="**Monitoring** – live request count and latency", icon="📈")

st.caption(f"Model version {meta['model_version']} · trained {meta['trained_at'][:10]} · "
           f"data up to {meta['serving_cutoff']} · Amazon Reviews 2023 (Video Games), 5-core since 2018")
