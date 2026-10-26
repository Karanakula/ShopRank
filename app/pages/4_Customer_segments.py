import altair as alt
import streamlit as st
from common import segments, sidebar_status

st.set_page_config(page_title="Customer segments · ShopRank", page_icon="👥", layout="wide")
sidebar_status()
st.title("👥 Customer segments")
st.caption("K-Means (k = 6) on six standardized habits per shopper: purchases, days since last purchase, "
           "average price paid, average rating given, share of verified purchases and number of stores. "
           "Fit on training data; every known shopper is assigned to its nearest segment.")

users = segments()
order = users["segment_name"].value_counts().index.tolist()
color = alt.Color("segment_name:N", title="Segment", sort=order)

left, right = st.columns([1.6, 1])
with left:
    st.markdown("#### Shoppers in 2-D (PCA)")
    sample = users.sample(min(6000, len(users)), random_state=42)
    st.altair_chart(
        alt.Chart(sample).mark_circle(size=12, opacity=0.5).encode(
            x=alt.X("pc1:Q", title="PC1"), y=alt.Y("pc2:Q", title="PC2"), color=color,
            tooltip=["segment_name", "n_purchases", "avg_price_paid", "avg_rating_given", "days_since_last"],
        ).properties(height=460).interactive(),
        width="stretch",
    )
    st.caption(f"Random sample of {len(sample):,} of {len(users):,} shoppers.")

with right:
    st.markdown("#### Segment sizes")
    sizes = users["segment_name"].value_counts().rename_axis("segment_name").reset_index(name="shoppers")
    st.altair_chart(
        alt.Chart(sizes).mark_bar().encode(
            x=alt.X("shoppers:Q", title="Shoppers"), y=alt.Y("segment_name:N", sort=order, title=None),
            color=alt.Color("segment_name:N", sort=order, legend=None), tooltip=["segment_name", "shoppers"],
        ).properties(height=460),
        width="stretch",
    )

st.markdown("#### Profile of each segment (averages)")
profile = users.groupby("segment_name").agg(
    shoppers=("user_id", "size"), purchases=("n_purchases", "mean"), days_since_last=("days_since_last", "mean"),
    avg_price_paid=("avg_price_paid", "mean"), avg_rating_given=("avg_rating_given", "mean"),
    verified_share=("verified_share", "mean"), stores=("n_stores", "mean"),
).loc[order]
st.dataframe(
    profile, width="stretch",
    column_config={
        "shoppers": st.column_config.NumberColumn("Shoppers", format="%d"),
        "purchases": st.column_config.NumberColumn("Purchases", format="%.1f"),
        "days_since_last": st.column_config.NumberColumn("Days since last", format="%.0f"),
        "avg_price_paid": st.column_config.NumberColumn("Avg price paid", format="$%.2f"),
        "avg_rating_given": st.column_config.NumberColumn("Avg rating given", format="%.2f ★"),
        "verified_share": st.column_config.ProgressColumn("Verified share", format="%.2f", min_value=0,
                                                          max_value=1),
        "stores": st.column_config.NumberColumn("Stores", format="%.1f"),
    },
)
