import streamlit as st
from common import APIError, post, show_api_error, sidebar_status

st.set_page_config(page_title="Review analyzer · ShopRank", page_icon="💬", layout="wide")
sidebar_status()
st.title("💬 Review analyzer")
st.caption("TF-IDF + logistic regression trained on 140k review titles and texts (macro F1 ≈ 0.90 on "
           "validation). The predicted rating maps the sentiment score to the average stars of training "
           "reviews that scored the same.")

EXAMPLES = {
    "Happy": "Absolutely love this controller. Battery lasts forever and it feels great in my hands.",
    "Unhappy": "Stopped charging after two weeks. Customer support never answered. Waste of money.",
    "Mixed": "The game is fun with friends, but the story is short and it crashed twice.",
}
if "review" not in st.session_state:
    st.session_state.review = EXAMPLES["Mixed"]
cols = st.columns(len(EXAMPLES) + 3)
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(f"{name} example"):
        st.session_state.review = text

text = st.text_area("Paste a review", key="review", height=150, max_chars=10_000)
if not text.strip():
    st.stop()

try:
    result = post("/sentiment", {"text": text})
except APIError as err:
    show_api_error(err)

c1, c2, c3 = st.columns(3)
c1.metric("Sentiment", "😊 Positive" if result["label"] == "positive" else "😞 Negative")
c2.metric("Confidence", f"{result['probability']:.0%}")
predicted = result["predicted_rating"]
c3.metric("Predicted rating", f"{predicted:.1f} ★" if predicted is not None else "—")
st.progress(result["p_positive"], text=f"Probability positive: {result['p_positive']:.1%}")
