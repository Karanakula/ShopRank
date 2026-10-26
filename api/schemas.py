"""Request and response models; FastAPI validates against these and documents them at /docs."""

from typing import Literal

from pydantic import BaseModel, Field


class Health(BaseModel):
    status: Literal["ok"]
    model_version: str
    trained_at: str
    serving_cutoff: str
    users: int
    products: int


class Reason(BaseModel):
    because_you_bought: str = Field(description="parent_asin of the most similar product in the user's history")
    title: str
    similarity: float = Field(description="cosine similarity of the two products' text embeddings")


class Product(BaseModel):
    parent_asin: str
    title: str
    price: float | None = Field(description="null when the catalogue has no price")
    store: str


class RecommendedItem(Product):
    rank: int = Field(ge=1)
    score: float | None = Field(description="ranker probability of a like; null for popular fallback")
    reason: Reason | None = None
    expected_rating: float | None = Field(None, ge=1, le=5, description="stars the user would likely give")


class Recommendations(BaseModel):
    user_id: str
    strategy: Literal["personalized", "popular"]
    k: int
    model_version: str
    items: list[RecommendedItem]


class HistoryItem(Product):
    rating: int = Field(ge=1, le=5)
    date: str


class History(BaseModel):
    user_id: str
    segment: str | None
    items: list[HistoryItem] = Field(description="newest first")


class SearchItem(Product):
    review_count: int


class SearchResults(BaseModel):
    query: str
    items: list[SearchItem]


class SimilarItem(Product):
    similarity: float
    source: Literal["co_purchase", "content"] = Field(
        description="co_purchase = item-KNN on buyers; content = text embeddings (used to fill up to k)")


class SimilarItems(BaseModel):
    item: Product
    items: list[SimilarItem]


class SentimentRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000, examples=["Great game, my kids play it every day."])


class SentimentResponse(BaseModel):
    label: Literal["positive", "negative"]
    probability: float = Field(ge=0, le=1, description="probability of the returned label")
    p_positive: float = Field(ge=0, le=1)
    predicted_rating: float | None = Field(None, ge=1, le=5,
                                           description="average stars of training reviews with this sentiment")


class RatingRequest(BaseModel):
    user_id: str = Field(min_length=1, examples=["AE222HFZDH6BPTYFOUWGGU63YSIQ"])
    parent_asin: str = Field(min_length=1, examples=["B00Z9TLVK0"])


class RatingResponse(BaseModel):
    user_id: str
    parent_asin: str
    expected_rating: float = Field(ge=1, le=5)
    known_user: bool
    features: dict[str, float]
