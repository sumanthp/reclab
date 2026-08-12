import pytest

from reclab.architectures.hybrid_llm import HybridLLM
from reclab.datasets import SyntheticConfig, generate_synthetic_dataset


def _cold_start_dataset():
    cfg = SyntheticConfig(
        n_users=150,
        n_items=100,
        n_categories=5,
        median_sequence_length=12,
        cold_start_ratio=0.35,
        cold_start_max_interactions=2,
        seed=8,
    )
    return generate_synthetic_dataset(cfg)


def test_fit_requires_item_metadata_with_description():
    interactions, _ = _cold_start_dataset()
    model = HybridLLM(embedding_dim=8, epochs=3)
    with pytest.raises(ValueError):
        model.fit(interactions, item_metadata=None)


def test_fit_and_recommend_returns_k_items():
    interactions, item_metadata = _cold_start_dataset()
    model = HybridLLM(embedding_dim=8, epochs=3)
    model.fit(interactions, item_metadata)

    user_id = interactions["user_id"].iloc[0]
    recs = model.recommend(user_id, k=8)

    assert len(recs) == 8
    assert len(set(recs)) == 8


def test_can_recommend_cold_start_items():
    """The actual claim this architecture makes: it can surface items with
    very few (or zero) training interactions, which two_tower/sasrec can't,
    because the candidate pool explicitly includes cold items scored by text
    similarity rather than only re-ranking the encoder's own candidates."""
    interactions, item_metadata = _cold_start_dataset()
    model = HybridLLM(embedding_dim=8, epochs=3, candidate_pool_size=100)
    model.fit(interactions, item_metadata)

    cold_item_set = set(model.cold_items)
    assert cold_item_set  # sanity: the dataset actually has cold items

    recommended_cold_items = set()
    for user_id in interactions["user_id"].unique()[:30]:
        recs = model.recommend(user_id, k=10)
        recommended_cold_items |= set(recs) & cold_item_set

    assert recommended_cold_items, "expected at least one cold item recommended across 30 users"


def test_unseen_user_falls_back_to_popularity():
    interactions, item_metadata = _cold_start_dataset()
    model = HybridLLM(embedding_dim=8, epochs=3)
    model.fit(interactions, item_metadata)

    recs = model.recommend("someone-not-in-training-data", k=5)
    assert recs == model.popularity[:5]


def test_recommend_before_fit_raises():
    model = HybridLLM()
    with pytest.raises(RuntimeError):
        model.recommend("u0", k=5)
