import pandas as pd
import pytest

from reclab.architectures.sasrec import SASRec
from reclab.datasets import SyntheticConfig, generate_synthetic_dataset


def _small_dataset():
    cfg = SyntheticConfig(n_users=60, n_items=40, n_categories=3, median_sequence_length=10, seed=6)
    return generate_synthetic_dataset(cfg)


def test_fit_and_recommend_returns_k_items():
    interactions, _ = _small_dataset()
    model = SASRec(embedding_dim=8, epochs=3)
    model.fit(interactions)

    user_id = interactions["user_id"].iloc[0]
    recs = model.recommend(user_id, k=5)

    assert len(recs) == 5
    assert len(set(recs)) == 5


def test_recommendations_exclude_seen_items():
    interactions, _ = _small_dataset()
    model = SASRec(embedding_dim=8, epochs=3)
    model.fit(interactions)

    user_id = interactions["user_id"].iloc[0]
    seen = set(interactions[interactions["user_id"] == user_id]["item_id"])
    recs = model.recommend(user_id, k=10)

    assert not (set(recs) & seen)


def test_unseen_user_falls_back_to_popularity():
    interactions, _ = _small_dataset()
    model = SASRec(embedding_dim=8, epochs=3)
    model.fit(interactions)

    recs = model.recommend("someone-not-in-training-data", k=5)
    assert recs == model.popularity[:5]


def test_recommend_before_fit_raises():
    model = SASRec()
    with pytest.raises(RuntimeError):
        model.recommend("u0", k=5)


def test_multi_head_not_implemented():
    with pytest.raises(NotImplementedError):
        SASRec(n_heads=2)


def test_fit_rejects_single_item_dataset():
    df = pd.DataFrame({"user_id": ["u1", "u2"], "item_id": ["i1", "i1"], "timestamp": [1, 2]})
    model = SASRec()
    with pytest.raises(ValueError):
        model.fit(df)
