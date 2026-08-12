from reclab.data_profiler import profile_interactions
from reclab.datasets import SyntheticConfig, generate_synthetic_dataset


def test_generate_synthetic_dataset_basic_shape():
    cfg = SyntheticConfig(n_users=50, n_items=40, n_categories=4, median_sequence_length=10)
    interactions, item_metadata = generate_synthetic_dataset(cfg)

    assert set(interactions.columns) == {"user_id", "item_id", "timestamp"}
    assert set(item_metadata.columns) == {"item_id", "category", "description"}
    assert interactions["user_id"].nunique() <= 50
    assert item_metadata["item_id"].nunique() == 40


def test_generate_synthetic_dataset_is_reproducible():
    cfg = SyntheticConfig(n_users=30, n_items=20, seed=7)
    a_interactions, a_meta = generate_synthetic_dataset(cfg)
    b_interactions, b_meta = generate_synthetic_dataset(cfg)

    assert a_interactions.equals(b_interactions)
    assert a_meta.equals(b_meta)


def test_cold_start_ratio_roughly_tracks_config():
    cfg = SyntheticConfig(
        n_users=300,
        n_items=200,
        n_categories=5,
        median_sequence_length=20,
        cold_start_ratio=0.4,
        cold_start_max_interactions=2,
        seed=1,
    )
    interactions, item_metadata = generate_synthetic_dataset(cfg)
    profile = profile_interactions(
        interactions,
        cold_start_threshold=3,
        item_metadata=item_metadata,
        text_col="description",
    )

    # Not exact (probabilistic sampling + fixed catalog), but should land
    # in a sane neighborhood of the configured ratio.
    assert 0.2 <= profile.cold_start_ratio <= 0.6
    assert profile.has_item_text is True


def test_timestamps_increase_within_each_user():
    cfg = SyntheticConfig(n_users=20, n_items=30, median_sequence_length=8, seed=3)
    interactions, _ = generate_synthetic_dataset(cfg)

    for _, group in interactions.groupby("user_id"):
        ts = group["timestamp"].tolist()
        assert ts == sorted(ts)
