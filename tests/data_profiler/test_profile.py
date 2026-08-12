import pandas as pd
import pytest

from reclab.data_profiler import profile_interactions


def _interactions(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def test_profile_basic_counts():
    df = _interactions(
        [("u1", "i1"), ("u1", "i2"), ("u2", "i1"), ("u2", "i2"), ("u2", "i3")]
    )
    profile = profile_interactions(df)

    assert profile.n_users == 2
    assert profile.n_items == 3
    assert profile.n_interactions == 5
    assert profile.median_sequence_length == 2.5


def test_profile_cold_start_ratio():
    # i1 has 3 interactions, i2 has 1 -> with threshold=2, i2 is cold-start
    df = _interactions([("u1", "i1"), ("u2", "i1"), ("u3", "i1"), ("u1", "i2")])
    profile = profile_interactions(df, cold_start_threshold=2)

    assert profile.cold_start_ratio == pytest.approx(0.5)


def test_profile_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        profile_interactions(pd.DataFrame(columns=["user_id", "item_id"]))


def test_profile_detects_item_text_metadata():
    interactions = _interactions([("u1", "i1"), ("u2", "i2")])
    item_metadata = pd.DataFrame({"item_id": ["i1", "i2"], "description": ["a", "b"]})

    profile = profile_interactions(
        interactions, item_metadata=item_metadata, text_col="description"
    )

    assert profile.has_item_text is True
    assert profile.has_item_image is False
