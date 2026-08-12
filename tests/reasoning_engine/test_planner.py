from reclab.data_profiler import DataProfile
from reclab.reasoning_engine import recommend_architectures


def _profile(**overrides) -> DataProfile:
    defaults = dict(
        n_users=1000,
        n_items=500,
        n_interactions=20000,
        sparsity=0.96,
        cold_start_ratio=0.1,
        median_sequence_length=15.0,
        has_item_text=False,
        has_item_image=False,
    )
    defaults.update(overrides)
    return DataProfile(**defaults)


def test_recommend_returns_all_registered_architectures():
    recs = recommend_architectures(_profile())
    names = {r.architecture for r in recs}
    assert names == {"two_tower", "sasrec", "hybrid_llm"}


def test_ranks_are_sequential_and_sorted_by_score_desc():
    recs = recommend_architectures(_profile())
    assert [r.rank for r in recs] == [1, 2, 3]
    scores = [r.score for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_high_cold_start_and_item_text_favors_hybrid_llm():
    """This is the core case from the MVP plan: sparse + rich item text +
    high cold-start should push hybrid_llm to the top of the shortlist."""
    profile = _profile(
        cold_start_ratio=0.6,
        sparsity=0.995,
        has_item_text=True,
        median_sequence_length=3.0,
    )
    recs = recommend_architectures(profile)
    assert recs[0].architecture == "hybrid_llm"


def test_short_sequences_without_cold_start_favor_two_tower_over_sasrec():
    profile = _profile(median_sequence_length=2.0, cold_start_ratio=0.05)
    recs = {r.architecture: r for r in recommend_architectures(profile)}
    assert recs["two_tower"].score > recs["sasrec"].score


def test_every_recommendation_has_a_nonempty_rationale():
    for rec in recommend_architectures(_profile()):
        assert rec.rationale
