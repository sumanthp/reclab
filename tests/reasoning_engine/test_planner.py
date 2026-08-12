import pytest

from reclab.data_profiler import DataProfile
from reclab.reasoning_engine import recommend_architectures
from reclab.reasoning_engine.planner import LOW_CONFIDENCE_MARGIN


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


def test_every_recommendation_has_structured_factors():
    for rec in recommend_architectures(_profile()):
        assert rec.factors
        for factor in rec.factors:
            assert factor.detail
            assert isinstance(factor.effect, float)


def test_rationale_is_factors_joined():
    for rec in recommend_architectures(_profile()):
        assert rec.rationale == "; ".join(f.detail for f in rec.factors)


def test_margin_to_next_is_none_only_for_last_rank():
    recs = recommend_architectures(_profile())
    assert recs[-1].margin_to_next is None
    for rec in recs[:-1]:
        assert rec.margin_to_next is not None
        assert rec.margin_to_next >= 0


def test_margin_to_next_matches_score_difference():
    recs = recommend_architectures(_profile())
    for i in range(len(recs) - 1):
        assert recs[i].margin_to_next == pytest.approx(recs[i].score - recs[i + 1].score)


def test_low_confidence_flagged_on_a_real_thin_margin():
    # Amazon Reviews' All_Beauty profile (see benchmarks/results/) — the
    # exact real case the LOW_CONFIDENCE_MARGIN constant is modeled on.
    # two_tower (0.50) beats sasrec/hybrid_llm (0.35 each) by 0.15, and the
    # measured winner turned out to be a different architecture entirely.
    profile = _profile(
        n_users=253,
        n_items=356,
        n_interactions=2535,
        sparsity=0.9718545987476129,
        cold_start_ratio=0.0,
        median_sequence_length=7.0,
        has_item_text=True,
    )
    recs = recommend_architectures(profile)
    assert recs[0].architecture == "two_tower"
    assert recs[0].low_confidence is True


def test_low_confidence_false_on_a_wide_margin():
    # cold-start-heavy synthetic scenario — hybrid_llm's margin over #2 is
    # wide (0.25+), a confident pick either way.
    profile = _profile(cold_start_ratio=0.6, sparsity=0.995, has_item_text=True)
    recs = recommend_architectures(profile)
    assert recs[0].architecture == "hybrid_llm"
    assert recs[0].margin_to_next is not None
    assert recs[0].margin_to_next > LOW_CONFIDENCE_MARGIN
    assert recs[0].low_confidence is False


def test_low_confidence_is_only_ever_set_on_rank_one():
    for profile in [_profile(), _profile(cold_start_ratio=0.6, sparsity=0.995, has_item_text=True)]:
        for rec in recommend_architectures(profile)[1:]:
            assert rec.low_confidence is False
