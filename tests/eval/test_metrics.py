import pytest

from reclab.eval.metrics import coverage_at_k, ndcg_at_k, recall_at_k


def test_recall_at_k_full_hit():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_k():
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_recall_at_k_no_relevant_items():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == pytest.approx(1.0)


def test_ndcg_rewards_earlier_hits():
    # A relevant item ranked first should score higher than the same item ranked second.
    ndcg_first = ndcg_at_k(["a", "x"], {"a"}, k=2)
    ndcg_second = ndcg_at_k(["x", "a"], {"a"}, k=2)
    assert ndcg_first > ndcg_second


def test_coverage_at_k():
    all_recs = [["a", "b"], ["b", "c"]]
    assert coverage_at_k(all_recs, catalog_size=4, k=2) == pytest.approx(0.75)  # {a,b,c} of 4


def test_coverage_at_k_empty_catalog():
    assert coverage_at_k([["a"]], catalog_size=0, k=1) == 0.0
