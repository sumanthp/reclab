from __future__ import annotations

from reclab.eval.comparison import summarize_comparison
from reclab.eval.harness import EvalResult
from reclab.reasoning_engine import Recommendation


def _rec(architecture: str, rank: int, score: float) -> Recommendation:
    return Recommendation(architecture=architecture, rank=rank, score=score, rationale="")


def _result(
    architecture: str, recall: float, cold_recall: float | None = None
) -> EvalResult:
    return EvalResult(
        architecture=architecture,
        k=10,
        n_test_users=10,
        recall_at_k=recall,
        ndcg_at_k=recall / 2,
        coverage_at_k=0.5,
        cold_start_recall_at_k=cold_recall,
        cold_start_surfaced_rate=0.0,
    )


def test_matches_when_shortlist_pick_has_best_recall():
    shortlist = [_rec("sasrec", 1, 0.75), _rec("two_tower", 2, 0.5)]
    eval_results = {"sasrec": _result("sasrec", 0.20), "two_tower": _result("two_tower", 0.10)}

    summary = summarize_comparison(shortlist, eval_results)

    assert summary.shortlist_pick == "sasrec"
    assert summary.measured_best_recall == "sasrec"
    assert summary.matches_on_recall is True
    assert summary.note is None


def test_mismatch_without_cold_start_explanation():
    shortlist = [_rec("two_tower", 1, 0.5), _rec("hybrid_llm", 2, 0.35)]
    eval_results = {
        "two_tower": _result("two_tower", 0.05, cold_recall=0.0),
        "hybrid_llm": _result("hybrid_llm", 0.09, cold_recall=0.05),
    }

    summary = summarize_comparison(shortlist, eval_results)

    assert summary.shortlist_pick == "two_tower"
    assert summary.measured_best_recall == "hybrid_llm"
    assert summary.matches_on_recall is False
    # two_tower isn't the cold-start winner either, so no softening note
    assert summary.note is None


def test_mismatch_with_cold_start_explanation():
    shortlist = [_rec("hybrid_llm", 1, 0.75), _rec("two_tower", 2, 0.3)]
    eval_results = {
        "hybrid_llm": _result("hybrid_llm", 0.10, cold_recall=0.22),
        "two_tower": _result("two_tower", 0.24, cold_recall=0.0),
    }

    summary = summarize_comparison(shortlist, eval_results)

    assert summary.matches_on_recall is False
    assert summary.measured_best_cold_start_recall == "hybrid_llm"
    assert summary.note is not None
    assert "cold-start recall" in summary.note


def test_no_eval_results_yields_unmeasured_summary():
    shortlist = [_rec("sasrec", 1, 0.75)]

    summary = summarize_comparison(shortlist, {})

    assert summary.shortlist_pick == "sasrec"
    assert summary.measured_best_recall is None
    assert summary.matches_on_recall is None


def test_no_cold_start_items_in_test_set():
    shortlist = [_rec("two_tower", 1, 0.5)]
    eval_results = {"two_tower": _result("two_tower", 0.1, cold_recall=None)}

    summary = summarize_comparison(shortlist, eval_results)

    assert summary.measured_best_cold_start_recall is None
