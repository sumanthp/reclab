"""Offline evaluation harness: runs candidate architectures against held-out
interactions and computes the metrics the reasoning engine's Phase 0 claims
get validated against."""

from reclab.eval.harness import EvalResult, run_eval, temporal_train_test_split
from reclab.eval.metrics import coverage_at_k, ndcg_at_k, recall_at_k

__all__ = [
    "recall_at_k",
    "ndcg_at_k",
    "coverage_at_k",
    "EvalResult",
    "run_eval",
    "temporal_train_test_split",
]
