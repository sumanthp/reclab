"""Numerical gradient check for SASRec's hand-derived backward pass.

Hand-written backprop is exactly the kind of code that looks plausible and is
subtly wrong. This test is the actual evidence that `_forward_backward` is
correct, not just that the model trains without crashing — it compares the
analytic gradient against a finite-difference approximation for a
representative sample of parameters across every gradient tensor.
"""

from __future__ import annotations

import numpy as np

from reclab.architectures.sasrec import SASRec

EPS = 1e-5
TOLERANCE = 2e-3


def _make_model() -> SASRec:
    model = SASRec(embedding_dim=4, max_sequence_length=10, epochs=0, seed=0)
    model.item_to_idx = {f"i{i}": i for i in range(5)}
    model.idx_to_item = [f"i{i}" for i in range(5)]

    rng = np.random.default_rng(1)
    d = model.embedding_dim
    model.E = rng.normal(0, 0.3, size=(5, d))
    model.P = rng.normal(0, 0.3, size=(model.max_sequence_length, d))
    model.Wq = rng.normal(0, 0.3, size=(d, d))
    model.Wk = rng.normal(0, 0.3, size=(d, d))
    model.Wv = rng.normal(0, 0.3, size=(d, d))
    return model


def _numerical_grad(model: SASRec, idx_seq: list[int], param: np.ndarray, i: int, j: int) -> float:
    original = param[i, j]

    param[i, j] = original + EPS
    loss_plus, _ = model._forward_backward(idx_seq)

    param[i, j] = original - EPS
    loss_minus, _ = model._forward_backward(idx_seq)

    param[i, j] = original
    return (loss_plus - loss_minus) / (2 * EPS)


def test_sasrec_gradients_match_finite_differences():
    model = _make_model()
    idx_seq = [0, 1, 2, 3, 1, 4]

    _, grads = model._forward_backward(idx_seq)
    length = len(idx_seq)

    checks = [
        (model.Wq, grads["dWq"], [(0, 0), (2, 3), (3, 1)]),
        (model.Wk, grads["dWk"], [(1, 1), (0, 2)]),
        (model.Wv, grads["dWv"], [(2, 2), (3, 0)]),
        (model.E, grads["dE"], [(0, 0), (1, 2), (4, 3)]),
        (model.P[:length], grads["dP"], [(0, 0), (2, 1), (length - 1, 2)]),
    ]

    for param, analytic_grad, positions in checks:
        for i, j in positions:
            numerical = _numerical_grad(model, idx_seq, param, i, j)
            analytic = analytic_grad[i, j]
            assert abs(numerical - analytic) < TOLERANCE, (
                f"gradient mismatch at ({i},{j}): analytic={analytic:.6f}, "
                f"numerical={numerical:.6f}"
            )
