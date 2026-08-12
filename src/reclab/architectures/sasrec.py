"""SASRec: self-attentive sequential recommendation, implemented as a single-
head causal self-attention block over item embeddings, trained via next-item
prediction with hand-derived backpropagation in plain NumPy.

Why hand-written instead of a framework: PyTorch can't be imported in this
project's development sandbox (the current PyPI wheel requires CUDA runtime
libraries even for CPU-only use, and the CPU-only wheel index isn't reachable
here — see docs/architecture/mvp-plan.md and CONTRIBUTING.md). Rather than
fake a transformer, this implements the real mechanism — causal self-attention
producing a sequence representation, trained end-to-end — just without a deep
learning framework underneath it. The math is checked against numerical
gradients in tests/architectures/test_sasrec_gradients.py; don't trust hand
derivatives that aren't checked that way.

Current scope: single attention head, one layer, no feed-forward block.
Multi-head / multi-layer is real future work (`n_heads` > 1 raises
NotImplementedError rather than silently ignoring the setting), not
implemented here to keep the manual backward pass tractable and verifiable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reclab.architectures._common import build_vocab, popularity_ranking, user_sequences
from reclab.architectures.base import Architecture, ArchitectureInfo


def _softmax_causal(scores: np.ndarray) -> np.ndarray:
    """Row-wise softmax where row i is only defined over columns 0..i
    (causal masking) — position i can only attend to itself and the past."""
    length = scores.shape[0]
    attn = np.zeros_like(scores)
    for i in range(length):
        row = scores[i, : i + 1]
        row = row - row.max()
        exp = np.exp(row)
        attn[i, : i + 1] = exp / exp.sum()
    return attn


class SASRec(Architecture):
    @classmethod
    def info(cls) -> ArchitectureInfo:
        return ArchitectureInfo(
            name="sasrec",
            description=(
                "Single-head causal self-attention over a user's interaction "
                "history, trained via next-item prediction (plain NumPy, "
                "hand-derived backprop — see module docstring). Captures "
                "order and recency signal that two-tower ignores; needs "
                "reasonably long user sequences to earn its extra training cost."
            ),
            strengths=[
                "Captures sequence order and recency",
                "Strong on session-based / repeat-engagement patterns",
            ],
            weaknesses=[
                "Needs sufficient per-user sequence length to outperform simpler baselines",
                "Still weak on cold-start items with no interaction history",
                "More expensive to train than two-tower",
            ],
            relative_train_cost="medium",
            relative_serving_latency="medium",
        )

    def __init__(
        self,
        embedding_dim: int = 16,
        max_sequence_length: int = 50,
        epochs: int = 40,
        learning_rate: float = 0.05,
        l2_reg: float = 0.0005,
        n_heads: int = 1,
        seed: int = 42,
    ) -> None:
        if n_heads != 1:
            raise NotImplementedError(
                "SASRec currently only supports n_heads=1 (single-head attention); "
                "multi-head is real future work, not a silently-ignored setting"
            )
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.n_heads = n_heads
        self.seed = seed
        self._fitted = False

    def fit(self, interactions: pd.DataFrame, item_metadata: pd.DataFrame | None = None) -> None:
        rng = np.random.default_rng(self.seed)
        d = self.embedding_dim

        self.item_to_idx, self.idx_to_item = build_vocab(interactions["item_id"])
        n_items = len(self.idx_to_item)
        if n_items < 2:
            raise ValueError("SASRec needs at least 2 distinct items to train against")

        self.E = rng.normal(0, 0.1, size=(n_items, d))
        self.P = rng.normal(0, 0.1, size=(self.max_sequence_length, d))
        scale = 1.0 / np.sqrt(d)
        self.Wq = rng.normal(0, scale, size=(d, d))
        self.Wk = rng.normal(0, scale, size=(d, d))
        self.Wv = rng.normal(0, scale, size=(d, d))

        sequences = user_sequences(interactions)
        self.sequences_by_user = {
            u: seq[-self.max_sequence_length :] for u, seq in sequences.items()
        }
        self.popularity = popularity_ranking(interactions)

        trainable_users = [u for u, seq in self.sequences_by_user.items() if len(seq) >= 2]

        for _ in range(self.epochs):
            rng.shuffle(trainable_users)
            for user_id in trainable_users:
                idx_seq = [self.item_to_idx[i] for i in self.sequences_by_user[user_id]]
                _, grads = self._forward_backward(idx_seq)
                self._apply_grads(grads)

        self._fitted = True

    def _forward(self, idx_seq: list[int]) -> tuple[np.ndarray, tuple]:
        """Run the attention block forward. Returns (O, cache) where O[t] is
        the output representation at sequence position t."""
        length = len(idx_seq)
        d = self.embedding_dim
        x = self.E[idx_seq] + self.P[:length]
        q = x @ self.Wq
        k = x @ self.Wk
        v = x @ self.Wv
        scores = (q @ k.T) / np.sqrt(d)
        attn = _softmax_causal(scores)
        z = attn @ v
        o = z + x
        return o, (x, q, k, v, attn)

    def _forward_backward(self, idx_seq: list[int]) -> tuple[float, dict]:
        length = len(idx_seq)
        d = self.embedding_dim
        o, (x, q, k, v, attn) = self._forward(idx_seq)

        d_o = np.zeros((length, d))
        d_e_out = np.zeros_like(self.E)
        total_loss = 0.0
        n_targets = 0

        for t in range(length - 1):
            target = idx_seq[t + 1]
            logits = o[t] @ self.E.T
            logits = logits - logits.max()
            exp = np.exp(logits)
            probs = exp / exp.sum()
            total_loss += -np.log(probs[target] + 1e-12)
            n_targets += 1

            d_logits = probs.copy()
            d_logits[target] -= 1.0
            d_o[t] += d_logits @ self.E
            d_e_out += np.outer(d_logits, o[t])

        n = max(n_targets, 1)

        # Backward through the residual + attention block.
        d_z = d_o.copy()
        d_x = d_o.copy()  # residual: O = Z + X

        d_attn = d_z @ v.T
        d_v = attn.T @ d_z

        d_scores = np.zeros_like(attn)
        for i in range(length):
            a_row = attn[i, : i + 1]
            da_row = d_attn[i, : i + 1]
            dot = a_row @ da_row
            d_scores[i, : i + 1] = a_row * (da_row - dot)

        d_q = (d_scores @ k) / np.sqrt(d)
        d_k = (d_scores.T @ q) / np.sqrt(d)

        d_wq = x.T @ d_q
        d_wk = x.T @ d_k
        d_wv = x.T @ d_v

        d_x += d_q @ self.Wq.T + d_k @ self.Wk.T + d_v @ self.Wv.T

        d_e_in = np.zeros_like(self.E)
        for t, item_idx in enumerate(idx_seq):
            d_e_in[item_idx] += d_x[t]
        d_p = d_x

        grads = {
            "dE": (d_e_in + d_e_out) / n,
            "dP": d_p / n,
            "dWq": d_wq / n,
            "dWk": d_wk / n,
            "dWv": d_wv / n,
            "length": length,
        }
        return total_loss / n, grads

    def _apply_grads(self, grads: dict) -> None:
        lr, reg, length = self.learning_rate, self.l2_reg, grads["length"]
        self.E -= lr * (grads["dE"] + reg * self.E)
        self.P[:length] -= lr * (grads["dP"] + reg * self.P[:length])
        self.Wq -= lr * (grads["dWq"] + reg * self.Wq)
        self.Wk -= lr * (grads["dWk"] + reg * self.Wk)
        self.Wv -= lr * (grads["dWv"] + reg * self.Wv)

    def recommend(self, user_id: Any, k: int = 10) -> list[Any]:
        if not self._fitted:
            raise RuntimeError("SASRec.recommend called before fit()")

        seq = self.sequences_by_user.get(user_id, [])
        idx_seq = [self.item_to_idx[i] for i in seq if i in self.item_to_idx]
        if not idx_seq:
            return self.popularity[:k]

        o, _ = self._forward(idx_seq)
        last_representation = o[-1]
        scores = self.E @ last_representation

        for item_idx in set(idx_seq):
            scores[item_idx] = -np.inf

        top_k = np.argsort(-scores)[:k]
        return [self.idx_to_item[idx] for idx in top_k]
