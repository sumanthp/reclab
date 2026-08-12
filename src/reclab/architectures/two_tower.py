"""Two-tower baseline: separate user and item embedding towers, scored by dot
product, trained with a Bayesian Personalized Ranking (BPR) pairwise loss via
plain NumPy SGD. This is a real, gradient-trained matrix-factorization
recommender — not a mock — chosen over a deep-learning framework because
PyTorch isn't runnable in this project's development sandbox (see
docs/architecture/mvp-plan.md and the Phase 1 notes in CONTRIBUTING.md for
why). Its job in the benchmark suite is to be the floor every other
architecture has to beat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from reclab.architectures._common import build_vocab, popularity_ranking, user_sequences
from reclab.architectures.base import Architecture, ArchitectureInfo


class TwoTower(Architecture):
    @classmethod
    def info(cls) -> ArchitectureInfo:
        return ArchitectureInfo(
            name="two_tower",
            description=(
                "Separate user and item embedding towers trained with a "
                "BPR pairwise ranking loss (plain NumPy SGD, a real "
                "matrix-factorization recommender). Fast to train and serve; "
                "handles dense collaborative signal well but has no native "
                "answer for cold-start items or sequence order."
            ),
            strengths=[
                "Fast to train and serve",
                "Strong on dense interaction data",
                "Simple to reason about and debug",
            ],
            weaknesses=[
                "Weak on cold-start items with few interactions",
                "Ignores interaction order / recency",
                "No use of item text or image metadata",
            ],
            relative_train_cost="low",
            relative_serving_latency="low",
        )

    def __init__(
        self,
        embedding_dim: int = 16,
        epochs: int = 40,
        learning_rate: float = 0.05,
        l2_reg: float = 0.001,
        seed: int = 42,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.seed = seed
        self._fitted = False

    def fit(self, interactions: pd.DataFrame, item_metadata: pd.DataFrame | None = None) -> None:
        rng = np.random.default_rng(self.seed)

        self.user_to_idx, self.idx_to_user = build_vocab(interactions["user_id"])
        self.item_to_idx, self.idx_to_item = build_vocab(interactions["item_id"])
        n_users, n_items = len(self.idx_to_user), len(self.idx_to_item)

        if n_items < 2:
            raise ValueError("TwoTower needs at least 2 distinct items to train against")

        self.U = rng.normal(0, 0.1, size=(n_users, self.embedding_dim))
        self.V = rng.normal(0, 0.1, size=(n_items, self.embedding_dim))

        self.seen_items_by_user: dict[Any, set[int]] = {
            uid: {self.item_to_idx[i] for i in items}
            for uid, items in user_sequences(interactions).items()
        }
        self.popularity = popularity_ranking(interactions)

        pairs = [
            (self.user_to_idx[row.user_id], self.item_to_idx[row.item_id])
            for row in interactions.itertuples()
        ]

        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for u_idx, i_idx in pairs:
                seen = self.seen_items_by_user[self.idx_to_user[u_idx]]
                j_idx = int(rng.integers(0, n_items))
                attempts = 0
                while j_idx in seen and attempts < 10:
                    j_idx = int(rng.integers(0, n_items))
                    attempts += 1

                u_vec, i_vec, j_vec = self.U[u_idx], self.V[i_idx], self.V[j_idx]
                x_uij = u_vec @ (i_vec - j_vec)
                sig = 1.0 / (1.0 + np.exp(-x_uij))
                # d(-log sigmoid(x))/dx = -(1-sigmoid(x)); sign folded into the updates below
                grad = 1.0 - sig

                self.U[u_idx] += self.learning_rate * (
                    grad * (i_vec - j_vec) - self.l2_reg * u_vec
                )
                self.V[i_idx] += self.learning_rate * (grad * u_vec - self.l2_reg * i_vec)
                self.V[j_idx] += self.learning_rate * (-grad * u_vec - self.l2_reg * j_vec)

        self._fitted = True

    def recommend(self, user_id: Any, k: int = 10) -> list[Any]:
        if not self._fitted:
            raise RuntimeError("TwoTower.recommend called before fit()")

        if user_id not in self.user_to_idx:
            return self.popularity[:k]

        u_idx = self.user_to_idx[user_id]
        scores = self.V @ self.U[u_idx]

        seen = self.seen_items_by_user.get(user_id, set())
        for item_idx in seen:
            scores[item_idx] = -np.inf

        top_k = np.argsort(-scores)[:k]
        return [self.idx_to_item[idx] for idx in top_k]
