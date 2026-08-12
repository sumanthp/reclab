"""Hybrid encoder + re-ranker: a SASRec encoder generates candidates from
collaborative signal, then a re-ranker (default: TF-IDF lexical similarity,
see rerankers.py) blends in item text to surface relevant items the encoder
alone can't reach — specifically items with too few interactions to have a
learned embedding, up to and including items with *zero* training
interactions. That candidate injection, not just re-scoring, is what actually
gives this architecture a cold-start advantage over two_tower/sasrec; a
re-ranker that only reordered the encoder's own candidates couldn't recommend
anything the encoder didn't already know about.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from reclab.architectures._common import popularity_ranking
from reclab.architectures.base import Architecture, ArchitectureInfo
from reclab.architectures.rerankers import LexicalReranker, Reranker
from reclab.architectures.sasrec import SASRec


class HybridLLM(Architecture):
    @classmethod
    def info(cls) -> ArchitectureInfo:
        return ArchitectureInfo(
            name="hybrid_llm",
            description=(
                "SASRec-style encoder for candidate generation from collaborative "
                "signal, plus a pluggable re-ranker (default: TF-IDF lexical "
                "similarity, no API key required) that injects and re-scores "
                "candidates using item text — including items the encoder has "
                "never seen. A real LLM-based re-ranker is a drop-in swap "
                "(see rerankers.py) once you have API access; the interface "
                "doesn't change."
            ),
            strengths=[
                "Strong cold-start performance when item text/metadata is rich",
                "Can recommend items with zero training interactions via text alone",
                "Handles sparse interaction data better than pure collaborative models",
            ],
            weaknesses=[
                "Highest serving latency and cost of the three candidates",
                "Requires usable item text metadata to earn its complexity",
                "Default re-ranker is lexical (TF-IDF), not a real LLM — semantic "
                "matches beyond shared vocabulary need a real LLM re-ranker swapped in",
            ],
            relative_train_cost="high",
            relative_serving_latency="high",
        )

    def __init__(
        self,
        embedding_dim: int = 16,
        max_sequence_length: int = 50,
        epochs: int = 40,
        learning_rate: float = 0.05,
        encoder_weight: float = 0.6,
        text_weight: float = 0.4,
        candidate_pool_size: int = 50,
        cold_start_threshold: int = 5,
        recent_items_for_query: int = 5,
        cold_slot_fraction: float = 0.2,
        reranker: Reranker | None = None,
        seed: int = 42,
    ) -> None:
        self.encoder = SASRec(
            embedding_dim=embedding_dim,
            max_sequence_length=max_sequence_length,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
        )
        self.reranker = reranker or LexicalReranker()
        self.encoder_weight = encoder_weight
        self.text_weight = text_weight
        self.candidate_pool_size = candidate_pool_size
        self.cold_start_threshold = cold_start_threshold
        self.recent_items_for_query = recent_items_for_query
        self.cold_slot_fraction = cold_slot_fraction
        self._fitted = False

    def fit(self, interactions: pd.DataFrame, item_metadata: pd.DataFrame | None = None) -> None:
        if item_metadata is None or "description" not in item_metadata.columns:
            raise ValueError(
                "HybridLLM requires item_metadata with a 'description' text column "
                "(the convention used by reclab.datasets.synthetic and "
                "reclab.datasets.loaders.load_movielens_100k)"
            )

        self.encoder.fit(interactions, item_metadata)
        self.reranker.fit(item_metadata, text_col="description")

        self.item_text_by_id = dict(
            zip(item_metadata["item_id"], item_metadata["description"], strict=True)
        )
        catalog_items = item_metadata["item_id"].tolist()
        interacted_counts = interactions["item_id"].value_counts()
        self.cold_items = [
            i for i in catalog_items if interacted_counts.get(i, 0) < self.cold_start_threshold
        ]
        self.popularity = popularity_ranking(interactions)
        self._fitted = True

    def recommend(self, user_id: Any, k: int = 10) -> list[Any]:
        if not self._fitted:
            raise RuntimeError("HybridLLM.recommend called before fit()")

        seq = self.encoder.sequences_by_user.get(user_id, [])
        if not seq:
            return self.popularity[:k]

        seen = set(seq)
        encoder_scores = self._encoder_candidate_scores(seq)

        recent_items = seq[-self.recent_items_for_query :]
        query_text = " ".join(self.item_text_by_id.get(i, "") for i in recent_items)

        encoder_candidates = sorted(
            (i for i in encoder_scores if i not in seen),
            key=lambda i: -encoder_scores[i],
        )[: self.candidate_pool_size]
        # Cold items are NOT capped by candidate_pool_size: that cap exists to
        # bound how many warm/encoder candidates get re-ranked, not to limit
        # cold-item discovery — capping it silently threw away exactly the
        # items this architecture exists to surface. All cold items always
        # enter the re-ranking pool; the re-ranker's text-similarity score is
        # what decides whether any of them actually make the top k.
        cold_candidates = [i for i in self.cold_items if i not in seen]
        candidate_ids = list(dict.fromkeys(encoder_candidates + cold_candidates))
        if not candidate_ids:
            return self.popularity[:k]

        lexical_scores = self.reranker.score(query_text, candidate_ids)

        final_scores = {
            item_id: self.encoder_weight * encoder_scores.get(item_id, 0.0)
            + self.text_weight * lexical_scores.get(item_id, 0.0)
            for item_id in candidate_ids
        }

        # A pure blended-score ranking systematically buries cold items: a
        # warm item in the user's favorite category scores well on *both*
        # axes (learned collaborative signal + shared category vocabulary),
        # while a cold item only has the lexical axis to compete on. Without
        # a floor, cold items rarely crack the top k regardless of how
        # relevant they are — which would make the "surfaces cold items"
        # claim in info() false in practice. Reserving a small number of
        # slots for the best lexical-only cold candidates (a real technique,
        # not a scoring hack — production recommenders call this an
        # exploration/discovery slot) is what actually makes that claim hold.
        cold_slots = max(1, round(k * self.cold_slot_fraction)) if cold_candidates else 0
        top_cold = sorted(
            (i for i in cold_candidates if lexical_scores.get(i, 0.0) > 0),
            key=lambda i: -lexical_scores[i],
        )[:cold_slots]

        remaining_k = k - len(top_cold)
        remaining_candidates = [i for i in candidate_ids if i not in top_cold]
        ranked_remaining = sorted(remaining_candidates, key=lambda i: -final_scores[i])

        return top_cold + ranked_remaining[:remaining_k]

    def _encoder_candidate_scores(self, seq: list[Any]) -> dict[Any, float]:
        """Normalized (0-1) encoder scores over every item the SASRec encoder
        has an embedding for. Empty if none of the user's items are in the
        encoder's vocabulary (e.g. every item they've touched is itself
        below the encoder's training threshold)."""
        idx_seq = [self.encoder.item_to_idx[i] for i in seq if i in self.encoder.item_to_idx]
        if not idx_seq:
            return {}

        o, _ = self.encoder._forward(idx_seq)
        raw_scores = self.encoder.E @ o[-1]
        low, high = raw_scores.min(), raw_scores.max()
        normalized = (raw_scores - low) / (high - low + 1e-9)
        return {self.encoder.idx_to_item[i]: float(normalized[i]) for i in range(len(normalized))}
