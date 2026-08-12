"""Pluggable re-ranker interface for HybridLLM.

The class is named `hybrid_llm` because that's the eventual target (an LLM
re-ranking a shortlist using natural-language item context) — but calling out
to a real LLM API needs credentials this project can't assume every user or
CI run has. `LexicalReranker` is the default: a real, working TF-IDF/cosine
similarity re-ranker that needs no API key and no network access, so the
architecture is genuinely usable out of the box.

Swapping in a real LLM is meant to be a small, obvious change: implement
`Reranker` against an LLM API of your choice and pass it to
`HybridLLM(reranker=YourReranker())`. See CONTRIBUTING.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Reranker(ABC):
    @abstractmethod
    def fit(self, item_metadata: pd.DataFrame, text_col: str, id_col: str = "item_id") -> None:
        """Index item text so `score` can be called cheaply per recommendation."""

    @abstractmethod
    def score(self, query_text: str, candidate_item_ids: list[Any]) -> dict[Any, float]:
        """Return a relevance score in roughly [0, 1] for each candidate,
        given a natural-language query built from the user's recent items."""


class LexicalReranker(Reranker):
    """TF-IDF cosine similarity between the query text and each candidate's
    item text. No API key, no network call, no LLM — this is the honest
    default, not a placeholder pretending to be an LLM."""

    def fit(self, item_metadata: pd.DataFrame, text_col: str, id_col: str = "item_id") -> None:
        self._id_col = id_col
        self._text_col = text_col
        texts = item_metadata[text_col].fillna("").astype(str).tolist()
        self._item_ids = item_metadata[id_col].tolist()
        self._id_to_row = {item_id: row for row, item_id in enumerate(self._item_ids)}

        self._vectorizer = TfidfVectorizer()
        try:
            self._matrix = self._vectorizer.fit_transform(texts)
        except ValueError:
            # Empty vocabulary (e.g. all-blank descriptions) — degrade to
            # "no signal" rather than crashing the whole architecture.
            self._vectorizer = None
            self._matrix = None

    def score(self, query_text: str, candidate_item_ids: list[Any]) -> dict[Any, float]:
        if not query_text.strip() or self._vectorizer is None:
            return dict.fromkeys(candidate_item_ids, 0.0)

        query_vec = self._vectorizer.transform([query_text])
        rows = [self._id_to_row[i] for i in candidate_item_ids if i in self._id_to_row]
        if not rows:
            return dict.fromkeys(candidate_item_ids, 0.0)

        sims = cosine_similarity(query_vec, self._matrix[rows])[0]
        row_to_sim = dict(zip(rows, sims, strict=True))

        return {
            item_id: float(row_to_sim.get(self._id_to_row.get(item_id, -1), 0.0))
            for item_id in candidate_item_ids
        }
