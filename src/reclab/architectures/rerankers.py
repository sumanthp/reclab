"""Pluggable re-ranker interface for HybridLLM.

Two real implementations exist: `LexicalReranker` (TF-IDF/cosine similarity,
the default — no API key, no network access, so the architecture is usable
out of the box) and `AnthropicReranker` (a real Claude API call, opt-in via
`pip install reclab[llm]` since it needs credentials this project can't
assume every user or CI run has).

A third-party LLM is a small, obvious addition: implement `Reranker` against
the API of your choice and pass it to `HybridLLM(reranker=YourReranker())`.
See CONTRIBUTING.md.
"""

from __future__ import annotations

import json
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


class AnthropicReranker(Reranker):
    """Re-ranks candidates by asking Claude to judge relevance to a
    natural-language query built from the user's recent items — the real LLM
    re-ranker `hybrid_llm`'s name has been promising since `LexicalReranker`
    was the only option.

    Needs the optional `llm` dependency group (`pip install reclab[llm]`) and
    an `ANTHROPIC_API_KEY`. Pass an existing `anthropic.Anthropic()` (or any
    object exposing the same `.messages.create(...)` shape) via `client` to
    reuse a configured client or to inject a fake one in tests — the `anthropic`
    package itself is only imported lazily, on first real use, so it's not a
    hard dependency of the default install.

    Any failure — no API key, network error, rate limit, or a response that
    doesn't parse as the expected JSON — degrades to "no signal" (score 0.0
    for every candidate) rather than crashing the whole comparison job. By
    the time re-ranking runs, training has already succeeded; a flaky LLM
    call shouldn't take that down with it.
    """

    def __init__(
        self,
        client: Any | None = None,
        model: str = "claude-sonnet-5",
        max_candidates: int = 20,
        max_text_chars: int = 200,
    ) -> None:
        self._client = client
        self._model = model
        self._max_candidates = max_candidates
        self._max_text_chars = max_text_chars
        self._id_to_text: dict[Any, str] = {}

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def fit(self, item_metadata: pd.DataFrame, text_col: str, id_col: str = "item_id") -> None:
        texts = item_metadata[text_col].fillna("").astype(str)
        self._id_to_text = dict(zip(item_metadata[id_col], texts, strict=True))

    def score(self, query_text: str, candidate_item_ids: list[Any]) -> dict[Any, float]:
        if not query_text.strip() or not candidate_item_ids:
            return dict.fromkeys(candidate_item_ids, 0.0)

        # Only the first max_candidates are sent to the LLM (cost/latency);
        # the rest fall back to "no signal" rather than being dropped.
        rated = candidate_item_ids[: self._max_candidates]
        prompt = self._build_prompt(query_text, rated)

        try:
            client = self._get_client()
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            return dict.fromkeys(candidate_item_ids, 0.0)

        raw = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        parsed = self._parse_scores(raw, rated)
        return {item_id: parsed.get(item_id, 0.0) for item_id in candidate_item_ids}

    def _build_prompt(self, query_text: str, candidate_item_ids: list[Any]) -> str:
        lines = [
            f"- id={item_id!r}: "
            f"{self._id_to_text.get(item_id, '')[: self._max_text_chars] or '(no description)'}"
            for item_id in candidate_item_ids
        ]
        return (
            "A user's recent activity, summarized as:\n"
            f"{query_text}\n\n"
            "Rate how relevant each of the following candidate items is to that "
            "user, on a 0.0-1.0 scale where 1.0 is highly relevant. Respond with "
            "ONLY a JSON object mapping each item id (as a string) to its score — "
            "no other text, no markdown fences.\n\n" + "\n".join(lines)
        )

    @staticmethod
    def _parse_scores(raw: str, candidate_item_ids: list[Any]) -> dict[Any, float]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}

        str_to_id = {str(item_id): item_id for item_id in candidate_item_ids}
        scores: dict[Any, float] = {}
        for key, value in parsed.items():
            item_id = str_to_id.get(str(key))
            if item_id is None:
                continue
            try:
                scores[item_id] = max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                continue
        return scores
