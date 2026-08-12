from types import SimpleNamespace
from typing import Any

import pandas as pd

from reclab.architectures.rerankers import AnthropicReranker, LexicalReranker


def test_lexical_reranker_favors_similar_text():
    item_metadata = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "description": [
                "space adventure science fiction robots",
                "space adventure science fiction aliens",
                "romantic comedy wedding drama",
            ],
        }
    )
    reranker = LexicalReranker()
    reranker.fit(item_metadata, text_col="description")

    scores = reranker.score("science fiction space robots", ["a", "b", "c"])

    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["c"]


def test_lexical_reranker_handles_empty_query():
    item_metadata = pd.DataFrame({"item_id": ["a", "b"], "description": ["x y", "y z"]})
    reranker = LexicalReranker()
    reranker.fit(item_metadata, text_col="description")

    scores = reranker.score("   ", ["a", "b"])
    assert scores == {"a": 0.0, "b": 0.0}


def test_lexical_reranker_handles_blank_corpus_gracefully():
    item_metadata = pd.DataFrame({"item_id": ["a", "b"], "description": ["", ""]})
    reranker = LexicalReranker()
    reranker.fit(item_metadata, text_col="description")

    scores = reranker.score("anything", ["a", "b"])
    assert scores == {"a": 0.0, "b": 0.0}


class FakeAnthropicClient:
    """Stands in for `anthropic.Anthropic()` — no real package, no network,
    no API key needed to exercise AnthropicReranker's own logic."""

    def __init__(self, reply_text: str | None = None, raises: Exception | None = None) -> None:
        self._reply_text = reply_text
        self._raises = raises
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._reply_text)])


def _item_metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "description": ["space robots", "space aliens", "wedding drama"],
        }
    )


def test_anthropic_reranker_parses_scores_from_response():
    client = FakeAnthropicClient(reply_text='{"a": 0.9, "b": 0.4, "c": 0.1}')
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("likes space movies", ["a", "b", "c"])

    assert scores == {"a": 0.9, "b": 0.4, "c": 0.1}
    assert len(client.calls) == 1


def test_anthropic_reranker_strips_markdown_code_fences():
    client = FakeAnthropicClient(reply_text='```json\n{"a": 0.7}\n```')
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a"])
    assert scores == {"a": 0.7}


def test_anthropic_reranker_clamps_scores_to_unit_range():
    client = FakeAnthropicClient(reply_text='{"a": 5.0, "b": -2.0}')
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b"])
    assert scores == {"a": 1.0, "b": 0.0}


def test_anthropic_reranker_degrades_to_zero_on_malformed_json():
    client = FakeAnthropicClient(reply_text="not json at all")
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b", "c"])
    assert scores == {"a": 0.0, "b": 0.0, "c": 0.0}


def test_anthropic_reranker_degrades_to_zero_on_api_error():
    client = FakeAnthropicClient(raises=RuntimeError("rate limited"))
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b"])
    assert scores == {"a": 0.0, "b": 0.0}


def test_anthropic_reranker_skips_the_api_call_for_an_empty_query():
    client = FakeAnthropicClient(reply_text="{}")
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("   ", ["a", "b"])

    assert scores == {"a": 0.0, "b": 0.0}
    assert client.calls == []


def test_anthropic_reranker_caps_candidates_sent_to_the_llm():
    client = FakeAnthropicClient(reply_text='{"a": 0.8}')
    reranker = AnthropicReranker(client=client, max_candidates=1)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b", "c"])

    assert scores == {"a": 0.8, "b": 0.0, "c": 0.0}
    prompt = client.calls[0]["messages"][0]["content"]
    assert "id='a'" in prompt
    assert "id='b'" not in prompt


def test_anthropic_reranker_ignores_unknown_ids_in_response():
    client = FakeAnthropicClient(reply_text='{"a": 0.5, "not-a-real-id": 0.9}')
    reranker = AnthropicReranker(client=client)
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b"])
    assert scores == {"a": 0.5, "b": 0.0}


def test_anthropic_reranker_degrades_gracefully_without_the_anthropic_package():
    """No `client=` injected, and the optional `anthropic` package isn't
    part of the default install — this must degrade to no signal like any
    other API failure, not crash the whole compare job. Also confirms the
    `anthropic` import is lazy: constructing/fitting the reranker never
    imports it, only a real `score()` call does."""
    reranker = AnthropicReranker()
    reranker.fit(_item_metadata(), text_col="description")

    scores = reranker.score("query", ["a", "b"])
    assert scores == {"a": 0.0, "b": 0.0}
