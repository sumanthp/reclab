import pandas as pd

from reclab.architectures.rerankers import LexicalReranker


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
