import pandas as pd

from reclab.architectures.two_tower import TwoTower
from reclab.datasets import SyntheticConfig, generate_synthetic_dataset
from reclab.eval.harness import run_eval, temporal_train_test_split


def test_temporal_split_holds_out_last_n_per_user():
    df = pd.DataFrame(
        {
            "user_id": ["u1"] * 4 + ["u2"] * 2,
            "item_id": ["a", "b", "c", "d", "x", "y"],
            "timestamp": [1, 2, 3, 4, 1, 2],
        }
    )
    train, test = temporal_train_test_split(df, holdout_n=1)

    assert set(train[train["user_id"] == "u1"]["item_id"]) == {"a", "b", "c"}
    assert set(test[test["user_id"] == "u1"]["item_id"]) == {"d"}
    assert set(train[train["user_id"] == "u2"]["item_id"]) == {"x"}
    assert set(test[test["user_id"] == "u2"]["item_id"]) == {"y"}


def test_temporal_split_excludes_users_with_too_few_interactions():
    df = pd.DataFrame({"user_id": ["u1"], "item_id": ["a"], "timestamp": [1]})
    train, test = temporal_train_test_split(df, holdout_n=1)

    assert len(train) == 1
    assert len(test) == 0


def test_temporal_split_missing_timestamp_column_raises():
    df = pd.DataFrame({"user_id": ["u1", "u1"], "item_id": ["a", "b"]})
    try:
        temporal_train_test_split(df)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_eval_produces_sane_result():
    cfg = SyntheticConfig(
        n_users=80, n_items=60, n_categories=4, median_sequence_length=10, seed=11
    )
    interactions, _ = generate_synthetic_dataset(cfg)
    train, test = temporal_train_test_split(interactions, holdout_n=1)

    result = run_eval(TwoTower(embedding_dim=8, epochs=5), train, test, k=10)

    assert result.architecture == "TwoTower"
    assert result.n_test_users > 0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.ndcg_at_k <= 1.0
    assert 0.0 <= result.coverage_at_k <= 1.0
    assert 0.0 <= result.cold_start_surfaced_rate <= 1.0


def test_run_eval_reports_cold_start_recall_when_applicable():
    cfg = SyntheticConfig(
        n_users=150,
        n_items=100,
        n_categories=5,
        median_sequence_length=12,
        cold_start_ratio=0.35,
        cold_start_max_interactions=2,
        seed=12,
    )
    interactions, item_metadata = generate_synthetic_dataset(cfg)
    train, test = temporal_train_test_split(interactions, holdout_n=1)

    from reclab.architectures.hybrid_llm import HybridLLM

    result = run_eval(
        HybridLLM(embedding_dim=8, epochs=3, candidate_pool_size=100),
        train,
        test,
        item_metadata=item_metadata,
        k=10,
    )

    # cold_start_recall_at_k may legitimately be None if no held-out test
    # interaction happens to be on a cold item, but the field should at
    # least be well-formed when it isn't.
    if result.cold_start_recall_at_k is not None:
        assert 0.0 <= result.cold_start_recall_at_k <= 1.0


def test_hybrid_llm_surfaces_cold_items_more_than_baselines():
    """Locks in the actual measured finding from benchmarks/results/
    synthetic-cold-start-heavy.json: on a sparse, high-cold-start,
    rich-item-text dataset, hybrid_llm's candidate-injection mechanism
    should surface cold-start items more often than two_tower/sasrec, which
    can only recommend items they have a (weak) learned embedding for."""
    from reclab.architectures.hybrid_llm import HybridLLM
    from reclab.architectures.sasrec import SASRec
    from reclab.architectures.two_tower import TwoTower

    cfg = SyntheticConfig(
        n_users=200,
        n_items=150,
        n_categories=5,
        median_sequence_length=15,
        cold_start_ratio=0.4,
        cold_start_max_interactions=2,
        seed=21,
    )
    interactions, item_metadata = generate_synthetic_dataset(cfg)
    train, test = temporal_train_test_split(interactions, holdout_n=1)

    two_tower_result = run_eval(TwoTower(embedding_dim=8, epochs=10), train, test, k=10)
    sasrec_result = run_eval(SASRec(embedding_dim=8, epochs=10), train, test, k=10)
    hybrid_result = run_eval(
        HybridLLM(embedding_dim=8, epochs=10),
        train,
        test,
        item_metadata=item_metadata,
        k=10,
    )

    assert hybrid_result.cold_start_surfaced_rate > two_tower_result.cold_start_surfaced_rate
    assert hybrid_result.cold_start_surfaced_rate > sasrec_result.cold_start_surfaced_rate
