#!/usr/bin/env python
"""Phase 0/1 benchmark runner — the actual check for whether the reasoning
engine's shortlist means anything.

Loads a dataset, profiles it, asks the reasoning engine for a ranked
architecture shortlist, then genuinely trains and evaluates all three
architectures on a temporal train/test split so the shortlist can be checked
against what actually wins.

Usage:
    uv run python scripts/run_benchmark.py --dataset synthetic
    uv run python scripts/run_benchmark.py --dataset synthetic --cold-start-heavy
    uv run python scripts/run_benchmark.py --csv path/to/interactions.csv
    uv run python scripts/run_benchmark.py --movielens-100k path/to/ml-100k(.zip)

Dataset options:
  --dataset synthetic          A reproducible synthetic dataset (see
                                reclab.datasets.synthetic) — used because the
                                real public benchmarks (MovieLens, Amazon
                                Reviews) aren't reachable from this project's
                                development sandbox. Real-dataset loaders
                                exist (reclab.datasets.loaders) and are
                                believed correct but unverified end-to-end;
                                running them here is real validation, not a
                                formality.
  --csv PATH                   A local interactions CSV (no train/eval of
                                architectures needing item text unless you
                                also pass --item-metadata-csv).
  --movielens-100k PATH        Real MovieLens 100K, as a directory or .zip.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from reclab.architectures import REGISTRY  # noqa: E402
from reclab.data_profiler import profile_interactions  # noqa: E402
from reclab.datasets import SyntheticConfig, generate_synthetic_dataset  # noqa: E402
from reclab.datasets.loaders import load_movielens_100k  # noqa: E402
from reclab.eval import run_eval, temporal_train_test_split  # noqa: E402
from reclab.reasoning_engine import recommend_architectures  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results"


def load_csv(path: str, user_col: str, item_col: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    return pd.read_csv(path), None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=["synthetic"], help="built-in dataset generator")
    source.add_argument("--csv", help="path to a local interactions CSV")
    source.add_argument("--movielens-100k", help="path to ml-100k directory or .zip")

    parser.add_argument(
        "--cold-start-heavy",
        action="store_true",
        help="(--dataset synthetic only) generate a sparser, higher-cold-start-ratio "
        "dataset — the case hybrid_llm is supposed to win",
    )
    parser.add_argument(
        "--item-metadata-csv", help="item metadata CSV (needs a description column)"
    )
    parser.add_argument("--user-col", default="user_id")
    parser.add_argument("--item-col", default="item_id")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    item_metadata = None
    if args.dataset == "synthetic":
        cfg = (
            SyntheticConfig(cold_start_ratio=0.4, cold_start_max_interactions=2, seed=args.seed)
            if args.cold_start_heavy
            else SyntheticConfig(seed=args.seed)
        )
        interactions, item_metadata = generate_synthetic_dataset(cfg)
        dataset_label = "synthetic-cold-start-heavy" if args.cold_start_heavy else "synthetic"
    elif args.movielens_100k:
        interactions, item_metadata = load_movielens_100k(args.movielens_100k)
        dataset_label = "movielens-100k"
    else:
        interactions, _ = load_csv(args.csv, args.user_col, args.item_col)
        if args.item_metadata_csv:
            item_metadata = pd.read_csv(args.item_metadata_csv)
        dataset_label = Path(args.csv).stem

    profile = profile_interactions(
        interactions,
        user_col=args.user_col,
        item_col=args.item_col,
        item_metadata=item_metadata,
        text_col="description" if item_metadata is not None else None,
    )
    shortlist = recommend_architectures(profile)

    print(f"\n=== reclab benchmark: {dataset_label} ===\n")
    print("Data profile:")
    for field, value in asdict(profile).items():
        print(f"  {field}: {value}")

    print("\nReasoning engine shortlist:")
    for rec in shortlist:
        print(f"  #{rec.rank} {rec.architecture} (score={rec.score:.2f}) — {rec.rationale}")

    has_timestamp = "timestamp" in interactions.columns
    eval_results = {}
    if has_timestamp:
        train, test = temporal_train_test_split(
            interactions, user_col=args.user_col, item_col=args.item_col
        )
        print(f"\nTrain/test split: {len(train)} train rows, {len(test)} test rows\n")
        print("Training and evaluating all registered architectures...")

        for name, arch_cls in REGISTRY.items():
            try:
                result = run_eval(
                    arch_cls(),
                    train,
                    test,
                    item_metadata=item_metadata,
                    k=args.k,
                    user_col=args.user_col,
                    item_col=args.item_col,
                )
                eval_results[name] = asdict(result)
                cold_recall = (
                    f"{result.cold_start_recall_at_k:.3f}"
                    if result.cold_start_recall_at_k is not None
                    else "n/a"
                )
                print(
                    f"  {name}: Recall@{args.k}={result.recall_at_k:.3f} "
                    f"NDCG@{args.k}={result.ndcg_at_k:.3f} "
                    f"Coverage@{args.k}={result.coverage_at_k:.3f} "
                    f"ColdStartRecall@{args.k}={cold_recall} "
                    f"ColdStartSurfacedRate={result.cold_start_surfaced_rate:.3f}"
                )
            except ValueError as exc:
                print(f"  {name}: skipped — {exc}")
                eval_results[name] = {"skipped": str(exc)}

        scored = {name: r for name, r in eval_results.items() if "recall_at_k" in r}
        best_recall = max(scored, key=lambda n: scored[n]["recall_at_k"], default=None)
        best_cold_recall = max(
            (n for n in scored if scored[n]["cold_start_recall_at_k"] is not None),
            key=lambda n: scored[n]["cold_start_recall_at_k"],
            default=None,
        )

        print(f"\nReasoning engine's #1 pick: {shortlist[0].architecture}")
        print(f"Measured best by Recall@{args.k}: {best_recall}")
        if best_cold_recall:
            print(f"Measured best by ColdStartRecall@{args.k}: {best_cold_recall}")

        if best_recall and shortlist[0].architecture != best_recall:
            note = (
                " — but it does win on ColdStartRecall, which is plausibly what the "
                "planner's rationale was actually optimizing for; a single Recall@K "
                "comparison doesn't capture that trade-off"
                if shortlist[0].architecture == best_cold_recall
                else ""
            )
            print(
                f"\nMISMATCH on Recall@{args.k}{note}. This is exactly the signal Phase 0 "
                "exists to surface — see benchmarks/README.md before trusting the "
                "planner's rationale as-is."
            )
    else:
        print(
            "\nNo timestamp column — skipping train/eval "
            "(temporal_train_test_split needs one). Profile and shortlist only."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / f"{dataset_label}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
    result_path.write_text(
        json.dumps(
            {
                "dataset": dataset_label,
                "profile": asdict(profile),
                "reasoning_engine_shortlist": [asdict(r) for r in shortlist],
                "eval_results": eval_results,
            },
            indent=2,
        )
    )
    print(f"\nSaved result to {result_path}")


if __name__ == "__main__":
    main()
