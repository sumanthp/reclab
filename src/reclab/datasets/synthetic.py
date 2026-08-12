"""Synthetic dataset generator.

Generates a reproducible interactions log + item metadata table with
controllable sparsity, cold-start ratio, sequence length, and category
structure (which drives item text similarity). This exists so the reasoning
engine, architectures, and eval harness can be developed and benchmarked
without network access to the real public datasets referenced in the MVP
plan — see the module docstring in `loaders.py` for what that means for
real-dataset validation.

Design choices that matter for benchmark interpretation:

- Items are assigned to categories. Item descriptions share vocabulary within
  a category and not across categories, so a lexical/text-similarity re-ranker
  (like the one `hybrid_llm` uses) has real signal to exploit — this isn't
  just noise dressed up as "item text."
- Users have 1-2 preferred categories and interact with them 80% of the time,
  so there's real collaborative signal for two_tower/sasrec to learn from.
- Cold-start items are capped at a small number of total interactions across
  *all* users, so cold_start_ratio in the resulting DataProfile actually
  reflects genuinely sparse items, not an artifact of sampling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SyntheticConfig:
    n_users: int = 500
    n_items: int = 300
    n_categories: int = 8
    median_sequence_length: int = 15
    cold_start_ratio: float = 0.3
    cold_start_max_interactions: int = 3
    preferred_category_weight: float = 0.8
    include_item_text: bool = True
    seed: int = 42


def generate_synthetic_dataset(
    config: SyntheticConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate (interactions, item_metadata) per `config`.

    interactions columns: user_id, item_id, timestamp
    item_metadata columns: item_id, category, description (if include_item_text)
    """
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    item_ids = [f"i{idx}" for idx in range(cfg.n_items)]
    categories = rng.integers(0, cfg.n_categories, size=cfg.n_items)

    n_cold = int(round(cfg.cold_start_ratio * cfg.n_items))
    cold_idx = set(rng.choice(cfg.n_items, size=n_cold, replace=False).tolist())
    warm_idx_by_category: dict[int, list[int]] = {c: [] for c in range(cfg.n_categories)}
    cold_idx_by_category: dict[int, list[int]] = {c: [] for c in range(cfg.n_categories)}
    for idx, cat in enumerate(categories):
        (cold_idx_by_category if idx in cold_idx else warm_idx_by_category)[int(cat)].append(idx)

    item_metadata = _build_item_metadata(item_ids, categories, cfg, rng)

    cold_interaction_counts = dict.fromkeys(cold_idx, 0)

    user_ids = [f"u{idx}" for idx in range(cfg.n_users)]
    rows: list[tuple[str, str, int]] = []
    base_time = 1_700_000_000  # arbitrary fixed epoch so runs are comparable

    for user_id in user_ids:
        n_preferred = rng.integers(1, 3)
        preferred_categories = rng.choice(cfg.n_categories, size=n_preferred, replace=False)

        # Poisson around the target median gives a realistic right-skewed
        # sequence-length distribution while keeping the median controllable.
        seq_len = max(1, int(rng.poisson(cfg.median_sequence_length)))

        t = base_time + int(rng.integers(0, 1_000_000))
        for _ in range(seq_len):
            use_preferred = rng.random() < cfg.preferred_category_weight
            category = (
                int(rng.choice(preferred_categories))
                if use_preferred
                else int(rng.integers(0, cfg.n_categories))
            )

            item_idx = _sample_item(
                category,
                warm_idx_by_category,
                cold_idx_by_category,
                cold_interaction_counts,
                cfg,
                rng,
            )
            t += int(rng.integers(60, 3600))  # 1 minute to 1 hour between interactions
            rows.append((user_id, item_ids[item_idx], t))

    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"])
    return interactions, item_metadata


def _sample_item(
    category: int,
    warm_idx_by_category: dict[int, list[int]],
    cold_idx_by_category: dict[int, list[int]],
    cold_interaction_counts: dict[int, int],
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> int:
    available_cold = [
        idx
        for idx in cold_idx_by_category.get(category, [])
        if cold_interaction_counts[idx] < cfg.cold_start_max_interactions
    ]
    warm_pool = warm_idx_by_category.get(category) or _any_nonempty(warm_idx_by_category)

    want_cold = available_cold and rng.random() < cfg.cold_start_ratio
    if want_cold:
        item_idx = int(rng.choice(available_cold))
        cold_interaction_counts[item_idx] += 1
        return item_idx

    return int(rng.choice(warm_pool))


def _any_nonempty(pools: dict[int, list[int]]) -> list[int]:
    for pool in pools.values():
        if pool:
            return pool
    raise ValueError("no items available to sample from — n_items too small for n_categories")


def _build_item_metadata(
    item_ids: list[str],
    categories: np.ndarray,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not cfg.include_item_text:
        return pd.DataFrame({"item_id": item_ids, "category": categories})

    # 20 tags per category, 5 sampled without replacement per item: enough
    # combinations (C(20,5) = 15504) that items within the same category are
    # lexically distinguishable from each other, not just from other
    # categories — otherwise a lexical re-ranker can narrow down to "the
    # right category" but has no real signal for which specific item within
    # it, which understates hybrid_llm's actual cold-start behavior.
    tokens_per_category = {
        c: [f"cat{c}_tag{k}" for k in range(20)] for c in range(cfg.n_categories)
    }
    descriptions = []
    for idx, cat in enumerate(categories):
        tags = rng.choice(tokens_per_category[int(cat)], size=5, replace=False)
        descriptions.append(f"category {cat} item {idx} " + " ".join(tags))

    return pd.DataFrame(
        {"item_id": item_ids, "category": categories, "description": descriptions}
    )
