"""Common interface every candidate architecture must implement.

Keeping this interface small and stable is what lets the reasoning engine and
eval harness treat every architecture interchangeably in the mix-and-match
sandbox, and what lets a contributor add a new architecture without touching
either of those.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ArchitectureInfo:
    """Static metadata about an architecture, used by the reasoning engine's
    planner to reason about fit *before* any training happens."""

    name: str
    description: str
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    # Rough relative cost/latency signals used for trade-off display, not
    # meant to be precise — refined once eval data exists.
    relative_train_cost: str = "medium"  # "low" | "medium" | "high"
    relative_serving_latency: str = "medium"  # "low" | "medium" | "high"


class Architecture(ABC):
    """Base class for a candidate recommendation architecture.

    Implementations live under `reclab/architectures/<name>/`. Each one must
    expose static `info()` metadata plus `fit` / `recommend` so the eval
    harness can run it without knowing anything else about it.
    """

    @classmethod
    @abstractmethod
    def info(cls) -> ArchitectureInfo:
        """Static metadata used for planning, independent of any dataset."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame, item_metadata: pd.DataFrame | None = None) -> None:
        """Train the model on a user-item interaction log."""

    @abstractmethod
    def recommend(self, user_id: Any, k: int = 10) -> list[Any]:
        """Return up to `k` recommended item ids for `user_id`."""
