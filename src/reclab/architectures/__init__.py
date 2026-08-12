"""Candidate recommendation architectures, all implementing a common interface.

Phase 0 ships three: `two_tower` (baseline), `sasrec` (sequential transformer),
and `hybrid_llm` (SASRec encoder + LLM re-ranker). See CONTRIBUTING.md for how
to add a new one — the reasoning engine and eval harness only depend on the
`Architecture` interface in `base.py`, not on any specific implementation.
"""

from reclab.architectures.base import Architecture, ArchitectureInfo
from reclab.architectures.hybrid_llm import HybridLLM
from reclab.architectures.sasrec import SASRec
from reclab.architectures.two_tower import TwoTower

# Registry the reasoning engine and API use to look up architectures by name.
REGISTRY: dict[str, type[Architecture]] = {
    "two_tower": TwoTower,
    "sasrec": SASRec,
    "hybrid_llm": HybridLLM,
}

__all__ = [
    "Architecture",
    "ArchitectureInfo",
    "TwoTower",
    "SASRec",
    "HybridLLM",
    "REGISTRY",
]
