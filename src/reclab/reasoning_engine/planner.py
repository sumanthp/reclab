"""Heuristic planner: DataProfile -> ranked architecture shortlist + rationale.

This is intentionally a simple, readable heuristic for Phase 0, not a learned
model. The entire point of Phase 0 is to run it against public benchmarks and
see whether "why" it gives actually tracks which architecture wins in
practice. If it doesn't, this file is what gets replaced first — everything
downstream (sandbox UI, eval harness) is built to not care how this function
is implemented internally, only that it returns a Recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass

from reclab.architectures import REGISTRY
from reclab.data_profiler import DataProfile

# Thresholds are deliberately named and centralized so Phase 0 benchmark runs
# can tune them from evidence instead of digging through scoring logic.
SPARSITY_HIGH = 0.98
COLD_START_HIGH = 0.3
SEQUENCE_LENGTH_MIN_FOR_SASREC = 8


@dataclass
class Recommendation:
    """One architecture's ranked position plus the plain-language reason it's
    there. `rationale` is what both the Layer 1 summary and the Layer 3
    technical detail trace back to — one explanation, two renderings."""

    architecture: str
    rank: int
    score: float
    rationale: str


def recommend_architectures(profile: DataProfile) -> list[Recommendation]:
    """Score every registered architecture against a data profile and return
    them ranked best-first."""
    scored = [_score(name, profile) for name in REGISTRY]
    scored.sort(key=lambda r: r.score, reverse=True)
    return [
        Recommendation(
            architecture=r.architecture, rank=i + 1, score=r.score, rationale=r.rationale
        )
        for i, r in enumerate(scored)
    ]


def _score(name: str, profile: DataProfile) -> Recommendation:
    if name == "two_tower":
        return _score_two_tower(profile)
    if name == "sasrec":
        return _score_sasrec(profile)
    if name == "hybrid_llm":
        return _score_hybrid_llm(profile)
    raise ValueError(f"No scoring rule registered for architecture '{name}'")


def _score_two_tower(profile: DataProfile) -> Recommendation:
    score = 0.5
    reasons = ["solid general-purpose baseline for dense collaborative signal"]

    if profile.sparsity > SPARSITY_HIGH:
        score -= 0.2
        reasons.append("penalized: very sparse interactions weaken pure collaborative filtering")
    if profile.cold_start_ratio > COLD_START_HIGH:
        score -= 0.2
        reasons.append("penalized: high cold-start ratio with no content signal to fall back on")

    return Recommendation(
        architecture="two_tower", rank=0, score=score, rationale="; ".join(reasons)
    )


def _score_sasrec(profile: DataProfile) -> Recommendation:
    score = 0.5
    reasons = []

    if profile.median_sequence_length >= SEQUENCE_LENGTH_MIN_FOR_SASREC:
        score += 0.25
        reasons.append(
            f"boosted: median sequence length {profile.median_sequence_length:.1f} "
            "gives the sequence model enough signal to use"
        )
    else:
        score -= 0.15
        reasons.append(
            f"penalized: median sequence length {profile.median_sequence_length:.1f} is short, "
            "limiting what a sequential model can learn over a two-tower baseline"
        )

    if profile.cold_start_ratio > COLD_START_HIGH:
        score -= 0.15
        reasons.append("penalized: still no native answer for cold-start items")

    return Recommendation(architecture="sasrec", rank=0, score=score, rationale="; ".join(reasons))


def _score_hybrid_llm(profile: DataProfile) -> Recommendation:
    score = 0.35  # starts lower: highest cost/complexity, needs to earn the upgrade
    reasons = []

    if profile.cold_start_ratio > COLD_START_HIGH and profile.has_item_text:
        score += 0.4
        reasons.append(
            "boosted: high cold-start ratio plus usable item text is exactly the "
            "gap an LLM re-ranker fills that collaborative models can't"
        )
    elif profile.cold_start_ratio > COLD_START_HIGH and not profile.has_item_text:
        score -= 0.1
        reasons.append(
            "penalized: high cold-start ratio, but no item text available to re-rank on — "
            "the usual reason to reach for this architecture doesn't apply without metadata"
        )
    elif not profile.has_item_text:
        score -= 0.15
        reasons.append("penalized: no item text metadata to justify the added complexity/cost")

    if profile.sparsity > SPARSITY_HIGH and profile.has_item_text:
        score += 0.1
        reasons.append("boosted: content signal helps compensate for very sparse interactions")

    if not reasons:
        reasons.append("no strong signal either way; likely not worth the extra cost here")

    return Recommendation(
        architecture="hybrid_llm", rank=0, score=score, rationale="; ".join(reasons)
    )
