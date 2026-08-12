"""Heuristic planner: DataProfile -> ranked architecture shortlist + rationale.

This is intentionally a simple, readable heuristic for Phase 0, not a learned
model. The entire point of Phase 0 is to run it against public benchmarks and
see whether "why" it gives actually tracks which architecture wins in
practice. If it doesn't, this file is what gets replaced first — everything
downstream (sandbox UI, eval harness) is built to not care how this function
is implemented internally, only that it returns a Recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reclab.architectures import REGISTRY
from reclab.data_profiler import DataProfile

# Thresholds are deliberately named and centralized so Phase 0 benchmark runs
# can tune them from evidence instead of digging through scoring logic.
SPARSITY_HIGH = 0.98
COLD_START_HIGH = 0.3
SEQUENCE_LENGTH_MIN_FOR_SASREC = 8

# How close the #1 and #2 scores need to be before the shortlist's top pick
# is flagged as low-confidence rather than presented flatly as "the" answer.
# Set from a real case, not a guess: on Amazon Reviews' All_Beauty category
# the planner picked two_tower (0.50) over hybrid_llm/sasrec (0.35 each) — a
# 0.15 margin — and that pick was wrong (see benchmarks/README.md). The
# margin was visible in the score all along; nothing surfaced it as a close
# call. This constant is that margin, not retuned scoring weights — changing
# *how* architectures get scored on n=4 real/quasi-real data points would be
# overfitting to a handful of runs, not calibration.
#
# Set to 0.16, not 0.15, deliberately: 0.50 - 0.35 is 0.15000000000000002 in
# float arithmetic, not exactly 0.15, and a strict "<" comparison against
# 0.15 would miss the very case this threshold is modeled on. 0.16 catches
# any margin that's "0.15" in the real-world sense regardless of which way
# floating-point error happens to round.
LOW_CONFIDENCE_MARGIN = 0.16


@dataclass
class ScoreFactor:
    """One factor that moved an architecture's score — the structured form
    of what `Recommendation.rationale` says in prose. `effect` is the signed
    score delta this factor contributed (0.0 for a factor that's descriptive
    but didn't change the score, e.g. the base "solid baseline" line)."""

    detail: str
    effect: float = 0.0


@dataclass
class Recommendation:
    """One architecture's ranked position plus why it's there.

    `rationale` is the prose form (semicolon-joined `factors[i].detail`) —
    kept for CLI output and backward compatibility. `factors` is the
    structured form a UI should render instead of parsing `rationale`
    client-side (see docs/architecture/ui-ux-plan.md section 5's open
    question — this is what resolves it)."""

    architecture: str
    rank: int
    score: float
    rationale: str
    factors: list[ScoreFactor] = field(default_factory=list)
    # Score gap to the next-ranked architecture; None for the last rank
    # (nothing below it to compare against).
    margin_to_next: float | None = None
    # True when margin_to_next is thin enough that this pick shouldn't be
    # read as confident — only ever set on rank 1, where "confidence" is a
    # meaningful question (it's about whether *this* is really the answer).
    low_confidence: bool = False


@dataclass
class _Scored:
    """Same as Recommendation minus the fields recommend_architectures fills
    in after sorting (rank, margin_to_next, low_confidence) — an individual
    _score_* function can't know those without seeing every other score."""

    architecture: str
    score: float
    factors: list[ScoreFactor]

    @property
    def rationale(self) -> str:
        return "; ".join(f.detail for f in self.factors)


def recommend_architectures(profile: DataProfile) -> list[Recommendation]:
    """Score every registered architecture against a data profile and return
    them ranked best-first."""
    scored = [_score(name, profile) for name in REGISTRY]
    scored.sort(key=lambda r: r.score, reverse=True)

    recommendations = []
    for i, r in enumerate(scored):
        margin_to_next = scored[i].score - scored[i + 1].score if i + 1 < len(scored) else None
        recommendations.append(
            Recommendation(
                architecture=r.architecture,
                rank=i + 1,
                score=r.score,
                rationale=r.rationale,
                factors=r.factors,
                margin_to_next=margin_to_next,
                low_confidence=(
                    i == 0 and margin_to_next is not None and margin_to_next < LOW_CONFIDENCE_MARGIN
                ),
            )
        )
    return recommendations


def _score(name: str, profile: DataProfile) -> _Scored:
    if name == "two_tower":
        return _score_two_tower(profile)
    if name == "sasrec":
        return _score_sasrec(profile)
    if name == "hybrid_llm":
        return _score_hybrid_llm(profile)
    raise ValueError(f"No scoring rule registered for architecture '{name}'")


def _score_two_tower(profile: DataProfile) -> _Scored:
    score = 0.5
    factors = [
        ScoreFactor(detail="solid general-purpose baseline for dense collaborative signal")
    ]

    if profile.sparsity > SPARSITY_HIGH:
        score -= 0.2
        factors.append(
            ScoreFactor(
                detail="penalized: very sparse interactions weaken pure collaborative filtering",
                effect=-0.2,
            )
        )
    if profile.cold_start_ratio > COLD_START_HIGH:
        score -= 0.2
        factors.append(
            ScoreFactor(
                detail="penalized: high cold-start ratio with no content signal to fall back on",
                effect=-0.2,
            )
        )

    return _Scored(architecture="two_tower", score=score, factors=factors)


def _score_sasrec(profile: DataProfile) -> _Scored:
    score = 0.5
    factors = []

    if profile.median_sequence_length >= SEQUENCE_LENGTH_MIN_FOR_SASREC:
        score += 0.25
        factors.append(
            ScoreFactor(
                detail=(
                    f"boosted: median sequence length {profile.median_sequence_length:.1f} "
                    "gives the sequence model enough signal to use"
                ),
                effect=0.25,
            )
        )
    else:
        score -= 0.15
        factors.append(
            ScoreFactor(
                detail=(
                    f"penalized: median sequence length {profile.median_sequence_length:.1f} is "
                    "short, limiting what a sequential model can learn over a two-tower baseline"
                ),
                effect=-0.15,
            )
        )

    if profile.cold_start_ratio > COLD_START_HIGH:
        score -= 0.15
        factors.append(
            ScoreFactor(
                detail="penalized: still no native answer for cold-start items", effect=-0.15
            )
        )

    return _Scored(architecture="sasrec", score=score, factors=factors)


def _score_hybrid_llm(profile: DataProfile) -> _Scored:
    score = 0.35  # starts lower: highest cost/complexity, needs to earn the upgrade
    factors = []

    if profile.cold_start_ratio > COLD_START_HIGH and profile.has_item_text:
        score += 0.4
        factors.append(
            ScoreFactor(
                detail=(
                    "boosted: high cold-start ratio plus usable item text is exactly the "
                    "gap an LLM re-ranker fills that collaborative models can't"
                ),
                effect=0.4,
            )
        )
    elif profile.cold_start_ratio > COLD_START_HIGH and not profile.has_item_text:
        score -= 0.1
        factors.append(
            ScoreFactor(
                detail=(
                    "penalized: high cold-start ratio, but no item text available to re-rank "
                    "on — the usual reason to reach for this architecture doesn't apply "
                    "without metadata"
                ),
                effect=-0.1,
            )
        )
    elif not profile.has_item_text:
        score -= 0.15
        factors.append(
            ScoreFactor(
                detail="penalized: no item text metadata to justify the added complexity/cost",
                effect=-0.15,
            )
        )

    if profile.sparsity > SPARSITY_HIGH and profile.has_item_text:
        score += 0.1
        factors.append(
            ScoreFactor(
                detail="boosted: content signal helps compensate for very sparse interactions",
                effect=0.1,
            )
        )

    if not factors:
        factors.append(
            ScoreFactor(detail="no strong signal either way; likely not worth the extra cost here")
        )

    return _Scored(architecture="hybrid_llm", score=score, factors=factors)
