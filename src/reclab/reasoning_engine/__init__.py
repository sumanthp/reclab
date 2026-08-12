"""The reasoning engine: maps a DataProfile to a ranked shortlist of candidate
architectures with a plain-language rationale for each.

Phase 0 goal: validate that this heuristic planner's ranking actually tracks
which architecture wins on public benchmarks (MovieLens, Amazon Reviews)
before any UI or sandbox is built on top of it. If it doesn't hold up, this
module gets rebuilt before anything else does.
"""

from reclab.reasoning_engine.planner import Recommendation, recommend_architectures

__all__ = ["Recommendation", "recommend_architectures"]
