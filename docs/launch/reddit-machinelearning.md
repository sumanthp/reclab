**Title:** [P] reclab — an open-source tool that reasons about which rec-sys architecture fits your data (and checks its own claims against real eval numbers)

**Post body:**

Sharing a project I've been building: **reclab**, a self-hosted, open-source
(Apache 2.0) tool for reasoning about, configuring, and benchmarking
recommendation system architectures against your own data.

**The problem it's trying to solve:** choosing between a two-tower model,
SASRec-style sequential model, or an LLM-hybrid re-ranker is mostly tribal
knowledge or trial and error unless you already have a rec-sys specialist on
the team. reclab profiles your interaction data and proposes a ranked
shortlist of architectures with a plain-language rationale, then lets you
train and evaluate all of them on your own data to check that rationale.

**Implementation notes that might interest this sub specifically:**

- All three architectures — BPR matrix factorization, a single-head causal
  self-attention sequential model (SASRec-style), and a hybrid encoder +
  pluggable re-ranker — are implemented in plain NumPy, including a
  hand-derived backward pass for the self-attention block. I validated it
  against numerical gradients rather than trusting the derivation
  (`tests/architectures/test_sasrec_gradients.py`).
- The eval harness does a proper temporal (leave-last-item-out) train/test
  split and reports Recall@K, NDCG@K, coverage, plus two cold-start metrics
  (exact-match recall on held-out cold items, and a softer "did any cold item
  get surfaced at all" rate — I added the second one after finding the first
  was too high-variance at small sample sizes to be useful on its own).
- I ran the reasoning engine's shortlist against actual measured
  train/eval results (`benchmarks/README.md`) rather than just asserting it
  works. Finding: the planner correctly identifies which architecture wins
  on the specific metric its own rationale is about (e.g., cold-start recall
  for the hybrid architecture), but its ranking conflates that with overall
  Recall@K, which doesn't hold at the scale I tested. That's flagged as an
  open item, not glossed over.

**What's not done:** real public benchmark validation (MovieLens, Amazon
Reviews) — currently validated against a synthetic dataset generator with
controllable sparsity/cold-start-ratio/sequence-length, since my dev
environment couldn't reach the dataset hosts. The MovieLens loader is
written and unit-tested against fixture files but unverified end-to-end.

Repo: <github URL once pushed>. Feedback on the reasoning engine's scoring
heuristics or the eval methodology especially welcome — that's the part I'm
least confident is fully right yet.
