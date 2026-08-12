**Title:** [P] reclab — an open-source tool that reasons about which rec-sys architecture fits your data (and checks its own claims against real eval numbers)

**Post body:**

Sharing a project I've been building: **reclab**, a self-hosted, open-source
(Apache 2.0) tool for reasoning about, configuring, and benchmarking
recommendation system architectures against your own data. Live demo with
real results, no signup: https://sumanthp.github.io/reclab/

**The problem it's trying to solve:** choosing between a two-tower model,
SASRec-style sequential model, or an LLM-hybrid re-ranker is mostly tribal
knowledge or trial and error unless you already have a rec-sys specialist on
the team. reclab profiles your interaction data and proposes a ranked
shortlist of architectures with a plain-language rationale, then lets you
train and evaluate all of them on your own data to check that rationale —
now via a real end-to-end web UI (upload a CSV, watch it train, inspect
results), not just a CLI.

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
  train/eval results on MovieLens 100K and two Amazon Reviews 2023
  categories (`benchmarks/README.md`), not just synthetic data. Finding:
  on MovieLens the top pick matched the measured winner; on both Amazon
  categories it didn't — but the engine's own `low_confidence` signal
  (score margin between rank 1 and rank 2) had flagged both of those exact
  cases *before* I checked. Two for two. I added that signal as a real,
  structured field in the API/UI rather than just a prose caveat, and the
  underlying ranking-calibration gap (the shortlist still conflates "best
  overall" with "best on the dimension its rationale invokes") is
  documented as an open item, not glossed over.
- Running real data also caught an actual bug: `coverage_at_k` could exceed
  its theoretical max of 1.0 for architectures with a candidate pool wider
  than the eval catalog. Fixed, with a regression test built from the real
  case that surfaced it.

**What's not done:** calibrating the reasoning engine's ranking against the
finding above (evidence, not yet a fix); a real LLM re-ranker exists now
(Anthropic API, optional, tested with a mocked client) but hasn't been
benchmarked against the default lexical one on real data yet; only two
Amazon Reviews categories run so far, both small.

Repo: https://github.com/sumanthp/reclab. Feedback on the reasoning
engine's scoring heuristics or the eval methodology especially welcome —
that's the part I'm least confident is fully right yet.
