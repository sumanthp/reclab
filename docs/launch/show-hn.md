**Title:** Show HN: reclab – reasons about which rec-sys architecture fits your data, then proves it

**Post body:**

I kept running into the same problem: you can find dozens of recommendation
system architectures (two-tower, SASRec-style sequential models, LLM-hybrid
re-rankers) but almost nothing that helps you reason about *which one fits
your actual data* before you commit weeks to building it. Managed services
(Amazon Personalize, Vertex AI) hide the architecture entirely; research
libraries (RecBole, Transformers4Rec) hand you 90 models with no guidance.

reclab profiles your interaction data (sparsity, cold-start ratio, sequence
length, item text availability), reasons about which architectures are
likely to work well and why, and lets you compare candidates side by side on
your own data. Self-hosted, Apache 2.0. Live demo with real results, no
signup: https://sumanthp.github.io/reclab/

A few things I think are worth a look if you're into this kind of thing:

- All three candidate architectures (BPR matrix factorization, a causal
  self-attention sequential model, and a hybrid encoder + re-ranker) are
  implemented from scratch in plain NumPy — no PyTorch. Not by choice
  initially: the current PyTorch wheel on PyPI requires CUDA runtime
  libraries even for CPU-only use, which wasn't installable in my dev
  environment. So the self-attention model's backward pass is hand-derived,
  and checked against numerical gradients in
  `tests/architectures/test_sasrec_gradients.py` — I wanted actual proof it's
  correct, not just that training doesn't crash.
- The eval harness (temporal train/test split, Recall@K, NDCG@K, coverage,
  and two different cold-start metrics) exists to check the reasoning
  engine's own claims against measured results, not just to produce a demo.
  `benchmarks/README.md` has the actual findings from doing that on
  MovieLens 100K and two Amazon Reviews 2023 categories: on MovieLens the
  planner's top pick matched the measured winner; on both Amazon
  categories it didn't. But the planner's own confidence signal — a real
  `low_confidence` flag computed from the score margin between its #1 and
  #2 pick — had already flagged both of those exact cases before I checked
  whether they were right. Two for two. That's now a real field in the API
  and UI, not a post-hoc excuse in a README, and the deeper issue it points
  at (the shortlist ranking conflates "best overall" with "best on the
  specific dimension its own rationale invokes") is logged as an open
  finding, not smoothed over.
- Running real data also caught a genuine bug in my own eval code:
  `coverage_at_k` could score above its theoretical max of 1.0 for an
  architecture whose candidate pool is wider than the eval catalog. Fixed,
  with a regression test built from the exact case that caught it.
- What started as a CLI-only reasoning engine is now a real end-to-end
  product: a FastAPI backend with an async job queue for training runs, a
  React frontend (upload → profile → shortlist → compare → results, with
  run history and mid-run cancellation), a static GitHub Pages demo
  reusing the same components against precomputed real results, structured
  JSON request logging, and a Playwright suite that drives the actual
  backend and frontend together as an integration check, alongside ~150
  unit tests across both.
- `scripts/demo.sh` still runs the whole reasoning-engine pipeline —
  profile, reason, train, evaluate — in under a minute on a laptop CPU, no
  GPU/API keys/cloud account needed.

Biggest open items: calibrating the reasoning engine's ranking against the
finding above, and running against a genuinely denser Amazon Reviews
category (the two I've run so far are both small). A real LLM re-ranker
also exists now (optional, via the Anthropic API) but hasn't been
benchmarked against the default lexical one yet.

Repo: https://github.com/sumanthp/reclab

Happy to answer questions about the reasoning engine's scoring logic, the
hand-derived attention backprop, the low-confidence signal, or anything
else.
