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
your own data. Self-hosted, Apache 2.0.

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
  `benchmarks/README.md` has the actual finding from doing that: the planner
  correctly identifies which architecture wins on the *specific* dimension
  its rationale invokes (e.g., the hybrid architecture really does win on
  cold-start recall exactly when it says it should), but its overall ranking
  currently conflates that with "best on every metric," which isn't true at
  the scale I tested. Logged as an open finding, not smoothed over.
- `scripts/demo.sh` runs the whole pipeline — profile, reason, train, evaluate
  — in under a minute on a laptop CPU, no GPU/API keys/cloud account needed.

Real public benchmark validation (MovieLens, Amazon Reviews) is the biggest
open item — my dev sandbox couldn't reach the hosting domains, so right now
it's validated against a synthetic dataset generator with controllable
sparsity/cold-start/sequence-length. If anyone runs it against MovieLens and
it breaks (or works), I'd genuinely like to know.

Repo: <github URL once pushed>

Happy to answer questions about the reasoning engine's scoring logic, the
hand-derived attention backprop, or anything else.
