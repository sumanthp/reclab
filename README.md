# reclab

**An open-source platform for reasoning about, configuring, and benchmarking recommendation system architectures against your own data.**

Most rec-sys tooling makes you choose upfront: a managed black box (Amazon Personalize, Vertex AI, Algolia Recommend) that hides the architecture from you, or a research library (RecBole, Transformers4Rec, NVIDIA Merlin) that hands you dozens of models with no guidance on which one fits your data. reclab sits in between — it profiles your dataset, reasons about which architectures are likely to work well and why, and lets you compare candidates side by side before you commit to one.

Self-hosted by design: `docker compose up` locally, or deploy the same containers on any cloud. Your data never leaves your machine.

## What it does

1. **Profile** — point reclab at your interaction data; it profiles sparsity, cold-start ratio, item metadata richness, interaction volume, and sequence length.
2. **Reason** — a planner maps that profile to a ranked shortlist of candidate architectures, with a plain-language rationale for each — not just a score.
3. **Compare** — evaluate candidates side by side on your actual data: Recall@K, NDCG@K, coverage, cold-start slice performance.

## Status: Phase 0/1 — reasoning engine validated, with an honest gap found

Phase 0 asked: **does the reasoning engine's shortlist actually track which architecture wins, or is it just a plausible-sounding heuristic?** The answer, after actually training and evaluating all three architectures on real (synthetic, see below) data: partially, and the gap is documented rather than hidden. See [`benchmarks/README.md`](benchmarks/README.md) for the full writeup — short version: the planner correctly identifies *which architecture is best on the specific dimension its rationale invokes* (e.g. `hybrid_llm` does win on cold-start recall exactly when the planner says it should), but its shortlist *ranking* currently conflates that with "best overall," which a single Recall@K comparison doesn't validate. That's a concrete, non-cosmetic next step for the planner, not a passing grade.

What's real today:
- Data profiling (`src/reclab/data_profiler/`) — computes the signals the planner reasons over.
- The reasoning engine (`src/reclab/reasoning_engine/`) — a heuristic planner scoring three candidate architectures against a data profile.
- **All three architectures actually train and recommend** (`src/reclab/architectures/`) — real NumPy implementations (BPR matrix factorization, a hand-derived self-attention sequential model with backprop checked against numerical gradients, and a hybrid encoder + pluggable re-ranker), not stubs. No PyTorch dependency — see the architectures' module docstrings and `pyproject.toml` for why.
- A full eval harness (`src/reclab/eval/`) — temporal train/test splitting, Recall@K, NDCG@K, coverage, and two cold-start metrics.
- A synthetic dataset generator (`src/reclab/datasets/synthetic.py`) with controllable sparsity, cold-start ratio, and item-text structure — used because the real public benchmarks aren't reachable from this project's development environment (see below).
- A FastAPI service (`src/reclab/api/`) exposing profiling + reasoning over HTTP.
- `scripts/run_benchmark.py` and `scripts/demo.sh` — the actual train-and-evaluate pipeline, runnable in under a minute, no GPU or API keys required.

What's not done yet (see [`CONTRIBUTING.md`](CONTRIBUTING.md)):
- **Validation against real public datasets** (MovieLens, Amazon Reviews). The loader for MovieLens 100K is written and unit-tested against fixture files but not run against the real dataset — the sandbox this repo was built in can't reach files.grouplens.org/huggingface.co/S3. Running it on a machine that can, and reporting back, is the highest-value open item.
- Calibrating the reasoning engine's ranking logic against the multi-metric finding above.
- Any UI. The dashboard/sandbox described in the plan doc is intentionally sequenced *after* this validation work, not before.

## Candidate architectures

| Architecture | Idea | Good for |
|---|---|---|
| `two_tower` | Separate user/item embedding towers, BPR pairwise loss (real matrix factorization) | Dense collaborative signal, fast to train/serve |
| `sasrec` | Single-head causal self-attention over interaction history, next-item prediction | Longer per-user sequences where order/recency matters |
| `hybrid_llm` | SASRec-style encoder + pluggable re-ranker (default: TF-IDF lexical similarity, no API key needed) with reserved cold-item exploration slots | Sparse data, high cold-start ratio, rich item text |

`hybrid_llm`'s re-ranker is an interface, not a fixed implementation — swapping in a real LLM API call is meant to be a small, obvious change (`HybridLLM(reranker=YourLLMReranker())`); see `src/reclab/architectures/rerankers.py`.

## Repo layout

```
src/reclab/
  data_profiler/       # dataset profiling (sparsity, cold-start ratio, metadata richness, sequence length)
  datasets/             # synthetic dataset generator + real dataset loaders (MovieLens, Amazon Reviews)
  architectures/        # candidate architectures behind a common interface (two_tower, sasrec, hybrid_llm)
  reasoning_engine/      # planner: data profile -> ranked architecture shortlist + rationale
  eval/                 # temporal train/test split + offline evaluation harness
  api/                  # FastAPI service exposing profiling/reasoning over HTTP
frontend/               # React + TypeScript UI (not started — sequenced after this validation work)
benchmarks/             # benchmark results, checked in and reproducible, with an honest write-up of findings
scripts/                # run_benchmark.py (train+eval pipeline) and demo.sh (the actual demo)
docs/architecture/       # design docs, including the full MVP plan
tests/                  # mirrors src/ layout, including a numerical-gradient check for SASRec's backprop
```

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+. No GPU, no API keys, no external services.

```bash
git clone https://github.com/<your-username>/reclab.git
cd reclab
uv sync
uv run pytest
```

Run the actual demo — profiles a dataset, gets the reasoning engine's shortlist, then trains and evaluates all three architectures, twice (a typical scenario and a cold-start-heavy one):

```bash
bash scripts/demo.sh
```

Or against your own interaction data (a CSV with at least `user_id`, `item_id`, and `timestamp` columns for full train/eval; `timestamp` is optional if you only want the profile + shortlist):

```bash
uv run python scripts/run_benchmark.py --csv path/to/interactions.csv
```

Or run the API:

```bash
docker compose up
# then: curl -F "interactions_csv=@path/to/interactions.csv" http://localhost:8000/profile
```

No database setup required for local use — see `docker-compose.yml` for the optional Postgres profile if you're running this on a shared server instead of locally.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a new architecture or extend the reasoning engine, and [`benchmarks/README.md`](benchmarks/README.md) for what the benchmark runs actually found.

## Why this exists

Choosing and validating a recommendation architecture is still mostly tribal knowledge or trial and error, especially for teams without a dedicated ML research function. reclab is an attempt to make that reasoning step explicit, inspectable, and reusable, in the open — built to be legible to a competent ML engineer evaluating it in an afternoon, not a black box.

## License

[Apache 2.0](LICENSE)
