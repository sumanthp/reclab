# reclab

**An open-source platform for reasoning about, configuring, and benchmarking recommendation system architectures against your own data.**

Most rec-sys tooling makes you choose upfront: a managed black box (Amazon Personalize, Vertex AI, Algolia Recommend) that hides the architecture from you, or a research library (RecBole, Transformers4Rec, NVIDIA Merlin) that hands you dozens of models with no guidance on which one fits your data. reclab sits in between — it profiles your dataset, reasons about which architectures are likely to work well and why, and lets you compare candidates side by side before you commit to one.

Self-hosted by design: `docker compose up` locally, or deploy the same containers on any cloud. Your data never leaves your machine.

## What it does

1. **Profile** — point reclab at your interaction data; it profiles sparsity, cold-start ratio, item metadata richness, interaction volume, and sequence length.
2. **Reason** — a planner maps that profile to a ranked shortlist of candidate architectures, with a plain-language rationale for each — not just a score.
3. **Compare** — evaluate candidates side by side on your actual data: Recall@K, NDCG@K, coverage, cold-start slice performance.

## Status: Phase 0/1 done, plus a working end-to-end UI ahead of schedule

Phase 0/1 (reasoning engine validated, all three architectures real) is done — see below. The dashboard/sandbox UI was originally sequenced *after* the Phase 2 launch (`docs/architecture/mvp-plan.md` section 9), but a working version exists now: `docker compose up`, upload a CSV, and the whole profile → shortlist → compare → results loop runs against the live API in the browser, with run history and cancellation. A post-launch re-audit of the whole platform (backend, frontend, ops) also caught and fixed a real correctness bug — `/profile` couldn't see uploaded item metadata at all, so the shortlist's `hybrid_llm` rationale was always penalized for "no item text" even when metadata was provided a moment later for `/compare`. What's *not* built yet is the fuller dual-audience dashboard from `docs/architecture/ui-ux-plan.md` (structured rationale detail, a sample of actual recommended items) — see "what's not done" below.

Phase 0 asked: **does the reasoning engine's shortlist actually track which architecture wins, or is it just a plausible-sounding heuristic?** The answer, after training and evaluating all three architectures on real MovieLens 100K data, a real Amazon Reviews 2023 category, and synthetic data: it's mixed, and that's reported rather than hidden. See [`benchmarks/README.md`](benchmarks/README.md) for the full writeup — short version: on MovieLens, the planner's #1 pick (`sasrec`) matched the measured overall winner on Recall@K and NDCG@K. On Amazon Reviews' `All_Beauty` category, it didn't — the planner itself flagged that case as low-confidence ("no strong signal either way"), and the guess was wrong. On the synthetic scenarios, the planner correctly identifies *which architecture is best on the specific dimension its rationale invokes* (e.g. `hybrid_llm` does win on cold-start recall exactly when the planner says it should), but its shortlist *ranking* currently conflates that with "best overall," which a single Recall@K comparison doesn't validate. That's a concrete, non-cosmetic next step for the planner, not a passing grade.

What's real today:
- Data profiling (`src/reclab/data_profiler/`) — computes the signals the planner reasons over.
- The reasoning engine (`src/reclab/reasoning_engine/`) — a heuristic planner scoring three candidate architectures against a data profile.
- **All three architectures actually train and recommend** (`src/reclab/architectures/`) — real NumPy implementations (BPR matrix factorization, a hand-derived self-attention sequential model with backprop checked against numerical gradients, and a hybrid encoder + pluggable re-ranker), not stubs. No PyTorch dependency — see the architectures' module docstrings and `pyproject.toml` for why.
- A full eval harness (`src/reclab/eval/`) — temporal train/test splitting, Recall@K, NDCG@K, coverage, and two cold-start metrics.
- A synthetic dataset generator (`src/reclab/datasets/synthetic.py`) with controllable sparsity, cold-start ratio, and item-text structure, plus real loaders for MovieLens 100K and Amazon Reviews 2023 (`src/reclab/datasets/loaders.py`) — all exercised end-to-end against real downloads (see [`benchmarks/README.md`](benchmarks/README.md)).
- A FastAPI service (`src/reclab/api/`) exposing profiling, reasoning, and a full async train+eval comparison job (`/profile`, `/compare`, `/runs`, `/runs/{id}`, `/runs/{id}/cancel`, `/architectures`) over HTTP.
- **A working end-to-end web UI** (`frontend/`, React + TypeScript) — upload a CSV, see the profile and shortlist (with each architecture's strengths/weaknesses/cost profile from `/architectures`), kick off the full architecture comparison, cancel a run mid-flight, and browse past runs — all against the live API, not a mockup. Not the full dual-audience dashboard from the design doc (`docs/architecture/ui-ux-plan.md`) yet — the shortlist's rationale is still prose, not structured (see the plan's open question).
- `scripts/run_benchmark.py` and `scripts/demo.sh` — the actual train-and-evaluate pipeline, runnable in under a minute, no GPU or API keys required.
- Operational basics for running this beyond a single quick local try: upload size limits, a concurrency cap on training jobs (extra requests queue instead of piling up threads), job-store retention pruning, and a `docker-compose` health-check gate — see [Configuration](#configuration) below. A frontend test suite (32 Vitest + React Testing Library tests) now sits alongside the backend's 86 pytest tests, both run in CI.

What's not done yet (see [`CONTRIBUTING.md`](CONTRIBUTING.md)):
- **Calibrating the reasoning engine's ranking logic** against the multi-metric finding above — both real-dataset runs now point at the same underlying gap (shortlist conflates "best overall" with "best on the dimension its own rationale invokes").
- **A `coverage_at_k` definition fix** for architectures (like `hybrid_llm`) whose candidate pool isn't limited to the train/test interaction catalog — see the Amazon Reviews findings in `benchmarks/README.md`.
- The rest of the dual-audience dashboard from `docs/architecture/ui-ux-plan.md` — a structured (not parsed-prose) rationale breakdown, and a sample of actual recommended items per test user.

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
  api/                  # FastAPI service: /profile, /compare (async job), /runs/{id}, /architectures
frontend/               # React + TypeScript UI — upload, profile, shortlist, compare, results
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

Or run the full stack — API + web UI — with Docker:

```bash
docker compose up
# then open http://localhost:5173
```

Upload an interactions CSV (needs at least `user_id`/`item_id` columns, plus `timestamp` if you want to run the full comparison, not just the profile + shortlist) and, optionally, an item metadata CSV (`item_id`, `description` columns) to enable `hybrid_llm`'s item-text signal. Or hit the API directly:

```bash
curl -F "interactions_csv=@path/to/interactions.csv" http://localhost:8000/profile
```

No database setup required for local use — see `docker-compose.yml` for the optional Postgres profile if you're running this on a shared server instead of locally.

For frontend development with hot reload instead of the Docker build, run the two pieces separately (requires [Node.js](https://nodejs.org/) 20+):

```bash
uv run uvicorn reclab.api.main:app --reload   # terminal 1, http://localhost:8000
cd frontend && npm install && npm run dev     # terminal 2, http://localhost:5173
```

Vite's dev server proxies API calls to `localhost:8000` (see `frontend/vite.config.ts`) — no CORS setup needed.

### Configuration

All optional, all environment variables read by the API (`src/reclab/api/main.py`, `src/reclab/api/jobs.py`) — none need to be set for local/single-user use, the defaults are sized for that case:

| Variable | Default | What it controls |
|---|---|---|
| `RECLAB_STORAGE` | `sqlite:///./reclab.db` | Where job state (`/compare` run history) persists. |
| `RECLAB_CORS_ORIGINS` | `*` | Comma-separated allowed origins. Permissive by default since this is a self-hosted single-user tool with no auth; lock it down for anything more exposed. |
| `RECLAB_MAX_UPLOAD_MB` | `100` | Per-file upload size limit for `/profile` and `/compare`. |
| `RECLAB_MAX_CONCURRENT_JOBS` | `2` | How many `/compare` training jobs can run at once; extra requests queue (status `pending`) rather than piling up unbounded background threads. |
| `RECLAB_MAX_RETAINED_RUNS` | `200` | Oldest runs beyond this count are pruned whenever a new one starts, so the job store doesn't grow forever. |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for how to add a new architecture or extend the reasoning engine, and [`benchmarks/README.md`](benchmarks/README.md) for what the benchmark runs actually found.

## Why this exists

Choosing and validating a recommendation architecture is still mostly tribal knowledge or trial and error, especially for teams without a dedicated ML research function. reclab is an attempt to make that reasoning step explicit, inspectable, and reusable, in the open — built to be legible to a competent ML engineer evaluating it in an afternoon, not a black box.

## License

[Apache 2.0](LICENSE)
