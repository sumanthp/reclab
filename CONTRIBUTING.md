# Contributing to reclab

reclab is early — right now it's mostly a solo effort validating whether the
reasoning engine's architecture recommendations hold up on real data (see
`docs/architecture/mvp-plan.md`). Contributions are welcome, but the most
valuable ones right now are things that test the core claim, not new surface
area.

## Dev setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
git clone https://github.com/<your-username>/reclab.git
cd reclab
uv sync
uv run pytest
```

No extras needed — there's deliberately no deep-learning framework dependency
(see the comment in `pyproject.toml` and the module docstring in
`architectures/sasrec.py` for why: the current PyTorch wheel on PyPI requires
CUDA runtime libraries even for CPU-only use, which wasn't installable in the
environment this repo was built in). If you want to add a PyTorch/HF-backed
architecture as an alternative, that's welcome — the `Architecture` interface
doesn't care what's underneath it — just don't make it a dependency of the
default install path.

Lint and type-check before opening a PR:

```bash
uv run ruff check .
uv run mypy src
```

## What's most useful to contribute right now

1. **Running this against a real public dataset.** All three architectures
   train and evaluate for real now, but only against a synthetic dataset —
   the sandbox this repo was built in can't reach files.grouplens.org,
   huggingface.co, or S3. If you can run
   `scripts/run_benchmark.py --movielens-100k path/to/ml-100k.zip` on a
   machine with normal internet access, the result (or the parsing bug it
   surfaces) is the single most valuable contribution possible right now.
   See `benchmarks/README.md`.
2. **Calibrating the reasoning engine.** `benchmarks/README.md` documents a
   real finding: the planner's shortlist ranking conflates "best overall"
   with "best on the dimension it's reasoning about" (e.g. `hybrid_llm`
   correctly wins on cold-start recall but the planner's ranking implies it
   should win on Recall@K overall, which it doesn't at this scale). Scoring
   architectures against the metric each rationale actually invokes, instead
   of one blended ranking, is the concrete next step — see
   `src/reclab/reasoning_engine/planner.py`.
3. **A fourth architecture** (a GNN-based approach is the natural next
   candidate — see the MVP plan's out-of-scope-for-v0.1 list). Same interface,
   same registration process as the three that exist.
4. **A real LLM re-ranker** for `hybrid_llm`, implementing the `Reranker`
   interface in `src/reclab/architectures/rerankers.py` against an actual LLM
   API, as an alternative to the default TF-IDF lexical one.

## Adding a new architecture

1. Create `src/reclab/architectures/<name>.py` implementing the `Architecture`
   base class (`src/reclab/architectures/base.py`): a static `info()` and
   instance `fit()` / `recommend()`.
2. Register it in `src/reclab/architectures/__init__.py`'s `REGISTRY`.
3. Add a scoring function for it in
   `src/reclab/reasoning_engine/planner.py` (`_score_<name>`), so the
   reasoning engine can actually reason about when to suggest it.
4. Add tests under `tests/architectures/` mirroring the existing ones.

## Code style

- Formatting/linting via `ruff` (config in `pyproject.toml`) — run
  `uv run ruff check .` before committing.
- Type hints throughout; `mypy` runs in CI.
- Keep the `Architecture` interface (`base.py`) small. If a change to it is
  needed to support a new architecture, that's worth a design discussion in
  an issue first, since every existing architecture and the eval harness
  depend on it staying stable.

## Pull requests

- Keep PRs scoped to one thing (one architecture, one bug fix, one benchmark
  run) — easier to review and easier to revert if something's wrong.
- CI (lint, type-check, tests) must pass.
- For anything touching the reasoning engine's scoring logic, include the
  benchmark evidence that motivated the change, not just the code diff.

## Reporting issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For reasoning-engine
mismatches specifically, include the data profile and the actual measured
architecture ranking — that's the fastest way to make the report actionable.
