# benchmarks

This directory is the credibility mechanism for reclab's core claim: that the
reasoning engine's ranked shortlist actually tracks which architecture
performs best on data, not just a plausible-sounding heuristic. Phase 0's job
was to test that claim, honestly, and report what it finds — including when
the answer is "not yet."

## What goes here

- `../scripts/run_benchmark.py` profiles a dataset, asks the reasoning engine
  for a ranked shortlist, then genuinely trains and evaluates all three
  architectures on a temporal train/test split so the shortlist can be
  checked against measured results.
- `results/` (checked in) holds the output of each run: the data profile, the
  shortlist + rationale, and the actual Recall@K / NDCG@K / coverage /
  cold-start numbers per architecture.
- `../scripts/demo.sh` runs both scenarios below in one command — the actual
  demo for this project, not a screen recording (see the MVP plan's Phase 2
  notes on why a real, runnable demo matters more than a GIF here).

## Datasets

**Public benchmarks (MovieLens, Amazon Reviews) are not yet run.** The
development sandbox this repo was built in can't reach files.grouplens.org,
huggingface.co, or S3, so `reclab.datasets.loaders.load_movielens_100k` is
written and unit-tested against fixture files that mimic the real format, but
not verified end-to-end against the real dataset. Running
`scripts/run_benchmark.py --movielens-100k path/to/ml-100k.zip` on a machine
with normal internet access and reporting back — success or a parsing bug —
is the single most valuable open item for Phase 0. See CONTRIBUTING.md.

In the meantime, `reclab.datasets.synthetic` generates a reproducible dataset
with controllable sparsity, cold-start ratio, sequence length, and item-text
structure. Two scenarios are checked in:

| Result file | Scenario | What it's testing |
|---|---|---|
| `synthetic-default.json` | Moderate sparsity, 30% cold-start ratio, median sequence length 15 | Does the planner correctly favor a sequence-aware architecture when sequences are informative? |
| `synthetic-cold-start-heavy.json` | Sparser, 40% cold-start ratio, same item text | Does the planner correctly favor `hybrid_llm` when cold-start + item text is the dominant characteristic? |

## What Phase 0 actually found

Both scenarios were run with the tuned defaults in each architecture class
(`embedding_dim=16, epochs=40, learning_rate=0.05` — reached by checking that
under-training wasn't the reason SASRec looked weak; see git history of
`two_tower.py`/`sasrec.py`/`hybrid_llm.py` defaults). Two real, non-cherry-picked
findings came out of it, both left as-is rather than tuned away:

**1. On Recall@10 alone, the planner's #1 pick did not win in either scenario.**
`two_tower` had the best Recall@10 in both the default run (0.220 vs sasrec's
0.196 vs hybrid_llm's 0.176) and the cold-start-heavy run (0.242 vs sasrec's
0.238 vs hybrid_llm's 0.218), even though the planner picked `sasrec` and
`hybrid_llm` respectively. At this dataset scale (500 users, 300 items), a
well-tuned BPR matrix-factorization baseline is genuinely hard to beat — which
is itself a legitimate, well-known result in recommendation systems (simple
baselines often win at small-to-moderate scale), not a sign the other
architectures are broken. **This is an open calibration gap in the reasoning
engine's heuristic thresholds** (`src/reclab/reasoning_engine/planner.py`),
flagged rather than hidden.

**2. But on ColdStartRecall@10, the planner's rationale holds up exactly.**
`hybrid_llm` had the best cold-start recall in both scenarios (0.105 and 0.222
respectively, vs. 0.000 for both `two_tower` and `sasrec` in every run) and a
much higher cold-start *surfaced* rate (items appearing in top-10 at all:
1.000 vs ≤0.02 for the baselines — see `EvalResult.cold_start_surfaced_rate`
in `src/reclab/eval/harness.py` for why this softer metric is reported
alongside the stricter exact-match one). The planner's rationale for
`hybrid_llm` ("high cold-start ratio plus usable item text is exactly the gap
an LLM re-ranker fills") is about cold-start performance specifically, not
overall Recall@K — and on the metric it's actually reasoning about, it's
right.

**The actual takeaway:** the reasoning engine's *shortlist ranking* currently
conflates "best architecture overall" with "best architecture on the
dimension it's reasoning about," and a single Recall@K number can't validate
that distinction. The concrete next step (not yet done) is scoring
architectures against the specific metric each one's rationale invokes —
`sasrec`'s rationale is about sequence signal, so it should be checked
against something like sequence-aware NDCG, not aggregate Recall@K — rather
than implicitly ranking everything by one metric. That's a real Phase 0
finding, not a passing or failing grade, and it's a better next action item
than a false "it all matches" would have been.

## Reproducing this

```bash
uv sync
bash scripts/demo.sh
```

Or individually:

```bash
uv run python scripts/run_benchmark.py --dataset synthetic
uv run python scripts/run_benchmark.py --dataset synthetic --cold-start-heavy
```

Both run in well under a minute on a laptop CPU — no GPU, no API keys, no
network access required (see CONTRIBUTING.md and pyproject.toml for why
there's deliberately no PyTorch/deep-learning dependency here).

## The bar

Per the MVP plan (`docs/architecture/mvp-plan.md`, section 5): the reasoning
engine's shortlist and rationale need to hold up against benchmark results,
reproducibly, by anyone who clones the repo and runs this script. Where it
doesn't hold up — see above — that's logged here, not smoothed over.
