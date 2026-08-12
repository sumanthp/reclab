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

**Both public benchmarks have now been run end-to-end.** MovieLens 100K
(`results/movielens-100k-*.json`) and a category slice of Amazon Reviews 2023
(`results/amazon-reviews-*.json`) — see findings below for both.
`load_movielens_100k` parsed the real files with no changes needed;
`load_amazon_reviews_category` needed one real fix (see below) once run
against the actual dataset.

`reclab.datasets.synthetic` remains useful alongside them: it generates a reproducible dataset
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

## What the real MovieLens 100K run found

`uv run python scripts/run_benchmark.py --movielens-100k path/to/ml-100k` —
943 users, 1682 items, 100K ratings, 93.7% sparse, 19.8% cold-start ratio,
median sequence length 65. No parsing issues; the loader worked against the
real files unmodified.

**The planner's #1 pick won on the metric that matters most.** The planner
ranked `sasrec` first ("median sequence length 65.0 gives the sequence model
enough signal to use"), and `sasrec` did in fact post the best Recall@10
(0.141 vs `two_tower`'s 0.120 and `hybrid_llm`'s 0.119) and the best NDCG@10
(0.069 vs 0.062 and 0.045). This is the first real-dataset case where the
shortlist's #1 pick matches the measured overall winner — the synthetic runs
below never had that agreement, so this is a genuinely different (positive)
result, not a rerun of the same finding.

**Cold-start recall was 0.000 for all three architectures on this dataset**,
which softens how much to read into the above: MovieLens's 19.8% cold-start
ratio is implicit in a way the synthetic cold-start-heavy scenario isn't (its
"cold" items are ones with few ratings in the temporal test window, not
items with deliberately withheld interaction history), and no architecture
recovers any of them at `k=10` here. `hybrid_llm` still surfaces a cold item
in the top-10 for 100% of users (`cold_start_surfaced_rate=1.0`, vs
0.001/0.0 for the baselines) — consistent with the synthetic finding — but
that doesn't convert into an exact-match recall win on this dataset, so it
isn't a confirmation of the cold-start rationale the way the synthetic runs
were, just a non-contradiction.

**Net effect on the open calibration gap:** the calibration gap described
below (shortlist conflates "best overall" with "best on the dimension the
rationale invokes") is not fully resolved — this run simply didn't stress
that distinction, because sequence length was the dominant signal and both
the planner and Recall@10 agreed on it. The concrete next step is unchanged:
score each architecture against the specific metric its rationale invokes,
rather than implicitly ranking by Recall@K alone.

## What the real Amazon Reviews 2023 run found

`uv run python scripts/run_benchmark.py --amazon-reviews All_Beauty.csv
--amazon-metadata meta_All_Beauty.jsonl` — the `All_Beauty` category's
`benchmark/5core/rating_only` slice (253 users, 356 items, 2535 ratings,
97.2% sparse, 0% cold-start ratio by the profiler's definition, median
sequence length 7) plus the real per-product metadata file for titles.

**A real parsing bug was caught and fixed before this result was usable.**
`load_amazon_reviews_category`'s first pass built `item_metadata` by
renaming the metadata file's `title` column to `description` and selecting
`["item_id", "description"]` — but the raw `meta_<Category>.jsonl` records
already have their own `description` field (a list of bullet points),
so the rename produced *two* columns both named `description`, and
`item_metadata["description"]` downstream silently returned a DataFrame
instead of a Series, crashing `hybrid_llm.fit()` with `'DataFrame' object
has no attribute 'tolist'`. Fixed by constructing `item_metadata` from named
Series directly instead of rename+select (`src/reclab/datasets/loaders.py`).
This is exactly the kind of thing the loaders' unit fixtures didn't catch,
because the fixtures were hand-written and never had a field name collision
— real-file validation is what caught it.

**On this dataset, the planner's #1 pick did not match the measured winner.**
The planner picked `two_tower` (score 0.50, "solid general-purpose
baseline" — the median sequence length of 7 wasn't enough to boost `sasrec`,
and cold-start ratio of 0 gave no signal for `hybrid_llm`). The measured
Recall@10 winner was `hybrid_llm` (0.091 vs `sasrec`'s 0.079 vs
`two_tower`'s 0.047), and NDCG@10 agreed (0.040 vs 0.033 vs 0.023). This is
a second real mismatch, alongside the synthetic-data ones, on a dataset
where the planner's own rationale gave no strong reason to prefer any one
architecture (`hybrid_llm`'s rationale here was literally "no strong signal
either way" — the planner effectively guessed with 356 items, and the guess
was wrong).

**A metric-definition edge case also surfaced:** `hybrid_llm`'s
`coverage_at_k` came out to 1.528 — above 1.0, which shouldn't be possible
for a fraction. Traced to `hybrid_llm`'s re-ranker drawing candidates from
the full item metadata catalog (112,590 products in `All_Beauty`), not just
the 356 items with interactions, so it can recommend items outside the
train/test catalog that `coverage_at_k`'s denominator (`catalog_size`,
computed from train+test only) doesn't count. This is arguably correct
behavior for `hybrid_llm` specifically — recommending truly cold items with
zero interaction history is the entire point of a content-based re-ranker —
but it means `coverage_at_k` as currently defined isn't a fair comparison
across architectures with different candidate universes. Left as-is and
flagged here rather than silently patched, since narrowing `hybrid_llm`'s
candidates to the interaction catalog would remove the exact behavior being
evaluated; this needs a real design decision, not a guess.

**Net effect:** two real datasets now checked. MovieLens matched the
planner's pick; Amazon Reviews' `All_Beauty` slice didn't, on a case the
planner itself flagged as low-confidence. That's consistent with a planner
whose confidence should probably be reflected in the shortlist somehow
(e.g. surfacing "no strong signal" cases as closer scores, which it already
does here — 0.50 vs 0.35 vs 0.35 — the caller just isn't told 0.15 is a
weak margin) rather than a fixed calibration bug to patch.

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

For MovieLens 100K (network access needed once, to download it):

```bash
curl -o /tmp/ml-100k.zip https://files.grouplens.org/datasets/movielens/ml-100k.zip
uv run python scripts/run_benchmark.py --movielens-100k /tmp/ml-100k.zip
```

For Amazon Reviews 2023 (`All_Beauty` is a good first category — small):

```bash
curl -L -o /tmp/All_Beauty.csv "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/benchmark/5core/rating_only/All_Beauty.csv"
curl -L -o /tmp/meta_All_Beauty.jsonl "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl"
uv run python scripts/run_benchmark.py --amazon-reviews /tmp/All_Beauty.csv --amazon-metadata /tmp/meta_All_Beauty.jsonl
```

(The metadata file is ~200MB — the ratings-only CSV alone is enough to get a
shortlist and Recall@K numbers, just without `hybrid_llm`'s real item-text
signal.)

## The bar

Per the MVP plan (`docs/architecture/mvp-plan.md`, section 5): the reasoning
engine's shortlist and rationale need to hold up against benchmark results,
reproducibly, by anyone who clones the repo and runs this script. Where it
doesn't hold up — see above — that's logged here, not smoothed over.
