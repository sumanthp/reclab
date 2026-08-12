# UI/UX plan — the dashboard/sandbox

## 0. Where this fits

Per `mvp-plan.md` section 9: build this *after* Phase 0/1, not before. Phase
0/1 is done — the reasoning engine is validated against two real datasets
(see `benchmarks/README.md`) and the CLI/API prove the core loop works
without a UI. This doc is the Phase 2+ UI/UX plan referenced there.

`frontend/` currently has no files (just an empty `frontend/src/`
directory) — this is a from-scratch plan, not a redesign.

## 1. Who this is for, and what that implies

Per `mvp-plan.md` section 2: "a competent ML engineer evaluating this repo
in an afternoon." That person already trusts numbers over marketing copy —
the UI's job is to get out of the way of the data as fast as possible, not
to look like a SaaS product. Concretely:
- No onboarding flow, no empty-state illustrations, no marketing hero.
- The first screen after upload should show real numbers within seconds,
  not a guided tour.
- Every claim the UI makes (a rationale, a "recommended" badge) should be
  one click from the actual number backing it — this is what "dual-audience"
  means in practice, not two separate UIs.

## 2. The dual-audience pattern, concretely

`mvp-plan.md` calls this "plain-language summary layer, expandable into
full technical detail." Translated into actual components against the data
this repo already produces:

| Plain-language (Layer 1) | Technical detail (Layer 2, expand-in-place) |
|---|---|
| "sasrec — good fit: your users have long interaction histories" | `median_sequence_length: 65.0`, threshold `SEQUENCE_LENGTH_MIN_FOR_SASREC = 8`, score `0.75`, full rationale string from `Recommendation.rationale` |
| A single "Recall@10: 14%" stat tile | Full `EvalResult` table: NDCG@10, coverage, cold-start recall, cold-start surfaced rate, n_test_users |
| "✓ works well with your data" badge on an architecture card | `ArchitectureInfo.strengths` / `.weaknesses` list, relative train cost / serving latency |

This is **one data model rendered two ways**, not two separate views to
build and keep in sync — every Layer 1 string should be generatable from
the Layer 2 numbers already in `DataProfile` / `Recommendation` /
`EvalResult` / `ArchitectureInfo`. If a Layer 1 claim can't be traced to a
field that exists today, that's a sign the claim shouldn't ship yet rather
than a sign to invent a new backend field to justify it.

Mechanically: an expand/collapse disclosure on each card and stat tile, not
a separate "technical mode" toggle for the whole page — a person new to
rec-sys and a person who's shipped three of these will often want different
depth on *different* cards on the *same* screen (e.g. an ML engineer wants
technical detail on the shortlist rationale but doesn't need coverage@k
explained).

## 3. Screen-by-screen

### 3.1 Upload
- Drag-and-drop / file picker for an interactions CSV, plus optional item
  metadata CSV. Column mapping fields default to `user_id`/`item_id` (match
  `/profile`'s existing defaults) with inline override — don't force a
  rename before upload.
- No dataset size limits stated upfront; if the profiler chokes, surface
  that as an actual error from the API (`ValueError` from
  `profile_interactions`, already returned as HTTP 400), not a pre-emptive
  guess at limits nobody's validated.

### 3.2 Profile
Renders `DataProfile` directly: n_users, n_items, n_interactions, sparsity,
cold_start_ratio, median_sequence_length as stat tiles; has_item_text /
has_item_image as boolean badges. This is the existing `/profile` response,
unchanged — first screen ships against the API that exists today, no new
backend work.

### 3.3 Shortlist
Renders `recommendations` from the same `/profile` response: ranked cards,
one per architecture, Layer 1 = `rationale` string as prose, Layer 2 =
score + the specific thresholds it crossed (parseable from the rationale
text today by splitting on `"; "` — see open item in section 5 about
whether that's durable enough to build against).

### 3.4 Compare (the actual sandbox — the core feature per section 2 of
`mvp-plan.md`)
Trains and evaluates all registered architectures on the uploaded data,
shows `EvalResult` per architecture side by side, and — critically, given
what Phase 0 found — **shows the shortlist's #1 pick next to the measured
winner explicitly**, including when they disagree. `scripts/run_benchmark.py`
already prints exactly this comparison; the UI should not hide the mismatch
case behind a "success" framing. This is the screen that makes the honesty
in `benchmarks/README.md` a first-class UI feature instead of a thing only
CLI users see.

**This screen has no backend endpoint yet** — see section 4, this is the
real blocker, not frontend component work.

### 3.5 Result detail (per architecture)
Layer 2 for one architecture's `EvalResult`, plus (once available) a sample
of actual recommended items per test user, so "what does Recall@10=0.14
even mean" has a concrete answer a newcomer can eyeball.

## 4. Backend work this actually requires (do this before frontend polish)

The API today (`src/reclab/api/main.py`) only has `/health`,
`/architectures`, and `/profile`. There is no HTTP path to what
`scripts/run_benchmark.py` does — training + evaluating all architectures.
Building the Compare screen against nothing would mean either faking data
or blocking frontend work entirely, so this is sequenced first:

1. **A `/compare` (or `/runs`) endpoint that kicks off training+eval and
   returns a job id**, not a synchronous response. MovieLens 100K took
   long enough locally that it had to run in the background during Phase 0
   validation (see this session's history) — a single blocking HTTP request
   training three architectures is a bad default even for small datasets,
   and actively wrong for anything MovieLens-100K-sized or larger.
2. **A job status/result endpoint** (`GET /runs/{id}`) the frontend polls.
   SQLite (already the documented storage choice for local/single-user runs
   per `mvp-plan.md` section 7) is enough to persist job state — no queue
   infra needed at this scale.
3. Reuse `temporal_train_test_split` + `run_eval` + `REGISTRY` exactly as
   `run_benchmark.py` does — this is a thin API wrapper around existing,
   already-validated logic, not new eval logic.

This is real, scoped backend work — not just "wire up a frontend to
existing endpoints" — and should be scheduled explicitly rather than
discovered mid-frontend-build.

## 5. Open questions to resolve before / while building, not after

- **Rationale strings as UI data.** `Recommendation.rationale` is a
  semicolon-joined prose string (`planner.py`), fine for a CLI print but
  fragile to parse into structured Layer 2 detail in a UI. Either keep
  rationale as pure prose (Layer 2 = raw score + profile fields, not a
  parsed breakdown of the sentence) or extend `Recommendation` with a
  structured `reasons: list[{factor, effect, detail}]` field the planner
  populates directly. The latter is cleaner but touches the reasoning
  engine's return type — worth deciding once, not accreting parsing hacks
  in the frontend.
- **Mismatch framing.** When the shortlist's #1 pick doesn't match the
  measured winner (has happened on 2 of 3 real/synthetic-adjacent runs so
  far), does the UI state that neutrally (as `run_benchmark.py` does today)
  or flag it more prominently as a caveat? Given the project's whole
  credibility argument rests on not hiding this, default to neutral-but-
  visible, not buried in Layer 2.
- **Long-running jobs and dataset size limits.** No dataset size guidance
  exists yet (see 3.1) — once `/compare` exists, real runs against it should
  inform what "this might take a while" thresholds to show, rather than
  guessing them now.

## 6. Tech stack (confirms `mvp-plan.md` section 7, no changes)

React + TypeScript, calling the existing FastAPI service directly (no BFF
layer needed at this scale). No component library decision made yet —
pick one when building starts, informed by whichever makes the disclosure
(expand/collapse) pattern in section 2 least awkward, not upfront.

## 7. Phased build order

1. **Backend: `/compare` + job status endpoint** (section 4). Blocking
   dependency for the Compare screen; everything else can start in
   parallel.
2. **Upload → Profile → Shortlist** (3.1–3.3), against the API that exists
   today. Ships the dual-audience pattern (section 2) on real data with the
   least new backend work — the fastest path to something demoable.
3. **Compare + Result detail** (3.4–3.5), once the job endpoint exists.
   This is the actual differentiator per `mvp-plan.md` section 2 ("the
   architecture comparison sandbox — still the core feature") — steps 1–2
   are worth shipping first only because they're faster to reach, not
   because they matter more.
4. **Polish pass**: the mismatch-framing and rationale-structure decisions
   from section 5, once there's a working sandbox to make those decisions
   concrete against instead of in the abstract.

Don't build a design system or a component library speculatively ahead of
step 2 — three or four real screens against real API responses will make
the right abstractions obvious; guessing them now risks the same kind of
premature structure this project has otherwise avoided.
