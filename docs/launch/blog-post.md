# Building reclab: reasoning about recommendation architectures, and checking my own reasoning against real data

*(Draft — rewrite in your own voice before publishing.)*

## The problem

Recommendation systems are the core product for a lot of companies — social
platforms, retail, streaming — but choosing an architecture for one is still
mostly tribal knowledge. You either reach for a managed black box (Amazon
Personalize, Vertex AI) that makes the decision for you and hides it, or a
research library (RecBole, Transformers4Rec, NVIDIA Merlin) that hands you
dozens of models and no guidance on which one actually fits your data.

I wanted something in between: a tool that profiles your data, reasons about
which architecture is likely to work, and — this part matters — lets you
check that reasoning against real results instead of just trusting it.

## Where the idea started, and where it ended up

The first version of this I sketched out was much bigger: an enterprise
platform with bring-your-own-cloud deployment, a control plane, spend
guardrails, the works — aimed at companies where recommendations are the
core product. That's a real gap in the market, but it's also a company, not
a weekend project, and it assumes a sales motion and a buyer I wasn't
building for.

Scoping it down to "open-source, self-hosted, one person running it on their
own data" cut away most of the complexity that mattered for the enterprise
version — the trust boundary between a vendor and a customer's cloud account
just doesn't exist when you're both. What was left was the actual
interesting part: the reasoning engine, and whether its recommendations
would hold up. **[Try the live demo](https://sumanthp.github.io/reclab/)** —
it's the actual UI running against real precomputed results, not a mockup.

## Phase 0: does the reasoning actually work?

Before building any UI, before packaging anything nicely, the plan was to
validate the one thing everything else depends on: given a data profile
(sparsity, cold-start ratio, sequence length, item text availability), does
a heuristic planner's architecture shortlist actually track which
architecture wins?

This meant I needed working architectures to test against, not just a
planner that sounds reasonable.

## The PyTorch wall

My first plan was to implement the three candidate architectures — a
two-tower baseline, a SASRec-style sequential transformer, and a hybrid
encoder + re-ranker — using PyTorch, like everyone does. That didn't work:
the current PyTorch wheel on PyPI requires CUDA runtime libraries just to
`import torch`, even for CPU-only use, and the CPU-only wheel index wasn't
reachable from my environment.

Rather than blocking on that, I rewrote all three in plain NumPy. The
interesting one is the sequential model: implementing causal self-attention
by hand means implementing its backward pass by hand too, which is exactly
the kind of code that looks plausible and is subtly wrong. I didn't trust it
until I checked it against numerical gradients — comparing the analytic
gradient to a finite-difference approximation for a sample of parameters
across every gradient tensor. It passed on the first real attempt, which was
a genuinely good feeling, but I wouldn't have trusted the model's output
without that check regardless.

## What real data actually found

Synthetic data got the pipeline working, but the real test was whether any
of this held up on real interactions. I ran it against MovieLens 100K and
two categories of Amazon Reviews 2023 (`All_Beauty`, `Gift_Cards`).

The honest result: **still mixed, in an interesting way.** On MovieLens, the
planner's #1 pick (`sasrec`) matched the measured Recall@K/NDCG@K winner —
a clean hit. On both Amazon Reviews categories, it didn't. But here's the
part I actually care about: the planner itself flagged both of those picks
as low-confidence *before* I checked whether they were right — a real
`margin_to_next` computed from the score gap between its #1 and #2 pick, not
a post-hoc excuse. Two for two, the low-confidence flag was on the case that
turned out wrong.

That's not "the reasoning engine works." It's something more useful for a
tool whose whole premise is that you can trust its explanations: **it knows
when it doesn't know.** The confidence signal is now a real field in the API
response and the UI (a "close call" badge, not just a score bar), and the
underlying ranking-calibration problem — the shortlist still conflates "best
overall" with "best on the specific dimension its own rationale invokes" —
is documented as the concrete next step, not smoothed over.

Along the way, running real data also surfaced a real bug: `coverage_at_k`
was computing candidate-recommendation coverage against the wrong
denominator, letting one architecture score *above* the theoretical maximum
of 1.0 (1.53, on Amazon Reviews). Fixed, with a regression test that locks
in the case that caught it. I wrote all of this up in the repo
(`benchmarks/README.md`) rather than tuning the demo until it looked clean —
if the point of this project is building something legible and honest, that
has to include what's wrong, not just what's right.

## From a validated engine to an actual product

Once the reasoning engine had real evidence behind it, the rest became worth
building for real instead of mocking up: a FastAPI backend with an async job
queue for training runs, a React frontend that drives the whole upload →
profile → shortlist → compare → results loop against the live API (with run
history, cancellation, and per-user hit/miss examples you can inspect), and
a static demo build on GitHub Pages that reuses the exact same components
against precomputed real results, so anyone can see it work without running
anything locally. It's Docker Compose'd, has upload/concurrency limits and
structured JSON request logging, and a scoped Playwright suite that drives
the real backend and real frontend together as a final check that the
pieces actually integrate — on top of the unit-level pytest and Vitest
suites.

## What's next

The biggest open item is still calibrating the reasoning engine's ranking
against the multi-metric finding above — two real datasets now point at the
same gap. After that: running against a denser Amazon Reviews category
(the two I've run so far are both small), and benchmarking the real
Anthropic-backed re-ranker I added against the default lexical one to see
whether real semantic matching earns its cost.

Repo: <https://github.com/sumanthp/reclab>. Live demo:
<https://sumanthp.github.io/reclab/>. If you try it against your own data
and it breaks (or works), I'd like to know either way.
