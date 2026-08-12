# Building reclab: reasoning about recommendation architectures, and finding out my reasoning engine was half right

*(Draft — rewrite in your own voice before publishing. Placeholders in `<...>`.)*

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
would hold up.

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
encoder + LLM re-ranker — using PyTorch, like everyone does. That didn't
work: the current PyTorch wheel on PyPI requires CUDA runtime libraries just
to `import torch`, even for CPU-only use, and the CPU-only wheel index
wasn't reachable from my environment.

Rather than blocking on that, I rewrote all three in plain NumPy. The
interesting one is the sequential model: implementing causal self-attention
by hand means implementing its backward pass by hand too, which is exactly
the kind of code that looks plausible and is subtly wrong. I didn't trust it
until I checked it against numerical gradients — comparing the analytic
gradient to a finite-difference approximation for a sample of parameters
across every gradient tensor. It passed on the first real attempt, which was
a genuinely good feeling, but I wouldn't have trusted the model's output
without that check regardless.

## What the benchmarks actually found

Once all three architectures could really train and recommend, I ran the
reasoning engine's shortlist against actual measured results on two
scenarios: a typical dataset, and a sparse, high-cold-start, rich-item-text
one where the planner should confidently favor the hybrid architecture.

The honest result: **partial confirmation.** On raw Recall@10, the planner's
top pick didn't win in either scenario — a well-tuned matrix factorization
baseline was hard to beat at the scale I tested, which is a known,
legitimate pattern in recommendation systems, not a sign my other
implementations were broken. But on cold-start recall specifically, the
hybrid architecture won exactly when the planner said it should, by a wide
margin, in both scenarios.

That told me something more useful than either a clean pass or a clean
failure would have: the reasoning engine's ranking currently conflates "best
architecture overall" with "best architecture on the dimension it's actually
reasoning about." A rationale about cold-start performance should be checked
against cold-start metrics, not an aggregate Recall@K number. That's a real,
specific next step for the planner's scoring logic — and a much better thing
to find in Phase 0 than either "it's perfect" or "it's wrong," neither of
which would have told me what to fix.

I wrote this up in full in the repo (`benchmarks/README.md`) rather than
tuning the demo until it looked clean. If the point of this project is
building something legible and honest, that has to include what doesn't
work yet.

## What's next

The biggest open item is validating this against real public benchmarks
(MovieLens, Amazon Reviews) instead of the synthetic dataset generator I
built for development — my environment couldn't reach the dataset hosts, so
that's genuinely untested end-to-end. After that: calibrating the reasoning
engine against the multi-metric finding above, and only then — per the
original plan — building the dashboard UI, once there's something real
worth putting a UI in front of.

Repo: <github URL>. If you try it against real data and it breaks (or
works), I'd like to know either way.
