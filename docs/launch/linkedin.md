I open-sourced a project I've been building: **reclab**, a self-hosted tool
that helps you reason about which recommendation system architecture
actually fits your data — before you spend weeks building the wrong one.

Most teams either use a managed black-box service that hides the
architecture entirely, or a research library that hands you dozens of models
with no guidance on which one fits. reclab profiles your interaction data
(sparsity, cold-start ratio, sequence length, item metadata) and proposes a
ranked shortlist of architectures with a plain-language explanation of why —
then actually trains and evaluates each one on your data so you can check
that reasoning against real results, not just trust it. There's a live demo
you can click through with real results, no signup required:
sumanthp.github.io/reclab

A couple of things I'm proud of from building this:

- All three candidate architectures, including a self-attention-based
  sequential model, are implemented from scratch — with the trickiest part
  (backpropagation through the attention mechanism) validated against
  numerical gradient checks, not just "it trained without crashing."
- I ran the reasoning engine's own recommendations against measured results
  on MovieLens and two Amazon Reviews categories, and published what I
  found — including that the engine's own "this is a close call" confidence
  flag was right both times: on the two picks it flagged as low-confidence,
  the guess was wrong. That's a real signal now surfaced in the product, not
  a caveat buried in a README.
- Running real data also caught a genuine bug in one of my own eval
  metrics (a coverage calculation that could exceed its theoretical max) —
  fixed and documented, not swept under the rug.
- It went from a validated reasoning engine to a full working product: a
  FastAPI backend, a React frontend with run history and cancellation, a
  real (optional) LLM re-ranker via the Anthropic API, structured JSON
  logging, and a Playwright end-to-end suite testing the real backend and
  frontend together, on top of ~150 unit tests across both.

It's Apache 2.0 and self-hosted — runs locally via Docker, your data never
leaves your machine. Repo: github.com/sumanthp/reclab. Live demo:
sumanthp.github.io/reclab.

Would love feedback from anyone who's built recommendation systems at scale,
especially on where the reasoning engine's logic doesn't match your
real-world experience.
