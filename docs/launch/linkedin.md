I open-sourced a project I've been building: **reclab**, a self-hosted tool
that helps you reason about which recommendation system architecture
actually fits your data — before you spend weeks building the wrong one.

Most teams either use a managed black-box service that hides the
architecture entirely, or a research library that hands you dozens of models
with no guidance on which one fits. reclab profiles your interaction data
(sparsity, cold-start ratio, sequence length, item metadata) and proposes a
ranked shortlist of architectures with a plain-language explanation of why —
then actually trains and evaluates each one on your data so you can check
that reasoning against real results, not just trust it.

A couple of things I'm proud of from building this:

- All three candidate architectures, including a self-attention-based
  sequential model, are implemented from scratch — with the trickiest part
  (backpropagation through the attention mechanism) validated against
  numerical gradient checks, not just "it trained without crashing."
- I ran the reasoning engine's own recommendations against measured results
  and published what I found, including a real gap: it correctly identifies
  which architecture wins on the specific thing it's reasoning about, but its
  overall ranking needs more calibration. Publishing the miss alongside the
  hit felt more honest than only showing the clean result.

It's Apache 2.0 and self-hosted — runs locally via Docker, your data never
leaves your machine. Link in comments / repo: <github URL once pushed>.

Would love feedback from anyone who's built recommendation systems at scale,
especially on where the reasoning engine's logic doesn't match your
real-world experience.
