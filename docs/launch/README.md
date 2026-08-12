# Launch drafts

Per the MVP plan (`docs/architecture/mvp-plan.md`, Phase 2): "post it somewhere
real — Show HN, r/MachineLearning, LinkedIn — rather than letting it sit
unannounced."

These are drafts, not published posts — publishing to GitHub, Hacker News,
Reddit, or LinkedIn needs your own accounts and your own final say on
wording, so nothing here has been posted. Review, edit to sound like you
(check `docs/architecture/mvp-plan.md` and your own judgment on tone), and
post whichever of these you want, whenever you're ready.

**Status: the prerequisites are done.** The repo is pushed
(<https://github.com/sumanthp/reclab>), the live demo is deployed
(<https://sumanthp.github.io/reclab/>), and `scripts/demo.sh` runs clean
from a fresh clone. All four drafts were refreshed to reflect what's
actually true today — real MovieLens 100K + two Amazon Reviews 2023
category results (not just synthetic), the `low_confidence` signal (and
that it's been right both times it fired on real data so far), the
`coverage_at_k` bug found and fixed along the way, the working end-to-end
UI, the optional Anthropic-backed re-ranker, and the Playwright E2E suite.
Reread them once more before posting in case anything's shifted since —
these are current as of the commit that added this note, not a living
document that updates itself.

Suggested order, per the plan: post Show HN first (it's the highest-signal,
highest-effort audience), then r/MachineLearning, then LinkedIn, and publish
the blog post either alongside Show HN or shortly after as the longer-form
companion piece people can link back to.

- `show-hn.md` — Hacker News
- `reddit-machinelearning.md` — r/MachineLearning
- `linkedin.md` — shorter, less technical
- `blog-post.md` — the "why I built this and what I learned" write-up the
  plan calls out as worth more for the portfolio goal than the code alone
