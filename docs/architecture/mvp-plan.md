# Recommendation Systems Platform — MVP Plan (open-source, self-hosted)

## 1. Positioning recap

**What it is:** An open-source platform that helps someone reason about, configure, and benchmark recommendation architectures — transformer-based sequential models, GNN approaches, and LLM-hybrid re-ranking — against their own data. Self-hosted, deployable on any cloud or locally via Docker. Public GitHub repo, MIT/Apache-licensed, built in the open.

**Why this framing changes the plan a lot:** the earlier version assumed a vendor/customer split — your infrastructure vs. a customer's cloud account, with a trust boundary (BYOC, scoped IAM roles, spend guardrails) between them. Open source and self-hosted collapses that split. The person running it *is* the person whose cloud it runs on. That removes a large chunk of engineering complexity that isn't needed anymore:
- No cross-account IAM role assumption flow
- No spend-cap/guardrail system protecting someone else's budget
- No SSO/RBAC/audit logging (needed for a paid enterprise product, not for a self-hosted tool one person or a small team runs)
- No sales motion, no design-partner POC process

What stays valuable regardless of business model:
- The reasoning engine (the core differentiator) — still the thing worth validating first
- The dual-audience UI concept (plain-language + technical detail) — still a nice differentiator, and a good portfolio signal of product thinking, not just ML skill
- The architecture comparison sandbox — still the core feature

**Goals, in priority order:** (1) learning — hands-on depth across rec-sys architectures, MLOps, and product/infra design, (2) portfolio — a real, documented, working system that's a strong artifact for the frontier-lab job search, (3) optionality — if it gets real traction, there's a path to expand (open-core, sponsorship, or eventually a company), but that's not the goal driving early decisions.

---

## 2. Who this is for

Primary users at this stage: ML engineers and small teams experimenting with recommendation architectures who want to compare approaches on their own data without wiring together Transformers4Rec, RecBole, and a custom LLM re-ranker from scratch. Realistically, in the first months, "users" mostly means: you, on your own test data and public benchmarks, plus whoever else discovers the repo and tries it.

Don't design for enterprise or startup buyer personas yet — that was a business-model framing for a company that isn't being built right now. Design for "a competent ML engineer evaluating this repo in an afternoon." If it's not useful and legible to that person, no downstream audience matters.

---

## 3. License

Recommend **Apache 2.0** over MIT or AGPL:
- Adds a patent grant MIT lacks, which matters if this ever touches anything patentable in the model/architecture space and is a checkbox enterprise legal teams look for if adoption ever grows that direction.
- Unlike AGPL, it doesn't create friction for adoption — AGPL is the right call when license enforcement is your monetization mechanism (stopping hyperscalers from repackaging your hosted service), but that's a company-stage concern, not a portfolio-project one. AGPL would mostly just scare off casual adopters and contributors right now.
- It's what most current trending AI infra repos (Gemini CLI, Codex CLI) ship with, so it reads as a normal, expected choice rather than a decision someone has to think about before trying the repo.

Set this on day one — retrofitting a license change after other people have contributed code is a real headache (needs contributor sign-off), not something to defer.

---

## 4. MVP technical scope

**In scope for v0.1:**
- **Cloud-agnostic by default, not cloud-specific.** Since there's no BYOC trust boundary to broker anymore, the right target is "runs anywhere Docker runs" — a `docker-compose up` for local/single-VM use, with docs for deploying the same containers on any cloud's basic compute (a GCE/EC2/DigitalOcean VM, or a simple Kubernetes deployment later). This is a big scope reduction from the earlier IAM-role-assumption flow.
- **Data connectors:** start with local file / S3-compatible object storage (works against AWS S3, but also MinIO, GCS via S3-compatible APIs, and R2) rather than building cloud-specific connectors one at a time.
- **Three architecture options**, same as before: two-tower baseline, SASRec-style sequential transformer, hybrid SASRec encoder + LoRA-fine-tuned LLM re-ranker.
- **The reasoning/planner layer:** data profile in, ranked architecture shortlist + rationale out. Still the core IP and the first thing to validate — this doesn't change with the OSS reframe.
- **Offline evaluation harness:** Recall@K, NDCG@K, coverage, cold-start slice performance, temporal splits.
- **A real demo, not just a working repo.** For an OSS project, a live-feeling demo (a hosted playground, a notebook, or a short screen recording in the README) matters more than it would for an enterprise POC — most people deciding whether to star or try a repo never run the code first.

**Explicitly out of scope for MVP:**
- Multi-tenant hosting, billing, or a managed SaaS version of this (that's a "if it grows" decision, not a v0.1 decision)
- Real-time streaming ingestion
- GNN architectures
- Enterprise auth (SSO/RBAC) — irrelevant for a self-hosted single-user tool

---

## 5. What "working" looks like

Replace the design-partner POC bar from the earlier version with something suited to an open-source learning project:

1. The reasoning engine's shortlist and rationale hold up against published benchmark results (MovieLens, Amazon Reviews) — reproducible by anyone who clones the repo and runs the included benchmark script.
2. A newcomer can go from `git clone` to a working comparison run in well under an hour, ideally under 15 minutes — this is the single biggest lever on whether the repo gets real engagement versus being starred once and ignored.
3. The README documents *why* the architecture comparison approach is useful, not just *what* the code does — this is what makes it read as a genuine engineering artifact rather than a tutorial clone, and it's what makes it useful for the job-search side of the goal too.

---

## 6. Suggested phased build

**Phase 0 — Validate the reasoning engine offline**
Same as before: build the data-profiling + architecture-recommendation logic, test against known-good results on public benchmarks. Doesn't need Docker, doesn't need a UI. This is still the first real risk to retire.

**Phase 1 — Minimum working repo**
Wire up the 3 architectures behind a common interface, the eval harness with temporal splits, and a `docker-compose up` path to run it locally against a sample dataset. This is the version you'd actually push to GitHub as v0.1.

**Phase 2 — README, demo, and a real launch**
Write the README as if it's the most important file in the repo, because for adoption purposes it is: what this does, why the architecture-comparison angle is useful, a GIF or short clip of it running, clear setup instructions, and what's coming next. Post it somewhere real — Show HN, r/MachineLearning, LinkedIn — rather than letting it sit unannounced. Given your existing interest in content creation, a short write-up (Substack-style, matching what you've done before) walking through *why* you built it and what you learned is worth more for the portfolio goal than the code alone.

**Phase 3 — Iterate on real feedback**
Whatever issues or PRs show up (or don't) tell you more about what's actually useful than any amount of solo planning would. Treat the first few weeks post-launch as the real validation step for the reasoning engine and the UX, same role the design-partner phase played in the earlier version.

**Phase 4 — Decide on expansion, if warranted**
Only after there's real signal (stars, forks, actual usage, contributors showing up) — decide whether to add the dual-audience dashboard UI, GNN support, a hosted playground, or an open-core commercial layer. Let usage tell you what's missing rather than pre-building it.

---

## 7. Tech stack

Simplified relative to the enterprise version — no separate control-plane service is needed yet since there's no multi-tenant orchestration problem to solve.

| Layer | Choice | Why |
|---|---|---|
| Reasoning engine, eval harness, ML pipelines | Python (FastAPI if/when an API surface is needed) | Same reasoning as before — this is where the ecosystem (HF, PyTorch, RecBole, Transformers4Rec) lives, and it's the part that's actually your IP. |
| Packaging / deployment | Docker + docker-compose | Makes "runs on any cloud" true without building cloud-specific infrastructure. A Helm chart is a reasonable v0.2 addition once someone actually asks for Kubernetes. |
| Frontend (dashboard, sandbox) | React + TypeScript | Same as before — you've got reps here from the Provenance mockups, and it's the fastest path to the layered-card UI. Can ship after the CLI/API works; don't block v0.1 on it. |
| Storage | SQLite for local/single-user runs, Postgres as a documented option for anyone running it on a shared server | Don't require Postgres setup just to try the repo locally — that's exactly the kind of friction that kills a 15-minute first-run experience. |
| Go / control-plane service | **Not needed yet** | This was solving a multi-tenant, cross-account orchestration problem that doesn't exist in a self-hosted single-user tool. Revisit only if this becomes a hosted/managed offering later. |

---

## 8. Repo structure and OSS practices worth doing from day one

- `README.md` — the most important file, per section 5/6 above.
- `LICENSE` — Apache 2.0, set immediately.
- `CONTRIBUTING.md` — even a short one signals the repo is meant to be used, not just published.
- GitHub Actions CI running the benchmark suite on every PR — doubles as proof the reasoning engine's claims are checked automatically, not just asserted.
- Issue templates (bug report, feature request) — low effort, makes the repo look maintained.
- A `benchmarks/` directory with the public-dataset comparison results checked in and reproducible — this is both the credibility mechanism from section 5 and a strong, concrete thing to point to in interviews or referral conversations.

---

## 9. UI/UX

The dual-audience design from before (plain-language summary layer, expandable into full technical detail) is still worth building — it's a good differentiator even for an OSS tool, since most rec-sys comparison tooling is written by and for ML engineers only. But sequence it after Phase 1–2: a working CLI/API with a solid benchmark story is what earns early attention; the dashboard UI is what turns early attention into people actually using it past the first run. Don't let UI work delay the Phase 0/1 validation.

---

## 10. Growth path if it takes off

Kept deliberately light, since this isn't the goal driving current decisions:
- **Sponsorship / GitHub Sponsors** — plausible if it gets real usage, low effort to set up, doesn't change any technical decisions.
- **Open-core** — if a hosted/managed version ever makes sense, the earlier enterprise-plan sections (BYOC, control plane, dual-audience dashboard, executive summary export) are still there to pull from — nothing from that work is wasted, just deferred.
- **Job search value** — independent of whether it grows, a well-documented working system with real benchmark results is a concrete artifact for the frontier-lab conversations already in progress. Worth treating the README and the benchmarks directory with the same care you'd put into the Oasis AI-Engine write-up.

---

## 11. Immediate next step

Same as before, and still the right first move regardless of the business-model reframe: **pressure-test the reasoning engine against 2–3 public benchmark datasets.** Cheap, fast, and tells you within days whether the core idea holds up — before spending time on Docker packaging, README polish, or UI.
