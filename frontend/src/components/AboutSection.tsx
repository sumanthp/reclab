// Demo-only (see App.tsx's DEMO_MODE gate) — this is the public-facing
// explainer for a first-time visitor landing on the GitHub Pages demo, not
// something a self-hosted user needs cluttering their working screen once
// they already know what the tool does.
export function AboutSection() {
  return (
    <section className="about">
      <div className="about-intro">
        <h1>Reason about your recommender before you build it</h1>
        <p>
          Choosing a recommendation system architecture is still mostly tribal knowledge. Managed
          services (Amazon Personalize, Vertex AI) hide the decision behind a black box; research
          libraries (RecBole, Transformers4Rec) hand you dozens of models with no guidance on
          which one fits your data. reclab sits in between: it profiles your interaction data,
          reasons about which architecture is likely to work and why, and then actually trains and
          evaluates the candidates on your data so you can check that reasoning against real
          results — instead of just trusting it.
        </p>
      </div>

      <div className="about-steps">
        <div className="about-step">
          <span className="about-step-num">1</span>
          <div>
            <h3>Profile</h3>
            <p>
              Sparsity, cold-start ratio, item metadata richness, interaction volume, sequence
              length — the signals that actually distinguish one architecture from another.
            </p>
          </div>
        </div>
        <div className="about-step">
          <span className="about-step-num">2</span>
          <div>
            <h3>Reason</h3>
            <p>
              A planner scores each candidate architecture against that profile and ranks them,
              with a plain-language, factor-by-factor rationale for each — not just a score.
            </p>
          </div>
        </div>
        <div className="about-step">
          <span className="about-step-num">3</span>
          <div>
            <h3>Compare</h3>
            <p>
              Every candidate is genuinely trained and evaluated on a temporal train/test split,
              so the shortlist's top pick gets checked against Recall@K, NDCG@K, coverage, and
              cold-start metrics — not left as an unverified claim.
            </p>
          </div>
        </div>
      </div>

      <div className="about-callout">
        <span className="about-callout-tag">Checked against real data</span>
        <p>
          The planner also flags when its own top pick is a close call — a{" "}
          <code>low_confidence</code> signal computed from the score margin over the runner-up, not
          a hedge added after the fact. Run against MovieLens 100K and two Amazon Reviews 2023
          categories, that flag has fired twice on real data — and both times, the flagged pick
          turned out to be wrong. That's the part of this project I trust most: not that the
          shortlist is always right, but that it tells you when it doesn't know.{" "}
          <a
            href="https://github.com/sumanthp/reclab/blob/main/benchmarks/README.md"
            target="_blank"
            rel="noreferrer"
          >
            Full write-up →
          </a>
        </p>
      </div>

      <div className="about-grid">
        <div>
          <h3>How it's built</h3>
          <p>
            All three candidate architectures — matrix factorization, a SASRec-style sequential
            transformer, and a hybrid encoder + re-ranker — are implemented from scratch in plain
            NumPy, no PyTorch dependency. The sequential model's hand-derived attention backward
            pass is checked against numerical gradients, not just "it trained without crashing."
            Self-hosted, Apache 2.0: <code>docker compose up</code> and your data never leaves your
            machine.
          </p>
        </div>
        <div>
          <h3>Who it's for</h3>
          <p>
            A team without a dedicated rec-sys specialist deciding what to build first. An ML
            engineer sanity-checking their own instinct against measured numbers before committing
            weeks to an architecture. Anyone who'd rather see the reasoning — and its failure
            modes — than trust a black box.
          </p>
        </div>
      </div>

      <div className="about-cta">
        <a
          className="btn btn-primary"
          href="https://github.com/sumanthp/reclab"
          target="_blank"
          rel="noreferrer"
        >
          View the repo
        </a>
        <a
          className="btn btn-ghost"
          href="https://github.com/sumanthp/reclab#getting-started"
          target="_blank"
          rel="noreferrer"
        >
          Self-host it →
        </a>
      </div>
    </section>
  );
}
