import type { ArchitectureInfo, Recommendation } from "../lib/types";

// Layer 1 is `rationale` (prose, always shown) plus the architecture's
// one-line description. Layer 2 is the raw score and the architecture's
// static strengths/weaknesses/cost profile from /architectures — not a
// parsed breakdown of the rationale sentence, deliberately: the backend's
// Recommendation.rationale is a semicolon-joined prose string meant for a
// CLI print, and parsing it client-side to reconstruct structured
// "boosted/penalized" chips would be exactly the fragile approach flagged
// as an open question in docs/architecture/ui-ux-plan.md section 5. If
// that field becomes structured on the backend, this is where the richer
// per-factor detail would go.
interface Props {
  shortlist: Recommendation[];
  architectureInfo: Record<string, ArchitectureInfo>;
}

export function ShortlistCards({ shortlist, architectureInfo }: Props) {
  return (
    <div className="shortlist">
      {shortlist.map((rec) => {
        const info = architectureInfo[rec.architecture];
        return (
          <div key={rec.architecture} className={`rec-card ${rec.rank === 1 ? "is-first" : ""}`}>
            <div className="rec-top">
              <span className="rank-badge">#{rec.rank}</span>
              <span className="arch-name">{rec.architecture}</span>
              <span className="score-track">
                <span className="score-fill" style={{ width: `${rec.score * 100}%` }} />
              </span>
              <span className="score-num">{rec.score.toFixed(2)}</span>
            </div>
            <p className="rec-rationale">{capitalize(rec.rationale)}</p>
            <details className="layer2">
              <summary>
                <span className="chev">▸</span> Details
              </summary>
              <div className="layer2-body">
                <div className="layer2-score">score = {rec.score.toFixed(3)}</div>
                {info && (
                  <>
                    <p className="layer2-desc">{info.description}</p>
                    <div className="cost-badges">
                      <span className={`cost-badge cost-${info.relative_train_cost}`}>
                        train cost: {info.relative_train_cost}
                      </span>
                      <span className={`cost-badge cost-${info.relative_serving_latency}`}>
                        serving latency: {info.relative_serving_latency}
                      </span>
                    </div>
                    {info.strengths.length > 0 && (
                      <div className="pro-con">
                        <span className="pro-con-label strengths">Strengths</span>
                        <ul>
                          {info.strengths.map((s) => (
                            <li key={s}>{s}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    {info.weaknesses.length > 0 && (
                      <div className="pro-con">
                        <span className="pro-con-label weaknesses">Weaknesses</span>
                        <ul>
                          {info.weaknesses.map((w) => (
                            <li key={w}>{w}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </>
                )}
              </div>
            </details>
          </div>
        );
      })}
    </div>
  );
}

function capitalize(s: string): string {
  return s.length ? s[0].toUpperCase() + s.slice(1) : s;
}
