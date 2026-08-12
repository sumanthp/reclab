import type { ArchitectureInfo, Recommendation } from "../lib/types";

// Layer 1 is `rationale` (prose, always shown) plus a "close call" flag on
// rank 1 when its score margin over #2 is thin (see
// reasoning_engine/planner.py's LOW_CONFIDENCE_MARGIN — evidence-grounded
// from a real case where a 0.15-point margin turned out to be the wrong
// pick, see benchmarks/README.md). Layer 2 shows the real structured
// `factors` the backend scored this architecture on, plus its static
// strengths/weaknesses/cost profile from /architectures — no client-side
// parsing of the prose `rationale` string, which used to be the only
// option here (see docs/architecture/ui-ux-plan.md section 5).
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
              {rec.rank === 1 && rec.low_confidence && (
                <span
                  className="confidence-flag"
                  title={
                    rec.margin_to_next !== null
                      ? `Only ${rec.margin_to_next.toFixed(2)} points ahead of #2 — not a strong signal either way`
                      : undefined
                  }
                >
                  close call
                </span>
              )}
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
                <div className="factors-list">
                  {rec.factors.map((f) => (
                    <div className="factor-row" key={f.detail}>
                      <span
                        className={`factor-effect ${
                          f.effect > 0 ? "boost" : f.effect < 0 ? "penalty" : "neutral"
                        }`}
                      >
                        {f.effect === 0 ? "base" : `${f.effect > 0 ? "+" : ""}${f.effect.toFixed(2)}`}
                      </span>
                      <span className="factor-detail">{f.detail}</span>
                    </div>
                  ))}
                </div>
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
