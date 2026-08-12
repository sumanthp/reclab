import type { Recommendation } from "../lib/types";

// Layer 1 is `rationale` (prose, always shown). Layer 2 is the raw score —
// not a parsed breakdown of the rationale sentence, deliberately: the
// backend's Recommendation.rationale is a semicolon-joined prose string
// meant for a CLI print, and parsing it client-side to reconstruct
// structured "boosted/penalized" chips would be exactly the fragile
// approach flagged as an open question in docs/architecture/ui-ux-plan.md
// section 5. If that field becomes structured on the backend, this
// component is where the richer Layer 2 detail would go.
export function ShortlistCards({ shortlist }: { shortlist: Recommendation[] }) {
  return (
    <div className="shortlist">
      {shortlist.map((rec) => (
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
              <span className="chev">▸</span> Raw score
            </summary>
            <div className="layer2-body">score = {rec.score.toFixed(3)}</div>
          </details>
        </div>
      ))}
    </div>
  );
}

function capitalize(s: string): string {
  return s.length ? s[0].toUpperCase() + s.slice(1) : s;
}
