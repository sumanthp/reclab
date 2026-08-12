import type { ComparisonSummary } from "../lib/types";

// Neutral-but-visible on a mismatch, per docs/architecture/ui-ux-plan.md
// section 5 — this is the one place the UI must not soften a mismatch into
// a false "success" framing, since the project's whole credibility argument
// (see benchmarks/README.md) rests on not hiding this.
export function VerdictBanner({ comparison }: { comparison: ComparisonSummary }) {
  if (comparison.matches_on_recall === null) {
    return null;
  }

  if (comparison.matches_on_recall) {
    return (
      <div className="verdict-banner match">
        <span className="icon">✓</span>
        <p>
          <strong>Shortlist's #1 pick matched the measured winner.</strong>{" "}
          <code>{comparison.shortlist_pick}</code> was ranked #1 by the reasoning engine and
          posted the best measured Recall@K.
        </p>
      </div>
    );
  }

  return (
    <div className="verdict-banner mismatch">
      <span className="icon">✕</span>
      <p>
        <strong>Shortlist's #1 pick did not match the measured winner.</strong> The planner picked{" "}
        <code>{comparison.shortlist_pick}</code>, but <code>{comparison.measured_best_recall}</code>{" "}
        had the best measured Recall@K.
        {comparison.note && <> — {comparison.note}.</>}
      </p>
    </div>
  );
}
