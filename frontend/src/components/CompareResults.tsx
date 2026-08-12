import type { CompareResult, EvalResult } from "../lib/types";
import { isSkipped } from "../lib/types";
import { pct } from "../lib/format";
import { VerdictBanner } from "./VerdictBanner";

export function CompareResults({ result }: { result: CompareResult }) {
  const { eval_results, comparison } = result;
  const scored = Object.entries(eval_results).filter(
    (entry): entry is [string, EvalResult] => !isSkipped(entry[1]),
  );
  const anyAnomaly = scored.some(([, r]) => r.coverage_at_k > 1);

  return (
    <div>
      {comparison ? (
        <VerdictBanner comparison={comparison} />
      ) : (
        <div className="verdict-banner cancelled">
          <span className="icon">⏸</span>
          <p>
            <strong>Run was cancelled before enough architectures finished for a verdict.</strong>{" "}
            Numbers below are for whatever completed before the cancellation.
          </p>
        </div>
      )}

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Architecture</th>
              <th>Recall@{scored[0]?.[1].k ?? 10}</th>
              <th>NDCG@{scored[0]?.[1].k ?? 10}</th>
              <th>Coverage</th>
              <th>Cold-start recall</th>
              <th>Cold-start surfaced</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(eval_results).map(([name, r]) =>
              isSkipped(r) ? (
                <tr key={name} className="skipped-row">
                  <td>{name}</td>
                  <td colSpan={5}>skipped — {r.skipped}</td>
                </tr>
              ) : (
                <tr key={name}>
                  <td>
                    {name}
                    <span className="row-tags">
                      {comparison?.shortlist_pick === name && (
                        <span className="tag pick">#1 pick</span>
                      )}
                      {comparison?.measured_best_recall === name && (
                        <span className="tag best">measured best</span>
                      )}
                    </span>
                  </td>
                  <td className={comparison?.measured_best_recall === name ? "best-cell" : ""}>
                    {pct(r.recall_at_k)}
                  </td>
                  <td className={comparison?.measured_best_recall === name ? "best-cell" : ""}>
                    {pct(r.ndcg_at_k)}
                  </td>
                  <td className={r.coverage_at_k > 1 ? "anomaly" : ""}>
                    {pct(r.coverage_at_k)}
                    {r.coverage_at_k > 1 && <span className="anomaly-flag">⚠</span>}
                  </td>
                  <td
                    className={
                      comparison?.measured_best_cold_start_recall === name ? "best-cell" : ""
                    }
                  >
                    {r.cold_start_recall_at_k === null ? "n/a" : pct(r.cold_start_recall_at_k)}
                  </td>
                  <td>{pct(r.cold_start_surfaced_rate)}</td>
                </tr>
              ),
            )}
            {scored.length === 0 && Object.keys(eval_results).length === 0 && (
              <tr>
                <td colSpan={6}>Cancelled before any architecture finished training.</td>
              </tr>
            )}
          </tbody>
        </table>
        {anyAnomaly && (
          <div className="table-footnote">
            ⚠ Coverage above 100% means that architecture's candidate pool isn't limited to the
            train/test interaction catalog (e.g. hybrid_llm can recommend items with zero
            training interactions by design) — flagged, not hidden. See CONTRIBUTING.md.
          </div>
        )}
      </div>

      {scored.length > 0 && (
        <div className="detail-grid">
          {scored.map(([name, r]) => (
            <div className="detail-card" key={name}>
              <h3>{name}</h3>
              <Metric k="Test users" v={r.n_test_users.toLocaleString()} />
              <Metric k="Recall@K" v={r.recall_at_k.toFixed(3)} />
              <Metric k="NDCG@K" v={r.ndcg_at_k.toFixed(3)} />
              <Metric k="Coverage@K" v={r.coverage_at_k.toFixed(3)} />
              <Metric
                k="Cold-start recall@K"
                v={
                  r.cold_start_recall_at_k === null ? "n/a" : r.cold_start_recall_at_k.toFixed(3)
                }
              />
              <Metric k="Cold-start surfaced rate" v={r.cold_start_surfaced_rate.toFixed(3)} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Metric({ k, v }: { k: string; v: string }) {
  return (
    <div className="metric-row">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}
