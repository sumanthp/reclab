import type { RunSummary } from "../lib/types";

interface Props {
  runs: RunSummary[];
  onSelect: (jobId: string) => void;
  activeRunId?: string;
}

const STATUS_LABEL: Record<string, string> = {
  pending: "Queued",
  running: "Running",
  done: "Done",
  error: "Failed",
  cancelled: "Cancelled",
};

export function RunHistory({ runs, onSelect, activeRunId }: Props) {
  if (runs.length === 0) return null;

  return (
    <div className="run-history">
      <div className="run-history-label">Recent runs</div>
      <ul>
        {runs.map((r) => (
          <li key={r.id}>
            <button
              type="button"
              className={`run-history-item ${r.id === activeRunId ? "active" : ""}`}
              onClick={() => onSelect(r.id)}
            >
              <span className={`run-status-dot status-${r.status}`} aria-hidden="true" />
              <span className="run-history-item-text">{r.dataset_label ?? "untitled"}</span>
              <span className="run-history-status">{STATUS_LABEL[r.status] ?? r.status}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
