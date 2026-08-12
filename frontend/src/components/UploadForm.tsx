import { useState, type FormEvent } from "react";

interface Props {
  onAnalyze: (
    interactionsCsv: File,
    itemMetadataCsv: File | null,
    userCol: string,
    itemCol: string,
  ) => void;
  loading: boolean;
  error: string | null;
}

export function UploadForm({ onAnalyze, loading, error }: Props) {
  const [interactionsCsv, setInteractionsCsv] = useState<File | null>(null);
  const [itemMetadataCsv, setItemMetadataCsv] = useState<File | null>(null);
  const [userCol, setUserCol] = useState("user_id");
  const [itemCol, setItemCol] = useState("item_id");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!interactionsCsv) return;
    onAnalyze(interactionsCsv, itemMetadataCsv, userCol, itemCol);
  }

  return (
    <div className="upload-card">
      <h1>Analyze your data</h1>
      <p className="intro">
        Upload an interactions CSV to profile it and get the reasoning engine's ranked
        architecture shortlist. A <code>timestamp</code> column is required if you also want to
        run the full training + evaluation comparison afterward.
      </p>
      <form onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="interactions">Interactions CSV (required)</label>
          <input
            id="interactions"
            type="file"
            accept=".csv"
            required
            onChange={(e) => setInteractionsCsv(e.target.files?.[0] ?? null)}
          />
          <div className="hint">Needs at least the user/item columns named below.</div>
        </div>

        <div className="field">
          <label htmlFor="metadata">Item metadata CSV (optional)</label>
          <input
            id="metadata"
            type="file"
            accept=".csv"
            onChange={(e) => setItemMetadataCsv(e.target.files?.[0] ?? null)}
          />
          <div className="hint">
            Needs <code>item_id</code> and <code>description</code> columns — enables the
            item-text signal (and lets <code>hybrid_llm</code> train).
          </div>
        </div>

        <div className="col-row">
          <div className="field">
            <label htmlFor="user-col">User column</label>
            <input
              id="user-col"
              type="text"
              value={userCol}
              onChange={(e) => setUserCol(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="item-col">Item column</label>
            <input
              id="item-col"
              type="text"
              value={itemCol}
              onChange={(e) => setItemCol(e.target.value)}
            />
          </div>
        </div>

        <div className="upload-actions">
          <button type="submit" className="btn btn-primary" disabled={!interactionsCsv || loading}>
            {loading ? "Profiling…" : "Profile dataset"}
          </button>
        </div>

        {error && (
          <div className="error-banner">
            <strong>Error:</strong> {error}
          </div>
        )}
      </form>
    </div>
  );
}
