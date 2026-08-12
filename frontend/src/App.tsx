import { useEffect, useRef, useState } from "react";
import "./App.css";
import { UploadForm } from "./components/UploadForm";
import { ProfileStats } from "./components/ProfileStats";
import { ShortlistCards } from "./components/ShortlistCards";
import { CompareResults } from "./components/CompareResults";
import { getRun, profileDataset, startCompare } from "./lib/api";
import { ApiError, type ProfileResponse, type RunResponse } from "./lib/types";

const POLL_INTERVAL_MS = 1500;

interface Dataset {
  interactionsCsv: File;
  itemMetadataCsv: File | null;
  userCol: string;
  itemCol: string;
}

function App() {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [profileResp, setProfileResp] = useState<ProfileResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);

  const [run, setRun] = useState<RunResponse | null>(null);
  const [compareStarting, setCompareStarting] = useState(false);
  const [compareStartError, setCompareStartError] = useState<string | null>(null);
  const pollHandle = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (pollHandle.current !== null) window.clearInterval(pollHandle.current);
    };
  }, []);

  async function handleAnalyze(
    interactionsCsv: File,
    itemMetadataCsv: File | null,
    userCol: string,
    itemCol: string,
  ) {
    setProfileLoading(true);
    setProfileError(null);
    try {
      const resp = await profileDataset(interactionsCsv, userCol, itemCol);
      setDataset({ interactionsCsv, itemMetadataCsv, userCol, itemCol });
      setProfileResp(resp);
      setRun(null);
    } catch (e) {
      setProfileError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setProfileLoading(false);
    }
  }

  async function handleRunCompare() {
    if (!dataset) return;
    setCompareStarting(true);
    setCompareStartError(null);
    try {
      const { job_id } = await startCompare(
        dataset.interactionsCsv,
        dataset.itemMetadataCsv,
        dataset.userCol,
        dataset.itemCol,
      );
      const initial = await getRun(job_id);
      setRun(initial);
      pollHandle.current = window.setInterval(async () => {
        const latest = await getRun(job_id);
        setRun(latest);
        if (latest.status === "done" || latest.status === "error") {
          if (pollHandle.current !== null) window.clearInterval(pollHandle.current);
        }
      }, POLL_INTERVAL_MS);
    } catch (e) {
      setCompareStartError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setCompareStarting(false);
    }
  }

  function handleReset() {
    if (pollHandle.current !== null) window.clearInterval(pollHandle.current);
    setDataset(null);
    setProfileResp(null);
    setProfileError(null);
    setRun(null);
    setCompareStartError(null);
  }

  return (
    <>
      <div className="header">
        <span className="wordmark">
          reclab<span className="dot">·</span>
        </span>
        <span className="header-sub">recommendation architecture lab</span>
        {profileResp && (
          <div className="header-actions">
            <button className="btn btn-ghost" onClick={handleReset}>
              Analyze new dataset
            </button>
          </div>
        )}
      </div>

      {!profileResp && (
        <UploadForm onAnalyze={handleAnalyze} loading={profileLoading} error={profileError} />
      )}

      {profileResp && (
        <>
          <section className="section" id="profile">
            <div className="section-head">
              <h2>Data profile</h2>
            </div>
            <p className="section-sub">
              Sparsity, cold-start ratio, and sequence length are what the shortlist below reasons
              over.
            </p>
            <ProfileStats profile={profileResp.profile} />
          </section>

          <section className="section" id="shortlist">
            <div className="section-head">
              <h2>Reasoning engine shortlist</h2>
            </div>
            <p className="section-sub">
              Ranked candidates with a plain-language rationale for each.
            </p>
            <ShortlistCards shortlist={profileResp.recommendations} />
          </section>

          <section className="section" id="compare">
            <div className="section-head">
              <h2>Compare</h2>
            </div>
            <p className="section-sub">
              Trains and evaluates every candidate architecture on a temporal split, then checks
              the shortlist's #1 pick against what actually won.
            </p>

            {!run && (
              <div className="compare-cta">
                <p>
                  This trains all three architectures for real — it can take from a few seconds to
                  a few minutes depending on dataset size.
                </p>
                <button
                  className="btn btn-primary"
                  onClick={handleRunCompare}
                  disabled={compareStarting}
                >
                  {compareStarting ? "Starting…" : "Run full comparison"}
                </button>
              </div>
            )}
            {compareStartError && (
              <div className="error-banner">
                <strong>Error:</strong> {compareStartError}
              </div>
            )}

            {run && (run.status === "pending" || run.status === "running") && (
              <div className="status-line">
                <span className="spinner" /> Training and evaluating architectures… ({run.status})
              </div>
            )}

            {run && run.status === "error" && (
              <div className="error-banner">
                <strong>Run failed:</strong> {run.error}
              </div>
            )}

            {run && run.status === "done" && run.result && <CompareResults result={run.result} />}
          </section>
        </>
      )}

      <footer className="app-footer">
        reclab — self-hosted, no data leaves your machine. See{" "}
        <a href="https://github.com/sumanthp/reclab" target="_blank" rel="noreferrer">
          the repo
        </a>{" "}
        for benchmarks, docs, and source.
      </footer>
    </>
  );
}

export default App;
