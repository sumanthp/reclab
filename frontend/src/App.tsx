import { useEffect, useRef, useState } from "react";
import "./App.css";
import { UploadForm } from "./components/UploadForm";
import { ProfileStats } from "./components/ProfileStats";
import { ShortlistCards } from "./components/ShortlistCards";
import { CompareResults } from "./components/CompareResults";
import { RunHistory } from "./components/RunHistory";
import { DemoBanner } from "./components/DemoBanner";
import { AboutSection } from "./components/AboutSection";
import { DemoPicker } from "./components/DemoPicker";
import {
  cancelRun,
  fetchArchitectures,
  getRun,
  listRuns,
  profileDataset,
  startCompare,
} from "./lib/api";
import { DEMO_ARCHITECTURES, demoFixtureList, demoRun, demoRunSummaries } from "./demo/fixtures";
import {
  ApiError,
  type ArchitectureInfo,
  type ProfileResponse,
  type RunResponse,
  type RunSummary,
} from "./lib/types";

// Set at build time (see vite.config.ts / .github/workflows/pages.yml) — the
// GitHub Pages build has no backend to talk to, so it ships pre-computed
// real results instead of calling the network functions above at all.
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE === "true";

const POLL_INTERVAL_MS = 1500;
const TERMINAL_STATUSES = new Set(["done", "error"]);

interface Dataset {
  interactionsCsv: File;
  itemMetadataCsv: File | null;
  userCol: string;
  itemCol: string;
}

function App() {
  const [view, setView] = useState<"upload" | "results">("upload");
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [profileResp, setProfileResp] = useState<ProfileResponse | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);

  const [architectureInfo, setArchitectureInfo] = useState<Record<string, ArchitectureInfo>>({});

  const [run, setRun] = useState<RunResponse | null>(null);
  const [compareStarting, setCompareStarting] = useState(false);
  const [compareStartError, setCompareStartError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const [history, setHistory] = useState<RunSummary[]>([]);
  const pollHandle = useRef<number | null>(null);

  useEffect(() => {
    if (DEMO_MODE) {
      setArchitectureInfo(Object.fromEntries(DEMO_ARCHITECTURES.map((a) => [a.name, a])));
      setHistory(demoRunSummaries());
      return;
    }
    fetchArchitectures()
      .then((list) => {
        setArchitectureInfo(Object.fromEntries(list.map((a) => [a.name, a])));
      })
      .catch(() => {
        // Non-critical: the shortlist still works without Layer 2 detail.
      });
    refreshHistory();
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, []);

  function stopPolling() {
    if (pollHandle.current !== null) {
      window.clearInterval(pollHandle.current);
      pollHandle.current = null;
    }
  }

  async function refreshHistory() {
    try {
      setHistory(await listRuns());
    } catch {
      // Non-critical: history is a convenience, not required for the core flow.
    }
  }

  function isRunSettled(r: RunResponse): boolean {
    // 'cancelled' has two sub-states: just-requested (result still null,
    // the background job hasn't noticed yet) and finalized (result
    // populated, or permanently null if the job never started at all —
    // see _run_compare_job's early-return in main.py). Keep polling through
    // the former so partial results aren't missed.
    return TERMINAL_STATUSES.has(r.status) || (r.status === "cancelled" && r.result !== null);
  }

  async function attachToRun(jobId: string) {
    stopPolling();
    const initial = await getRun(jobId);
    setRun(initial);
    if (initial.result) {
      setProfileResp({
        profile: initial.result.profile,
        recommendations: initial.result.reasoning_engine_shortlist,
      });
    }
    if (!isRunSettled(initial)) {
      pollHandle.current = window.setInterval(async () => {
        const latest = await getRun(jobId);
        setRun(latest);
        if (latest.result) {
          setProfileResp({
            profile: latest.result.profile,
            recommendations: latest.result.reasoning_engine_shortlist,
          });
        }
        if (isRunSettled(latest)) {
          stopPolling();
          refreshHistory();
        }
      }, POLL_INTERVAL_MS);
    }
  }

  async function handleAnalyze(
    interactionsCsv: File,
    itemMetadataCsv: File | null,
    userCol: string,
    itemCol: string,
  ) {
    stopPolling();
    setProfileLoading(true);
    setProfileError(null);
    try {
      const resp = await profileDataset(interactionsCsv, itemMetadataCsv, userCol, itemCol);
      setDataset({ interactionsCsv, itemMetadataCsv, userCol, itemCol });
      setProfileResp(resp);
      setRun(null);
      setCompareStartError(null);
      setView("results");
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
      await attachToRun(job_id);
      refreshHistory();
    } catch (e) {
      setCompareStartError(e instanceof ApiError ? e.message : "Something went wrong.");
    } finally {
      setCompareStarting(false);
    }
  }

  function handleLoadDemo(id: string) {
    const found = demoRun(id);
    if (!found?.result) return;
    setDataset(null);
    setProfileError(null);
    setCompareStartError(null);
    setProfileResp({
      profile: found.result.profile,
      recommendations: found.result.reasoning_engine_shortlist,
    });
    setRun(found);
    setView("results");
  }

  async function handleViewRun(jobId: string) {
    if (DEMO_MODE) {
      handleLoadDemo(jobId);
      return;
    }
    setDataset(null); // no local files for a historical run — Compare CTA stays hidden
    setProfileError(null);
    setCompareStartError(null);
    setProfileResp(null);
    setView("results");
    await attachToRun(jobId);
  }

  async function handleCancel() {
    if (!run) return;
    setCancelling(true);
    try {
      const updated = await cancelRun(run.id);
      setRun(updated);
    } catch (e) {
      setCompareStartError(e instanceof ApiError ? e.message : "Could not cancel the run.");
    } finally {
      setCancelling(false);
    }
  }

  function handleReset() {
    stopPolling();
    setDataset(null);
    setProfileResp(null);
    setProfileError(null);
    setRun(null);
    setCompareStartError(null);
    setView("upload");
  }

  return (
    <>
      <div className="header">
        <span className="wordmark">
          reclab<span className="dot">·</span>
        </span>
        <span className="header-sub">recommendation architecture lab</span>
        {view === "results" && (
          <div className="header-actions">
            <button className="btn btn-ghost" onClick={handleReset}>
              {DEMO_MODE ? "Try another dataset" : "Analyze new dataset"}
            </button>
          </div>
        )}
      </div>

      {DEMO_MODE && <DemoBanner />}
      {DEMO_MODE && view === "upload" && <AboutSection />}

      <RunHistory runs={history} onSelect={handleViewRun} activeRunId={run?.id} />

      {view === "upload" &&
        (DEMO_MODE ? (
          <DemoPicker fixtures={demoFixtureList()} onSelect={handleLoadDemo} />
        ) : (
          <UploadForm onAnalyze={handleAnalyze} loading={profileLoading} error={profileError} />
        ))}

      {view === "results" && (
        <>
          {profileResp && (
            <>
              <section className="section" id="profile">
                <div className="section-head">
                  <h2>Data profile</h2>
                </div>
                <p className="section-sub">
                  Sparsity, cold-start ratio, and sequence length are what the shortlist below
                  reasons over.
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
                <ShortlistCards
                  shortlist={profileResp.recommendations}
                  architectureInfo={architectureInfo}
                />
              </section>
            </>
          )}

          <section className="section" id="compare">
            <div className="section-head">
              <h2>Compare</h2>
            </div>
            <p className="section-sub">
              Trains and evaluates every candidate architecture on a temporal split, then checks
              the shortlist's #1 pick against what actually won.
            </p>

            {!run && dataset && (
              <div className="compare-cta">
                <p>
                  This trains all three architectures for real — it can take from a few seconds
                  to a few minutes depending on dataset size.
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
                <span className="status-line-text">
                  <span className="spinner" /> Training and evaluating architectures… (
                  {run.status})
                </span>
                <button className="btn btn-ghost" onClick={handleCancel} disabled={cancelling}>
                  {cancelling ? "Cancelling…" : "Cancel"}
                </button>
              </div>
            )}

            {run && run.status === "cancelled" && !run.result && (
              <div className="status-line">Cancelled before any architecture finished.</div>
            )}

            {run && run.status === "error" && (
              <div className="error-banner">
                <strong>Run failed:</strong> {run.error}
              </div>
            )}

            {run && run.result && (
              <CompareResults result={run.result} datasetLabel={run.dataset_label} />
            )}
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
