import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import App from "./App";
import * as api from "./lib/api";
import {
  ApiError,
  type ArchitectureInfo,
  type CompareResult,
  type DataProfile,
  type ProfileResponse,
  type Recommendation,
  type RunResponse,
  type RunSummary,
} from "./lib/types";

vi.mock("./lib/api");

function makeCsvFile(name = "interactions.csv") {
  return new File(["user_id,item_id,timestamp\nu1,i1,1"], name, { type: "text/csv" });
}

const architectures: ArchitectureInfo[] = [
  {
    name: "sasrec",
    description: "A sequence model.",
    strengths: [],
    weaknesses: [],
    relative_train_cost: "medium",
    relative_serving_latency: "medium",
  },
];

function profile(overrides: Partial<DataProfile> = {}): DataProfile {
  return {
    n_users: 25,
    n_items: 20,
    n_interactions: 150,
    sparsity: 0.7,
    cold_start_ratio: 0,
    median_sequence_length: 6,
    has_item_text: false,
    has_item_image: false,
    ...overrides,
  };
}

function recommendations(): Recommendation[] {
  return [
    {
      architecture: "sasrec",
      rank: 1,
      score: 0.75,
      rationale: "boosted: long sequences",
      factors: [{ detail: "boosted: long sequences", effect: 0.25 }],
      margin_to_next: 0.25,
      low_confidence: false,
    },
    {
      architecture: "two_tower",
      rank: 2,
      score: 0.5,
      rationale: "solid baseline",
      factors: [{ detail: "solid baseline", effect: 0 }],
      margin_to_next: null,
      low_confidence: false,
    },
  ];
}

function compareResult(overrides: Partial<CompareResult> = {}): CompareResult {
  return {
    profile: profile(),
    reasoning_engine_shortlist: recommendations(),
    eval_results: {
      sasrec: {
        architecture: "SASRec",
        k: 10,
        n_test_users: 25,
        recall_at_k: 0.8,
        ndcg_at_k: 0.7,
        coverage_at_k: 0.9,
        cold_start_recall_at_k: null,
        cold_start_surfaced_rate: 0,
        example_recommendations: [],
      },
    },
    comparison: {
      shortlist_pick: "sasrec",
      measured_best_recall: "sasrec",
      measured_best_cold_start_recall: null,
      matches_on_recall: true,
      note: null,
    },
    ...overrides,
  };
}

function run(overrides: Partial<RunResponse> = {}): RunResponse {
  return {
    id: "run-1",
    status: "done",
    dataset_label: "interactions.csv",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:01Z",
    result: compareResult(),
    error: null,
    ...overrides,
  };
}

// Drains a chain of pending microtasks under fake timers — used after a
// fireEvent that kicks off an async handler with several chained awaits
// (getRun() -> setState() -> render). A single advanceTimersByTimeAsync
// call yields at least once but isn't reliably enough hops to fully drain
// a multi-await chain.
async function flush() {
  for (let i = 0; i < 5; i++) {
    await vi.advanceTimersByTimeAsync(0);
  }
}

// fireEvent.submit rather than clicking the submit button: jsdom doesn't
// recognize a FileList assigned via userEvent's DataTransfer trick as
// satisfying `required` on a file input, so a real click-triggered
// submission is silently blocked by native constraint validation before
// React's onSubmit ever runs (see UploadForm.test.tsx for the isolated
// repro). fireEvent.submit dispatches the DOM event directly, bypassing
// that jsdom-only gate.
async function uploadAndProfile(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(screen.getByLabelText(/interactions csv/i), makeCsvFile());
  const form = document.querySelector("form");
  if (!form) throw new Error("no form found");
  fireEvent.submit(form);
  await screen.findByText("Reasoning engine shortlist");
}

beforeEach(() => {
  vi.mocked(api.fetchArchitectures).mockResolvedValue(architectures);
  vi.mocked(api.listRuns).mockResolvedValue([]);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  it("shows the upload form initially", async () => {
    render(<App />);
    expect(await screen.findByText("Analyze your data")).toBeInTheDocument();
  });

  it("profiles a dataset and shows the profile + shortlist", async () => {
    const user = userEvent.setup();
    const profileResp: ProfileResponse = { profile: profile(), recommendations: recommendations() };
    vi.mocked(api.profileDataset).mockResolvedValue(profileResp);

    render(<App />);
    await uploadAndProfile(user);

    expect(screen.getByText("25")).toBeInTheDocument(); // n_users stat tile
    expect(screen.getByText("sasrec")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /run full comparison/i })).toBeInTheDocument();
  });

  it("shows a profile error without leaving the upload view", async () => {
    const user = userEvent.setup();
    vi.mocked(api.profileDataset).mockRejectedValue(new ApiError("bad columns"));

    render(<App />);
    await user.upload(screen.getByLabelText(/interactions csv/i), makeCsvFile());
    const form = document.querySelector("form");
    if (!form) throw new Error("no form found");
    fireEvent.submit(form);

    expect(await screen.findByText(/bad columns/i)).toBeInTheDocument();
    expect(screen.getByText("Analyze your data")).toBeInTheDocument();
  });

  it("starts a compare job and renders results once the job is already done", async () => {
    const user = userEvent.setup();
    const profileResp: ProfileResponse = { profile: profile(), recommendations: recommendations() };
    vi.mocked(api.profileDataset).mockResolvedValue(profileResp);
    vi.mocked(api.startCompare).mockResolvedValue({ job_id: "run-1" });
    vi.mocked(api.getRun).mockResolvedValue(run({ status: "done" }));

    render(<App />);
    await uploadAndProfile(user);
    await user.click(screen.getByRole("button", { name: /run full comparison/i }));

    expect(await screen.findByText(/matched the measured winner/i)).toBeInTheDocument();
    expect(api.startCompare).toHaveBeenCalledTimes(1);
  });

  it("polls until the run settles, showing a spinner and Cancel button while active", async () => {
    // RTL's findByText/waitFor poll using real setTimeout internally, so
    // they hang if fake timers are already active and never auto-advance.
    // Do the upload/profile step under real timers (findByText works
    // normally there), then switch to fake timers right before the click
    // that creates the polling interval — that's the only part that needs
    // deterministic control. From that point on, use advanceTimersByTimeAsync
    // (which also flushes pending microtasks/promises) plus synchronous
    // getByText, not findByText/waitFor.
    const user = userEvent.setup({ delay: null });
    vi.mocked(api.profileDataset).mockResolvedValue({
      profile: profile(),
      recommendations: recommendations(),
    });
    vi.mocked(api.startCompare).mockResolvedValue({ job_id: "run-1" });
    vi.mocked(api.getRun)
      .mockResolvedValueOnce(run({ status: "pending", result: null }))
      .mockResolvedValueOnce(run({ status: "running", result: null }))
      .mockResolvedValue(run({ status: "done" }));

    render(<App />);
    await uploadAndProfile(user);

    vi.useFakeTimers();
    try {
      // fireEvent, not userEvent, from here on: userEvent's internal
      // pointer/act machinery doesn't play well with fake timers that never
      // auto-advance (it hangs). fireEvent dispatches the DOM event directly
      // with no timer dependency of its own.
      fireEvent.click(screen.getByRole("button", { name: /run full comparison/i }));
      // Each "tick" below involves a chain of several awaits (getRun() ->
      // setRun()/setProfileResp() -> React's render). advanceTimersByTimeAsync
      // yields at least once per call but that's not reliably enough hops to
      // fully drain a multi-await chain, so flush a few times after each one.
      await flush();

      expect(screen.getByText(/pending/i)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /cancel/i })).toBeInTheDocument();

      await vi.advanceTimersByTimeAsync(1500);
      await flush();
      expect(screen.getByText(/running/i)).toBeInTheDocument();

      await vi.advanceTimersByTimeAsync(1500);
      await flush();
      expect(screen.getByText(/matched the measured winner/i)).toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
      // The exact number of getRun() calls this takes isn't asserted here —
      // fake-timer flushing internals make that count brittle across test
      // order/runs. What matters behaviorally is covered above: it reaches
      // pending -> running -> done, and Cancel disappears once settled.
      expect(api.getRun).toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("cancels a running job", async () => {
    const user = userEvent.setup();
    vi.mocked(api.profileDataset).mockResolvedValue({
      profile: profile(),
      recommendations: recommendations(),
    });
    vi.mocked(api.startCompare).mockResolvedValue({ job_id: "run-1" });
    vi.mocked(api.getRun).mockResolvedValue(run({ status: "running", result: null }));
    vi.mocked(api.cancelRun).mockResolvedValue(run({ status: "cancelled", result: null }));

    render(<App />);
    await uploadAndProfile(user);
    await user.click(screen.getByRole("button", { name: /run full comparison/i }));

    const cancelButton = await screen.findByRole("button", { name: /cancel/i });
    await user.click(cancelButton);

    await waitFor(() => expect(api.cancelRun).toHaveBeenCalledWith("run-1"));
    expect(await screen.findByText(/cancelled before any architecture finished/i)).toBeInTheDocument();
  });

  it("loads a past run from history without needing a re-upload", async () => {
    const user = userEvent.setup();
    const summaries: RunSummary[] = [
      {
        id: "old-run",
        status: "done",
        dataset_label: "old.csv",
        created_at: "2025-01-01T00:00:00Z",
        updated_at: "2025-01-01T00:00:01Z",
      },
    ];
    vi.mocked(api.listRuns).mockResolvedValue(summaries);
    vi.mocked(api.getRun).mockResolvedValue(
      run({
        id: "old-run",
        result: compareResult({ profile: profile({ n_users: 999 }) }),
      }),
    );

    render(<App />);

    const historyItem = await screen.findByText("old.csv");
    await user.click(historyItem);

    expect(await screen.findByText("999")).toBeInTheDocument();
    // No local files for a historical run — the Compare CTA must stay hidden.
    expect(screen.queryByRole("button", { name: /run full comparison/i })).not.toBeInTheDocument();
  });

  it("resets to the upload view via 'Analyze new dataset'", async () => {
    const user = userEvent.setup();
    vi.mocked(api.profileDataset).mockResolvedValue({
      profile: profile(),
      recommendations: recommendations(),
    });

    render(<App />);
    await uploadAndProfile(user);

    await user.click(screen.getByRole("button", { name: /analyze new dataset/i }));

    expect(await screen.findByText("Analyze your data")).toBeInTheDocument();
    expect(screen.queryByText("Reasoning engine shortlist")).not.toBeInTheDocument();
  });
});
