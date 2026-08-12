import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { CompareResults } from "./CompareResults";
import type { CompareResult } from "../lib/types";

function result(overrides: Partial<CompareResult> = {}): CompareResult {
  return {
    profile: {
      n_users: 10,
      n_items: 20,
      n_interactions: 100,
      sparsity: 0.5,
      cold_start_ratio: 0,
      median_sequence_length: 10,
      has_item_text: false,
      has_item_image: false,
    },
    reasoning_engine_shortlist: [],
    eval_results: {
      two_tower: {
        architecture: "TwoTower",
        k: 10,
        n_test_users: 10,
        recall_at_k: 0.5,
        ndcg_at_k: 0.4,
        coverage_at_k: 0.6,
        cold_start_recall_at_k: null,
        cold_start_surfaced_rate: 0,
        example_recommendations: [
          { user_id: "u1", recommended: ["a", "b"], held_out: ["a"], hit: true },
          { user_id: "u2", recommended: ["c", "d"], held_out: ["z"], hit: false },
        ],
      },
    },
    comparison: {
      shortlist_pick: "two_tower",
      measured_best_recall: "two_tower",
      measured_best_cold_start_recall: null,
      matches_on_recall: true,
      note: null,
    },
    ...overrides,
  };
}

describe("CompareResults", () => {
  let createObjectURL: ReturnType<typeof vi.fn>;
  let revokeObjectURL: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    createObjectURL = vi.fn(() => "blob:mock-url");
    revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("shows example recommendations with hit/miss state once expanded", async () => {
    const user = userEvent.setup();
    render(<CompareResults result={result()} />);

    await user.click(screen.getByText(/example recommendations/i));

    expect(screen.getByText("user u1")).toBeInTheDocument();
    expect(screen.getByText("hit")).toBeInTheDocument();
    expect(screen.getByText("miss")).toBeInTheDocument();
  });

  it("downloads the result as JSON when clicked", async () => {
    const user = userEvent.setup();
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});

    render(<CompareResults result={result()} datasetLabel="My Dataset.csv" />);
    await user.click(screen.getByRole("button", { name: /download results/i }));

    expect(createObjectURL).toHaveBeenCalledTimes(1);
    expect(clickSpy).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");

    clickSpy.mockRestore();
  });

  it("slugifies the dataset label into the downloaded filename", async () => {
    const user = userEvent.setup();
    let capturedHref = "";
    let capturedDownload = "";
    const clickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(function (this: HTMLAnchorElement) {
        capturedHref = this.href;
        capturedDownload = this.download;
      });

    render(<CompareResults result={result()} datasetLabel="My Dataset.csv" />);
    await user.click(screen.getByRole("button", { name: /download results/i }));

    expect(capturedDownload).toBe("reclab-my-dataset-csv.json");
    expect(capturedHref).toContain("blob:mock-url");

    clickSpy.mockRestore();
  });
});
