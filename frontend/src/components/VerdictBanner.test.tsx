import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { VerdictBanner } from "./VerdictBanner";
import type { ComparisonSummary } from "../lib/types";

describe("VerdictBanner", () => {
  it("renders a match banner when the shortlist pick matches the measured best", () => {
    const comparison: ComparisonSummary = {
      shortlist_pick: "sasrec",
      measured_best_recall: "sasrec",
      measured_best_cold_start_recall: null,
      matches_on_recall: true,
      note: null,
    };
    render(<VerdictBanner comparison={comparison} />);

    expect(screen.getByText(/matched the measured winner/i)).toBeInTheDocument();
    expect(screen.getAllByText("sasrec").length).toBeGreaterThan(0);
  });

  it("renders a mismatch banner naming both picks, never softened into a match", () => {
    const comparison: ComparisonSummary = {
      shortlist_pick: "two_tower",
      measured_best_recall: "hybrid_llm",
      measured_best_cold_start_recall: null,
      matches_on_recall: false,
      note: null,
    };
    render(<VerdictBanner comparison={comparison} />);

    expect(screen.getByText(/did not match the measured winner/i)).toBeInTheDocument();
    expect(screen.getByText("two_tower")).toBeInTheDocument();
    expect(screen.getByText("hybrid_llm")).toBeInTheDocument();
  });

  it("includes the cold-start-recall note when present, on a mismatch", () => {
    const comparison: ComparisonSummary = {
      shortlist_pick: "hybrid_llm",
      measured_best_recall: "two_tower",
      measured_best_cold_start_recall: "hybrid_llm",
      matches_on_recall: false,
      note: "it does win on cold-start recall",
    };
    render(<VerdictBanner comparison={comparison} />);

    expect(screen.getByText(/it does win on cold-start recall/i)).toBeInTheDocument();
  });

  it("renders nothing when nothing was measured (matches_on_recall is null)", () => {
    const comparison: ComparisonSummary = {
      shortlist_pick: "sasrec",
      measured_best_recall: null,
      measured_best_cold_start_recall: null,
      matches_on_recall: null,
      note: null,
    };
    const { container } = render(<VerdictBanner comparison={comparison} />);

    expect(container).toBeEmptyDOMElement();
  });
});
