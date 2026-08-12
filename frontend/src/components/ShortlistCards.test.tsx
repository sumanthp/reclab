import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ShortlistCards } from "./ShortlistCards";
import type { ArchitectureInfo, Recommendation } from "../lib/types";

const shortlist: Recommendation[] = [
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

const architectureInfo: Record<string, ArchitectureInfo> = {
  sasrec: {
    name: "sasrec",
    description: "A sequence model.",
    strengths: ["Handles order and recency"],
    weaknesses: ["Needs enough sequence length"],
    relative_train_cost: "medium",
    relative_serving_latency: "medium",
  },
};

describe("ShortlistCards", () => {
  it("renders each architecture's rank, name, score, and rationale", () => {
    render(<ShortlistCards shortlist={shortlist} architectureInfo={{}} />);

    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("sasrec")).toBeInTheDocument();
    expect(screen.getByText("0.75")).toBeInTheDocument();
    // Layer 1 (rationale) and Layer 2 (factors) legitimately share this
    // text — factors compose the rationale — so there are two matches.
    expect(screen.getAllByText(/boosted: long sequences/i).length).toBeGreaterThanOrEqual(1);
  });

  it("capitalizes the first letter of the rationale", () => {
    render(<ShortlistCards shortlist={shortlist} architectureInfo={{}} />);
    expect(screen.getByText(/^Boosted: long sequences/)).toBeInTheDocument();
  });

  it("expands the Details disclosure (native <details>, closed by default) on click", async () => {
    // Content is present in the DOM either way — a real browser hides it via
    // the `details:not([open])` UA stylesheet rule, which jsdom doesn't lay
    // out. What this component controls is the `open` attribute itself.
    const user = userEvent.setup();
    render(<ShortlistCards shortlist={shortlist} architectureInfo={architectureInfo} />);

    const [firstDetails] = document.querySelectorAll("details.layer2");
    expect(firstDetails).not.toHaveAttribute("open");

    const [firstSummary] = screen.getAllByText("Details");
    await user.click(firstSummary);

    expect(firstDetails).toHaveAttribute("open");
    expect(screen.getByText(/handles order and recency/i)).toBeInTheDocument();
    expect(screen.getByText(/train cost: medium/i)).toBeInTheDocument();
  });

  it("degrades gracefully when architecture info is unavailable for a card", async () => {
    const user = userEvent.setup();
    render(<ShortlistCards shortlist={shortlist} architectureInfo={{}} />);

    // two_tower has no entry in architectureInfo — expanding its Details
    // should still show its scoring factors, just no strengths/weaknesses.
    const detailsButtons = screen.getAllByText("Details");
    await user.click(detailsButtons[1]);

    const factorDetails = document.querySelectorAll(".factor-detail");
    expect(factorDetails[1]).toHaveTextContent(/solid baseline/i);
  });

  it("shows a 'close call' flag on rank 1 when low_confidence is true", () => {
    const lowConfidenceShortlist: Recommendation[] = [
      { ...shortlist[0], low_confidence: true },
      shortlist[1],
    ];
    render(<ShortlistCards shortlist={lowConfidenceShortlist} architectureInfo={{}} />);

    expect(screen.getByText(/close call/i)).toBeInTheDocument();
  });

  it("does not show a 'close call' flag when low_confidence is false", () => {
    render(<ShortlistCards shortlist={shortlist} architectureInfo={{}} />);
    expect(screen.queryByText(/close call/i)).not.toBeInTheDocument();
  });

  it("renders each factor's effect and detail", async () => {
    const user = userEvent.setup();
    render(<ShortlistCards shortlist={shortlist} architectureInfo={{}} />);

    await user.click(screen.getAllByText("Details")[0]);

    expect(screen.getByText("+0.25")).toBeInTheDocument();
  });
});
