import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ShortlistCards } from "./ShortlistCards";
import type { ArchitectureInfo, Recommendation } from "../lib/types";

const shortlist: Recommendation[] = [
  { architecture: "sasrec", rank: 1, score: 0.75, rationale: "boosted: long sequences" },
  { architecture: "two_tower", rank: 2, score: 0.5, rationale: "solid baseline" },
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
    expect(screen.getByText(/boosted: long sequences/i)).toBeInTheDocument();
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
    // should still show the raw score, just no strengths/weaknesses.
    const detailsButtons = screen.getAllByText("Details");
    await user.click(detailsButtons[1]);

    expect(screen.getByText(/score = 0\.500/)).toBeInTheDocument();
  });
});
