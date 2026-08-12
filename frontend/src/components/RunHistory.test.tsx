import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RunHistory } from "./RunHistory";
import type { RunSummary } from "../lib/types";

function run(overrides: Partial<RunSummary> = {}): RunSummary {
  return {
    id: "run-1",
    status: "done",
    dataset_label: "interactions.csv",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:01Z",
    ...overrides,
  };
}

describe("RunHistory", () => {
  it("renders nothing when there are no runs", () => {
    const { container } = render(<RunHistory runs={[]} onSelect={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("lists each run's dataset label and status", () => {
    render(
      <RunHistory
        runs={[run({ id: "a", dataset_label: "a.csv", status: "done" }), run({ id: "b", dataset_label: "b.csv", status: "error" })]}
        onSelect={vi.fn()}
      />,
    );

    expect(screen.getByText("a.csv")).toBeInTheDocument();
    expect(screen.getByText("b.csv")).toBeInTheDocument();
    expect(screen.getByText("Done")).toBeInTheDocument();
    expect(screen.getByText("Failed")).toBeInTheDocument();
  });

  it("falls back to 'untitled' when dataset_label is null", () => {
    render(<RunHistory runs={[run({ dataset_label: null })]} onSelect={vi.fn()} />);
    expect(screen.getByText("untitled")).toBeInTheDocument();
  });

  it("calls onSelect with the run's id when clicked", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(<RunHistory runs={[run({ id: "the-run-id" })]} onSelect={onSelect} />);

    await user.click(screen.getByRole("button"));

    expect(onSelect).toHaveBeenCalledWith("the-run-id");
  });

  it("marks the active run distinctly", () => {
    render(
      <RunHistory
        runs={[run({ id: "a" }), run({ id: "b", dataset_label: "b.csv" })]}
        onSelect={vi.fn()}
        activeRunId="b"
      />,
    );

    const buttons = screen.getAllByRole("button");
    const activeButton = buttons.find((b) => b.textContent?.includes("b.csv"));
    expect(activeButton).toHaveClass("active");
  });
});
