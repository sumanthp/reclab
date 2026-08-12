import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { UploadForm } from "./UploadForm";

function makeCsvFile(name = "interactions.csv") {
  return new File(["user_id,item_id,timestamp\nu1,i1,1"], name, { type: "text/csv" });
}

// Submitting via fireEvent rather than clicking the submit button: jsdom
// doesn't recognize a FileList assigned via userEvent's DataTransfer trick
// as satisfying `required` on a file input, so a real click-triggered
// submission is silently blocked by native constraint validation before
// React's onSubmit ever runs. fireEvent.submit dispatches the DOM event
// directly, bypassing that (jsdom-only) gate.
function submit(container: HTMLElement) {
  const form = container.querySelector("form");
  if (!form) throw new Error("no form in container");
  fireEvent.submit(form);
}

describe("UploadForm", () => {
  it("disables submit until an interactions file is chosen", async () => {
    const user = userEvent.setup();
    render(<UploadForm onAnalyze={vi.fn()} loading={false} error={null} />);

    expect(screen.getByRole("button", { name: /profile dataset/i })).toBeDisabled();

    const input = screen.getByLabelText(/interactions csv/i);
    await user.upload(input, makeCsvFile());

    expect(screen.getByRole("button", { name: /profile dataset/i })).toBeEnabled();
  });

  it("calls onAnalyze with the chosen file and column names on submit", async () => {
    const user = userEvent.setup();
    const onAnalyze = vi.fn();
    const { container } = render(
      <UploadForm onAnalyze={onAnalyze} loading={false} error={null} />,
    );

    const interactionsFile = makeCsvFile();
    await user.upload(screen.getByLabelText(/interactions csv/i), interactionsFile);

    const userColInput = screen.getByLabelText(/user column/i);
    await user.clear(userColInput);
    await user.type(userColInput, "custom_user");

    submit(container);

    expect(onAnalyze).toHaveBeenCalledTimes(1);
    expect(onAnalyze).toHaveBeenCalledWith(interactionsFile, null, "custom_user", "item_id");
  });

  it("includes the item metadata file when one is chosen", async () => {
    const user = userEvent.setup();
    const onAnalyze = vi.fn();
    const { container } = render(
      <UploadForm onAnalyze={onAnalyze} loading={false} error={null} />,
    );

    const interactionsFile = makeCsvFile("interactions.csv");
    const metadataFile = makeCsvFile("meta.csv");
    await user.upload(screen.getByLabelText(/interactions csv/i), interactionsFile);
    await user.upload(screen.getByLabelText(/item metadata csv/i), metadataFile);
    submit(container);

    expect(onAnalyze).toHaveBeenCalledWith(interactionsFile, metadataFile, "user_id", "item_id");
  });

  it("does not call onAnalyze if submitted with no interactions file", () => {
    const onAnalyze = vi.fn();
    const { container } = render(
      <UploadForm onAnalyze={onAnalyze} loading={false} error={null} />,
    );

    submit(container);

    expect(onAnalyze).not.toHaveBeenCalled();
  });

  it("shows the loading label and disables submit while loading", () => {
    render(<UploadForm onAnalyze={vi.fn()} loading={true} error={null} />);
    expect(screen.getByRole("button", { name: /profiling/i })).toBeDisabled();
  });

  it("renders an error message when given one", () => {
    render(<UploadForm onAnalyze={vi.fn()} loading={false} error="something broke" />);
    expect(screen.getByText(/something broke/i)).toBeInTheDocument();
  });
});
