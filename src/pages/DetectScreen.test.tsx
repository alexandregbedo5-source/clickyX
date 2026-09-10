import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { LocalAiProvider } from "../ui/local-ai";
import { DetectScreen } from "./DetectScreen";

describe("DetectScreen", () => {
  it("renders the drop zone and a disabled analyze action", () => {
    render(
      <LocalAiProvider>
        <DetectScreen />
      </LocalAiProvider>,
    );
    expect(screen.getByText("AI image detection")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyze image" })).toBeDisabled();
    expect(screen.getByText(/Drop an image/)).toBeInTheDocument();
  });
});
