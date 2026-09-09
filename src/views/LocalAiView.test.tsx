import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AppProvider } from "../context/AppContext";
import { LocalAiView } from "./LocalAiView";

describe("LocalAiView", () => {
  it("exposes detect, results, history, settings and help", async () => {
    const user = userEvent.setup();
    render(
      <AppProvider>
        <LocalAiView />
      </AppProvider>,
    );
    expect(screen.getByTestId("local-ai-view")).toBeInTheDocument();
    expect(screen.getByText("AI image detection")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Results" }));
    expect(screen.getByText(/No analysis yet/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "History" }));
    expect(screen.getByText("Local history")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Help" }));
    expect(screen.getByText("How this works")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(screen.getByText("Offline mode")).toBeInTheDocument();
    expect(screen.getByText("AI detector")).toBeInTheDocument();
  });
});
