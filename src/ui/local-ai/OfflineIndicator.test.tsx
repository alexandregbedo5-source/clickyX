import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { OfflineIndicator } from "./OfflineIndicator";

describe("OfflineIndicator", () => {
  it("renders a compact connectivity chip", async () => {
    render(<OfflineIndicator compact />);
    expect(await screen.findByRole("button")).toBeInTheDocument();
  });
});
