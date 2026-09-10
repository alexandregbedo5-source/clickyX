import { describe, expect, it } from "vitest";
import { isAcceptedImage, validateImageFile } from "./image";

describe("image helpers", () => {
  it("accepts common image types", () => {
    const file = new File(["x"], "shot.png", { type: "image/png" });
    expect(isAcceptedImage(file)).toBe(true);
    expect(validateImageFile(file)).toBeNull();
  });

  it("rejects non-images", () => {
    const file = new File(["x"], "notes.txt", { type: "text/plain" });
    expect(validateImageFile(file)).toMatch(/PNG/);
  });
});
