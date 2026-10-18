import { describe, expect, it } from "vitest";
import { clockDuration } from "../src/renderer/src/ClockWorkspacePanel";

describe("clock duration presentation", () => {
  it.each([
    [0, "0:00.0"], [-1, "0:00.0"], [999, "0:01.0"],
    [59949, "0:59.9"], [59999, "1:00.0"],
    [3599999, "1:00:00.0"], [3661000, "1:01:01.0"]
  ])("rounds %s milliseconds with valid carry", (milliseconds, expected) => {
    expect(clockDuration(milliseconds)).toBe(expected);
  });
  it("does not invent a measured duration for non-finite input", () => {
    expect(clockDuration(Number.NaN)).toBe("—");
    expect(clockDuration(Number.POSITIVE_INFINITY)).toBe("—");
  });
});
