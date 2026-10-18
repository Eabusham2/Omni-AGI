import { describe, expect, it } from "vitest";
import { attachmentMenuPosition } from "../src/renderer/src/attachmentMenuPosition";

describe("top-layer attachment placement", () => {
  it("keeps the short-window popup above the composer and inside both edges", () => {
    const result = attachmentMenuPosition({ left: 680, top: 405, bottom: 440 }, { width: 210, height: 340 }, { width: 720, height: 480 });
    expect(result).toEqual({ left: 502, top: 109, maxHeight: 288 });
    expect(result.top + Math.min(340, result.maxHeight)).toBeLessThanOrEqual(405 - 8);
  });
  it("opens below a top-edge trigger without exceeding the current viewport", () => {
    const result = attachmentMenuPosition({ left: 4, top: 12, bottom: 46 }, { width: 210, height: 340 }, { width: 320, height: 480 });
    expect(result).toEqual({ left: 8, top: 54, maxHeight: 288 });
    expect(result.top + result.maxHeight).toBeLessThanOrEqual(480 - 8);
  });
});
