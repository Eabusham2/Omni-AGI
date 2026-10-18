import { describe, expect, it } from "vitest";
import { nativeConnectionLabel } from "../src/renderer/src/neuralRuntimePresentation";

describe("native graph contact labels", () => {
  it("separates exact signs from measured strength and repeated contacts", () => {
    expect(nativeConnectionLabel({ effectiveWeight: -1, connectionStrength: 16 })).toBe("−1 × 16");
    expect(nativeConnectionLabel({ effectiveWeight: 1, connectionStrength: 4, isRepeatedContact: true })).toBe("+1 × 4 · repeated");
    expect(nativeConnectionLabel({ effectiveWeight: 0, connectionStrength: 7 })).toBe("0 × 7");
  });
  it("never invents missing gain or turns an invalid weight into zero", () => {
    expect(nativeConnectionLabel({ effectiveWeight: 1 })).toBe("+1");
    expect(nativeConnectionLabel({ effectiveWeight: 1, connectionStrength: NaN })).toBe("+1");
    expect(nativeConnectionLabel({ effectiveWeight: 0.5 })).toBe("—");
  });
});
