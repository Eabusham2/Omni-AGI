import { describe, expect, it, vi } from "vitest";
import { hydrateLibrarySubstrateTotals } from "../src/renderer/src/librarySubstrateTotals";

describe("Library authoritative substrate hydration", () => {
  it("retries transient pointer races and resolves each missing card independently", async () => {
    const calls = new Map<string, number>();
    const onResolved = vi.fn();
    await hydrateLibrarySubstrateTotals({
      brainIds: ["copy", "original"],
      maximumAttempts: 4,
      cancelled: () => false,
      wait: async () => undefined,
      load: async (brainId) => {
        const count = (calls.get(brainId) ?? 0) + 1;
        calls.set(brainId, count);
        if (brainId === "copy" && count < 3) {
          if (count === 1) throw new Error("pointer swapped during read");
          return null;
        }
        return {
          brainId,
          revision: brainId.repeat(64).slice(0, 64),
          source: "validated-persisted-substrate" as const,
          totals: brainId === "copy"
            ? { neurons: 15_200, assemblies: 165, synapses: 4_818_791 }
            : { neurons: 14_973, assemblies: 121, synapses: 4_812_443 }
        };
      },
      onResolved
    });

    expect(calls.get("original")).toBe(1);
    expect(calls.get("copy")).toBe(3);
    expect(onResolved).toHaveBeenCalledTimes(2);
    expect(onResolved).toHaveBeenCalledWith(expect.objectContaining({
      brainId: "copy",
      totals: { neurons: 15_200, assemblies: 165, synapses: 4_818_791 }
    }));
  });

  it("stops retries when Library unmounts", async () => {
    let cancelled = false;
    const load = vi.fn(async () => null);
    await hydrateLibrarySubstrateTotals({
      brainIds: ["copy"],
      maximumAttempts: 5,
      cancelled: () => cancelled,
      wait: async () => {
        cancelled = true;
      },
      load,
      onResolved: vi.fn()
    });
    expect(load).toHaveBeenCalledOnce();
  });
});
