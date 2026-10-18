import { describe, expect, it, vi } from "vitest";
import { visitChatDirectionTurns } from "../src/renderer/src/chatDirectionQueries";
import type { SteerDirectionRecord } from "../src/shared/steerDirections";

describe("exact direction lookup covers history without stale ownership", () => {
  it("visits all201 unique displayed turns in100/100/1 batches", async () => {
    const ids = Array.from({ length: 201 }, (_, i) => `turn-${i}`), visited: string[] = [];
    const forTurns = vi.fn(async (brainId: string, batch: string[]): Promise<SteerDirectionRecord[]> => {
      expect(brainId).toBe("brain-A"); expect(batch.length).toBeLessThanOrEqual(100); visited.push(...batch); return [];
    });
    const accept = vi.fn();
    await visitChatDirectionTurns({ forTurns }, "brain-A", [...ids, ids[0]!], accept, () => true);
    expect(visited).toEqual(ids); expect(forTurns.mock.calls.map(([, batch]) => batch.length)).toEqual([100, 100, 1]);
    expect(accept).toHaveBeenCalledTimes(3);
  });
  it("does not accept a late old-instance read or issue its next batch", async () => {
    let current = true;
    const forTurns = vi.fn(async (): Promise<SteerDirectionRecord[]> => { current = false; return []; });
    const accept = vi.fn();
    await visitChatDirectionTurns({ forTurns }, "brain-A", Array.from({ length: 201 }, (_, i) => `turn-${i}`), accept, () => current);
    expect(forTurns).toHaveBeenCalledTimes(1); expect(accept).not.toHaveBeenCalled();
  });
  it("does not start a stale query and propagates a current query error", async () => {
    const forTurns = vi.fn(async (): Promise<SteerDirectionRecord[]> => { throw new Error("Exact host query failed"); });
    await visitChatDirectionTurns({ forTurns }, "brain-A", ["turn"], () => {}, () => false);
    expect(forTurns).not.toHaveBeenCalled();
    await expect(visitChatDirectionTurns({ forTurns }, "brain-A", ["turn"], () => {}, () => true)).rejects.toThrow("Exact host query failed");
  });
});
