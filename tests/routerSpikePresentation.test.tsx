import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { checkedRouterViewport, RouterSpikeView, routerChannelIdentity } from "../src/renderer/src/RouterSpikeView";
import type { RouterSpikeActivity } from "../src/shared/routerActivity";

const receipt: RouterSpikeActivity = { brainId: "brain-a", source: "actual-lif-hard-spikes", owner: "root.router.population", channelIdentity: "brainId:root.router.population:channelIndex", available: true, enabled: true, observed: true, captureId: "capture-a", epoch: "epoch-a", totalChannels: 100000, channelStart: 9000, admittedCount: 2, requestedCount: 2, ringBytes: 1024, frameCapacity: 10, lastSequence: 4, firstAvailableSequence: 1, nextSequence: 4, droppedFrames: 0, missedFrames: 0, hasMore: false, frames: [{ sequence: 4, observedAtUnix: 1, monotonicNs: "12345678901234567890", channelStart: 9000, spikes: [1, 0] }], semanticNeuronMapping: false, synapseFiringObserved: false };
describe("actual router viewport presentation", () => {
  it("stays off until requested and never renders made-up spike cells", () => {
    const observe = vi.fn(); const html = renderToStaticMarkup(<RouterSpikeView brainId="brain-a" observe={observe} />);
    expect(observe).not.toHaveBeenCalled(); expect(html).toContain("Observation off"); expect(html).not.toContain('data-spike=');
    expect(html).toContain("Connections to idea assemblies are not mapped");
  });
  it("preserves actual binary silence, exact channel IDs and monotonic strings", () => {
    const value = checkedRouterViewport(receipt, "brain-a", { id: "capture-a", epoch: "epoch-a", afterSequence: 3 });
    expect(value.frames[0].spikes).toEqual([1, 0]); expect(value.frames[0].monotonicNs).toBe("12345678901234567890");
    expect(routerChannelIdentity("brain-a", 9001)).toBe("brain-a:root.router.population:9001");
  });
  it("rejects cross-brain, old-epoch, non-binary and repeated frame receipts", () => {
    const capture = { id: "capture-a", epoch: "epoch-a", afterSequence: 3 };
    for (const value of [{ ...receipt, brainId: "other" }, { ...receipt, epoch: "old" }, { ...receipt, semanticNeuronMapping: true },
      { ...receipt, frames: [{ ...receipt.frames[0], spikes: [0.5, 0] }] }]) expect(() => checkedRouterViewport(value as RouterSpikeActivity, "brain-a", capture)).toThrow();
    expect(() => checkedRouterViewport(receipt, "brain-a", { ...capture, afterSequence: 4 })).toThrow();
  });
});
