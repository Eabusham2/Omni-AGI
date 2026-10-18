import { expect, it } from "vitest";
import { retainRouterObservation } from "../src/renderer/src/RouterSpikeView";
import type { RouterSpikeActivity } from "../src/shared/routerActivity";

const sample: RouterSpikeActivity = {
  brainId: "fixture", source: "actual-lif-hard-spikes", owner: "root.router.population",
  channelIdentity: "brainId:root.router.population:channelIndex", available: true, enabled: true,
  observed: true, captureId: "capture", epoch: "a".repeat(32), totalChannels: 2,
  channelStart: 0, admittedCount: 2, requestedCount: 2, ringBytes: 512, frameCapacity: 8,
  lastSequence: 1, firstAvailableSequence: 1, nextSequence: 1, droppedFrames: 0, missedFrames: 0, hasMore: false,
  frames: [{ sequence: 1, observedAtUnix: 1_800_000_000, monotonicNs: "100", channelStart: 0, spikes: [1, 0] }],
  transmissionPage: { source: "actual-packed-contact-transmissions", owner: "root.router.synapses", available: true,
    enabled: true, observed: true, captureEpoch: "a".repeat(32), epoch: "b".repeat(32), layoutDigest: "c".repeat(64),
    selectedContactOffset: 0, selectedContactCount: 1, ringCapacity: 8, lastSequence: 1, firstAvailableSequence: 1,
    nextSequence: 1, droppedRecords: 0, missedRecords: 0, pageSize: 128, hasMore: false, postsynapticFiringObserved: false,
    events: [{ sequence: 1, operationSequence: 1, observedAtUnix: 1_800_000_000, monotonicNs: "100",
      contactId: "fixture:root.router.synapses:r0_c0:1:0", blockId: "r0_c0", sourceChannel: 0, targetChannel: 1,
      sourceId: "fixture:root.router.population:0", targetId: "fixture:root.router.population:1", sourceInput: 1,
      effectiveWeight: -1, connectionStrength: 2, connectionResistance: 0, effectiveContribution: -2,
      classification: "inhibitory", transmissionObserved: true, postsynapticFiringObserved: false }] },
  semanticNeuronMapping: false, synapseFiringObserved: false
};
const emptyPoll = (): RouterSpikeActivity => ({ ...sample, frames: [], transmissionPage: { ...sample.transmissionPage!, events: [] } });

it("keeps the last actual event/time on an empty transport poll without changing raw cursors", () => {
  const incoming = emptyPoll(), view = retainRouterObservation(sample, incoming);
  expect(view.frames).toEqual(sample.frames); expect(view.transmissionPage?.events).toEqual(sample.transmissionPage?.events);
  expect(incoming.frames).toEqual([]); expect(incoming.transmissionPage?.events).toEqual([]);
  expect(view.nextSequence).toBe(incoming.nextSequence); expect(view.transmissionPage?.nextSequence).toBe(incoming.transmissionPage?.nextSequence);
});
it("replaces a prior highlight with a real new silent frame and zero transmission", () => {
  const incoming = emptyPoll(); incoming.frames = [{ ...sample.frames[0]!, sequence: 2, spikes: [0, 0] }];
  incoming.transmissionPage!.events = [{ ...sample.transmissionPage!.events[0]!, sequence: 2, sourceInput: 0, effectiveContribution: 0, classification: "zero" }];
  expect(retainRouterObservation(sample, incoming).frames[0]?.spikes).toEqual([0, 0]);
  expect(retainRouterObservation(sample, incoming).transmissionPage?.events[0]?.classification).toBe("zero");
});
it("does not carry events across owners, capture/layout/page epochs, viewport changes or Stop", () => {
  for (const changed of [{ brainId: "other" }, { captureId: "other" }, { epoch: "d".repeat(32) }, { channelStart: 1 }, { enabled: false }]) {
    expect(retainRouterObservation(sample, { ...emptyPoll(), ...changed }).frames).toEqual([]);
  }
  for (const changed of [{ epoch: "d".repeat(32) }, { layoutDigest: "e".repeat(64) }, { selectedContactOffset: 128 }]) {
    const next = emptyPoll(); next.transmissionPage = { ...next.transmissionPage!, ...changed };
    expect(retainRouterObservation(sample, next).transmissionPage?.events).toEqual([]);
  }
});
