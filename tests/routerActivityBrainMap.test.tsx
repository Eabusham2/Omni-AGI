import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RouterActivityBrainMap, routerActivityGraph } from "../src/renderer/src/RouterActivityBrainMap";
import type { RouterContactRecord, RouterSpikeActivity } from "../src/shared/routerActivity";

const base: RouterContactRecord = {
  id: "brain-a:root.router.synapses:r0_c0:11:10", owner: "root.router.synapses",
  sourceChannel: 10, targetChannel: 11,
  sourceId: "brain-a:root.router.population:10", targetId: "brain-a:root.router.population:11",
  blockId: "r0_c0", parallelLayer: 0, effectiveWeight: 1, connectionStrength: 4,
  connectionResistance: 3, effectiveContribution: 4, eligibility: 2, stability: 0.5, uses: 7,
  isRepeatedContact: false, parallelOrdinal: 0,
  parallelRootId: "brain-a:root.router.synapses:r0_c0:11:10", firingObserved: false
};
const repeated: RouterContactRecord = { ...base,
  id: "brain-a:root.router.synapses:r0_c0_p1:11:10", blockId: "r0_c0_p1", parallelLayer: 1,
  effectiveWeight: -1, connectionStrength: 2, effectiveContribution: -2,
  isRepeatedContact: true, parallelOrdinal: 1
};
const activity: RouterSpikeActivity = {
  brainId: "brain-a", source: "actual-lif-hard-spikes", owner: "root.router.population",
  channelIdentity: "brainId:root.router.population:channelIndex", available: true, enabled: true,
  observed: true, captureId: "capture", epoch: "a".repeat(32), totalChannels: 1_000,
  channelStart: 10, admittedCount: 3, requestedCount: 3, ringBytes: 512, frameCapacity: 8,
  lastSequence: 2, firstAvailableSequence: 1, nextSequence: 2, droppedFrames: 0, missedFrames: 0,
  hasMore: false, frames: [
    { sequence: 1, observedAtUnix: 1_800_000_001, monotonicNs: "100", channelStart: 10, spikes: [1, 0, 0] },
    { sequence: 2, observedAtUnix: 1_800_000_002, monotonicNs: "200", channelStart: 10, spikes: [1, 0, 1] }
  ],
  contactSnapshot: { source: "actual-allocated-router-contacts", owner: "root.router.synapses",
    available: true, observed: true, capturedAtUnix: 1_800_000_000, topologyGeneration: 4,
    totalContacts: 2, offset: 0, returned: 2, pageSize: 128, hasMore: false,
    contacts: [base, repeated], synapseFiringObserved: false },
  transmissionPage: { source: "actual-packed-contact-transmissions", owner: "root.router.synapses",
    available: true, enabled: true, observed: true, captureEpoch: "a".repeat(32), epoch: "b".repeat(32),
    layoutDigest: "c".repeat(64), selectedContactOffset: 0, selectedContactCount: 2,
    ringCapacity: 4096, lastSequence: 2, firstAvailableSequence: 1, nextSequence: 2,
    droppedRecords: 0, missedRecords: 0, pageSize: 128, hasMore: false,
    events: [
      { sequence: 1, operationSequence: 1, observedAtUnix: 1_800_000_003, monotonicNs: "300",
        contactId: base.id, blockId: base.blockId, sourceChannel: 10, targetChannel: 11,
        sourceId: base.sourceId, targetId: base.targetId, sourceInput: 1, effectiveWeight: -1,
        connectionStrength: 3, connectionResistance: 2, effectiveContribution: -3,
        classification: "inhibitory", transmissionObserved: true, postsynapticFiringObserved: false },
      { sequence: 2, operationSequence: 1, observedAtUnix: 1_800_000_003, monotonicNs: "300",
        contactId: repeated.id, blockId: repeated.blockId, sourceChannel: 10, targetChannel: 11,
        sourceId: repeated.sourceId, targetId: repeated.targetId, sourceInput: 0, effectiveWeight: -1,
        connectionStrength: 2, connectionResistance: 0, effectiveContribution: 0,
        classification: "zero", transmissionObserved: true, postsynapticFiringObserved: false }
    ], postsynapticFiringObserved: false },
  semanticNeuronMapping: false, synapseFiringObserved: false
};

describe("exact router activity brain-map projection", () => {
  it("highlights only recorded hard-spike channels and never stored contacts", () => {
    const graph = routerActivityGraph(activity);
    expect(graph.nodes.map(node => [node.id, node.firingEvent?.count ?? 0])).toEqual([
      ["brain-a:root.router.population:10", 2], ["brain-a:root.router.population:11", 0],
      ["brain-a:root.router.population:12", 1]
    ]);
    expect(graph.nodes.every(node => node.activationObserved === false)).toBe(true);
    expect(graph.edges).toHaveLength(2);
    expect(graph.edges.map(edge => [edge.id, edge.signedStrength, edge.firingEvent])).toEqual([
      [base.id, 4, undefined], [repeated.id, -2, undefined]
    ]);
    expect(graph.edges[1]).toMatchObject({ isRepeatedContact: true, parallelOrdinal: 1,
      parallelRootId: base.id });
    expect(graph.edges[0]!.transmissionEvent).toMatchObject({ classification: "inhibitory",
      contribution: -3, count: 1 });
    expect(graph.edges.every(edge => edge.firingEvent === undefined)).toBe(true);
  });

  it("renders capture-time, later transmission operands and unobserved postsynaptic firing separately", () => {
    const html = renderToStaticMarkup(<RouterActivityBrainMap activity={activity} />);
    expect(html.match(/data-firing="observed"/g)).toHaveLength(2);
    expect(html.match(/data-contact-id=/g)).toHaveLength(2);
    expect(html.match(/data-contact-id=[^>]+data-firing="unobserved"/g)).toHaveLength(2);
    expect(html).toContain('data-transmission="inhibitory"');
    expect(html).toContain('data-transmission="zero"');
    expect(html).toContain("Postsynaptic/path firing remains unobserved");
    expect(html).toContain("No token or semantic-neuron mapping");
    expect(html).toContain("captured contacts");
  });

  it("rejects a semantic or cross-viewport identity instead of inventing a layout edge", () => {
    expect(() => routerActivityGraph({ ...activity, semanticNeuronMapping: true as false })).toThrow();
    const outside = { ...base, sourceChannel: 9,
      sourceId: "brain-a:root.router.population:9",
      id: "brain-a:root.router.synapses:r0_c0:11:9",
      parallelRootId: "brain-a:root.router.synapses:r0_c0:11:9" };
    expect(() => routerActivityGraph({ ...activity, contactSnapshot: { ...activity.contactSnapshot!,
      totalContacts: 1, returned: 1, contacts: [outside] } })).toThrow(/owner\/channel identity/);
  });
});
