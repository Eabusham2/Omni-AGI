import { describe, expect, it } from "vitest";
import { checkedRouterSpikeActivity, validateRouterSpikeQuery } from "../src/main/routerSpikeProtocol";
import type { RouterContactRecord, RouterSpikeActivity, RouterSpikeQuery } from "../src/shared/routerActivity";

const captureEpoch = "a".repeat(32);
const query: RouterSpikeQuery = { enabled: true, captureId: "capture", epoch: captureEpoch,
  contactOffset: 0, contactPageSize: 2, transmissionPageSize: 2 };
const contact = (changes: Partial<RouterContactRecord> = {}): RouterContactRecord => ({
  id: "brain-a:root.router.synapses:r0_c0:4:3", owner: "root.router.synapses",
  sourceChannel: 3, targetChannel: 4,
  sourceId: "brain-a:root.router.population:3", targetId: "brain-a:root.router.population:4",
  blockId: "r0_c0", parallelLayer: 0, effectiveWeight: 1, connectionStrength: 3,
  connectionResistance: 2, effectiveContribution: 3, eligibility: 1, stability: 0.25, uses: 4,
  isRepeatedContact: false, parallelOrdinal: 0,
  parallelRootId: "brain-a:root.router.synapses:r0_c0:4:3", firingObserved: false, ...changes
});
const activity = (contacts: RouterContactRecord[] = [contact()]): RouterSpikeActivity => ({
  brainId: "brain-a", source: "actual-lif-hard-spikes", owner: "root.router.population",
  channelIdentity: "brainId:root.router.population:channelIndex", available: true, enabled: true,
  observed: true, captureId: "capture", epoch: captureEpoch, totalChannels: 10,
  channelStart: 3, admittedCount: 3, requestedCount: 3, ringBytes: 256, frameCapacity: 4,
  lastSequence: 1, firstAvailableSequence: 1, nextSequence: 1, droppedFrames: 0,
  missedFrames: 0, hasMore: false,
  frames: [{ sequence: 1, observedAtUnix: 1_800_000_001, monotonicNs: "123", channelStart: 3, spikes: [1, 0, 0] }],
  contactSnapshot: { source: "actual-allocated-router-contacts", owner: "root.router.synapses",
    available: true, observed: true, capturedAtUnix: 1_800_000_000, topologyGeneration: 8,
    totalContacts: 3, offset: 0, returned: contacts.length, pageSize: 2, hasMore: contacts.length < 3,
    contacts, synapseFiringObserved: false },
  transmissionPage: { source: "actual-packed-contact-transmissions", owner: "root.router.synapses",
    available: true, enabled: true, observed: true, captureEpoch, epoch: "b".repeat(32),
    layoutDigest: "c".repeat(64), selectedContactOffset: 0, selectedContactCount: contacts.length,
    ringCapacity: 4096, lastSequence: 1, firstAvailableSequence: 1, nextSequence: 1,
    droppedRecords: 0, missedRecords: 0, pageSize: 2, hasMore: false,
    events: contacts.length ? [{ sequence: 1, operationSequence: 1, observedAtUnix: 1_800_000_002,
      monotonicNs: "456", contactId: contacts[0]!.id, blockId: contacts[0]!.blockId,
      sourceChannel: contacts[0]!.sourceChannel, targetChannel: contacts[0]!.targetChannel,
      sourceId: contacts[0]!.sourceId, targetId: contacts[0]!.targetId,
      sourceInput: 2, effectiveWeight: 1, connectionStrength: 3, connectionResistance: 2,
      effectiveContribution: 6, classification: "excitatory", transmissionObserved: true,
      postsynapticFiringObserved: false }] : [], reason: "", postsynapticFiringObserved: false },
  semanticNeuronMapping: false, synapseFiringObserved: false
});

describe("exact router contact protocol", () => {
  it("validates independent contact paging, exact owner IDs and capture time", () => {
    const checked = checkedRouterSpikeActivity(activity(), "brain-a", query);
    expect(checked.contactSnapshot?.capturedAtUnix).toBeLessThan(checked.frames[0]!.observedAtUnix);
    expect(checked.contactSnapshot?.contacts[0]).toMatchObject({ effectiveWeight: 1,
      connectionStrength: 3, effectiveContribution: 3, firingObserved: false });
    expect(checked.transmissionPage?.events[0]).toMatchObject({ sourceInput: 2,
      effectiveContribution: 6, classification: "excitatory",
      postsynapticFiringObserved: false });
    expect(validateRouterSpikeQuery({ enabled: true, captureId: "c", contactOffset: 300,
      contactPageSize: 256 })).toMatchObject({ contactOffset: 300, contactPageSize: 256 });
    expect(() => validateRouterSpikeQuery({ enabled: true, captureId: "c", contactPageSize: 257 })).toThrow();
    expect(() => validateRouterSpikeQuery({ enabled: true, captureId: "c", afterTransmissionSequence: 1 })).toThrow(/selection epoch/);
  });

  it("rejects inferred firing, cross-owner endpoints, malformed layers and false coverage", () => {
    const invalid = [
      contact({ firingObserved: true as false }),
      contact({ sourceId: "brain-a:root.router.population:4" }),
      contact({ sourceChannel: 2, sourceId: "brain-a:root.router.population:2" }),
      contact({ effectiveContribution: 2 }),
      contact({ blockId: "r0_c0_p1", parallelLayer: 0 }),
      contact({ parallelRootId: "brain-a:root.router.synapses:r0_c0_p1:4:3" })
    ];
    for (const value of invalid) expect(() => checkedRouterSpikeActivity(activity([value]), "brain-a", query)).toThrow();
    const wrongPage = activity(); wrongPage.contactSnapshot = { ...wrongPage.contactSnapshot!, offset: 1 };
    expect(() => checkedRouterSpikeActivity(wrongPage, "brain-a", query)).toThrow(/page differs/);
    const unavailable = activity(); unavailable.contactSnapshot = { ...unavailable.contactSnapshot!, available: false,
      observed: false, capturedAtUnix: null, topologyGeneration: null, totalContacts: null, returned: 0,
      hasMore: false, contacts: [] };
    unavailable.transmissionPage = { ...unavailable.transmissionPage!, available: false, enabled: false,
      observed: false, captureEpoch: null, epoch: null, layoutDigest: null, selectedContactCount: 0,
      ringCapacity: 0, lastSequence: 0, firstAvailableSequence: 0, nextSequence: 0,
      events: [] };
    expect(checkedRouterSpikeActivity(unavailable, "brain-a", query).contactSnapshot?.available).toBe(false);
  });

  it("rejects operand fabrication or any implication of postsynaptic firing", () => {
    const wrongContribution = activity(); wrongContribution.transmissionPage = { ...wrongContribution.transmissionPage!,
      events: [{ ...wrongContribution.transmissionPage!.events[0]!, effectiveContribution: 5 }] };
    expect(() => checkedRouterSpikeActivity(wrongContribution, "brain-a", query)).toThrow(/transmission/);
    const firing = activity(); firing.transmissionPage = { ...firing.transmissionPage!,
      events: [{ ...firing.transmissionPage!.events[0]!, postsynapticFiringObserved: true as false }] };
    expect(() => checkedRouterSpikeActivity(firing, "brain-a", query)).toThrow(/transmission/);
    const oldOwner = activity(); oldOwner.transmissionPage = { ...oldOwner.transmissionPage!, captureEpoch: "d".repeat(32) };
    expect(() => checkedRouterSpikeActivity(oldOwner, "brain-a", query)).toThrow(/owner\/layout/);
  });
});
