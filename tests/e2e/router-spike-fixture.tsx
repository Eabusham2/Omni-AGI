import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { RouterSpikeView } from "../../src/renderer/src/RouterSpikeView";
import type { RouterContactRecord, RouterContactSnapshotPage, RouterContactTransmissionPage, RouterSpikeActivity, RouterSpikeQuery } from "../../src/shared/routerActivity";
import "../../src/renderer/src/styles.css";

let sequence = 0, live = false, retired = false, capturedAt = 0;
let viewport = { start: 0, count: 64 }, captureEpoch = "";
let transmissionOffset = -1, transmissionEpoch = "", transmissionBase = 0;
const calls: string[] = [];
function contactPage(brainId: string, offset: number, size: number): RouterContactSnapshotPage {
  const width = Math.min(viewport.count, 8), totalContacts = width > 1 ? 140 : 0;
  const contacts: RouterContactRecord[] = Array.from({ length: Math.max(0, Math.min(size, totalContacts - offset)) }, (_, i) => {
    const n = offset + i, pair = n % (width * width), layer = Math.floor(n / (width * width));
    const sourceChannel = viewport.start + pair % width, targetChannel = viewport.start + Math.floor(pair / width) % width;
    // Every fixture ID is still internally exact; these are synthetic values,
    // not a claim that a live native router emitted or learned them.
    const repeated = layer > 0, blockId = `r0_c0${repeated ? `_p${layer}` : ""}`;
    const effectiveWeight = layer % 2 ? -1 : 1, connectionStrength = repeated ? 2 : 4;
    return { id: `${brainId}:root.router.synapses:${blockId}:${targetChannel}:${sourceChannel}`,
      owner: "root.router.synapses", sourceChannel, targetChannel,
      sourceId: `${brainId}:root.router.population:${sourceChannel}`,
      targetId: `${brainId}:root.router.population:${targetChannel}`, blockId,
      parallelLayer: layer, effectiveWeight, connectionStrength, connectionResistance: 3,
      effectiveContribution: effectiveWeight * connectionStrength, eligibility: 2, stability: 0.5, uses: 7,
      isRepeatedContact: repeated, parallelOrdinal: layer,
      parallelRootId: `${brainId}:root.router.synapses:r0_c0:${targetChannel}:${sourceChannel}`, firingObserved: false };
  });
  return { source: "actual-allocated-router-contacts", owner: "root.router.synapses", available: true, observed: true,
    capturedAtUnix: capturedAt, topologyGeneration: 4, totalContacts, offset, returned: contacts.length,
    pageSize: size, hasMore: offset + contacts.length < totalContacts, contacts, synapseFiringObserved: false };
}
const observe = async (brainId: string, query: RouterSpikeQuery): Promise<RouterSpikeActivity> => {
  const configure = query.epoch === undefined;
  if (configure) { calls.push(`${brainId}:configure`); live = query.enabled; retired = false; sequence = 0;
    viewport = { start: query.start ?? 0, count: Math.min(query.count ?? 64, 1024 - (query.start ?? 0)) };
    capturedAt = Date.now() / 1000; captureEpoch = crypto.randomUUID().replaceAll("-", ""); transmissionOffset = -1; }
  if (!query.enabled) { live = false; calls.push(`${brainId}:stop`); }
  const first = viewport.start, count = viewport.count;
  const enabled = live && !retired, epoch = captureEpoch;
  const contactSnapshot = enabled ? contactPage(brainId, query.contactOffset ?? 0, query.contactPageSize ?? 128) : undefined;
  if (contactSnapshot && transmissionOffset !== contactSnapshot.offset) {
    transmissionOffset = contactSnapshot.offset; transmissionEpoch = crypto.randomUUID().replaceAll("-", ""); transmissionBase = sequence;
  }
  const after = query.afterTransmissionSequence ?? 0, last = (sequence - transmissionBase) * 3;
  const firstTransmission = last > 0 ? Math.max(1, last - 2) : 0;
  const chosen = contactSnapshot ? [contactSnapshot.contacts[0], contactSnapshot.contacts[64] ?? contactSnapshot.contacts[1], contactSnapshot.contacts.at(-1)].filter((c): c is RouterContactRecord => Boolean(c)) : [];
  const events = chosen.flatMap((contact, index) => {
    const eventSequence = last - 2 + index;
    if (!last || eventSequence <= after) return [];
    return [{ sequence: eventSequence, operationSequence: sequence, observedAtUnix: Date.now() / 1000,
      monotonicNs: String(1_000_000_000 + eventSequence), contactId: contact.id, blockId: contact.blockId,
      sourceChannel: contact.sourceChannel, targetChannel: contact.targetChannel,
      sourceId: contact.sourceId, targetId: contact.targetId, sourceInput: 1,
      effectiveWeight: contact.effectiveWeight, connectionStrength: contact.connectionStrength,
      connectionResistance: contact.connectionResistance, effectiveContribution: contact.effectiveContribution,
      classification: contact.effectiveContribution < 0 ? "inhibitory" as const : "excitatory" as const,
      transmissionObserved: true as const, postsynapticFiringObserved: false as const }];
  });
  const transmissionPage: RouterContactTransmissionPage | undefined = enabled ? {
    source: "actual-packed-contact-transmissions", owner: "root.router.synapses", available: true, enabled: true,
    observed: last > 0, captureEpoch, epoch: transmissionEpoch, layoutDigest: "a".repeat(64),
    selectedContactOffset: transmissionOffset, selectedContactCount: contactSnapshot?.returned ?? 0,
    ringCapacity: 3, lastSequence: last, firstAvailableSequence: firstTransmission,
    nextSequence: events.at(-1)?.sequence ?? after, droppedRecords: Math.max(0, last - 3),
    missedRecords: Math.max(0, firstTransmission - after - 1), pageSize: query.transmissionPageSize ?? 128,
    hasMore: false, events, postsynapticFiringObserved: false } : undefined;
  return { brainId, source: "actual-lif-hard-spikes", owner: "root.router.population", channelIdentity: "brainId:root.router.population:channelIndex", available: !retired,
    enabled, observed: enabled && sequence > 0, captureId: query.captureId, epoch, totalChannels: retired ? null : 1024, channelStart: first, admittedCount: enabled ? count : 0, requestedCount: count, ringBytes: enabled ? 2048 : 0, frameCapacity: enabled ? 16 : 0, lastSequence: sequence,
    firstAvailableSequence: sequence ? 1 : 0, nextSequence: sequence, droppedFrames: 0, missedFrames: 0, hasMore: false,
    frames: enabled && sequence > (query.afterSequence ?? 0) ? [{ sequence, observedAtUnix: Date.now() / 1000, monotonicNs: "12345678901234567890", channelStart: first, spikes: Array.from({ length: count }, (_, index) => (index === 2 || index === 5) ? 1 : 0) }] : [],
    ...(contactSnapshot ? { contactSnapshot } : {}), ...(transmissionPage ? { transmissionPage } : {}),
    semanticNeuronMapping: false, synapseFiringObserved: false, ...(retired ? { reason: "Synthetic router epoch retired" } : {}) };
};
function Fixture() {
  const [brainId, setBrainId] = useState("original"), [dark, setDark] = useState(false), [receipt, setReceipt] = useState("");
  const [compact, setCompact] = useState(false);
  const [controlsOnly, setControlsOnly] = useState(false);
  return <main style={{ maxWidth: compact ? 420 : 960, padding: 12, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Router activity UI check</h1><p>Actual viewer, explicitly synthetic binary event receipts. No brain, LIF step, forward or neural load runs.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button className="button" onClick={() => { sequence++; }}>Deliver synthetic spike frame</button>
      <button className="button" onClick={() => { retired = true; }}>Retire synthetic observer</button>
      <button className="button" onClick={() => setBrainId(value => value === "original" ? "copy" : "original")}>Switch fixture owner</button>
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(value => !value)}>Compact content</button>
      <button className="button" onClick={() => setControlsOnly(value => !value)}>Toggle controls-only view</button>
      <button className="button" onClick={() => setReceipt(calls.join("\n") || "No observer calls")}>Inspect synthetic calls</button>
    </div><RouterSpikeView brainId={brainId} observe={observe} showMap={!controlsOnly} /><pre style={{ whiteSpace: "pre-wrap" }}>{receipt}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
