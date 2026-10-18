import { useState } from "react";
import type { RouterContactRecord, RouterSpikeActivity } from "../../shared/routerActivity";
import { BrainMapSubstrateGraph, type BrainMapGraphEdge, type BrainMapGraphNode } from "./BrainMapSubstrateGraph";
import "./routerActivityBrainMap.css";

export function routerChannelIdentity(brainId: string, index: number): string { return `${brainId}:root.router.population:${index}`; }

function contactEdge(contact: RouterContactRecord, brainId: string, start: number, end: number): BrainMapGraphEdge {
  const sourceId = routerChannelIdentity(brainId, contact.sourceChannel), targetId = routerChannelIdentity(brainId, contact.targetChannel);
  const id = `${brainId}:root.router.synapses:${contact.blockId}:${contact.targetChannel}:${contact.sourceChannel}`;
  const block = /^r(?:0|[1-9][0-9]*)_c(?:0|[1-9][0-9]*)(?:_p([1-9][0-9]*))?$/.exec(contact.blockId);
  const layer = Number(block?.[1] ?? 0), baseBlock = contact.blockId.replace(/_p[1-9][0-9]*$/, "");
  const rootId = `${brainId}:root.router.synapses:${baseBlock}:${contact.targetChannel}:${contact.sourceChannel}`;
  if (contact.owner !== "root.router.synapses" || contact.sourceId !== sourceId || contact.targetId !== targetId || contact.id !== id ||
      !block || contact.sourceChannel < start || contact.sourceChannel >= end || contact.targetChannel < start || contact.targetChannel >= end ||
      contact.parallelLayer !== layer || contact.parallelOrdinal !== layer || contact.isRepeatedContact !== (layer > 0) || contact.parallelRootId !== rootId ||
      ![-1, 0, 1].includes(contact.effectiveWeight) || !Number.isSafeInteger(contact.connectionStrength) ||
      contact.connectionStrength < 1 || contact.connectionStrength > 16 || contact.firingObserved !== false ||
      contact.effectiveContribution !== contact.effectiveWeight * contact.connectionStrength) {
    throw new Error("Router contact presentation lost its exact owner/channel identity.");
  }
  return { id: contact.id, sourceId, targetId, effectiveWeight: contact.effectiveWeight,
    connectionStrength: contact.connectionStrength, effectiveContribution: contact.effectiveContribution,
    signedStrength: contact.effectiveContribution, stability: contact.stability, eligibility: contact.eligibility,
    uses: contact.uses, pathways: 1, isRepeatedContact: contact.isRepeatedContact,
    parallelOrdinal: contact.parallelOrdinal, parallelRootId: contact.parallelRootId,
    nativeEndpoints: { sourceId, targetId } };
}

/** Exact router owner projection. Semantic substrate UUIDs are never joined. */
export function routerActivityGraph(activity: RouterSpikeActivity): { nodes: BrainMapGraphNode[]; edges: BrainMapGraphEdge[] } {
  if (activity.owner !== "root.router.population" || activity.channelIdentity !== "brainId:root.router.population:channelIndex" ||
      activity.semanticNeuronMapping !== false || activity.synapseFiringObserved !== false) throw new Error("Router graph requires exact native channel ownership.");
  const firing = new Map<number, { count: number; observedAt: string }>();
  for (const frame of activity.frames) frame.spikes.forEach((spike, index) => {
    if (spike !== 1) return; const channel = frame.channelStart + index, observedAt = new Date(frame.observedAtUnix * 1000).toISOString();
    const prior = firing.get(channel); firing.set(channel, { count: (prior?.count ?? 0) + 1, observedAt });
  });
  const nodes: BrainMapGraphNode[] = Array.from({ length: activity.admittedCount }, (_, offset) => {
    const channel = activity.channelStart + offset, event = firing.get(channel), id = routerChannelIdentity(activity.brainId, channel);
    return { id, label: `Router channel ${channel.toLocaleString()}`, region: "root.router.population · exact LIF channels",
      activation: 0, activationObserved: false, importance: 0, members: [id],
      ...(event ? { firingEvent: { source: "recorded-spike-event" as const, observedAt: event.observedAt, count: event.count },
        lastActivatedAt: event.observedAt } : {}) };
  });
  const snapshot = activity.contactSnapshot;
  if (!snapshot?.available) return { nodes, edges: [] };
  if (snapshot.source !== "actual-allocated-router-contacts" || snapshot.owner !== "root.router.synapses" || snapshot.synapseFiringObserved !== false)
    throw new Error("Router graph contact snapshot source is invalid.");
  const transmissions = new Map<string, NonNullable<BrainMapGraphEdge["transmissionEvent"]>>();
  const page = activity.transmissionPage;
  if (page?.available) {
    if (page.source !== "actual-packed-contact-transmissions" || page.owner !== "root.router.synapses" ||
        page.postsynapticFiringObserved !== false || page.captureEpoch !== activity.epoch)
      throw new Error("Router graph transmission source is invalid.");
    for (const event of page.events) {
      if (event.transmissionObserved !== true || event.postsynapticFiringObserved !== false || !Number.isFinite(event.effectiveContribution) ||
          event.classification !== (event.effectiveContribution < 0 ? "inhibitory" : event.effectiveContribution > 0 ? "excitatory" : "zero"))
        throw new Error("Router graph transmission operands are invalid.");
      const prior = transmissions.get(event.contactId);
      transmissions.set(event.contactId, { source: "recorded-contact-transmission",
        observedAt: new Date(event.observedAtUnix * 1000).toISOString(), count: (prior?.count ?? 0) + 1,
        contribution: event.effectiveContribution, classification: event.classification });
    }
  }
  const end = activity.channelStart + activity.admittedCount;
  const edges = snapshot.contacts.map(contact => {
    const edge = contactEdge(contact, activity.brainId, activity.channelStart, end);
    const transmissionEvent = transmissions.get(contact.id);
    return transmissionEvent ? { ...edge, transmissionEvent } : edge;
  });
  if ([...transmissions].some(([id]) => !snapshot.contacts.some(contact => contact.id === id)))
    throw new Error("Router graph transmission is outside the selected contact page.");
  return { nodes, edges };
}

export function RouterActivityBrainMap({ activity, onContactOffset }: {
  activity: RouterSpikeActivity; onContactOffset?(offset: number): void;
}) {
  const graph = routerActivityGraph(activity), [selected, setSelected] = useState(graph.nodes[0]?.id ?? "");
  const contacts = activity.contactSnapshot;
  const transmissions = activity.transmissionPage;
  return <section className="router-activity-map" aria-label="Recorded router firing map">
    <div className="router-activity-map__heading"><div><h3>Recorded router firing</h3>
      <p>Exact <code>root.router.population</code> channels and allocated recurrent contacts. No token or semantic-neuron mapping.</p></div>
      <span>{graph.nodes.length.toLocaleString()} channels · {contacts?.totalContacts?.toLocaleString() ?? "unknown"} captured contacts · {transmissions?.events.length.toLocaleString() ?? "0"} last recorded transmissions</span></div>
    {graph.nodes.length ? <BrainMapSubstrateGraph nodes={graph.nodes} edges={graph.edges} selectedId={selected}
      onSelect={node => setSelected(node.id)} /> : <p>No admitted router channels.</p>}
    <p className="router-activity-map__scope">Hard-spike node highlights use recorded LIF frames. Contact weights were captured at {contacts?.capturedAtUnix
      ? new Date(contacts.capturedAtUnix * 1000).toLocaleString() : "an unavailable time"}; highlighted contact transmissions use later packed recurrent operands before STDP. Postsynaptic/path firing remains unobserved.</p>
    {transmissions?.reason ? <p>{transmissions.reason}</p> : null}
    {contacts?.reason ? <p>{contacts.reason}</p> : null}
    {contacts?.available ? <div className="router-activity-map__paging">
      <span>{contacts.returned ? `${contacts.offset + 1}–${contacts.offset + contacts.returned}` : "0"} of {contacts.totalContacts?.toLocaleString() ?? "unknown"} contacts</span>
      <button type="button" className="button" disabled={!onContactOffset || contacts.offset === 0}
        onClick={() => onContactOffset?.(Math.max(0, contacts.offset - contacts.pageSize))}>Previous contacts</button>
      <button type="button" className="button" disabled={!onContactOffset || !contacts.hasMore}
        onClick={() => onContactOffset?.(contacts.offset + contacts.returned)}>Next contacts</button>
    </div> : null}
  </section>;
}
