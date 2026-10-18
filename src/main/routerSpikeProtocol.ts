import type { RouterSpikeActivity, RouterSpikeQuery } from "../shared/routerActivity";

const record = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const epoch = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{32}$/.test(value);

export function validateRouterSpikeQuery(query: RouterSpikeQuery): RouterSpikeQuery {
  if (!record(query) || typeof query.enabled !== "boolean" || typeof query.captureId !== "string" || !query.captureId || query.captureId.length > 128)
    throw new Error("Invalid router spike capture ownership.");
  for (const field of ["start", "count", "afterSequence", "pageSize", "contactOffset", "contactPageSize", "afterTransmissionSequence", "transmissionPageSize"] as const) {
    const value = query[field];
    if (value !== undefined && (!count(value) || (["count", "pageSize", "contactPageSize", "transmissionPageSize"] as string[]).includes(field) && value < 1 ||
      field === "contactPageSize" && value > 256 || field === "transmissionPageSize" && value > 128)) throw new Error("Invalid router spike viewport or sequence.");
  }
  if (query.epoch !== undefined && !epoch(query.epoch) || (query.afterSequence ?? 0) > 0 && query.epoch === undefined)
    throw new Error("Router spike continuation requires its exact capture epoch.");
  if (query.transmissionEpoch !== undefined && !epoch(query.transmissionEpoch) ||
      (query.afterTransmissionSequence ?? 0) > 0 && query.transmissionEpoch === undefined)
    throw new Error("Router transmission continuation requires its exact selection epoch.");
  return { enabled: query.enabled, captureId: query.captureId, start: query.start, count: query.count,
    epoch: query.epoch, afterSequence: query.afterSequence, pageSize: query.pageSize,
    contactOffset: query.contactOffset, contactPageSize: query.contactPageSize,
    transmissionEpoch: query.transmissionEpoch,
    afterTransmissionSequence: query.afterTransmissionSequence,
    transmissionPageSize: query.transmissionPageSize };
}

function checkedContactSnapshot(value: unknown, brainId: string, query: RouterSpikeQuery, channelStart: number,
  admittedCount: number, totalChannels: number | null): void {
  if (value === undefined) return; // Old renderer fixtures; current native source always sends this field.
  if (!record(value) || value.source !== "actual-allocated-router-contacts" || value.owner !== "root.router.synapses" ||
      typeof value.available !== "boolean" || typeof value.observed !== "boolean" || value.synapseFiringObserved !== false ||
      !count(value.offset) || !count(value.returned) || !count(value.pageSize) || value.pageSize < 1 || value.pageSize > 256 ||
      typeof value.hasMore !== "boolean" || !Array.isArray(value.contacts) || value.contacts.length !== value.returned || value.contacts.length > value.pageSize ||
      value.totalContacts !== null && !count(value.totalContacts) || value.topologyGeneration !== null && !count(value.topologyGeneration) ||
      value.capturedAtUnix !== null && (typeof value.capturedAtUnix !== "number" || !Number.isFinite(value.capturedAtUnix) || value.capturedAtUnix <= 0) ||
      value.reason !== undefined && typeof value.reason !== "string") throw new Error("Invalid exact router contact snapshot.");
  if (value.offset !== (query.contactOffset ?? 0) || value.pageSize !== (query.contactPageSize ?? 128)) throw new Error("Router contact page differs from its query.");
  if (!value.available) {
    if (value.observed || value.contacts.length || value.totalContacts !== null || value.capturedAtUnix !== null || value.topologyGeneration !== null || value.hasMore)
      throw new Error("Unavailable router contacts cannot claim observed topology.");
    return;
  }
  const contactOffset = value.offset as number, returned = value.returned as number;
  const totalContacts = value.totalContacts as number;
  if (!value.observed || value.totalContacts === null || value.capturedAtUnix === null || value.topologyGeneration === null ||
      contactOffset + returned > totalContacts || value.hasMore !== (contactOffset + returned < totalContacts))
    throw new Error("Router contact snapshot coverage is invalid.");
  const end = channelStart + admittedCount;
  for (const raw of value.contacts) {
    if (!record(raw)) throw new Error("Invalid exact router contact record.");
    const blockId = raw.blockId;
    const blockMatch = typeof blockId === "string"
      ? /^r(?:0|[1-9][0-9]*)_c(?:0|[1-9][0-9]*)(?:_p([1-9][0-9]*))?$/.exec(blockId)
      : null;
    const weight = raw.effectiveWeight, strength = raw.connectionStrength;
    if (raw.owner !== "root.router.synapses" || raw.firingObserved !== false || !count(raw.sourceChannel) || !count(raw.targetChannel) ||
        raw.sourceChannel < channelStart || raw.sourceChannel >= end || raw.targetChannel < channelStart || raw.targetChannel >= end ||
        totalChannels !== null && (raw.sourceChannel >= totalChannels || raw.targetChannel >= totalChannels) ||
        !blockMatch || !count(raw.parallelLayer) || raw.parallelLayer !== Number(blockMatch[1] ?? 0) ||
        raw.isRepeatedContact !== (raw.parallelLayer > 0) || raw.parallelOrdinal !== raw.parallelLayer ||
        typeof weight !== "number" || ![-1, 0, 1].includes(weight) || !count(strength) || strength < 1 || strength > 16 ||
        !count(raw.connectionResistance) || raw.connectionResistance > 15 || raw.effectiveContribution !== weight * strength ||
        typeof raw.eligibility !== "number" || !Number.isFinite(raw.eligibility) || typeof raw.stability !== "number" || !Number.isFinite(raw.stability) ||
        typeof raw.uses !== "number" || !Number.isFinite(raw.uses) || raw.uses < 0) throw new Error("Invalid exact router contact record.");
    const sourceId = `${brainId}:root.router.population:${raw.sourceChannel}`, targetId = `${brainId}:root.router.population:${raw.targetChannel}`;
    const id = `${brainId}:root.router.synapses:${blockId}:${raw.targetChannel}:${raw.sourceChannel}`;
    const baseBlock = (blockId as string).replace(/_p[1-9][0-9]*$/, ""), rootId = `${brainId}:root.router.synapses:${baseBlock}:${raw.targetChannel}:${raw.sourceChannel}`;
    if (raw.sourceId !== sourceId || raw.targetId !== targetId || raw.id !== id || raw.parallelRootId !== rootId) throw new Error("Router contact owner/channel identity differs.");
  }
}

function checkedTransmissionPage(value: unknown, brainId: string, query: RouterSpikeQuery,
  captureEpoch: unknown, contacts: unknown): void {
  if (value === undefined) return; // Old fixtures only; current native source always emits this field.
  if (!record(value) || value.source !== "actual-packed-contact-transmissions" || value.owner !== "root.router.synapses" ||
      typeof value.available !== "boolean" || typeof value.enabled !== "boolean" || typeof value.observed !== "boolean" ||
      value.postsynapticFiringObserved !== false || !count(value.selectedContactOffset) || !count(value.selectedContactCount) ||
      !count(value.ringCapacity) || !count(value.lastSequence) || !count(value.firstAvailableSequence) || !count(value.nextSequence) ||
      !count(value.droppedRecords) || !count(value.missedRecords) || !count(value.pageSize) || value.pageSize < 1 || value.pageSize > 128 ||
      typeof value.hasMore !== "boolean" || !Array.isArray(value.events) || value.events.length > value.pageSize ||
      value.reason !== undefined && typeof value.reason !== "string") throw new Error("Invalid exact router transmission page.");
  if (value.pageSize !== (query.transmissionPageSize ?? 128) || value.selectedContactOffset !== (query.contactOffset ?? 0))
    throw new Error("Router transmission page differs from its query.");
  if (!value.available) {
    if (value.enabled || value.observed || value.captureEpoch !== null && (!epoch(value.captureEpoch) || value.captureEpoch !== captureEpoch) ||
        value.epoch !== null || value.layoutDigest !== null ||
        value.selectedContactCount !== 0 || value.ringCapacity !== 0 || value.lastSequence !== 0 || value.firstAvailableSequence !== 0 ||
        value.nextSequence !== 0 || value.droppedRecords !== 0 || value.missedRecords !== 0 || value.hasMore || value.events.length)
      throw new Error("Unavailable router transmission cannot claim observed operands.");
    return;
  }
  if (!epoch(value.captureEpoch) || value.captureEpoch !== captureEpoch || !epoch(value.epoch) ||
      typeof value.layoutDigest !== "string" || !/^[a-f0-9]{64}$/.test(value.layoutDigest) || value.ringCapacity < 1)
    throw new Error("Router transmission owner/layout identity differs.");
  if (!record(contacts) || contacts.available !== true || !Array.isArray(contacts.contacts) ||
      value.selectedContactCount !== contacts.contacts.length)
    throw new Error("Router transmission selection differs from its contact page.");
  const byId = new Map<string, Record<string, unknown>>();
  for (const item of contacts.contacts) if (record(item) && typeof item.id === "string") byId.set(item.id, item);
  const returnedEpoch = value.epoch as string;
  const requestedAfter = query.transmissionEpoch === returnedEpoch ? query.afterTransmissionSequence ?? 0 : 0;
  let previous = requestedAfter;
  let previousOperation = 0;
  for (const event of value.events) {
    if (!record(event) || !count(event.sequence) || event.sequence <= previous || event.sequence > value.lastSequence ||
        !count(event.operationSequence) || event.operationSequence < 1 || event.operationSequence < previousOperation ||
        typeof event.observedAtUnix !== "number" || !Number.isFinite(event.observedAtUnix) || event.observedAtUnix <= 0 ||
        typeof event.monotonicNs !== "string" || !/^[0-9]{1,24}$/.test(event.monotonicNs) ||
        event.transmissionObserved !== true || event.postsynapticFiringObserved !== false ||
        typeof event.contactId !== "string" || typeof event.blockId !== "string" || !count(event.sourceChannel) || !count(event.targetChannel) ||
        typeof event.sourceInput !== "number" || !Number.isFinite(event.sourceInput) ||
        typeof event.effectiveWeight !== "number" || ![-1, 0, 1].includes(event.effectiveWeight) ||
        !count(event.connectionStrength) || event.connectionStrength < 1 || event.connectionStrength > 16 ||
        !count(event.connectionResistance) || event.connectionResistance > 15 ||
        typeof event.effectiveContribution !== "number" || !Number.isFinite(event.effectiveContribution) ||
        event.effectiveContribution !== event.sourceInput * event.effectiveWeight * event.connectionStrength ||
        event.classification !== (event.effectiveContribution < 0 ? "inhibitory" : event.effectiveContribution > 0 ? "excitatory" : "zero"))
      throw new Error("Invalid observed router contact transmission.");
    const contact = byId.get(event.contactId);
    if (!contact || event.blockId !== contact.blockId || event.sourceChannel !== contact.sourceChannel ||
        event.targetChannel !== contact.targetChannel || event.sourceId !== contact.sourceId || event.targetId !== contact.targetId)
      throw new Error("Router transmission contact identity differs from its selected snapshot.");
    previous = event.sequence;
    previousOperation = event.operationSequence;
  }
  const last = value.lastSequence as number, capacity = value.ringCapacity as number;
  const first = last ? Math.max(1, last-capacity+1) : 0;
  if (value.firstAvailableSequence !== first || value.droppedRecords !== Math.max(0, last-capacity) ||
      value.missedRecords !== Math.max(0, first-requestedAfter-1) ||
      value.nextSequence !== previous || value.observed !== (last > 0) || value.hasMore !== (previous < last))
    throw new Error("Router transmission cursor accounting diverged.");
}

export function checkedRouterSpikeActivity(value: unknown, brainId: string, query: RouterSpikeQuery): RouterSpikeActivity {
  if (!record(value) || value.brainId !== brainId || value.captureId !== query.captureId || value.source !== "actual-lif-hard-spikes" ||
      value.owner !== "root.router.population" || value.channelIdentity !== "brainId:root.router.population:channelIndex" ||
      value.semanticNeuronMapping !== false || value.synapseFiringObserved !== false ||
      typeof value.available !== "boolean" || typeof value.enabled !== "boolean" || typeof value.observed !== "boolean" ||
      value.epoch !== null && !epoch(value.epoch) || value.totalChannels !== null && !count(value.totalChannels) ||
      !Array.isArray(value.frames) || value.frames.length > 32) throw new Error("Invalid owned native router spike receipt.");
  for (const field of ["channelStart", "admittedCount", "requestedCount", "ringBytes", "frameCapacity", "lastSequence", "firstAvailableSequence", "nextSequence", "droppedFrames", "missedFrames"])
    if (!count(value[field])) throw new Error("Invalid native router spike accounting.");
  const admitted = value.admittedCount as number;
  if (admitted > 256 || (value.enabled && (!value.available || !epoch(value.epoch))) ||
      query.epoch !== undefined && value.available && value.epoch !== query.epoch ||
      value.totalChannels !== null && (value.channelStart as number) + admitted > (value.totalChannels as number) ||
      typeof value.hasMore !== "boolean" || value.reason !== undefined && typeof value.reason !== "string")
    throw new Error("Native router spike epoch or channel range differs.");
  if (!value.available) {
    if (value.enabled || value.observed || value.frames.length) throw new Error("Unavailable router activity cannot claim observed frames.");
    checkedContactSnapshot(value.contactSnapshot, brainId, query, value.channelStart as number,
      admitted, value.totalChannels as number | null);
    checkedTransmissionPage(value.transmissionPage, brainId, query, value.epoch,
      value.contactSnapshot);
    return value as unknown as RouterSpikeActivity;
  }
  let previous = query.afterSequence ?? 0;
  for (const frame of value.frames) {
    if (!record(frame) || !count(frame.sequence) || frame.sequence <= previous || frame.sequence > (value.lastSequence as number) ||
        frame.channelStart !== value.channelStart || typeof frame.observedAtUnix !== "number" || !Number.isFinite(frame.observedAtUnix) || frame.observedAtUnix <= 0 ||
        typeof frame.monotonicNs !== "string" || !/^[0-9]{1,24}$/.test(frame.monotonicNs) || !Array.isArray(frame.spikes) || frame.spikes.length !== admitted ||
        frame.spikes.some((spike) => spike !== 0 && spike !== 1)) throw new Error("Invalid exact LIF event frame.");
    previous = frame.sequence;
  }
  if (value.nextSequence !== previous || value.hasMore !== (value.enabled && previous < (value.lastSequence as number)))
    throw new Error("Native router spike continuation diverged.");
  checkedContactSnapshot(value.contactSnapshot, brainId, query, value.channelStart as number, admitted,
    value.totalChannels as number | null);
  checkedTransmissionPage(value.transmissionPage, brainId, query, value.epoch,
    value.contactSnapshot);
  return value as unknown as RouterSpikeActivity;
}
