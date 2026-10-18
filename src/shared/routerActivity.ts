/** Exact root LIF channel events, never semantic-assembly or contact inference. */
export interface RouterSpikeQuery {
  enabled: boolean;
  captureId: string;
  start?: number;
  count?: number;
  epoch?: string;
  afterSequence?: number;
  pageSize?: number;
  /** Page through the immutable contact snapshot captured when observation begins. */
  contactOffset?: number;
  contactPageSize?: number;
  /** Cursor for the current exact contact-page transmission selection. */
  transmissionEpoch?: string;
  afterTransmissionSequence?: number;
  transmissionPageSize?: number;
}

export interface RouterSpikeFrame {
  sequence: number;
  observedAtUnix: number;
  /** Integer string preserves a monotonic nanosecond clock beyond JS precision. */
  monotonicNs: string;
  channelStart: number;
  spikes: Array<0 | 1>;
}

export interface RouterContactRecord {
  id: string;
  owner: "root.router.synapses";
  sourceChannel: number;
  targetChannel: number;
  sourceId: string;
  targetId: string;
  blockId: string;
  parallelLayer: number;
  effectiveWeight: -1 | 0 | 1;
  connectionStrength: number;
  connectionResistance: number;
  effectiveContribution: number;
  eligibility: number;
  stability: number;
  uses: number;
  isRepeatedContact: boolean;
  parallelOrdinal: number;
  parallelRootId: string;
  /** The enable-time contact snapshot itself never implies propagation/firing. */
  firingObserved: false;
}

export interface RouterContactSnapshotPage {
  source: "actual-allocated-router-contacts";
  owner: "root.router.synapses";
  available: boolean;
  observed: boolean;
  /** Capture/configuration time, distinct from later hard-spike frame times. */
  capturedAtUnix: number | null;
  topologyGeneration: number | null;
  totalContacts: number | null;
  offset: number;
  returned: number;
  pageSize: number;
  hasMore: boolean;
  contacts: RouterContactRecord[];
  reason?: string;
  synapseFiringObserved: false;
}

export interface RouterContactTransmissionEvent {
  sequence: number;
  operationSequence: number;
  observedAtUnix: number;
  monotonicNs: string;
  contactId: string;
  blockId: string;
  sourceChannel: number;
  targetChannel: number;
  sourceId: string;
  targetId: string;
  /** Actual recurrent input operand at packed computation time. */
  sourceInput: number;
  effectiveWeight: -1 | 0 | 1;
  connectionStrength: number;
  connectionResistance: number;
  effectiveContribution: number;
  classification: "inhibitory" | "excitatory" | "zero";
  transmissionObserved: true;
  /** A contact contribution is not evidence that the target neuron fired. */
  postsynapticFiringObserved: false;
}

export interface RouterContactTransmissionPage {
  source: "actual-packed-contact-transmissions";
  owner: "root.router.synapses";
  available: boolean;
  enabled: boolean;
  observed: boolean;
  captureEpoch: string | null;
  epoch: string | null;
  /** Process-local live-layout custody, never portable model provenance. */
  layoutDigest: string | null;
  selectedContactOffset: number;
  selectedContactCount: number;
  ringCapacity: number;
  lastSequence: number;
  firstAvailableSequence: number;
  nextSequence: number;
  droppedRecords: number;
  missedRecords: number;
  pageSize: number;
  hasMore: boolean;
  events: RouterContactTransmissionEvent[];
  reason?: string;
  postsynapticFiringObserved: false;
}

export interface RouterSpikeActivity {
  brainId: string;
  source: "actual-lif-hard-spikes";
  owner: "root.router.population";
  channelIdentity: "brainId:root.router.population:channelIndex";
  available: boolean;
  enabled: boolean;
  observed: boolean;
  captureId: string;
  epoch: string | null;
  totalChannels: number | null;
  channelStart: number;
  admittedCount: number;
  requestedCount: number;
  ringBytes: number;
  frameCapacity: number;
  lastSequence: number;
  firstAvailableSequence: number;
  nextSequence: number;
  droppedFrames: number;
  missedFrames: number;
  hasMore: boolean;
  frames: RouterSpikeFrame[];
  /** Optional for old saved UI fixtures; current native capture always emits it. */
  contactSnapshot?: RouterContactSnapshotPage;
  /** Optional for old fixtures; current native observation always emits it. */
  transmissionPage?: RouterContactTransmissionPage;
  reason?: string;
  semanticNeuronMapping: false;
  synapseFiringObserved: false;
}
