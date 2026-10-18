import type { DeviceResourceBudget, DeviceResourceTopology } from "../shared/deviceResources";
import { retainedDeviceStorageSupported } from "../shared/deviceResources";
import type { SavedTensorShape } from "./nativeCheckpointMemory";

/** Complete direct packed-owner buffers from authenticated safetensor headers. */
export function checkpointPackedOwnerMinimum(tensors: ReadonlyMap<string, SavedTensorShape>): number | undefined {
  const prefixes = new Set([...tensors.keys()].filter(name => /\._packed_forward_weight$/.test(name))
    .map(name => name.slice(0, -"_packed_forward_weight".length)));
  const owners = new Map([...prefixes].map(prefix => [prefix, 4])); // Actual nonpersistent packed autograd trigger.
  for (const [name, tensor] of tensors) {
    const prefix = name.slice(0, name.lastIndexOf(".") + 1);
    if (owners.has(prefix)) owners.set(prefix, owners.get(prefix)! + tensor.bytes);
  }
  let maximum = 0;
  for (const bytes of owners.values()) {
    if (!Number.isSafeInteger(bytes)) throw new Error("Packed owner minimum exceeds exact accounting.");
    maximum = Math.max(maximum, bytes);
  }
  return maximum || undefined;
}

/** Exact header bytes outside packed owners/tiled recurrent matrix backing. */
export function checkpointNonpackedControls(tensors: ReadonlyMap<string, SavedTensorShape>): number {
  const prefixes = new Set([...tensors.keys()].filter(name => /\._packed_forward_weight$/.test(name))
    .map(name => name.slice(0, -"_packed_forward_weight".length)));
  let bytes = 0;
  for (const [name, tensor] of tensors) {
    if (prefixes.has(name.slice(0, name.lastIndexOf(".") + 1))) continue;
    if (/(?:^|\.)synapses(?:\.blocks\.[^.]+)?\.(?:_packed_weights|_connection_state|eligibility_accumulator|stability|uses)$/.test(name)) continue;
    bytes += tensor.bytes;
  }
  if (!Number.isSafeInteger(bytes)) throw new Error("Nonpacked control inventory exceeds exact accounting.");
  return bytes;
}

export interface RetainedContextPool {
  deviceId: string; executionDevice: string; memoryGroupId: string; budgetBytes: number;
  selectedCeilingBytes: number; effectiveCopyBytesPerSecond: number | null; copyTimingMeasured: boolean;
  partitionMode?: "manual-admitted-minimums" | "manual-conservative-missing-minimum";
  minimumPackedOwnerBytes?: number; controlReserveBytes?: number; transferReserveBytes?: number;
  pageBytes?: number;
}
export interface RetainedContextMinimum { packedOwnerBytes: number; controlBytes: number; transferBytes: number }
const count = (value: unknown): value is number => Number.isSafeInteger(value) && Number(value) >= 0;
const rate = (value: { bytesPerSecond?: number } | undefined): number | null =>
  typeof value?.bytesPerSecond === "number" && Number.isFinite(value.bytesPerSecond) && value.bytesPerSecond > 0 ? value.bytesPerSecond : null;

/** Same current-capacity contract as the native pager; never the broad diagnostic projection. */
export function retainedContextPools(topology: DeviceResourceTopology, budgets: DeviceResourceBudget[], options: {
  manual?: boolean; minimums?: Record<string, RetainedContextMinimum>; pageBytes?: number;
} = {}): RetainedContextPool[] {
  const ramGroups = new Set(["ram:system", ...topology.devices.filter(item => item.kind === "ram").map(item => item.memoryGroupId)]);
  const selected = new Map(budgets.map(item => [item.deviceId, item]));
  const links = new Map(topology.links.map(item => [`${item.from}>${item.to}`, rate(item.measurement)]));
  const groups = new Map<string, RetainedContextPool>();
  for (const device of topology.devices) {
    const budget = selected.get(device.id), execution = device.executionDevice;
    if (!retainedDeviceStorageSupported(device) || ramGroups.has(device.memoryGroupId)
      || !budget?.enabled || budget.memoryGroupId !== device.memoryGroupId || !count(device.totalBytes) || !device.totalBytes
      || !count(device.availableBytes) || !count(budget.ceilingBytes)) continue;
    const ceiling = Math.min(device.totalBytes, budget.ceilingBytes);
    const local = rate(device.measurement), upload = links.get(`ram:system>${device.id}`), download = links.get(`${device.id}>ram:system`);
    const effective = local && upload && download ? 1 / (1 / local + 1 / upload + 1 / download) : null;
    const pool: RetainedContextPool = { deviceId: device.id, executionDevice: execution!, memoryGroupId: device.memoryGroupId,
      budgetBytes: Math.floor(ceiling * 30 / 100), selectedCeilingBytes: ceiling,
      effectiveCopyBytesPerSecond: effective, copyTimingMeasured: effective !== null };
    if (options.manual) {
      const minimum = options.minimums?.[execution!];
      if (minimum && Object.keys(minimum).length === 3 && [minimum.packedOwnerBytes, minimum.controlBytes, minimum.transferBytes].every(count)) {
        pool.budgetBytes = Math.max(0, ceiling - minimum.packedOwnerBytes - minimum.controlBytes - minimum.transferBytes);
        Object.assign(pool, { partitionMode: "manual-admitted-minimums", minimumPackedOwnerBytes: minimum.packedOwnerBytes,
          controlReserveBytes: minimum.controlBytes, transferReserveBytes: minimum.transferBytes });
      } else pool.partitionMode = "manual-conservative-missing-minimum";
    }
    const pageBytes = options.pageBytes ?? 1;
    if (!count(pageBytes) || pageBytes < 1) throw new Error("Retained context requires an exact positive page granularity.");
    pool.budgetBytes = Math.floor(pool.budgetBytes / pageBytes) * pageBytes;
    if (pageBytes !== 1) pool.pageBytes = pageBytes;
    const previous = groups.get(device.memoryGroupId);
    if (!previous || pool.budgetBytes > previous.budgetBytes || pool.budgetBytes === previous.budgetBytes
      && ((effective ?? 0) > (previous.effectiveCopyBytesPerSecond ?? 0) || effective === previous.effectiveCopyBytesPerSecond
        && execution! > previous.executionDevice)) groups.set(device.memoryGroupId, pool);
  }
  return [...groups.values()].sort((a, b) => (b.effectiveCopyBytesPerSecond ?? 0) - (a.effectiveCopyBytesPerSecond ?? 0)
    || (a.deviceId < b.deviceId ? -1 : a.deviceId > b.deviceId ? 1 : 0));
}

export function contextSnapshotBytes(tokens: number, geometry: { kvBytesPerToken: number; indexBytesPerToken: number; pageTokens: number; layers: number }): number {
  const bytes = tokens * (geometry.kvBytesPerToken + geometry.indexBytesPerToken)
    + Math.ceil(tokens / geometry.pageTokens) * 2 * geometry.layers * 65536;
  if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Context checkpoint exceeds exact byte accounting.");
  return bytes;
}

export interface ContextCapacityInput {
  ramWorkingBytes: number; deviceContextBytes: number; kvBytesPerToken: number; indexBytesPerToken: number;
  modelLimitTokens: number; spillCapacityBytes: number; pagedKvBytesPerToken: number; checkpointCapacityBytes: number;
  pageTokens: number; layers: number;
  sharedStorageCapacityBytes?: number;
}
export function contextColdSpillBytes(tokens: number, input: Pick<ContextCapacityInput,
  "ramWorkingBytes" | "deviceContextBytes" | "kvBytesPerToken" | "indexBytesPerToken" | "pagedKvBytesPerToken" | "pageTokens">): number {
  const hot = Math.max(0, input.ramWorkingBytes - tokens * input.indexBytesPerToken) + input.deviceContextBytes;
  const cold = Math.max(0, tokens * input.kvBytesPerToken - hot);
  return Math.ceil(cold / Math.max(1, input.kvBytesPerToken * input.pageTokens)) * input.pageTokens * input.pagedKvBytesPerToken;
}
export function contextCapacity(input: ContextCapacityInput): {
  residentMaximumTokens: number; storageMaximumTokens: number; maximumTokens: number; metadataMaximumTokens: number;
} {
  if (Object.values(input).some(value => !count(value)) || input.pageTokens < 1 || input.layers < 1)
    throw new Error("Context capacity requires nonnegative exact geometry and budgets.");
  const metadata = Math.floor(input.ramWorkingBytes / Math.max(1, input.indexBytesPerToken));
  const resident = Math.min(input.modelLimitTokens, metadata,
    Math.floor((input.ramWorkingBytes + input.deviceContextBytes) / Math.max(1, input.kvBytesPerToken + input.indexBytesPerToken)));
  let low = 0, high = Math.min(input.modelLimitTokens, metadata,
    Math.floor(input.checkpointCapacityBytes / Math.max(1, input.kvBytesPerToken + input.indexBytesPerToken)));
  while (low < high) {
    const middle = low + Math.ceil((high - low) / 2);
    const snapshot = contextSnapshotBytes(middle, input), cold = contextColdSpillBytes(middle, input);
    if (snapshot <= input.checkpointCapacityBytes && cold <= input.spillCapacityBytes
      && snapshot + cold <= (input.sharedStorageCapacityBytes ?? input.checkpointCapacityBytes)) low = middle;
    else high = middle - 1;
  }
  return { residentMaximumTokens: Math.min(resident, low), storageMaximumTokens: input.spillCapacityBytes ? low : 0,
    maximumTokens: low, metadataMaximumTokens: metadata };
}
