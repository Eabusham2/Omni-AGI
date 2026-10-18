import type {
  DeviceResourceAllocation, DeviceResourceBudget, DeviceResourceDemand, DeviceResourcePlan,
  DeviceResourceSettings, DeviceResourceTopology, RegionalExecutionSettings, ResourceDevice, ResourceRateMeasurement
} from "../shared/deviceResources";

const ID = /^[A-Za-z0-9][A-Za-z0-9:._-]{0,255}$/;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const count = (value: unknown): value is number => typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
const nullableCount = (value: unknown) => value === null || count(value);

export function validateRegionalExecutionSettings(value: unknown): RegionalExecutionSettings {
  if (!record(value) || value.version !== 1 || typeof value.enabled !== "boolean" || value.mode !== "manual" || !Array.isArray(value.routes)
    || Object.keys(value).some(key => !["version", "enabled", "mode", "routes"].includes(key))) throw new Error("Invalid manual regional execution settings.");
  const seen = new Set<string>();
  const routes = value.routes.map(route => {
    if (!record(route) || typeof route.regionId !== "string" || !route.regionId || route.regionId.includes("\0") || seen.has(route.regionId)
      || typeof route.deviceId !== "string" || !ID.test(route.deviceId) || !["native", "packed-cpu-tile-fallback"].includes(String(route.executionMode))
      || Object.keys(route).some(key => !["regionId", "deviceId", "executionMode", "acknowledgePackedCpuFallback"].includes(key))) throw new Error("Invalid or duplicated regional route.");
    seen.add(route.regionId);
    if (route.executionMode === "packed-cpu-tile-fallback" ? route.acknowledgePackedCpuFallback !== true : "acknowledgePackedCpuFallback" in route) {
      throw new Error("Regional fallback requires its own explicit acknowledgment; native routes must not carry it.");
    }
    return { regionId: route.regionId, deviceId: route.deviceId, executionMode: route.executionMode as "native" | "packed-cpu-tile-fallback",
      ...(route.acknowledgePackedCpuFallback === true ? { acknowledgePackedCpuFallback: true as const } : {}) };
  });
  return { version: 1, enabled: value.enabled, mode: "manual", routes };
}

export function validateDeviceResourceSettings(value: unknown): DeviceResourceSettings {
  if (!record(value) || value.version !== 1 || !Array.isArray(value.allocations) || !Array.isArray(value.storageDirectories)
    || typeof value.allowStorageSpill !== "boolean" || typeof value.autoIncreaseStorage !== "boolean"
    || Object.keys(value).some((key) => !["version", "allocations", "storageDirectories", "allowStorageSpill", "autoIncreaseStorage", "regionalExecution"].includes(key))) throw new Error("Invalid device resource settings.");
  const seen = new Set<string>();
  const allocations: DeviceResourceAllocation[] = value.allocations.map((entry) => {
    if (!record(entry) || typeof entry.deviceId !== "string" || !ID.test(entry.deviceId) || seen.has(entry.deviceId)
      || typeof entry.enabled !== "boolean" || typeof entry.autoIncrease !== "boolean" || !["auto", "manual"].includes(String(entry.mode))
      || typeof entry.percent !== "number" || !Number.isFinite(entry.percent) || entry.percent < (entry.deviceId === "ram:system" ? 30 : 1) || entry.percent > 100
      || Object.keys(entry).some((key) => !["deviceId", "enabled", "mode", "percent", "autoIncrease"].includes(key))) throw new Error("Invalid or duplicated device allocation.");
    seen.add(entry.deviceId);
    return { deviceId: entry.deviceId, enabled: entry.enabled, mode: entry.mode as "auto" | "manual", percent: entry.percent, autoIncrease: entry.autoIncrease };
  });
  const storageDirectories = value.storageDirectories.map((path) => {
    if (typeof path !== "string" || !path || path.includes("\0")) throw new Error("Invalid selected storage directory.");
    return path;
  });
  if (new Set(storageDirectories).size !== storageDirectories.length) throw new Error("Selected storage directories must be unique.");
  const regionalExecution = value.regionalExecution === undefined ? undefined : validateRegionalExecutionSettings(value.regionalExecution);
  if (regionalExecution?.enabled && regionalExecution.routes.some(route => !allocations.some(entry => entry.deviceId === route.deviceId && entry.enabled))) {
    throw new Error("Each regional route requires an explicitly enabled device allocation.");
  }
  return { version: 1, allocations, storageDirectories, allowStorageSpill: value.allowStorageSpill, autoIncreaseStorage: value.autoIncreaseStorage,
    ...(regionalExecution ? { regionalExecution } : {}) };
}

function rate(value: unknown): ResourceRateMeasurement | undefined {
  if (value === undefined) return undefined;
  if (!record(value) || !["bytesPerSecond", "sampleBytes", "runs", "medianNanoseconds"].every((key) => count(value[key]) && value[key] > 0)
    || typeof value.measuredAt !== "string" || !Number.isFinite(Date.parse(value.measuredAt))
    || !["memory-copy", "device-transfer", "durable-file-write"].includes(String(value.scope))) throw new Error("Invalid resource rate measurement.");
  return value as unknown as ResourceRateMeasurement;
}

export function validateDeviceResourceTopology(value: unknown): DeviceResourceTopology {
  if (!record(value) || value.format !== "omni-device-resource-topology" || value.version !== 1 || value.neuralModelConstructed !== false
    || typeof value.measurementRequested !== "boolean" || typeof value.measuredAt !== "string" || !Number.isFinite(Date.parse(value.measuredAt))
    || typeof value.platform !== "string" || typeof value.architecture !== "string" || !Array.isArray(value.devices)
    || !Array.isArray(value.links) || !Array.isArray(value.warnings) || value.warnings.some((entry) => typeof entry !== "string")) throw new Error("Invalid device topology.");
  const ids = new Set<string>();
  const devices = value.devices.map((entry): ResourceDevice => {
    if (!record(entry) || typeof entry.id !== "string" || !ID.test(entry.id) || ids.has(entry.id)
      || typeof entry.label !== "string" || typeof entry.memoryGroupId !== "string" || !ID.test(entry.memoryGroupId)
      || !["ram", "gpu", "npu", "storage"].includes(String(entry.kind))
      || !["cpu", "cuda", "mps", "xpu", "npu", "directml", "filesystem"].includes(String(entry.backend))
      || !["totalBytes", "availableBytes", "allocatedBytes", "reservedBytes"].every((key) => nullableCount(entry[key]))
      || !(entry.unifiedMemory === null || typeof entry.unifiedMemory === "boolean") || typeof entry.nativeComputeSupported !== "boolean"
      || (entry.executionDevice !== undefined && (typeof entry.executionDevice !== "string" || !/^(cpu|cuda:\d+|mps|xpu:\d+|npu:\d+|privateuseone(?::\d+)?)$/.test(entry.executionDevice)))
      || (entry.recommendedLimitBytes !== undefined && !nullableCount(entry.recommendedLimitBytes))
      || (entry.poolDirectory !== undefined && (typeof entry.poolDirectory !== "string" || !entry.poolDirectory || entry.poolDirectory.includes("\0")))
      || (entry.filesystemDeviceId !== undefined && (typeof entry.filesystemDeviceId !== "string" || !/^[0-9]{1,32}$/.test(entry.filesystemDeviceId)))
      || (entry.readOnly !== undefined && typeof entry.readOnly !== "boolean")) throw new Error("Invalid or duplicated resource device.");
    ids.add(entry.id);
    rate(entry.measurement);
    return structuredClone(entry) as unknown as ResourceDevice;
  });
  const links = value.links.map((entry) => {
    if (!record(entry) || typeof entry.from !== "string" || !ids.has(entry.from) || typeof entry.to !== "string" || !ids.has(entry.to)
      || entry.from === entry.to || !(entry.peerAccess === null || typeof entry.peerAccess === "boolean")) throw new Error("Invalid resource transfer link.");
    rate(entry.measurement);
    return structuredClone(entry) as unknown as DeviceResourceTopology["links"][number];
  });
  if (new Set(links.map((link) => `${link.from}>${link.to}`)).size !== links.length) throw new Error("Resource transfer links must be unique.");
  return { format: "omni-device-resource-topology", version: 1, measuredAt: value.measuredAt,
    platform: value.platform, architecture: value.architecture, devices, links,
    measurementRequested: value.measurementRequested, neuralModelConstructed: false, warnings: [...value.warnings] as string[] };
}

export function planDeviceResources(rawTopology: DeviceResourceTopology, rawSettings: DeviceResourceSettings, demand: DeviceResourceDemand): DeviceResourcePlan {
  const topology = validateDeviceResourceTopology(rawTopology);
  const settings = validateDeviceResourceSettings(rawSettings);
  const demandKeys = ["modelBytes", "fixedRuntimeBytes", "contextBytesPerToken", "requestedContextTokens", "systemRamBudgetBytes", "diskReserveBytes"];
  if (!record(demand) || demandKeys.some((key) => !count(demand[key]))
    || Object.keys(demand).some((key) => !demandKeys.includes(key))
    || demand.contextBytesPerToken < 1) throw new Error("Device resource demand requires measured nonnegative byte counts.");
  const selections = new Map(settings.allocations.map((entry) => [entry.deviceId, entry]));
  const blockers: string[] = [];
  const budgets = topology.devices.map((device): DeviceResourceBudget => {
    const selected = selections.get(device.id) ?? { deviceId: device.id, enabled: device.kind === "ram" || device.kind === "storage" || device.nativeComputeSupported,
      mode: "auto" as const, percent: 90, autoIncrease: false };
    let capacity = device.totalBytes, reserve = 0, ceiling = 0;
    if (device.kind === "ram") {
      ceiling = Math.min(capacity ?? 0, demand.systemRamBudgetBytes);
      reserve = Math.max(0, (capacity ?? 0) - ceiling);
      if (selected.mode === "manual") ceiling = Math.min(ceiling, Math.floor((capacity ?? 0) * selected.percent / 100));
    } else if (device.kind === "storage") {
      reserve = demand.diskReserveBytes;
      capacity = device.availableBytes;
      ceiling = Math.floor(Math.max(0, (capacity ?? 0) - reserve) * selected.percent / 100);
    } else {
      capacity ??= device.recommendedLimitBytes ?? null;
      reserve = Math.floor((capacity ?? 0) / 10);
      ceiling = Math.floor(Math.max(0, (capacity ?? 0) - reserve) * selected.percent / 100);
      if (device.memoryGroupId === "ram:system") ceiling = Math.min(ceiling, demand.systemRamBudgetBytes);
      if (device.recommendedLimitBytes !== undefined && device.recommendedLimitBytes !== null) ceiling = Math.min(ceiling, device.recommendedLimitBytes);
    }
    const enabled = selected.enabled && !(device.kind === "storage" && (!settings.allowStorageSpill || device.readOnly));
    if (enabled && capacity === null) blockers.push(`${device.label}: memory capacity is unknown.`);
    const acknowledgedFallback = device.backend === "cuda" && settings.regionalExecution?.enabled
      && settings.regionalExecution.routes.some(route => route.deviceId === device.id && route.executionMode === "packed-cpu-tile-fallback" && route.acknowledgePackedCpuFallback === true);
    if (enabled && device.kind !== "storage" && !device.nativeComputeSupported && !acknowledgedFallback) blockers.push(`${device.label}: its native compute backend is unavailable.`);
    return { deviceId: device.id, memoryGroupId: device.memoryGroupId, kind: device.kind, enabled,
      capacityBytes: capacity, ceilingBytes: enabled ? ceiling : 0, currentlyAvailableBytes: device.availableBytes,
      reserveBytes: reserve, selection: selected.mode, autoIncrease: selected.autoIncrease,
      ...(capacity === null ? { reason: "capacity-unavailable" } : {}) };
  });
  const devices = new Map(topology.devices.map((device) => [device.id, device]));
  for (const selection of settings.allocations) if (selection.enabled && !devices.has(selection.deviceId)) blockers.push(`Selected device ${selection.deviceId} is unavailable.`);
  if (!budgets.some((budget) => budget.enabled && budget.kind !== "storage" && devices.get(budget.deviceId)!.nativeComputeSupported)) blockers.push("Select a supported execution device.");
  const executionBlockers = [...blockers];
  const memoryGroupCeilings: Record<string, number> = Object.create(null);
  const physicalGroups = new Map<string, number>();
  const memoryGroups = new Map<string, number>();
  const storageGroups = new Map<string, number>();
  for (const budget of budgets) {
    const device = devices.get(budget.deviceId)!;
    if (device.kind !== "storage" && device.totalBytes !== null) physicalGroups.set(device.memoryGroupId, Math.max(physicalGroups.get(device.memoryGroupId) ?? 0, device.totalBytes));
    if (!budget.enabled) continue;
    memoryGroupCeilings[budget.memoryGroupId] = Math.max(memoryGroupCeilings[budget.memoryGroupId] ?? 0, budget.ceilingBytes);
    const groups = budget.kind === "storage" ? storageGroups : memoryGroups;
    groups.set(budget.memoryGroupId, Math.max(groups.get(budget.memoryGroupId) ?? 0, budget.ceilingBytes));
  }
  const sum = (values: Iterable<number>) => { let total = 0; for (const value of values) total += value; if (!Number.isSafeInteger(total)) throw new Error("Resource capacity exceeds exact integer accounting."); return total; };
  const physicalMemoryCapacityBytes = sum(physicalGroups.values());
  const selectedMemoryCapacityBytes = sum(memoryGroups.values());
  const selectedStorageCapacityBytes = sum(storageGroups.values());
  const usable = sum([Math.floor(selectedMemoryCapacityBytes * 0.9), Math.floor(selectedStorageCapacityBytes * 0.75)]);
  const fixed = sum([demand.modelBytes, demand.fixedRuntimeBytes]);
  const spendable = Math.max(0, usable - fixed);
  const maximum = Math.floor(spendable / demand.contextBytesPerToken);
  if (fixed > usable) blockers.push("The weighted model and fixed runtime reserve do not fit the selected pools.");
  if (demand.requestedContextTokens > maximum) blockers.push("The requested context, weighted state, and runtime reserve do not fit the selected pools.");
  const transferRates = new Map(topology.links.map((link) => [`${link.from}>${link.to}`, link.measurement?.bytesPerSecond]));
  const effectiveSpeed = (id: string) => {
    const device = devices.get(id)!;
    const local = device.measurement?.bytesPerSecond;
    if (!local) return 0;
    if (device.kind === "ram" || device.kind === "storage") return local;
    const transfer = transferRates.get(`ram:system>${id}`);
    return transfer ? 1 / (1 / local + 1 / transfer) : 0;
  };
  const placementOrder = budgets.filter((budget) => budget.enabled && budget.ceilingBytes > 0).map((budget) => budget.deviceId)
    .sort((a, b) => effectiveSpeed(b) - effectiveSpeed(a) || a.localeCompare(b));
  return { version: 1, topologyMeasuredAt: topology.measuredAt, budgets, placementOrder, memoryGroupCeilings,
    physicalMemoryCapacityBytes, selectedMemoryCapacityBytes, selectedStorageCapacityBytes,
    projectedContextMaximumTokens: maximum, maximumBasis: "weighted-state-and-reserves-deduplicated-physical-pools",
    automaticContextTokens: Math.min(maximum, demand.requestedContextTokens), fits: blockers.length === 0, blockers, executionBlockers };
}
