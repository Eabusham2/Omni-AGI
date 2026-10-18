export type ResourceDeviceKind = "ram" | "gpu" | "npu" | "storage";

export interface ResourceRateMeasurement {
  bytesPerSecond: number;
  sampleBytes: number;
  runs: number;
  medianNanoseconds: number;
  measuredAt: string;
  scope: "memory-copy" | "device-transfer" | "durable-file-write";
}

export interface ResourceDevice {
  id: string;
  label: string;
  kind: ResourceDeviceKind;
  backend: "cpu" | "cuda" | "mps" | "xpu" | "npu" | "directml" | "filesystem";
  /** Runtime index/name may change; preferences bind to id and current topology. */
  executionDevice?: string;
  /** Devices in one group share the same physical bytes. Never sum them. */
  memoryGroupId: string;
  totalBytes: number | null;
  availableBytes: number | null;
  allocatedBytes: number | null;
  reservedBytes: number | null;
  recommendedLimitBytes?: number | null;
  unifiedMemory: boolean | null;
  measurement?: ResourceRateMeasurement;
  nativeComputeSupported: boolean;
  unavailableReason?: string;
  /** Present only for an explicitly supplied storage-pool directory. */
  poolDirectory?: string;
  /** Exact OS device identity observed with the selected directory. */
  filesystemDeviceId?: string;
  readOnly?: boolean;
}

export interface ResourceTransferLink {
  from: string;
  to: string;
  peerAccess: boolean | null;
  measurement?: ResourceRateMeasurement;
  unavailableReason?: string;
}

export interface DeviceResourceTopology {
  format: "omni-device-resource-topology";
  version: 1;
  measuredAt: string;
  platform: string;
  architecture: string;
  devices: ResourceDevice[];
  links: ResourceTransferLink[];
  measurementRequested: boolean;
  neuralModelConstructed: false;
  warnings: string[];
}

export interface DeviceResourceAllocation {
  deviceId: string;
  enabled: boolean;
  mode: "auto" | "manual";
  /** Percentage of this physical device after its reserve; automatic ceilings retain headroom. */
  percent: number;
  autoIncrease: boolean;
}

export interface DeviceResourceSettings {
  version: 1;
  allocations: DeviceResourceAllocation[];
  allowStorageSpill: boolean;
  autoIncreaseStorage: boolean;
  /** Paths come from the user's storage selection, never from neural tool arguments. */
  storageDirectories: string[];
  /** Portable preferences only. Load/import never arms the process-local lease. */
  regionalExecution?: RegionalExecutionSettings;
}

export interface RegionalExecutionRoute {
  /** Exact registered DecoderBlock path, including connected parent namespaces. */
  regionId: string;
  deviceId: string;
  executionMode: "native" | "packed-cpu-tile-fallback";
  acknowledgePackedCpuFallback?: true;
}

export interface RegionalExecutionSettings {
  version: 1;
  enabled: boolean;
  mode: "manual";
  routes: RegionalExecutionRoute[];
}

export interface RegionalExecutionRegion {
  regionId: string;
  dimensions: number;
  heads: number;
  feedForward: number;
  packedOwners: number;
  packedOwnerBytes: number;
  maximumPackedOwnerBytes: number;
  /** Repeated-contact pages stay CPU/paged; this is the accelerator owner bound. */
  maximumDevicePackedOwnerBytes: number;
}

export interface RegionalExecutionStatus {
  format: "omni-manual-regional-execution-status";
  version: 1;
  ownerId: string;
  active: boolean;
  /** An armed but invalidated lease refuses dispatch; it never silently falls back. */
  leaseValid: boolean;
  epoch: string | null;
  reason: string;
  registeredRegions: number;
  activeCalls: number;
  routes: Array<RegionalExecutionRoute & { executionDevice: string; memoryGroupId: string; nativeComputeSupported: boolean }>;
  completedRegionDispatches: number;
  actualActivationTransfers: number;
  actualActivationTransferBytes: number;
  trainingInapplicableCalls: number;
  estimatedKnownTransferSeconds: number;
  unknownTransferCosts: number;
  trainingSupported: false;
  automaticDefault: false;
  contextGpuCapacityCredited: false;
  wholePathPerformanceMeasured: false;
  trace: Array<{ regionId: string; executionDevice: string; callerDevice: string;
    executionMode: RegionalExecutionRoute["executionMode"]; completed: boolean; actualTransferBytes: number; elapsedSeconds: number;
    estimatedTransferSeconds: number | null; unknownTransferCosts: number }>;
}

export interface RegionalExecutionReview {
  format: "omni-manual-regional-execution-review";
  version: 1;
  ownerId: string;
  policySha256: string;
  topologyIdentitySha256: string;
  registrySha256: string;
  topology: DeviceResourceTopology;
  settings: RegionalExecutionSettings;
  start: number;
  total: number;
  regions: RegionalExecutionRegion[];
  status: RegionalExecutionStatus;
}

/** Main observes/validates topology; renderer never supplies it as authority. */
export type RegionalExecutionActivationReview = Pick<RegionalExecutionReview,
  "policySha256" | "topologyIdentitySha256" | "registrySha256" | "topology">;

export const DEFAULT_DEVICE_RESOURCE_SETTINGS: Readonly<DeviceResourceSettings> = Object.freeze({
  version: 1,
  allocations: [],
  allowStorageSpill: true,
  autoIncreaseStorage: false,
  storageDirectories: []
});

export interface DeviceResourceBudget {
  deviceId: string;
  memoryGroupId: string;
  kind: ResourceDeviceKind;
  enabled: boolean;
  capacityBytes: number | null;
  ceilingBytes: number;
  currentlyAvailableBytes: number | null;
  reserveBytes: number;
  selection: "auto" | "manual";
  autoIncrease: boolean;
  reason?: string;
}

export interface DeviceResourcePlan {
  version: 1;
  topologyMeasuredAt: string;
  budgets: DeviceResourceBudget[];
  /** Ordered by measured effective copy/transfer cost; unknown speeds sort last. */
  placementOrder: string[];
  memoryGroupCeilings: Record<string, number>;
  physicalMemoryCapacityBytes: number;
  selectedMemoryCapacityBytes: number;
  selectedStorageCapacityBytes: number;
  projectedContextMaximumTokens: number;
  /** Current running pager limit; expansion requires its concrete placement consumer. */
  admittedContextMaximumTokens?: number;
  maximumBasis: "weighted-state-and-reserves-deduplicated-physical-pools";
  automaticContextTokens: number;
  fits: boolean;
  blockers: string[];
  /** Availability/selection errors, separate from the non-authoritative capacity projection. */
  executionBlockers: string[];
  /** Real selected-drive quotas. Diagnostic raw hardware projections above do not replace these. */
  appliedStorage?: { largestSharedPoolBytes: number; chargedBytes: number; pendingBytes: number;
    groups: Array<{ memoryGroupId: string; ceilingBytes: number; availableAfterReserveBytes: number;
      chargedBytes: number; pendingBytes: number; unmaterializedPromiseBytes: number }> };
}

export interface DeviceResourceDemand {
  modelBytes: number;
  fixedRuntimeBytes: number;
  contextBytesPerToken: number;
  requestedContextTokens: number;
  /** Existing process-wide safe RAM ceiling, retained across resource pressure. */
  systemRamBudgetBytes: number;
  diskReserveBytes: number;
}

/** Main-owned authority/ledger observations, never accepted as renderer input. */
export interface AuthorizedSpillStorageSnapshot {
  groups: Array<{
    physicalGroup: string; ceilingBytes: number; availableBytes: number; allocatedBytes: number;
    pendingBytes: number; unmaterializedPromiseBytes: number; ownerBytes: number;
    primaryVolume: boolean; allocationUnitBytes: number;
  }>;
  largestSharedPoolBytes: number;
  sharedUsedBytes: number;
  sharedPendingBytes: number;
  ownerBytes: number;
  authorized: true;
}
