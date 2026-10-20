import { randomUUID } from "node:crypto";
import { execFile as execFileCallback } from "node:child_process";
import { open, readFile, rm, stat, statfs, writeFile } from "node:fs/promises";
import { freemem, platform, totalmem } from "node:os";
import { join, resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { promisify } from "node:util";
import type {
  BrainConfig,
  HardwareTier,
  WorkingMemoryPlanRequest,
  WorkingMemoryResourcePlan
} from "../shared/types";
import {
  defaultGroundUpWorkingMemoryItems,
  capacityDerivedNativeArchitectureProfile,
  ramFirstNativeArchitectureProfile,
  balancedMeasuredNativeArchitectureProfile,
  groundUpArchitectureProfile,
  type GroundUpArchitectureProfile
} from "./omniArchitectureProfile";
import { nativeCoreInventory, packedOwnerMetadataReserve, packedLearningScratchBytes } from "./nativeCoreInventory";
import { nativeCheckpointMemory, type SavedTensorShape } from "./nativeCheckpointMemory";
import { connectedResourceProfile, type NativeContextGeometry } from "./connectedResourceProfile";
import { authorizedSpillCapacity } from "./authorizedSpillCapacity";
import { planDeviceResources, validateDeviceResourceSettings } from "./deviceResourcePlan";
import { retainedContextPools, checkpointPackedOwnerMinimum, checkpointNonpackedControls, contextCapacity, contextSnapshotBytes, contextColdSpillBytes,
  type RetainedContextPool, type RetainedContextMinimum } from "./retainedContextCapacity";
import type { AuthorizedSpillStorageSnapshot, DeviceResourceSettings, DeviceResourceTopology } from "../shared/deviceResources";
import {
  validateNativeProjectionComputeProfile,
  type NativeComputeMeasurementCollector,
  type NativeComputeMeasurementRequest,
  type NativeProjectionComputeProfile
} from "./nativeComputeMeasurement";
import {
  adaptiveDiskReserve,
  calculateDiskSpaceReport
} from "./diskSpace";

export { MANDATORY_FREE_DISK_BYTES } from "./diskSpace";

export const GIB = 1024 ** 3;
export const MIB = 1024 ** 2;
export const MODEL_CHECKPOINT_HEADROOM_RATIO = 0.2;
export const MODEL_WORKING_SET_RATIO = 1.2;

// One recurrent/paged memory item is not a dense-attention token. The RAM
// estimate includes an active vector, routing/index metadata, and nearby hot
// synapses. Its colder disk form is packed/compressed independently.
export const RAM_BYTES_PER_MEMORY_ITEM = 4 * 1024;
export const DISK_BYTES_PER_MEMORY_ITEM = 768;

const BENCHMARK_SCHEMA = "omni-storage-benchmark-2";
const BENCHMARK_BYTES = 16 * 1024 * 1024;
const BENCHMARK_RUNS = 3;
const BENCHMARK_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;
const execFile = promisify(execFileCallback);

export interface ResourceSnapshot {
  totalMemoryBytes: number;
  availableMemoryBytes: number;
  diskTotalBytes: number;
  diskFreeBytes: number;
  diskBlockSizeBytes?: number;
}

export interface StorageBenchmark {
  measuredAt: string;
  sampleBytes: number;
  memoryBytesPerSecond: number;
  storageBytesPerSecond: number;
  cacheHit: boolean;
  volumeId?: string;
  storageMetric?: "durable-sequential-write-fsync-median";
}

export interface ResourcePlannerDependencies {
  readResources?: () => Promise<ResourceSnapshot>;
  benchmark?: () => Promise<StorageBenchmark>;
  now?: () => Date;
  /** Explicit main hardware-profiling dependency; never a renderer override. */
  measurementCollector?: NativeComputeMeasurementCollector;
  deviceTopology?: (settings: DeviceResourceSettings) => Promise<DeviceResourceTopology>;
  spillCapacity?: (settings: DeviceResourceSettings, topology: DeviceResourceTopology,
    context: { brainId?: string; selectionId?: string }) => Promise<AuthorizedSpillStorageSnapshot>;
}

export function parseLinuxMemoryInfo(text: string): {
  totalMemoryBytes: number;
  availableMemoryBytes: number;
} | undefined {
  const values = new Map<string, number>();
  for (const line of text.split(/\r?\n/)) {
    const match = /^([^:]+):\s*(\d+)\s*kB\s*$/i.exec(line.trim());
    if (!match) continue;
    values.set(match[1]!, Number(match[2]!) * 1024);
  }
  const total = values.get("MemTotal");
  const available = values.get("MemAvailable") ?? [
    "MemFree",
    "Buffers",
    "Cached",
    "SReclaimable"
  ].reduce((sum, key) => sum + (values.get(key) ?? 0), 0);
  return total && available
    ? { totalMemoryBytes: total, availableMemoryBytes: available }
    : undefined;
}

/** Apply a Linux cgroup/app memory ceiling to the host memory reading. */
export function applyMemoryLimit(
  host: { totalMemoryBytes: number; availableMemoryBytes: number },
  limitBytes: number | undefined,
  currentBytes: number | undefined
): { totalMemoryBytes: number; availableMemoryBytes: number } {
  if (
    !Number.isFinite(limitBytes) ||
    !Number.isFinite(currentBytes) ||
    (limitBytes as number) <= 0 ||
    (currentBytes as number) < 0 ||
    (limitBytes as number) >= host.totalMemoryBytes
  ) {
    return host;
  }
  const effectiveTotal = Math.floor(limitBytes as number);
  const effectiveAvailable = Math.max(
    0,
    Math.min(
      host.availableMemoryBytes,
      effectiveTotal - Math.floor(currentBytes as number)
    )
  );
  return {
    totalMemoryBytes: effectiveTotal,
    availableMemoryBytes: effectiveAvailable
  };
}

/** OS/app headroom scales down on constrained and mobile-class memory limits. */
export function adaptiveRamReserve(totalMemoryBytes: number): number {
  const total = Math.max(0, safeWhole(totalMemoryBytes));
  if (total <= 0) return 384 * MIB;
  if (total <= 4 * GIB) {
    return Math.min(384 * MIB, Math.max(128 * MIB, Math.ceil(total * 0.09375)));
  }
  if (total <= 8 * GIB) return 512 * MIB;
  if (total <= 16 * GIB) return GIB;
  return Math.min(2 * GIB, Math.max(GIB, Math.ceil(total * 0.0625)));
}

export function parseMacMemory(
  vmStat: string,
  totalMemoryBytes: number
): { totalMemoryBytes: number; availableMemoryBytes: number } | undefined {
  const pageMatch = /page size of\s+(\d+)\s+bytes/i.exec(vmStat);
  if (!pageMatch || !Number.isFinite(totalMemoryBytes) || totalMemoryBytes <= 0) {
    return undefined;
  }
  const pages = new Map<string, number>();
  for (const line of vmStat.split(/\r?\n/).slice(1)) {
    const match = /^([^:]+):\s*(\d+)\.?\s*$/.exec(line.trim());
    if (match) pages.set(match[1]!, Number(match[2]!));
  }
  const reclaimablePages = [
    "Pages free",
    "Pages inactive",
    "Pages purgeable"
  ].reduce((sum, key) => sum + (pages.get(key) ?? 0), 0);
  if (reclaimablePages <= 0) return undefined;
  return {
    totalMemoryBytes,
    availableMemoryBytes: reclaimablePages * Number(pageMatch[1])
  };
}

/**
 * macOS's `vm_stat` buckets are intentionally conservative and can
 * substantially understate memory that the kernel can reclaim without
 * swapping. `memory_pressure -Q` reports that reclaimable system-wide pool.
 * Keep the two probes independent and use the larger *measured* value; neither
 * is allowed to exceed physical memory.
 */
export function parseMacMemoryPressure(
  text: string,
  totalMemoryBytes: number
): number | undefined {
  const match = /System-wide memory free percentage:\s*(\d+(?:\.\d+)?)%/i.exec(
    text
  );
  if (!match || !Number.isFinite(totalMemoryBytes) || totalMemoryBytes <= 0) {
    return undefined;
  }
  const percent = Number(match[1]);
  if (!Number.isFinite(percent) || percent < 0 || percent > 100) {
    return undefined;
  }
  return Math.min(
    Math.floor(totalMemoryBytes),
    Math.max(0, Math.floor((totalMemoryBytes * percent) / 100))
  );
}

export async function measuredSystemMemory(): Promise<{
  totalMemoryBytes: number;
  availableMemoryBytes: number;
}> {
  try {
    if (platform() === "darwin") {
      const [{ stdout: totalText }, { stdout: vmStat }, pressure] = await Promise.all([
        execFile("sysctl", ["-n", "hw.memsize"], { timeout: 2_000 }),
        execFile("vm_stat", [], { timeout: 2_000 }),
        execFile("memory_pressure", ["-Q"], { timeout: 2_000 }).catch(() => ({
          stdout: ""
        }))
      ]);
      const total = Number(totalText.trim());
      const parsed = parseMacMemory(vmStat, total);
      const pressureAvailable = parseMacMemoryPressure(pressure.stdout, total);
      if (parsed) {
        return {
          ...parsed,
          availableMemoryBytes: Math.max(
            parsed.availableMemoryBytes,
            pressureAvailable ?? 0
          )
        };
      }
    } else if (platform() === "linux") {
      const parsed = parseLinuxMemoryInfo(await readFile("/proc/meminfo", "utf8"));
      if (parsed) {
        // cgroup v2 first, then v1. Containers may report host RAM through
        // /proc/meminfo even though the process is allowed much less.
        for (const [limitPath, currentPath] of [
          ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"],
          [
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
            "/sys/fs/cgroup/memory/memory.usage_in_bytes"
          ]
        ] as const) {
          try {
            const [limitText, currentText] = await Promise.all([
              readFile(limitPath, "utf8"),
              readFile(currentPath, "utf8")
            ]);
            const limit = limitText.trim() === "max" ? undefined : Number(limitText.trim());
            const current = Number(currentText.trim());
            const constrained = applyMemoryLimit(parsed, limit, current);
            if (constrained.totalMemoryBytes < parsed.totalMemoryBytes) return constrained;
          } catch {
            // This cgroup layout is not active; try the other known layout.
          }
        }
        return parsed;
      }
    } else if (platform() === "win32") {
      // libuv implements Node's totalmem/freemem on Windows with
      // GlobalMemoryStatusEx, which is the native physical-memory source.
      return {
        totalMemoryBytes: totalmem(),
        availableMemoryBytes: freemem()
      };
    }
  } catch {
    // Fall through to the portable Node probe.
  }
  return {
    totalMemoryBytes: totalmem(),
    availableMemoryBytes: freemem()
  };
}

const AUTO_SYSTEM_RAM_SHARE_BY_TIER: Record<HardwareTier, number> = {
  micro: 55,
  personal: 65,
  gpu: 75,
  workstation: 80
};

// The baseline is intentionally the only tier-fixed context target. Auto and
// Extended are derived below from the live model/device envelope. These
// floors are large enough for useful conversation while remaining
// conservative for each class's minimum supported memory.
export const CONTEXT_FLOOR_TOKENS_BY_TIER: Record<HardwareTier, number> = {
  micro: 2_048,
  personal: 4_096,
  gpu: 8_192,
  workstation: 16_384
};

// Native rotary positions and streamed attention have no product window cap.
// JSON/token indexes remain exact safe integers; physical admission is lower.
export const DEFAULT_MODEL_CONTEXT_LIMIT_TOKENS = Number.MAX_SAFE_INTEGER;
const CONTEXT_TOKEN_QUANTUM = 256;

const CONTEXT_MODEL_SHAPE_BY_TIER: Record<
  HardwareTier,
  { hiddenSize: number; layers: number }
> = {
  // Conservative fallback for standalone planning without a verified native
  // architecture or checkpoint shape. Product Builds pass their exact profile.
  micro: { hiddenSize: 2_048, layers: 24 },
  personal: { hiddenSize: 2_048, layers: 24 },
  gpu: { hiddenSize: 2_048, layers: 32 },
  workstation: { hiddenSize: 2_048, layers: 32 }
};

// Electron, the supervised worker, tokenizer state, and routing metadata are
// resident alongside the native core. Auto accounts for that baseline instead
// of pretending raw tensor bytes are the entire process working set.
const BASE_RUNTIME_MEMORY_BYTES = 256 * MIB;

const TRAINING_WINDOW_BY_TIER: Record<HardwareTier, number> = {
  micro: 256,
  personal: 1024,
  gpu: 2048,
  workstation: 4096
};

function positiveIntegerText(value: unknown): string | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const text = String(value).trim();
  if (!/^\d+$/.test(text)) return undefined;
  const normalized = text.replace(/^0+(?=\d)/, "");
  return normalized === "0" ? undefined : normalized;
}

function compareIntegerText(left: string, right: string): number {
  const a = left.replace(/^0+(?=\d)/, "");
  const b = right.replace(/^0+(?=\d)/, "");
  if (a.length !== b.length) return a.length < b.length ? -1 : 1;
  return a === b ? 0 : a < b ? -1 : 1;
}

function boundedPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(95, Math.round(value)));
}

function safeWhole(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(Number.MAX_SAFE_INTEGER, Math.floor(value)));
}

/** Persist measured byte rates without leaking fractional timing noise. */
export function roundedStorageBytesPerSecond(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0;
  return Math.max(
    1,
    Math.min(Number.MAX_SAFE_INTEGER, Math.round(value))
  );
}

function tierForConfig(config?: BrainConfig, fallback: HardwareTier = "personal"): HardwareTier {
  const mode = config?.workingMemoryMode;
  void mode;
  return fallback;
}

function prospectiveGroundUpWorkingMemoryItems(
  request: WorkingMemoryPlanRequest,
  hardwareTier: HardwareTier,
  config?: BrainConfig
): number {
  const requested = positiveIntegerText(request.requestedItems);
  if (request.mode === "manual" && requested !== undefined) {
    const numeric = Number(requested);
    if (Number.isSafeInteger(numeric) && numeric > 0) return numeric;
  }
  if (
    request.mode === "manual" &&
    Number.isSafeInteger(config?.workingMemorySlots) &&
    (config?.workingMemorySlots ?? 0) > 0
  ) {
    return config!.workingMemorySlots;
  }
  return defaultGroundUpWorkingMemoryItems(hardwareTier, request.mode);
}

export function planWorkingMemory(input: {
  /** Main-derived current tensor-storage pools; never renderer capacity authority. */
  retainedContextPools?: RetainedContextPool[];
  spillStorage?: AuthorizedSpillStorageSnapshot;
  allowStorageSpill?: boolean;
  request: WorkingMemoryPlanRequest;
  resources: ResourceSnapshot;
  benchmark: StorageBenchmark;
  hardwareTier: HardwareTier;
  modelBytes: number;
  /** Host-only module/registry/cache allowance; it is not checkpoint payload. */
  runtimeMetadataReserveBytes?: number;
  /** Exact project-owned architecture accounting for a new native build. */
  architectureProfile?: GroundUpArchitectureProfile;
  acceleratorAvailable?: boolean;
  modelContextLimitTokens?: number;
  modelHiddenSize?: number;
  modelLayers?: number;
  modelContextGeometry?: NativeContextGeometry;
  maximumNativeBufferBytes?: number;
  maximumPackedOwnerBytes?: number;
  nonpackedControlBytes?: number;
  modelUpdateScratchBytes?: number;
  /** Saved learned latent geometry, independent of runtime context mode. */
  modelWorkspaceSlots?: number;
  fixedWorkingMemoryItems?: number;
  /**
   * Existing device-wide Auto reservation. Auto may grow this ceiling but
   * must not silently shrink it when a later settings preflight does not have
   * the original dataset byte count available.
   */
  minimumStoragePoolBytes?: number;
  /** Existing checkpoint admission does not reserve future growth again. */
  admissionScope?: "capacity" | "existing-runtime";
  /** Existing checkpoints may retain a smaller recorded context. */
  enforceContextFloor?: boolean;
}): WorkingMemoryResourcePlan {
  const { request, resources, hardwareTier } = input;
  const architectureProfile = input.architectureProfile;
  const benchmark: StorageBenchmark = {
    ...input.benchmark,
    storageBytesPerSecond: roundedStorageBytesPerSecond(
      input.benchmark.storageBytesPerSecond
    )
  };
  const modelBytes = safeWhole(input.modelBytes);
  const runtimeMetadataReserveBytes = safeWhole(input.runtimeMetadataReserveBytes ?? 0);
  const admissionScope = input.admissionScope ?? "capacity";
  const allowStorageSpill = input.allowStorageSpill !== false;
  const selectedDrives = input.spillStorage;
  const diskReserveBytes = adaptiveDiskReserve({
    platform: platform(),
    diskTotalBytes: resources.diskTotalBytes,
    diskFreeBytes: resources.diskFreeBytes
  });
  const checkpointHeadroomBytes = Math.ceil(
    modelBytes * MODEL_CHECKPOINT_HEADROOM_RATIO
  );
  const initialModelWriteBytes = architectureProfile ? modelBytes : 0;
  const immutableOriginWriteBytes = architectureProfile ? modelBytes : 0;
  const checkpointWriteBytes = safeWhole(
    immutableOriginWriteBytes + checkpointHeadroomBytes
  );
  const fixedBuildWriteBytes = safeWhole(
    initialModelWriteBytes + checkpointWriteBytes
  );
  // A native checkpoint includes non-weight metaplastic/router metadata.
  // Runtime residency is based on packed ternary learned weights plus a
  // conservative metadata reserve, never a full-precision learned mirror.
  const modelWorkingSetBytes = Math.ceil(
    (architectureProfile?.residentInferenceStateBytes ?? (modelBytes + runtimeMetadataReserveBytes)) *
      MODEL_WORKING_SET_RATIO
  );
  const ramReserveBytes = adaptiveRamReserve(resources.totalMemoryBytes);
  // Capacity is a persisted commitment derived from physical RAM, not a
  // snapshot of whatever happens to be free while Build is open. Current
  // pressure is reported separately below so another application can make the
  // cortex wait/retry, but can never silently shrink its configured brain.
  const safeRamPoolBytes = Math.max(
    0,
    resources.totalMemoryBytes - ramReserveBytes
  );
  const availableSafeRamBytes = Math.max(
    0,
    Math.min(resources.availableMemoryBytes, resources.totalMemoryBytes) -
      ramReserveBytes
  );
  const requestedSystemShare = request.systemRamSharePercent;
  const storageClass = benchmark.storageBytesPerSecond < 100 * 1024 * 1024
    ? "slow-storage" as const
    : benchmark.storageBytesPerSecond < 500 * 1024 * 1024
      ? "moderate-storage" as const
      : "fast-storage" as const;
  const contextShape = CONTEXT_MODEL_SHAPE_BY_TIER[hardwareTier];
  const modelHiddenSize = Math.max(
    1,
    safeWhole(input.modelHiddenSize ?? contextShape.hiddenSize)
  );
  const modelLayers = input.modelContextGeometry?.attentionLayers ?? Math.max(
    1,
    safeWhole(input.modelLayers ?? contextShape.layers)
  );
  // Packed learned weights are ternary; attention activations are float32.
  // Two K/V arrays per layer are paged independently from transient tiles.
  const estimatedKvActivationBytesPerToken = input.modelContextGeometry?.kvBytesPerToken ?? modelLayers * modelHiddenSize * 8;
  const residentTokenBytesPerToken = 96;
  // Conservative Python descriptor/path/checksum allowance, not tensor bytes.
  // The worker coalesces incremental tails into bounded 256-token pages.
  let pageTokens = 256;
  let contextMetadataBytesPerToken = Math.ceil(2048 * 2 * modelLayers / pageTokens);
  let residentIndexBytesPerToken = residentTokenBytesPerToken + contextMetadataBytesPerToken;
  const workspaceSlots = input.modelWorkspaceSlots ?? architectureProfile?.workspaceLatents ?? 8;
  let workspaceResidentBytes = input.modelContextGeometry
    ? input.modelContextGeometry.workspaceStateBytes + input.modelContextGeometry.connectedActivityBytes
    : Math.ceil(4 * workspaceSlots * (2 * modelHiddenSize + 1));
  const modelContextLimitTokens = Math.max(
    8,
    safeWhole(
      input.modelContextLimitTokens ?? DEFAULT_MODEL_CONTEXT_LIMIT_TOKENS
    )
  );
  const contextFloorTokens = Math.min(
    modelContextLimitTokens,
    CONTEXT_FLOOR_TOKENS_BY_TIER[hardwareTier]
  );
  const acceleratorAvailable =
    input.acceleratorAvailable ?? request.acceleratorAvailable ??
    (hardwareTier === "gpu" || hardwareTier === "workstation");
  const estimatedResidentModelBytes = architectureProfile
    ? Math.max(
        8 * MIB,
        architectureProfile.residentInferenceStateBytes +
          architectureProfile.packedUpdateScratchBytes + (architectureProfile.packedForwardTensorReserveBytes ?? 0)
      )
    : Math.max(512 * MIB, Math.ceil(modelBytes * 1.5) + 128 * MIB,
        modelBytes + (input.modelUpdateScratchBytes ?? 0) + 128 * MIB) + runtimeMetadataReserveBytes;
  const fullFoundationRuntimeBytes =
    Math.max(modelWorkingSetBytes, estimatedResidentModelBytes) +
    BASE_RUNTIME_MEMORY_BYTES;
  const maximumPackedOwnerBytes = input.maximumPackedOwnerBytes ?? architectureProfile?.maximumProjectionBytes;
  const inventory = architectureProfile?.nativeArchitecture?.inventory;
  const nonpackedControlBytes = input.nonpackedControlBytes ?? (inventory
    ? inventory.routerNonweightTensorBytes + inventory.freshRouteControlBytes + inventory.liquidStateBytes : undefined);
  const actualManualCoreMinimum = request.mode === "manual" && maximumPackedOwnerBytes !== undefined && nonpackedControlBytes !== undefined
    ? Math.max(8 * MIB, maximumPackedOwnerBytes + nonpackedControlBytes + (architectureProfile?.pythonOwnerMetadataReserveBytes ?? runtimeMetadataReserveBytes))
    : undefined;
  const minimumLayerResidencyBytes = actualManualCoreMinimum ?? (architectureProfile
    ? Math.max(
        8 * MIB,
        architectureProfile.fixedControlBufferReserveBytes +
          architectureProfile.pythonOwnerMetadataReserveBytes +
          architectureProfile.packedMetaplasticityReserveBytes +
          architectureProfile.packedWorkspaceTableBytes +
          architectureProfile.workspaceConnectionStateBytes +
          architectureProfile.packedUpdateScratchBytes +
          (architectureProfile.packedForwardTensorReserveBytes ?? 0) +
          Math.ceil(
            (architectureProfile.packedLearnedStateBytes -
              architectureProfile.packedWorkspaceTableBytes - architectureProfile.workspaceConnectionStateBytes) /
              Math.max(1, architectureProfile.layers)
          ),
        architectureProfile.maximumProjectionBytes + architectureProfile.packedUpdateScratchBytes +
          (architectureProfile.packedForwardTensorReserveBytes ?? 0) +
          architectureProfile.fixedControlBufferReserveBytes + architectureProfile.pythonOwnerMetadataReserveBytes
      )
    : Math.max(input.modelUpdateScratchBytes ?? 0,
        192 * MIB,
        Math.min(512 * MIB, Math.floor((estimatedResidentModelBytes - runtimeMetadataReserveBytes) / 4))
      ) + runtimeMetadataReserveBytes);
  const minimumFoundationContextTokens = Math.min(
    256,
    modelContextLimitTokens
  );
  const minimumContextWorkspaceBytes = Math.max(8 * MIB,
    workspaceResidentBytes + minimumFoundationContextTokens *
      (estimatedKvActivationBytesPerToken + residentIndexBytesPerToken));
  // Slow storage receives more of the already-safe RAM pool so Auto avoids
  // paging. Current external pressure is separate from this saved ceiling.
  const autoBaseSharePercent =
    AUTO_SYSTEM_RAM_SHARE_BY_TIER[hardwareTier] +
    (benchmark.storageBytesPerSecond > 0 ? storageClass === "slow-storage" ? 10 : storageClass === "moderate-storage" ? 5 : 0 : 0);
  // Same base policy as ResourcePolicy._adaptive_system_ram_share. Model size
  // cannot silently increase the selected RAM ceiling; use admitted paging or
  // an explicit revised allocation when the packed model is larger.
  const autoSystemRamSharePercent = Math.min(90, autoBaseSharePercent);
  const manualSystemShareValid =
    request.systemRamMode !== "manual" ||
    (typeof requestedSystemShare === "number" &&
      Number.isFinite(requestedSystemShare) &&
      requestedSystemShare >= 30 &&
      requestedSystemShare <= 100);
  const systemRamMode = request.systemRamMode ?? "auto";
  const systemRamSharePercent =
    systemRamMode === "manual" && manualSystemShareValid
      ? Math.round(requestedSystemShare as number)
      : autoSystemRamSharePercent;
  const systemRamBudgetBytes = Math.floor(
    safeRamPoolBytes * systemRamSharePercent / 100
  );
  const safeRamBytes = Math.min(safeRamPoolBytes, systemRamBudgetBytes);
  const currentOmniAvailableBytes = Math.min(
    systemRamBudgetBytes,
    availableSafeRamBytes
  );
  const currentOmniShortfallBytes = Math.max(
    0,
    systemRamBudgetBytes - currentOmniAvailableBytes
  );
  const usableDiskAfterReserveBytes = Math.max(
    0,
    resources.diskFreeBytes - diskReserveBytes - fixedBuildWriteBytes
  );
  const minimumResidentFoundationBytes =
    BASE_RUNTIME_MEMORY_BYTES + minimumLayerResidencyBytes;
  // Match the worker's one shared post-baseline 55/30/15 envelope. Full packed
  // core residency is preferred; cold core and K/V can spill independently.
  const residualRamBytes = Math.max(0, safeRamBytes - BASE_RUNTIME_MEMORY_BYTES);
  const transferBudgetBytes = Math.floor(residualRamBytes * 0.15);
  const attentionWidths = input.modelContextGeometry?.leaves.map(leaf => leaf.width) ?? [modelHiddenSize];
  const maximumAttentionWidth = Math.max(modelHiddenSize, ...attentionWidths);
  pageTokens = Math.max(1, Math.min(256, Math.floor(transferBudgetBytes / Math.max(1, 16 * maximumAttentionWidth))));
  const retainedContextPageBytes = Math.max(...attentionWidths.map(width =>
    Math.max(1, Math.min(256, Math.floor(transferBudgetBytes / Math.max(1, 16 * width)))) * width * 4));
  contextMetadataBytesPerToken = Math.ceil(2048 * 2 * modelLayers / pageTokens);
  residentIndexBytesPerToken = residentTokenBytesPerToken + contextMetadataBytesPerToken;
  const querySlotsPerChunk = Math.max(1, Math.min(256, workspaceSlots,
    Math.floor(transferBudgetBytes / Math.max(1, 64 * modelHiddenSize))));
  const queryChunks = input.modelContextGeometry ? input.modelContextGeometry.leaves.reduce((sum, leaf) => sum
    + Math.ceil(leaf.workspaceSlots / Math.max(1, Math.min(256, leaf.workspaceSlots, Math.floor(transferBudgetBytes / Math.max(1, 64 * leaf.width))))), 0)
    : Math.ceil(workspaceSlots / querySlotsPerChunk);
  workspaceResidentBytes += queryChunks * 8192 + 4096 * modelLayers;
  const defaultDiskUnit = Math.max(platform() === "win32" ? 65536 : 4096, safeWhole(resources.diskBlockSizeBytes ?? 16384));
  const diskBlockSizeBytes = selectedDrives?.groups.reduce((largest, group) => Math.max(largest, group.allocationUnitBytes), defaultDiskUnit) ?? defaultDiskUnit;
  const pagedKvBytesPerToken = estimatedKvActivationBytesPerToken + Math.ceil(2 * modelLayers * diskBlockSizeBytes / pageTokens);
  const manualCoreRebalance = request.mode === "manual" && maximumPackedOwnerBytes !== undefined && maximumPackedOwnerBytes > 0;
  const plannedResidentModelBytes = Math.min(estimatedResidentModelBytes, manualCoreRebalance
    ? minimumLayerResidencyBytes : Math.max(minimumLayerResidencyBytes, Math.floor(residualRamBytes * 0.55)));
  const planningFoundationResidentBytes = BASE_RUNTIME_MEMORY_BYTES + plannedResidentModelBytes;
  const safeRamAfterModelBytes = Math.max(
    0,
    safeRamBytes - planningFoundationResidentBytes
  );
  const contextResidentBudgetBytes = Math.max(0, manualCoreRebalance
    ? safeRamAfterModelBytes - transferBudgetBytes
    : Math.min(safeRamAfterModelBytes - Math.floor(residualRamBytes * 0.15), Math.floor(residualRamBytes * 0.30)));
  const trainingSourceBytes = safeWhole(request.trainingSourceBytes ?? 0);
  const trainingScratchBytes = Math.max(512 * MIB,
    Math.ceil(trainingSourceBytes * 0.1), Math.ceil(modelBytes * 0.5));
  const requestedItemReserve = safeWhole(Number(positiveIntegerText(request.requestedItems) ?? 0));
  const baseGrowthHeadroomBytes = Math.max(2 * GIB, Math.ceil(modelBytes * 1.2),
    Math.ceil(requestedItemReserve * DISK_BYTES_PER_MEMORY_ITEM * 0.5));
  const plannedModelSpillBytes = Math.max(0, estimatedResidentModelBytes - plannedResidentModelBytes);
  const plannedCoreScratchBytes = plannedModelSpillBytes > 0
    ? Math.max(64 * MIB, Math.ceil(plannedModelSpillBytes * 1.25) + 8 * MIB) : 0;
  const primaryPoolCapacityBytes = Math.max(0, resources.diskFreeBytes - diskReserveBytes);
  // Only locally authorized, physically distinct groups add placement capacity.
  // The primary drive still owns all checkpoints, origins and persistent data.
  const maximumStoragePoolBytes = safeWhole(primaryPoolCapacityBytes + (selectedDrives?.groups
    .filter((group) => !group.primaryVolume).reduce((total, group) => total + group.ceilingBytes, 0) ?? 0));
  const storagePoolMode = request.storagePoolMode ?? "auto";
  const requestedStoragePoolText = positiveIntegerText(request.storagePoolBytes);
  const boundedStoragePoolCapacity = storagePoolMode === "manual"
    ? Math.min(maximumStoragePoolBytes, safeWhole(Number(requestedStoragePoolText ?? 0)))
    : maximumStoragePoolBytes;
  const primaryRequiredBytes = admissionScope === "existing-runtime" ? checkpointHeadroomBytes
    : fixedBuildWriteBytes + trainingScratchBytes + baseGrowthHeadroomBytes;
  const selectedDriveCapacity = selectedDrives ? authorizedSpillCapacity(selectedDrives, {
    selectedPoolBytes: boundedStoragePoolCapacity, primaryRequiredBytes
  }) : undefined;
  const contextSpillCapacityBytes = allowStorageSpill ? Math.max(0,
    (selectedDriveCapacity?.capacityBytes ?? (boundedStoragePoolCapacity - primaryRequiredBytes)) - plannedCoreScratchBytes) : 0;
  const contextResidentWorkingBytes = Math.max(0, contextResidentBudgetBytes - workspaceResidentBytes);
  const retainedDeviceContextBytes = safeWhole((input.retainedContextPools ?? []).reduce((sum, pool) => sum + pool.budgetBytes, 0));
  // Every selected window must remain serializable on the checkpoint volume,
  // independently of whether optional *runtime* disk spill is enabled.
  const contextCheckpointCapacityBytes = Math.max(0, Math.min(primaryPoolCapacityBytes, boundedStoragePoolCapacity)
    - primaryRequiredBytes - plannedCoreScratchBytes - workspaceResidentBytes);
  const contextSharedStorageCapacityBytes = Math.max(0,
    boundedStoragePoolCapacity - primaryRequiredBytes - plannedCoreScratchBytes - workspaceResidentBytes);
  // Selected-drive capacity constrains runtime cold spill, NOT snapshots.
  // Checkpoints stay on primary even when only extra drives were authorized
  // for spill. The bounded shared pool already caps the deduplicated physical
  // union (primary once + nonprimary groups), while the two independent
  // capacity predicates above constrain primary snapshot and cold placement.
  const capacity = contextCapacity({ ramWorkingBytes: safeWhole(contextResidentWorkingBytes), deviceContextBytes: retainedDeviceContextBytes,
    kvBytesPerToken: safeWhole(estimatedKvActivationBytesPerToken), indexBytesPerToken: safeWhole(residentIndexBytesPerToken),
    modelLimitTokens: modelContextLimitTokens, spillCapacityBytes: safeWhole(contextSpillCapacityBytes),
    pagedKvBytesPerToken: safeWhole(pagedKvBytesPerToken), checkpointCapacityBytes: safeWhole(contextCheckpointCapacityBytes), pageTokens, layers: modelLayers,
    sharedStorageCapacityBytes: safeWhole(contextSharedStorageCapacityBytes) });
  const residentContextMaximumTokens = capacity.residentMaximumTokens;
  const contextMaximumTokens = capacity.maximumTokens;
  // Auto favors RAM on slow drives. Extended may use the explicitly budgeted
  // attention spill, but storage bandwidth never masquerades as RAM capacity.
  const autoContextRamFraction = Math.min(
    0.82,
    Math.max(
      0.5,
      (storageClass === "fast-storage"
        ? 0.7
        : storageClass === "moderate-storage"
          ? 0.64
          : 0.58) + (acceleratorAvailable ? 0.08 : 0)
    )
  );
  const quantizeContext = (tokens: number): number => {
    const bounded = Math.max(0, Math.min(contextMaximumTokens, safeWhole(tokens)));
    if (bounded < CONTEXT_TOKEN_QUANTUM) return bounded;
    return Math.floor(bounded / CONTEXT_TOKEN_QUANTUM) * CONTEXT_TOKEN_QUANTUM;
  };
  const measuredAutoContext = quantizeContext(
    (contextResidentWorkingBytes + retainedDeviceContextBytes) * autoContextRamFraction /
      (estimatedKvActivationBytesPerToken + residentIndexBytesPerToken)
  );
  const declaredResidentBaseline = architectureProfile?.nativeArchitecture?.sizing.selectionMode === "ram-first-headroom-default"
    ? safeWhole(Number(architectureProfile.nativeArchitecture.sizing.baselineContextTokens)) : 0;
  const autoContextTokens = residentContextMaximumTokens < contextFloorTokens
    ? residentContextMaximumTokens
    : Math.max(contextFloorTokens, Math.min(residentContextMaximumTokens,
        Math.max(measuredAutoContext, declaredResidentBaseline)));
  const extendedContextTokens = quantizeContext(
    Math.max(
      autoContextTokens,
      autoContextTokens * 1.5,
      autoContextTokens + Math.floor(contextFloorTokens / 2)
    )
  );
  const requestedContextText = positiveIntegerText(
    request.requestedContextTokens
  );
  const requestedContextFits =
    requestedContextText !== undefined &&
    compareIntegerText(
      requestedContextText,
      String(contextMaximumTokens)
    ) <= 0;
  const selectedContextTokens =
    request.mode === "manual" && request.requestedContextTokens !== undefined
      ? requestedContextFits
        ? Number(requestedContextText)
        : 0
      : request.mode === "extended"
        ? extendedContextTokens
        : autoContextTokens;
  const selectedContextIndexBytes = selectedContextTokens * residentIndexBytesPerToken;
  const selectedContextKvBytes = selectedContextTokens * estimatedKvActivationBytesPerToken;
  const selectedContextKvResidentBytes = Math.min(selectedContextKvBytes,
    Math.max(0, contextResidentWorkingBytes - selectedContextIndexBytes));
  const selectedContextDeviceBytes = Math.min(retainedDeviceContextBytes, Math.max(0, selectedContextKvBytes - selectedContextKvResidentBytes));
  const selectedContextSpillBytes = Math.max(0, selectedContextKvBytes - selectedContextKvResidentBytes - selectedContextDeviceBytes);
  const contextCheckpointReserveBytes = contextSnapshotBytes(selectedContextTokens, {
    kvBytesPerToken: estimatedKvActivationBytesPerToken, indexBytesPerToken: residentIndexBytesPerToken, pageTokens, layers: modelLayers
  }) + workspaceResidentBytes;
  // The full paused snapshot is reserved separately. Runtime spill covers the
  // allocation-rounded cold remainder, not an artificial choose-one tier or
  // a second copy of every hot RAM/VRAM page. New pressure pauses, never crops.
  const contextOffloadBudgetBytes = allowStorageSpill ? contextColdSpillBytes(selectedContextTokens, {
    ramWorkingBytes: contextResidentWorkingBytes, deviceContextBytes: retainedDeviceContextBytes,
    kvBytesPerToken: estimatedKvActivationBytesPerToken, indexBytesPerToken: residentIndexBytesPerToken,
    pagedKvBytesPerToken, pageTokens
  }) : 0;
  const selectedContextResidentBytes = workspaceResidentBytes + selectedContextIndexBytes + selectedContextKvResidentBytes;
  // Maximum-context capacity can borrow cold core cache, but a smaller
  // selected window should retain a fitting model and avoid needless spill.
  const residentModelBytes = manualCoreRebalance ? Math.min(estimatedResidentModelBytes,
    Math.max(minimumLayerResidencyBytes, Math.min(Math.max(minimumLayerResidencyBytes, Math.floor(residualRamBytes * 0.55)),
      safeRamBytes - BASE_RUNTIME_MEMORY_BYTES - selectedContextResidentBytes - transferBudgetBytes))) : plannedResidentModelBytes;
  const residentFoundationBytes =
    BASE_RUNTIME_MEMORY_BYTES + residentModelBytes;
  const modelSpillBytes = Math.max(
    0,
    estimatedResidentModelBytes - residentModelBytes
  );
  const modelOffloadScratchBytes = modelSpillBytes > 0
    ? architectureProfile
      ? Math.max(64 * MIB, Math.ceil(modelSpillBytes * 1.25) + 8 * MIB)
      : Math.max(
          256 * MIB,
          Math.ceil(modelBytes * 1.25) + 64 * MIB
        )
    : 0;
  const ramBytesPerMemoryItem = Math.max(RAM_BYTES_PER_MEMORY_ITEM, 4 * modelHiddenSize + 512);
  const diskBytesPerMemoryItem = Math.max(DISK_BYTES_PER_MEMORY_ITEM, 4 * modelHiddenSize + 512);
  const residentMemoryBytes = Math.max(
    0,
    safeRamBytes - residentFoundationBytes - selectedContextResidentBytes - transferBudgetBytes
  );
  const primaryEphemeralBytes = !selectedDrives || selectedDrives.groups.some((group) => group.primaryVolume)
    ? modelOffloadScratchBytes + contextOffloadBudgetBytes : 0;
  const diskForPagedMemoryBytes = allowStorageSpill ? Math.max(
    0,
    Math.min(usableDiskAfterReserveBytes, boundedStoragePoolCapacity - fixedBuildWriteBytes)
      - primaryEphemeralBytes - contextCheckpointReserveBytes - trainingScratchBytes - baseGrowthHeadroomBytes
  ) : 0;
  const residentItems = safeWhole(
    residentMemoryBytes / ramBytesPerMemoryItem
  );
  const pagedItems = safeWhole(
    diskForPagedMemoryBytes / diskBytesPerMemoryItem
  );
  const sliderMaximumItems = safeWhole(residentItems + pagedItems);

  const autoFastActivityBytes = Math.floor(contextResidentBudgetBytes * 0.20);
  const autoTarget = Math.min(residentItems, Math.max(1,
    safeWhole(autoFastActivityBytes / (ramBytesPerMemoryItem + 2 * modelHiddenSize + 1))));
  const extendedTarget = Math.min(
    sliderMaximumItems,
    Math.max(autoTarget, Math.floor(autoTarget * 1.5))
  );
  const requestedText = positiveIntegerText(request.requestedItems);
  const requestedFits =
    requestedText !== undefined &&
    compareIntegerText(requestedText, String(sliderMaximumItems)) <= 0;
  const derivedManualItems = Math.min(
    sliderMaximumItems,
    Math.max(
      1,
      Math.round(
        autoTarget *
          (selectedContextTokens / Math.max(1, autoContextTokens))
      )
    )
  );
  const fixedItems = input.fixedWorkingMemoryItems;
  if (fixedItems !== undefined && (!Number.isSafeInteger(fixedItems) || fixedItems < 1)) {
    throw new Error("Saved learned workspace capacity is invalid.");
  }
  const selectedItems = fixedItems ?? (
    request.mode === "manual"
      ? request.requestedItems === undefined
        ? derivedManualItems
        : requestedFits
          ? Number(requestedText)
          : 0
      : request.mode === "extended"
        ? extendedTarget
        : autoTarget);
  const memorySpillItems = Math.max(0, selectedItems - residentItems);
  const memorySpillBytes = memorySpillItems * diskBytesPerMemoryItem;
  const configuredMemorySpillBytes =
    modelOffloadScratchBytes + memorySpillBytes + selectedContextSpillBytes;
  const offloadRequired = modelSpillBytes > 0 || memorySpillItems > 0 || selectedContextSpillBytes > 0;
  // Source datasets remain referenced in place; this is transactional replay,
  // staging, and recovery space sized from the selected corpus rather than a
  // duplicate of every source byte.
  const futureGrowthHeadroomBytes = Math.max(
    baseGrowthHeadroomBytes,
    Math.ceil(selectedItems * DISK_BYTES_PER_MEMORY_ITEM * 0.5)
  );
  const requiredStoragePoolBytes = safeWhole(
    fixedBuildWriteBytes +
      modelOffloadScratchBytes +
      memorySpillBytes +
      contextOffloadBudgetBytes +
      contextCheckpointReserveBytes +
      trainingScratchBytes +
      futureGrowthHeadroomBytes
  );
  const diskSpace = calculateDiskSpaceReport({
    diskTotalBytes: resources.diskTotalBytes,
    diskFreeBytes: resources.diskFreeBytes,
    mandatoryReserveBytes: diskReserveBytes,
    components: {
      selectedDatasetBytes: trainingSourceBytes,
      modelBytes: initialModelWriteBytes,
      checkpointBytes: checkpointWriteBytes + contextCheckpointReserveBytes,
      maximumWorkingMemorySpillBytes:
        admissionScope === "existing-runtime" ? 0 : memorySpillBytes + (primaryEphemeralBytes > 0 ? contextOffloadBudgetBytes : 0),
      futureGrowthBytes:
        admissionScope === "existing-runtime" ? 0 : futureGrowthHeadroomBytes,
      operationWriteBytes: admissionScope === "existing-runtime"
        ? 0
        : (primaryEphemeralBytes > 0 ? modelOffloadScratchBytes : 0) + trainingScratchBytes
    }
  });
  const minimumStoragePoolBytes = safeWhole(input.minimumStoragePoolBytes ?? 0);
  const manualStoragePoolValid =
    storagePoolMode !== "manual" || requestedStoragePoolText !== undefined;
  const requestedStoragePoolFits =
    requestedStoragePoolText !== undefined &&
    compareIntegerText(
      requestedStoragePoolText,
      String(maximumStoragePoolBytes)
    ) <= 0;
  // Auto reservations are reusable capacity hints, not irrevocable disk
  // allocations. Preserve an existing reservation while it still fits behind
  // the mandatory free-space reserve, but do not let an old/high-water value
  // permanently lock a brain out after available disk changes. The current
  // workload still has to fit in full; manual reservations remain strict.
  const staleAutoReservationRightSized =
    admissionScope === "capacity" &&
    storagePoolMode === "auto" &&
    minimumStoragePoolBytes > maximumStoragePoolBytes &&
    requiredStoragePoolBytes <= maximumStoragePoolBytes;
  const retainedAutoMinimumStoragePoolBytes =
    storagePoolMode === "auto" &&
    (admissionScope === "existing-runtime" ||
      minimumStoragePoolBytes <= maximumStoragePoolBytes)
      ? minimumStoragePoolBytes
      : 0;
  const sharedStoragePoolBytes = storagePoolMode === "manual"
    ? requestedStoragePoolFits || (
        admissionScope === "existing-runtime" &&
        requestedStoragePoolText !== undefined
      )
      ? Number(requestedStoragePoolText)
      : 0
    : Math.max(requiredStoragePoolBytes, retainedAutoMinimumStoragePoolBytes);
  const minimumLiveFoundationFits =
    minimumResidentFoundationBytes + selectedContextResidentBytes <= safeRamBytes;
  const diskFitsBase =
    resources.diskFreeBytes >
    diskReserveBytes + (
      admissionScope === "existing-runtime"
        ? checkpointHeadroomBytes + contextCheckpointReserveBytes
        : fixedBuildWriteBytes + contextCheckpointReserveBytes + (primaryEphemeralBytes > 0 ? modelOffloadScratchBytes : 0)
    );
  const diskFitsSelection =
    !diskSpace.paused;
  const storagePoolFits = admissionScope === "existing-runtime" || (
    Math.max(sharedStoragePoolBytes, selectedDrives?.largestSharedPoolBytes ?? 0) >= requiredStoragePoolBytes &&
    sharedStoragePoolBytes <= maximumStoragePoolBytes
  );
  const manualItemsValid =
    request.requestedItems === undefined || requestedText !== undefined;
  const manualContextValid =
    request.requestedContextTokens === undefined ||
    requestedContextText !== undefined;
  const manualValid =
    request.mode !== "manual" ||
    (manualItemsValid &&
      manualContextValid &&
      (request.requestedItems !== undefined ||
        request.requestedContextTokens !== undefined));
  const itemSelectionFits =
    fixedItems !== undefined ? fixedItems <= sliderMaximumItems :
    request.mode !== "manual" ||
    request.requestedItems === undefined ||
    (requestedFits && selectedItems > 0 && selectedItems <= sliderMaximumItems);
  const contextSelectionFits =
    request.mode !== "manual" ||
    request.requestedContextTokens === undefined ||
    (requestedContextFits &&
      selectedContextTokens > 0 &&
      selectedContextTokens <= contextMaximumTokens);
  const enforceContextFloor = input.enforceContextFloor !== false;
  const blockers: string[] = [];
  const warnings: string[] = [];
  if (!allowStorageSpill && offloadRequired) blockers.push("Storage spill is off. Increase the RAM allocation or enable storage spill before loading this selection.");
  if (selectedDriveCapacity) {
    const ephemeralBytes = modelOffloadScratchBytes + contextOffloadBudgetBytes;
    if (ephemeralBytes > selectedDriveCapacity.currentlyAvailableBytes) blockers.push("Selected spill drives are temporarily occupied or have pending writes. The selected context is retained; free space or wait for those operations to finish.");
    const largestFile = Math.max(retainedContextPageBytes, modelSpillBytes > 0 ? input.maximumNativeBufferBytes ?? architectureProfile?.maximumProjectionBytes ?? 0 : 0);
    if (largestFile > selectedDriveCapacity.largestSingleAllocationBytes) blockers.push("No selected drive can hold the largest native page. Increase one drive's allocation; several drives cannot split one mapped page.");
  }
  if (!manualSystemShareValid) {
    blockers.push(
      "The Omni RAM cap must be between 30% and 100% of safely available memory."
    );
  }
  if (!manualStoragePoolValid) {
    blockers.push("Enter a positive whole-number shared storage-pool size.");
  } else if (!storagePoolFits) {
    blockers.push(
      sharedStoragePoolBytes < requiredStoragePoolBytes
        ? `The shared storage pool must be at least ${requiredStoragePoolBytes.toLocaleString()} bytes for the selected model, training scratch, and future neural growth.`
        : "The shared storage pool would cross the mandatory free-space reserve."
    );
  }
  if (!diskFitsBase) {
    blockers.push(
      architectureProfile
        ? `Not enough free storage for the native model, immutable origin, candidate headroom, required state-offload scratch, and the ${diskReserveBytes.toLocaleString()}-byte device safety reserve.`
        : `Not enough free storage for checkpoint headroom, required state-offload scratch, and the ${diskReserveBytes.toLocaleString()}-byte device safety reserve.`
    );
  }
  if (!minimumLiveFoundationFits) {
    blockers.push(
      "The Omni RAM envelope cannot keep the live neural tile, context indexes and runtime resident. Increase the RAM cap or reduce context; cold attention and packed core pages may use the designated storage pool."
    );
  }
  if (enforceContextFloor && contextMaximumTokens < contextFloorTokens) {
    blockers.push(
      `The RAM and designated storage pool cannot fit the ${contextFloorTokens.toLocaleString()}-token device baseline (physical maximum ${contextMaximumTokens.toLocaleString()} tokens).`
    );
  }
  if (!manualValid) {
    blockers.push("Enter a positive whole-number active-context or memory size.");
  } else if (!itemSelectionFits) {
    blockers.push(
      `The requested memory needs more physical RAM and storage than this device can safely provide (safe maximum ${sliderMaximumItems.toLocaleString()} items).`
    );
  }
  if (!contextSelectionFits) {
    blockers.push(
      `The requested active context cannot fit RAM plus the designated storage pool (physical/model maximum ${contextMaximumTokens.toLocaleString()} tokens).`
    );
  } else if (
    request.mode === "manual" &&
    request.requestedContextTokens !== undefined &&
    selectedContextTokens < contextFloorTokens
  ) {
    const message =
      `This context is below the ${contextFloorTokens.toLocaleString()}-token device baseline.`;
    if (enforceContextFloor) blockers.push(message);
    else warnings.push(`${message} It is retained only for checkpoint compatibility.`);
  }
  if (!diskFitsSelection && diskFitsBase) {
    blockers.push(
      admissionScope === "existing-runtime"
        ? "The next checkpoint headroom would cross the device-adaptive free-space reserve. Existing neural data remains intact and readable."
        : "The selected working memory would consume storage reserved for checkpoints or cross the device-adaptive free-space boundary."
    );
  }
  if (sliderMaximumItems < 1 && blockers.length === 0) {
    blockers.push("No safe working-memory capacity remains after the model and device reserves.");
  }

  const storageRatio = Math.max(
    0.001,
    Math.min(1, benchmark.storageBytesPerSecond / benchmark.memoryBytesPerSecond)
  );
  const offloadedFraction = offloadRequired
    ? (modelSpillBytes + memorySpillBytes + selectedContextSpillBytes) /
      Math.max(
        1,
        fullFoundationRuntimeBytes +
          selectedContextResidentBytes +
          selectedItems * RAM_BYTES_PER_MEMORY_ITEM
      )
    : 0;
  const estimatedSlowdownPercent = boundedPercent(
    offloadedFraction * (1 - storageRatio) * 100
  );
  if (offloadRequired && minimumLiveFoundationFits && diskFitsSelection) {
    warnings.push(
      `Storage offload is required and is estimated to reduce memory-heavy speed by about ${estimatedSlowdownPercent}%.`
    );
  }
  const minimumCurrentLiveBytes =
    minimumResidentFoundationBytes + minimumContextWorkspaceBytes;
  if (currentOmniAvailableBytes < minimumCurrentLiveBytes) {
    warnings.push(
      `Other applications currently occupy ${currentOmniShortfallBytes.toLocaleString()} bytes of Omni's saved RAM envelope. Close memory-heavy applications; the cortex keeps its configured capacity and waits/retries instead of shrinking it.`
    );
  } else if (currentOmniShortfallBytes > 0) {
    warnings.push(
      `Only ${currentOmniAvailableBytes.toLocaleString()} of ${systemRamBudgetBytes.toLocaleString()} committed RAM bytes are free right now. Close memory-heavy applications for full speed; the saved capacity remains locked and grows into the allocation as the OS releases memory.`
    );
  }
  if (systemRamMode === "manual") {
    if (systemRamSharePercent <= 40) {
      warnings.push(
        "This RAM cap is very low; training will use smaller batches and windows than Auto while still visiting the full dataset."
      );
    } else if (systemRamSharePercent < autoSystemRamSharePercent) {
      warnings.push(
        `This cap is below Auto's ${autoSystemRamSharePercent}% recommendation and will usually train more slowly.`
      );
    }
    if (systemRamSharePercent >= 90) {
      warnings.push(
        "This cap is near the safe-pool boundary; the OS reserve remains protected, but background applications may force training to pause."
      );
    }
  }
  if (admissionScope === "existing-runtime" && diskFitsSelection) {
    warnings.push(
      "Runtime admission reused the existing checkpoint and did not reserve future growth, training scratch, or spill capacity again. Every actual checkpoint and mutation write remains live reserve-gated."
    );
  }
  if (
    storagePoolMode === "auto" &&
    minimumStoragePoolBytes > requiredStoragePoolBytes &&
    minimumStoragePoolBytes <= maximumStoragePoolBytes &&
    sharedStoragePoolBytes <= maximumStoragePoolBytes
  ) {
    warnings.push(
      "The existing shared storage reservation is retained so another brain or a previous training selection does not lose its reusable scratch and growth capacity."
    );
  }
  if (staleAutoReservationRightSized) {
    warnings.push(
      `The previous Auto storage reservation no longer fits behind the device-adaptive free-space reserve. Auto safely right-sized reusable scratch and growth capacity to ${sharedStoragePoolBytes.toLocaleString()} bytes; neural data was not deleted.`
    );
  }
  if (
    storageClass === "slow-storage" &&
    configuredMemorySpillBytes > Math.max(GIB, systemRamBudgetBytes * 0.2)
  ) {
    warnings.push(
      "Heavy spill is projected on slow storage; Auto favors RAM and pauses instead of using storage as per-step virtual memory."
    );
  }
  if (benchmark.memoryBytesPerSecond < 2 * GIB) {
    warnings.push(
      `The bounded RAM-copy check measured ${Math.round(benchmark.memoryBytesPerSecond / MIB)} MiB/s; memory-heavy training may be slower even without storage spill.`
    );
  }
  const suitableMinimum = Math.max(1, Math.floor(autoTarget * 0.6));
  const suitableMaximum = Math.max(
    suitableMinimum,
    Math.min(sliderMaximumItems, Math.ceil(autoTarget * 1.75))
  );
  const suitableContextMaximum = quantizeContext(
    Math.max(
      autoContextTokens,
      autoContextTokens * 1.2,
      autoContextTokens + Math.floor(contextFloorTokens / 4)
    )
  );
  if (request.mode === "manual" && selectedItems > suitableMaximum) {
    warnings.push(
      "This working-context selection is above the device's suitable range and is likely to be slower than Auto."
    );
  }
  if (
    request.mode === "manual" &&
    selectedContextTokens > suitableContextMaximum
  ) {
    warnings.push(
      "This context is above the resource-derived green range. Cold attention may use storage and leave less capacity for training, media and hot neural patterns."
    );
  }
  // The bounded packed-update block needs real training RAM. Activation
  // tensors and temporary gradients are budgeted separately below; none is
  // counted as a resident learned FP32 master or optimizer copy.
  const additionalTrainingStateBytes = architectureProfile
    ? Math.max(
        0,
        architectureProfile.minimumTrainingStateBytes -
          architectureProfile.residentInferenceStateBytes
      )
    : 0;
  const trainingHeadroomBytes = Math.max(
    0,
    safeRamBytes -
      residentFoundationBytes -
      selectedContextResidentBytes -
      Math.min(selectedItems, residentItems) * ramBytesPerMemoryItem -
      additionalTrainingStateBytes
  );
  const requestedWindowTokens = TRAINING_WINDOW_BY_TIER[hardwareTier];
  const requestedPhysicalBatch = hardwareTier === "workstation"
    ? 4
    : hardwareTier === "gpu"
      ? 2
      : 1;
  // A conservative cross-backend activation allowance. The worker performs a
  // second live RAM/VRAM measurement using the exact mutable parameter count.
  const activationBytesPerSample = 512 * 1024 * 1024;
  const physicalBatchSize = Math.max(
    1,
    Math.min(
      requestedPhysicalBatch,
      Math.floor(trainingHeadroomBytes / activationBytesPerSample) || 1
    )
  );
  const windowScale = Math.max(
    0.125,
    Math.min(1, trainingHeadroomBytes / Math.max(1, 2 * activationBytesPerSample))
  );
  const windowTokens = Math.max(
    64,
    Math.min(requestedWindowTokens, Math.floor(requestedWindowTokens * windowScale))
  );
  const gradientAccumulation = Math.max(
    1,
    Math.ceil((requestedPhysicalBatch * 8) / physicalBatchSize)
  );
  const minimumScratchIntervalSeconds =
    storageClass === "slow-storage"
      ? 3600
      : storageClass === "moderate-storage"
        ? 900
        : 300;

  return {
    schemaVersion: 1,
    diskSpace,
    ...(architectureProfile
      ? {
          architecture: {
            ...architectureProfile
          }
        }
      : {}),
    mode: request.mode,
    allowed: blockers.length === 0,
    blockers,
    warnings,
    hardwareTier,
    selectedItems,
    selectedItemsText: String(selectedItems),
    sliderMaximumItems,
    sliderMaximumItemsText: String(sliderMaximumItems),
    suitableRange: {
      minimumItems: suitableMinimum,
      autoItems: autoTarget,
      maximumItems: suitableMaximum,
      extendedItems: extendedTarget
    },
    context: {
      selectedTokens: selectedContextTokens,
      floorTokens: contextFloorTokens,
      autoTokens: autoContextTokens,
      extendedTokens: extendedContextTokens,
      maximumTokens: contextMaximumTokens,
      suitableMinimumTokens: contextFloorTokens,
      suitableMaximumTokens: suitableContextMaximum,
      evidence: {
        source: "measured-device-theoretical-allocation",
        modelHiddenSize,
        modelLayers,
        estimatedResidentMemoryItemBytes: ramBytesPerMemoryItem,
        estimatedPagedMemoryItemBytes: diskBytesPerMemoryItem,
        reservedTrainingTransferBytes: transferBudgetBytes,
        modelContextLimitTokens,
        safeRamAfterModelBytes,
        contextResidentBudgetBytes,
        estimatedKvActivationBytesPerToken,
        contextWorkspaceMultiplier: 1,
        minimumContextWorkspaceBytes,
        selectedContextResidentBytes,
        selectedContextSpillBytes,
        contextOffloadBudgetBytes,
        retainedDeviceContextBytes,
        selectedContextDeviceBytes,
        contextCheckpointReserveBytes,
        contextCheckpointCapacityBytes,
        maximumPackedOwnerBytes,
        nonpackedControlBytes,
        manualCoreRebalance,
        retainedContextPageBytes,
        residentContextMaximumTokens,
        contextSpillCapacityBytes,
        ...(selectedDriveCapacity ? { selectedDriveCapacityBytes: selectedDriveCapacity.capacityBytes,
          selectedDriveAvailableBytes: selectedDriveCapacity.currentlyAvailableBytes } : {}),
        residentTokenBytesPerToken,
        contextMetadataBytesPerToken,
        workspaceResidentBytes,
        pageTokens,
        pagedKvBytesPerToken,
        diskBlockSizeBytes,
        acceleratorAvailable,
        storageClass,
        measuredStorageBytesPerSecond: benchmark.storageBytesPerSecond,
        autoContextRamFraction
      }
    },
    resources: {
      ...resources,
      ramReserveBytes,
      mandatoryFreeDiskBytes: diskReserveBytes,
      checkpointHeadroomBytes,
      usableDiskAfterReserveBytes,
      modelBytes,
      ...(architectureProfile
        ? {
            modelParameterCount:
              architectureProfile.exactLogicalParameterCount,
            modelParameterCountBasis:
              architectureProfile.parameterCountBasis,
            modelStorageBasis: architectureProfile.checkpointByteBasis
          }
        : {}),
      modelWorkingSetBytes,
      modelMetadataReserveBytes: architectureProfile?.pythonOwnerMetadataReserveBytes ?? runtimeMetadataReserveBytes,
      estimatedResidentModelBytes,
      fullFoundationRuntimeBytes,
      minimumLayerResidencyBytes,
      residentFoundationBytes,
      runtimeOverheadBytes: BASE_RUNTIME_MEMORY_BYTES,
      configuredMemorySpillBytes,
      safeRamPoolBytes,
      availableSafeRamBytes,
      systemRamBudgetBytes,
      currentOmniAvailableBytes,
      currentOmniShortfallBytes,
      systemRamMode,
      systemRamSharePercent,
      autoSystemRamSharePercent,
      storagePoolMode,
      sharedStoragePoolBytes,
      requiredStoragePoolBytes,
      maximumStoragePoolBytes,
      trainingSourceBytes,
      trainingScratchBytes,
      futureGrowthHeadroomBytes
    },
    offload: {
      required: offloadRequired,
      residentMemoryItems: Math.min(selectedItems, residentItems),
      pagedMemoryItems: memorySpillItems,
      modelSpillBytes,
      modelOffloadScratchBytes,
      memorySpillBytes,
      contextSpillBytes: selectedContextSpillBytes,
      estimatedSlowdownPercent,
      benchmark: {
        measuredAt: benchmark.measuredAt,
        sampleBytes: benchmark.sampleBytes,
        memoryBytesPerSecond: benchmark.memoryBytesPerSecond,
        storageBytesPerSecond: benchmark.storageBytesPerSecond,
        cacheHit: benchmark.cacheHit
      }
    },
    semantics: {
      unit: "recurrent-paged-memory-item",
      denseAttentionClaim: false,
      contextFloorTokens,
      contextPagedToStorage: selectedContextSpillBytes > 0,
      capacityPersistsAcrossPressure: true,
      storagePoolShareRule: "largest-brain-not-sum",
      hotRamPriority: [
        "currently firing",
        "frequently used",
        "stable or rooted",
        "unfinished activity"
      ],
      spillOrder: [
        "cold scratch trail",
        "replay batches",
        "cold metaplasticity metadata",
        "inactive working patterns",
        "cold attention K/V pages"
      ]
    },
    training: {
      policy: "ram-first-adaptive-streaming",
      physicalBatchSize,
      gradientAccumulation,
      windowTokens,
      allSourceBytesVisited: true,
      scratchMode: "emergency-checkpoint-only",
      scratchUsedAsVirtualRam: false,
      sequentialScratchWrites: true,
      minimumScratchIntervalSeconds,
      storageClass,
      measuredStorageBytesPerSecond: benchmark.storageBytesPerSecond
    }
  };
}

async function defaultResourceSnapshot(path: string): Promise<ResourceSnapshot> {
  const disk = await statfs(path);
  const memory = await measuredSystemMemory();
  return {
    ...memory,
    diskTotalBytes: disk.blocks * disk.bsize,
    diskFreeBytes: disk.bavail * disk.bsize,
    diskBlockSizeBytes: disk.bsize
  };
}

function validBenchmark(
  value: unknown,
  now: Date,
  volumeId: string
): value is StorageBenchmark {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const record = value as Record<string, unknown>;
  const measured = Date.parse(String(record.measuredAt ?? ""));
  return (
    Number.isFinite(measured) &&
    measured <= now.getTime() + 5 * 60 * 1000 &&
    now.getTime() - measured <= BENCHMARK_MAX_AGE_MS &&
    record.schema === BENCHMARK_SCHEMA &&
    record.volumeId === volumeId &&
    record.storageMetric === "durable-sequential-write-fsync-median" &&
    record.sampleBytes === BENCHMARK_BYTES &&
    typeof record.memoryBytesPerSecond === "number" &&
    Number.isFinite(record.memoryBytesPerSecond) &&
    record.memoryBytesPerSecond > 0 &&
    typeof record.storageBytesPerSecond === "number" &&
    Number.isFinite(record.storageBytesPerSecond) &&
    record.storageBytesPerSecond > 0
  );
}

async function boundedStorageBenchmark(root: string, now: Date): Promise<StorageBenchmark> {
  const volumeId = String((await stat(root)).dev);
  const cachePath = join(root, ".resource-benchmark-v2.json");
  try {
    const cached = JSON.parse(await readFile(cachePath, "utf8")) as unknown;
    if (validBenchmark(cached, now, volumeId)) return { ...cached, cacheHit: true };
  } catch {
    // A cache miss or corrupt diagnostic cache is safe to replace.
  }

  const sample = Buffer.alloc(BENCHMARK_BYTES, 0xa5);
  const memoryTarget = Buffer.allocUnsafe(BENCHMARK_BYTES);
  const median = (values: number[]): number =>
    [...values].sort((left, right) => left - right)[Math.floor(values.length / 2)] ?? 1;
  const memoryRates: number[] = [];
  for (let run = 0; run < BENCHMARK_RUNS; run += 1) {
    const memoryStart = performance.now();
    for (let index = 0; index < 8; index += 1) sample.copy(memoryTarget);
    const elapsed = Math.max(0.1, performance.now() - memoryStart);
    memoryRates.push((BENCHMARK_BYTES * 8 * 1000) / elapsed);
  }
  // This is a bounded buffer-copy comparison used only by the slowdown
  // estimate; it is not presented as a full DRAM bandwidth benchmark.
  const memoryBytesPerSecond = median(memoryRates);
  const storageRates: number[] = [];
  for (let run = 0; run < BENCHMARK_RUNS; run += 1) {
    const probePath = join(root, `.resource-probe-${randomUUID()}.tmp`);
    try {
      const handle = await open(probePath, "wx", 0o600);
      const storageStart = performance.now();
      try {
        await handle.writeFile(sample);
        await handle.sync();
      } finally {
        await handle.close();
      }
      const elapsed = Math.max(0.1, performance.now() - storageStart);
      storageRates.push((BENCHMARK_BYTES * 1000) / elapsed);
    } finally {
      await rm(probePath, { force: true }).catch(() => undefined);
    }
  }
  const storageBytesPerSecond = median(storageRates);
  const result: StorageBenchmark = {
    measuredAt: now.toISOString(),
    sampleBytes: BENCHMARK_BYTES,
    memoryBytesPerSecond,
    storageBytesPerSecond,
    cacheHit: false,
    volumeId,
    storageMetric: "durable-sequential-write-fsync-median"
  };
  await writeFile(
    cachePath,
    JSON.stringify({ schema: BENCHMARK_SCHEMA, ...result, cacheHit: false }, null, 2),
    { encoding: "utf8", mode: 0o600 }
  ).catch(() => undefined);
  return result;
}

interface CheckpointModelProfile {
  modelBytes?: number;
  runtimeMetadataReserveBytes?: number;
  hardwareTier?: HardwareTier;
  modelHiddenSize?: number;
  modelLayers?: number;
  modelWorkspaceSlots?: number;
  workingMemoryItems?: number;
  modelContextGeometry?: NativeContextGeometry;
  maximumNativeBufferBytes?: number;
  maximumPackedOwnerBytes?: number;
  nonpackedControlBytes?: number;
  modelUpdateScratchBytes?: number;
}

async function checkpointModelProfile(
  brainDirectory: string
): Promise<CheckpointModelProfile> {
  const metadataPath = join(brainDirectory, "engine", "brain.json");
  let metadataRead = false;
  try {
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as {
      connected_regions?: unknown;
      config?: {
        hardware_tier?: unknown;
        d_model?: unknown;
        n_layers?: unknown;
        working_memory_slots?: unknown;
      };
    };
    metadataRead = true;
    const recordedTier = String(metadata.config?.hardware_tier ?? "");
    const hardwareTier = ["micro", "personal", "gpu", "workstation"].includes(
      recordedTier
    )
      ? recordedTier as HardwareTier
      : undefined;
    const paths = [
      join(brainDirectory, "engine", "core.safetensors"),
      join(brainDirectory, "engine", "plasticity.safetensors")
    ];
    let tensors: ReadonlyMap<string, SavedTensorShape> = new Map();
    const memory = await nativeCheckpointMemory(paths, (inventory) => { tensors = inventory; });
    const modelContextGeometry = metadata.connected_regions === undefined ? undefined
      : await connectedResourceProfile(join(brainDirectory, "engine"), tensors);
    const total = memory.projectedCheckpointBytes;
    const hidden = Number(metadata.config?.d_model);
    const layers = Number(metadata.config?.n_layers);
    const items = Number(metadata.config?.working_memory_slots);
    let maximumNativeBufferBytes = 0, modelUpdateScratchBytes = 0;
    for (const [name, tensor] of tensors) if (/^(decoder|memory_bridge|idea_adapter|liquid|modalities|router)\./.test(name)) {
      maximumNativeBufferBytes = Math.max(maximumNativeBufferBytes, tensor.bytes);
      const packed = /(_packed_forward_weight|_packed_forward_bias|_packed_weights)$/.exec(name);
      if (packed) {
        const prefix = name.slice(0, -packed[1]!.length);
        const state = tensors.get(prefix + (packed[1] === "_packed_forward_bias" ? "_bias_connection_state" : "_connection_state"));
        const rows = tensor.shape[0]!, width = state?.shape[1] ?? tensor.shape[1]! * 4;
        if (state && (state.shape.length !== 2 || state.shape[0] !== rows || Math.ceil(width / 4) !== tensor.shape[1])) throw new Error("Saved native connection dimensions disagree with packed signs.");
        modelUpdateScratchBytes = Math.max(modelUpdateScratchBytes, packedLearningScratchBytes(rows, width));
      }
    }
    return {
      ...(total > 0 ? { modelBytes: total } : {}),
      runtimeMetadataReserveBytes: memory.runtimeMetadataReserveBytes,
      modelContextGeometry,
      maximumNativeBufferBytes,
      maximumPackedOwnerBytes: checkpointPackedOwnerMinimum(tensors),
      nonpackedControlBytes: checkpointNonpackedControls(tensors),
      modelUpdateScratchBytes,
      ...(hardwareTier ? { hardwareTier } : {}),
      ...(Number.isSafeInteger(hidden) && hidden > 0
        ? { modelHiddenSize: hidden }
        : {}),
      ...(Number.isSafeInteger(layers) && layers > 0
        ? { modelLayers: layers }
        : {}),
      ...(Number.isSafeInteger(items) && items > 0
        ? { workingMemoryItems: items, modelWorkspaceSlots: Math.max(8, Math.floor(items / 4)) }
        : {})
    };
  } catch (error) {
    if (metadataRead || (error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return {};
  }
}

interface ResourcePlannerOptions {
  retainedContextPools?: RetainedContextPool[];
  spillStorage?: AuthorizedSpillStorageSnapshot;
  allowStorageSpill?: boolean;
  hardwareTier?: HardwareTier;
  modelBytes?: number;
  brainDirectory?: string;
  config?: BrainConfig;
  acceleratorAvailable?: boolean;
  modelContextLimitTokens?: number;
  modelHiddenSize?: number;
  modelLayers?: number;
  modelContextGeometry?: NativeContextGeometry;
  maximumNativeBufferBytes?: number;
  maximumPackedOwnerBytes?: number;
  nonpackedControlBytes?: number;
  modelUpdateScratchBytes?: number;
  modelWorkspaceSlots?: number;
  fixedWorkingMemoryItems?: number;
  minimumStoragePoolBytes?: number;
  admissionScope?: "capacity" | "existing-runtime";
  enforceContextFloor?: boolean;
  nativeSizingMode?: "ram-first" | "physical-capacity" | "balanced-measured";
  nativeComputeDevice?: string;
  nativePrimitiveWorkBudgetMicroseconds?: number;
}

export class ResourcePlanner {
  readonly root: string;
  private readonly dependencies: ResourcePlannerDependencies;

  constructor(root: string, dependencies: ResourcePlannerDependencies = {}) {
    this.root = resolve(root);
    this.dependencies = dependencies;
  }

  private async resources(): Promise<ResourceSnapshot> {
    return this.dependencies.readResources
      ? this.dependencies.readResources()
      : defaultResourceSnapshot(this.root);
  }

  private async benchmark(): Promise<StorageBenchmark> {
    const now = this.dependencies.now?.() ?? new Date();
    return this.dependencies.benchmark
      ? this.dependencies.benchmark()
      : boundedStorageBenchmark(this.root, now);
  }

  async measureNativeCompute(request: NativeComputeMeasurementRequest): Promise<NativeProjectionComputeProfile | undefined> {
    try {
      const profile = await this.dependencies.measurementCollector?.(request);
      return profile ? validateNativeProjectionComputeProfile(profile) : undefined;
    } catch {
      return undefined;
    }
  }

  /** Select a reviewed RAM ceiling without constructing a hypothetical model.
   * Connected merges use this before native allocation; it is not a fit proof.
   */
  async reviewedRamEnvelope(config: BrainConfig, hardwareTier?: HardwareTier): Promise<{
    systemRamBudgetBytes: number; systemRamMode: "auto" | "manual"; systemRamSharePercent: number;
    totalMemoryBytes: number; ramReserveBytes: number; safeRamPoolBytes: number; availableSafeRamBytes: number;
    hardwareTier: HardwareTier; modelFitEvaluated: false;
  }> {
    const resources = await this.resources();
    if (!Number.isSafeInteger(resources.totalMemoryBytes) || resources.totalMemoryBytes <= 0
      || !Number.isSafeInteger(resources.availableMemoryBytes) || resources.availableMemoryBytes < 0) throw new Error("RAM capacity measurement is unavailable.");
    const tier = hardwareTier ?? config.nativeArchitecture?.hardwareTier ?? tierForConfig(config);
    if (!(tier in AUTO_SYSTEM_RAM_SHARE_BY_TIER)) throw new Error("Invalid reviewed hardware tier.");
    const selected = config.deviceResources === undefined ? undefined : validateDeviceResourceSettings(config.deviceResources);
    const ram = selected?.allocations.find((entry) => entry.deviceId === "ram:system");
    const mode = ram?.mode ?? config.systemRamMode;
    if (mode !== "auto" && mode !== "manual") throw new Error("Invalid reviewed RAM mode.");
    const manual = ram?.percent ?? config.systemRamSharePercent;
    if (mode === "manual" && (!Number.isFinite(manual) || manual < 30 || manual > 100)) throw new Error("Reviewed RAM share must be 30–100 percent.");
    const speed = roundedStorageBytesPerSecond(config.storageBytesPerSecond);
    const auto = Math.min(90, AUTO_SYSTEM_RAM_SHARE_BY_TIER[tier] + (speed > 0 && speed < 100 * MIB ? 10 : speed > 0 && speed < 500 * MIB ? 5 : 0));
    const share = mode === "manual" ? manual : auto;
    const reserve = adaptiveRamReserve(resources.totalMemoryBytes);
    const safe = Math.max(0, resources.totalMemoryBytes - reserve);
    return { systemRamBudgetBytes: Math.floor(safe * share / 100), systemRamMode: mode, systemRamSharePercent: share,
      totalMemoryBytes: resources.totalMemoryBytes, ramReserveBytes: reserve, safeRamPoolBytes: safe,
      availableSafeRamBytes: Math.max(0, Math.min(resources.totalMemoryBytes, resources.availableMemoryBytes) - reserve),
      hardwareTier: tier, modelFitEvaluated: false };
  }

  async plan(request: WorkingMemoryPlanRequest, options: ResourcePlannerOptions = {}): Promise<WorkingMemoryResourcePlan> {
    const supplied = request.deviceResources ?? options.config?.deviceResources;
    const settings = supplied === undefined ? undefined : validateDeviceResourceSettings(supplied);
    const ram = settings?.allocations.find((entry) => entry.deviceId === "ram:system");
    const selectedRequest = ram ? { ...request, systemRamMode: ram.mode, systemRamSharePercent: ram.mode === "manual" ? ram.percent : undefined } : request;
    if (!settings) return this.planCore(selectedRequest, options);
    let base: WorkingMemoryResourcePlan | undefined;
    try {
      if (!this.dependencies.deviceTopology) throw new Error("Selected device allocations await hardware discovery.");
      const topology = await this.dependencies.deviceTopology(settings);
      let spillStorage: AuthorizedSpillStorageSnapshot | undefined;
      if (settings.allowStorageSpill && settings.storageDirectories.length > 0) {
        if (!this.dependencies.spillCapacity) throw new Error("Selected drives await local spill authorization.");
        spillStorage = await this.dependencies.spillCapacity(settings, topology, { brainId: request.brainId,
          selectionId: request.spillPoolSelectionId });
      }
      base = await this.planCore(selectedRequest, { ...options, spillStorage, allowStorageSpill: settings.allowStorageSpill });
      const workspaceBytes = base.context.evidence.workspaceResidentBytes;
      const tokenIndexBytes = base.context.evidence.residentTokenBytesPerToken;
      const pageMetadataBytes = base.context.evidence.contextMetadataBytesPerToken;
      if (workspaceBytes === undefined || tokenIndexBytes === undefined || pageMetadataBytes === undefined) throw new Error("The saved context geometry is unavailable for device placement.");
      const placement = (plan: WorkingMemoryResourcePlan) => planDeviceResources(topology, settings, {
        modelBytes: plan.resources.modelBytes,
        fixedRuntimeBytes: plan.resources.runtimeOverheadBytes + (plan.resources.modelMetadataReserveBytes ?? 0)
          + workspaceBytes + (plan.architecture?.packedUpdateScratchBytes ?? 0)
          + (plan.architecture?.packedForwardTensorReserveBytes ?? 0),
        contextBytesPerToken: plan.context.evidence.estimatedKvActivationBytesPerToken + tokenIndexBytes + pageMetadataBytes,
        requestedContextTokens: plan.context.selectedTokens,
        systemRamBudgetBytes: plan.resources.systemRamBudgetBytes,
        diskReserveBytes: plan.resources.mandatoryFreeDiskBytes
      });
      let selected = placement(base);
      const minimums: Record<string, RetainedContextMinimum> = {};
      for (const device of topology.devices) {
        if (!device.executionDevice) continue;
        const compute = device.nativeComputeSupported || settings.regionalExecution?.enabled
          && settings.regionalExecution.routes.some(route => route.deviceId === device.id);
        if (!compute && base.context.evidence.nonpackedControlBytes !== undefined) minimums[device.executionDevice] = {
          packedOwnerBytes: 0, controlBytes: base.context.evidence.nonpackedControlBytes,
          transferBytes: (base.context.evidence.retainedContextPageBytes ?? 0) * 4 };
        else if (base.context.evidence.maximumPackedOwnerBytes !== undefined && base.context.evidence.nonpackedControlBytes !== undefined) {
          minimums[device.executionDevice] = { packedOwnerBytes: base.context.evidence.maximumPackedOwnerBytes,
            controlBytes: base.context.evidence.nonpackedControlBytes,
            transferBytes: base.context.evidence.reservedTrainingTransferBytes };
        }
      }
      const pools = retainedContextPools(topology, selected.budgets, { manual: request.mode === "manual", minimums,
        pageBytes: base.context.evidence.retainedContextPageBytes });
      if (pools.length) {
        base = await this.planCore(selectedRequest, { ...options, spillStorage, allowStorageSpill: settings.allowStorageSpill,
          retainedContextPools: pools });
        selected = placement(base);
      }
      selected.admittedContextMaximumTokens = base.context.maximumTokens;
      if (spillStorage) selected.appliedStorage = {
        largestSharedPoolBytes: Math.max(spillStorage.largestSharedPoolBytes, base.resources.sharedStoragePoolBytes),
        chargedBytes: spillStorage.sharedUsedBytes, pendingBytes: spillStorage.sharedPendingBytes,
        groups: spillStorage.groups.map((group) => ({ memoryGroupId: group.physicalGroup, ceilingBytes: group.ceilingBytes,
          availableAfterReserveBytes: group.availableBytes, chargedBytes: group.allocatedBytes,
          pendingBytes: group.pendingBytes, unmaterializedPromiseBytes: group.unmaterializedPromiseBytes }))
      };
      return { ...base, deviceResources: selected, allowed: base.allowed && selected.executionBlockers.length === 0,
        blockers: [...base.blockers, ...selected.executionBlockers] };
    } catch (error) {
      base ??= await this.planCore(selectedRequest, { ...options, allowStorageSpill: settings.allowStorageSpill });
      return { ...base, allowed: false, blockers: [...base.blockers,
        `Device resource measurement is unavailable: ${error instanceof Error ? error.message : String(error)}`] };
    }
  }

  private async planCore(
    request: WorkingMemoryPlanRequest,
    options: ResourcePlannerOptions = {}
  ): Promise<WorkingMemoryResourcePlan> {
    const checkpoint = options.brainDirectory
      ? await checkpointModelProfile(options.brainDirectory)
      : {};
    const hardwareTier =
      options.hardwareTier ??
      checkpoint.hardwareTier ??
      options.config?.nativeArchitecture?.hardwareTier ??
      tierForConfig(options.config);
    const systemRamMode = request.systemRamMode ?? options.config?.systemRamMode ?? "auto";
    const systemRamSharePercent = request.systemRamSharePercent ??
      (systemRamMode === "manual" ? options.config?.systemRamSharePercent : undefined);
    const storagePoolMode =
      request.storagePoolMode ?? options.config?.storagePoolMode ?? "auto";
    const storagePoolBytes = request.storagePoolBytes ??
      (storagePoolMode === "manual" && options.config?.storagePoolBytes
        ? String(options.config.storagePoolBytes)
        : undefined);
    const sharedInput = {
      retainedContextPools: options.retainedContextPools,
      spillStorage: options.spillStorage,
      allowStorageSpill: options.allowStorageSpill,
      request: {
        ...request,
        systemRamMode,
        ...(systemRamSharePercent !== undefined ? { systemRamSharePercent } : {}),
        storagePoolMode,
        ...(storagePoolBytes !== undefined ? { storagePoolBytes } : {})
      },
      resources: await this.resources(),
      benchmark: await this.benchmark(),
      hardwareTier,
      acceleratorAvailable:
        request.acceleratorAvailable ?? options.acceleratorAvailable,
      modelContextLimitTokens: options.modelContextLimitTokens,
      modelHiddenSize: options.modelHiddenSize ?? checkpoint.modelHiddenSize,
      modelLayers: options.modelLayers ?? checkpoint.modelLayers,
      modelContextGeometry: options.modelContextGeometry ?? checkpoint.modelContextGeometry,
      maximumNativeBufferBytes: options.maximumNativeBufferBytes ?? checkpoint.maximumNativeBufferBytes,
      maximumPackedOwnerBytes: options.maximumPackedOwnerBytes ?? checkpoint.maximumPackedOwnerBytes,
      nonpackedControlBytes: options.nonpackedControlBytes ?? checkpoint.nonpackedControlBytes,
      modelUpdateScratchBytes: options.modelUpdateScratchBytes ?? checkpoint.modelUpdateScratchBytes,
      modelWorkspaceSlots: options.modelWorkspaceSlots ?? checkpoint.modelWorkspaceSlots ??
        (options.brainDirectory && options.config ? options.config.nativeArchitecture?.shape.workspaceLatents ?? Math.max(8, Math.floor(options.config.workingMemorySlots / 4)) : undefined),
      fixedWorkingMemoryItems: options.fixedWorkingMemoryItems ?? checkpoint.workingMemoryItems ??
        (options.brainDirectory ? options.config?.workingMemorySlots : undefined),
      minimumStoragePoolBytes:
        options.minimumStoragePoolBytes ??
        (storagePoolMode === "auto"
          ? options.config?.storagePoolBytes
          : undefined),
      admissionScope: options.admissionScope,
      enforceContextFloor: options.enforceContextFloor
    };
    const recordedModelBytes = options.modelBytes ?? checkpoint.modelBytes;
    if (recordedModelBytes !== undefined) {
      // Explicit bytes and materialized checkpoints are legacy/import/runtime
      // facts. Never relabel them as a fresh ground-up architecture estimate.
      return planWorkingMemory({
        ...sharedInput,
        modelBytes: recordedModelBytes,
        runtimeMetadataReserveBytes: checkpoint.runtimeMetadataReserveBytes
      });
    }

    if (options.brainDirectory && options.config) {
      // A saved identity without materialized size facts is not authorization
      // to size/reseal a fresh recommended-tier cortex. Use its existing
      // explicit descriptor, or its legacy recorded tier/item geometry.
      const descriptor = options.config.nativeArchitecture;
      const legacy = groundUpArchitectureProfile(hardwareTier, options.config.workingMemorySlots);
      // The archived descriptor remains unchanged. Unit-strength migration of
      // a native v1 checkpoint needs the current connection-state allocation.
      const runtimeInventory = descriptor ? nativeCoreInventory(descriptor.shape,
        descriptor.sizing.routerStorageLayout as "block-sparse-v1" | undefined) : undefined;
      return planWorkingMemory({
        ...sharedInput,
        modelBytes: runtimeInventory ? runtimeInventory.packedLearnedStateBytes! + runtimeInventory.staticNonweightTensorBytes : legacy.checkpointTensorBytes,
        runtimeMetadataReserveBytes: runtimeInventory ? packedOwnerMetadataReserve(runtimeInventory.packedOwners) : legacy.pythonOwnerMetadataReserveBytes,
        modelHiddenSize: descriptor?.shape.dModel ?? sharedInput.modelHiddenSize ?? legacy.dModel,
        modelLayers: descriptor?.shape.layers ?? sharedInput.modelLayers ?? legacy.layers,
      });
    }

    // Before construction there is no checkpoint to stat. Resolve the exact
    // project-owned parameter inventory from the same tier/workspace contract
    // the worker is required to instantiate. A second pass makes constrained
    // Auto plans exact when the first pass safely lowers the requested items.
    let profileItems = prospectiveGroundUpWorkingMemoryItems(
      request,
      hardwareTier,
      options.config
    );
    const bootstrap = planWorkingMemory({ ...sharedInput,
      architectureProfile: groundUpArchitectureProfile(hardwareTier, profileItems),
      modelBytes: groundUpArchitectureProfile(hardwareTier, profileItems).checkpointTensorBytes });
    if (request.mode !== "manual" && bootstrap.selectedItems > 0) profileItems = bootstrap.selectedItems;
    const sizingInput = {
      hardwareTier,
      selectedSystemRamBudgetBytes: bootstrap.resources.systemRamBudgetBytes,
      runtimeBaselineReserveBytes: bootstrap.resources.runtimeOverheadBytes,
      selectedStoragePoolBytes: bootstrap.resources.sharedStoragePoolBytes,
      measuredStorageBytesPerSecond: sharedInput.benchmark.storageBytesPerSecond,
      estimatedTrainingSourceBytes: bootstrap.resources.trainingSourceBytes,
      acceleratorAvailable: sharedInput.acceleratorAvailable ?? false,
      liquidMode: "cfc" as const
    };
    const baselineReferenceShape = CONTEXT_MODEL_SHAPE_BY_TIER[hardwareTier];
    const baselineResidual = Math.max(0, sizingInput.selectedSystemRamBudgetBytes - sizingInput.runtimeBaselineReserveBytes);
    const baselineActivityPartition = Math.floor(baselineResidual * 0.30);
    const baselineContextTokens = Math.max(CONTEXT_FLOOR_TOKENS_BY_TIER[hardwareTier],
      Math.floor(baselineActivityPartition * 0.70 /
        (8 * baselineReferenceShape.hiddenSize * baselineReferenceShape.layers + 96 + 16 * baselineReferenceShape.layers) / 256) * 256);
    const baselineResidentItems = Math.max(1, Math.floor(baselineActivityPartition * 0.20 /
      (Math.max(4096, 4 * baselineReferenceShape.hiddenSize + 512) + 2 * baselineReferenceShape.hiddenSize + 1)));
    // User selected RAM-first headroom, not a primitive-time ceiling. Optional
    // experiments remain main-only; they cannot silently become Auto policy.
    const workBudget = options.nativePrimitiveWorkBudgetMicroseconds;
    const hasWorkBudget = Number.isSafeInteger(workBudget) && Number(workBudget) > 0;
    const measured = options.nativeSizingMode === "balanced-measured" && hasWorkBudget
      ? await this.measureNativeCompute({ device: options.nativeComputeDevice ?? "cpu",
          ramBudgetBytes: bootstrap.resources.systemRamBudgetBytes, hardwareTier })
      : undefined;
    let measurementWarning = options.nativeSizingMode === "balanced-measured" && (!hasWorkBudget || !measured)
      ? "Optional packed-primitive measurement/work budget is unavailable; the shipped native shape policy remains available. No neural throughput or quality is claimed."
      : undefined;
    const freshProfile = (items: number): GroundUpArchitectureProfile => {
      const input = { ...sizingInput, workingMemoryItems: items,
        estimatedNeuralGrowthReserveBytes: bootstrap.resources.futureGrowthHeadroomBytes,
        trainingScratchReserveBytes: bootstrap.resources.trainingScratchBytes };
      if (options.nativeSizingMode === "physical-capacity") return capacityDerivedNativeArchitectureProfile(input);
      if (options.nativeSizingMode === "balanced-measured" && measured && hasWorkBudget) {
        const candidate = balancedMeasuredNativeArchitectureProfile(input, measured, workBudget!);
        if (candidate.nativeArchitecture?.sizing.fitsPrimitiveWorkProxyBudget !== false) return candidate;
        measurementWarning = "The explicit primitive-work proxy budget does not admit the minimum candidate shape; the shipped native policy is retained, not blocked. This is not a measured neural-latency result.";
      }
      return ramFirstNativeArchitectureProfile({ ...input, baselineContextTokens,
        baselineResidentWorkingMemoryItems: request.mode === "auto" ? items : Math.min(items, baselineResidentItems) });
    };
    let planned: WorkingMemoryResourcePlan | undefined;
    const seenItems: number[] = [];
    for (let pass = 0; pass < 8; pass += 1) {
      seenItems.push(profileItems);
      const architectureProfile = freshProfile(profileItems);
      planned = planWorkingMemory({
        ...sharedInput,
        modelBytes: architectureProfile.checkpointTensorBytes,
        modelHiddenSize: architectureProfile.dModel,
        modelLayers: architectureProfile.layers,
        architectureProfile
      });
      if (architectureProfile.nativeArchitecture) {
        const fits = architectureProfile.nativeArchitecture.sizing.fitsPolicyTarget !== false;
        planned = { ...planned, nativeArchitecture: architectureProfile.nativeArchitecture,
          allowed: planned.allowed && fits,
          blockers: fits ? planned.blockers : [...planned.blockers, "The minimum native shape does not fit the selected core/storage sizing reserve."],
          warnings: [...planned.warnings,
            "Native counts are exact shape inventory; resource/work reserves and any primitive-to-wide-shape extrapolation are estimates, not neural throughput or quality proof.",
            ...(measurementWarning ? [measurementWarning] : [])] };
      }
      if (
        planned.selectedItems < 1 ||
        planned.selectedItems === profileItems
      ) {
        return planned;
      }
      profileItems = planned.selectedItems;
    }
    // Integer Auto feedback can alternate by a few items. Resolve once to
    // the lowest observed safe selection and recompute the *same* exact shape
    // used by the worker, instead of returning a descriptor for another count.
    const stableItems = Math.max(1, Math.min(...seenItems, planned!.selectedItems));
    const architectureProfile = freshProfile(stableItems);
    const stable = planWorkingMemory({ ...sharedInput,
      request: { ...sharedInput.request, mode: "manual", requestedItems: String(stableItems) },
      modelBytes: architectureProfile.checkpointTensorBytes, modelHiddenSize: architectureProfile.dModel,
      modelLayers: architectureProfile.layers, architectureProfile });
    const fits = architectureProfile.nativeArchitecture?.sizing.fitsPolicyTarget !== false;
    return { ...stable, mode: request.mode,
      allowed: stable.allowed && fits,
      blockers: fits ? stable.blockers : [...stable.blockers, "The minimum native shape does not fit the selected core/storage sizing reserve."],
      nativeArchitecture: architectureProfile.nativeArchitecture,
      warnings: [...stable.warnings, "Auto item rounding was stabilized before sealing the exact native shape.",
        ...(measurementWarning ? [measurementWarning] : [])] };
  }
}
