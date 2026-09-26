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
  groundUpArchitectureProfile,
  type GroundUpArchitectureProfile
} from "./omniArchitectureProfile";
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

async function measuredSystemMemory(): Promise<{
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

const AUTO_MEMORY_ITEMS_BY_TIER: Record<HardwareTier, number> = {
  micro: defaultGroundUpWorkingMemoryItems("micro"),
  personal: defaultGroundUpWorkingMemoryItems("personal"),
  gpu: defaultGroundUpWorkingMemoryItems("gpu"),
  workstation: defaultGroundUpWorkingMemoryItems("workstation")
};

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

// This is a product-level v1 active-context ceiling, not a capability copied
// from an upstream model. Rotary tables are native, lazy OmniCortex state.
export const DEFAULT_MODEL_CONTEXT_LIMIT_TOKENS = 32_768;
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
  request: WorkingMemoryPlanRequest;
  resources: ResourceSnapshot;
  benchmark: StorageBenchmark;
  hardwareTier: HardwareTier;
  modelBytes: number;
  /** Exact project-owned architecture accounting for a new native build. */
  architectureProfile?: GroundUpArchitectureProfile;
  acceleratorAvailable?: boolean;
  modelContextLimitTokens?: number;
  modelHiddenSize?: number;
  modelLayers?: number;
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
  const admissionScope = input.admissionScope ?? "capacity";
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
    (architectureProfile?.residentInferenceStateBytes ?? modelBytes) *
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
  const modelLayers = Math.max(
    1,
    safeWhole(input.modelLayers ?? contextShape.layers)
  );
  // K and V per layer in two-byte runtime precision, plus the current hidden
  // activation. The 4 KiB floor conservatively covers chunked attention score
  // and allocator overhead; cold recurrent pages are counted independently.
  const estimatedKvActivationBytesPerToken = Math.max(
    4_096,
    modelLayers * modelHiddenSize * 4 + modelHiddenSize * 2
  );
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
          architectureProfile.packedUpdateScratchBytes
      )
    : Math.max(512 * MIB, Math.ceil(modelBytes * 1.5) + 128 * MIB);
  const fullFoundationRuntimeBytes =
    Math.max(modelWorkingSetBytes, estimatedResidentModelBytes) +
    BASE_RUNTIME_MEMORY_BYTES;
  const minimumLayerResidencyBytes = architectureProfile
    ? Math.max(
        8 * MIB,
        architectureProfile.fixedControlBufferReserveBytes +
          architectureProfile.packedMetaplasticityReserveBytes +
          architectureProfile.packedWorkspaceTableBytes +
          architectureProfile.packedUpdateScratchBytes +
          Math.ceil(
            (architectureProfile.packedTernaryWeightBytes -
              architectureProfile.packedWorkspaceTableBytes) /
              Math.max(1, architectureProfile.layers)
          )
      )
    : Math.max(
        192 * MIB,
        Math.min(512 * MIB, Math.floor(estimatedResidentModelBytes / 4))
      );
  const minimumFoundationContextTokens = Math.min(
    256,
    modelContextLimitTokens
  );
  const minimumContextWorkspaceBytes = Math.max(
    128 * MIB,
    estimatedKvActivationBytesPerToken *
      minimumFoundationContextTokens *
      2
  );
  const contextFloorWorkspaceBytes = Math.max(
    minimumContextWorkspaceBytes,
    contextFloorTokens * estimatedKvActivationBytesPerToken * 2
  );
  // Slow storage receives more of the already-safe RAM pool so Auto avoids
  // paging. This is still bounded by the OS reserve and the live free pool.
  const autoBaseSharePercent =
    AUTO_SYSTEM_RAM_SHARE_BY_TIER[hardwareTier] +
    (storageClass === "slow-storage" ? 10 : storageClass === "moderate-storage" ? 5 : 0);
  const autoFoundationSharePercent = safeRamPoolBytes > 0
    ? Math.ceil(
      (
        fullFoundationRuntimeBytes + contextFloorWorkspaceBytes
      ) * 100 /
        safeRamPoolBytes
    )
    : 100;
  // Auto can expand to fit the actual packed foundation and runtime while it
  // remains inside the safe pool. A user-selected manual cap stays exact.
  const autoSystemRamSharePercent = Math.min(
    90,
    Math.max(autoBaseSharePercent, autoFoundationSharePercent)
  );
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
  const fullResidentWithFloorFits =
    fullFoundationRuntimeBytes + contextFloorWorkspaceBytes <= safeRamBytes;
  const planningFoundationResidentBytes = fullResidentWithFloorFits
    ? fullFoundationRuntimeBytes
    : minimumResidentFoundationBytes;
  const safeRamAfterModelBytes = Math.max(
    0,
    safeRamBytes - planningFoundationResidentBytes
  );
  const contextMaximumTokens = Math.min(
    modelContextLimitTokens,
    safeWhole(
      safeRamAfterModelBytes /
        Math.max(1, estimatedKvActivationBytesPerToken * 2)
    )
  );
  // Fast storage cannot hold live attention state, but it can page colder
  // replay/assembly pages and therefore lets Auto devote more of the remaining
  // resident envelope to active context. Accelerator availability similarly
  // reduces CPU-side contention without assuming unmeasured VRAM capacity.
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
    safeRamAfterModelBytes * autoContextRamFraction /
      (estimatedKvActivationBytesPerToken * 2)
  );
  const autoContextTokens = contextMaximumTokens < contextFloorTokens
    ? contextMaximumTokens
    : Math.max(contextFloorTokens, measuredAutoContext);
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
  const selectedContextResidentBytes = Math.max(
    minimumContextWorkspaceBytes,
    selectedContextTokens * estimatedKvActivationBytesPerToken * 2
  );
  const residentModelBudgetBytes = Math.max(
    0,
    safeRamBytes - BASE_RUNTIME_MEMORY_BYTES - selectedContextResidentBytes
  );
  const residentModelBytes = Math.min(
    estimatedResidentModelBytes,
    Math.max(minimumLayerResidencyBytes, residentModelBudgetBytes)
  );
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
  const residentMemoryBytes = Math.max(
    0,
    safeRamBytes - residentFoundationBytes - selectedContextResidentBytes
  );
  const diskForPagedMemoryBytes = Math.max(
    0,
    usableDiskAfterReserveBytes - modelOffloadScratchBytes
  );
  const residentItems = safeWhole(
    residentMemoryBytes / RAM_BYTES_PER_MEMORY_ITEM
  );
  const pagedItems = safeWhole(
    diskForPagedMemoryBytes / DISK_BYTES_PER_MEMORY_ITEM
  );
  const sliderMaximumItems = safeWhole(residentItems + pagedItems);

  const autoTarget = Math.min(
    sliderMaximumItems,
    AUTO_MEMORY_ITEMS_BY_TIER[hardwareTier]
  );
  const extendedTarget = Math.min(
    sliderMaximumItems,
    AUTO_MEMORY_ITEMS_BY_TIER[hardwareTier] * 4
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
  const selectedItems =
    request.mode === "manual"
      ? request.requestedItems === undefined
        ? derivedManualItems
        : requestedFits
          ? Number(requestedText)
          : 0
      : request.mode === "extended"
        ? extendedTarget
        : autoTarget;
  const memorySpillItems = Math.max(0, selectedItems - residentItems);
  const memorySpillBytes = memorySpillItems * DISK_BYTES_PER_MEMORY_ITEM;
  const configuredMemorySpillBytes =
    modelOffloadScratchBytes + memorySpillBytes;
  const offloadRequired = modelSpillBytes > 0 || memorySpillItems > 0;
  const trainingSourceBytes = safeWhole(request.trainingSourceBytes ?? 0);
  // Source datasets remain referenced in place; this is transactional replay,
  // staging, and recovery space sized from the selected corpus rather than a
  // duplicate of every source byte.
  const trainingScratchBytes = Math.max(
    512 * MIB,
    Math.ceil(trainingSourceBytes * 0.1),
    Math.ceil(modelBytes * 0.5)
  );
  const futureGrowthHeadroomBytes = Math.max(
    2 * GIB,
    Math.ceil(modelBytes * 1.2),
    Math.ceil(selectedItems * DISK_BYTES_PER_MEMORY_ITEM * 0.5)
  );
  const requiredStoragePoolBytes = safeWhole(
    fixedBuildWriteBytes +
      modelOffloadScratchBytes +
      memorySpillBytes +
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
      checkpointBytes: checkpointWriteBytes,
      maximumWorkingMemorySpillBytes:
        admissionScope === "existing-runtime" ? 0 : memorySpillBytes,
      futureGrowthBytes:
        admissionScope === "existing-runtime" ? 0 : futureGrowthHeadroomBytes,
      operationWriteBytes: admissionScope === "existing-runtime"
        ? 0
        : modelOffloadScratchBytes + trainingScratchBytes
    }
  });
  const maximumStoragePoolBytes = Math.max(
    0,
    resources.diskFreeBytes - diskReserveBytes
  );
  const storagePoolMode = request.storagePoolMode ?? "auto";
  const requestedStoragePoolText = positiveIntegerText(request.storagePoolBytes);
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
        ? checkpointHeadroomBytes
        : fixedBuildWriteBytes + modelOffloadScratchBytes
    );
  const diskFitsSelection =
    !diskSpace.paused;
  const storagePoolFits = admissionScope === "existing-runtime" || (
    sharedStoragePoolBytes >= requiredStoragePoolBytes &&
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
      "The Omni RAM envelope cannot keep one complete neural layer, the selected active context, and the runtime resident at the same time. Increase the RAM cap or reduce active context; storage cannot replace this minimum live tier."
    );
  }
  if (enforceContextFloor && contextMaximumTokens < contextFloorTokens) {
    blockers.push(
      `This live RAM envelope cannot keep the ${contextFloorTokens.toLocaleString()}-token device baseline resident after the model and runtime (resident maximum ${contextMaximumTokens.toLocaleString()} tokens).`
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
      `The requested active context cannot stay resident on this device (safe/model maximum ${contextMaximumTokens.toLocaleString()} tokens).`
    );
  } else if (
    request.mode === "manual" &&
    request.requestedContextTokens !== undefined &&
    selectedContextTokens < contextFloorTokens
  ) {
    const message =
      `This context is below the ${contextFloorTokens.toLocaleString()}-token baseline measured for this device.`;
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
    ? (modelSpillBytes + memorySpillBytes) /
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
      "This active context is above the measured green range. It remains resident, but leaves less RAM for training, media, agents, and hot neural patterns."
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
      Math.min(selectedItems, residentItems) * RAM_BYTES_PER_MEMORY_ITEM -
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
        source: "live-device-model-measurement",
        modelContextLimitTokens,
        safeRamAfterModelBytes,
        contextResidentBudgetBytes: safeRamAfterModelBytes,
        estimatedKvActivationBytesPerToken,
        contextWorkspaceMultiplier: 2,
        minimumContextWorkspaceBytes,
        selectedContextResidentBytes,
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
      contextPagedToStorage: false,
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
        "inactive working patterns"
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
    diskFreeBytes: disk.bavail * disk.bsize
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
  hardwareTier?: HardwareTier;
  modelHiddenSize?: number;
  modelLayers?: number;
}

async function checkpointModelProfile(
  brainDirectory: string
): Promise<CheckpointModelProfile> {
  const metadataPath = join(brainDirectory, "engine", "brain.json");
  try {
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as {
      config?: {
        hardware_tier?: unknown;
        d_model?: unknown;
        n_layers?: unknown;
      };
    };
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
    let total = 0;
    for (const path of paths) total += (await stat(path)).size;
    const hidden = Number(metadata.config?.d_model);
    const layers = Number(metadata.config?.n_layers);
    return {
      ...(total > 0 ? { modelBytes: total } : {}),
      ...(hardwareTier ? { hardwareTier } : {}),
      ...(Number.isSafeInteger(hidden) && hidden > 0
        ? { modelHiddenSize: hidden }
        : {}),
      ...(Number.isSafeInteger(layers) && layers > 0
        ? { modelLayers: layers }
        : {})
    };
  } catch {
    return {};
  }
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

  async plan(
    request: WorkingMemoryPlanRequest,
    options: {
      hardwareTier?: HardwareTier;
      modelBytes?: number;
      brainDirectory?: string;
      config?: BrainConfig;
      acceleratorAvailable?: boolean;
      modelContextLimitTokens?: number;
      modelHiddenSize?: number;
      modelLayers?: number;
      minimumStoragePoolBytes?: number;
      admissionScope?: "capacity" | "existing-runtime";
      enforceContextFloor?: boolean;
    } = {}
  ): Promise<WorkingMemoryResourcePlan> {
    const checkpoint = options.brainDirectory
      ? await checkpointModelProfile(options.brainDirectory)
      : {};
    const hardwareTier =
      options.hardwareTier ??
      checkpoint.hardwareTier ??
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
        modelBytes: recordedModelBytes
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
    let planned: WorkingMemoryResourcePlan | undefined;
    for (let pass = 0; pass < 3; pass += 1) {
      const architectureProfile = groundUpArchitectureProfile(
        hardwareTier,
        profileItems
      );
      planned = planWorkingMemory({
        ...sharedInput,
        modelBytes: architectureProfile.checkpointTensorBytes,
        modelHiddenSize: architectureProfile.dModel,
        modelLayers: architectureProfile.layers,
        architectureProfile
      });
      if (
        planned.selectedItems < 1 ||
        planned.selectedItems === profileItems
      ) {
        return planned;
      }
      profileItems = planned.selectedItems;
    }
    return planned!;
  }
}
