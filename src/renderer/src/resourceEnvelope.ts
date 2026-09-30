import type {
  BrainConfig,
  SystemRamMode,
  WorkingMemoryResourcePlan
} from "../../shared/types";

export const MIN_SYSTEM_RAM_SHARE_PERCENT = 30;
export const MAX_SYSTEM_RAM_SHARE_PERCENT = 100;
export const DEFAULT_MANUAL_SYSTEM_RAM_SHARE_PERCENT = 65;

export const STORAGE_BOUNDARY_COPY =
  "Active tiles, token indexes and the hottest pathways stay in RAM. Cold attention and saved activity may use the designated storage pool alongside cold memory, replay batches, and checkpoints. Resource pressure pauses work without resizing learned workspace tables.";

export interface ContextCapacityBandStyle {
  "--capacity-black-low": string;
  "--capacity-red-low": string;
  "--capacity-orange-low": string;
  "--capacity-yellow-low": string;
  "--capacity-green-start": string;
  "--capacity-green-end": string;
  "--capacity-yellow-high": string;
  "--capacity-orange-high": string;
  "--capacity-red-high": string;
}

function percentOf(value: number, maximum: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(maximum) || maximum <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, value / maximum * 100));
}

/** Build a device-specific risk band around the measured suitable range. */
export function contextCapacityBandStyle(
  plan: WorkingMemoryResourcePlan
): ContextCapacityBandStyle {
  const maximum = Math.max(1, plan.context.maximumTokens);
  const greenStart = percentOf(
    plan.context.suitableMinimumTokens,
    maximum
  );
  const greenEnd = Math.max(
    greenStart,
    percentOf(plan.context.suitableMaximumTokens, maximum)
  );
  const highSpan = Math.max(0, 100 - greenEnd);
  return {
    "--capacity-black-low": `${greenStart * 0.22}%`,
    "--capacity-red-low": `${greenStart * 0.48}%`,
    "--capacity-orange-low": `${greenStart * 0.72}%`,
    "--capacity-yellow-low": `${greenStart * 0.92}%`,
    "--capacity-green-start": `${greenStart}%`,
    "--capacity-green-end": `${greenEnd}%`,
    "--capacity-yellow-high": `${greenEnd + highSpan * 0.2}%`,
    "--capacity-orange-high": `${greenEnd + highSpan * 0.46}%`,
    "--capacity-red-high": `${greenEnd + highSpan * 0.72}%`
  };
}

export function clampSystemRamSharePercent(value: number): number {
  if (!Number.isFinite(value)) return MIN_SYSTEM_RAM_SHARE_PERCENT;
  return Math.max(
    MIN_SYSTEM_RAM_SHARE_PERCENT,
    Math.min(MAX_SYSTEM_RAM_SHARE_PERCENT, Math.round(value))
  );
}

export function initialManualSystemRamSharePercent(config: BrainConfig): number {
  return config.systemRamMode === "manual"
    ? clampSystemRamSharePercent(config.systemRamSharePercent)
    : DEFAULT_MANUAL_SYSTEM_RAM_SHARE_PERCENT;
}

/**
 * Keep the persisted runtime envelope synchronized with the exact plan the
 * person reviewed. Auto deliberately stores a zero manual percentage so the
 * next launch re-runs device detection instead of pinning an old recommendation.
 */
export function configWithResourceEnvelope(
  config: BrainConfig,
  plan: WorkingMemoryResourcePlan,
  systemRamMode: SystemRamMode,
  systemRamSharePercent: number
): BrainConfig {
  return {
    ...config,
    // Learned latent-table geometry belongs to the saved architecture. This
    // helper is for Device settings, not the separately preflighted new Build.
    workingMemorySlots: config.workingMemorySlots,
    contextWindowTokens: plan.context.selectedTokens,
    workingMemoryMode: plan.mode,
    extendedWorkingMemory: plan.mode === "extended",
    memoryOffloadBytes: plan.resources.configuredMemorySpillBytes,
    contextOffloadBudgetBytes: plan.context.evidence?.contextOffloadBudgetBytes ?? config.contextOffloadBudgetBytes ?? 0,
    memoryResidentItems: Math.max(1, Math.min(config.workingMemorySlots, plan.offload.residentMemoryItems)),
    memoryOffloadSlowdownPercent: plan.offload.estimatedSlowdownPercent,
    systemRamMode,
    systemRamSharePercent:
      systemRamMode === "manual"
        ? clampSystemRamSharePercent(systemRamSharePercent)
        : 0,
    storagePoolMode: plan.resources.storagePoolMode,
    storagePoolBytes: plan.resources.sharedStoragePoolBytes,
    storageBytesPerSecond: plan.offload.benchmark.storageBytesPerSecond
  };
}
