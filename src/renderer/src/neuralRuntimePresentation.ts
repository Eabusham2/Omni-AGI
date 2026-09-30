import type { NeuralParameterAccounting } from "../../shared/types";

type RuntimeRow = readonly [label: string, value: string];

export interface NeuralParameterAccountingPresentation
  extends NeuralParameterAccounting {
  compactTotal: string;
  exactLabel: string;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function safeCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

export function compactParameterCount(value: number): string {
  const count = safeCount(value);
  if (count === undefined) return "—";
  const units = [
    { size: 1, suffix: "" },
    { size: 1_000, suffix: "K" },
    { size: 1_000_000, suffix: "M" },
    { size: 1_000_000_000, suffix: "B" },
    { size: 1_000_000_000_000, suffix: "T" }
  ] as const;
  let unitIndex = 0;
  while (unitIndex < units.length - 1 && count >= units[unitIndex + 1]!.size) {
    unitIndex += 1;
  }
  let scaled = Number((count / units[unitIndex]!.size).toFixed(3));
  if (scaled >= 1_000 && unitIndex < units.length - 1) {
    unitIndex += 1;
    scaled = Number((count / units[unitIndex]!.size).toFixed(3));
  }
  return `${scaled.toLocaleString("en-US", {
    useGrouping: false,
    minimumFractionDigits: 0,
    maximumFractionDigits: 3
  })}${units[unitIndex]!.suffix}`;
}

/** Parse only the native worker's non-overlapping parameter counter. */
export function neuralParameterAccounting(
  runtimeCard: Record<string, unknown> | null | undefined
): NeuralParameterAccountingPresentation | undefined {
  const accounting = record(runtimeCard?.parameterAccounting);
  if (!accounting) return undefined;
  const mutableDenseParameters = safeCount(accounting.mutableDenseParameters);
  const substrateVectorParameters = accounting.substrateVectorParameters === undefined
    ? 0
    : safeCount(accounting.substrateVectorParameters);
  const substrateDynamicSparseSynapses = safeCount(
    accounting.substrateDynamicSparseSynapses
  );
  const dynamicSparseSynapses = safeCount(accounting.dynamicSparseSynapses);
  const totalNeuralParameters = safeCount(accounting.totalNeuralParameters);
  const countingRule = typeof accounting.countingRule === "string"
    ? accounting.countingRule.replaceAll("\0", "").trim()
    : "";
  if (
    mutableDenseParameters === undefined ||
    substrateVectorParameters === undefined ||
    substrateDynamicSparseSynapses === undefined ||
    dynamicSparseSynapses === undefined ||
    totalNeuralParameters === undefined ||
    totalNeuralParameters < 1 ||
    substrateDynamicSparseSynapses !== dynamicSparseSynapses ||
    ["foundationEffectiveParameters", "sequenceDynamicSparseSynapses", "fixedSequenceStatisticalCapacity"]
      .some((field) => Object.hasOwn(accounting, field)) ||
    mutableDenseParameters + substrateVectorParameters + dynamicSparseSynapses !==
      totalNeuralParameters ||
    !countingRule ||
    countingRule.length > 512
  ) {
    return undefined;
  }
  const exact = (value: number): string => value.toLocaleString("en-US");
  return {
    mutableDenseParameters,
    ...(accounting.substrateVectorParameters !== undefined ? { substrateVectorParameters } : {}),
    substrateDynamicSparseSynapses,
    dynamicSparseSynapses,
    totalNeuralParameters,
    countingRule,
    compactTotal: compactParameterCount(totalNeuralParameters),
    exactLabel:
      `${exact(totalNeuralParameters)} logical neural weights and connections, counted once in one brain. ` +
      `${exact(mutableDenseParameters)} core weights and ` +
      (substrateVectorParameters > 0
        ? `${exact(substrateVectorParameters)} learned memory weights and `
        : "") +
      `${exact(dynamicSparseSynapses)} grown connections. ${countingRule}`
  };
}

function formatBytes(value: number): string {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GiB`;
  if (value >= 1024 ** 2) return `${Math.round(value / 1024 ** 2)} MiB`;
  if (value >= 1024) return `${Math.round(value / 1024)} KiB`;
  return `${Math.round(value)} B`;
}

/** Display worker-measured physical memory, never a desktop-side estimate. */
export function neuralResourceRuntimeRows(
  runtimeCard: Record<string, unknown> | undefined
): RuntimeRow[] {
  const offload = record(runtimeCard?.state_offload);
  const resources = record(offload?.resources);
  if (!resources) return [];
  const current = finiteNumber(resources.processMemoryBytes);
  const peak = finiteNumber(resources.processPeakMemoryBytes);
  if (current === undefined && peak === undefined) return [];
  const parts = [
    current === undefined ? undefined : `${formatBytes(current)} current`,
    peak === undefined ? undefined : `${formatBytes(peak)} peak`
  ].filter((value): value is string => value !== undefined);
  return [["Python neural memory", `${parts.join(" · ")} · physical footprint`]];
}
