export interface CognitionProgress {
  phase: "ponder" | "best-work";
  active: boolean;
  passes?: number;
  step?: number;
  branch?: number;
  admittedBranches?: number;
  completedBranches?: number;
  elapsedMs?: number;
  delta?: number;
  entropy?: number;
  ordinary?: boolean;
  schedulerYield?: boolean;
  selectedAction?: "talk" | "tool" | "imagine" | "agent" | "ponder" | "learn" | "evolve" | "stop";
  stopReason?: string;
  concurrency?: "cooperative-interleaved";
}

/** Operational measurements only: never a completion event or instruction. */
export function normalizeCognitionProgress(value: unknown): CognitionProgress | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const record = value as Record<string, unknown>;
  if (record.phase !== "ponder" && record.phase !== "best-work") return undefined;
  if (record.active !== undefined && typeof record.active !== "boolean") return undefined;
  const result: CognitionProgress = { phase: record.phase, active: record.active !== false };
  for (const key of ["passes", "step", "branch", "admittedBranches", "completedBranches"] as const) {
    const item = record[key];
    if (item === undefined) continue;
    if (typeof item !== "number" || !Number.isSafeInteger(item) || item < 0) return undefined;
    result[key] = item;
  }
  for (const key of ["elapsedMs", "delta", "entropy"] as const) {
    const item = record[key];
    if (item === undefined) continue;
    if (typeof item !== "number" || !Number.isFinite(item) || item < 0) return undefined;
    result[key] = item;
  }
  for (const key of ["ordinary", "schedulerYield"] as const) {
    if (record[key] === undefined) continue;
    if (typeof record[key] !== "boolean") return undefined;
    result[key] = record[key];
  }
  if (record.selectedAction !== undefined) {
    if (!["talk", "tool", "imagine", "agent", "ponder", "learn", "evolve", "stop"].includes(String(record.selectedAction))) return undefined;
    result.selectedAction = record.selectedAction as CognitionProgress["selectedAction"];
  }
  if (record.concurrency !== undefined) {
    if (record.concurrency !== "cooperative-interleaved") return undefined;
    result.concurrency = record.concurrency;
  }
  if (record.stopReason !== undefined) {
    if (typeof record.stopReason !== "string" || !/^[a-z0-9][a-z0-9 -]{0,99}$/i.test(record.stopReason)) return undefined;
    result.stopReason = record.stopReason;
  }
  if (result.completedBranches !== undefined && result.admittedBranches !== undefined &&
      result.completedBranches > result.admittedBranches) return undefined;
  return result;
}
