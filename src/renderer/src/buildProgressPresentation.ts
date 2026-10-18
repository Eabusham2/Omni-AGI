import type { BrainStorageOperationEvent, BuildProgressEvent } from "../../shared/types";

export interface ProgressTimingPresentation {
  percent: number;
  elapsedMs?: number;
  remainingMs?: number;
  totalMs?: number;
  elapsedText: string;
  remainingText: string;
  totalText?: string;
  ariaLabel: string;
  terminal: boolean;
}

function duration(value: number): string {
  const seconds = Math.max(0, Math.floor(value / 1_000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60), remainder = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainder.toString().padStart(2, "0")}s`;
  const hours = Math.floor(minutes / 60), minuteRemainder = minutes % 60;
  return `${hours}h ${minuteRemainder.toString().padStart(2, "0")}m`;
}

function measured(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

function phaseLabel(value: string): string {
  const normalized = value.replace(/[-_]+/g, " ").replace(/\s+/g, " ").trim();
  return normalized ? normalized.slice(0, 1).toLocaleUpperCase() + normalized.slice(1) : "Current phase";
}

/**
 * Uses only producer-measured monotonic duration and an explicit ETA. Weighted
 * build percentages are presentation stages, not a time model.
 */
export function buildProgressTimingPresentation(input: {
  phase: BuildProgressEvent["phase"];
  progress: number;
  elapsedMs?: number;
  sinceUpdateMs?: number;
  etaMs?: number;
  terminal?: boolean;
}): ProgressTimingPresentation {
  const terminal = input.phase === "complete" && input.terminal !== false;
  const progress = typeof input.progress === "number" && Number.isFinite(input.progress)
    ? Math.max(0, Math.min(1, input.progress)) : 0;
  const percent = terminal ? 100 : Math.min(99, Math.round(progress * 100));
  const baseElapsed = measured(input.elapsedMs), sinceUpdate = terminal ? 0 : measured(input.sinceUpdateMs) ?? 0;
  const elapsedMs = baseElapsed === undefined ? undefined : baseElapsed + sinceUpdate;
  const sampledEta = measured(input.etaMs);
  const remainingMs = !terminal && sampledEta !== undefined && sampledEta > sinceUpdate
    ? sampledEta - sinceUpdate : undefined;
  const totalMs = elapsedMs !== undefined && (terminal || remainingMs !== undefined)
    ? elapsedMs + (remainingMs ?? 0) : undefined;
  const elapsedText = elapsedMs === undefined ? "Elapsed time measuring…" : `${duration(elapsedMs)} elapsed`;
  const remainingText = terminal ? "Durably complete"
    : remainingMs !== undefined ? `${duration(remainingMs)} remaining`
      : `Estimating remaining time · ${phaseLabel(input.phase)}`;
  return { percent, ...(elapsedMs !== undefined ? { elapsedMs } : {}), ...(remainingMs !== undefined ? { remainingMs } : {}),
    ...(totalMs !== undefined ? { totalMs, totalText: duration(totalMs) } : {}),
    elapsedText, remainingText, ariaLabel: `${percent}% complete. ${elapsedText}. ${remainingText}.`, terminal };
}

export function storageOperationTimingPresentation(input: Pick<BrainStorageOperationEvent, "state" | "phase" | "elapsedMs" | "etaMs"> & {
  sinceUpdateMs?: number;
}): Omit<ProgressTimingPresentation, "percent"> {
  const terminal = ["complete", "failed", "cancelled"].includes(input.state);
  const active = ["queued", "running", "cancelling"].includes(input.state);
  const sinceUpdate = active ? measured(input.sinceUpdateMs) ?? 0 : 0;
  const baseElapsed = measured(input.elapsedMs), elapsedMs = baseElapsed === undefined ? undefined : baseElapsed + sinceUpdate;
  const sampledEta = measured(input.etaMs);
  const remainingMs = active && sampledEta !== undefined && sampledEta > sinceUpdate ? sampledEta - sinceUpdate : undefined;
  const totalMs = elapsedMs !== undefined && (input.state === "complete" || remainingMs !== undefined)
    ? elapsedMs + (remainingMs ?? 0) : undefined;
  const elapsedText = elapsedMs === undefined ? "Elapsed time measuring…" : `${duration(elapsedMs)} elapsed`;
  const remainingText = input.state === "complete" ? "Durably complete"
    : input.state === "failed" ? "Ended with error"
      : input.state === "cancelled" ? "Cancelled"
        : input.state === "paused" ? "Paused · remaining time not advancing"
          : remainingMs !== undefined ? `${duration(remainingMs)} remaining`
            : `Estimating remaining time · ${phaseLabel(input.phase)}`;
  return { ...(elapsedMs !== undefined ? { elapsedMs } : {}), ...(remainingMs !== undefined ? { remainingMs } : {}),
    ...(totalMs !== undefined ? { totalMs, totalText: duration(totalMs) } : {}), elapsedText,
    remainingText, ariaLabel: `${elapsedText}. ${remainingText}.`, terminal };
}
