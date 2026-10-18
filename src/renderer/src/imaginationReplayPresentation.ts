import type { ImaginationAssembly, ImaginationAssemblyOptions, ImaginationPlanEvent } from "../../shared/imaginationAssembly";

export interface ReplayOptionsDraft {
  width: string; height: string; fps: string; historicalTailSeconds: string; streamOffsets: string;
}
export const EMPTY_REPLAY_OPTIONS: ReplayOptionsDraft = { width: "", height: "", fps: "", historicalTailSeconds: "", streamOffsets: "" };

function milliseconds(raw: string): number | undefined {
  if (!/^\d+(?:\.\d{1,3})?$/.test(raw)) return undefined;
  const value = Math.round(Number(raw) * 1000);
  return Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

/** Empty fields leave Auto intact. Overrides are explicit, never substituted silently. */
export function replayOptions(draft: ReplayOptionsDraft): Partial<ImaginationAssemblyOptions> | undefined {
  const options: Partial<ImaginationAssemblyOptions> = {};
  for (const key of ["width", "height", "fps"] as const) {
    const raw = draft[key].trim(); if (!raw) continue;
    const value = Number(raw);
    if (!Number.isFinite(value) || value <= 0 || ((key === "width" || key === "height") && !Number.isSafeInteger(value))) {
      throw new Error(`${key === "fps" ? "Frame rate" : key === "width" ? "Width" : "Height"} must be a positive ${key === "fps" ? "number" : "whole number"}.`);
    }
    options[key] = value;
  }
  if (draft.historicalTailSeconds.trim()) {
    const value = milliseconds(draft.historicalTailSeconds.trim());
    if (value === undefined) throw new Error("Ending hold must be nonnegative seconds with millisecond precision.");
    options.historicalTailMs = value;
  }
  const lines = draft.streamOffsets.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
  if (lines.length) {
    const seen = new Set<string>();
    options.presentationOffsets = lines.map(line => {
      const split = line.lastIndexOf("=");
      const streamId = split < 0 ? "" : line.slice(0, split).trim(), value = milliseconds(line.slice(split + 1).trim());
      if (!streamId || seen.has(streamId) || value === undefined) {
        throw new Error("Use one unique stream ID = source seconds per line, with nonnegative millisecond precision.");
      }
      seen.add(streamId); return { streamId, offsetMs: value };
    });
  }
  return Object.keys(options).length ? options : undefined;
}

export function replayRecordingId(value: ImaginationAssembly): string { return value.currentRecordingId ?? value.recordingId; }
export function mergeReplayJobs(current: ImaginationAssembly[], incoming: ImaginationAssembly[], brainId: string, recordingId: string): ImaginationAssembly[] {
  const byId = new Map(current.filter(value => value.brainId === brainId && replayRecordingId(value) === recordingId).map(value => [value.id, value]));
  for (const value of incoming) {
    if (value.brainId !== brainId || replayRecordingId(value) !== recordingId) continue;
    if (!byId.has(value.id) || byId.get(value.id)!.revision <= value.revision) byId.set(value.id, value);
  }
  return [...byId.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id));
}

export function replayState(value: ImaginationAssembly): string {
  const label = value.state === "complete" ? "Ready" : value.state === "cancelled" ? "Stopped · originals kept" : value.state === "dormant" ? "Paused copy" :
    value.state === "failed" ? "Replay failed · originals kept" : value.state === "queued" ? "Waiting" : value.state === "preparing" ? "Preparing replay" : "Making replay";
  return value.totalSpans > 0 ? `${label} · ${value.completedSpans.toLocaleString()} of ${value.totalSpans.toLocaleString()} spans` : label;
}

export function replayIsActive(value: ImaginationAssembly): boolean { return ["queued", "preparing", "encoding"].includes(value.state); }

export function mergeReplayPlanProgress(current: ImaginationPlanEvent | undefined, incoming: ImaginationPlanEvent, brainId: string, requestId: string): ImaginationPlanEvent | undefined {
  if (incoming.brainId !== brainId || incoming.requestId !== requestId || !Number.isSafeInteger(incoming.sequence) || incoming.sequence < 0) return current;
  if (current?.requestId === requestId && current.sequence >= incoming.sequence) return current;
  return incoming;
}
export function replayPlanStatus(value: ImaginationPlanEvent): string {
  const phase = { snapshotting: "Copying recording snapshot", runtime: "Preparing video runtime", sources: "Inspecting source media", spooling: "Preparing timeline", ready: "Replay plan ready", cancelling: "Stopping preparation", cancelled: "Preparation stopped", failed: "Preparation failed", dormant: "Preparation paused" }[value.state];
  if (value.runtime?.completedBytes !== undefined) return `${phase} · ${value.runtime.completedBytes.toLocaleString()}${value.runtime.totalBytes && value.runtime.totalBytes > 0 ? ` of ${value.runtime.totalBytes.toLocaleString()}` : ""} bytes`;
  if (value.completedFragments !== undefined) return `${phase} · ${value.completedFragments.toLocaleString()}${value.totalFragments && value.totalFragments > 0 ? ` of ${value.totalFragments.toLocaleString()}` : ""} fragments`;
  return phase;
}
