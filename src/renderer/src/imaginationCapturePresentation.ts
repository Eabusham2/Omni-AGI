import type { ImaginationCapturedFragment, ImaginationRecording } from "../../shared/imaginationCapture";

export const CAPTURE_WINDOW_PRESETS = [15, 30, 60, 180, 300] as const;
export function parseCaptureSeconds(raw: string, allowZero = false): number {
  if (!/^\d+(?:\.\d{1,3})?$/.test(raw.trim())) throw new Error("Use seconds, with at most three decimal places.");
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0 || (!allowZero && value === 0) || value > 300) {
    throw new Error(allowZero ? "Look back from 0 to 300 seconds." : "The rolling window must be greater than 0 and no more than 300 seconds.");
  }
  return value;
}
export function captureTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "Unavailable";
  return seconds >= 60 ? `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s` : `${Math.floor(seconds)}s`;
}
export function captureRecordingState(record: ImaginationRecording): string {
  return ({ recording: "Recording", complete: "Saved", cancelled: "Stopped · saved fragments kept", dormant: "Saved · paused", "resource-paused": "Paused · resource pressure" })[record.state];
}
export function mergeCaptureRecordings(current: ImaginationRecording[], incoming: ImaginationRecording[], brainId: string): ImaginationRecording[] {
  const records = new Map<string, ImaginationRecording>();
  for (const item of [...current, ...incoming]) {
    if (item.brainId !== brainId) continue;
    const prior = records.get(item.id);
    if (!prior || item.revision >= prior.revision) records.set(item.id, item);
  }
  return [...records.values()].sort((a, b) => b.sequence - a.sequence);
}
export function mergeCaptureFragments(current: ImaginationCapturedFragment[], incoming: ImaginationCapturedFragment[], recordingId: string): ImaginationCapturedFragment[] {
  const fragments = new Map<number, ImaginationCapturedFragment>();
  for (const item of [...current, ...incoming]) if (item.recordingId === recordingId) fragments.set(item.index, item);
  return [...fragments.values()].sort((a, b) => a.index - b.index);
}
