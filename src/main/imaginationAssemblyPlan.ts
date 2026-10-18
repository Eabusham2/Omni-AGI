import type { ImaginationCapturedFragment, ImaginationRecording } from "../shared/imaginationCapture";
import type { ImaginationAssemblyManifest, ImaginationAssemblyOptions, ImaginationReplayState, ImaginationReplayTrack } from "../shared/imaginationAssembly";
import { nativeMediaTimeline } from "../shared/nativeMediaTimeline";

export function checkedAssemblyOptions(value: ImaginationAssemblyOptions): ImaginationAssemblyOptions {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !["width", "height", "fps", "historicalTailMs", "presentationOffsets"].includes(key))) throw new Error("Unsupported exact composition options.");
  for (const key of ["width", "height", "fps"] as const) if (!Number.isSafeInteger(value[key]) || value[key] < 1) throw new Error("Composition dimensions and FPS must be positive integers.");
  if (value.width % 2 || value.height % 2 || value.fps > 65535 || !Number.isSafeInteger(value.width * value.height * 64)) throw new Error("Composition needs even dimensions, a portable timebase and a representable resource estimate.");
  if (value.historicalTailMs !== undefined && (!Number.isFinite(value.historicalTailMs) || value.historicalTailMs < 0)) throw new Error("Historical tail must be an explicit nonnegative duration.");
  const ids = new Set<string>(); for (const row of value.presentationOffsets ?? []) {
    if (!row || Object.keys(row).sort().join(",") !== "offsetMs,streamId" || typeof row.streamId !== "string" || !row.streamId || ids.has(row.streamId) || !Number.isFinite(row.offsetMs) || row.offsetMs < 0) throw new Error("Presentation offsets must be unique exact stream identities and nonnegative source times."); ids.add(row.streamId);
  }
  return structuredClone(value);
}
export function deliveredReplayState(row: Omit<ImaginationCapturedFragment, "mediaUrl" | "available">, track: ImaginationReplayTrack, firstOffsetMs: number, explicitOffset: number | undefined, startMs: number): ImaginationReplayState {
  const base: ImaginationReplayState = { streamId: track.streamId, slot: track.slot, fragmentId: row.id, visual: "none", audio: "none" }, timeline = nativeMediaTimeline(row.mediaTimeline);
  if (timeline?.unit === "image" && row.modality === "image") return { ...base, visual: "snapshot" };
  if (!timeline || timeline.unit === "image") return { ...base, issue: "unknown-source-timing" };
  if (track.anchor === "unknown") return { ...base, issue: "truncated-start-needs-offset" };
  const sourceTimeMs = Math.max(0, explicitOffset !== undefined ? explicitOffset + startMs - firstOffsetMs : row.deliveryAnchor!.elapsedMs + startMs - row.offsetMs);
  const duration = timeline.count * timeline.timebaseNumerator * 1000 / timeline.timebaseDenominator, exhausted = sourceTimeMs >= duration;
  return { ...base, sourceTimeMs, visual: timeline.unit === "frames" ? exhausted ? "hold" : "play" : "none",
    audio: !exhausted && (timeline.unit === "samples" || !!timeline.containerAudio) ? "play" : "none", ...(exhausted ? { issue: "source-exhausted" as const } : {}) };
}
export function planDeliveredTimeline(recording: ImaginationRecording, input: Omit<ImaginationCapturedFragment, "mediaUrl" | "available">[], rawOptions: ImaginationAssemblyOptions): ImaginationAssemblyManifest {
  const options = checkedAssemblyOptions(rawOptions), fragments = structuredClone(input);
  if (!fragments.length) throw new Error("A recording needs a verified delivered fragment before assembly.");
  const byId = new Set<string>(); for (const row of fragments) {
    if (row.recordingId !== recording.id || byId.has(row.id) || !Number.isFinite(row.offsetMs) || row.offsetMs < 0 || !/^[a-f0-9]{64}$/.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 1) throw new Error("Invalid exact source timeline."); byId.add(row.id);
    if (row.mediaTimeline !== undefined && !nativeMediaTimeline(row.mediaTimeline)) throw new Error("Invalid byte-producer timeline.");
  }
  fragments.sort((a, b) => a.offsetMs - b.offsetMs || a.index - b.index);
  const last = fragments.at(-1)!.offsetMs, measuredEnd = recording.timelineEndMs !== undefined;
  if (!measuredEnd && options.historicalTailMs === undefined) throw new Error("Historical recording has no measured end; declare its presentation tail explicitly.");
  const end = recording.timelineEndMs ?? last + options.historicalTailMs!;
  if (!Number.isFinite(end) || end < last || end <= 0) throw new Error("The measured or declared composition end does not cover its source timeline.");
  const audioOnly = fragments.every(row => row.modality === "audio"), presentationRate = audioOnly ? 48000 : options.fps;
  const totalFrames = Math.max(1, Math.round(end * presentationRate / 1000)); if (!Number.isSafeInteger(totalFrames)) throw new Error("Composition duration exceeds its exact frame representation.");
  const firstByStream = new Map<string, typeof fragments[number]>(); for (const row of fragments) if (!firstByStream.has(row.streamId)) firstByStream.set(row.streamId, row);
  const explicit = new Map((options.presentationOffsets ?? []).map(row => [row.streamId, row.offsetMs]));
  for (const key of explicit.keys()) if (!firstByStream.has(key)) throw new Error("An explicit offset names a stream outside this recording.");
  const tracks = [...firstByStream].map(([streamId, row], slot) => ({ streamId, slot, firstDeliveryMs: row.offsetMs,
    truncatedStart: row.deliveryAnchor?.truncatedStart ?? true,
    anchor: explicit.has(streamId) ? "explicit-retained-source-offset" as const : row.deliveryAnchor && !row.deliveryAnchor.truncatedStart ? "first-verified-main-delivery" as const : "unknown" as const }));
  const events = new Map<number, typeof fragments>(); const supersededInSameFrame: string[] = [];
  for (const row of fragments) { const frame = Math.min(totalFrames, Math.max(0, Math.round(row.offsetMs * presentationRate / 1000))); const current = events.get(frame) ?? [];
    const previous = current.find(item => item.streamId === row.streamId); if (previous) supersededInSameFrame.push(previous.id);
    events.set(frame, [...current.filter(item => item.streamId !== row.streamId), row]); }
  const boundaries = [...new Set([0, totalFrames, ...events.keys()])].sort((a, b) => a - b), current = new Map<string, typeof fragments[number]>();
  const spans: ImaginationAssemblyManifest["spans"] = [];
  for (let n = 0; n + 1 < boundaries.length; n++) {
    const startFrame = boundaries[n]!, endFrame = boundaries[n + 1]!, startMs = startFrame * 1000 / presentationRate, endMs = endFrame * 1000 / presentationRate;
    for (const row of events.get(startFrame) ?? []) current.set(row.streamId, row);
    const states: ImaginationReplayState[] = [];
    for (const track of tracks) { const row = current.get(track.streamId); if (!row) continue;
      const base: ImaginationReplayState = { streamId: track.streamId, slot: track.slot, fragmentId: row.id, visual: "none", audio: "none" }, timeline = nativeMediaTimeline(row.mediaTimeline);
      if (timeline?.unit === "image" && row.modality === "image") { states.push({ ...base, visual: "snapshot" }); continue; }
      if (!timeline || timeline.unit === "image") { states.push({ ...base, issue: "unknown-source-timing" }); continue; }
      if (track.anchor === "unknown") { states.push({ ...base, issue: "truncated-start-needs-offset" }); continue; }
      const first = firstByStream.get(row.streamId)!;
      const sourceTimeMs = Math.max(0, explicit.has(row.streamId) ? explicit.get(row.streamId)! + startMs - first.offsetMs : row.deliveryAnchor!.elapsedMs + startMs - row.offsetMs);
      const duration = timeline.count * timeline.timebaseNumerator * 1000 / timeline.timebaseDenominator, exhausted = sourceTimeMs >= duration;
      states.push({ ...base, sourceTimeMs,
        visual: timeline.unit === "frames" ? exhausted ? "hold" : "play" : "none",
        audio: !exhausted && (timeline.unit === "samples" || !!timeline.containerAudio) ? "play" : "none",
        ...(exhausted ? { issue: "source-exhausted" as const } : {}) });
    }
    spans.push({ startFrame, endFrame, startMs, endMs, states });
  }
  return { format: "omni-delivered-timeline-composition", version: 1, policy: "delivered-timeline-v1", outputKind: audioOnly ? "audio-only" : "audiovisual", recording: structuredClone(recording), options, fragments, tracks, spans,
    durationMs: totalFrames * 1000 / presentationRate, measuredEnd, captureGapCount: recording.gapCount,
    presentation: { layout: "fixed-stream-grid", missingVisual: "neutral-black", exhaustedVisual: "last-verified-frame-hold", missingAudio: "declared-silence", independentAudio: "delivery-timed-equal-weight-mix", quantization: audioOnly ? "nearest-output-sample" : "nearest-output-frame", maximumQuantizationMs: 1000 / presentationRate, nativeSynchronizationClaimed: false, supersededInSameFrame } };
}
