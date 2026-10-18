import { describe, expect, it } from "vitest";
import { planDeliveredTimeline } from "../src/main/imaginationAssemblyPlan";
import { nativeMediaTimeline, type NativeMediaTimeline } from "../src/shared/nativeMediaTimeline";
import type { ImaginationCapturedFragment, ImaginationRecording } from "../src/shared/imaginationCapture";
const recording: ImaginationRecording = { format: "omni-imagination-recording", version: 1, id: "recording", brainId: "brain", sequence: 1, requestId: "record", mode: "manual", state: "complete", revision: 5, controlRevision: 5, startedAt: "2026-10-13T12:00:00Z", requestedLookbackSeconds: 0, availableLookbackSeconds: 0, fragmentCount: 2, bytes: 2, gapCount: 1, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video", timelineEndMs: 1000 };
const timeline = (unit: NativeMediaTimeline["unit"], count: number, rate: number): NativeMediaTimeline => ({ format: "omni-native-media-timeline", version: 1, producer: "native-byte-encoder", layout: unit === "image" ? "snapshot" : "replacement-from-zero", unit, start: 0, count, timebaseNumerator: 1, timebaseDenominator: rate });
function fragment(index: number, offsetMs: number, extra: Partial<ImaginationCapturedFragment> = {}): Omit<ImaginationCapturedFragment,"mediaUrl"|"available"> { return { id: `fragment-${index}`, recordingId: recording.id, index, streamId: "stream", observedAt: recording.startedAt, offsetMs, origin: { kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic" }, kind: "preview-fragment", modality: "audio", mimeType: "audio/wav", sha256: String(index + 1).padStart(64, "0"), bytes: 1, mediaTimeline: timeline("samples", 16000, 16000), deliveryAnchor: { version: 1, kind: "first-verified-main-delivery", firstObservedAt: recording.startedAt, elapsedMs: offsetMs, truncatedStart: false }, ...extra }; }
describe("delivered source clock policy without a model", () => {
  it("replaces every overlapping prefix at its delivery offset instead of concatenating or repeating it", () => {
    const plan = planDeliveredTimeline(recording, [fragment(0, 0), fragment(1, 500, { mediaTimeline: timeline("samples", 32000, 16000) })], { width: 2, height: 2, fps: 10 });
    expect(plan.fragments).toHaveLength(2); expect(plan.spans[1]?.states[0]).toMatchObject({ fragmentId: "fragment-1", sourceTimeMs: 500, audio: "play" }); expect(plan.durationMs).toBe(1000); expect(plan.outputKind).toBe("audio-only");
  });
  it("retains a known earlier delivery anchor when retrospective recording begins mid stream", () => {
    const row = fragment(0, 0, { deliveryAnchor: { version: 1, kind: "first-verified-main-delivery", firstObservedAt: recording.startedAt, elapsedMs: 800, truncatedStart: false } });
    const plan = planDeliveredTimeline({ ...recording, fragmentCount: 1 }, [row], { width: 2, height: 2, fps: 10 }); expect(plan.spans[0]?.states[0]?.sourceTimeMs).toBe(800);
  });
  it("marks truncated prefix starts as gaps until an explicit source offset is declared", () => {
    const row = fragment(0, 0, { deliveryAnchor: undefined }), args = { width: 2, height: 2, fps: 10 };
    expect(planDeliveredTimeline(recording, [row], args).spans[0]?.states[0]).toMatchObject({ audio: "none", issue: "truncated-start-needs-offset" });
    expect(planDeliveredTimeline(recording, [row], { ...args, presentationOffsets: [{ streamId: "stream", offsetMs: 250 }] }).spans[0]?.states[0]).toMatchObject({ sourceTimeMs: 250, audio: "play" });
  });
  it("does not fabricate historical end time or discard unknown audio timing", () => {
    expect(() => planDeliveredTimeline({ ...recording, timelineEndMs: undefined }, [fragment(0, 0)], { width: 2, height: 2, fps: 10 })).toThrow(/declare/);
    const plan = planDeliveredTimeline({ ...recording, timelineEndMs: undefined }, [fragment(0, 0, { mediaTimeline: undefined })], { width: 2, height: 2, fps: 10, historicalTailMs: 100 });
    expect(plan.fragments).toHaveLength(1); expect(plan.measuredEnd).toBe(false); expect(plan.spans[0]?.states[0]?.issue).toBe("unknown-source-timing");
  });
  it("keeps same-frame revisions in provenance and declares quantization instead of fake native precision", () => {
    const rows = [fragment(0, 0, { modality: "image", mimeType: "image/png", mediaTimeline: timeline("image", 1, 1) }), fragment(1, 1, { modality: "image", mimeType: "image/png", mediaTimeline: timeline("image", 1, 1) })];
    const plan = planDeliveredTimeline(recording, rows, { width: 32, height: 16, fps: 10 }); expect(plan.fragments).toHaveLength(2); expect(plan.presentation.supersededInSameFrame).toEqual(["fragment-0"]); expect(plan.presentation.nativeSynchronizationClaimed).toBe(false);
  });
  it("accepts only measured byte ranges and genuine bounded stability fields", () => {
    expect(nativeMediaTimeline(timeline("samples", 160, 16000))).toBeDefined(); expect(nativeMediaTimeline({ ...timeline("samples", 160, 16000), stableCount: 161 })).toBeUndefined(); expect(nativeMediaTimeline({ ...timeline("frames", 2, 8), containerAudio: { sampleCount: 100, sampleRate: 16000, startSample: 0 } })).toBeUndefined();
  });
});
