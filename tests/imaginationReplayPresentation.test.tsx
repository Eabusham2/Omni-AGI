import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ImaginationReplayControls } from "../src/renderer/src/ImaginationReplayControls";
import { EMPTY_REPLAY_OPTIONS, mergeReplayJobs, mergeReplayPlanProgress, replayIsActive, replayOptions, replayPlanStatus, replayState } from "../src/renderer/src/imaginationReplayPresentation";
import type { ImaginationAssembly, ImaginationAssemblyApi } from "../src/shared/imaginationAssembly";
import type { ImaginationRecording } from "../src/shared/imaginationCapture";
const job = (id: string, revision: number): ImaginationAssembly => ({ format: "omni-imagination-assembly", version: 1, id, brainId: "brain-a", recordingId: "recording-a", recordingRevision: 2, requestId: "request-a", revision, state: "queued", createdAt: "2026-10-13T00:00:00Z", updatedAt: "2026-10-13T00:00:00Z", completedSpans: 0, totalSpans: 0, durationMs: 0, manifestSha256: "a".repeat(64), label: "Delivered timeline replay — presentation timing, not native synchronization" });
const recording: ImaginationRecording = { format: "omni-imagination-recording", version: 1, id: "recording-a", brainId: "brain-a", sequence: 1, requestId: "capture-a", mode: "manual", state: "complete", revision: 2, controlRevision: 2, startedAt: "2026-10-13T00:00:00Z", timelineEndMs: 1000, requestedLookbackSeconds: 0, availableLookbackSeconds: 0, fragmentCount: 2, bytes: 200, gapCount: 0, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video" };
describe("recorded imagination replay controls", () => {
  it("leaves Auto unspecified and preserves explicit millisecond timing without binary-roundoff rejection", () => {
    expect(replayOptions(EMPTY_REPLAY_OPTIONS)).toBeUndefined();
    expect(replayOptions({ ...EMPTY_REPLAY_OPTIONS, width: "1920", height: "1080", fps: "29.97", historicalTailSeconds: "1.001", streamOffsets: "first = 0.029\nsecond = 2.125" })).toEqual({ width: 1920, height: 1080, fps: 29.97, historicalTailMs: 1001, presentationOffsets: [{ streamId: "first", offsetMs: 29 }, { streamId: "second", offsetMs: 2125 }] });
  });
  it("rejects impossible overrides and duplicate, negative, or guessed precision offsets", () => {
    for (const change of [{ width: "1.2" }, { height: "0" }, { fps: "Infinity" }, { historicalTailSeconds: "0.0001" }, { streamOffsets: "one = 1\none = 2" }, { streamOffsets: "one = -2" }, { streamOffsets: "one = unknown" }]) expect(() => replayOptions({ ...EMPTY_REPLAY_OPTIONS, ...change })).toThrow();
  });
  it("keeps new receipts and exact recording ownership without treating stopped or copied replay as active", () => {
    const latest = { ...job("one", 3), state: "complete" as const };
    const values = mergeReplayJobs([latest], [job("one", 1), { ...job("foreign", 9), brainId: "brain-b" }, { ...job("other-record", 9), recordingId: "recording-b" }], "brain-a", "recording-a");
    expect(values).toEqual([latest]); expect(replayIsActive(latest)).toBe(false);
    expect(replayIsActive({ ...latest, state: "dormant" })).toBe(false);
    expect(replayState({ ...latest, state: "cancelled" })).toContain("originals kept"); expect(replayState(job("unknown-total", 1))).not.toContain("0 of 0");
  });
  it("does not plan, encode, resume, load a model, or render fabricated playback on opening markup", () => {
    const api = new Proxy({}, { get: () => { throw new Error("Unexpected replay operation"); } }) as ImaginationAssemblyApi;
    const html = renderToStaticMarkup(<ImaginationReplayControls recording={recording} api={api} />);
    expect(html).toContain("Prepare replay"); expect(html).toContain("Replay options"); expect(html).not.toContain("Make replay");
    expect(html).not.toContain("<video"); expect(html).not.toContain("<audio"); expect(html).not.toContain("native synchronization");
    const activeHtml = renderToStaticMarkup(<ImaginationReplayControls recording={{ ...recording, state: "recording" }} api={api} />);
    expect(activeHtml).toContain("Stop recording to prepare a fixed replay"); expect(activeHtml).toContain("disabled=");
  });
  it("uses only the main-projected current recording alias for copied history, leaving the immutable source identity unchanged", () => {
    const copied = { ...job("copy", 4), recordingId: "original-parent-recording", currentRecordingId: "recording-a", state: "dormant" as const };
    expect(mergeReplayJobs([], [copied], "brain-a", "recording-a")).toEqual([copied]);
    expect(copied.recordingId).toBe("original-parent-recording");
    expect(mergeReplayJobs([], [copied], "brain-b", "recording-a")).toEqual([]);
  });
  it("keeps exact preparation ownership and monotonic progress without inventing zero totals", () => {
    const ready = { brainId: "brain-a", requestId: "prepare-a", sequence: 4, state: "ready" as const };
    expect(mergeReplayPlanProgress(ready, { ...ready, sequence: 3, state: "runtime" }, "brain-a", "prepare-a")).toBe(ready);
    expect(mergeReplayPlanProgress(undefined, ready, "brain-b", "prepare-a")).toBeUndefined();
    expect(mergeReplayPlanProgress(undefined, ready, "brain-a", "prepare-b")).toBeUndefined();
    expect(replayPlanStatus({ ...ready, state: "sources", completedFragments: 0, totalFragments: 0 })).toBe("Inspecting source media · 0 fragments");
    expect(replayPlanStatus({ ...ready, state: "runtime", runtime: { state: "downloading", target: "fixture", message: "", completedBytes: 25, totalBytes: 100 } })).toContain("25 of 100 bytes");
  });
});
