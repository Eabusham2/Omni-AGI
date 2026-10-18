import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ImaginationCaptureControls } from "../src/renderer/src/ImaginationCaptureControls";
import { captureRecordingState, captureTime, mergeCaptureFragments, mergeCaptureRecordings, parseCaptureSeconds } from "../src/renderer/src/imaginationCapturePresentation";
import type { ImaginationCaptureApi, ImaginationCapturedFragment, ImaginationRecording } from "../src/shared/imaginationCapture";

const recording = (id: string, revision = 1): ImaginationRecording => ({ format: "omni-imagination-recording", version: 1, id, brainId: "brain-a", sequence: 1, requestId: "request-a", mode: "manual", state: "complete", revision, controlRevision: 1, startedAt: "2026-10-13T00:00:00Z", requestedLookbackSeconds: 15, availableLookbackSeconds: 2, fragmentCount: 1, bytes: 10, gapCount: 0, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video" });
describe("imagination capture presentation", () => {
  it("validates millisecond-precision windows and permits zero only for record-now", () => {
    expect(parseCaptureSeconds("0", true)).toBe(0); expect(parseCaptureSeconds("15.125")).toBe(15.125); expect(parseCaptureSeconds("300")).toBe(300);
    for (const value of ["0", "301", "-1", "NaN", "1e2", "15.1234"]) expect(() => parseCaptureSeconds(value)).toThrow();
    expect(captureTime(61.9)).toBe("1m 1s"); expect(captureTime(NaN)).toBe("Unavailable");
  });
  it("keeps older pages, rejects another owner and never regresses a newer receipt", () => {
    const old = recording("old", 5), fresh = { ...recording("new"), sequence: 2 };
    const values = mergeCaptureRecordings([old], [recording("old", 2), fresh, { ...recording("foreign"), brainId: "brain-b" }], "brain-a");
    expect(values.map(value => value.id)).toEqual(["new", "old"]); expect(values[1].revision).toBe(5);
    expect(captureRecordingState({ ...old, state: "dormant" })).toBe("Saved · paused");
    expect(captureRecordingState({ ...old, state: "cancelled" })).toContain("saved fragments kept");
  });
  it("does not mix fragment pages from another recording", () => {
    const fragment = { id: "one", index: 0, recordingId: "old", available: false } as ImaginationCapturedFragment;
    expect(mergeCaptureFragments([fragment], [{ ...fragment, id: "new", recordingId: "new" }], "old")).toEqual([fragment]);
  });
  it("does not start, arm, query or render imaginary media during server rendering", () => {
    const api = new Proxy({}, { get: () => { throw new Error("Unexpected capture operation"); } }) as ImaginationCaptureApi;
    const html = renderToStaticMarkup(<ImaginationCaptureControls brainId="brain-a" api={api} />);
    expect(html).toContain("Imagination recording"); expect(html).toContain("Loading recording controls");
    expect(html).not.toContain("<video"); expect(html).not.toContain("<audio"); expect(html).not.toContain("<img");
    expect(html).not.toContain("checked="); expect(html).not.toContain("Cancel");
  });
});
