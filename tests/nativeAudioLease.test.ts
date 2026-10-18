import { describe, expect, it, vi } from "vitest";
import { RuntimeJobManager } from "../src/main/brainService";
import type { NativeAudioCaptureLeaseRequest } from "../src/main/nativeAudioChat";

function fixture() {
  const record = { session: { id: "mic-lease", brainId: "audio-brain", state: "active", modalities: ["audio"], maxPacketBytes: 1000,
    permission: { granted: true, scope: "session", source: "microphone" } }, workerPid: 101 };
  const requireInstanceLoaded = vi.fn();
  const owner = { observationSessions: new Map([["mic-lease", record]]), service: { requireInstanceLoaded }, engine: { pid: 101 } } as unknown as RuntimeJobManager;
  const request: NativeAudioCaptureLeaseRequest = { brainId: "audio-brain", turnId: "audio-turn", sessionId: "mic-lease", utteranceId: "clip-one",
    sequence: 0, capturedAt: "2026-10-11T14:00:00Z", bytes: 100, contentSha256: "a".repeat(64) };
  const check = (value = request) => RuntimeJobManager.prototype.requireNativeAudioCaptureLease.call(owner, value);
  return { record, owner, request, check, requireInstanceLoaded };
}

describe("current per-brain microphone lease (constructor-free host)", () => {
  it("renderer-supplied session strings cannot create authority", () => {
    const f = fixture();
    expect(() => f.check({ ...f.request, sessionId: "invented" })).toThrow(/current active/);
    expect(() => f.check({ ...f.request, brainId: "another-mind" })).toThrow(/current active/);
    expect(f.check().permissionRevision).toBe(1); expect(f.requireInstanceLoaded).toHaveBeenCalled();
  });
  it("inactive, permission-revoked and replaced-worker leases cannot dispatch", () => {
    const f = fixture(); f.record.session.state = "stopped";
    expect(() => f.check()).toThrow(/current active/);
    f.record.session.state = "active"; f.record.session.permission.granted = false;
    expect(() => f.check()).toThrow(/current active/);
    f.record.session.permission.granted = true; f.record.workerPid = 102;
    expect(() => f.check()).toThrow(/current active/);
  });
  it("resource envelope is checked before decoding and exact retry does not create another claim", () => {
    const f = fixture();
    expect(() => f.check({ ...f.request, bytes: 1001, contentSha256: "" })).toThrow(/byte envelope/);
    expect(f.check({ ...f.request, contentSha256: "" }).maxBytes).toBe(1000);
    expect(f.check()).toEqual(f.check());
    expect(() => f.check({ ...f.request, turnId: "different-turn" })).toThrow(/different audio bytes/);
    expect(f.check({ ...f.request, sequence: 1, utteranceId: "clip-two" }).permissionRevision).toBe(1);
    expect(f.check()).toEqual({ permissionRevision: 1, maxBytes: 1000 });
  });
});
