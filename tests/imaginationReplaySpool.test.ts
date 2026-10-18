import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, appendFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { expect, it } from "vitest";
import { buildReplaySpool, ReplaySpool, verifyReplaySpool } from "../src/main/imaginationReplaySpool";
import type { ImaginationRecording } from "../src/shared/imaginationCapture";

async function fixture(count: number, end: number) {
  const root = await mkdtemp(join(tmpdir(), "omni-replay-spool-")), at = "2026-10-13T12:00:00Z";
  const recording: ImaginationRecording = { format: "omni-imagination-recording", version: 1, id: "recording", brainId: "brain", sequence: 1, requestId: "record", mode: "manual", state: "complete", revision: count + 2, controlRevision: count + 2, startedAt: at, requestedLookbackSeconds: 0, availableLookbackSeconds: 0, fragmentCount: count, bytes: count, gapCount: 2, representation: "verified-delivered-fragment-timeline", label: "Delivered fragments — not a completed video", timelineEndMs: end };
  const db = new DatabaseSync(join(root, "source.sqlite3")); db.exec("CREATE TABLE fragments(recording_id TEXT,fragment_index INTEGER,body TEXT,sha TEXT); BEGIN IMMEDIATE");
  for (let index = 0; index < count; index++) {
    const value = { id: `fragment-${index}`, recordingId: recording.id, index, streamId: "stream", observedAt: at, offsetMs: index, origin: { kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic" }, kind: "preview-fragment", modality: "image", mimeType: "image/png", sha256: "1".repeat(64), bytes: 1,
      mediaTimeline: { format: "omni-native-media-timeline", version: 1, producer: "native-byte-encoder", layout: "snapshot", unit: "image", start: 0, count: 1, timebaseNumerator: 1, timebaseDenominator: 1 },
      deliveryAnchor: { version: 1, kind: "first-verified-main-delivery", firstObservedAt: at, elapsedMs: index, truncatedStart: false } };
    const body = JSON.stringify(value); db.prepare("INSERT INTO fragments VALUES(?,?,?,?)").run(recording.id, index, body, createHash("sha256").update(body).digest("hex"));
  }
  db.exec("COMMIT"); db.close(); return { root, recording };
}
it("streams thousands of revisions to a sealed SQLite/JSON manifest and exposes only bounded pages", async () => {
  const { root, recording } = await fixture(2001, 2100); let largest = 0, inspections = 0;
  try {
    const result = await buildReplaySpool({ directory: root, recording, refreshHz: 60, signal: new AbortController().signal, reserve: async () => undefined, admit: async bytes => { largest = Math.max(largest, bytes); },
      inspect: async () => { inspections++; return { path: "owned-fixture", mimeType: "image/png", width: 32, height: 16, frames: 1, durationMs: 40, hasAudio: false }; } });
    expect(inspections).toBe(1); expect(result.summary.fragmentCount).toBe(2001); expect(result.summary.options.fps).toBe(60); expect(result.summary.presentation.supersededInSameFrameCount).toBeGreaterThan(1000); expect(largest).toBeLessThan(1024 * 1024);
    await verifyReplaySpool(root, result.seal); const spool = new ReplaySpool(root, result.summary);
    try { expect(spool.page("fragments", -1, 100).entries).toHaveLength(100); expect(spool.page("fragments", 1999, 100).entries).toHaveLength(1); expect(() => spool.page("fragments", -1, 101)).toThrow(/bounded/); let count = 0; for (const _row of spool.fragments()) count++; expect(count).toBe(2001); } finally { spool.close(); }
    // Small developer fixture readback only; production never loads this whole array.
    expect(JSON.parse(await readFile(join(root, "manifest.json"), "utf8")).fragments).toHaveLength(2001);
    await appendFile(join(root, "manifest.json"), "\n"); await expect(verifyReplaySpool(root, result.seal)).rejects.toThrow(/changed/);
  } finally { await rm(root, { recursive: true, force: true }); }
}, 15000);
it("checkpoints an hour-long held state without imposing the five-minute lookback limit on recordings", async () => {
  const { root, recording } = await fixture(1, 3600000);
  try {
    const result = await buildReplaySpool({ directory: root, recording, refreshHz: 120, signal: new AbortController().signal, reserve: async () => undefined, admit: async () => undefined,
      inspect: async () => ({ path: "owned-fixture", mimeType: "image/png", width: 32, height: 16, frames: 1, durationMs: 40, hasAudio: false }) });
    expect(result.summary.durationMs).toBe(3600000); expect(result.summary.spanCount).toBe(360); const spool = new ReplaySpool(root, result.summary); try { expect(spool.span(359).endMs).toBe(3600000); expect(spool.span(0).endMs).toBe(10000); } finally { spool.close(); }
  } finally { await rm(root, { recursive: true, force: true }); }
});
