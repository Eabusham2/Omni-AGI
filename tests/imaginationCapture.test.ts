import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ImaginationCaptureService, type DeliveredImaginationFragment } from "../src/main/imaginationCapture";
import { ImaginationCaptureStore } from "../src/main/imaginationCaptureStore";
import { MediaArtifactRegistry } from "../src/main/mediaArtifactRegistry";
import { snapshotMutableSqliteIsolated } from "../src/main/mutableFileIsolation";

const png = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a7O8AAAAASUVORK5CYII=", "base64");
const sha = (data: Uint8Array) => createHash("sha256").update(data).digest("hex");
describe("actual fragment custody and rolling controls (synthetic native events, no model)", () => {
  let root: string, now: number, service: ImaginationCaptureService;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-imagination-capture-")); await mkdir(join(root, "engine", "artifacts"), { recursive: true }); await writeFile(join(root, "brain.json"), '{"id":"brain"}'); now = 0;
    service = new ImaginationCaptureService({ brainDirectory: () => root, now: () => now, wall: () => new Date(Date.parse("2026-10-13T12:00:00Z") + now).toISOString(),
      reserve: async () => undefined, admit: () => undefined, mediaUrl: async (_brain, hash) => `omni-media://fixture/${hash}` }); });
  afterEach(async () => { await service.dispose(); await rm(root, { recursive: true, force: true }); });
  async function fragment(index: number, kind: DeliveredImaginationFragment["kind"] = "preview-fragment"): Promise<DeliveredImaginationFragment> {
    const data = Buffer.concat([png, Buffer.from([index])]), path = join(root, `source-${index}.png`); await writeFile(path, data);
    return { brainId: "brain", streamId: "job:brain:organic-job", origin: { kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic-action" }, kind,
      modality: "image", revision: index, resolve: async () => ({ path, sha256: sha(data), mimeType: "image/png", size: data.length }) };
  }

  it("uses requested preset/custom windows, default always-save Off, and only verified spontaneous references", async () => {
    let status = await service.status("brain"); expect(status.settings).toMatchObject({ bufferSeconds: 300, alwaysSave: false, dormant: false });
    for (const seconds of [15, 30, 60, 180, 300, 12.5]) status = await service.settings({ brainId: "brain", expectedRevision: status.settings.revision, bufferSeconds: seconds, alwaysSave: false });
    await expect(service.settings({ brainId: "brain", expectedRevision: status.settings.revision, bufferSeconds: 301, alwaysSave: true })).rejects.toThrow(/at most 300/);
    const actual = await fragment(0), resolve = vi.fn(actual.resolve);
    await service.delivered({ ...actual, origin: { kind: "unclassified", authority: "native-brain-action" }, resolve });
    await service.delivered({ ...actual, origin: { kind: "user-requested", authority: "main-ui", requestId: "requested" }, resolve }); expect(resolve).not.toHaveBeenCalled();
    await service.delivered(actual); expect((await service.status("brain")).bufferedFragments).toBe(1); expect((await service.recordings("brain")).total).toBe(0);
    now = 13_000; expect((await service.status("brain")).bufferedFragments).toBe(0); expect(await readFile(join(root, "source-0.png"))).toEqual(Buffer.concat([png, Buffer.from([0])]));
  });

  it("back-records only available prior fragments and future fragments until Stop, without claiming a completed video", async () => {
    await service.delivered(await fragment(0)); now = 4000; await service.delivered(await fragment(1)); now = 6000;
    const record = await service.record({ brainId: "brain", requestId: "manual-record", lookbackSeconds: 5 });
    expect(record).toMatchObject({ fragmentCount: 1, requestedLookbackSeconds: 5, availableLookbackSeconds: 2, label: "Delivered fragments — not a completed video" });
    now = 9000; await service.delivered(await fragment(2, "final-artifact")); const active = (await service.status("brain")).recording!;
    expect(active.fragmentCount).toBe(2); const stopped = await service.stop({ brainId: "brain", recordingId: active.id, expectedRevision: active.revision }); expect(stopped.state).toBe("complete");
    now = 10000; await service.delivered(await fragment(3)); expect((await service.fragments("brain", record.id)).entries.map(value => value.revision)).toEqual([1, 2]);
    expect((await service.fragments("brain", record.id)).entries.map(value => value.offsetMs)).toEqual([0, 5000]);
    expect((await service.record({ brainId: "brain", requestId: "manual-record", lookbackSeconds: 5 })).id).toBe(record.id);
  });

  it("cancels future recording only, preserves committed blobs, and closes active state across unload without late recapture", async () => {
    const record = await service.record({ brainId: "brain", requestId: "record-cancel", lookbackSeconds: 0 }); await service.delivered(await fragment(0));
    const active = (await service.status("brain")).recording!; await service.cancel({ brainId: "brain", recordingId: record.id, expectedRevision: active.revision });
    const [saved] = (await service.fragments("brain", record.id)).entries; expect(await readFile(join(root, "imagination", "blobs", `${saved!.sha256}.data`))).toEqual(Buffer.concat([png, Buffer.from([0])]));
    await service.release("brain"); await service.delivered(await fragment(1)); expect((await service.status("brain")).bufferedFragments).toBe(1);
    expect((await service.recordings("brain")).entries[0]?.state).toBe("cancelled"); service.activate("brain"); await service.delivered(await fragment(2)); expect((await service.status("brain")).bufferedFragments).toBe(2);
  });

  it("always-save records only new delivered events and resource refusal pauses honestly without fabricated fragments", async () => {
    const status = await service.status("brain"); await service.settings({ brainId: "brain", expectedRevision: status.settings.revision, bufferSeconds: 300, alwaysSave: true });
    await service.delivered(await fragment(0)); const actual = (await service.status("brain")).recording!; expect(actual.mode).toBe("always-save");
    const bad = await fragment(1); await service.delivered({ ...bad, resolve: async () => { throw new Error("Synthetic native reference expired"); } });
    expect((await service.recordings("brain")).entries[0]).toMatchObject({ state: "resource-paused", fragmentCount: 1, gapCount: 1 }); expect((await service.status("brain")).autoPaused).toBe(true);
    await service.delivered(await fragment(2)); expect((await service.recordings("brain")).total).toBe(1);
    expect((await service.fragments("brain", actual.id)).entries).toHaveLength(1);
  });

  it("seals copied active recordings dormant while original exact fragments and hashes remain unchanged", async () => {
    const record = await service.record({ brainId: "brain", requestId: "copy-record", lookbackSeconds: 0 }); await service.delivered(await fragment(0));
    const store = (await ImaginationCaptureStore.open(root, "brain", true))!, before = store.fragments(record.id).entries; store.close();
    const copyRoot = join(root, "isolated-copy"); await mkdir(copyRoot); await snapshotMutableSqliteIsolated(join(root, "imagination", "workspace.sqlite3"), join(copyRoot, "imagination", "workspace.sqlite3"));
    const writer = (await ImaginationCaptureStore.open(copyRoot, "brain", true))!; writer.rekey("copied"); writer.close(); const copied = (await ImaginationCaptureStore.open(copyRoot, "copied"))!;
    try { copied.validate(); expect(copied.settings().dormant).toBe(true); expect(copied.get(record.id)).toMatchObject({ brainId: "copied", state: "dormant", inheritedFrom: { brainId: "brain" } }); expect(copied.fragments(record.id).entries).toEqual(before); }
    finally { copied.close(); }
  });

  it("allows Stop across new frame revisions but rejects a stale control from before explicit Resume", async () => {
    const initial = await service.record({ brainId: "brain", requestId: "frame-control", lookbackSeconds: 0 });
    await service.delivered(await fragment(0)); await service.delivered(await fragment(1));
    expect((await service.status("brain")).recording?.revision).toBeGreaterThan(initial.revision);
    expect((await service.stop({ brainId: "brain", recordingId: initial.id, expectedRevision: initial.revision })).state).toBe("complete");
    const next = await service.record({ brainId: "brain", requestId: "retained-control", lookbackSeconds: 0 }); await service.release("brain");
    const row = (await service.recordings("brain")).entries.find(value => value.id === next.id)!;
    const resumed = await service.resume({ brainId: "brain", recordingId: row.id, expectedRevision: row.revision });
    await expect(service.stop({ brainId: "brain", recordingId: row.id, expectedRevision: next.revision })).rejects.toThrow(/control changed/);
    expect((await service.stop({ brainId: "brain", recordingId: row.id, expectedRevision: resumed.revision })).state).toBe("complete");
  });

  it("persists an admitted-recording queue refusal as a real gap, without inventing a fragment", async () => {
    await service.dispose(); let refuse = false;
    service = new ImaginationCaptureService({ brainDirectory: () => root, reserve: async () => undefined, admit: () => { if (refuse) throw new Error("Synthetic queue budget"); }, mediaUrl: async () => "not-used" });
    const recording = await service.record({ brainId: "brain", requestId: "resource-gap", lookbackSeconds: 0 }); refuse = true; await service.delivered(await fragment(0)); refuse = false;
    const stopped = await service.stop({ brainId: "brain", recordingId: recording.id, expectedRevision: recording.revision });
    expect(stopped).toMatchObject({ state: "complete", gapCount: 1, fragmentCount: 0, bytes: 0 });
  });
});

it("registry keeps user/unknown finals, but gives genuine spontaneous final only an expiring lease and strips inline retained pixels", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-spontaneous-registry-")); await mkdir(join(root, "engine", "artifacts"), { recursive: true });
  const registry = new MediaArtifactRegistry(() => root), offers: DeliveredImaginationFragment[] = []; registry.setCaptureSink(async fragment => { offers.push(fragment); });
  try {
    const hash = sha(png), requested = join(root, "engine", "artifacts", `${hash}.png`); await writeFile(requested, png);
    registry.noteCaptureOrigin("job:brain:human", { kind: "user-requested", authority: "main-ui", requestId: "human" });
    await registry.completeJob("brain", "human", { path: requested, mimeType: "image/png", artifactSha256: hash, modality: "image" });
    const scratch = join(root, "engine", ".spontaneous-imagination", "c".repeat(64)); await mkdir(scratch, { recursive: true });
    const temporary = join(scratch, `${hash}.png`); await writeFile(temporary, png);
    await writeFile(join(scratch, "owner.json"), JSON.stringify({ format: "omni-spontaneous-output-owner", version: 1, brainId: "brain", jobId: "organic", leaseId: "c".repeat(64) }));
    registry.noteCaptureOrigin("job:brain:organic", { kind: "spontaneous", authority: "main-organic-action", actionEventId: "organic" });
    registry.noteSpontaneousOutputLease("brain", "organic", "c".repeat(64));
    const output = await registry.completeJob("brain", "organic", { path: temporary, mimeType: "image/png", artifactSha256: hash, modality: "image", dataUrl: `data:image/png;base64,${png.toString("base64")}` }) as Record<string, unknown>;
    expect(output.artifactRetention).toBe("ephemeral-spontaneous"); expect(output.dataUrl).toBeUndefined(); expect(output.path).toBeUndefined();
    expect((await registry.listArtifacts("brain")).totalArtifacts).toBe(1); expect(await readFile(requested)).toEqual(png); expect(offers).toHaveLength(1);
    expect((await offers[0]!.resolve()).sha256).toBe(hash);
  } finally { registry.dispose(); await rm(root, { recursive: true, force: true }); }
});
