import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, writeFile, rm, readdir } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ImaginationCaptureService, type DeliveredImaginationFragment } from "../src/main/imaginationCapture";
import { ImaginationCaptureStore } from "../src/main/imaginationCaptureStore";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { mergeImaginationWorkspace, validateImaginationWorkspace } from "../src/main/imaginationCaptureWorkspace";

describe("complete unsanitized imagination custody (no model)", () => {
  let root: string, source: string, now: number, capture: ImaginationCaptureService;
  const services: ImaginationCaptureService[] = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-imagination-custody-")); source = join(root, "source"); now = 0; await managed(source, "source"); capture = service(source); });
  afterEach(async () => { for (const instance of services.splice(0)) await instance.dispose(); await rm(root, { recursive: true, force: true }); });
  async function managed(path: string, id: string) { await mkdir(join(path, "engine"), { recursive: true }); await writeFile(join(path, "brain.json"), JSON.stringify({ id })); }
  function service(path: string) { const result = new ImaginationCaptureService({ brainDirectory: () => path, now: () => now, wall: () => new Date(Date.parse("2026-10-13T12:00:00Z") + now).toISOString(),
    reserve: async () => undefined, admit: () => undefined, mediaUrl: async (_id, sha) => `omni-media://fixture/${sha}` }); services.push(result); return result; }
  async function fragment(index: number): Promise<DeliveredImaginationFragment> { const bytes = Buffer.from(`Synthetic admitted native reference ${index}`), path = join(root, `source-${index}.data`); await writeFile(path, bytes);
    return { brainId: "source", streamId: "job:source:owned", origin: { kind: "spontaneous", authority: "main-organic-action", actionEventId: "owned-action" }, modality: "image", kind: "preview-fragment", revision: index,
      resolve: async () => ({ path, sha256: createHash("sha256").update(bytes).digest("hex"), mimeType: "image/png", size: bytes.length }) }; }

  it("exports admitted live window and active recording, then keeps copied clocks and pending state dormant until explicit action", async () => {
    await capture.delivered(await fragment(0)); now = 3000; const recording = await capture.record({ brainId: "source", requestId: "pending-record", lookbackSeconds: 3 });
    await capture.delivered(await fragment(1)); const archive = (await snapshotLearningWorkspace(source, join(root, "archive"), "source"))!;
    const target = join(root, "copy"); await managed(target, "copy"); await adoptLearningWorkspace(archive, target, "copy");
    const copied = service(target); const first = await copied.status("copy"); expect(first.bufferedFragments).toBe(2); expect(first.settings.dormant).toBe(true);
    expect((await copied.recordings("copy")).entries[0]).toMatchObject({ id: recording.id, state: "dormant", fragmentCount: 2 });
    now += 600_000; const paused = await copied.status("copy"); expect(paused.bufferedFragments).toBe(2); expect(paused.availableLookbackSeconds).toBe(first.availableLookbackSeconds);
    const store = (await ImaginationCaptureStore.open(target, "copy"))!; const window = store.window(); store.close();
    for (const value of window) expect(await readFile(join(target, "imagination", "window-blobs", `${value.sha256}.data`))).toEqual(await readFile(join(root, `source-${value.source.revision}.data`)));
    const dormant = (await copied.recordings("copy")).entries[0]!; await copied.resume({ brainId: "copy", recordingId: dormant.id, expectedRevision: dormant.revision });
    copied.activate("copy"); const fresh = await fragment(2); await copied.delivered({ ...fresh, brainId: "copy", streamId: "job:copy:new-owned" });
    expect((await copied.status("copy")).recording?.fragmentCount).toBe(3);
    expect((await capture.status("source")).recording?.state).toBe("recording");
  });

  it("snapshot waits for admitted pending fragment IO instead of omitting it, and Fresh alone retires unrecorded window", async () => {
    let release!: () => void; const held = new Promise<void>(resolve => { release = resolve; }), actual = await fragment(0);
    const pending = capture.delivered({ ...actual, resolve: async () => { await held; return actual.resolve(); } });
    await Promise.resolve(); let finished = false;
    const snapshot = snapshotLearningWorkspace(source, join(root, "drained"), "source").then(value => { finished = true; return value; });
    await Promise.resolve(); expect(finished).toBe(false); release(); await pending; const archive = (await snapshot)!;
    const target = join(root, "target"); await managed(target, "target"); await adoptLearningWorkspace(archive, target, "target");
    const copied = service(target); expect((await copied.status("target")).bufferedFragments).toBe(1);
    await capture.release("source", true); expect((await capture.status("source")).bufferedFragments).toBe(0); expect((await capture.status("source")).settings.alwaysSave).toBe(false);
    expect((await copied.status("target")).bufferedFragments).toBe(1);
  });

  it("merges complete recordings and pending windows once for shared content while retaining raw database archives", async () => {
    const record = await capture.record({ brainId: "source", requestId: "retained-record", lookbackSeconds: 0 }); await capture.delivered(await fragment(0));
    const archive = (await snapshotLearningWorkspace(source, join(root, "archive"), "source"))!, target = join(root, "joined"); await managed(target, "joined");
    for (const parentId of ["parent-a", "parent-b"]) { const stage = join(root, parentId); await managed(stage, "joined"); await adoptLearningWorkspace(archive, stage, "joined");
      await mergeImaginationWorkspace(stage, target, "joined", parentId, async () => undefined); }
    const store = (await ImaginationCaptureStore.open(target, "joined"))!;
    try { store.validate(); expect(store.list().total).toBe(1); expect(store.list().entries[0]).toMatchObject({ state: "dormant", fragmentCount: 1, inheritedFrom: { recordingId: record.id } }); expect(store.window()).toHaveLength(1); }
    finally { store.close(); }
    expect((await readdir(join(target, "imagination", "archives"))).length).toBeGreaterThan(0); await validateImaginationWorkspace(target, "joined");
    const second = (await snapshotLearningWorkspace(target, join(root, "second"), "joined"))!, copy = join(root, "second-copy"); await managed(copy, "second-copy"); await adoptLearningWorkspace(second, copy, "second-copy");
    await validateImaginationWorkspace(copy, "second-copy"); expect((await service(copy).recordings("second-copy")).total).toBe(1);
  });
});
