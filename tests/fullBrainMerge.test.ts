import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CONFIG } from "../src/shared/types";
import { FullBrainMergeCoordinator, type FullMergeAssemblyInput, type FullMergeHooks,
  type FullMergeRequest } from "../src/main/fullBrainMerge";

const cleanup: string[] = [];
afterEach(async () => { for (const path of cleanup.splice(0)) await rm(path, { recursive: true, force: true }); });

const sha = (data: string | Buffer): string => createHash("sha256").update(data).digest("hex");
const request = (): FullMergeRequest => ({ operationId: "merge-one", parentIds: ["a", "b"],
  name: "Connected", config: { ...DEFAULT_CONFIG, name: "Connected" } });

async function fixture(): Promise<{ root: string; hooks: FullMergeHooks; observations: FullMergeAssemblyInput[] }> {
  const root = await mkdtemp(join(tmpdir(), "omni-full-merge-"));
  cleanup.push(root);
  for (const parent of ["a", "b"]) {
    await mkdir(join(root, parent, "engine"), { recursive: true });
    await writeFile(join(root, parent, "engine", "core.safetensors"), "same-packed-weights");
    await writeFile(join(root, parent, "engine", "brain.json"), JSON.stringify({ brain_id: parent, revision: 1 }));
    // Same event ID with different actual content must never be collapsed.
    await writeFile(join(root, parent, "history.json"), JSON.stringify([{ id: "shared-id", fact: parent }]));
  }
  const observations: FullMergeAssemblyInput[] = [];
  const hooks: FullMergeHooks = {
    withParentsFrozen: async (_ids, action) => action(),
    sealParent: async (id) => ({ brainId: id, name: id, ancestry: ["a".repeat(64)],
      generationSha256: sha(await readFile(join(root, id, "engine", "brain.json"))) }),
    validateSettings: async () => undefined,
    reserve: async () => undefined,
    assemble: async (input) => {
      observations.push(input);
      await mkdir(join(input.targetDirectory, "engine"), { recursive: true });
      await writeFile(join(input.targetDirectory, "brain.json"), JSON.stringify({ id: input.targetBrainId }));
      await writeFile(join(input.targetDirectory, "engine", "brain.json"), JSON.stringify({ regions: input.parents.map((p) => p.brainId) }));
    },
    verify: async (input) => {
      const all = [...input.files()];
      if (all.length < 6) throw new Error("Incomplete parent snapshot");
      for (const file of all) expect(sha(await readFile(input.blobPath(file.sha256)))).toBe(file.sha256);
    },
  };
  return { root, hooks, observations };
}

describe("complete native merge transaction", () => {
  it("publishes a new instance, preserves parents, and deduplicates only exact bytes", async () => {
    const { root, hooks, observations } = await fixture();
    const before = await Promise.all(["a", "b"].map((id) => readFile(join(root, id, "engine", "brain.json"), "utf8")));
    const coordinator = new FullBrainMergeCoordinator(root, hooks);
    const done = await coordinator.start(request());
    expect(done.state).toBe("complete");
    expect(done.filesCopied).toBe(8); // six parent files and two sealed learning inventories
    expect(done.uniqueBlobBytes).toBeLessThan(done.copiedBytes);
    expect(done.targetBrainId).not.toBe("a");
    expect(observations).toHaveLength(1);
    expect(await Promise.all(["a", "b"].map((id) => readFile(join(root, id, "engine", "brain.json"), "utf8")))).toEqual(before);
    const index = (await readFile(join(root, done.targetBrainId, "merge", "parents.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
    const histories = index.filter((row) => row.path === "history.json");
    expect(histories).toHaveLength(2);
    expect(histories[0].sha256).not.toBe(histories[1].sha256);
    const weights = index.filter((row) => row.path === "engine/core.safetensors");
    expect(weights[0].sha256).toBe(weights[1].sha256);
  });

  it("resumes a paused copy using the same operation and never exposes a partial instance", async () => {
    const { root, hooks } = await fixture();
    let coordinator: FullBrainMergeCoordinator;
    let interrupted = false;
    hooks.onProgress = (progress) => {
      if (progress.filesCopied > 0 && !interrupted) { interrupted = true; coordinator.pause(progress.operationId); }
    };
    coordinator = new FullBrainMergeCoordinator(root, hooks);
    await expect(coordinator.start(request())).rejects.toThrow("paused");
    const paused = await coordinator.status("merge-one");
    expect(paused.state).toBe("paused");
    expect(await readdir(root)).not.toContain(paused.targetBrainId);
    const resumed = await coordinator.start(request());
    expect(resumed.state).toBe("complete");
    expect(resumed.targetBrainId).toBe(paused.targetBrainId);
    expect(resumed.filesCopied).toBe(8);
  });

  it("rejects a changed parent and rejects changed reviewed settings on resume", async () => {
    const { root, hooks } = await fixture();
    let coordinator: FullBrainMergeCoordinator;
    hooks.onProgress = (progress) => { if (progress.state === "copying" && progress.filesCopied > 0) coordinator.pause(progress.operationId); };
    coordinator = new FullBrainMergeCoordinator(root, hooks);
    await expect(coordinator.start(request())).rejects.toThrow();
    await expect(coordinator.start({ ...request(), name: "Different" })).rejects.toThrow("review a new");
    // Preserve the original job request after rejection.
    await writeFile(join(root, "a", "engine", "brain.json"), JSON.stringify({ brain_id: "a", revision: 2 }));
    await expect(coordinator.start(request())).rejects.toThrow("parent changed");
  });

  it("captures SQLite WAL rows consistently and keeps the source unchanged", async () => {
    const { root, hooks } = await fixture();
    const path = join(root, "a", "history.sqlite3");
    const db = new DatabaseSync(path);
    try {
      db.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE events (body TEXT); INSERT INTO events VALUES ('first'),('second')");
      const done = await new FullBrainMergeCoordinator(root, hooks).start(request());
      const index = (await readFile(join(root, done.targetBrainId, "merge", "parents.jsonl"), "utf8")).trim().split("\n").map((line) => JSON.parse(line));
      const entry = index.find((row) => row.path === "history.sqlite3");
      const saved = new DatabaseSync(join(root, done.targetBrainId, "merge", "blobs", entry.sha256), { readOnly: true });
      try { expect(saved.prepare("SELECT body FROM events").all()).toEqual([{ body: "first" }, { body: "second" }]); } finally { saved.close(); }
      expect(db.prepare("SELECT COUNT(*) AS n FROM events").get()?.n).toBe(2);
      expect(index.some((row) => String(row.path).endsWith("-wal"))).toBe(false);
    } finally { db.close(); }
  });

  it("does not publish failed assembly and explicit cancellation removes only its staging directory", async () => {
    const { root, hooks } = await fixture();
    hooks.assemble = async () => { throw new Error("native integration failed"); };
    const coordinator = new FullBrainMergeCoordinator(root, hooks);
    await expect(coordinator.start(request())).rejects.toThrow("native integration failed");
    const state = await coordinator.status("merge-one");
    expect(await readdir(root)).not.toContain(state.targetBrainId);
    await coordinator.cancel("merge-one");
    expect(await readdir(root)).toEqual(expect.arrayContaining(["a", "b"]));
    expect(await readdir(join(root, ".full-merges"))).toEqual([]);
  });

  it("recovers post-publication receipt retries without duplicate assembly", async () => {
    const { root, hooks } = await fixture();
    const assemble = vi.spyOn(hooks, "assemble");
    const coordinator = new FullBrainMergeCoordinator(root, hooks);
    const first = await coordinator.start(request());
    const second = await coordinator.start(request());
    expect(second).toEqual(first);
    expect(assemble).toHaveBeenCalledTimes(1);
    await expect(coordinator.start({ ...request(), name: "A different reviewed destination" })).rejects.toThrow("request changed");
    expect((await coordinator.status(request().operationId)).state).toBe("complete");
    await expect(coordinator.cancel("merge-one")).rejects.toThrow("published");
  });

  it("revalidates copied state before publication and leaves all parents intact", async () => {
    const { root, hooks } = await fixture();
    const original = hooks.assemble;
    hooks.assemble = async (input) => {
      await original(input);
      const file = [...input.files()][0]!;
      await writeFile(input.blobPath(file.sha256), "tampered");
    };
    const coordinator = new FullBrainMergeCoordinator(root, hooks);
    await expect(coordinator.start(request())).rejects.toThrow("changed parent bytes");
    const state = await coordinator.status("merge-one");
    expect(await readdir(root)).not.toContain(state.targetBrainId);
    expect(await readFile(join(root, "a", "engine", "core.safetensors"), "utf8")).toBe("same-packed-weights");
  });
});
