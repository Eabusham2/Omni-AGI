import { createHash } from "node:crypto";
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DatasetManifestStore, CrawlFrontierStore } from "../src/main/dataIngestion";
import { snapshotLearningWorkspace, adoptLearningWorkspace, learningWorkspaceArchive } from "../src/main/learningWorkspacePortability";
import { LearningTaskStore, learningHash } from "../src/main/learningTaskStore";
import { makeWebLearningPolicy } from "../src/main/webLearningPolicy";
import { acquireCrawlSourceLease } from "../src/main/crawlSourceLease";
import { openLearningRelocations } from "../src/main/learningSourceRelocation";
import { adoptFullMergeLearningWorkspace } from "../src/main/fullMergeLearningWorkspace";

describe("identity-safe unsanitized learning workspace portability", () => {
  let root: string, source: string, target: string, datasets: DatasetManifestStore;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-learning-portability-")); source = join(root, "source"); target = join(root, "target");
    await mkdir(join(source, "engine"), { recursive: true }); await mkdir(join(target, "engine"), { recursive: true });
    await writeFile(join(source, "engine", "core.safetensors"), "native-fixture"); await writeFile(join(target, "engine", "core.safetensors"), "native-fixture");
    datasets = new DatasetManifestStore(() => source);
  });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  it("copies WAL task commits and exact sources, rekeys owner, retains cursors, and supports another fork", async () => {
    const path = join(root, "selected.jsonl"); await writeFile(path, '{"text":"a whole experience"}\n{"text":"its suffix"}\n');
    const manifest = await datasets.create("source", [path]); const before = await datasets.progress("source", manifest.id);
    const canonicalEntries = []; for await (const entry of datasets.entries("source", manifest.id)) canonicalEntries.push(entry);
    const tasks = await LearningTaskStore.open(source, "source");
    tasks.create({ format: "omni-learning-task", version: 1, id: "task", brainId: "source", requestId: "request", requestSha256: learningHash("request"), kind: "specific-learn", state: "running",
      createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), revision: 1, policy: makeWebLearningPolicy({}, "Read it", 1), sources: [], epochs: 1, phase: "sources", label: "Learning", pendingQuestions: [], traversalComplete: false },
      [{ id: "selected", kind: "dataset", label: "selected.jsonl", state: "running", manifestId: manifest.id, spec: { manifestSha256: manifest.manifestHash } }]);
    tasks.bindNativeTurn("task", "retained-turn");
    const archive = await snapshotLearningWorkspace(source, join(root, "archive"), "source"); tasks.close();
    expect(archive?.missingSources).toBe(0); await rm(path);
    await adoptLearningWorkspace(archive!, target, "target");
    const adopted = await LearningTaskStore.open(target, "target");
    expect(adopted.get("task")).toMatchObject({ brainId: "target", state: "paused" }); expect(adopted.nativeTurn("retained-turn")?.taskId).toBe("task"); adopted.close();
    const retained = new DatasetManifestStore(() => target), entries = [];
    for await (const entry of retained.entries("target", manifest.id)) entries.push(entry);
    expect(await readFile(entries[0]!.path, "utf8")).toContain("its suffix"); expect(entries[0]!.sourceRelocation?.originalPath).toBe(canonicalEntries[0]!.path);
    expect(await retained.progress("target", manifest.id)).toEqual(before);
    const second = await snapshotLearningWorkspace(target, join(root, "second"), "target");
    const third = join(root, "third"); await mkdir(join(third, "engine"), { recursive: true }); await writeFile(join(third, "engine", "core.safetensors"), "native-fixture");
    await adoptLearningWorkspace(second!, third, "third");
    const thirdStore = new DatasetManifestStore(() => third);
    for await (const entry of thirdStore.entries("third", manifest.id)) expect(await readFile(entry.path, "utf8")).toContain("whole experience");
    const different = join(root, "different"); await mkdir(join(different, "engine"), { recursive: true });
    await writeFile(join(different, "engine", "core.safetensors"), "native-fixture");
    await writeFile(join(different, "engine", "brain.json"), JSON.stringify({ substrate: { persistence: { contentSha256: "b".repeat(64) } } }));
    await adoptLearningWorkspace(archive!, different, "different");
    const blocked = await LearningTaskStore.open(different, "different");
    try { expect(blocked.get("task").resumeBlocked?.reason).toBe("neural-generation-mismatch"); }
    finally { blocked.close(); }
  });
  it("retains real web lease/crawl bytes and rejects unlisted payloads", async () => {
    const lease = await acquireCrawlSourceLease({ directory: join(source, "datasets", "web-cache"), url: "https://example.org/chapter", kind: "text", text: "source bytes", extension: ".txt",
      contentHash: createHash("sha256").update("source bytes").digest("hex"), contentType: "text/plain" });
    const crawl = await CrawlFrontierStore.create(source, "source", lease.url, "crawl"); crawl.close();
    const archive = await snapshotLearningWorkspace(source, join(root, "archive"), "source");
    await adoptLearningWorkspace(archive!, target, "target");
    const relocations = await openLearningRelocations(target, "target");
    try { expect(relocations?.generic(join(target, "datasets", "web-cache", lease.file))).toMatchObject({ originalPath: lease.path, sourceName: lease.url }); }
    finally { relocations?.close(); }
    const database = new DatabaseSync(join(target, "datasets", "crawls", "crawl.sqlite3"), { readOnly: true });
    try { expect(database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value).toBe("target"); } finally { database.close(); }
    await writeFile(join(archive!.root, "blobs", `${"0".repeat(64)}.data`), "unlisted");
    await expect(learningWorkspaceArchive(archive!.root)).rejects.toThrow(/unlisted payload/);
  });
  it("never accepts an interrupted inventory as a complete empty workspace", async () => {
    await mkdir(join(source, "learning", "inputs"), { recursive: true }); await writeFile(join(source, "learning", "inputs", "data.txt"), "retained data");
    const packageRoot = join(root, "interrupted");
    await expect(snapshotLearningWorkspace(source, packageRoot, "source", { checkpoint: async () => { throw new Error("fixture interrupted"); } })).rejects.toThrow(/interrupted/);
    await expect(learningWorkspaceArchive(packageRoot)).rejects.toMatchObject({ code: "LEARNING_WORKSPACE_INCOMPLETE" });
  });

  it("namespaces two parents' executable tasks/sources and binds only the declared ancestral frontier", async () => {
    const parents = [];
    for (const id of ["a", "b"]) {
      const parentRoot = join(root, id); await mkdir(join(parentRoot, "engine"), { recursive: true });
      await writeFile(join(parentRoot, "engine", "core.safetensors"), "native-fixture");
      const data = join(root, `${id}.jsonl`); await writeFile(data, `{"text":"${id} distinct experience"}\n`);
      const manifest = await new DatasetManifestStore(() => parentRoot).create(id, [data]);
      const store = await LearningTaskStore.open(parentRoot, id);
      store.create({ format: "omni-learning-task", version: 1, id: "shared-task-id", brainId: id, requestId: "same-request-id", requestSha256: learningHash(id), kind: "specific-learn",
        state: "paused", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), revision: 1, policy: makeWebLearningPolicy({}, "Read it", 1), sources: [], epochs: 1,
        phase: "sources", label: "Paused", pendingQuestions: [], traversalComplete: false,
        ...(id === "a" ? { retainedNativeTurn: { turnId: "original-turn", inputSha256: "a".repeat(64) } } : {}) },
        [{ id: "source", kind: "dataset", label: `${id}.jsonl`, state: "paused", manifestId: manifest.id, spec: { manifestSha256: manifest.manifestHash } }]);
      if (id === "a") store.bindNativeTurn("shared-task-id", "original-turn"); store.close();
      await snapshotLearningWorkspace(parentRoot, join(parentRoot, "portable-learning-workspace"), id);
      parents.push({ brainId: id, name: id, generationSha256: learningHash(`generation-${id}`), ancestry: [] });
      await rm(data);
    }
    await writeFile(join(target, "engine", "brain.json"), JSON.stringify({ connected_executions: { entries: [
      { parentId: "a", originalTurnId: "original-turn", rootTurnId: "ancestral-root-turn", id: "ancestral-execution" }
    ] } }));
    await adoptFullMergeLearningWorkspace({ targetBrainId: "target", targetDirectory: target, parents,
      request: {} as never, files: () => [], blobPath: () => { throw new Error("not used"); }, materializeParent: async id => join(root, id), reserve: async () => undefined, checkpoint: async () => undefined });
    const store = await LearningTaskStore.open(target, "target");
    try {
      const tasks = store.list(); expect(tasks).toHaveLength(2); expect(new Set(tasks.map(task => task.id)).size).toBe(2);
      const inherited = tasks.find(task => task.inheritedFrom?.parentId === "a")!;
      expect(inherited).toMatchObject({ state: "paused", retainedNativeTurn: { turnId: "ancestral-root-turn" }, inheritedFrom: { taskId: "shared-task-id", executionId: "ancestral-execution" } });
      expect(store.nativeTurn("ancestral-root-turn")?.taskId).toBe(inherited.id);
      const manifests = new DatasetManifestStore(() => target);
      for (const task of tasks) for await (const entry of manifests.entries("target", task.sources[0]!.manifestId!))
        expect(await readFile(entry.path, "utf8")).toContain(`${task.inheritedFrom!.parentId} distinct experience`);
    } finally { store.close(); }
  });
});
