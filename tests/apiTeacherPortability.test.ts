/** Tiny SQLite custody only. No provider, credential, worker, model, or neural learning. */
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import {
  ApiTeacherStore,
  apiTeacherCanonical,
  apiTeacherSha256,
  type ApiTeacherStoredItem,
  type ApiTeacherStoredJob
} from "../src/main/apiTeacherStore";
import {
  mergeApiTeacherWorkspace,
  rekeyApiTeacherWorkspace,
  validateApiTeacherWorkspace
} from "../src/main/apiTeacherPortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function completeSource(root: string, owner: string): Promise<string> {
  const store = await ApiTeacherStore.open(root, owner, async () => undefined);
  if (!store) throw new Error("fixture store unavailable");
  const now = "2026-10-15T12:00:00.000Z", id = "teacher-source-job", prompt = "Explain a durable cursor.",
    response = "A durable cursor advances only after an authenticated commit.", learnedResponseSha256 = apiTeacherSha256(response);
  const trajectory = JSON.stringify({ format: "omni-api-teacher-trajectory", version: 1, deliveryId: "delivery-1",
    input: prompt, output: response }), experienceSha256 = apiTeacherSha256(trajectory);
  const job: ApiTeacherStoredJob = { format: "omni-api-teacher-job", version: 1, id, brainId: owner,
    originalBrainId: owner, originalJobId: id, requestId: "stable-request-1", requestSha256: apiTeacherSha256("request"),
    provider: "openai", model: "fixture-model", maxOutputTokens: 32, promptCount: 1, learned: 0, failed: 0,
    skipped: 0, nextIndex: 0, state: "queued", revision: 1, createdAt: now, updatedAt: now };
  const item: ApiTeacherStoredItem = { format: "omni-api-teacher-item", version: 1, jobId: id, index: 0, prompt,
    promptSha256: apiTeacherSha256(prompt), deliveryId: "delivery-1", state: "pending" };
  try {
    await store.createJob(job, [item]);
    await store.updateItem(id, 0, { state: "complete", attemptId: "attempt-1", requestSha256: apiTeacherSha256("provider-request"),
      rawResponseSha256: apiTeacherSha256(response), learnedResponseSha256, response, experienceSha256,
      receipt: { brainId: owner, sourceId: "source-1", contentHash: experienceSha256, learnedResponseSha256,
        deliveryId: "delivery-1", learnedIdeas: 1, learnedConcepts: 1, learnedSynapses: 2,
        learnedParameterSteps: 1, parametersChanged: true } }, "fixture-learning-committed");
    await store.updateJob(id, { learned: 1, nextIndex: 1, state: "complete" }, "fixture-complete");
  } finally { store.close(); }
  return response;
}

describe("API teacher portable dormant custody", () => {
  it("captures and adopts teacher history through the actual shared workspace path without rewriting sealed lineage bytes", async () => {
    const source = await mkdtemp(join(tmpdir(), "omni-api-teacher-hook-")), target = await mkdtemp(join(tmpdir(), "omni-api-teacher-copy-")),
      archiveRoot = await mkdtemp(join(tmpdir(), "omni-api-teacher-capture-")), secondArchive = await mkdtemp(join(tmpdir(), "omni-api-teacher-second-"));
    roots.push(source, target, archiveRoot, secondArchive);
    const response = await completeSource(source, "brain-a");
    const archive = await snapshotLearningWorkspace(source, archiveRoot, "brain-a");
    const inventory = new DatabaseSync(join(archiveRoot, "inventory.sqlite3"), { readOnly: true });
    try { expect(inventory.prepare("SELECT path FROM files WHERE path=?").get("api-teachers/workspace.sqlite3")).toBeDefined(); }
    finally { inventory.close(); }
    await adoptLearningWorkspace(archive!, target, "brain-b");
    await validateApiTeacherWorkspace(target, "brain-b");
    await snapshotLearningWorkspace(target, secondArchive, "brain-b");
    const nextInventory = new DatabaseSync(join(secondArchive, "inventory.sqlite3"), { readOnly: true });
    try { expect(nextInventory.prepare("SELECT path FROM files WHERE path GLOB 'api-teachers/archives/*.sqlite3'").get()).toBeDefined(); }
    finally { nextInventory.close(); }
    const store = await ApiTeacherStore.open(target, "brain-b", async () => undefined, { create: false });
    try { expect(store!.jobs()[0]).toMatchObject({ brainId: "brain-b", originalBrainId: "brain-a", state: "dormant" });
      expect(store!.items(store!.jobs()[0]!.id)[0]?.response).toBe(response); }
    finally { store?.close(); }
  });
  it("retains nested exact source databases while every copied current job remains dormant", async () => {
    const source = await mkdtemp(join(tmpdir(), "omni-api-teacher-source-")), target = await mkdtemp(join(tmpdir(), "omni-api-teacher-target-"));
    roots.push(source, target); const response = await completeSource(source, "brain-a"), reserved: number[] = [];
    await rekeyApiTeacherWorkspace(source, "brain-a", "brain-b", { checkDisk: async bytes => { reserved.push(bytes); } });
    await rekeyApiTeacherWorkspace(source, "brain-b", "brain-c", { checkDisk: async bytes => { reserved.push(bytes); } });
    await mergeApiTeacherWorkspace(source, target, "brain-c", "parent-a", async bytes => { reserved.push(bytes); });
    await validateApiTeacherWorkspace(target, "brain-c");
    const current = await ApiTeacherStore.open(target, "brain-c", async () => undefined, { create: false });
    if (!current) throw new Error("merged fixture store unavailable");
    try {
      expect(current.jobs()).toHaveLength(1);
      expect(current.jobs()[0]).toMatchObject({ brainId: "brain-c", originalBrainId: "brain-a", state: "dormant" });
      expect(current.items(current.jobs()[0]!.id)[0]).toMatchObject({ state: "dormant", response,
        receipt: { brainId: "brain-a" } });
      expect(current.lineages().length).toBeGreaterThanOrEqual(3);
      for (const record of current.lineages()) {
        const database = new DatabaseSync(join(target, ...record.relativePath.split("/")), { readOnly: true });
        try {
          const metadata = Object.fromEntries(database.prepare("SELECT key,value FROM meta").all().map(row => [String(row.key), String(row.value)]));
          expect(metadata.brainId).toBe(record.sourceBrainId);
        } finally { database.close(); }
      }
      const originals = current.lineages().filter(record => record.sourceBrainId === "brain-a");
      expect(originals).toHaveLength(1);
      const archive = new DatabaseSync(join(target, ...originals[0]!.relativePath.split("/")), { readOnly: true });
      try {
        const payload = String(archive.prepare("SELECT payload FROM items LIMIT 1").get()?.payload);
        expect(apiTeacherCanonical(JSON.parse(payload))).toBe(payload); expect(payload).toContain(response);
      } finally { archive.close(); }
    } finally { current.close(); }
    expect(reserved.length).toBeGreaterThan(0);
  });
});
