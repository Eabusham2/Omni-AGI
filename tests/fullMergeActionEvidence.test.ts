import { createHash, randomUUID } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { adoptFullMergeActionEvidence } from "../src/main/fullMergeActionEvidence";
import { CompletedActionEvidenceStore, type CompletedActionEvidenceJob } from "../src/main/completedActionEvidence";
import type { FullMergeAssemblyInput, FullMergeStoredFile } from "../src/main/fullBrainMerge";
import { DEFAULT_CONFIG } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const sha = (value: string): string => createHash("sha256").update(value).digest("hex");

it("rekeys distinct result learning, preserves native cursors, settled receipts and orphan bytes without dispatching effects", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-result-merge-")); roots.push(root);
  await mkdir(join(root, "engine"), { recursive: true });
  const files: FullMergeStoredFile[] = [], queue: unknown[] = [];
  const oldId = sha("same-original-id");
  for (const parentId of ["a", "b", "c"]) {
    const engine = join(root, "merge", "parents", parentId, "engine");
    await mkdir(join(engine, "action-result-learning", "evidence"), { recursive: true });
    const executionId = randomUUID();
    const provenance = { format: "omni-completed-action-result-v1", originBrainId: parentId,
      actionEventId: "old-action", toolId: "web.fetch", action: "fetch", executionState: "complete", completedAt: "2026-10-01T00:00:00.000Z" };
    const raw = JSON.stringify({ ...provenance, evidenceId: oldId, executionId, startedAt: provenance.completedAt,
      arguments: { url: "https://example.invalid" }, outputPresent: true, output: { exactPrivateText: parentId } }) + "\n";
    const evidenceSha256 = sha(raw);
    const path = `engine/action-result-learning/evidence/${oldId}.jsonl`;
    await writeFile(join(engine, "action-result-learning", "evidence", `${oldId}.jsonl`), raw);
    files.push({ parentId, path, sha256: evidenceSha256, bytes: Buffer.byteLength(raw), mode: 0o600 });
    const job: CompletedActionEvidenceJob = { format: "omni-action-result-learning-job-v1", brainId: parentId, evidenceId: oldId,
      executionId, evidenceSha256, state: parentId === "b" ? "complete" : "pending", createdAt: provenance.completedAt,
      updatedAt: provenance.completedAt, attempts: 1, provenance, routeComplete: parentId !== "a",
      ...(parentId === "a" ? { route: { eventId: "12345678-1234-4234-8234-123456789012", utterance: "Read the chosen page",
        toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.invalid" } } } : {}),
      ...(parentId === "b" ? { receipt: { brainId: parentId, evidenceId: oldId, executionId, evidenceSha256,
        committed: true as const, processed: true, duplicate: false } } : {}) };
    const source = new CompletedActionEvidenceStore(engine, parentId, async () => undefined);
    if (parentId !== "c") await source.save(job); // immutable result captured just before source queue publication
    const completed = parentId === "c" ? [{ format: "omni-completed-ingestion", formatVersion: 1, transactionId: oldId,
      contentHash: evidenceSha256, sourceNameHash: sha(`action-result-${oldId}`), policy: "pretrain", epoch: 0,
      coverage: { complete: true }, sourceIdentity: sha("source"), parameterChecksumAfter: sha("committed") }] : [];
    await writeFile(join(engine, "brain.json"), JSON.stringify({ completed_ingestions: completed }));
    if (parentId === "a") queue.push({ id: sha("queued-a"), parentId, status: "queued", checkpoint: {
      transactionId: oldId, contentHash: evidenceSha256, sourceNameHash: sha(`action-result-${oldId}`),
      policy: "pretrain", epoch: 0, committedRecords: 7, activeRecordWindow: { fragment: "exact" }
    } });
  }
  const metadata = { connected_regions: { parents: ["a", "b", "c"].map((parentId) => ({ parentId, regionId: `r-${parentId}` })) }, connected_pending_ingestions: queue };
  await writeFile(join(root, "engine", "brain.json"), JSON.stringify(metadata));
  const input: FullMergeAssemblyInput = { targetBrainId: "merged", targetDirectory: root,
    request: { operationId: "fixture", parentIds: ["a", "b", "c"], name: "Merged", config: DEFAULT_CONFIG },
    parents: ["a", "b", "c"].map((brainId) => ({ brainId, name: brainId, ancestry: [], generationSha256: sha(brainId) })),
    files: (id) => files.filter((file) => !id || file.parentId === id), blobPath: () => { throw new Error("Unused"); },
    materializeParent: async (id) => join(root, "merge", "parents", id), reserve: async () => undefined, checkpoint: async () => undefined };
  await adoptFullMergeActionEvidence(input);
  const store = new CompletedActionEvidenceStore(join(root, "engine"), "merged", async () => undefined);
  expect(await store.pendingCount()).toBe(1);
  const [pending] = await store.pending();
  expect(pending!.evidenceId).not.toBe(oldId);
  expect(pending!.mergeContinuation).toEqual({ pendingId: sha("queued-a"), originalEvidenceId: oldId, parentId: "a" });
  expect(pending!.route?.eventId).toMatch(/^[a-f0-9]{8}-[a-f0-9]{4}-8[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$/);
  expect(await readFile(store.evidencePath(pending!.evidenceId), "utf8")).toBe(await readFile(join(root, "merge", "parents", "a", "engine", "action-result-learning", "evidence", `${oldId}.jsonl`), "utf8"));
  await adoptFullMergeActionEvidence(input);
  expect(await store.pendingCount()).toBe(1);
  expect(JSON.parse(await readFile(join(root, "engine", "brain.json"), "utf8"))).toEqual(metadata);

  // A merged parent has remapped filenames but retains the original immutable
  // execution header. A second merge must not reinterpret that as a new effect.
  const nested = await mkdtemp(join(tmpdir(), "omni-result-remerge-")); roots.push(nested);
  await mkdir(join(nested, "merge", "parents"), { recursive: true });
  await cp(root, join(nested, "merge", "parents", "merged"), { recursive: true });
  await mkdir(join(nested, "merge", "parents", "other", "engine"), { recursive: true });
  await writeFile(join(nested, "merge", "parents", "other", "engine", "brain.json"), "{}");
  await mkdir(join(nested, "engine"), { recursive: true });
  await writeFile(join(nested, "engine", "brain.json"), JSON.stringify({
    connected_regions: { parents: [{ parentId: "merged", regionId: "nested-r" }, { parentId: "other", regionId: "other-r" }] },
    connected_pending_ingestions: [{ ...(queue[0] as object), id: sha("second-queue"), parentId: "merged" }]
  }));
  const nestedFiles: FullMergeStoredFile[] = [];
  for (const kind of ["jobs", "evidence"]) for (const name of await readdir(join(root, "engine", "action-result-learning", kind))) {
    const path = `engine/action-result-learning/${kind}/${name}`;
    const raw = await readFile(join(root, path), "utf8");
    nestedFiles.push({ parentId: "merged", path, sha256: sha(raw), bytes: Buffer.byteLength(raw), mode: 0o600 });
  }
  await adoptFullMergeActionEvidence({ ...input, targetBrainId: "twice", targetDirectory: nested,
    parents: ["merged", "other"].map((brainId) => ({ brainId, name: brainId, ancestry: [], generationSha256: sha(brainId) })),
    files: (id) => nestedFiles.filter((file) => !id || file.parentId === id) });
  const twice = new CompletedActionEvidenceStore(join(nested, "engine"), "twice", async () => undefined);
  const [continued] = await twice.pending();
  expect(await twice.pendingCount()).toBe(1);
  expect(continued!.mergeContinuation).toEqual({ pendingId: sha("second-queue"), originalEvidenceId: oldId, parentId: "merged" });
  expect(continued!.evidenceId).not.toBe(pending!.evidenceId);
  expect(await readFile(twice.evidencePath(continued!.evidenceId), "utf8")).toBe(await readFile(store.evidencePath(pending!.evidenceId), "utf8"));
});
