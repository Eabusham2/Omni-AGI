import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { SPARSE_SUBSTRATE_FORMAT, SPARSE_SUBSTRATE_SCOPE, type SparseSubstrateBundle,
  type SparseSubstrateJob, type SparseSubstratePlan } from "../src/shared/sparseSubstrateTransfer";
import { SPARSE_TRANSFER_DIRECTORY, SparseSubstrateTransferStore } from "../src/main/sparseSubstrateTransferStore";
import { adoptSparseSubstrateTransferWorkspace, listSparseSubstrateLineage, mergeSparseSubstrateTransferWorkspace,
  resolveSparseSubstrateLineageArtifact, validateSparseSubstrateTransferWorkspace } from "../src/main/sparseSubstrateTransferPortability";

const roots: string[] = [];
const sha = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
const canonical = (value: unknown): string => {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
};
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });

async function directory(label: string): Promise<string> {
  const root = await mkdtemp(join(tmpdir(), `omni-sparse-portable-${label}-`)); roots.push(root); return root;
}

async function seeded(root: string, brainId: string, label: string, state: SparseSubstrateJob["state"]): Promise<void> {
  const store = await SparseSubstrateTransferStore.open(root, brainId);
  try {
    const bundleId = `bundle-${label}`, bundleBytes = Buffer.from(`bundle-bytes-${label}`), bundleSha = sha(bundleBytes);
    const bundleManifest = canonical({ format: SPARSE_SUBSTRATE_FORMAT, version: 1, scope: SPARSE_SUBSTRATE_SCOPE, rows: 2,
      sourceBrainId: brainId, sourceGenerationSha256: "1".repeat(64), cortexStateIncluded: false, sourceAuthorityIncluded: false });
    const bundle: SparseSubstrateBundle = { format: SPARSE_SUBSTRATE_FORMAT, version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      id: bundleId, sha256: bundleSha, bytes: bundleBytes.length, manifestJson: bundleManifest, manifestSha256: sha(bundleManifest),
      sourceBrainId: brainId, sourceGenerationSha256: "1".repeat(64), rows: 2, sourceBinding: "verified-owned-export",
      relativePath: `${SPARSE_TRANSFER_DIRECTORY}/bundles/${bundleId}.sqlite3` };
    await mkdir(join(root, SPARSE_TRANSFER_DIRECTORY, "bundles"), { recursive: true });
    await writeFile(join(root, bundle.relativePath), bundleBytes);
    store.putArtifact("bundle", bundle);

    const planId = `plan-${label}`, planBytes = Buffer.from(`plan-bytes-${label}`), planSha = sha(planBytes);
    const planManifest = canonical({ format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      rows: 2, brainId, bundleSha256: bundleSha, mode: "append" });
    const plan: SparseSubstratePlan = { format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      id: planId, brainId, bundleId, bundleSha256: bundleSha, mode: "append", sha256: planSha, bytes: planBytes.length,
      manifestJson: planManifest, manifestSha256: sha(planManifest), relativePath: `${SPARSE_TRANSFER_DIRECTORY}/plans/${planId}.sqlite3`,
      rows: 2, additions: 2, replacements: 0, identical: 0, conflicts: 0, targetGenerationSha256: "2".repeat(64),
      selectedPreimagesSha256: "3".repeat(64), changesSha256: "4".repeat(64) };
    await mkdir(join(root, SPARSE_TRANSFER_DIRECTORY, "plans"), { recursive: true });
    await writeFile(join(root, plan.relativePath), planBytes);
    store.putArtifact("plan", plan);

    const requestId = `request-${label}`, requestSha256 = sha(`request-${label}`);
    store.beginRequest(requestId, "apply", requestSha256); store.finishRequest(requestId, { id: `job-${label}`, planId });
    const now = "2026-10-15T00:00:00.000Z";
    const job: SparseSubstrateJob = { format: "omni-sparse-substrate-job", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      id: `job-${label}`, brainId, originalBrainId: brainId, requestId, requestSha256, planId, planSha256: planSha,
      authorizationRevision: `authority-${label}`, state, createdAt: now, updatedAt: now, revision: 1, liveAuthorityPersisted: false };
    store.createJob(job);
    const receiptBody = { format: "omni-sparse-substrate-applied", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      brainId, jobId: job.id, requestSha256, planSha256: planSha, bundleSha256: bundleSha,
      selectedPreimagesSha256: plan.selectedPreimagesSha256, changesSha256: plan.changesSha256, appliedRows: 2,
      committedAt: now, generationSha256: "5".repeat(64), duplicate: false, weightsAveraged: false,
      cortexParametersChanged: false, originChanged: false };
    const receiptJson = canonical(receiptBody), receiptSha256 = sha(receiptJson);
    store.archiveHead(brainId, receiptSha256, receiptJson);
  } finally { store.close(); }
  await validateSparseSubstrateTransferWorkspace(root, brainId);
}

function rows(path: string): Record<string, unknown[]> {
  const database = new DatabaseSync(path, { readOnly: true, allowExtension: false });
  try {
    return Object.fromEntries(["artifacts", "requests", "jobs", "heads", "history"].map(table => [table,
      [...database.prepare(`SELECT * FROM ${table} ORDER BY rowid`).iterate()].map(row => ({ ...row }))]));
  } finally { database.close(); }
}

describe("sparse substrate transfer portability", () => {
  it("archives every original row/artifact before owner adoption and exposes only dormant current projections", async () => {
    const root = await directory("adopt"); await seeded(root, "brain-source", "one", "applying");
    const originalRows = rows(join(root, SPARSE_TRANSFER_DIRECTORY, "workspace.sqlite3"));
    const reserve: number[] = [];
    const archive = await adoptSparseSubstrateTransferWorkspace(root, "brain-source", "brain-copy", {
      reserveWrite: async bytes => { reserve.push(bytes); }
    });
    expect(archive).toBeDefined();
    expect(reserve.some(bytes => bytes > 0)).toBe(true);
    const lineage = await listSparseSubstrateLineage(root);
    expect(lineage).toHaveLength(1);
    expect(lineage[0]).toMatchObject({ sourceBrainId: "brain-source", databaseOwner: "brain-source", files: [
      expect.objectContaining({ kind: "bundle", id: "bundle-one" }), expect.objectContaining({ kind: "plan", id: "plan-one" })
    ] });
    expect(rows(join(root, lineage[0]!.relativePath, "workspace.sqlite3"))).toEqual(originalRows);
    const plan = await resolveSparseSubstrateLineageArtifact(root, "brain-source", lineage[0]!.workspaceSha256, "plan", "plan-one");
    expect(await readFile(plan.path, "utf8")).toBe("plan-bytes-one");
    expect(sha(await readFile(plan.path))).toBe(plan.descriptor.sha256);

    const current = await SparseSubstrateTransferStore.open(root, "brain-copy");
    try {
      expect(current.db.prepare("SELECT COUNT(*) AS n FROM artifacts").get()?.n).toBe(0);
      expect(current.db.prepare("SELECT COUNT(*) AS n FROM requests").get()?.n).toBe(0);
      expect(current.db.prepare("SELECT COUNT(*) AS n FROM heads").get()?.n).toBe(0);
      const job = current.job("job-one");
      expect(job).toMatchObject({ brainId: "brain-copy", originalBrainId: "brain-source", state: "uncertain", liveAuthorityPersisted: false });
      expect(job.planId).toBe("plan-one");
      expect(job.authorizationRevision).not.toBe("authority-one");
    } finally { current.close(); }
    await validateSparseSubstrateTransferWorkspace(root, "brain-copy");
    expect(await adoptSparseSubstrateTransferWorkspace(root, "brain-source", "brain-copy")).toBeUndefined();
  });

  it("keeps colliding multi-parent plans and histories in isolated lineage archives without a first-parent winner", async () => {
    const first = await directory("parent-a"), second = await directory("parent-b"), target = await directory("target");
    await seeded(first, "brain-parent-a", "shared", "queued");
    await seeded(second, "brain-parent-b", "shared", "complete");
    await mergeSparseSubstrateTransferWorkspace(first, target, "brain-merged", "brain-parent-a");
    await mergeSparseSubstrateTransferWorkspace(second, target, "brain-merged", "brain-parent-b");
    const lineage = await listSparseSubstrateLineage(target);
    expect(lineage.map(item => item.sourceBrainId)).toEqual(["brain-parent-a", "brain-parent-b"]);
    expect(lineage[0]!.workspaceSha256).not.toBe(lineage[1]!.workspaceSha256);
    expect((await resolveSparseSubstrateLineageArtifact(target, "brain-parent-a", lineage[0]!.workspaceSha256, "plan", "plan-shared")).path)
      .not.toBe((await resolveSparseSubstrateLineageArtifact(target, "brain-parent-b", lineage[1]!.workspaceSha256, "plan", "plan-shared")).path);
    const store = await SparseSubstrateTransferStore.open(target, "brain-merged");
    try {
      expect(store.db.prepare("SELECT COUNT(*) AS n FROM artifacts").get()?.n).toBe(0);
      expect(store.db.prepare("SELECT COUNT(*) AS n FROM requests").get()?.n).toBe(0);
      const jobs = store.jobs("", 20).entries;
      expect(jobs).toHaveLength(2);
      expect(new Set(jobs.map(job => job.id)).size).toBe(2);
      expect(jobs.every(job => job.brainId === "brain-merged" && job.state === "dormant" && job.liveAuthorityPersisted === false)).toBe(true);
      expect(new Set(jobs.map(job => job.originalBrainId))).toEqual(new Set(["brain-parent-a", "brain-parent-b"]));
    } finally { store.close(); }
    await validateSparseSubstrateTransferWorkspace(target, "brain-merged");
  });

  it("refuses changed artifact bytes before replacing current ownership", async () => {
    const root = await directory("corrupt"); await seeded(root, "brain-source", "bad", "queued");
    await writeFile(join(root, SPARSE_TRANSFER_DIRECTORY, "plans", "plan-bad.sqlite3"), "changed");
    await expect(adoptSparseSubstrateTransferWorkspace(root, "brain-source", "brain-copy")).rejects.toThrow(/checksum|differs/i);
    const source = await SparseSubstrateTransferStore.open(root, "brain-source");
    try { expect(source.job("job-bad").brainId).toBe("brain-source"); } finally { source.close(); }
    expect(await listSparseSubstrateLineage(root)).toEqual([]);
  });

  it("preserves nested adoption lineage when that copied parent is merged again", async () => {
    const adopted = await directory("nested-adopted"), merged = await directory("nested-merged");
    await seeded(adopted, "brain-original", "nested", "queued");
    const originalRows = rows(join(adopted, SPARSE_TRANSFER_DIRECTORY, "workspace.sqlite3"));
    await adoptSparseSubstrateTransferWorkspace(adopted, "brain-original", "brain-copy");
    const before = await listSparseSubstrateLineage(adopted);
    expect(before.map(item => item.sourceBrainId)).toEqual(["brain-original"]);

    await mergeSparseSubstrateTransferWorkspace(adopted, merged, "brain-merged", "brain-copy");
    const after = await listSparseSubstrateLineage(merged);
    expect(after.map(item => item.sourceBrainId)).toEqual(["brain-copy", "brain-original"]);
    const original = after.find(item => item.sourceBrainId === "brain-original")!;
    expect(rows(join(merged, original.relativePath, "workspace.sqlite3"))).toEqual(originalRows);
    const plan = await resolveSparseSubstrateLineageArtifact(merged, "brain-original", original.workspaceSha256, "plan", "plan-nested");
    expect(await readFile(plan.path, "utf8")).toBe("plan-bytes-nested");
    expect(plan.descriptor.sha256).toBe(sha("plan-bytes-nested"));
    await validateSparseSubstrateTransferWorkspace(merged, "brain-merged");
  });
});
