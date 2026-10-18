import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { expect, it } from "vitest";
import { cloneLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { SparseSubstrateTransferStore } from "../src/main/sparseSubstrateTransferStore";
import { listSparseSubstrateLineage, resolveSparseSubstrateLineageArtifact } from "../src/main/sparseSubstrateTransferPortability";

it("carries exact sparse SQLite artifact bytes through the actual shared workspace copy hook", async () => {
  // A host workspace and inert SQLite bytes, not a brain/core/model fixture.
  const root = await mkdtemp(join(tmpdir(), "omni-sparse-copy-hook-"));
  const source = join(root, "source"), target = join(root, "target"), hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
  try {
    const store = await SparseSubstrateTransferStore.open(source, "source-owner");
    await mkdir(join(source, "substrate-transfers", "bundles"));
    const relativePath = "substrate-transfers/bundles/inert-fixture.sqlite3", path = join(source, relativePath);
    const payload = new DatabaseSync(path); payload.exec("CREATE TABLE inert(value TEXT); INSERT INTO inert VALUES('exact original byte payload');"); payload.close();
    const bytes = await readFile(path), manifestJson = JSON.stringify({ format: "omni-sparse-substrate-transfer", version: 1,
      scope: "native-sparse-assemblies", rows: 0, sourceBrainId: "source-owner", sourceGenerationSha256: "a".repeat(64), cortexStateIncluded: false, sourceAuthorityIncluded: false });
    try { store.putArtifact("bundle", { format: "omni-sparse-substrate-transfer", version: 1, scope: "native-sparse-assemblies", id: "inert-fixture",
      sha256: hash(bytes), bytes: bytes.length, manifestJson, manifestSha256: hash(manifestJson), sourceBrainId: "source-owner",
      sourceGenerationSha256: "a".repeat(64), rows: 0, relativePath, sourceBinding: "verified-owned-export" }); }
    finally { store.close(); }
    await cloneLearningWorkspace(source, target, "source-owner", "copied-owner");
    const lineage = await listSparseSubstrateLineage(target); expect(lineage).toHaveLength(1);
    const artifact = await resolveSparseSubstrateLineageArtifact(target, "source-owner", lineage[0]!.workspaceSha256, "bundle", "inert-fixture");
    expect(await readFile(artifact.path)).toEqual(bytes);
    expect(await readFile(path)).toEqual(bytes);
    const current = await SparseSubstrateTransferStore.open(target, "copied-owner");
    try { expect(current.db.prepare("SELECT COUNT(*) AS count FROM artifacts").get()?.count).toBe(0); }
    finally { current.close(); }
  } finally { await rm(root, { recursive: true, force: true }); }
});
