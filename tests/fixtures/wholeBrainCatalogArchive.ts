import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { wholeBrainCatalogCanonical, wholeBrainCatalogDigest, wholeBrainCatalogBinding, type WholeBrainCatalogFile,
  type WholeBrainCatalogLineageEntry } from "../../src/main/wholeBrainCatalogPortability";

/** Constructor-free JSON/SQLite custody fixture; never a runnable brain. */
export async function tinyCatalog(root: string, brainId = "source-brain", savedMetadata?: Record<string, unknown>) {
  const metadata: Record<string, unknown> = savedMetadata ? structuredClone(savedMetadata) : { brain_id: brainId,
    mutable_state: { activeGeneration: "1".repeat(64) },
    substrate: { persistence: { activeGeneration: "2".repeat(64) } }, attention_overlay: {} };
  const record = { kind: "region", id: `region-${brainId}`, parentRegionId: null, label: "Whole brain",
    regionKind: "whole-brain", lineage: { brainId }, evidence: "actual-brain-owner" };
  const body = { format: "omni-whole-brain-catalog", version: 1, previousSha256: null, records: [record] };
  const shard = Buffer.from(wholeBrainCatalogCanonical(body)), shardSha = createHash("sha256").update(shard).digest("hex");
  const shardPath = join(root, "whole-brain-catalog", "blobs", `${shardSha}.json`);
  await mkdir(dirname(shardPath), { recursive: true }); await writeFile(shardPath, shard);
  const temporaryIndex = join(root, "whole-brain-catalog", "index", "temporary.sqlite3");
  await mkdir(dirname(temporaryIndex), { recursive: true });
  const database = new DatabaseSync(temporaryIndex);
  try {
    database.exec("CREATE TABLE records(id TEXT PRIMARY KEY,kind TEXT,shard TEXT,ordinal INTEGER,owner_id TEXT,region_id TEXT,parent_id TEXT,source_id TEXT,target_id TEXT,rank INTEGER)");
    database.prepare("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)").run(record.id, record.kind, shardSha, 0, null, null, null, null, null, 0);
  } finally { database.close(); }
  const index = await readFile(temporaryIndex), indexSha = createHash("sha256").update(index).digest("hex");
  await writeFile(join(dirname(temporaryIndex), `${indexSha}.sqlite3`), index); await rm(temporaryIndex);
  if (metadata.brain_id !== brainId) throw new Error("Catalog fixture owner must match exact saved metadata.");
  const binding = wholeBrainCatalogBinding(metadata);
  metadata.whole_brain_catalog = { format: "omni-whole-brain-catalog", version: 1, brainId, binding,
    bindingSha256: wholeBrainCatalogDigest(binding), tailSha256: shardSha, shards: 1, records: 1,
    counts: { regions: 1, owners: 0, aliases: 0, memberships: 0, bridges: 0, inspections: 0 },
    index: { path: `whole-brain-catalog/index/${indexSha}.sqlite3`, sha256: indexSha, bytes: index.length },
    semanticAxisMapping: false, activityObserved: false };
  return { root, metadata, shardPath };
}

export async function materializeCatalogFixture(root: string, files: Map<string, WholeBrainCatalogFile>): Promise<void> {
  for (const file of files.values()) {
    const path = join(root, ...file.relativePath.split("/")); await mkdir(dirname(path), { recursive: true });
    if (file.sourcePath) await copyFile(file.sourcePath, path); else await writeFile(path, file.contents!);
  }
}

export async function installLineageFixture(root: string, metadata: Record<string, unknown>,
  entries: WholeBrainCatalogLineageEntry[]): Promise<void> {
  const body = { format: "omni-whole-brain-catalog-lineage-shard", version: 1, previousSha256: null, entries };
  const bytes = Buffer.from(wholeBrainCatalogCanonical(body)), sha256 = createHash("sha256").update(bytes).digest("hex");
  const path = join(root, "whole-brain-catalog", "lineage", "blobs", `${sha256}.json`);
  await mkdir(dirname(path), { recursive: true }); await writeFile(path, bytes);
  const pointer = { format: "omni-whole-brain-catalog-lineage", version: 1, tailSha256: sha256, shards: 1, entries: entries.length };
  metadata.whole_brain_catalog_lineage = { ...pointer, manifestSha256: wholeBrainCatalogDigest(pointer) };
}

export async function catalogTemporaryDirectory(): Promise<string> { return mkdtemp(join(tmpdir(), "omni-format2-catalog-")); }
