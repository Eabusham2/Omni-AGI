import { createHash } from "node:crypto";
import { copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, win32 } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { adoptWholeBrainCatalog, collectWholeBrainCatalog, collectWholeBrainCatalogLineage,
  wholeBrainCatalogCanonical, wholeBrainCatalogDigest, wholeBrainCatalogSqlitePath, type WholeBrainCatalogFile } from "../src/main/wholeBrainCatalogPortability";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function fixture(brainId = "source-brain") {
  const root = await mkdtemp(join(tmpdir(), "omni-whole-catalog-")); roots.push(root);
  const metadata: Record<string, unknown> = { brain_id: brainId,
    mutable_state: { activeGeneration: "1".repeat(64) },
    substrate: { persistence: { activeGeneration: "2".repeat(64) } },
    attention_overlay: {} };
  const record = { kind: "region", id: "region-root", parentRegionId: null, label: "Whole brain",
    regionKind: "whole-brain", lineage: { brainId }, evidence: "actual-brain-owner" };
  const shard = { format: "omni-whole-brain-catalog", version: 1, previousSha256: null, records: [record] };
  const shardBytes = Buffer.from(wholeBrainCatalogCanonical(shard)), shardSha = createHash("sha256").update(shardBytes).digest("hex");
  const shardPath = join(root, "whole-brain-catalog", "blobs", `${shardSha}.json`);
  await mkdir(dirname(shardPath), { recursive: true }); await writeFile(shardPath, shardBytes);
  const temporaryIndex = join(root, "whole-brain-catalog", "index", "temporary.sqlite3"); await mkdir(dirname(temporaryIndex), { recursive: true });
  const database = new DatabaseSync(temporaryIndex);
  database.exec(`CREATE TABLE records(id TEXT PRIMARY KEY,kind TEXT,shard TEXT,ordinal INTEGER,owner_id TEXT,region_id TEXT,
    parent_id TEXT,source_id TEXT,target_id TEXT,rank INTEGER)`);
  database.prepare("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?,?)").run(record.id, record.kind, shardSha, 0, null, null, null, null, null, 0);
  database.close();
  const indexBytes = await readFile(temporaryIndex), indexSha = createHash("sha256").update(indexBytes).digest("hex");
  const indexPath = join(root, "whole-brain-catalog", "index", `${indexSha}.sqlite3`); await writeFile(indexPath, indexBytes); await rm(temporaryIndex);
  const binding = { brainId, coreGeneration: "1".repeat(64), substrateGeneration: "2".repeat(64),
    attentionOverlaySha256: wholeBrainCatalogDigest({}), connectedDescriptorSha256: null, connectedGraphStatesSha256: null };
  metadata.whole_brain_catalog = { format: "omni-whole-brain-catalog", version: 1, brainId, binding,
    bindingSha256: wholeBrainCatalogDigest(binding), tailSha256: shardSha, shards: 1, records: 1,
    counts: { regions: 1, owners: 0, aliases: 0, memberships: 0, bridges: 0, inspections: 0 },
    index: { path: `whole-brain-catalog/index/${indexSha}.sqlite3`, sha256: indexSha, bytes: indexBytes.byteLength },
    semanticAxisMapping: false, activityObserved: false };
  return { root, metadata, shardPath, indexPath };
}
async function materialize(root: string, files: Map<string, WholeBrainCatalogFile>): Promise<void> {
  for (const file of files.values()) {
    const target = join(root, ...file.relativePath.split("/")); await mkdir(dirname(target), { recursive: true });
    if (file.sourcePath) await copyFile(file.sourcePath, target); else await writeFile(target, file.contents!);
  }
}

describe("whole-brain catalog exact portability closure", () => {
  it("preserves authenticated long drive and UNC index paths when SQLite needs the Windows namespace", () => {
    const suffix = win32.join("whole-brain-catalog", "lineage", "sources", "a".repeat(64),
      "whole-brain-catalog", "index", `${"b".repeat(64)}.sqlite3`);
    const drive = win32.join("C:\\Users\\runneradmin\\AppData\\Local\\Temp\\omni-whole-catalog-long-fixture", suffix);
    const unc = win32.join("\\\\server\\share\\saved-catalog", suffix);
    expect(drive.length).toBeGreaterThan(260);
    expect(wholeBrainCatalogSqlitePath(drive, "win32")).toBe(`file:${encodeURIComponent(`\\\\?\\${drive}`)}?vfs=win32-longpath`);
    expect(wholeBrainCatalogSqlitePath(unc, "win32")).toBe(`file:${encodeURIComponent(`\\\\?\\UNC\\${unc.slice(2)}`)}?vfs=win32-longpath`);
    expect(wholeBrainCatalogSqlitePath(`\\\\?\\${drive}`, "win32")).toBe(`file:${encodeURIComponent(`\\\\?\\${drive}`)}?vfs=win32-longpath`);
    expect(wholeBrainCatalogSqlitePath("/owned/catalog/index.sqlite3", "darwin")).toBe("/owned/catalog/index.sqlite3");
  });
  it("visits only the pointer-referenced JSON chain and immutable location index", async () => {
    const value = await fixture();
    await writeFile(join(value.root, "whole-brain-catalog", "blobs", `${"f".repeat(64)}.json`), "unreferenced and corrupt");
    const closure = await collectWholeBrainCatalog(value.root, value.metadata);
    expect([...closure!.files.keys()].sort()).toEqual([
      `whole-brain-catalog/blobs/${(value.metadata.whole_brain_catalog as Record<string, unknown>).tailSha256}.json`,
      (value.metadata.whole_brain_catalog as { index: { path: string } }).index.path
    ].sort());
    await writeFile(value.shardPath, "changed");
    await expect(collectWholeBrainCatalog(value.root, value.metadata)).rejects.toThrow(/checksum/);
  });

  it("preserves a native descriptor checksum whose numeric lexeme cannot round-trip through JavaScript", async () => {
    const value = await fixture();
    const descriptorSha256 = createHash("sha256").update('{"gain":1.0}').digest("hex");
    value.metadata.connected_regions = { gain: 1, sha256: descriptorSha256 };
    const pointer = value.metadata.whole_brain_catalog as Record<string, unknown>;
    const binding = { ...(pointer.binding as Record<string, unknown>), connectedDescriptorSha256: descriptorSha256 };
    pointer.binding = binding; pointer.bindingSha256 = wholeBrainCatalogDigest(binding);
    await expect(collectWholeBrainCatalog(value.root, value.metadata)).resolves.toMatchObject({ sourceBrainId: "source-brain" });
  });

  it("refuses undeclared authority-like fields in a portable catalog pointer", async () => {
    const value = await fixture();
    (value.metadata.whole_brain_catalog as Record<string, unknown>).approvalToken = "must-not-travel";
    await expect(collectWholeBrainCatalog(value.root, value.metadata)).rejects.toThrow(/shape differs/);
  });

  it("keeps same-owner current bytes but makes a new owner lineage-only without rewriting source IDs", async () => {
    const value = await fixture(), same = await adoptWholeBrainCatalog(value.root, value.metadata,
      structuredClone(value.metadata), "source-brain", { sameIdentity: true });
    expect(same.current).toBe(true); expect(same.metadata.whole_brain_catalog).toEqual(value.metadata.whole_brain_catalog);

    const targetMetadata = { ...structuredClone(value.metadata), brain_id: "target-brain" };
    const adopted = await adoptWholeBrainCatalog(value.root, value.metadata, targetMetadata, "target-brain", { sameIdentity: false });
    expect(adopted.current).toBe(false); expect(adopted.metadata.whole_brain_catalog).toBeUndefined();
    expect(adopted.metadata.whole_brain_catalog_lineage).toMatchObject({ format: "omni-whole-brain-catalog-lineage", entries: 1 });
    const sourcePointer = value.metadata.whole_brain_catalog as Record<string, unknown>;
    const closurePath = [...adopted.files.keys()].find(path => path.endsWith(`/${sourcePointer.tailSha256}.json`) && path.includes("/sources/"));
    expect(closurePath).toBeTruthy();
    const targetRoot = await mkdtemp(join(tmpdir(), "omni-whole-target-")); roots.push(targetRoot);
    await materialize(targetRoot, adopted.files);
    const lineage = await collectWholeBrainCatalogLineage(targetRoot, adopted.metadata);
    expect(lineage.entries.size).toBe(1);
    const entry = [...lineage.entries.values()][0]!;
    expect(entry.sourceBrainId).toBe("source-brain");
    expect(entry.catalogPointer).toEqual(sourcePointer);
    expect(entry.closureRoot).toBe(`whole-brain-catalog/lineage/sources/${entry.entryKeySha256}`);
  });

  it("flattens prior lineage recursively and deduplicates only its exact authenticated entry", async () => {
    const source = await fixture(), first = await adoptWholeBrainCatalog(source.root, source.metadata,
      { ...structuredClone(source.metadata), brain_id: "copy-one" }, "copy-one", { sameIdentity: false });
    const firstRoot = await mkdtemp(join(tmpdir(), "omni-whole-first-")); roots.push(firstRoot); await materialize(firstRoot, first.files);
    const second = await adoptWholeBrainCatalog(firstRoot, first.metadata,
      { ...structuredClone(first.metadata), brain_id: "copy-two" }, "copy-two", { sameIdentity: false });
    const secondRoot = await mkdtemp(join(tmpdir(), "omni-whole-second-")); roots.push(secondRoot); await materialize(secondRoot, second.files);
    const lineage = await collectWholeBrainCatalogLineage(secondRoot, second.metadata);
    expect(lineage.entries.size).toBe(1);
    expect([...lineage.entries.values()][0]!.sourceBrainId).toBe("source-brain");
  });

  it("replaces a private recovery closure exactly and declares every restored catalog file", async () => {
    const source = await fixture(), repositoryRoot = await mkdtemp(join(tmpdir(), "omni-whole-repository-"));
    roots.push(repositoryRoot);
    const repository = new BrainRepository(join(repositoryRoot, "brains"));
    await repository.initialize();
    const destinationEngine = join(repositoryRoot, "saved", "engine");
    const stale = join(destinationEngine, "whole-brain-catalog", "blobs", `${"f".repeat(64)}.json`);
    await mkdir(dirname(stale), { recursive: true }); await writeFile(stale, "later unreferenced catalog state");
    const copy = (repository as unknown as { copySavedContinuation(
      sourceEngine: string, destinationEngine: string, metadata: unknown
    ): Promise<{ paths: string[]; hashes: string[] }> }).copySavedContinuation.bind(repository);
    const result = await copy(source.root, destinationEngine, source.metadata);
    const closure = await collectWholeBrainCatalog(destinationEngine, source.metadata);
    expect(new Set(result.paths)).toEqual(new Set([...closure!.files.keys()].map(path => `engine/${path}`)));
    expect(result.hashes).toHaveLength(closure!.files.size);
    await expect(readFile(stale)).rejects.toMatchObject({ code: "ENOENT" });
  });
});
