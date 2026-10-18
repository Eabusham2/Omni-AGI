import { readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { assertOmniBundleFeatures, collectWholeBrainCatalogArchive, OMNI_BUNDLE_REQUIRED_FEATURES,
  rewriteNativeCatalogFields, validateWholeBrainCatalogArchive } from "../src/main/wholeBrainCatalogArchive";
import { adoptWholeBrainCatalog, collectWholeBrainCatalogLineage, readWholeBrainCatalogMetadata } from "../src/main/wholeBrainCatalogPortability";
import { BrainRepository } from "../src/main/brainRepository";
import { rewriteNativeIdentity } from "../src/main/connectedBrainArchive";
import { extractStreamingZip, writeStreamingZip, type ExtractedZipArchive } from "../src/main/streamingZip";
import { catalogTemporaryDirectory, installLineageFixture, materializeCatalogFixture, tinyCatalog } from "./fixtures/wholeBrainCatalogArchive";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function directory() { const path = await catalogTemporaryDirectory(); roots.push(path); return path; }
async function saved() { return tinyCatalog(await directory()); }
async function archive(scopes: Parameters<typeof collectWholeBrainCatalogArchive>[0]): Promise<ExtractedZipArchive> {
  const closure = await collectWholeBrainCatalogArchive(scopes), root = await directory(), path = join(root, "catalog.omni");
  await writeStreamingZip(path, closure.sources);
  return extractStreamingZip(path, root);
}
function scopes(current: { root: string; metadata: Record<string, unknown> }, origin = current) {
  return { current: { engineRoot: current.root, metadata: current.metadata }, origin: { engineRoot: origin.root, metadata: origin.metadata } };
}

describe("format2 whole-brain archive closure", () => {
  it("requires complete format2 support and rejects unknown required features before adoption", () => {
    expect(assertOmniBundleFeatures({ format: "omni-brain", formatVersion: 2, requiredFeatures: OMNI_BUNDLE_REQUIRED_FEATURES })).toBe(2);
    expect(assertOmniBundleFeatures({ format: "omni-brain", formatVersion: 1 })).toBe(1);
    expect(() => assertOmniBundleFeatures({ format: "omni-brain", formatVersion: 2 })).toThrow(/complete/);
    expect(() => assertOmniBundleFeatures({ format: "omni-brain", formatVersion: 2, requiredFeatures: ["foreign-feature"] })).toThrow(/Unsupported required/);
    expect(() => assertOmniBundleFeatures({ format: "omni-brain", formatVersion: 1, requiredFeatures: OMNI_BUNDLE_REQUIRED_FEATURES })).toThrow(/format2/);
  });

  it("round-trips exact current/origin catalogs and ignores only unreferenced local history", async () => {
    const current = await saved(), origin = await tinyCatalog(await directory(), "origin-brain");
    const original = await readFile(current.shardPath);
    await writeFile(join(current.root, "whole-brain-catalog", "blobs", `${"f".repeat(64)}.json`), "unreferenced old state");
    const exported = await archive(scopes(current, origin));
    const validated = await validateWholeBrainCatalogArchive(exported, 2, { current: current.metadata, origin: origin.metadata });
    expect(validated.files.size).toBe(4);
    expect([...validated.files.keys()].every(path => /^catalog\/(current|origin)\/whole-brain-catalog\//.test(path))).toBe(true);
    expect([...exported.entries.keys()].some(path => path.includes("f".repeat(64)))).toBe(false);
    expect(await readFile(current.shardPath)).toEqual(original);
  });

  it("refuses missing or fabricated empty current/origin catalogs instead of loading or inventing one", async () => {
    const current = await saved(), missing = { root: await directory(), metadata: { ...current.metadata } };
    delete missing.metadata.whole_brain_catalog;
    await expect(collectWholeBrainCatalogArchive(scopes(current, missing))).rejects.toThrow(/real saved origin/);
    await expect(collectWholeBrainCatalogArchive(scopes(missing, current))).rejects.toThrow(/real saved current/);
    const empty = structuredClone(current.metadata);
    const pointer = empty.whole_brain_catalog as Record<string, unknown>;
    pointer.records = 0; pointer.counts = { regions: 0, owners: 0, aliases: 0, memberships: 0, bridges: 0, inspections: 0 };
    // The underlying nonempty chain cannot masquerade as a zero-record registry.
    await expect(collectWholeBrainCatalogArchive(scopes({ ...current, metadata: empty }, current))).rejects.toThrow(/lost records/);
  });

  it("rejects catalog-bearing or dangling format1 namespaces while allowing catalog-free native metadata", async () => {
    const current = await saved(), exported = await archive(scopes(current));
    await expect(validateWholeBrainCatalogArchive(exported, 1, { current: current.metadata, origin: current.metadata })).rejects.toThrow(/requires format2/);
    const emptyArchive: ExtractedZipArchive = { root: await directory(), entries: new Map() };
    await expect(validateWholeBrainCatalogArchive(emptyArchive, 1, { current: {}, origin: {} })).resolves.toMatchObject({ files: new Map() });
    emptyArchive.entries.set("arbitrary/whole-brain-catalog/blobs/orphan.json", {
      name: "arbitrary/whole-brain-catalog/blobs/orphan.json", path: "unused", compressedBytes: 0, uncompressedBytes: 0, crc32: 0 });
    await expect(validateWholeBrainCatalogArchive(emptyArchive, 1, { current: {}, origin: {} })).rejects.toThrow(/requires format2/);
  });

  it("does not accept a declared region count that disagrees with authenticated shard kinds", async () => {
    const value = await saved(), pointer = value.metadata.whole_brain_catalog as Record<string, unknown>;
    await expect(collectWholeBrainCatalogArchive(scopes(value))).resolves.toMatchObject({ files: expect.any(Map) });
    pointer.counts = { regions: 0, owners: 1, aliases: 0, memberships: 0, bridges: 0, inspections: 0 };
    await expect(collectWholeBrainCatalogArchive(scopes(value))).rejects.toThrow(/record-kind counts differ/);
  });

  it("re-exports two generations of copied custody without fabricating a destination registry", async () => {
    const source = await saved(), sourceBytes = await readFile(source.shardPath), origin = await tinyCatalog(await directory(), "origin-brain");
    let previous = source;
    for (const brainId of ["copy-one", "copy-two"]) {
      const adopted = await adoptWholeBrainCatalog(previous.root, previous.metadata, { ...previous.metadata, brain_id: brainId }, brainId, { sameIdentity: false });
      const copy = { root: await directory(), metadata: adopted.metadata };
      await materializeCatalogFixture(copy.root, adopted.files);
      expect(copy.metadata.whole_brain_catalog).toBeUndefined();
      const exported = await archive(scopes(copy, origin));
      await validateWholeBrainCatalogArchive(exported, 2, { current: copy.metadata, origin: origin.metadata });
      const projection = JSON.parse(await readFile(exported.entries.get("catalog/current/projection.json")!.path, "utf8"));
      expect(projection).toMatchObject({ relationship: "preserved-current-state", destinationRegistry: false,
        stateBinding: { brainId }, sourceBrainId: "source-brain" });
      const lineage = await collectWholeBrainCatalogLineage(copy.root, copy.metadata);
      expect(lineage.entries.size).toBe(1); expect([...lineage.entries.values()][0]!.sourceBrainId).toBe("source-brain");
      previous = { ...previous, ...copy };
    }
    expect(await readFile(source.shardPath)).toEqual(sourceBytes);
  });

  it("refuses stale state and ambiguous distinct source registries rather than guessing a copied map", async () => {
    const source = await saved(), copyId = "copied", copied = await adoptWholeBrainCatalog(source.root, source.metadata,
      { ...source.metadata, brain_id: copyId }, copyId, { sameIdentity: false });
    const copy = { root: await directory(), metadata: copied.metadata }; await materializeCatalogFixture(copy.root, copied.files);
    const stale = structuredClone(copy.metadata); stale.attention_overlay = { changed: true };
    await expect(collectWholeBrainCatalogArchive(scopes({ ...copy, metadata: stale }, source))).rejects.toThrow(/real saved current/);
    const other = await tinyCatalog(await directory(), "distinct-source"), otherCopy = await adoptWholeBrainCatalog(other.root, other.metadata,
      { ...other.metadata, brain_id: copyId }, copyId, { sameIdentity: false });
    await materializeCatalogFixture(copy.root, otherCopy.files);
    const first = await collectWholeBrainCatalogLineage(copy.root, copy.metadata), second = await collectWholeBrainCatalogLineage(copy.root, otherCopy.metadata);
    await installLineageFixture(copy.root, copy.metadata, [...first.entries.values(), ...second.entries.values()]);
    await expect(collectWholeBrainCatalogArchive(scopes(copy, source))).rejects.toThrow(/Ambiguous current/);
  });

  it("rejects a tampered archive-only projection and extra authenticated catalog payload", async () => {
    const source = await saved(), adopted = await adoptWholeBrainCatalog(source.root, source.metadata,
      { ...source.metadata, brain_id: "copy" }, "copy", { sameIdentity: false });
    const copy = { root: await directory(), metadata: adopted.metadata }; await materializeCatalogFixture(copy.root, adopted.files);
    const exported = await archive(scopes(copy, source)), entry = exported.entries.get("catalog/current/projection.json")!;
    const original = await readFile(entry.path), changed = Buffer.from(original.toString("utf8").replace('"destinationRegistry":false', '"destinationRegistry":true '));
    expect(changed.length).toBe(original.length); await writeFile(entry.path, changed);
    await expect(validateWholeBrainCatalogArchive(exported, 2, { current: copy.metadata, origin: source.metadata })).rejects.toThrow(/checksum|projection differs/);
    await writeFile(entry.path, original);
    exported.entries.set("catalog/current/orphan.json", { ...entry, name: "catalog/current/orphan.json" });
    await expect(validateWholeBrainCatalogArchive(exported, 2, { current: copy.metadata, origin: source.metadata })).rejects.toThrow(/exact current\/origin\/lineage closure/);
  });

  it("preserves every unrelated native JSON lexeme when identity and catalog fields are rekeyed", () => {
    const raw = '{"brain_id":"source","name":"Old","connected_regions":{"gain":1.0,"big":9007199254740993},"whole_brain_catalog":{"old":true},"config":{"name":"Old"}}';
    const result = rewriteNativeCatalogFields(rewriteNativeIdentity(raw, "copy", "New"), { whole_brain_catalog_lineage: { retained: true } });
    expect(result).toContain('"gain":1.0,"big":9007199254740993');
    expect(result).toContain('"brain_id":"copy"'); expect(result).toContain('"config":{"name":"New"}');
    expect(JSON.parse(result)).not.toHaveProperty("whole_brain_catalog");
    expect(JSON.parse(result).whole_brain_catalog_lineage).toEqual({ retained: true });
    expect(() => rewriteNativeCatalogFields('{"brain_id":"x","brain_id":"y"}', {})).toThrow(/Duplicate/);
  });

  it("admits exact JSON scratch before parsing and honors cancellation after an awaited admission", async () => {
    const path = join(await directory(), "invalid.json"); await writeFile(path, "not JSON");
    const reject = vi.fn(() => { throw new Error("selected app RAM is full"); });
    await expect(readWholeBrainCatalogMetadata(path, { admit: reject })).rejects.toThrow("selected app RAM is full");
    expect(reject).toHaveBeenCalledWith(8 * 8 + 65536, "JSON scratch");
    const controller = new AbortController();
    await expect(readWholeBrainCatalogMetadata(path, { signal: controller.signal, admit: async () => { controller.abort(); } })).rejects.toThrow(/abort/i);
    expect(await readFile(path, "utf8")).toBe("not JSON");
  });

  it("threads repository catalog admission into exact recovery-copy reads", async () => {
    const source = await saved(), root = await directory(), admitted = vi.fn(() => { throw new Error("catalog RAM admission withheld"); });
    const repository = new BrainRepository(join(root, "brains"), undefined, { catalogAdmission: admitted }); await repository.initialize();
    const copy = (repository as unknown as { copySavedContinuation(a: string, b: string, metadata: unknown): Promise<unknown> }).copySavedContinuation.bind(repository);
    await expect(copy(source.root, join(root, "stage", "engine"), source.metadata)).rejects.toThrow("catalog RAM admission withheld");
    expect(admitted).toHaveBeenCalledWith(1024 * 1024, "pointer/index scratch");
    expect(await readFile(source.shardPath, "utf8")).toContain("source-brain");
  });
});
