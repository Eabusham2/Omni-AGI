/** Read-only selected custody hashes for the opt-in native host fixture. No neural imports. */
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, readFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
async function fileHash(path: string): Promise<string> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error(`Selected custody file is not regular: ${path}`);
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}
async function metadataFile(path: string): Promise<{ bytes: Buffer; value: Record<string, unknown> }> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 16 * 1024 ** 2) throw new Error("Selected fixture metadata is not a bounded regular file.");
  const bytes = await readFile(path), value: unknown = JSON.parse(bytes.toString("utf8"));
  if (!record(value)) throw new Error("Selected fixture metadata is not an object.");
  return { bytes, value };
}
function generationPath(value: unknown, label: string): { relative: string; sha256: string } {
  if (!record(value) || typeof value.activeGeneration !== "string" || !/^[a-f0-9]{64}$/.test(value.activeGeneration) || typeof value.generationManifest !== "string" ||
    value.generationManifest !== `generations/${value.activeGeneration}/manifest.json` ||
    typeof value.generationManifestSha256 !== "string" || !/^[a-f0-9]{64}$/.test(value.generationManifestSha256)) {
    throw new Error(`Selected custody ${label} generation declaration is invalid.`);
  }
  return { relative: value.generationManifest, sha256: value.generationManifestSha256 };
}

export async function selectedInheritedSourceInventory(source: string): Promise<Record<string, string>> {
  const selected: Record<string, string> = {}, root = join(resolve(source), "merged"), engineRoot = join(root, "engine"), origin = join(engineRoot, "origin");
  const originMetadata = await metadataFile(join(origin, "brain.json")), descriptor = originMetadata.value.connected_regions;
  if (!record(descriptor) || !Array.isArray(descriptor.parents) || !Array.isArray(descriptor.regions)) throw new Error("Selected source has no connected-parent declaration.");
  const current = await metadataFile(join(engineRoot, "brain.json"));
  for (const owner of ["a", "b"]) {
    const aliases = descriptor.parents.filter(value => record(value) && value.parentId === owner);
    const alias = aliases[0];
    if (aliases.length !== 1 || !record(alias)) throw new Error("Selected source parent alias is ambiguous.");
    const declarations = descriptor.regions.filter(value => record(value) && value.regionId === alias.regionId);
    const declaration = declarations[0];
    if (declarations.length !== 1 || !record(declaration) || declaration.parentId !== owner ||
      typeof declaration.enginePath !== "string" || !/^merge\/parents\/[ab]\/engine$/.test(declaration.enginePath)) {
      throw new Error("This selected two-distinct-parent fixture declaration differs.");
    }
    // Exactly the native origin resolver's declared aliases; an absent metadata
    // dependency is not repaired from a graph cache or guessed from its config.
    const candidates = [join(origin, "connected-inputs", declaration.enginePath), join(engineRoot, declaration.enginePath)];
    if (isDeepStrictEqual(current.value.connected_regions, descriptor)) {
      candidates.push(join(engineRoot, "connected-inputs", declaration.enginePath), join(root, declaration.enginePath));
    }
    let engine: string | undefined;
    for (const candidate of candidates) {
      try { await lstat(join(candidate, "brain.json")); engine = candidate; break; }
      catch (error) { if (!record(error) || error.code !== "ENOENT") throw error; }
    }
    if (!engine) throw new Error(`Declared parent ${owner}/${String(declaration.regionId)} lacks authenticated metadata ${String(declaration.metadataSha256)} in all native aliases: ${candidates.join(", ")}`);
    const metadataPath = join(engine, "brain.json"), { bytes, value: metadata } = await metadataFile(metadataPath);
    if (metadata.brain_id !== owner || !record(metadata.substrate)) throw new Error("Selected parent owner/substrate binding is invalid.");
    selected[metadataPath] = createHash("sha256").update(bytes).digest("hex");
    if (selected[metadataPath] !== declaration.metadataSha256) throw new Error("Selected parent metadata differs from its connected declaration.");
    for (const [name, expected] of [["core.safetensors", declaration.coreSha256], ["plasticity.safetensors", declaration.plasticitySha256], ["conversation.sqlite3", undefined]] as const) {
      const path = join(engine, name); selected[path] = await fileHash(path);
      if (expected !== undefined && selected[path] !== expected) throw new Error("Selected parent neural file differs from its connected declaration.");
    }
    for (const [directory, pointer] of [["state", metadata.mutable_state], ["substrate", metadata.substrate.persistence]] as const) {
      const declared = generationPath(pointer, `${owner}/${directory}`), path = join(engine, directory, ...declared.relative.split("/"));
      const actual = await fileHash(path);
      if (actual !== declared.sha256) throw new Error(`Selected custody generation checksum differs: ${path}`);
      selected[path] = actual;
    }
  }
  for (const name of ["brain.json", "core.safetensors", "plasticity.safetensors", "connected-origin.json"]) {
    const path = join(resolve(source), "merged/engine/origin", name); selected[path] = await fileHash(path);
  }
  if (Object.keys(selected).length !== 16) throw new Error("Selected fixture custody inventory is not exactly sixteen unique files.");
  return selected;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = process.argv[2];
  if (!source) throw new Error("Pass the retained trained fixture directory. This command only reads selected files.");
  process.stdout.write(`${JSON.stringify({ format: "omni-current-inherited-selected-source-inventory", version: 1,
    source: resolve(source), files: await selectedInheritedSourceInventory(source) }, null, 2)}\n`);
}
