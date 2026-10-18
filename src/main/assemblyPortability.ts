import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, opendir, readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { copyMutableFileIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
export function safeAssemblyWorkspacePath(value: string): boolean { return /^assembly\/[a-f0-9]{32}\//.test(value) && !value.includes("\\") && !value.includes("\0") && value.split("/").every(part => part && part !== "." && part !== ".."); }
async function exists(path: string) { return !!await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; }); }
async function* inventory(root: string, prefix = ""): AsyncGenerator<{ name: string; path: string; bytes: number; sha256: string }> {
  const info = await lstat(root); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Assembly archive contains an unsafe directory.");
  const entries = await opendir(root);
  for await (const entry of entries) {
    const path = join(root, entry.name), name = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isSymbolicLink()) throw new Error("Assembly archive contains a filesystem link.");
    if (entry.isDirectory()) { yield* inventory(path, name); continue; }
    if (!entry.isFile()) throw new Error("Assembly archive contains a nonregular output.");
    const before = await lstat(path), digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk);
    const after = await lstat(path); if (before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size || before.mtimeMs !== after.mtimeMs) throw new Error("Assembly output changed while archiving; no complete snapshot has been claimed.");
    yield { name, path, bytes: after.size, sha256: digest.digest("hex") };
  }
}
async function readJson(path: string): Promise<Record<string, any>> {
  const info = await lstat(path), need = info.size * 12 + 65536;
  if (!info.isFile() || info.isSymbolicLink() || !Number.isSafeInteger(need) || need > freemem() / 2 || need > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Assembly metadata awaits safe regular-file parse headroom.");
  const value = JSON.parse(await readFile(path, "utf8")); if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid saved assembly metadata."); return value;
}
export async function boundAssemblyOriginalId(root: string, currentId: string, compile: { artifactId: string }): Promise<string> {
  if (compile.artifactId === currentId) return currentId;
  const binding = await readJson(join(root, "binding.json")), receiptHash = createHash("sha256"); for await (const chunk of createReadStream(join(root, "compile.json"))) receiptHash.update(chunk);
  if (binding.format !== "omni-assembly-local-binding" || binding.version !== 1 || binding.artifactId !== currentId || binding.originalArtifactId !== compile.artifactId || binding.compileSha256 !== receiptHash.digest("hex")) throw new Error("Assembly alias differs from its exact retained source receipt.");
  return compile.artifactId;
}
export async function assemblyArtifactPage(root: string, after = "", limit = 50): Promise<{ entries: Array<Record<string, unknown>>; total: number; nextAfter?: string }> {
  if (after && !/^[a-f0-9]{32}$/.test(after) || !Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid assembly artifact page.");
  const base = join(root, "assembly"); if (!await exists(base)) return { entries: [], total: 0 };
  const info = await lstat(base); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Assembly artifact page requires a real directory.");
  const selected: string[] = []; let total = 0; const reader = await opendir(base);
  for await (const entry of reader) { if (!/^[a-f0-9]{32}$/.test(entry.name) || !entry.isDirectory() || entry.isSymbolicLink()) throw new Error("Unsafe assembly artifact directory."); total++;
    if (entry.name > after) { selected.push(entry.name); selected.sort(); if (selected.length > limit + 1) selected.pop(); } }
  const entries: Array<Record<string, unknown>> = [];
  for (const artifactId of selected.slice(0, limit)) { const path = join(base, artifactId), hasReceipt = await exists(join(path, "compile.json"));
    if (!hasReceipt) { entries.push({ artifactId, state: "incomplete", nativeProcessResumable: false }); continue; }
    const compile = await readJson(join(path, "compile.json")), originalArtifactId = await boundAssemblyOriginalId(path, artifactId, compile as { artifactId: string });
    entries.push({ artifactId, originalArtifactId, state: compile.state, kind: compile.kind, architecture: compile.architecture, platform: compile.platform, artifactSha256: compile.artifact?.sha256 }); }
  return { entries, total, ...(selected.length > limit ? { nextAfter: selected[limit - 1] } : {}) };
}
/** Opaque ordinary files are preserved, never evaluated or automatically executed. */
export async function validateAssemblyWorkspace(root: string): Promise<void> {
  const base = join(root, "assembly"); if (!await exists(base)) return; const info = await lstat(base); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Assembly workspace is not a real managed directory."); const entries = await opendir(base);
  for await (const entry of entries) {
    if (!/^[a-f0-9]{32}$/.test(entry.name) || !entry.isDirectory() || entry.isSymbolicLink()) throw new Error("Assembly workspace has an unsafe artifact identity.");
    const artifact = join(base, entry.name); for await (const _file of inventory(artifact)) { /* Verify the entire regular-file closure, not just completed output. */ }
    if (await exists(join(artifact, "compile.json"))) {
      const compile = await readJson(join(artifact, "compile.json"));
      if (compile.format !== "omni-assembly-receipt" || compile.version !== 1 || typeof compile.artifactId !== "string" || !/^[a-f0-9]{32}$/.test(compile.artifactId) || !Array.isArray(compile.partialFiles) || !Array.isArray(compile.processes)) throw new Error("Assembly compile receipt format is invalid.");
      await boundAssemblyOriginalId(artifact, entry.name, compile as { artifactId: string });
    }
    // A source/intent-only interrupted native compiler is kept as data, never
    // described as a resumable process or a compiled executable.
  }
}
/** Preserve distinct artifact conflicts under a new local ID, retaining raw receipts. */
export async function mergeAssemblyWorkspace(sourceRoot: string, targetRoot: string, parentId: string, checkDisk?: (bytes: number) => Promise<void>): Promise<void> {
  if (!await exists(join(sourceRoot, "assembly"))) return; await validateAssemblyWorkspace(sourceRoot); await validateAssemblyWorkspace(targetRoot); await mkdir(join(targetRoot, "assembly"), { recursive: true });
  const entries = await opendir(join(sourceRoot, "assembly"));
  for await (const entry of entries) {
    const source = join(sourceRoot, "assembly", entry.name); let localId = entry.name, target = join(targetRoot, "assembly", localId), conflict = false;
    const inventoryDigest = Buffer.alloc(32); let inventoryCount = 0;
    for await (const item of inventory(source)) { const digest = createHash("sha256").update(JSON.stringify([item.name, item.bytes, item.sha256])).digest(); inventoryCount++; for (let index = 0; index < 32; index++) inventoryDigest[index] = inventoryDigest[index]! ^ digest[index]!;
      const destination = join(target, ...item.name.split("/")); if (await exists(destination)) { const current = await lstat(destination); if (!current.isFile() || current.isSymbolicLink()) throw new Error("Target assembly output is unsafe.");
        const comparison = createHash("sha256"); for await (const chunk of createReadStream(destination)) comparison.update(chunk); if (comparison.digest("hex") !== item.sha256) conflict = true; } }
    if (conflict) { localId = hash(JSON.stringify({ originalId: entry.name, inventoryCount, inventory: inventoryDigest.toString("hex") })).slice(0, 32); target = join(targetRoot, "assembly", localId); }
    await mkdir(target, { recursive: true }); const sourceBinding = await exists(join(source, "binding.json")) ? await readJson(join(source, "binding.json")) : undefined;
    for await (const item of inventory(source)) {
      const name = item.name === "binding.json" && localId !== entry.name ? `provenance-binding-${item.sha256}.json` : item.name;
      const destination = join(target, ...name.split("/")); if (await exists(destination)) { const comparison = createHash("sha256"); for await (const chunk of createReadStream(destination)) comparison.update(chunk); if (comparison.digest("hex") !== item.sha256) throw new Error("An assembly identity alias collided with different bytes."); continue; }
      await checkDisk?.(item.bytes); await mkdir(dirname(destination), { recursive: true }); await copyMutableFileIsolated(item.path, destination, { expectedSha256: item.sha256 });
    }
    if (localId !== entry.name && await exists(join(source, "compile.json"))) {
      const compile = await readJson(join(source, "compile.json")), receipt = createHash("sha256"); for await (const chunk of createReadStream(join(source, "compile.json"))) receipt.update(chunk);
      const body = JSON.stringify({ format: "omni-assembly-local-binding", version: 1, artifactId: localId, originalArtifactId: compile.artifactId, compileSha256: receipt.digest("hex"), parentId, sourceArtifactId: entry.name, ...(sourceBinding ? { previousBinding: sourceBinding } : {}) });
      if (!await exists(join(target, "binding.json"))) { await checkDisk?.(Buffer.byteLength(body)); await writeMutableFileIsolated(join(target, "binding.json"), Buffer.from(body)); }
    }
  }
  await validateAssemblyWorkspace(targetRoot);
}
