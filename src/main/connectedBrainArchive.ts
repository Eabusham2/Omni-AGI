/** Connected checkpoints have dependencies outside their root tensor files.
 *
 * Archive original user files as inert, content-addressed payloads. Their
 * original names are restored only beneath the new app-owned instance; they
 * are never ZIP entry names, executable modules, permissions, or host paths.
 */
import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, readFile, readdir } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import { createInterface } from "node:readline";
import { DatabaseSync } from "node:sqlite";
import type { BrainStorageOperationHooks } from "./brainStorageOperations";
import { assertSafeArchivePath, streamFileSha256, type StreamingZipSource } from "./streamingZip";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
import { collectConnectedExecutionInputs, validateConnectedExecutionArchive } from "./connectedExecutionPortability";
import { sqliteFilename } from "./sqliteFilename";

const FORMAT = "omni-connected-dependency-archive";
const HASH = /^[a-f0-9]{64}$/;
const ID = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/;
const hash = (value: Uint8Array | string): string => createHash("sha256").update(value).digest("hex");
const fileHash = async (path: string, signal?: AbortSignal): Promise<string> => {
  const digest = createHash("sha256");
  for await (const bytes of createReadStream(path, { highWaterMark: 1024 * 1024, signal })) digest.update(bytes);
  return digest.digest("hex");
};
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid connected dependency declaration.");
  return value as Record<string, unknown>;
};
const canonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  return JSON.stringify(value);
};

interface DependencyDescriptor { sha256: string; bytes: number }
export interface ConnectedArchive {
  manifest: { format: typeof FORMAT; formatVersion: 1; files: Record<string, DependencyDescriptor> };
  /** Keys are validated logical paths relative to an instance, never host paths. */
  files: Map<string, StreamingZipSource>;
}
export interface ConnectedArchiveScope { engine: string; metadata: unknown }
export interface ConnectedSourceScope extends ConnectedArchiveScope { sourceEngine: string }
type SubstrateCollector = (engine: string, prefix: string, metadata: unknown, signal?: AbortSignal) =>
  Promise<{ sources: StreamingZipSource[] } | undefined>;

/** Rekey a mutable identity without reserializing hash-bound nested native
 * descriptors. In particular, Python's 1.0 must not silently become JS's 1.
 */
export function rewriteNativeIdentity(text: string, brainId: string, name?: string): string {
  const parsed = record(JSON.parse(text));
  const replacements = new Map<string, string>([[JSON.stringify(["brain_id"]), JSON.stringify(brainId)]]);
  if (name !== undefined) {
    if (Object.hasOwn(parsed, "name")) replacements.set(JSON.stringify(["name"]), JSON.stringify(name));
    if (parsed.config && Object.hasOwn(record(parsed.config), "name")) replacements.set(JSON.stringify(["config", "name"]), JSON.stringify(name));
  }
  if (!Object.hasOwn(parsed, "brain_id")) throw new Error("Native checkpoint has no identity to rekey.");
  const edits: Array<{ start: number; end: number; value: string }> = [];
  let cursor = 0;
  const whitespace = () => { while (/\s/.test(text[cursor] ?? "") && cursor < text.length) cursor += 1; };
  const string = (): string => {
    const start = cursor++;
    while (cursor < text.length) {
      const char = text[cursor++];
      if (char === "\\") cursor += 1;
      else if (char === '"') return JSON.parse(text.slice(start, cursor)) as string;
    }
    throw new Error("Invalid native JSON string.");
  };
  const value = (path: string[]): void => {
    whitespace(); const start = cursor, char = text[cursor];
    if (char === "{") {
      cursor += 1; whitespace(); const keys = new Set<string>();
      while (text[cursor] !== "}") {
        const key = string();
        if (keys.has(key)) throw new Error("Native checkpoint contains duplicate JSON keys.");
        keys.add(key); whitespace(); cursor += 1; value([...path, key]); whitespace();
        if (text[cursor] !== ",") break;
        cursor += 1; whitespace();
      }
      cursor += 1;
    } else if (char === "[") {
      cursor += 1; whitespace(); let index = 0;
      while (text[cursor] !== "]") {
        value([...path, String(index++)]); whitespace();
        if (text[cursor] !== ",") break;
        cursor += 1;
      }
      cursor += 1;
    } else if (char === '"') string();
    else { while (cursor < text.length && !/[\s,}\]]/.test(text[cursor]!)) cursor += 1; }
    const replacement = replacements.get(JSON.stringify(path));
    if (replacement !== undefined) edits.push({ start, end: cursor, value: replacement });
  };
  value([]);
  for (const edit of edits.sort((a, b) => b.start - a.start)) text = text.slice(0, edit.start) + edit.value + text.slice(edit.end);
  return text;
}

export function isConnectedDependencyPath(path: string): boolean {
  return path === "full-merge.json" || path.startsWith("merge/") ||
    /^engine\/(?:origin\/)?(?:connected-(?:state|inputs|execution-inputs)\/|connected-(?:runtime\.sqlite3|origin\.json|assembly\.json)$)/.test(path);
}

function safeDependencyPath(path: string): string {
  assertSafeArchivePath(path);
  // Also reject Windows alternate streams/devices and ambiguous trailing dots.
  if (path.split("/").some(part => /[:\x00-\x1f]/.test(part) || /[. ]$/.test(part) ||
    /^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part)) || !isConnectedDependencyPath(path)) {
    throw new Error("Connected dependency destination is not an app-managed relative path.");
  }
  // These are checkpoint loaders, not user document attachments. Never turn
  // an opaque payload into an unsafe native pickle checkpoint on import.
  if (/(?:^|\/)engine\/(?!artifacts\/|workspace\/).*\.(?:pt|pth|pkl|pickle|joblib)$/i.test(path)) {
    throw new Error("Unsafe native checkpoint content is not allowed in connected dependencies.");
  }
  if (/\.sqlite3?-(?:wal|shm|journal)$/.test(path)) throw new Error("SQLite sidecars are not portable connected state.");
  return path;
}

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
    throw error;
  }
}

/** Checks every component so a safe-looking final name cannot traverse a link. */
async function inside(root: string, relative: string, writing = false): Promise<string> {
  assertSafeArchivePath(relative);
  const base = await lstat(root);
  if (!base.isDirectory() || base.isSymbolicLink()) throw new Error("Connected dependency root is unsafe.");
  let path = root;
  for (const component of relative.split("/")) {
    path = join(path, component);
    const info = await lstat(path).catch((error: NodeJS.ErrnoException) => {
      if (writing && error.code === "ENOENT") return undefined;
      throw error;
    });
    if (info?.isSymbolicLink()) throw new Error("Connected dependencies cannot follow symbolic links.");
  }
  return path;
}

async function sourceBytes(source: StreamingZipSource): Promise<Buffer> {
  return source.contents ? Buffer.from(source.contents) : readFile(source.sourcePath!);
}

async function addFile(archive: ConnectedArchive, path: string, source: StreamingZipSource, signal?: AbortSignal): Promise<void> {
  safeDependencyPath(path);
  signal?.throwIfAborted();
  const info = source.sourcePath ? await lstat(source.sourcePath) : undefined;
  if (info && (!info.isFile() || info.isSymbolicLink())) throw new Error("Connected dependency is not a regular file.");
  const descriptor = { sha256: source.contents ? hash(source.contents) : await fileHash(source.sourcePath!, signal),
    bytes: info?.size ?? source.contents!.byteLength };
  if (archive.manifest.files[path] && canonical(archive.manifest.files[path]) !== canonical(descriptor)) {
    throw new Error("Connected dependency has conflicting committed content.");
  }
  archive.manifest.files[path] = descriptor;
  archive.files.set(path, source);
}

function emptyArchive(): ConnectedArchive {
  return { manifest: { format: FORMAT, formatVersion: 1, files: {} }, files: new Map() };
}

/** Collect only committed regional generations, synthesizing private mirrors.
 * Immutable parent trees retain ALL original files, including source code.
 */
export async function collectConnectedArchive(options: {
  instanceRoot: string;
  repositoryRoot?: string;
  scopes: ConnectedSourceScope[];
  stagingDirectory: string;
  collectSubstrate: SubstrateCollector;
  operation?: BrainStorageOperationHooks;
}): Promise<ConnectedArchive | undefined> {
  const { instanceRoot, scopes, operation } = options;
  if (!scopes.some(scope => record(scope.metadata).connected_regions !== undefined)) return undefined;
  const archive = emptyArchive();
  const add = (path: string, source: StreamingZipSource) => addFile(archive, path, source, operation?.signal);
  await mkdir(options.stagingDirectory, { recursive: true });
  const snapshotSqlite = async (source: string): Promise<string> => {
    operation?.signal.throwIfAborted();
    const wal = await lstat(source + "-wal").catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined;
      throw error;
    });
    if (wal?.isSymbolicLink()) throw new Error("Connected SQLite WAL is unsafe.");
    const target = join(options.stagingDirectory, `${randomUUID()}.sqlite3`);
    await operation?.checkDisk(options.stagingDirectory, (await lstat(source)).size + (wal?.size ?? 0));
    await snapshotMutableSqliteIsolated(source, target);
    operation?.signal.throwIfAborted();
    return target;
  };
  const walk = async (relative: string): Promise<void> => {
    const directory = await inside(instanceRoot, relative);
    for (const entry of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      operation?.signal.throwIfAborted();
      const path = `${relative}/${entry.name}`;
      if (entry.isSymbolicLink()) throw new Error("Connected saved history contains a symbolic link.");
      if (entry.isDirectory()) { await walk(path); continue; }
      if (!entry.isFile()) throw new Error("Connected saved history is not a regular file.");
      if (/\.sqlite3?-(?:wal|shm|journal)$/.test(entry.name)) {
        // Read-only WAL handles may create a nonempty process-local SHM
        // lock table. It contains no saved rows; a nonempty WAL/journal does.
        if (!entry.name.endsWith("-shm") && path.startsWith("merge/parents/") && (await lstat(join(directory, entry.name))).size > 0) {
          throw new Error(`Immutable parent history has a live SQLite sidecar: ${path}`);
        }
        continue;
      }
      const sourcePath = path === "merge/host-history.sqlite3"
        ? await snapshotSqlite(join(directory, entry.name)) : join(directory, entry.name);
      await add(path, { name: path, sourcePath });
      await operation?.checkpoint({ phase: "planning", label: "Validating complete connected parent history" });
    }
  };
  await walk("merge");
  await add("full-merge.json", { name: "full-merge.json", sourcePath: await inside(instanceRoot, "full-merge.json") });

  const collectGraphs = async (engine: string, destination: string, states: unknown): Promise<void> => {
    const graph = record(states), regions = record(graph.regions);
    if (graph.format !== "omni-connected-graph-states" || graph.formatVersion !== 1 || !HASH.test(String(graph.sha256))) {
      throw new Error("Invalid committed connected graph state.");
    }
    for (const [id, value] of Object.entries(regions)) {
      const region = record(value);
      if (!ID.test(id) || region.enginePath !== `engine/connected-state/${id}`) throw new Error("Unsafe connected graph path.");
      const sourceEngine = await inside(engine, `connected-state/${id}`);
      const prefix = `${destination}/connected-state/${id}`;
      const metadata = { substrate: record(region.substrate) };
      await add(`${prefix}/brain.json`, { name: "brain.json", contents: Buffer.from(canonical(metadata)) });
      const pointer = record(metadata.substrate.persistence);
      if (typeof pointer.generationManifest !== "string" || !/^generations\/[a-f0-9]{64}\/manifest\.json$/.test(pointer.generationManifest)) {
        throw new Error("Unsafe connected generation path.");
      }
      const generationPath = await inside(sourceEngine, `substrate/${pointer.generationManifest}`);
      const generation = record(JSON.parse(await readFile(generationPath, "utf8")));
      if (!Array.isArray(generation.shards)) throw new Error("Connected generation has no shard inventory.");
      for (const row of generation.shards) for (const key of ["records", "tensors"]) {
        const value = record(row)[key];
        if (key === "tensors" && value === null) continue;
        const descriptor = record(value);
        if (typeof descriptor.path !== "string" || !/^blobs\/[a-f0-9]{64}\.(?:json|safetensors)$/.test(descriptor.path)) {
          throw new Error("Unsafe connected shard path.");
        }
        await inside(sourceEngine, `substrate/${descriptor.path}`);
      }
      const substrate = await options.collectSubstrate(sourceEngine, `${prefix}/substrate`, metadata, operation?.signal);
      if (!substrate) throw new Error("Connected graph is missing its committed generation.");
      for (const source of substrate.sources) {
        if (source.sourcePath) await inside(sourceEngine, relative(sourceEngine, source.sourcePath).split(sep).join("/"));
        await add(source.name, source);
      }
      if (region.nestedGraphStates !== undefined) {
        if (region.nestedStateRoot !== `${region.enginePath}/nested`) throw new Error("Unsafe nested connected graph root.");
        await collectGraphs(join(sourceEngine, "nested", "engine"), `${prefix}/nested/engine`, region.nestedGraphStates);
      }
    }
  };
  for (const scope of scopes) {
    const metadata = record(scope.metadata);
    if (metadata.connected_regions === undefined) continue;
    if (!["engine", "engine/origin"].includes(scope.engine)) throw new Error("Unsupported connected engine scope.");
    if (await exists(join(scope.sourceEngine, ".connected-working-publication.json"))) {
      throw new Error("Recover the unfinished inherited working-page publication before copying this connected checkpoint.");
    }
    await collectGraphs(scope.sourceEngine, scope.engine, metadata.connected_graph_states);
    const collectInputs = async (relative: string): Promise<void> => {
      const sourceDirectory = await inside(scope.sourceEngine, relative);
      for (const entry of await readdir(sourceDirectory, { withFileTypes: true })) {
        operation?.signal.throwIfAborted();
        const path = `${relative}/${entry.name}`;
        if (entry.isSymbolicLink()) throw new Error("Connected native inputs contain a symbolic link.");
        if (entry.isDirectory()) { await collectInputs(path); continue; }
        if (!entry.isFile()) throw new Error("Connected native input is not a regular file.");
        const sourcePath = join(sourceDirectory, entry.name);
        if (/\.sqlite3?-(?:wal|shm|journal)$/.test(entry.name)) {
          if (!entry.name.endsWith("-shm") && (await lstat(sourcePath)).size > 0) throw new Error(`Immutable connected native input has a live SQLite sidecar: ${path}`);
          continue;
        }
        await add(`${scope.engine}/${path}`, { name: path, sourcePath });
      }
    };
    if (await exists(join(scope.sourceEngine, "connected-inputs"))) await collectInputs("connected-inputs");
    for (const [relative, path] of await collectConnectedExecutionInputs(scope.sourceEngine, metadata)) {
      // These SQL files are already frozen by the exact runtime seal; a
      // SQLite backup could change their bytes and invalidate that binding.
      await add(`${scope.engine}/${relative}`, { name: relative, sourcePath: path });
    }
    for (const name of ["connected-runtime.sqlite3", "connected-origin.json", "connected-assembly.json"]) {
      const source = join(scope.sourceEngine, name);
      if (!await exists(source)) {
        if (name === "connected-runtime.sqlite3" && metadata.connected_runtime_adoption !== undefined) throw new Error("Connected runtime provenance is missing.");
        continue;
      }
      await inside(scope.sourceEngine, name);
      const sourcePath = name.endsWith(".sqlite3") ? await snapshotSqlite(source) : source;
      await add(`${scope.engine}/${name}`, { name, sourcePath });
    }
  }
  // Older complete merges may refer to installation-level content blobs.
  // Only authenticated sha256 IDs are resolved, never saved absolute paths.
  const sharedHashes = new Set<string>();
  const collectReferences = (value: unknown): void => {
    if (Array.isArray(value)) { value.forEach(collectReferences); return; }
    if (!value || typeof value !== "object") return;
    for (const [key, child] of Object.entries(value)) {
      if (key === "blobHash") {
        if (typeof child !== "string" || !HASH.test(child)) throw new Error("Saved parent source blob reference is invalid.");
        sharedHashes.add(child);
      } else if (child && typeof child === "object") collectReferences(child);
    }
  };
  for (const [path, source] of archive.files) {
    if (/^merge\/parents\/.+\/(?:brain|origin)\.json$/.test(path)) collectReferences(JSON.parse((await sourceBytes(source)).toString("utf8")));
    if (/^merge\/parents\/.+\/activity\/ledger\.sqlite3$/.test(path)) {
      const db = new DatabaseSync(sqliteFilename(source.sourcePath!), { readOnly: true });
      try {
        for (const row of db.prepare("SELECT payload_json FROM training_source_versions ORDER BY sequence").iterate()) {
          collectReferences(JSON.parse(String(row.payload_json)));
        }
      } finally { db.close(); }
    }
  }
  for (const digest of sharedHashes) {
    const path = `merge/shared-blobs/${digest}`;
    if (!archive.files.has(path)) {
      if (!options.repositoryRoot) throw new Error("A saved parent source blob is missing from its connected archive.");
      const sourcePath = await inside(options.repositoryRoot, `.blobs/${digest}`);
      if (await streamFileSha256(sourcePath) !== digest) throw new Error("Saved parent source blob checksum failed.");
      await add(path, { name: path, sourcePath });
    }
  }
  await validateConnectedArchive(archive, scopes, operation?.signal);
  return archive;
}

export function connectedArchiveSources(archive: ConnectedArchive, prefix = "connected"): StreamingZipSource[] {
  assertSafeArchivePath(prefix);
  const sources: StreamingZipSource[] = [{ name: `${prefix}/manifest.json`, contents: Buffer.from(canonical(archive.manifest)) }];
  const seen = new Set<string>();
  for (const [path, source] of archive.files) {
    const sha = archive.manifest.files[path]!.sha256;
    if (seen.has(sha)) continue;
    seen.add(sha);
    sources.push({ ...source, name: `${prefix}/payloads/${sha}` });
  }
  return sources;
}

export async function readConnectedArchive(
  entries: Map<string, { path: string }>, prefix = "connected", signal?: AbortSignal
): Promise<ConnectedArchive | undefined> {
  const names = [...entries.keys()].filter(name => name.startsWith(`${prefix}/`));
  if (!names.length) return undefined;
  const entry = entries.get(`${prefix}/manifest.json`);
  if (!entry) throw new Error("Connected archive has no dependency manifest.");
  const manifest = record(JSON.parse(await readFile(entry.path, "utf8")));
  if (manifest.format !== FORMAT || manifest.formatVersion !== 1) throw new Error("Unsupported connected dependency archive.");
  const files = record(manifest.files), archive = emptyArchive(), declared = new Set([`${prefix}/manifest.json`]);
  const verifiedPayloads = new Map<string, DependencyDescriptor>();
  for (const [path, value] of Object.entries(files)) {
    signal?.throwIfAborted();
    safeDependencyPath(path);
    const descriptor = record(value);
    if (!HASH.test(String(descriptor.sha256)) || !Number.isSafeInteger(descriptor.bytes) || Number(descriptor.bytes) < 0 ||
      Object.keys(descriptor).some(key => !["sha256", "bytes"].includes(key))) throw new Error("Invalid connected payload descriptor.");
    const name = `${prefix}/payloads/${descriptor.sha256}`, payload = entries.get(name);
    if (!payload) throw new Error("Connected dependency payload is missing.");
    declared.add(name);
    const verified = verifiedPayloads.get(name);
    if (verified) {
      archive.manifest.files[path] = verified;
      archive.files.set(path, { name, sourcePath: payload.path });
    } else {
      await addFile(archive, path, { name, sourcePath: payload.path }, signal);
      verifiedPayloads.set(name, archive.manifest.files[path]!);
    }
    if (canonical(archive.manifest.files[path]) !== canonical(descriptor)) throw new Error("Connected dependency payload checksum failed.");
  }
  if (names.some(name => !declared.has(name))) throw new Error("Connected archive contains an undeclared payload.");
  return archive;
}

export async function readConnectedArchiveDirectory(root: string, prefix = "connected", signal?: AbortSignal): Promise<ConnectedArchive | undefined> {
  if (!await exists(join(root, prefix))) return undefined;
  const entries = new Map<string, { path: string }>();
  const visit = async (relative: string): Promise<void> => {
    const directory = await inside(root, relative);
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (entry.isSymbolicLink()) throw new Error("Connected archive payload is a symbolic link.");
      const name = `${relative}/${entry.name}`;
      if (entry.isDirectory()) await visit(name);
      else if (entry.isFile()) entries.set(name, { path: join(directory, entry.name) });
      else throw new Error("Connected archive payload is not a regular file.");
    }
  };
  await visit(prefix);
  return readConnectedArchive(entries, prefix, signal);
}

/** Verify both graph closure and immutable descriptor/index dependencies.
 * Hashes of native descriptors themselves remain the native loader's contract
 * (Python float canonicalization differs); their referenced file bytes are
 * independently authenticated here, without rewriting those descriptors.
 */
export async function validateConnectedArchive(archive: ConnectedArchive | undefined, scopes: ConnectedArchiveScope[], signal?: AbortSignal): Promise<void> {
  const connected = scopes.filter(scope => record(scope.metadata).connected_regions !== undefined);
  if (!connected.length) { if (archive) throw new Error("Unconnected engine has undeclared connected dependencies."); return; }
  if (!archive) throw new Error("Connected checkpoint is missing its portable dependency closure.");
  const requireFile = (path: string, expected?: unknown): StreamingZipSource => {
    signal?.throwIfAborted();
    safeDependencyPath(path);
    const source = archive.files.get(path), descriptor = archive.manifest.files[path];
    if (!source || !descriptor || (expected !== undefined && descriptor.sha256 !== expected)) throw new Error(`Connected dependency is missing or changed: ${path}`);
    return source;
  };
  const json = async (path: string, expected?: unknown): Promise<Record<string, unknown>> =>
    record(JSON.parse((await sourceBytes(requireFile(path, expected))).toString("utf8")));
  const checkedParents = new Set<string>();
  const parentClosure = async (descriptorValue: unknown, root: string): Promise<void> => {
    const descriptor = record(descriptorValue);
    if (descriptor.format !== "omni-connected-native-regions" || descriptor.formatVersion !== 1 || !HASH.test(String(descriptor.sha256)) ||
      !Array.isArray(descriptor.regions) || !descriptor.regions.length || !Array.isArray(descriptor.parents) || descriptor.parents.length < 2) {
      throw new Error("Invalid connected parent descriptor.");
    }
    const indexPath = `${root}merge/parents.jsonl`;
    if (!checkedParents.has(indexPath)) {
      checkedParents.add(indexPath);
      const source = requireFile(indexPath);
      const input = source.sourcePath ? createReadStream(source.sourcePath) : undefined;
      const lines: AsyncIterable<string> | Iterable<string> = input ? createInterface({ input, crlfDelay: Infinity }) :
        Buffer.from(source.contents!).toString("utf8").split(/\r?\n/);
      let first = true;
      try {
        for await (const line of lines) {
          signal?.throwIfAborted();
          if (!line.trim()) continue;
          const row = record(JSON.parse(line));
          if (first) {
            first = false;
            if (row.format !== "omni-full-merge-parent-index" || row.formatVersion !== 1 || !Array.isArray(row.parents)) throw new Error("Connected parent index is invalid.");
            const seals = new Map(row.parents.map(value => {
              const parent = record(value);
              return [String(parent.brainId ?? parent.parentId), parent.generationSha256];
            }));
            if (seals.size !== descriptor.parents.length || descriptor.parents.some(value => {
              const parent = record(value);
              return seals.get(String(parent.parentId)) !== parent.generationSha256;
            })) throw new Error("Connected parent index lost its sealed parent generations.");
            continue;
          }
          if (!ID.test(String(row.parentId)) || typeof row.path !== "string" || !HASH.test(String(row.sha256)) ||
            !Number.isSafeInteger(row.bytes) || Number(row.bytes) < 0) throw new Error("Connected parent index path or digest is unsafe.");
          assertSafeArchivePath(row.path);
          const path = `${root}merge/parents/${row.parentId}/${row.path}`;
          requireFile(path, row.sha256);
          if (archive.manifest.files[path]!.bytes !== row.bytes) throw new Error("Connected parent snapshot size changed.");
        }
      } finally { input?.destroy(); }
      if (first) throw new Error("Connected parent index is empty.");
      const marker = await json(`${root}full-merge.json`);
      if (marker.format !== "omni-full-brain-merge" || marker.formatVersion !== 1) throw new Error("Connected host merge provenance is invalid.");
    }
    const ids = new Set<string>();
    for (const value of descriptor.regions) {
      const region = record(value);
      if (!ID.test(String(region.regionId)) || ids.has(String(region.regionId)) || !ID.test(String(region.parentId)) ||
        region.enginePath !== `merge/parents/${region.parentId}/engine` ||
        ["metadataSha256", "coreSha256", "plasticitySha256", "generationSha256", "neuralSha256"].some(key => !HASH.test(String(region[key])))) {
        throw new Error("Connected immutable parent path or digest is unsafe.");
      }
      ids.add(String(region.regionId));
      const engine = `${root}${region.enginePath}`;
      const metadata = await json(`${engine}/brain.json`, region.metadataSha256);
      requireFile(`${engine}/core.safetensors`, region.coreSha256);
      requireFile(`${engine}/plasticity.safetensors`, region.plasticitySha256);
      if (metadata.brain_id !== region.parentId) throw new Error("Connected immutable parent identity changed.");
      if (metadata.connected_regions !== undefined) {
        await parentClosure(metadata.connected_regions, `${root}merge/parents/${region.parentId}/`);
        await inputAliases(metadata.connected_regions, engine);
      }
    }
    const parents = new Set<string>();
    for (const value of descriptor.parents) {
      const parent = record(value);
      if (!ID.test(String(parent.parentId)) || parents.has(String(parent.parentId)) || !ids.has(String(parent.regionId)) ||
        !HASH.test(String(parent.generationSha256))) throw new Error("Connected parent provenance is invalid.");
      parents.add(String(parent.parentId));
      requireFile(`${root}merge/parents/${parent.parentId}/brain.json`);
    }
  };
  const graphClosure = async (value: unknown, engine: string, descriptorValue?: unknown): Promise<void> => {
    const graph = record(value), regions = record(graph.regions);
    if (graph.format !== "omni-connected-graph-states" || graph.formatVersion !== 1 || !HASH.test(String(graph.sha256))) throw new Error("Invalid connected graph binding.");
    if (descriptorValue !== undefined) {
      const descriptor = record(descriptorValue);
      if (graph.descriptorSha256 !== descriptor.sha256 || canonical(Object.keys(regions).sort()) !==
        canonical((descriptor.regions as unknown[]).map(value => String(record(value).regionId)).sort())) throw new Error("Connected graph does not bind the retained owners.");
    }
    for (const [id, value] of Object.entries(regions)) {
      const region = record(value);
      if (!ID.test(id) || region.enginePath !== `engine/connected-state/${id}`) throw new Error("Unsafe connected graph path.");
      const prefix = `${engine}/connected-state/${id}`, substrate = record(region.substrate), pointer = record(substrate.persistence);
      if (canonical((await json(`${prefix}/brain.json`)).substrate) !== canonical(substrate) ||
        canonical(await json(`${prefix}/substrate/manifest.json`)) !== canonical(pointer)) throw new Error("Connected graph mirror differs from the root commit.");
      if (typeof pointer.generationManifest !== "string" || !/^generations\/[a-f0-9]{64}\/manifest\.json$/.test(pointer.generationManifest) ||
        !HASH.test(String(pointer.generationManifestSha256))) throw new Error("Unsafe connected generation path.");
      const generation = await json(`${prefix}/substrate/${pointer.generationManifest}`, pointer.generationManifestSha256);
      if (!Array.isArray(generation.shards) || generation.contentSha256 !== pointer.activeGeneration) throw new Error("Connected generation differs from its committed pointer.");
      for (const row of generation.shards) for (const key of ["records", "tensors"]) {
        const value = record(row)[key];
        if (key === "tensors" && value === null) continue;
        const blob = record(value);
        if (!HASH.test(String(blob.sha256)) || typeof blob.path !== "string" ||
          !new RegExp(`^blobs/${blob.sha256}\\.(?:json|safetensors)$`).test(blob.path)) throw new Error("Unsafe connected shard path.");
        const path = `${prefix}/substrate/${blob.path}`;
        requireFile(path, blob.sha256);
        if (archive.manifest.files[path]!.bytes !== blob.bytes) throw new Error("Connected shard size changed.");
      }
      if (region.nestedGraphStates !== undefined) {
        if (region.nestedStateRoot !== `${region.enginePath}/nested`) throw new Error("Unsafe nested graph root.");
        await graphClosure(region.nestedGraphStates, `${prefix}/nested/engine`);
      }
    }
  };
  const inputAliases = async (descriptorValue: unknown, engine: string): Promise<void> => {
    const descriptor = record(descriptorValue), prefix = `${engine}/connected-inputs/`;
    if (![...archive.files.keys()].some(path => path.startsWith(prefix))) return;
    for (const value of descriptor.regions as unknown[]) {
      const region = record(value);
      if (region.enginePath !== `merge/parents/${region.parentId}/engine` || !ID.test(String(region.parentId))) {
        throw new Error("Connected input alias has an unsafe parent path.");
      }
      const path = `${prefix}${region.enginePath}`;
      const metadata = await json(`${path}/brain.json`, region.metadataSha256);
      requireFile(`${path}/core.safetensors`, region.coreSha256);
      requireFile(`${path}/plasticity.safetensors`, region.plasticitySha256);
      if (metadata.brain_id !== region.parentId) throw new Error("Connected input alias changed parent identity.");
      if (metadata.connected_regions !== undefined) {
        await graphClosure(metadata.connected_graph_states, path, metadata.connected_regions);
        await inputAliases(metadata.connected_regions, path);
      }
      const originPath = `${path}/origin/brain.json`;
      if (archive.files.has(originPath)) {
        const origin = await json(originPath);
        if (origin.connected_regions !== undefined) {
          await graphClosure(origin.connected_graph_states, `${path}/origin`, origin.connected_regions);
          await inputAliases(origin.connected_regions, `${path}/origin`);
        }
      }
    }
  };
  const sqlite = (path: string): void => {
    const source = requireFile(path);
    if (!source.sourcePath) throw new Error("Connected SQLite state has no saved file.");
    const db = new DatabaseSync(sqliteFilename(source.sourcePath), { readOnly: true });
    try {
      if (db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Connected SQLite state failed integrity verification.");
    } finally { db.close(); }
  };
  sqlite("merge/host-history.sqlite3");
  for (const scope of connected) {
    const metadata = record(scope.metadata);
    await parentClosure(metadata.connected_regions, "");
    await graphClosure(metadata.connected_graph_states, scope.engine, metadata.connected_regions);
    await inputAliases(metadata.connected_regions, scope.engine);
    await validateConnectedExecutionArchive(metadata, scope.engine, (path) => archive.files.get(path));
    if (metadata.connected_runtime_adoption !== undefined) sqlite(`${scope.engine}/connected-runtime.sqlite3`);
    if (scope.engine === "engine/origin") requireFile(`${scope.engine}/connected-origin.json`);
  }
  for (const [path, descriptor] of Object.entries(archive.manifest.files)) {
    if (/^merge\/(?:shared-blobs|blobs)\/[a-f0-9]{64}$/.test(path) && path.split("/").at(-1) !== descriptor.sha256) {
      throw new Error("Connected content-addressed blob checksum failed.");
    }
  }
}

/** Write an opaque package, used by recovery points as well as ZIP exports. */
export async function saveConnectedArchive(archive: ConnectedArchive, root: string, operation?: BrainStorageOperationHooks): Promise<string> {
  for (const source of connectedArchiveSources(archive)) {
    operation?.signal.throwIfAborted();
    const target = await inside(root, source.name, true);
    await operation?.checkDisk(root, source.contents?.byteLength ?? (await lstat(source.sourcePath!)).size);
    if (source.contents) await writeMutableFileIsolated(target, source.contents);
    else await copyMutableFileIsolated(source.sourcePath!, target, { expectedSha256: source.name.split("/").at(-1), mode: 0o600 });
  }
  return hash(canonical(archive.manifest));
}

/** All outputs use private inodes/reflinks, including mutable SQLite/pointers.
 * Caller owns the unpublished directory and its cancellation/rollback lifetime.
 */
export async function materializeConnectedArchive(archive: ConnectedArchive, root: string, options: {
  operation?: BrainStorageOperationHooks;
  engineDirectory?: string;
  onSharedBlob?: (hash: string, source: string) => Promise<void>;
} = {}): Promise<void> {
  for (const [relative, source] of archive.files) {
    options.operation?.signal.throwIfAborted();
    safeDependencyPath(relative);
    const isEngine = relative.startsWith("engine/");
    const targetRoot = isEngine && options.engineDirectory ? options.engineDirectory : root;
    const path = isEngine && options.engineDirectory ? relative.slice(7) : relative;
    await mkdir(targetRoot, { recursive: true });
    const target = await inside(targetRoot, path, true), descriptor = archive.manifest.files[relative]!;
    await options.operation?.checkDisk(root, descriptor.bytes);
    if (source.contents) {
      if (hash(source.contents) !== descriptor.sha256) throw new Error("Connected payload changed before installation.");
      await writeMutableFileIsolated(target, source.contents);
    } else await copyMutableFileIsolated(source.sourcePath!, target, { expectedSha256: descriptor.sha256, mode: 0o600 });
    if (/^merge\/shared-blobs\/[a-f0-9]{64}$/.test(relative)) await options.onSharedBlob?.(descriptor.sha256, target);
    await options.operation?.checkpoint({ phase: "materializing", label: "Restoring verified connected graph and parent history" });
  }
}
