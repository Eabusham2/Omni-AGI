import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { dirname, join, relative, sep } from "node:path";
import { backup, DatabaseSync } from "node:sqlite";
import type { BrainDocument, ConversationLedgerSummary, ActionEvent, ChatMessage, ThoughtTrace, JournalEntry, TrainingSource } from "../shared/types";
import type { FullMergeHistoryPage, FullMergeParentSeal, FullMergeStartRequest } from "../shared/fullMerge";
import type { FullMergeAssemblyInput } from "./fullBrainMerge";
import { ConversationLedger } from "./conversationLedger";
import { BrainActivityLedger } from "./brainActivityLedger";
import { ArtifactIndexStore, parseArtifactIndex, type PersistedGeneratedArtifact } from "./artifactIndexStore";
import { captureMcpTaskCustody } from "./mcpTaskCustody";
import { MCP_TASK_DIRECTORY } from "./mcpTaskStore";
import { adoptFullMergeMcpTasks } from "./fullMergeMcpTasks";

export function mergeCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(mergeCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, child]) => `${JSON.stringify(key)}:${mergeCanonical(child)}`).join(",")}}`;
  const result = JSON.stringify(value);
  if (result === undefined) throw new Error("Full merge requires explicit JSON settings.");
  return result;
}
const hash = (value: string | Buffer): string => createHash("sha256").update(value).digest("hex");
const HASH = /^[a-f0-9]{64}$/;
const verifiedOrigins = new Map<string, { fingerprint: string; marker: Record<string, unknown> }>();
const record = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid native merge declaration.");
  return value as Record<string, unknown>;
};
export async function fullMergeFileHash(path: string, signal?: AbortSignal): Promise<string> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Full merge requires regular managed files.");
  const digest = createHash("sha256");
  for await (const bytes of createReadStream(path, { highWaterMark: 1024 * 1024, signal })) digest.update(bytes);
  return digest.digest("hex");
}

/** Native marker contains only integers, strings and booleans; no float canonicalization. */
export async function verifyConnectedNativeOrigin(engine: string, brainId: string,
  options: { allowAncestralIdentity?: boolean } = {}): Promise<Record<string, unknown>> {
  const metadata = record(JSON.parse(await readFile(join(engine, "brain.json"), "utf8")));
  const descriptor = record(metadata.connected_regions);
  const origin = join(engine, "origin");
  for (const path of [origin, join(origin, "packed-ternary")]) {
    const info = await lstat(path);
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Merged native origin is not an owned directory.");
  }
  const files = ["connected-origin.json", "brain.json", "core.safetensors", "plasticity.safetensors", "packed-ternary/manifest.json"];
  const fingerprint = async (): Promise<string> => {
    const stats = [];
    for (const path of files) {
      const info = await lstat(join(origin, path), { bigint: true });
      if (!info.isFile() || info.isSymbolicLink()) throw new Error("Merged native origin contains a nonregular file.");
      stats.push([path, info.dev, info.ino, info.size, info.mtimeNs, info.ctimeNs].map(String));
    }
    return JSON.stringify([brainId, descriptor.sha256, options.allowAncestralIdentity === true, stats]);
  };
  const before = await fingerprint();
  const cached = verifiedOrigins.get(engine);
  if (metadata.brain_id === brainId && cached?.fingerprint === before) return structuredClone(cached.marker);
  const marker = record(JSON.parse(await readFile(join(origin, "connected-origin.json"), "utf8")));
  const { sha256, ...body } = marker;
  if (metadata.brain_id !== brainId || body.format !== "omni-connected-native-origin" || body.formatVersion !== 1 ||
      (!options.allowAncestralIdentity && body.brainId !== brainId) || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(String(body.brainId)) ||
      body.descriptorSha256 !== descriptor.sha256 || !HASH.test(String(body.descriptorSha256)) ||
      body.initialCurriculumRerun !== false || hash(mergeCanonical(body)) !== sha256) {
    throw new Error("Merged native origin failed authentication.");
  }
  for (const [path, field] of [["brain.json", "metadataSha256"], ["core.safetensors", "coreSha256"],
    ["plasticity.safetensors", "plasticitySha256"], ["packed-ternary/manifest.json", "packedManifestSha256"]]) {
    if (await fullMergeFileHash(join(origin, path!)) !== body[field!]) throw new Error("Merged native origin bytes changed.");
  }
  const saved = record(JSON.parse(await readFile(join(origin, "brain.json"), "utf8")));
  const packed = record(JSON.parse(await readFile(join(origin, "packed-ternary", "manifest.json"), "utf8")));
  const packedMetadata = record(packed.metadata);
  if (saved.brain_id !== body.brainId || record(saved.connected_regions).sha256 !== body.descriptorSha256 ||
      record(saved.connected_runtime_adoption).sha256 !== body.runtimeAdoptionSha256 ||
      packedMetadata.brainId !== body.brainId || packedMetadata.parameterChecksum !== body.parameterChecksum) {
    throw new Error("Merged origin lost its native and runtime binding.");
  }
  if (await fingerprint() !== before) throw new Error("Merged native origin changed during verification.");
  if (verifiedOrigins.size >= 32) verifiedOrigins.delete(verifiedOrigins.keys().next().value!);
  verifiedOrigins.set(engine, { fingerprint: before, marker: structuredClone(marker) });
  return marker;
}

export function validateFullMergeReview(request: FullMergeStartRequest): void {
  if (!request || typeof request.name !== "string" || !request.name.trim() || !request.config ||
      typeof request.config !== "object" || Array.isArray(request.config)) throw new Error("Review the complete destination settings.");
  const policy = request?.accessPolicy;
  if (!["micro", "personal", "gpu", "workstation"].includes(request?.hardwareTier)) throw new Error("Review the destination hardware tier and RAM policy.");
  if (request.spillPoolSelectionId !== undefined && !/^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/.test(request.spillPoolSelectionId)) throw new Error("Invalid destination spill selection.");
  if (request.config.deviceResources?.storageDirectories?.length && !request.spillPoolSelectionId) throw new Error("Choose destination spill drives locally; parent drive permissions are not inherited.");
  if (!policy || policy.credentials !== "none" || policy.mcp !== "disabled" || !Array.isArray(policy.toolPermissions)) {
    throw new Error("Review the destination access policy. Credentials and MCP access are not inherited.");
  }
  const ids = new Set<string>();
  for (const permission of policy.toolPermissions) {
    if (!permission || !/^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}$/.test(permission.toolId) ||
      ids.has(permission.toolId) || !["off", "ask", "auto", "full"].includes(permission.level)) {
      throw new Error("Invalid reviewed destination tool permission.");
    }
    ids.add(permission.toolId);
  }
  if (request.config.idleCognition) throw new Error("Start the new instance's Active mode after its merge completes.");
  if (request.config.nativeArchitecture) throw new Error("Full merge preserves parent geometry; do not supply a single-stack architecture override.");
  if (request.config.name !== request.name.trim()) throw new Error("The reviewed name and destination settings disagree.");
}

/** Includes host receipts, native cursors, and WAL bytes, not only neural weights. */
export async function sealFullMergeParent(root: string, brain: BrainDocument, signal?: AbortSignal,
  reserve?: (bytes: number) => Promise<void>, admit?: (bytes: number, purpose: string) => Promise<void> | void): Promise<FullMergeParentSeal> {
  const digest = createHash("sha256");
  const visit = async (directory: string): Promise<void> => {
    for (const entry of (await readdir(directory, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))) {
      signal?.throwIfAborted();
      const path = join(directory, entry.name);
      const info = await lstat(path);
      if (info.isSymbolicLink()) throw new Error("A full-merge parent contains a symbolic link.");
      if (info.isDirectory()) {
        if (entry.name === MCP_TASK_DIRECTORY && /(?:^|\/)engine(?:\/origin)?$/.test(relative(root, directory).split(sep).join("/"))) {
          const stage = await mkdtemp(join(dirname(root), ".mcp-task-seal-"));
          try {
            const captured = await captureMcpTaskCustody(directory, join(stage, "engine"), { signal, admit, checkDisk: reserve });
            for (const [local, saved] of [...captured].sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)) {
              digest.update(mergeCanonical([`${relative(root, directory).split(sep).join("/")}/${local}`,
                (await lstat(saved)).size, await fullMergeFileHash(saved, signal)]));
            }
          } finally { await rm(stage, { recursive: true, force: true }); }
        } else await visit(path);
        continue;
      }
      if (/\.sqlite3?-(?:shm|wal)$/.test(entry.name)) continue; // represented by the consistent snapshot below
      let bytes = info.size, sha: string;
      const logical = relative(root, path).split(sep).join("/");
      const sealedLearning = /(?:^|\/)(?:learning\/(?:archives|source-blobs)\/[a-f0-9]{64}\.(?:sqlite3|data)|api-teachers\/archives\/[a-f0-9]{64}\.sqlite3)$/.test(logical);
      if (/(?:^|\/)api-teachers\//.test(logical)) {
        signal?.throwIfAborted();
        await admit?.((/\.sqlite3?$/.test(logical) && !sealedLearning ? 2 : 1) * 1024 * 1024,
          "teacher parent seal SQLite/hash windows");
        signal?.throwIfAborted();
      }
      if (/\.sqlite3?$/.test(entry.name) && !sealedLearning) {
        const wal = await lstat(`${path}-wal`).catch(() => undefined);
        await reserve?.(bytes + (wal?.size ?? 0));
        const temporary = await mkdtemp(join(dirname(root), ".full-merge-seal-"));
        const database = new DatabaseSync(path, { readOnly: true });
        try {
          database.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-1024; PRAGMA mmap_size=0;");
          const snapshot = join(temporary, "snapshot.sqlite3");
          await backup(database, snapshot, { rate: 128, progress: () => signal?.throwIfAborted() });
          bytes = (await lstat(snapshot)).size;
          sha = await fullMergeFileHash(snapshot, signal);
        } finally { database.close(); await rm(temporary, { recursive: true, force: true }); }
      } else {
        sha = await fullMergeFileHash(path, signal);
        if (sealedLearning && sha !== entry.name.split(".")[0]) throw new Error("An immutable learning archive changed before merge.");
      }
      digest.update(mergeCanonical([relative(root, path).split(sep).join("/"), bytes, sha]));
    }
  };
  await visit(root);
  const ancestry = [await fullMergeFileHash(join(root, "origin.json"))];
  try {
    const prior = record(JSON.parse(await readFile(join(root, "full-merge.json"), "utf8")));
    for (const value of prior.parents as FullMergeParentSeal[]) ancestry.push(value.generationSha256, ...value.ancestry);
  } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
  return { brainId: brain.id, name: brain.name, generationSha256: digest.digest("hex"), ancestry: [...new Set(ancestry)].sort() };
}

function hostDatabase(root: string, readOnly = false): DatabaseSync {
  const db = new DatabaseSync(join(root, "merge", "host-history.sqlite3"), { readOnly });
  if (!readOnly) db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA temp_store=FILE; PRAGMA cache_size=-1024;
    CREATE TABLE IF NOT EXISTS records(sequence INTEGER PRIMARY KEY, parent_id TEXT NOT NULL, path TEXT NOT NULL,
      source_id TEXT NOT NULL, kind TEXT NOT NULL, payload_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
      UNIQUE(parent_id,path,source_id,kind,payload_sha256));
    CREATE TABLE IF NOT EXISTS identities(parent_id TEXT, source_id TEXT, destination_id TEXT, PRIMARY KEY(parent_id,source_id));
    CREATE TABLE IF NOT EXISTS projections(kind TEXT,payload_sha256 TEXT,destination_id TEXT,PRIMARY KEY(kind,payload_sha256));`);
  return db;
}

export async function fullMergeHistory(root: string, brainId: string, after = 0, limit = 80): Promise<FullMergeHistoryPage> {
  if (!Number.isSafeInteger(after) || after < 0 || !Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid merge history cursor.");
  const db = hostDatabase(root, true);
  try {
    const rows = db.prepare("SELECT * FROM records WHERE sequence>? ORDER BY sequence LIMIT ?").all(after, limit + 1);
    const selected = rows.slice(0, limit);
    return { brainId, entries: selected.map((row) => ({ sequence: Number(row.sequence), parentId: String(row.parent_id),
      path: String(row.path), kind: String(row.kind), sourceId: String(row.source_id), payloadSha256: String(row.payload_sha256),
      payload: JSON.parse(String(row.payload_json)), historicalOnly: true })),
      ...(rows.length > limit ? { nextAfterSequence: Number(selected.at(-1)!.sequence) } : {}) };
  } finally { db.close(); }
}

/** Raw originals remain in snapshots; common indexes are explicit historical projections, never execution queues. */
export async function adoptFullMergeHostState(input: FullMergeAssemblyInput, sharedBlobSource?: (sha256: string) => string): Promise<ConversationLedgerSummary> {
  await adoptFullMergeMcpTasks(input);
  const db = hostDatabase(input.targetDirectory);
  const conversation = await ConversationLedger.open(input.targetDirectory, input.targetBrainId);
  const activity = await BrainActivityLedger.open(input.targetDirectory, input.targetBrainId);
  const artifacts = await ArtifactIndexStore.open(join(input.targetDirectory, "engine", "artifacts"), input.targetBrainId);
  const nativeIndexPath = join(input.targetDirectory, "engine", "connected-runtime.sqlite3");
  const nativeIndex = await lstat(nativeIndexPath).then(() => new DatabaseSync(nativeIndexPath, { readOnly: true })).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined;
    throw error;
  });
  const add = (parentId: string, path: string, sourceId: string, kind: string, payload: unknown, raw?: string): string => {
    const json = raw ?? JSON.stringify(payload);
    const sha = hash(json);
    db.prepare("INSERT OR IGNORE INTO records(parent_id,path,source_id,kind,payload_sha256,payload_json) VALUES(?,?,?,?,?,?)")
      .run(parentId, path, sourceId, kind, sha, json);
    const existing = db.prepare("SELECT destination_id FROM projections WHERE kind=? AND payload_sha256=?").get(kind, sha);
    const prior = db.prepare("SELECT destination_id FROM identities WHERE parent_id=? AND source_id=?").get(parentId, sourceId);
    const native = ["message", "trace"].includes(kind) ? nativeIndex?.prepare(
      "SELECT target_id FROM conversation_sources WHERE parent_id=? AND kind=? AND original_id=? ORDER BY sequence LIMIT 1"
    ).get(parentId, kind, sourceId) : undefined;
    const id = String(native?.target_id ?? existing?.destination_id ?? prior?.destination_id ?? hash(`${kind}:${sha}`));
    db.prepare("INSERT OR IGNORE INTO identities VALUES(?,?,?)").run(parentId, sourceId, id);
    db.prepare("INSERT OR IGNORE INTO projections VALUES(?,?,?)").run(kind, sha, id);
    return id;
  };
  try {
    for (const parent of input.parents) {
      const root = await input.materializeParent(parent.brainId);
      const sourceDocument = record(JSON.parse(await readFile(join(root, "brain.json"), "utf8")));
      const artifactIndexes = new Set<string>();
      for (const file of input.files(parent.brainId)) {
        input.signal?.throwIfAborted();
        // Complete file-level provenance includes questions, approval receipts,
        // live observation records and interrupted external-action journals.
        add(parent.brainId, file.path, file.path, "file", { sha256: file.sha256, bytes: file.bytes,
          snapshotPath: `merge/parents/${parent.brainId}/${file.path}`, historicalOnly: true });
        if (["engine/artifacts/index.sqlite3", "engine/artifacts/index.json"].includes(file.path)) artifactIndexes.add(file.path);
        if (!["conversation/ledger.sqlite3", "activity/ledger.sqlite3"].includes(file.path)) continue;
        const source = new DatabaseSync(join(root, ...file.path.split("/")), { readOnly: true });
        try {
          const owner = source.prepare("SELECT value FROM meta WHERE key='brainId'").get();
          if (owner?.value !== parent.brainId) throw new Error("Parent host ledger has the wrong identity.");
          const tables = file.path.startsWith("conversation") ? ["entries"] : ["journal_entries", "training_source_versions"];
          for (const table of tables) {
            let previous = "0".repeat(64), sequence = 0;
            for (const row of source.prepare(`SELECT * FROM ${table} ORDER BY sequence`).iterate()) {
              input.signal?.throwIfAborted();
              const json = String(row.payload_json), payload = record(JSON.parse(json));
              const kind = table === "entries" ? String(row.kind) : table === "journal_entries" ? "journal" : "training-source";
              const basis = table === "entries" ? { sequence: row.sequence, entryKey: row.entry_key, kind: row.kind, createdAt: row.created_at,
                attentionEpoch: row.attention_epoch, payloadSha256: row.payload_sha256, previousSha256: row.previous_sha256 }
                : table === "journal_entries" ? { sequence: row.sequence, entryId: row.entry_id, createdAt: row.created_at, kind: row.journal_kind,
                  payloadSha256: row.payload_sha256, previousSha256: row.previous_sha256 }
                  : { sequence: row.sequence, sourceId: row.source_id, importedAt: row.imported_at, payloadSha256: row.payload_sha256, previousSha256: row.previous_sha256 };
              if (row.sequence !== ++sequence || row.previous_sha256 !== previous || hash(json) !== row.payload_sha256 || hash(JSON.stringify(basis)) !== row.row_sha256) {
                throw new Error("Parent host history checksum failed.");
              }
              previous = String(row.row_sha256);
              add(parent.brainId, file.path, String(payload.id), kind, payload, json);
            }
            if (table === "entries" && sourceDocument.conversation) {
              const head = record(sourceDocument.conversation);
              // Display summary may lag append-only delivery receipts, but its
              // committed prefix may never disappear or change.
              const prefix = Number(head.headSequence) === 0 ? "0".repeat(64)
                : source.prepare("SELECT row_sha256 FROM entries WHERE sequence=?").get(Number(head.headSequence))?.row_sha256;
              if (Number(head.headSequence) > sequence || prefix !== head.headSha256) throw new Error("Parent conversation lost its committed head.");
            } else if (table !== "entries") {
              const prefix = table === "journal_entries" ? "journal" : "trainingSource";
              const count = source.prepare("SELECT value FROM meta WHERE key=?").get(`${prefix}HeadSequence`)?.value;
              const head = source.prepare("SELECT value FROM meta WHERE key=?").get(`${prefix}HeadSha256`)?.value;
              if (Number(count) !== sequence || head !== previous) throw new Error("Parent activity lost its committed head.");
            }
          }
        } finally { source.close(); }
      }
      const sourceArtifacts = join(root, "engine", "artifacts");
      const indexed = artifactIndexes.has("engine/artifacts/index.sqlite3");
      let legacy: PersistedGeneratedArtifact[] = [];
      if (!indexed && artifactIndexes.has("engine/artifacts/index.json")) {
        legacy = parseArtifactIndex(JSON.parse(await readFile(join(sourceArtifacts, "index.json"), "utf8")), parent.brainId).artifacts;
      }
      const source = indexed ? new DatabaseSync(join(sourceArtifacts, "index.sqlite3"), { readOnly: true }) : undefined;
      try {
        if (source && source.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== parent.brainId) throw new Error("Parent artifact ledger has the wrong identity.");
        let sequence = 0, previous = "0".repeat(64);
        const rows: Iterable<Record<string, unknown>> = source
          ? source.prepare("SELECT * FROM artifacts ORDER BY sequence").iterate()
          : legacy.map((entry) => ({ payload_json: JSON.stringify(entry) }));
        for (const row of rows) {
          input.signal?.throwIfAborted();
          const json = String(row.payload_json), artifact = JSON.parse(json) as PersistedGeneratedArtifact;
          if (row.payload_sha256 && hash(json) !== row.payload_sha256) throw new Error("Parent artifact metadata changed.");
          if (source) {
            const rowHash = hash(JSON.stringify({ sequence: row.sequence, artifactId: row.artifact_id,
              payloadSha256: row.payload_sha256, previousSha256: row.previous_sha256 }));
            if (row.sequence !== ++sequence || row.previous_sha256 !== previous || row.row_sha256 !== rowHash || artifact.id !== row.artifact_id) throw new Error("Parent artifact ledger chain changed.");
            previous = rowHash;
          }
          if (!/^artifacts\/[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$/.test(artifact.relativePath)) throw new Error("Unsafe parent artifact path.");
          const sourcePath = join(root, "engine", artifact.relativePath);
          if (await fullMergeFileHash(sourcePath, input.signal) !== artifact.sha256 || (await lstat(sourcePath)).size !== artifact.bytes) throw new Error("Parent artifact content changed.");
          const id = add(parent.brainId, "engine/artifacts/index", artifact.id, "artifact", artifact, json);
          const relativePath = `artifacts/${artifact.sha256}.media`;
          await input.reserve(artifact.bytes, "merged artifact relocation");
          await copyFile(sourcePath, join(input.targetDirectory, "engine", relativePath));
          artifacts.append([{ ...artifact, id, relativePath }]);
        }
        if (source && (Number(source.prepare("SELECT value FROM meta WHERE key='headSequence'").get()?.value) !== sequence ||
          source.prepare("SELECT value FROM meta WHERE key='headSha256'").get()?.value !== previous)) throw new Error("Parent artifact ledger lost its committed head.");
      } finally { source?.close(); }
    }
    // Identities are complete before projecting references, including traces
    // emitted after their message and copied artifact references.
    const rebind = (value: unknown, parentId: string, key = ""): unknown => {
      if (Array.isArray(value)) return value.map((child) => rebind(child, parentId));
      if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([field, child]) => [field, rebind(child, parentId, field)]));
      if (typeof value === "string" && /(?:^id$|Id$)/.test(key)) return db.prepare("SELECT destination_id FROM identities WHERE parent_id=? AND source_id=?").get(parentId, value)?.destination_id ?? value;
      return value;
    };
    for (const row of db.prepare("SELECT * FROM records WHERE kind IN ('message','trace','action','journal','training-source') ORDER BY sequence").iterate()) {
      input.signal?.throwIfAborted();
      const original = JSON.parse(String(row.payload_json)) as Record<string, unknown>;
      const value = rebind(original, String(row.parent_id)) as Record<string, unknown>;
      value.mergeSource = { parentId: row.parent_id, path: row.path, sourceId: row.source_id, payloadSha256: row.payload_sha256, historicalOnly: true };
      if (row.kind === "action") {
        value.brainId = input.targetBrainId;
        // Returned data is opaque, even when user keys end in *Id. Rebind only
        // host correlation metadata, never content from file/code/tool results.
        if (original.execution && typeof original.execution === "object" && !Array.isArray(original.execution) && Object.hasOwn(original.execution, "output"))
          record(value.execution).output = record(original.execution).output;
        delete value.mcpAppView;
        // No live tool journal/approval is created. Old unsettled proposals are
        // visible history, not new-identity authorization or retry candidates.
        if (!["complete", "failed", "stopped"].includes(String(value.state))) {
          value.mergeSource = { ...record(value.mergeSource), originalState: value.state };
          value.state = "stopped";
        }
        conversation.append([{ kind: "action", value: value as unknown as ActionEvent }]);
      } else if (row.kind === "message") {
        // Immutable attribution names the original App/source owner. It is not
        // a root execution ID or authority to resume an imported request.
        for (const key of ["inputProvenance", "input_provenance"]) if (Object.hasOwn(original, key)) value[key] = original[key];
        if (value.inputReceipt) { record(value.mergeSource).originalInputReceipt = original.inputReceipt; delete value.inputReceipt; }
        conversation.append([{ kind: "message", value: value as unknown as ChatMessage }]);
      } else if (row.kind === "trace") conversation.append([{ kind: "trace", value: value as unknown as ThoughtTrace }]);
      else if (row.kind === "journal") activity.appendJournals([value as unknown as JournalEntry]);
      else {
        if (original.blobHash !== undefined) {
          const sha = String(original.blobHash);
          if (!HASH.test(sha) || !sharedBlobSource) throw new Error("A retained shared source blob is unavailable.");
          const destination = join(input.targetDirectory, "merge", "shared-blobs", sha);
          await mkdir(join(input.targetDirectory, "merge", "shared-blobs"), { recursive: true });
          const present = await lstat(destination).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
          if (!present) {
            const source = sharedBlobSource(sha);
            if (await fullMergeFileHash(source, input.signal) !== sha) throw new Error("A retained shared source blob changed.");
            await input.reserve((await lstat(source)).size, "complete retained shared source blob");
            await copyFile(source, destination);
          }
          if (await fullMergeFileHash(destination, input.signal) !== sha) throw new Error("A merged shared source blob failed verification.");
        }
        activity.upsertTrainingSources([value as unknown as TrainingSource]);
      }
    }
    return conversation.integrity();
  } finally { nativeIndex?.close(); artifacts.close(); activity.close(); conversation.close(); db.close(); }
}
