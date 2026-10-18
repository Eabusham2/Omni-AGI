import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, mkdtemp, readdir, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { SteerDirectionRecord } from "../shared/steerDirections";
import { STEER_DIRECTIONS_PATH, SteerDirectionStore } from "./steerDirectionStore";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";

export function safeSteerDirectionWorkspacePath(value: string): boolean { return value === STEER_DIRECTIONS_PATH || /^directions\/archives\/[a-f0-9]{64}\.sqlite3$/.test(value); }
async function fileHash(path: string): Promise<string> { const hash = createHash("sha256"); for await (const chunk of createReadStream(path)) hash.update(chunk); return hash.digest("hex"); }
async function exists(path: string): Promise<boolean> { try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; } }
export async function validateSteerDirectionWorkspace(root: string, brainId: string): Promise<void> {
  const directory = join(root, "directions"); if (!await exists(directory)) return;
  const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe direction workspace directory.");
  for (const file of await readdir(directory, { withFileTypes: true })) {
    if (file.name === "archives" && file.isDirectory() && !file.isSymbolicLink()) {
      for (const archive of await readdir(join(directory, "archives"), { withFileTypes: true })) {
        const path = join(directory, "archives", archive.name);
        if (!archive.isFile() || archive.isSymbolicLink() || !/^[a-f0-9]{64}\.sqlite3$/.test(archive.name) || await fileHash(path) !== archive.name.slice(0, -8)) throw new Error("Immutable direction history archive changed.");
      }
    } else if (!file.isFile() || file.isSymbolicLink() || !/^workspace\.sqlite3(?:-(?:wal|shm|journal))?$/.test(file.name)) throw new Error("Direction workspace contains an unowned payload.");
  }
  const store = await SteerDirectionStore.open(root, brainId); if (!store) { if ((await readdir(directory)).length) throw new Error("Direction history is missing its owned registry."); return; }
  try { store.validate(); } finally { store.close(); }
}
/** Copy staging only. Immutable original input/receipt history is not rewritten. */
export async function rekeySteerDirectionWorkspace(root: string, sourceId: string, targetId: string): Promise<void> {
  await validateSteerDirectionWorkspace(root, sourceId); const reader = await SteerDirectionStore.open(root, sourceId); if (!reader) return; reader.close();
  const store = (await SteerDirectionStore.open(root, sourceId, true))!;
  try { store.rekey(targetId); } finally { store.close(); }
  await validateSteerDirectionWorkspace(root, targetId);
}
type Projection = Pick<SteerDirectionRecord, "turnId" | "predecessorTurnId" | "humanMessageId" | "humanTurnId" | "brainMessageId">;
/** A real host-history alias is distinct from immutable native receipt identity. */
export async function mergedSteerDirectionProjection(root: string, owner: string, parentId: string,
  source: SteerDirectionRecord, nativeTurns: ReadonlyMap<string, string>): Promise<Projection> {
  const historyPath = join(root, "merge", "host-history.sqlite3"), ledgerPath = join(root, "conversation", "ledger.sqlite3");
  const history = await exists(historyPath) ? new DatabaseSync(historyPath, { readOnly: true }) : undefined;
  const ledger = await exists(ledgerPath) ? new DatabaseSync(ledgerPath, { readOnly: true }) : undefined;
  const alias = (id: string | undefined): string | undefined => id === undefined ? undefined : history?.prepare("SELECT destination_id FROM identities WHERE parent_id=? AND source_id=?").get(parentId, id)?.destination_id as string | undefined;
  try {
    if (ledger && ledger.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== owner) throw new Error("Merged steering projection has the wrong history owner.");
    const humanMessageId = alias(source.humanMessageId), brainMessageId = alias(source.brainMessageId);
    let humanTurnId: string | undefined;
    if (humanMessageId) {
      const row = ledger?.prepare("SELECT * FROM entries WHERE entry_key=?").get(`message:${humanMessageId}`);
      const checksum = (text: string) => createHash("sha256").update(text).digest("hex");
      if (!row || row.kind !== "message" || checksum(String(row.payload_json)) !== row.payload_sha256 ||
          checksum(JSON.stringify({ sequence: row.sequence, entryKey: row.entry_key, kind: row.kind, createdAt: row.created_at,
            attentionEpoch: row.attention_epoch, payloadSha256: row.payload_sha256, previousSha256: row.previous_sha256 })) !== row.row_sha256) throw new Error("Merged steering human row is unavailable or changed.");
      const human = JSON.parse(String(row.payload_json));
      if (human.id !== humanMessageId || human.role !== "human" || human.content !== source.source.content) throw new Error("Merged steering does not match its actual human input.");
      humanTurnId = human.turnId ?? human.turn_id;
    }
    return { turnId: nativeTurns.get(source.turnId) ?? alias(source.turnId) ?? source.turnId,
      predecessorTurnId: nativeTurns.get(source.predecessorTurnId) ?? alias(source.predecessorTurnId) ?? source.predecessorTurnId,
      ...(humanMessageId ? { humanMessageId } : {}), ...(humanTurnId ? { humanTurnId } : {}), ...(brainMessageId ? { brainMessageId } : {}) };
  } finally { history?.close(); ledger?.close(); }
}
/** Historical union only. Never dispatches, resumes, learns, or grants access. */
export async function mergeSteerDirectionWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string,
  project: (record: SteerDirectionRecord) => Promise<Projection> = async record => ({ turnId: record.turnId,
    predecessorTurnId: record.predecessorTurnId, ...(record.humanMessageId ? { humanMessageId: record.humanMessageId } : {}),
    ...(record.humanTurnId ? { humanTurnId: record.humanTurnId } : {}), ...(record.brainMessageId ? { brainMessageId: record.brainMessageId } : {}) }),
  reason: "merged-history" | "recovered-history" = "merged-history", reserve?: (bytes: number) => Promise<void>): Promise<void> {
  await validateSteerDirectionWorkspace(sourceRoot, owner); const liveSource = await SteerDirectionStore.open(sourceRoot, owner); if (!liveSource) return; liveSource.close();
  const archiveRoot = join(targetRoot, "directions", "archives"); await mkdir(archiveRoot, { recursive: true, mode: 0o700 });
  const temporary = join(targetRoot, `.${randomUUID()}.direction-history.sqlite3`);
  const stage = await mkdtemp(join(targetRoot, ".direction-history-read-"));
  let source: SteerDirectionStore | undefined, target: SteerDirectionStore | undefined;
  try {
    const info = await lstat(join(sourceRoot, STEER_DIRECTIONS_PATH)), wal = await lstat(join(sourceRoot, `${STEER_DIRECTIONS_PATH}-wal`)).catch(() => undefined);
    await reserve?.(info.size + (wal?.size ?? 0) + 65536);
    await snapshotMutableSqliteIsolated(join(sourceRoot, STEER_DIRECTIONS_PATH), temporary);
    const digest = await fileHash(temporary), destination = join(archiveRoot, `${digest}.sqlite3`);
    if (!await exists(destination)) await rename(temporary, destination);
    // Project the SAME complete snapshot, not a later live history generation.
    await mkdir(join(stage, "directions")); await copyMutableFileIsolated(destination, join(stage, STEER_DIRECTIONS_PATH), { expectedSha256: digest });
    source = (await SteerDirectionStore.open(stage, owner))!; source.validate();
    const originals = join(sourceRoot, "directions", "archives"); if (await exists(originals)) for (const name of await readdir(originals)) {
      const path = join(originals, name); await reserve?.((await lstat(path)).size); await copyMutableFileIsolated(path, join(archiveRoot, name), { expectedSha256: name.slice(0, -8) });
    }
    await reserve?.(65536); target = (await SteerDirectionStore.open(targetRoot, owner, true))!;
    for (const row of source.db.prepare("SELECT id FROM directions ORDER BY sequence").iterate()) {
      const record = source.get(String(row.id))!;
      await reserve?.(Buffer.byteLength(JSON.stringify(record)) + 8192); target.adopt(record, await project(record), parentId, reason);
    } target.validate();
  } finally { source?.close(); target?.close(); await rm(temporary, { force: true }); await rm(stage, { recursive: true, force: true }); }
}
