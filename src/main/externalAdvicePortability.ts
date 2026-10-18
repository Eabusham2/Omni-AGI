import { randomUUID } from "node:crypto";
import { lstat, mkdir, opendir, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ExternalAdviceRequest } from "../shared/externalAdvice";
import {
  EXTERNAL_ADVICE_RELATIVE_PATH, externalAdviceCanonical, externalAdviceInheritedRequestId, externalAdviceInheritedReviewId,
  externalAdviceRequestId, externalAdviceSha256, validateExternalAdviceDatabaseFile, validateExternalAdviceRequest, validateExternalAdviceWorkspace
} from "./externalAdviceService";
import { copyMutableFileIsolated, snapshotMutableSqliteIsolated } from "./mutableFileIsolation";

export interface ExternalAdvicePortabilityHooks {
  signal?: AbortSignal;
  checkDisk?(bytes: number): Promise<void>;
}

type SqlRow = Record<string, unknown>;
type Original = { request: ExternalAdviceRequest; payload: string; sha256: string };
const SHA = /^[a-f0-9]{64}$/;

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

export function safeExternalAdviceWorkspacePath(value: string): boolean {
  return value === EXTERNAL_ADVICE_RELATIVE_PATH;
}

/** Reuse backend validation on the committed online-backup/CAS transaction. */
export async function validateCapturedExternalAdviceWorkspace(path: string, owner: string): Promise<void> {
  await validateExternalAdviceDatabaseFile(path, owner);
}

/** No install-private credentials, grants, controllers or unlisted sidecars. */
export async function validateExternalAdvicePortabilityWorkspace(root: string, owner: string): Promise<void> {
  externalAdviceRequestId(owner);
  const directory = join(root, "external-advice"), database = join(root, EXTERNAL_ADVICE_RELATIVE_PATH);
  if (!await exists(directory)) return;
  const info = await lstat(directory);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("External-advice custody needs an owned directory.");
  const entries = await opendir(directory);
  for await (const entry of entries) {
    if (!["workspace.sqlite3", "workspace.sqlite3-wal", "workspace.sqlite3-shm", "workspace.sqlite3-journal"].includes(entry.name) ||
        !entry.isFile() || entry.isSymbolicLink()) throw new Error("External-advice custody contains an unlisted or unsafe operational payload.");
  }
  if (!await exists(database)) throw new Error("External-advice custody is missing its ownership database.");
  await validateExternalAdviceWorkspace(root, owner);
}

function openDatabase(path: string, readOnly = false): DatabaseSync {
  const database = new DatabaseSync(path, { readOnly, allowExtension: false });
  database.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;");
  if (!readOnly) database.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;");
  return database;
}

function checkedPayload(payload: unknown, digest: unknown): string {
  if (typeof payload !== "string" || typeof digest !== "string" || !SHA.test(digest) || externalAdviceSha256(payload) !== digest)
    throw new Error("External-advice portable source hash changed.");
  const parsed: unknown = JSON.parse(payload);
  if (externalAdviceCanonical(parsed) !== payload) throw new Error("External-advice portable source is not canonical.");
  return payload;
}

function originalRequest(database: DatabaseSync, row: SqlRow, owner: string): Original {
  const payload = checkedPayload(row.payload, row.sha256), value = JSON.parse(payload) as ExternalAdviceRequest;
  let inherited: string | undefined;
  if (value.inheritedFrom) {
    const archive = database.prepare("SELECT payload FROM inherited_requests WHERE source_brain_id=? AND source_request_id=? AND source_sha256=?")
      .get(value.inheritedFrom.brainId, value.inheritedFrom.requestId, value.inheritedFrom.sha256);
    if (!archive) throw new Error("Inherited external-advice source is missing.");
    inherited = checkedPayload(archive.payload, value.inheritedFrom.sha256);
  }
  validateExternalAdviceRequest(value, owner, inherited);
  const source = inherited ?? payload, request = JSON.parse(source) as ExternalAdviceRequest;
  validateExternalAdviceRequest(request, request.brainId);
  if (request.inheritedFrom) throw new Error("External-advice ancestry must resolve to an exact original receipt.");
  return { request, payload: source, sha256: externalAdviceSha256(source) };
}

function preserveOriginal(database: DatabaseSync, original: Original): void {
  const { request, payload, sha256 } = original;
  const prior = database.prepare("SELECT payload FROM inherited_requests WHERE source_brain_id=? AND source_request_id=? AND source_sha256=?")
    .get(request.brainId, request.id, sha256);
  if (prior && prior.payload !== payload) throw new Error("External-advice immutable request ancestry collided.");
  database.prepare("INSERT OR IGNORE INTO inherited_requests VALUES(?,?,?,?)").run(request.brainId, request.id, sha256, payload);
}

function projection(original: Original, owner: string): ExternalAdviceRequest {
  const source = original.request;
  const value: ExternalAdviceRequest = { ...source, brainId: owner,
    id: externalAdviceInheritedRequestId(source.brainId, source.id, original.sha256, owner),
    state: source.state === "queued" || source.state === "streaming" ? "cancelled" : source.state,
    inheritedFrom: { brainId: source.brainId, requestId: source.id, sha256: original.sha256 } };
  return validateExternalAdviceRequest(value, owner, original.payload);
}

function insertProjection(database: DatabaseSync, value: ExternalAdviceRequest): void {
  const payload = externalAdviceCanonical(value), digest = externalAdviceSha256(payload);
  const prior = database.prepare("SELECT payload,sha256 FROM requests WHERE id=?").get(value.id);
  if (prior) {
    if (prior.payload !== payload || prior.sha256 !== digest) throw new Error("External-advice history identity collided with different bytes.");
    return;
  }
  database.prepare("INSERT INTO requests VALUES(?,?,?,?,?,?,?)")
    .run(value.id, value.channel, value.state, value.createdAt, value.updatedAt, payload, digest);
}

function insertReview(database: DatabaseSync, row: SqlRow, owner: string): string {
  const payload = checkedPayload(row.payload, row.sha256), sourceOwner = String(row.source_brain_id), reviewId = String(row.review_id);
  const id = owner === sourceOwner ? reviewId : externalAdviceInheritedReviewId(sourceOwner, reviewId, String(row.sha256), owner);
  const prior = database.prepare("SELECT owner_brain_id,source_brain_id,review_id,finished_at,payload,sha256 FROM reviews WHERE id=?").get(id);
  if (prior) {
    if (prior.owner_brain_id !== owner || prior.source_brain_id !== sourceOwner || prior.review_id !== reviewId ||
        prior.finished_at !== row.finished_at || prior.payload !== payload || prior.sha256 !== row.sha256)
      throw new Error("External-advice immutable parsed review ancestry collided.");
  } else database.prepare("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)")
    .run(id, owner, sourceOwner, reviewId, String(row.finished_at), payload, String(row.sha256));
  return id;
}

function archiveSettings(database: DatabaseSync, source: DatabaseSync, sourceOwner: string): void {
  for (const row of source.prepare("SELECT channel,payload,sha256 FROM settings ORDER BY channel").iterate()) {
    const payload = checkedPayload(row.payload, row.sha256);
    const prior = database.prepare("SELECT payload FROM inherited_settings WHERE source_brain_id=? AND channel=? AND source_sha256=?")
      .get(sourceOwner, String(row.channel), String(row.sha256));
    if (prior && prior.payload !== payload) throw new Error("External-advice immutable settings ancestry collided.");
    database.prepare("INSERT OR IGNORE INTO inherited_settings VALUES(?,?,?,?)").run(sourceOwner, String(row.channel), String(row.sha256), payload);
  }
  const metadata: Record<string, string | number> = {};
  for (const row of source.prepare("SELECT key,value FROM meta ORDER BY key").iterate()) metadata[String(row.key)] = String(row.value);
  const revision = Number(metadata.settingsRevision);
  if (!Number.isSafeInteger(revision) || revision < 1 || String(revision) !== metadata.settingsRevision)
    throw new Error("External-advice source settings revision is not canonical.");
  metadata.settingsRevision = revision;
  const payload = externalAdviceCanonical(metadata), digest = externalAdviceSha256(payload);
  const prior = database.prepare("SELECT payload FROM inherited_meta WHERE source_brain_id=? AND source_sha256=?").get(sourceOwner, digest);
  if (prior && prior.payload !== payload) throw new Error("External-advice immutable settings metadata collided.");
  database.prepare("INSERT OR IGNORE INTO inherited_meta VALUES(?,?,?)").run(sourceOwner, digest, payload);
}

async function requirePrivateWritable(path: string, hooks: ExternalAdvicePortabilityHooks): Promise<void> {
  hooks.signal?.throwIfAborted();
  const info = await lstat(path), estimate = info.size * 2 + 65536;
  if (!info.isFile() || info.isSymbolicLink() || info.nlink !== 1 || !Number.isSafeInteger(estimate))
    throw new Error("External-advice projection must use an independent private mutable database.");
  await hooks.checkDisk?.(estimate); hooks.signal?.throwIfAborted();
}

/** Current owner changes; exact source receipt/settings bytes remain archived. */
export async function rekeyExternalAdviceWorkspace(root: string, sourceOwner: string, targetOwner: string,
  hooks: ExternalAdvicePortabilityHooks = {}): Promise<void> {
  externalAdviceRequestId(sourceOwner); externalAdviceRequestId(targetOwner);
  const path = join(root, EXTERNAL_ADVICE_RELATIVE_PATH);
  if (!await exists(path)) { await validateExternalAdvicePortabilityWorkspace(root, sourceOwner); return; }
  let currentOwner = sourceOwner;
  try { await validateExternalAdvicePortabilityWorkspace(root, sourceOwner); }
  catch (error) { if (sourceOwner === targetOwner) throw error;
    try { await validateExternalAdvicePortabilityWorkspace(root, targetOwner); currentOwner = targetOwner; } catch { throw error; } }
  await requirePrivateWritable(path, hooks);
  const database = openDatabase(path);
  try {
    database.exec("BEGIN IMMEDIATE");
    archiveSettings(database, database, currentOwner);
    // Read the committed original rowid interval. New aliases append beyond
    // the captured head; mutation never grows an in-memory history array.
    const head = Number(database.prepare("SELECT COALESCE(MAX(rowid),0) AS n FROM requests").get()?.n ?? 0);
    let after = 0;
    while (true) {
      hooks.signal?.throwIfAborted();
      const row = database.prepare("SELECT rowid AS sequence,id,payload,sha256 FROM requests WHERE rowid>? AND rowid<=? ORDER BY rowid LIMIT 1").get(after, head);
      if (!row) break;
      after = Number(row.sequence);
      const original = originalRequest(database, row, currentOwner), value = projection(original, targetOwner);
      preserveOriginal(database, original);
      insertProjection(database, value);
      if (row.id !== value.id) database.prepare("DELETE FROM requests WHERE id=?").run(String(row.id));
    }
    const reviewHead = Number(database.prepare("SELECT COALESCE(MAX(rowid),0) AS n FROM reviews").get()?.n ?? 0);
    let afterReview = 0;
    while (true) {
      hooks.signal?.throwIfAborted();
      const row = database.prepare("SELECT rowid AS sequence,* FROM reviews WHERE rowid>? AND rowid<=? ORDER BY rowid LIMIT 1").get(afterReview, reviewHead);
      if (!row) break;
      afterReview = Number(row.sequence);
      const id = insertReview(database, row, targetOwner);
      if (row.id !== id) database.prepare("DELETE FROM reviews WHERE id=?").run(String(row.id));
    }
    database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetOwner);
    hooks.signal?.throwIfAborted(); database.exec("COMMIT");
  } catch (error) { try { database.exec("ROLLBACK"); } catch { /* A completed transaction is never replayed. */ } throw error; }
  finally { database.close(); }
  await validateExternalAdvicePortabilityWorkspace(root, targetOwner);
}

async function copyArchiveRows(source: DatabaseSync, target: DatabaseSync, hooks: ExternalAdvicePortabilityHooks): Promise<void> {
  for (const table of ["inherited_requests", "inherited_settings", "inherited_meta"] as const) {
    for (const row of source.prepare(`SELECT * FROM ${table}`).iterate()) {
      hooks.signal?.throwIfAborted();
      const payload = checkedPayload(row.payload, row.source_sha256);
      await hooks.checkDisk?.(Buffer.byteLength(payload) + 4096);
      const keys = table === "inherited_requests" ? ["source_brain_id", "source_request_id", "source_sha256"]
        : table === "inherited_settings" ? ["source_brain_id", "channel", "source_sha256"] : ["source_brain_id", "source_sha256"];
      const values = keys.map(key => String(row[key]));
      const prior = target.prepare(`SELECT payload FROM ${table} WHERE ${keys.map(key => `${key}=?`).join(" AND ")}`).get(...values);
      if (prior && prior.payload !== payload) throw new Error("External-advice inherited source identity collided.");
      target.prepare(`INSERT OR IGNORE INTO ${table} VALUES(${[...values, payload].map(() => "?").join(",")})`).run(...values, payload);
    }
  }
}

/** Same checked ancestry/bytes deduplicate; distinct responses never do. */
export async function mergeExternalAdviceWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentBrainId: string,
  reserve: (bytes: number) => Promise<void>, signal?: AbortSignal): Promise<void> {
  externalAdviceRequestId(owner); externalAdviceRequestId(parentBrainId);
  if (resolve(sourceRoot) === resolve(targetRoot)) throw new Error("External-advice merge cannot mutate its source parent.");
  await validateExternalAdvicePortabilityWorkspace(sourceRoot, owner);
  const sourcePath = join(sourceRoot, EXTERNAL_ADVICE_RELATIVE_PATH);
  if (!await exists(sourcePath)) return;
  signal?.throwIfAborted(); await mkdir(targetRoot, { recursive: true });
  const temporaryRoot = join(targetRoot, `.external-advice-parent-${randomUUID()}`), temporary = join(temporaryRoot, EXTERNAL_ADVICE_RELATIVE_PATH);
  const targetPath = join(targetRoot, EXTERNAL_ADVICE_RELATIVE_PATH), hooks = { signal, checkDisk: reserve };
  try {
    await reserve((await lstat(sourcePath)).size + 65536);
    await snapshotMutableSqliteIsolated(sourcePath, temporary);
    await validateExternalAdvicePortabilityWorkspace(temporaryRoot, owner);
    if (!await exists(targetPath)) {
      await mkdir(dirname(targetPath), { recursive: true });
      await copyMutableFileIsolated(temporary, targetPath);
      await rekeyExternalAdviceWorkspace(targetRoot, owner, owner, hooks);
      return;
    }
    await validateExternalAdvicePortabilityWorkspace(targetRoot, owner);
    await requirePrivateWritable(targetPath, hooks);
    const source = openDatabase(temporary, true), target = openDatabase(targetPath);
    try {
      target.exec("BEGIN IMMEDIATE");
      await copyArchiveRows(source, target, hooks);
      // Current chosen settings remain exact; retain every other parent's
      // source settings/meta rather than merging config values or authority.
      archiveSettings(target, source, owner);
      for (const row of source.prepare("SELECT id,payload,sha256 FROM requests ORDER BY rowid").iterate()) {
        signal?.throwIfAborted(); const original = originalRequest(source, row, owner);
        await reserve(Buffer.byteLength(original.payload) * 2 + 8192);
        preserveOriginal(target, original); insertProjection(target, projection(original, owner));
      }
      for (const row of source.prepare("SELECT * FROM reviews ORDER BY rowid").iterate()) {
        signal?.throwIfAborted(); await reserve(Buffer.byteLength(String(row.payload)) + 4096);
        insertReview(target, row, owner);
      }
      signal?.throwIfAborted(); target.exec("COMMIT");
    } catch (error) { try { target.exec("ROLLBACK"); } catch { /* No partial projection is promoted. */ } throw error; }
    finally { source.close(); target.close(); }
    await validateExternalAdvicePortabilityWorkspace(targetRoot, owner);
  } finally { await rm(temporaryRoot, { recursive: true, force: true }); }
}
