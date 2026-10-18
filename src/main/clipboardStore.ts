import { randomUUID } from "node:crypto";
import { constants } from "node:fs";
import { access, chmod, lstat, mkdir, open, opendir, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { CLIPBOARD_FORMAT, type ClipboardMutationAction, type ClipboardMutationReceipt, type ClipboardProvenance, type ClipboardVersionSummary } from "../shared/clipboard";
import { clipboardCanonical, clipboardHash, clipboardId, clipboardMutationReceipt, clipboardProvenance, clipboardSha } from "./clipboardFormat";
import { writeMutableFileIsolated } from "./mutableFileIsolation";

export const CLIPBOARD_WORKSPACE_PATH = "clipboard/workspace.sqlite3";
const WORKSPACE_FORMAT = "omni-clipboard-workspace-1";
const EMPTY = Buffer.alloc(0), EMPTY_SHA = clipboardHash(EMPTY), INITIAL_VERSION = "0".repeat(32);
const schema = new Map(Object.entries({
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  contents: "CREATE TABLE contents(sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL)",
  versions: "CREATE TABLE versions(id TEXT PRIMARY KEY,revision INTEGER NOT NULL UNIQUE,action TEXT NOT NULL,sha256 TEXT NOT NULL,created_at TEXT NOT NULL,provenance TEXT NOT NULL)",
  current_clipboard: "CREATE TABLE current_clipboard(singleton INTEGER PRIMARY KEY,version_id TEXT NOT NULL,revision INTEGER NOT NULL)",
  effects: "CREATE TABLE effects(request_id TEXT PRIMARY KEY,scope TEXT NOT NULL,action TEXT NOT NULL,arguments_sha256 TEXT NOT NULL,phase TEXT NOT NULL,payload TEXT NOT NULL,receipt TEXT NOT NULL)",
  version_page: "CREATE INDEX version_page ON versions(id)"
}));
const schemaSql = (value: unknown): string => String(value).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/\s+|;$/g, "").toUpperCase();

interface EffectPayload { scope: "instance" | "host"; action: ClipboardMutationAction; requestSha256: string; contentSha256: string; contentBytes: number }
export interface ClipboardEffectRecord extends EffectPayload { requestId: string; phase: "prepared" | "dispatched" | "terminal"; receipt?: ClipboardMutationReceipt }

async function privateDirectory(path: string): Promise<void> {
  await mkdir(path, { recursive: true, mode: 0o700 }); let info = await lstat(path);
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Clipboard workspace requires a private real directory.");
  if (process.platform !== "win32") {
    const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
    if (info.uid !== uid) throw new Error("Clipboard workspace has another operating-system owner.");
    if ((info.mode & 0o077) !== 0) { await chmod(path, 0o700); info = await lstat(path); }
    if (info.uid !== uid || (info.mode & 0o077) !== 0) throw new Error("Clipboard workspace is not private to its owner.");
  }
}
async function existingDirectory(path: string, label: string): Promise<void> {
  const info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error(`${label} must be an owned real directory.`);
  if (process.platform !== "win32") { const uid = typeof process.getuid === "function" ? process.getuid() : info.uid;
    if (info.uid !== uid) throw new Error(`${label} has another operating-system owner.`); }
}
function timestamp(value: unknown): string { if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error("Invalid clipboard timestamp."); return value; }
function effectPayload(value: unknown): EffectPayload {
  const row = typeof value === "string" ? JSON.parse(value) : value as EffectPayload;
  if (!row || typeof row !== "object" || !["instance", "host"].includes(String(row.scope)) || !["write", "clear"].includes(String(row.action)) ||
      !Number.isSafeInteger(row.contentBytes) || row.contentBytes < 0) throw new Error("Invalid clipboard effect payload.");
  clipboardSha(row.requestSha256); clipboardSha(row.contentSha256);
  return { scope: row.scope, action: row.action, requestSha256: row.requestSha256, contentSha256: row.contentSha256, contentBytes: row.contentBytes };
}
function utf8Page(bytes: Buffer, maximum: number, complete: boolean): { bytes: Buffer; text: string } {
  let length = complete ? bytes.length : Math.min(maximum, bytes.length);
  for (let drop = 0; drop <= 4 && length - drop >= 0; drop++) {
    const page = bytes.subarray(0, length - drop);
    try { const text = new TextDecoder("utf-8", { fatal: true }).decode(page); return { bytes: page, text }; } catch { /* Find the prior UTF-8 boundary. */ }
  }
  throw new Error("Clipboard page did not contain valid UTF-8 text at a bounded boundary.");
}

export class ClipboardStore {
  private constructor(readonly root: string, brainId: string, readonly base: string, readonly db: DatabaseSync) { this.brainId = brainId; }
  brainId: string;
  static async open(root: string, brainId: string, readOnly = false): Promise<ClipboardStore> {
    clipboardId(brainId, "clipboard brain owner"); const base = join(root, "clipboard"), blobs = join(base, "blobs"), path = join(root, CLIPBOARD_WORKSPACE_PATH);
    if (!readOnly) { await privateDirectory(base); await privateDirectory(blobs); }
    else { await existingDirectory(base, "Clipboard workspace"); await existingDirectory(blobs, "Clipboard blob directory");
      const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Clipboard workspace database is unsafe."); }
    const db = new DatabaseSync(path, { readOnly, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
        if (row.type === "index" && /^sqlite_autoindex_[A-Za-z0-9_]+_\d+$/.test(String(row.name)) && row.sql === null) continue;
        const declaration = schema.get(String(row.name));
        if (!declaration || row.type !== (row.name === "version_page" ? "index" : "table") || schemaSql(row.sql) !== schemaSql(declaration)) throw new Error("Clipboard workspace contains unsupported executable/schema objects.");
      }
      if (!readOnly) {
        db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS contents(sha256 TEXT PRIMARY KEY,bytes INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY,revision INTEGER NOT NULL UNIQUE,action TEXT NOT NULL,sha256 TEXT NOT NULL,created_at TEXT NOT NULL,provenance TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS version_page ON versions(id);
          CREATE TABLE IF NOT EXISTS current_clipboard(singleton INTEGER PRIMARY KEY,version_id TEXT NOT NULL,revision INTEGER NOT NULL);
          CREATE TABLE IF NOT EXISTS effects(request_id TEXT PRIMARY KEY,scope TEXT NOT NULL,action TEXT NOT NULL,arguments_sha256 TEXT NOT NULL,phase TEXT NOT NULL,payload TEXT NOT NULL,receipt TEXT NOT NULL);`);
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(WORKSPACE_FORMAT); db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
        if (!db.prepare("SELECT 1 AS n FROM current_clipboard WHERE singleton=1").get()) {
          await writeMutableFileIsolated(join(blobs, `${EMPTY_SHA}.data`), EMPTY);
          const now = new Date(0).toISOString(), p: ClipboardProvenance = { kind: "initial", capturedAt: now, brainId };
          db.exec("BEGIN IMMEDIATE"); try {
            db.prepare("INSERT OR IGNORE INTO contents VALUES(?,0)").run(EMPTY_SHA);
            db.prepare("INSERT INTO versions VALUES(?,0,'initial',?,?,?)").run(INITIAL_VERSION, EMPTY_SHA, now, clipboardCanonical(p));
            db.prepare("INSERT INTO current_clipboard VALUES(1,?,0)").run(INITIAL_VERSION); db.exec("COMMIT");
          } catch (error) { db.exec("ROLLBACK"); throw error; }
        }
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== WORKSPACE_FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Clipboard workspace has another format or owner.");
      return new ClipboardStore(root, brainId, base, db);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  blobPath(sha: string): string { return join(this.base, "blobs", `${clipboardSha(sha)}.data`); }
  private currentRow(): { version_id: string; revision: number } {
    if (Number(this.db.prepare("SELECT COUNT(*) AS n FROM current_clipboard").get()?.n ?? 0) !== 1) throw new Error("Clipboard workspace must have exactly one current pointer.");
    const row = this.db.prepare("SELECT version_id,revision FROM current_clipboard WHERE singleton=1").get();
    if (!row || typeof row.version_id !== "string" || !Number.isSafeInteger(row.revision) || Number(row.revision) < 0) throw new Error("Clipboard current pointer is invalid.");
    return row as { version_id: string; revision: number };
  }
  version(id?: string): ClipboardVersionSummary {
    const current = id === undefined ? this.currentRow() : undefined, wanted = id ?? current!.version_id, row = this.db.prepare("SELECT id,revision,action,sha256,created_at,provenance FROM versions WHERE id=?").get(clipboardId(wanted, "clipboard version ID"));
    if (!row || !Number.isSafeInteger(row.revision) || Number(row.revision) < 0 || !["initial", "write", "clear"].includes(String(row.action)) ||
        typeof row.provenance !== "string" || Buffer.byteLength(row.provenance) > 8192) throw new Error("Clipboard version was not found or is invalid.");
    const content = this.db.prepare("SELECT bytes FROM contents WHERE sha256=?").get(clipboardSha(row.sha256));
    if (!content || !Number.isSafeInteger(content.bytes) || Number(content.bytes) < 0) throw new Error("Clipboard version lost its content descriptor.");
    if (current && current.revision !== Number(row.revision)) throw new Error("Clipboard current revision differs from its version.");
    return { id: String(row.id), revision: Number(row.revision), action: row.action as ClipboardVersionSummary["action"], contentSha256: String(row.sha256), bytes: Number(content.bytes), createdAt: timestamp(row.created_at), provenance: clipboardProvenance(JSON.parse(String(row.provenance))) };
  }
  versions(after = "", limit = 50): { entries: ClipboardVersionSummary[]; nextAfter?: string } {
    if (after) clipboardId(after, "clipboard version cursor"); if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Clipboard version page must contain 1 to 200 entries.");
    const rows = this.db.prepare("SELECT id FROM versions WHERE id>? ORDER BY id LIMIT ?").all(after, limit + 1), entries = rows.slice(0, limit).map(row => this.version(String(row.id)));
    return { entries, ...(rows.length > limit ? { nextAfter: entries.at(-1)!.id } : {}) };
  }
  async read(versionId: string | undefined, offset: number, maximum: number): Promise<{ version: ClipboardVersionSummary; text: string; bytes: number; nextOffset?: number; complete: boolean }> {
    const version = this.version(versionId); if (!Number.isSafeInteger(offset) || offset < 0 || offset > version.bytes) throw new Error("Clipboard cursor is outside the selected content.");
    const remaining = version.bytes - offset, wanted = Math.min(remaining, maximum + 4), file = await open(this.blobPath(version.contentSha256), constants.O_RDONLY);
    try {
      const bytes = Buffer.alloc(wanted), result = await file.read(bytes, 0, wanted, offset); if (result.bytesRead !== wanted) throw new Error("Clipboard content ended before its declared length.");
      const complete = remaining <= maximum, page = utf8Page(bytes, maximum, complete);
      return { version, text: page.text, bytes: page.bytes.length, complete, ...(!complete ? { nextOffset: offset + page.bytes.length } : {}) };
    } finally { await file.close(); }
  }
  effect(requestId: string): ClipboardEffectRecord | undefined {
    const row = this.db.prepare("SELECT request_id,scope,action,arguments_sha256,phase,payload,receipt FROM effects WHERE request_id=?").get(clipboardId(requestId, "clipboard request ID"));
    if (!row) return undefined;
    if (typeof row.payload !== "string" || Buffer.byteLength(row.payload) > 32768 || typeof row.receipt !== "string" || Buffer.byteLength(row.receipt) > 32768) throw new Error("Clipboard effect metadata exceeds its bounded envelope.");
    const payload = effectPayload(row.payload);
    if (payload.scope !== row.scope || payload.action !== row.action || payload.requestSha256 !== row.arguments_sha256 || !["prepared", "dispatched", "terminal"].includes(String(row.phase))) throw new Error("Clipboard effect SQL binding differs from its payload.");
    if ((row.phase === "terminal") !== Boolean(row.receipt)) throw new Error("Clipboard effect phase differs from its terminal receipt.");
    const receipt = row.receipt ? clipboardMutationReceipt(JSON.parse(String(row.receipt))) : undefined;
    if (receipt && (receipt.requestId !== row.request_id || receipt.scope !== payload.scope || receipt.action !== payload.action || receipt.requestSha256 !== payload.requestSha256 ||
        receipt.contentSha256 !== payload.contentSha256 || receipt.contentBytes !== payload.contentBytes)) throw new Error("Clipboard effect receipt differs from its exact request/content owner.");
    return { requestId: String(row.request_id), ...payload, phase: row.phase as ClipboardEffectRecord["phase"], ...(receipt ? { receipt } : {}) };
  }
  prepareHostEffect(requestId: string, payload: EffectPayload): ClipboardEffectRecord {
    clipboardId(requestId, "clipboard request ID"); effectPayload(payload); const prior = this.effect(requestId);
    if (prior) { if (clipboardCanonical(effectPayload(prior)) !== clipboardCanonical(payload)) throw new Error("Clipboard request ID was reused with different scope/content."); return prior; }
    this.db.prepare("INSERT INTO effects VALUES(?,'host',?,?,'prepared',?,'')")
      .run(requestId, payload.action, payload.requestSha256, clipboardCanonical(payload));
    return this.effect(requestId)!;
  }
  dispatched(requestId: string): void {
    if (this.db.prepare("UPDATE effects SET phase='dispatched' WHERE request_id=? AND phase='prepared'").run(clipboardId(requestId)).changes !== 1) throw new Error("Clipboard mutation is no longer prepared for dispatch.");
  }
  terminal(requestId: string, receipt: ClipboardMutationReceipt): void {
    const checked = clipboardMutationReceipt(receipt), text = clipboardCanonical(checked); if (text.length > 32768 || checked.requestId !== requestId) throw new Error("Clipboard terminal receipt is invalid.");
    if (this.db.prepare("UPDATE effects SET phase='terminal',receipt=? WHERE request_id=? AND phase IN ('prepared','dispatched')").run(text, clipboardId(requestId)).changes !== 1) throw new Error("Clipboard mutation lost its effect-journal owner.");
  }
  recoverInterruptedHostEffects(reason = "Studio ended before the host clipboard mutation reached a terminal acknowledgement.", parentBrainId?: string): number {
    let count = 0; const now = new Date().toISOString();
    for (const row of this.db.prepare("SELECT request_id,phase,payload FROM effects WHERE scope='host' AND phase!='terminal'").all()) {
      const payload = effectPayload(row.payload), uncertain = row.phase === "dispatched";
      const receipt: ClipboardMutationReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: payload.action, requestId: String(row.request_id), requestSha256: payload.requestSha256,
        scope: "host", state: uncertain ? "write-uncertain" : "cancelled", contentType: "text/plain", contentBytes: payload.contentBytes,
        contentSha256: payload.contentSha256, writeMayHaveOccurred: uncertain, retrySafe: false,
        provenance: parentBrainId ? { kind: "inherited", capturedAt: now, brainId: this.brainId, parentBrainId }
          : { kind: "host-os-clipboard", capturedAt: now, brainId: this.brainId },
        error: uncertain ? `${reason} The effect may have occurred and this exact request will not be replayed.` : `${reason} Dispatch was not recorded and this exact request remains nonreplayable.` };
      this.terminal(String(row.request_id), receipt); count++;
    }
    return count;
  }
  async removeUnownedBlobs(): Promise<number> {
    const owned = new Set([...this.db.prepare("SELECT sha256 FROM contents").iterate()].map(row => `${clipboardSha(row.sha256)}.data`));
    let removed = 0; const directory = await opendir(join(this.base, "blobs")); try { for await (const entry of directory) {
      if (!entry.isFile() || entry.isSymbolicLink()) throw new Error("Clipboard blob directory contains an unsafe entry.");
      if (!owned.has(entry.name)) { await rm(join(this.base, "blobs", entry.name), { force: true }); removed++; }
    } } finally { await directory.close().catch(() => undefined); } return removed;
  }
  async mutateInstance(requestId: string, payload: EffectPayload, text: string, provenanceValue: ClipboardProvenance): Promise<ClipboardMutationReceipt> {
    if (payload.scope !== "instance") throw new Error("Instance clipboard mutation received a host payload.");
    const prior = this.effect(requestId); if (prior) {
      if (clipboardCanonical(effectPayload(prior)) !== clipboardCanonical(payload)) throw new Error("Clipboard request ID was reused with different content.");
      if (!prior.receipt) throw new Error("An earlier instance clipboard mutation has no terminal receipt and was not repeated.");
      return { ...prior.receipt, repeated: true };
    }
    const body = Buffer.from(text, "utf8"); if (body.byteLength !== payload.contentBytes || clipboardHash(body) !== payload.contentSha256) throw new Error("Instance clipboard bytes differ from their request checksum.");
    const contentAlreadyOwned = Boolean(this.db.prepare("SELECT 1 AS n FROM contents WHERE sha256=?").get(payload.contentSha256));
    await writeMutableFileIsolated(this.blobPath(payload.contentSha256), body);
    const revision = this.nextRevision(), versionId = randomUUID().replaceAll("-", ""), createdAt = new Date().toISOString();
    const receipt: ClipboardMutationReceipt = { format: CLIPBOARD_FORMAT, version: 1, action: payload.action, requestId, requestSha256: payload.requestSha256,
      scope: "instance", state: "complete", contentType: "text/plain", contentBytes: body.byteLength, contentSha256: payload.contentSha256,
      writeMayHaveOccurred: true, retrySafe: false, versionId, revision, provenance: provenanceValue };
    this.db.exec("BEGIN IMMEDIATE"); try {
      this.db.prepare("INSERT OR IGNORE INTO contents VALUES(?,?)").run(payload.contentSha256, body.byteLength);
      this.db.prepare("INSERT INTO versions VALUES(?,?,?,?,?,?)").run(versionId, revision, payload.action, payload.contentSha256, createdAt, clipboardCanonical(provenanceValue));
      this.db.prepare("UPDATE current_clipboard SET version_id=?,revision=? WHERE singleton=1").run(versionId, revision);
      this.db.prepare("INSERT INTO effects VALUES(?,'instance',?,?,'terminal',?,?)").run(requestId, payload.action, payload.requestSha256, clipboardCanonical(payload), clipboardCanonical(receipt));
      this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); if (!contentAlreadyOwned) await rm(this.blobPath(payload.contentSha256), { force: true }).catch(() => undefined); throw error; }
    return receipt;
  }
  allEffects(): ClipboardEffectRecord[] { return this.db.prepare("SELECT request_id FROM effects ORDER BY request_id").all().map(row => this.effect(String(row.request_id))!); }
  nextRevision(): number { const value = Number(this.db.prepare("SELECT COALESCE(MAX(revision),0) AS n FROM versions").get()?.n ?? 0); if (!Number.isSafeInteger(value) || value < 0 || value >= Number.MAX_SAFE_INTEGER) throw new Error("Clipboard revision space is exhausted."); return value + 1; }
  async validate(): Promise<void> {
    if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Clipboard workspace failed SQLite integrity validation.");
    const expected = new Set<string>();
    for (const row of this.db.prepare("SELECT sha256,bytes FROM contents").iterate()) {
      const sha = clipboardSha(row.sha256), bytes = Number(row.bytes); if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Clipboard content descriptor is invalid.");
      const path = this.blobPath(sha), info = await lstat(path); if (!info.isFile() || info.isSymbolicLink() || info.size !== bytes) throw new Error("Clipboard content blob differs from its descriptor.");
      const hash = (await import("node:crypto")).createHash("sha256"), handle = await open(path, constants.O_RDONLY); try { for await (const chunk of handle.createReadStream({ autoClose: false })) hash.update(chunk); } finally { await handle.close(); }
      if (hash.digest("hex") !== sha) throw new Error("Clipboard content blob checksum failed."); expected.add(`${sha}.data`);
    }
    const directory = await opendir(join(this.base, "blobs")); try { for await (const entry of directory) if (!entry.isFile() || !expected.has(entry.name)) throw new Error("Clipboard workspace contains an unowned blob."); } finally { await directory.close().catch(() => undefined); }
    this.version(); for (const row of this.db.prepare("SELECT id FROM versions").iterate()) this.version(String(row.id));
    for (const row of this.db.prepare("SELECT request_id FROM effects").iterate()) this.effect(String(row.request_id));
  }
  rekey(targetBrainId: string): void {
    clipboardId(targetBrainId, "target clipboard owner"); const sourceBrainId = this.brainId;
    this.db.exec("BEGIN IMMEDIATE"); try {
      this.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      this.db.exec("COMMIT"); this.brainId = targetBrainId;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
    this.recoverInterruptedHostEffects("The source clipboard workspace changed owners before this mutation reached a terminal acknowledgement.", sourceBrainId);
  }
  importVersion(summary: ClipboardVersionSummary, sourcePath: string, newId: string, revision: number, provenanceValue: ClipboardProvenance): void {
    clipboardId(newId, "merged clipboard version ID"); clipboardSha(summary.contentSha256); clipboardProvenance(provenanceValue);
    this.db.prepare("INSERT OR IGNORE INTO contents VALUES(?,?)").run(summary.contentSha256, summary.bytes);
    this.db.prepare("INSERT OR IGNORE INTO versions VALUES(?,?,?,?,?,?)").run(newId, revision, summary.action, summary.contentSha256, summary.createdAt, clipboardCanonical(provenanceValue));
    void sourcePath;
  }
  importEffect(source: ClipboardEffectRecord, requestId: string, versionAliases: ReadonlyMap<string, string>, parentBrainId: string): void {
    clipboardId(requestId, "merged clipboard request ID"); clipboardId(parentBrainId, "clipboard parent owner");
    const baseReceipt = source.receipt, capturedAt = baseReceipt?.provenance.capturedAt ?? new Date(0).toISOString();
    const aliasedVersion = baseReceipt?.versionId ? versionAliases.get(baseReceipt.versionId) ?? baseReceipt.versionId : undefined;
    const receipt: ClipboardMutationReceipt = baseReceipt ? { ...baseReceipt, requestId,
      ...(aliasedVersion ? { versionId: aliasedVersion, revision: this.version(aliasedVersion).revision } : {}),
      provenance: { kind: "inherited", capturedAt, brainId: this.brainId, parentBrainId } }
      : { format: CLIPBOARD_FORMAT, version: 1, action: source.action, requestId, requestSha256: source.requestSha256,
        scope: source.scope, state: source.phase === "dispatched" ? "write-uncertain" : "cancelled", contentType: "text/plain",
        contentBytes: source.contentBytes, contentSha256: source.contentSha256, writeMayHaveOccurred: source.phase === "dispatched", retrySafe: false,
        provenance: { kind: "inherited", capturedAt, brainId: this.brainId, parentBrainId },
        error: "The parent clipboard mutation had no terminal acknowledgement. It was retained as nonreplayable history." };
    const payload = { scope: source.scope, action: source.action, requestSha256: source.requestSha256, contentSha256: source.contentSha256, contentBytes: source.contentBytes };
    this.db.prepare("INSERT OR IGNORE INTO effects VALUES(?,?,?,?, 'terminal',?,?)")
      .run(requestId, source.scope, source.action, source.requestSha256, clipboardCanonical(payload), clipboardCanonical(receipt));
    const imported = this.effect(requestId); if (!imported?.receipt || imported.phase !== "terminal" || clipboardCanonical(imported.receipt) !== clipboardCanonical(receipt)) throw new Error("Merged clipboard effect conflicts with retained history.");
  }
  setCurrent(versionId: string, revision: number): void { const version = this.version(clipboardId(versionId)); if (version.revision !== revision) throw new Error("Clipboard current pointer revision differs from its version.");
    this.db.prepare("UPDATE current_clipboard SET version_id=?,revision=? WHERE singleton=1").run(version.id, revision); }
  remove(): Promise<void> { this.close(); return rm(this.base, { recursive: true, force: true }); }
}
