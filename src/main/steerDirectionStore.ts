import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { SteerDirectionPage, SteerDirectionPageQuery, SteerDirectionRecord, SteerDirectionSource, SteerDirectionState } from "../shared/steerDirections";
import { checkedNativeAudioInfo } from "../shared/nativeAudioChat";
import { checkedChatInputProvenance } from "../shared/chatInputProvenance";
import { sqliteFilename } from "./sqliteFilename";

export const STEER_DIRECTIONS_PATH = "directions/workspace.sqlite3";
const FORMAT = "omni-steer-directions-1", SHA = /^[a-f0-9]{64}$/, ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/;
const STATES = new Set<SteerDirectionState>(["pending", "received", "complete", "failed", "stopped", "steered"]);
const TABLES = {
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  directions: "CREATE TABLE directions(sequence INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT NOT NULL UNIQUE,binding_sha TEXT NOT NULL,source_json TEXT NOT NULL,view_json TEXT NOT NULL,view_sha TEXT NOT NULL,turn_id TEXT NOT NULL,human_turn_id TEXT,dismissed INTEGER NOT NULL)",
  events: "CREATE TABLE events(sequence INTEGER PRIMARY KEY,id TEXT NOT NULL,revision INTEGER NOT NULL,payload_json TEXT NOT NULL,payload_sha TEXT NOT NULL,previous_sha TEXT NOT NULL,row_sha TEXT NOT NULL)"
};
const INDEXES = { directions_turn: "CREATE INDEX directions_turn ON directions(turn_id)", directions_human_turn: "CREATE INDEX directions_human_turn ON directions(human_turn_id)", events_direction: "CREATE INDEX events_direction ON events(id,sequence)" };
type View = Omit<SteerDirectionRecord, "source" | "sequence">;
export const directionHash = (value: unknown): string => createHash("sha256").update(directionCanonical(value)).digest("hex");
export function directionCanonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(directionCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => `${JSON.stringify(key)}:${directionCanonical(child)}`).join(",")}}`;
  const text = JSON.stringify(value); if (text === undefined) throw new Error("Direction history requires explicit JSON."); return text;
}
export function directionId(value: unknown): string { if (typeof value !== "string" || !ID.test(value)) throw new Error("Invalid direction identity."); return value; }
const date = (value: unknown): value is string => typeof value === "string" && Number.isFinite(Date.parse(value));
const object = (value: unknown): Record<string, unknown> => { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid direction record."); return value as Record<string, unknown>; };
function exact(value: Record<string, unknown>, allowed: string[]): void { if (Object.keys(value).some(key => !allowed.includes(key))) throw new Error("Direction record contains undeclared fields."); }
export function validateDirectionSource(value: unknown): SteerDirectionSource {
  const row = object(value); exact(row, ["brainId", "turnId", "predecessorTurnId", "createdAt", "content", "submittedContent", "attachmentIds", "inputKind", "audio"]);
  [row.brainId, row.turnId, row.predecessorTurnId].forEach(directionId);
  if (row.turnId === row.predecessorTurnId || !date(row.createdAt) || typeof row.content !== "string" ||
      row.submittedContent !== undefined && typeof row.submittedContent !== "string" || !Array.isArray(row.attachmentIds) ||
      new Set(row.attachmentIds).size !== row.attachmentIds.length || !["text", "native-audio"].includes(String(row.inputKind))) throw new Error("Direction source binding is invalid.");
  row.attachmentIds.forEach(directionId);
  if (row.inputKind === "native-audio") {
    const audio = object(row.audio); exact(audio, ["inputSha256", "attachmentId", "sourceSha256", "sourceBytes", "mimeType", "utteranceId", "sequence", "capturedAt"]);
    if (row.content !== "" || !SHA.test(String(audio.inputSha256)) || !SHA.test(String(audio.sourceSha256)) ||
        !Number.isSafeInteger(audio.sourceBytes) || Number(audio.sourceBytes) < 1 || !Number.isSafeInteger(audio.sequence) || Number(audio.sequence) < 0 ||
        !date(audio.capturedAt) || typeof audio.mimeType !== "string" || !/^audio\/[a-z0-9.+-]+$/.test(audio.mimeType)) throw new Error("Direction audio source is invalid.");
    directionId(audio.attachmentId); directionId(audio.utteranceId);
  } else if (!row.content || row.audio !== undefined) throw new Error("Text direction requires its actual applied content.");
  return row as unknown as SteerDirectionSource;
}
function validateView(value: unknown, source: SteerDirectionSource, owner: string): View {
  const row = object(value); exact(row, ["format", "version", "id", "brainId", "turnId", "predecessorTurnId", "bindingSha256", "state", "revision", "createdAt", "updatedAt", "dismissedAt", "inputReceipt", "acceptedAttachmentIds", "humanMessageId", "humanTurnId", "brainMessageId", "reply", "failure", "dormant", "provenance"]);
  if (row.format !== "omni-steer-direction" || row.version !== 1 || row.brainId !== owner || !SHA.test(String(row.id)) || row.bindingSha256 !== directionHash(source) ||
      !STATES.has(row.state as SteerDirectionState) || !Number.isSafeInteger(row.revision) || Number(row.revision) < 1 || !date(row.createdAt) || !date(row.updatedAt) ||
      row.dismissedAt !== undefined && !date(row.dismissedAt)) throw new Error("Direction presentation failed ownership/integrity validation.");
  directionId(row.turnId); directionId(row.predecessorTurnId);
  if (row.humanMessageId !== undefined) directionId(row.humanMessageId);
  if (row.humanTurnId !== undefined) directionId(row.humanTurnId);
  if (row.brainMessageId !== undefined) directionId(row.brainMessageId);
  if (row.inputReceipt !== undefined) {
    const receipt = object(row.inputReceipt);
    exact(receipt, ["committed", "turnId", "inputSha256", "humanMessageId", "afterimageId", "createdAt", "attentionEpoch", "slowJobId", "nativeAudioInput", "taskAttachmentIds", "inputProvenance"]);
    if (receipt.committed !== true || ![source.turnId, row.turnId].includes(receipt.turnId as string) || receipt.inputSha256 !== (source.audio?.inputSha256 ?? createHash("sha256").update(source.content, "utf8").digest("hex")) ||
        !date(receipt.createdAt) || !Number.isSafeInteger(receipt.attentionEpoch) || Number(receipt.attentionEpoch) < 0) throw new Error("Direction receipt differs from its immutable applied source.");
    directionId(receipt.humanMessageId); directionId(receipt.afterimageId);
    if (receipt.slowJobId !== undefined) directionId(receipt.slowJobId);
    if (receipt.inputProvenance !== undefined && (source.audio || checkedChatInputProvenance(receipt.inputProvenance).inputSha256 !== receipt.inputSha256)) throw new Error("Direction receipt lost its original input attribution.");
    if (source.audio) { const audio = checkedNativeAudioInfo(receipt.nativeAudioInput); if (audio.inputSha256 !== source.audio.inputSha256 || audio.attachmentId !== source.audio.attachmentId ||
        audio.sourceSha256 !== source.audio.sourceSha256 || audio.sourceBytes !== source.audio.sourceBytes || audio.sourceBrainId !== source.brainId || audio.sourceTurnId !== source.turnId ||
        audio.mimeType !== source.audio.mimeType || audio.utteranceId !== source.audio.utteranceId || audio.sequence !== source.audio.sequence || audio.capturedAt !== source.audio.capturedAt) throw new Error("Direction audio receipt changed its original capture."); }
    else if (receipt.nativeAudioInput !== undefined) throw new Error("A text direction cannot acquire an audio receipt.");
    if (receipt.taskAttachmentIds !== undefined) { if (!Array.isArray(receipt.taskAttachmentIds)) throw new Error("Invalid received attachment identities."); receipt.taskAttachmentIds.forEach(directionId); }
    if ((!row.humanMessageId && !row.dormant) || !Array.isArray(row.acceptedAttachmentIds)) throw new Error("Received direction lost its current human/attachment projection.");
    row.acceptedAttachmentIds.forEach(directionId);
    if (new Set(row.acceptedAttachmentIds).size !== row.acceptedAttachmentIds.length || source.attachmentIds.some(id => !(row.acceptedAttachmentIds as string[]).includes(id)) ||
        directionCanonical(receipt.taskAttachmentIds ?? []) !== directionCanonical(row.acceptedAttachmentIds) ||
        source.audio && !(row.acceptedAttachmentIds as string[]).includes(source.audio.attachmentId)) throw new Error("Received direction lost an applied attachment.");
  }
  if (row.reply !== undefined) {
    const reply = object(row.reply); exact(reply, ["humanMessageId", "brainMessageId", "createdAt", "generationEnd"]);
    directionId(reply.humanMessageId); directionId(reply.brainMessageId);
    if (!date(reply.createdAt) || reply.generationEnd !== undefined && !["steered", "native-stop", "no-reply"].includes(String(reply.generationEnd))) throw new Error("Invalid direction reply receipt.");
    if ((!row.humanMessageId || !row.brainMessageId) && !row.dormant) throw new Error("Completed direction lost its host projection.");
  }
  if (row.state === "received" && !row.inputReceipt || row.state === "complete" && !row.reply) throw new Error("Direction state lacks its actual receipt.");
  if (row.failure !== undefined && !["dispatch-failed", "stopped", "post-reply-work-failed"].includes(String(row.failure))) throw new Error("Invalid direction failure state.");
  if (row.dormant !== undefined) { const dormant = object(row.dormant); exact(dormant, ["reason", "fromBrainId", "at"]); directionId(dormant.fromBrainId);
    if (!["copied-instance", "merged-history", "recovered-history", "process-interrupted", "saved-continuation"].includes(String(dormant.reason)) || !date(dormant.at)) throw new Error("Invalid dormant direction custody."); }
  if (row.provenance !== undefined) { if (!Array.isArray(row.provenance)) throw new Error("Invalid direction provenance."); for (const item of row.provenance) { const original = object(item); exact(original, ["brainId", "id"]); directionId(original.brainId); if (!SHA.test(String(original.id))) throw new Error("Invalid direction ancestor."); } }
  return row as unknown as View;
}
async function exists(path: string): Promise<boolean> { try { await lstat(path); return true; } catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; } }
const normalizeSql = (value: unknown) => String(value).replace(/\s+/g, " ").trim().toLowerCase();
export class SteerDirectionStore {
  private constructor(readonly db: DatabaseSync, readonly brainId: string, private readonly maxReadBytes: () => number) {}
  static async open(root: string, brainId: string, create = false, maxReadBytes = () => Math.floor(Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 8)): Promise<SteerDirectionStore | undefined> {
    directionId(brainId); const path = join(root, STEER_DIRECTIONS_PATH), present = await exists(path); if (!present && !create) return undefined;
    const parent = join(root, "directions"); if (create) await mkdir(parent, { recursive: true, mode: 0o700 });
    for (const directory of [root, parent]) { const info = await lstat(directory); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Direction workspace requires owned directories."); }
    if (present) { const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Direction workspace requires an owned regular database."); }
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: !create });
    try {
      db.exec("PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (!present) {
        db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; BEGIN IMMEDIATE;");
        try { for (const sql of [...Object.values(TABLES), ...Object.values(INDEXES)]) db.exec(sql);
          db.prepare("INSERT INTO meta VALUES('format',?)").run(FORMAT); db.prepare("INSERT INTO meta VALUES('brainId',?)").run(brainId);
          db.prepare("INSERT INTO meta VALUES('eventHeadSequence','0')").run(); db.prepare("INSERT INTO meta VALUES('eventHeadSha256',?)").run("0".repeat(64)); db.exec("COMMIT");
        } catch (error) { db.exec("ROLLBACK"); throw error; }
      }
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").iterate()) {
        const expected = row.type === "table" ? TABLES[row.name as keyof typeof TABLES] : row.type === "index" ? INDEXES[row.name as keyof typeof INDEXES] : undefined;
        if (!expected || normalizeSql(row.sql) !== normalizeSql(expected)) throw new Error("Direction workspace contains an unsupported schema/trigger.");
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Direction workspace has another owner or format.");
      if (create) db.exec("PRAGMA synchronous=FULL;");
      return new SteerDirectionStore(db, brainId, maxReadBytes);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  transaction<T>(work: () => T): T { this.db.exec("BEGIN IMMEDIATE"); try { const value = work(); this.db.exec("COMMIT"); return value; } catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  private checked(row: Record<string, unknown>): SteerDirectionRecord {
    const bytes = Buffer.byteLength(String(row.source_json), "utf8") + Buffer.byteLength(String(row.view_json), "utf8");
    if (bytes > this.maxReadBytes()) throw new Error("Direction history awaits enough transfer RAM; its contents were not truncated.");
    const source = validateDirectionSource(JSON.parse(String(row.source_json))), view = validateView(JSON.parse(String(row.view_json)), source, this.brainId);
    if (row.id !== view.id || row.binding_sha !== view.bindingSha256 || row.view_sha !== directionHash(view) || row.turn_id !== view.turnId || row.human_turn_id !== (view.humanTurnId ?? null) || row.dismissed !== (view.dismissedAt ? 1 : 0) ||
        !Number.isSafeInteger(row.sequence) || Number(row.sequence) < 1) throw new Error("Direction database row changed.");
    return { ...view, source, sequence: Number(row.sequence) };
  }
  get(id: string): SteerDirectionRecord | undefined {
    const size = this.db.prepare("SELECT length(CAST(source_json AS BLOB))+length(CAST(view_json AS BLOB)) AS bytes FROM directions WHERE id=?").get(id);
    if (!size) return undefined; if (Number(size.bytes) > this.maxReadBytes()) throw new Error("Direction history awaits enough transfer RAM; its contents were not truncated.");
    return this.checked(this.db.prepare("SELECT * FROM directions WHERE id=?").get(id)!);
  }
  forTurn(turnId: string): SteerDirectionRecord[] { directionId(turnId); const result: SteerDirectionRecord[] = []; let bytes = 0;
    for (const row of this.db.prepare("SELECT id,length(CAST(source_json AS BLOB))+length(CAST(view_json AS BLOB)) AS bytes FROM directions WHERE turn_id=? OR human_turn_id=? ORDER BY sequence").iterate(turnId, turnId)) {
      bytes += Number(row.bytes); if (bytes > this.maxReadBytes()) throw new Error("Direction lookup awaits transfer RAM."); result.push(this.get(String(row.id))!);
    } return result;
  }
  bytesForTurn(turnId: string): number { directionId(turnId); return Number(this.db.prepare("SELECT SUM(length(CAST(source_json AS BLOB))+length(CAST(view_json AS BLOB))) AS bytes FROM directions WHERE turn_id=? OR human_turn_id=?").get(turnId, turnId)?.bytes ?? 0); }
  list(query: SteerDirectionPageQuery = {}): SteerDirectionPage {
    if (!query || typeof query !== "object" || Object.keys(query).some(key => !["beforeSequence", "limit", "includeDismissed"].includes(key))) throw new Error("Invalid direction page query.");
    const limit = query.limit ?? 20, before = query.beforeSequence ?? Number.MAX_SAFE_INTEGER;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100 || !Number.isSafeInteger(before) || before < 1 || query.includeDismissed !== undefined && typeof query.includeDismissed !== "boolean") throw new Error("Invalid direction page bounds.");
    const where = query.includeDismissed ? "" : " AND dismissed=0", entries: SteerDirectionRecord[] = [];
    let bytes = 0;
    for (const row of this.db.prepare(`SELECT id,length(CAST(source_json AS BLOB))+length(CAST(view_json AS BLOB)) AS bytes FROM directions WHERE sequence<?${where} ORDER BY sequence DESC LIMIT ?`).iterate(before, limit)) {
      bytes += Number(row.bytes);
      if (bytes > this.maxReadBytes()) throw new Error("Direction page awaits transfer RAM; use a smaller page.");
      entries.push(this.get(String(row.id))!);
    }
    const total = Number(this.db.prepare(`SELECT COUNT(*) AS n FROM directions WHERE 1=1${where}`).get()?.n ?? 0);
    return { entries, total, ...(entries.length === limit ? { nextBeforeSequence: entries.at(-1)!.sequence } : {}) };
  }
  private event(view: View): void {
    const sequence = Number(this.db.prepare("SELECT value FROM meta WHERE key='eventHeadSequence'").get()?.value) + 1;
    const previousSha256 = String(this.db.prepare("SELECT value FROM meta WHERE key='eventHeadSha256'").get()?.value), payload = directionCanonical(view), payloadSha256 = directionHash(view);
    if (!Number.isSafeInteger(sequence) || sequence < 1 || !SHA.test(previousSha256)) throw new Error("Direction event head is invalid.");
    const sha = directionHash({ sequence, id: view.id, revision: view.revision, payloadSha256, previousSha256 });
    this.db.prepare("INSERT INTO events VALUES(?,?,?,?,?,?,?)").run(sequence, view.id, view.revision, payload, payloadSha256, previousSha256, sha);
    this.db.prepare("UPDATE meta SET value=? WHERE key='eventHeadSequence'").run(String(sequence)); this.db.prepare("UPDATE meta SET value=? WHERE key='eventHeadSha256'").run(sha);
  }
  create(source: SteerDirectionSource, at = new Date().toISOString()): SteerDirectionRecord {
    validateDirectionSource(source); const id = directionHash(source), prior = this.get(id); if (prior) return prior;
    if (this.forTurn(source.turnId).length) throw new Error("This successor turn already has a different immutable steering direction; use a new turn.");
    return this.transaction(() => {
      const view: View = { format: "omni-steer-direction", version: 1, id, brainId: this.brainId, turnId: source.turnId,
        predecessorTurnId: source.predecessorTurnId, bindingSha256: id, state: "pending", revision: 1, createdAt: at, updatedAt: at };
      this.insert(source, view); return this.get(id)!;
    });
  }
  private insert(source: SteerDirectionSource, view: View): void {
    validateView(view, source, this.brainId);
    this.db.prepare("INSERT INTO directions(id,binding_sha,source_json,view_json,view_sha,turn_id,human_turn_id,dismissed) VALUES(?,?,?,?,?,?,?,?)")
      .run(view.id, view.bindingSha256, directionCanonical(source), directionCanonical(view), directionHash(view), view.turnId, view.humanTurnId ?? null, view.dismissedAt ? 1 : 0); this.event(view);
  }
  update(id: string, work: (view: View, source: SteerDirectionSource) => View): SteerDirectionRecord {
    return this.transaction(() => {
      const current = this.get(id); if (!current) throw new Error("Direction record is unavailable.");
      const { source, sequence: _sequence, ...view } = current, next = work(structuredClone(view), structuredClone(source));
      if (next.id !== view.id || next.brainId !== view.brainId || next.bindingSha256 !== view.bindingSha256 || next.turnId !== view.turnId || next.predecessorTurnId !== view.predecessorTurnId || next.createdAt !== view.createdAt) throw new Error("Applied direction identity is immutable.");
      for (const key of ["inputReceipt", "acceptedAttachmentIds", "reply"] as const) if (view[key] !== undefined && directionCanonical(next[key]) !== directionCanonical(view[key])) throw new Error("An applied direction receipt is immutable.");
      for (const key of ["humanMessageId", "humanTurnId", "brainMessageId"] as const) if (view[key] !== undefined && next[key] !== view[key]) throw new Error("An applied direction host projection is immutable.");
      if (directionCanonical(next) === directionCanonical(view)) return current;
      next.revision = view.revision + 1; next.updatedAt = new Date().toISOString(); validateView(next, source, this.brainId);
      this.db.prepare("UPDATE directions SET view_json=?,view_sha=?,human_turn_id=?,dismissed=? WHERE id=?").run(directionCanonical(next), directionHash(next), next.humanTurnId ?? null, next.dismissedAt ? 1 : 0, id); this.event(next); return this.get(id)!;
    });
  }
  dismiss(id: string, expectedRevision: number, dismissed: boolean): SteerDirectionRecord {
    if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 1 || typeof dismissed !== "boolean") throw new Error("Invalid direction display revision.");
    return this.update(id, view => { if (view.revision !== expectedRevision) throw new Error("Direction presentation changed; refresh before dismissing.");
      if (dismissed && !view.dismissedAt) view.dismissedAt = new Date().toISOString(); else if (!dismissed) delete view.dismissedAt; return view; });
  }
  validate(): void {
    if (this.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Direction workspace failed SQLite integrity.");
    for (const row of this.db.prepare("SELECT id FROM directions ORDER BY sequence").iterate()) this.get(String(row.id));
    let sequence = 0, previousSha256 = "0".repeat(64);
    const preceding = this.db.prepare("SELECT revision FROM events WHERE id=? AND sequence<? ORDER BY sequence DESC LIMIT 1");
    for (const sizes of this.db.prepare("SELECT length(CAST(payload_json AS BLOB)) AS bytes FROM events").iterate()) if (Number(sizes.bytes) > this.maxReadBytes()) throw new Error("Direction history validation awaits enough transfer RAM; its events were not truncated.");
    for (const row of this.db.prepare("SELECT * FROM events ORDER BY sequence").iterate()) {
      const payload = JSON.parse(String(row.payload_json)), payloadSha256 = directionHash(payload);
      if (row.sequence !== ++sequence || row.previous_sha !== previousSha256 || row.payload_sha !== payloadSha256 || row.row_sha !== directionHash({ sequence, id: row.id, revision: row.revision, payloadSha256, previousSha256 }) ||
          payload.id !== row.id || payload.revision !== row.revision || (() => { const prior = preceding.get(String(row.id), Number(row.sequence)); return prior && Number(row.revision) !== Number(prior.revision) + 1; })()) throw new Error("Direction event history changed.");
      previousSha256 = String(row.row_sha);
    }
    if (String(sequence) !== this.db.prepare("SELECT value FROM meta WHERE key='eventHeadSequence'").get()?.value || previousSha256 !== this.db.prepare("SELECT value FROM meta WHERE key='eventHeadSha256'").get()?.value) throw new Error("Direction history lost its committed head.");
    const latest = this.db.prepare("SELECT payload_sha FROM events WHERE id=? ORDER BY sequence DESC LIMIT 1");
    for (const row of this.db.prepare("SELECT id,view_sha FROM directions").iterate()) if (latest.get(String(row.id))?.payload_sha !== row.view_sha) throw new Error("Direction view differs from its durable event head.");
  }
  /** Private isolated copy only; source bindings and original event history remain unchanged. */
  rekey(targetId: string, reason: "copied-instance" | "recovered-history" = "copied-instance"): void {
    directionId(targetId);
    this.transaction(() => { for (const row of this.db.prepare("SELECT id FROM directions ORDER BY sequence").iterate()) { const old = this.get(String(row.id))!;
      const { source: _source, sequence: _sequence, ...view } = old; view.brainId = targetId; view.revision += 1; view.updatedAt = new Date().toISOString();
      if (["pending", "received"].includes(view.state)) view.dormant = { reason, fromBrainId: this.brainId, at: view.updatedAt };
      view.provenance = [...(view.provenance ?? []), { brainId: this.brainId, id: view.id }];
      this.db.prepare("UPDATE directions SET view_json=?,view_sha=? WHERE id=?").run(directionCanonical(view), directionHash(view), view.id); this.event(view);
    } this.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetId); });
  }
  adopt(source: SteerDirectionRecord, projection: { turnId: string; predecessorTurnId: string; humanMessageId?: string; humanTurnId?: string; brainMessageId?: string }, parentId: string, reason: "merged-history" | "recovered-history"): void {
    const { source: binding, sequence: _sequence, ...view } = structuredClone(source);
    const identity = directionHash({ bindingSha256: source.bindingSha256, projection, inputReceipt: source.inputReceipt ?? null, reply: source.reply ?? null, state: source.state });
    let existingId = this.get(identity)?.id;
    // A selected recovery snapshot or shared ancestor can already contain the
    // same applied record under its original ID. Union that history once.
    if (!existingId) for (const row of this.db.prepare("SELECT id FROM directions WHERE binding_sha=? ORDER BY sequence").iterate(source.bindingSha256)) {
      const old = this.get(String(row.id))!;
      const oldProjection = { turnId: old.turnId, predecessorTurnId: old.predecessorTurnId,
        ...(old.humanMessageId ? { humanMessageId: old.humanMessageId } : {}), ...(old.humanTurnId ? { humanTurnId: old.humanTurnId } : {}),
        ...(old.brainMessageId ? { brainMessageId: old.brainMessageId } : {}) };
      if (directionHash({ bindingSha256: old.bindingSha256, projection: oldProjection, inputReceipt: old.inputReceipt ?? null, reply: old.reply ?? null, state: old.state }) === identity) { existingId = old.id; break; }
    }
    if (existingId) { this.update(existingId, existing => {
      const provenance = { brainId: parentId, id: source.id };
      if (!(existing.provenance ?? []).some(item => item.brainId === provenance.brainId && item.id === provenance.id)) existing.provenance = [...(existing.provenance ?? []), provenance];
      if (!source.dismissedAt) delete existing.dismissedAt; if (source.failure) existing.failure ??= source.failure; return existing;
    }); return; }
    view.id = identity; view.brainId = this.brainId; Object.assign(view, projection); view.revision += 1; view.updatedAt = new Date().toISOString();
    for (const key of ["humanMessageId", "humanTurnId", "brainMessageId"] as const) if (projection[key] === undefined) delete view[key];
    view.provenance = [...(view.provenance ?? []), { brainId: parentId, id: source.id }];
    view.dormant = { reason, fromBrainId: parentId, at: view.updatedAt };
    this.transaction(() => this.insert(binding, view));
  }
}
