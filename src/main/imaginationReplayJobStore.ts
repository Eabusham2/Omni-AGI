import { createHash } from "node:crypto";
import { constants, lstatSync } from "node:fs";
import { lstat, mkdir, open } from "node:fs/promises";
import { freemem } from "node:os";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { getHeapStatistics } from "node:v8";
import type { ImaginationAssembly } from "../shared/imaginationAssembly";
import { sqliteFilename } from "./sqliteFilename";

export interface ReplayJobSegment { index: number; path: string; sha256: string; bytes: number }
export interface ReplayJobHead { sequence: number; sha256: string; revision: number; segmentCount: number; segmentsSha256: string }
export interface ReplayJobStoreOptions {
  create?: ImaginationAssembly;
  readOnly?: boolean;
  signal?: AbortSignal;
  /** Additional current-owner admission. Called before a stored JSON body is fetched or parsed. */
  admitBytes?(bytes: number): void;
}
type Row = Record<string, unknown>;
type Kind = "create" | "view" | "segment" | "rekey";
interface Origin {
  format: "omni-replay-job-state"; version: 1; schemaSha256: string;
  brainId: string; assemblyId: string;
  invariant: Pick<ImaginationAssembly, "requestId" | "recordingId" | "recordingRevision" | "manifestSha256" | "createdAt" | "totalSpans" | "durationMs" | "label">;
}
interface CheckedEvent {
  sequence: number; kind: Kind; previousSha256: string; view: ImaginationAssembly; viewBody: string;
  segmentIndex: number | null; segmentSha256: string | null; segmentsSha256: string; sha256: string;
}
interface CheckedSegment { value: ReplayJobSegment; previousSha256: string; sha256: string; eventSequence: number }
const label = "Delivered timeline replay — presentation timing, not native synchronization";
const kinds = new Set<Kind>(["create", "view", "segment", "rekey"]);
const active = new Set<ImaginationAssembly["state"]>(["queued", "preparing", "encoding"]);
const states = new Set<ImaginationAssembly["state"]>([...active, "complete", "cancelled", "failed", "dormant"]);
const schema = new Map([
  ["meta", "CREATE TABLE meta(singleton INTEGER PRIMARY KEY CHECK(singleton=1),body TEXT NOT NULL,sha256 TEXT NOT NULL)"],
  ["events", "CREATE TABLE events(sequence INTEGER PRIMARY KEY CHECK(sequence>=1),kind TEXT NOT NULL,previous_sha256 TEXT NOT NULL,body TEXT NOT NULL,body_sha256 TEXT NOT NULL,segment_index INTEGER,segment_sha256 TEXT,segments_sha256 TEXT NOT NULL,event_sha256 TEXT NOT NULL)"],
  ["segments", "CREATE TABLE segments(idx INTEGER PRIMARY KEY CHECK(idx>=0),body TEXT NOT NULL,body_sha256 TEXT NOT NULL,previous_sha256 TEXT NOT NULL,segment_sha256 TEXT NOT NULL,event_sequence INTEGER NOT NULL UNIQUE)"],
  ["head", "CREATE TABLE head(singleton INTEGER PRIMARY KEY CHECK(singleton=1),sequence INTEGER NOT NULL,event_sha256 TEXT NOT NULL,segment_count INTEGER NOT NULL,segments_sha256 TEXT NOT NULL,sha256 TEXT NOT NULL)"]
]);
const hash = (text: string): string => createHash("sha256").update(text).digest("hex");
const schemaSha256 = hash(JSON.stringify([...schema]));
const sqlShape = (value: unknown): string => String(value).replace(/\s+|;$/g, "").toUpperCase();
const id = (value: unknown): string => {
  if (typeof value !== "string" || !/^[a-f0-9]{32}(?:[a-f0-9]{32})?$/.test(value)) throw new Error("Invalid replay assembly identity.");
  return value;
};
const owner = (value: unknown): string => {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value)) throw new Error("Invalid replay owner or request identity.");
  return value;
};
const sha = (value: unknown): string => {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Replay state has an invalid SHA-256 descriptor.");
  return value;
};
const integer = (value: unknown, minimum = 0): number => {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < minimum) throw new Error("Replay state has an invalid committed cursor.");
  return value;
};
function object(value: unknown): Record<string, any> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Replay state must contain a JSON object.");
  return value as Record<string, any>;
}
function timestamp(value: unknown): void {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error("Replay state has an invalid timestamp.");
}
function invariant(view: ImaginationAssembly): Origin["invariant"] {
  return { requestId: view.requestId, recordingId: view.recordingId, recordingRevision: view.recordingRevision,
    manifestSha256: view.manifestSha256, createdAt: view.createdAt, totalSpans: view.totalSpans,
    durationMs: view.durationMs, label: view.label };
}
function checkView(value: unknown): ImaginationAssembly {
  const view = object(value);
  if (view.format !== "omni-imagination-assembly" || view.version !== 1 || view.label !== label || !states.has(view.state)) throw new Error("Replay state has an unsupported view.");
  id(view.id); owner(view.brainId); owner(view.requestId); owner(view.recordingId); sha(view.manifestSha256);
  integer(view.revision, 1); integer(view.recordingRevision, 1); integer(view.totalSpans); integer(view.completedSpans);
  timestamp(view.createdAt); timestamp(view.updatedAt);
  if (view.completedSpans > view.totalSpans || typeof view.durationMs !== "number" || !Number.isFinite(view.durationMs) || view.durationMs < 0) throw new Error("Replay state has an inconsistent span cursor or duration.");
  if (view.inheritedFrom !== undefined) { const source = object(view.inheritedFrom); owner(source.brainId); id(source.assemblyId); }
  for (const key of ["artifactSha256", "receiptSha256"] as const) if (view[key] !== undefined) sha(view[key]);
  if (view.bytes !== undefined) integer(view.bytes, 1);
  if (view.mimeType !== undefined && !["audio/wav", "video/mp4"].includes(view.mimeType)) throw new Error("Replay state has an unsupported output type.");
  if (view.error !== undefined && typeof view.error !== "string" || view.mediaUrl !== undefined && typeof view.mediaUrl !== "string" || view.available !== undefined && typeof view.available !== "boolean") throw new Error("Replay state has an invalid presentation field.");
  if (view.currentAttempt !== undefined || view.currentSpan !== undefined) {
    if (typeof view.currentAttempt !== "string" || !/^[a-f0-9]{32}$/.test(view.currentAttempt) || integer(view.currentSpan) !== view.completedSpans || view.currentSpan >= view.totalSpans) throw new Error("Replay in-progress cursor is not the next uncommitted span.");
  }
  if (view.codec !== undefined) {
    const codec = object(view.codec);
    if (!["ready", "external"].includes(codec.state) || typeof codec.distributionLicense !== "string" || !Array.isArray(codec.encoders) || codec.encoders.some((entry: unknown) => typeof entry !== "string")) throw new Error("Replay codec provenance is invalid.");
    sha(codec.binarySha256); if (codec.artifactSha256 !== undefined) sha(codec.artifactSha256);
    if (codec.artifactId !== undefined && typeof codec.artifactId !== "string") throw new Error("Replay codec artifact identity is invalid.");
  }
  if (view.state === "complete" && (view.completedSpans !== view.totalSpans || !view.artifactSha256 || !view.receiptSha256 || !view.bytes || !view.mimeType)) throw new Error("Completed replay is missing its committed spans or result receipt.");
  return view as ImaginationAssembly;
}
function checkSegment(value: unknown): ReplayJobSegment {
  const segment = object(value), index = integer(segment.index), bytes = integer(segment.bytes, 1);
  if (typeof segment.path !== "string" || !new RegExp(`^work/[a-f0-9]{32}/segment-${index}\\.(?:mkv|wav)$`).test(segment.path)) throw new Error("Replay segment path does not match its exact committed index.");
  return { index, path: segment.path, sha256: sha(segment.sha256), bytes };
}
export function replayJobStatePath(root: string, assemblyId: string): string {
  return join(root, "imagination-assemblies", "jobs", id(assemblyId), "state.sqlite3");
}
function owned(info: NonNullable<ReturnType<typeof lstatSync>>, directory: boolean): void {
  if (info.isSymbolicLink() || (directory ? !info.isDirectory() : !info.isFile()) || !directory && info.nlink !== 1) throw new Error("Replay job state requires isolated real owned files and directories.");
  if (process.platform !== "win32" && typeof process.getuid === "function" && info.uid !== process.getuid()) throw new Error("Replay job state has another operating-system owner.");
}

/**
 * Disk-backed encode cursor, not a neural memory or an execution authority.
 * Reads check their exact row hashes before JSON parsing. Status is bounded;
 * validate() streams the entire append-only history at custody boundaries.
 * Hashes detect changed bytes/rows, not a hostile owner rewriting the entire DB;
 * portable custody must additionally seal the exact database file externally.
 */
export class ReplayJobStore {
  private depth = 0;
  private closed = false;
  private origin!: Origin;
  private originSha256!: string;
  private originSegmentsSha256!: string;
  private currentId: string;
  private currentOwner: string;
  private constructor(private readonly db: DatabaseSync, private readonly path: string, private readonly directories: string[],
    private readonly identity: { dev: number; ino: number }, private readonly pathId: string, savedId: string,
    savedOwner: string, private readonly options: ReplayJobStoreOptions) { this.currentId = savedId; this.currentOwner = savedOwner; }

  static open(root: string, assemblyId: string, brainId: string, options: ReplayJobStoreOptions = {}): Promise<ReplayJobStore> {
    return ReplayJobStore.openAt(root, id(assemblyId), owner(brainId), assemblyId, options);
  }
  private static async openAt(root: string, pathId: string, brainId: string, savedId: string, options: ReplayJobStoreOptions): Promise<ReplayJobStore> {
    if (options.readOnly && options.create) throw new Error("Read-only replay state cannot initialize a job.");
    options.signal?.throwIfAborted();
    const base = resolve(root), directories = [base, join(base, "imagination-assemblies"), join(base, "imagination-assemblies", "jobs"), join(base, "imagination-assemblies", "jobs", pathId)];
    for (const [index, directory] of directories.entries()) {
      if (options.create && index) await mkdir(directory, { mode: 0o700 }).catch(error => { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; });
      owned(await lstat(directory), true);
    }
    const path = replayJobStatePath(base, pathId);
    let present = await lstat(path).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; });
    const fresh = !present;
    if (fresh) {
      if (!options.create) throw new Error("Replay job state is missing; no cursor was silently recreated.");
      const file = await open(path, constants.O_CREAT | constants.O_EXCL | constants.O_WRONLY | (constants.O_NOFOLLOW ?? 0), 0o600);
      await file.close(); present = await lstat(path);
    }
    owned(present!, false);
    for (const suffix of ["-journal", "-wal", "-shm"]) {
      const info = await lstat(path + suffix).catch(error => { if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined; throw error; });
      if (info) { owned(info, false); if (suffix !== "-journal") throw new Error("Replay state requires a committed standalone SQLite snapshot, not detached WAL state."); }
    }
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: options.readOnly ?? false, allowExtension: false });
    const store = new ReplayJobStore(db, path, directories, { dev: present!.dev, ino: present!.ino }, pathId, savedId, brainId, options);
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-256; PRAGMA mmap_size=0; PRAGMA temp_store=FILE; PRAGMA busy_timeout=5000;");
      if (!fresh && db.prepare("PRAGMA journal_mode").get()?.journal_mode !== "delete") throw new Error("Replay state is not a committed standalone DELETE-journal database.");
      if (!options.readOnly) db.exec("PRAGMA synchronous=FULL;");
      if (fresh) {
        db.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;");
        store.transaction(true, () => {
          for (const declaration of schema.values()) db.exec(declaration);
          const view = checkView(store.json(options.create!).value);
          if (view.id !== pathId || view.brainId !== brainId || view.revision !== 1 || view.completedSpans !== 0 || view.inheritedFrom !== undefined) throw new Error("Initial replay state must be its exact new owner at cursor zero and revision one.");
          const origin: Origin = { format: "omni-replay-job-state", version: 1, schemaSha256, brainId, assemblyId: pathId, invariant: invariant(view) };
          const encoded = store.json(origin); db.prepare("INSERT INTO meta VALUES(1,?,?)").run(encoded.body, encoded.sha256);
          store.origin = origin; store.originSha256 = encoded.sha256; store.originSegmentsSha256 = hash(`${encoded.sha256}:segments`);
          store.append("create", view, null, { sequence: 0, sha256: encoded.sha256, revision: 0, segmentCount: 0, segmentsSha256: store.originSegmentsSha256 });
        });
      }
      store.transaction(false, () => { store.verifySchema(); store.readOrigin(); store.current(); });
      return store;
    } catch (error) { db.close(); store.closed = true; throw error; }
  }
  close(): void { if (!this.closed) { this.db.close(); this.closed = true; } }
  private assertFile(): void {
    if (this.closed) throw new Error("Replay job store is closed.");
    this.options.signal?.throwIfAborted();
    for (const directory of this.directories) owned(lstatSync(directory), true);
    const info = lstatSync(this.path); owned(info, false);
    if (info.dev !== this.identity.dev || info.ino !== this.identity.ino) throw new Error("Replay job database was replaced; reopen its checked custody.");
  }
  private transaction<T>(write: boolean, work: () => T): T {
    this.assertFile();
    if (write && this.options.readOnly) throw new Error("Read-only replay job cannot commit or rekey.");
    if (this.depth) return work();
    this.db.exec(write ? "BEGIN IMMEDIATE" : "BEGIN"); this.depth++;
    try { if (write && this.origin) this.verifySchema(); const value = work(); this.assertFile(); this.db.exec("COMMIT"); return value; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
    finally { this.depth--; }
  }
  private admit(bytes: unknown): number {
    const n = integer(bytes);
    this.options.signal?.throwIfAborted();
    const headroom = Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed);
    if (!Number.isSafeInteger(n * 12) || n * 12 > headroom / 2) throw new Error("Replay metadata awaits physical parse headroom; no history or cursor was truncated.");
    this.options.admitBytes?.(n); this.options.signal?.throwIfAborted(); return n;
  }
  private json<T>(value: T): { body: string; sha256: string; value: T } {
    const body = JSON.stringify(value);
    if (body === undefined) throw new Error("Replay state requires explicit JSON.");
    this.admit(Buffer.byteLength(body)); return { body, sha256: hash(body), value: JSON.parse(body) as T };
  }
  private parsed(body: unknown, expected: unknown, admittedBytes: number): any {
    if (typeof body !== "string" || Buffer.byteLength(body) !== admittedBytes || hash(body) !== sha(expected)) throw new Error("Replay row bytes differ from their exact committed checksum.");
    this.options.signal?.throwIfAborted(); return JSON.parse(body);
  }
  private verifySchema(): void {
    const seen = new Set<string>();
    for (const sizes of this.db.prepare("SELECT rowid,length(CAST(type AS BLOB)) AS type_bytes,length(CAST(name AS BLOB)) AS name_bytes,length(CAST(sql AS BLOB)) AS sql_bytes FROM sqlite_master").iterate()) {
      this.options.signal?.throwIfAborted();
      this.admit(integer(sizes.type_bytes) + integer(sizes.name_bytes) + (sizes.sql_bytes === null ? 0 : integer(sizes.sql_bytes)));
      const row = this.db.prepare("SELECT type,name,sql FROM sqlite_master WHERE rowid=?").get(integer(sizes.rowid, 1))!;
      if (row.type === "index" && row.name === "sqlite_autoindex_segments_1" && row.sql === null) continue;
      const name = String(row.name), expected = schema.get(name);
      if (row.type !== "table" || !expected || seen.has(name)) throw new Error("Replay state contains unsupported executable/schema objects.");
      if (sqlShape(row.sql) !== sqlShape(expected)) throw new Error("Replay state schema changed."); seen.add(name);
    }
    if (seen.size !== schema.size) throw new Error("Replay state schema is incomplete; it was not silently repaired.");
  }
  private readOrigin(): void {
    const size = this.db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes,length(sha256) AS hash_length FROM meta WHERE singleton=1").get();
    if (!size || size.hash_length !== 64) throw new Error("Replay state lost its original intent binding.");
    const bytes = this.admit(size.bytes), row = this.db.prepare("SELECT body,sha256 FROM meta WHERE singleton=1").get()!;
    const origin = object(this.parsed(row.body, row.sha256, bytes)) as unknown as Origin;
    if (origin.format !== "omni-replay-job-state" || origin.version !== 1 || origin.schemaSha256 !== schemaSha256) throw new Error("Replay state format/schema checksum changed.");
    owner(origin.brainId); id(origin.assemblyId);
    const fields = object(origin.invariant);
    owner(fields.requestId); owner(fields.recordingId); integer(fields.recordingRevision, 1); sha(fields.manifestSha256); timestamp(fields.createdAt); integer(fields.totalSpans);
    if (fields.label !== label || typeof fields.durationMs !== "number" || !Number.isFinite(fields.durationMs) || fields.durationMs < 0) throw new Error("Replay original source descriptor changed.");
    if (this.originSha256 && this.originSha256 !== row.sha256) throw new Error("Replay original intent binding was replaced while open.");
    this.origin = origin; this.originSha256 = String(row.sha256); this.originSegmentsSha256 = hash(`${this.originSha256}:segments`);
  }
  private sourceView(value: unknown): ImaginationAssembly {
    const view = checkView(value);
    if (JSON.stringify(invariant(view)) !== JSON.stringify(this.origin.invariant)) throw new Error("Replay source manifest or immutable identity changed.");
    const inherited = view.inheritedFrom;
    if ((view.id !== this.origin.assemblyId || view.brainId !== this.origin.brainId) &&
      (inherited?.assemblyId !== this.origin.assemblyId || inherited?.brainId !== this.origin.brainId) || inherited &&
      (inherited.assemblyId !== this.origin.assemblyId || inherited.brainId !== this.origin.brainId)) throw new Error("Replay inheritance no longer binds its original source owner.");
    return view;
  }
  private event(sequence: number): CheckedEvent {
    const size = this.db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes,length(previous_sha256) AS previous_length,length(body_sha256) AS body_length,length(segment_sha256) AS segment_length,length(segments_sha256) AS cursor_length,length(event_sha256) AS hash_length,length(kind) AS kind_length FROM events WHERE sequence=?").get(sequence);
    if (!size || size.previous_length !== 64 || size.body_length !== 64 || size.segment_length !== null && size.segment_length !== 64 || size.cursor_length !== 64 || size.hash_length !== 64 || Number(size.kind_length) > 7) throw new Error("Replay event chain is missing or malformed.");
    const bytes = this.admit(size.bytes), row = this.db.prepare("SELECT * FROM events WHERE sequence=?").get(sequence)!;
    if (integer(row.sequence, 1) !== sequence || !kinds.has(row.kind as Kind)) throw new Error("Replay event sequence or kind changed.");
    const descriptor = { sequence, kind: row.kind as Kind, previousSha256: sha(row.previous_sha256), viewSha256: sha(row.body_sha256),
      segmentIndex: row.segment_index === null ? null : integer(row.segment_index), segmentSha256: row.segment_sha256 === null ? null : sha(row.segment_sha256), segmentsSha256: sha(row.segments_sha256) };
    if (hash(JSON.stringify(descriptor)) !== sha(row.event_sha256)) throw new Error("Replay event chain checksum changed.");
    const view = this.sourceView(this.parsed(row.body, row.body_sha256, bytes));
    if (view.revision !== sequence || (descriptor.kind === "segment") !== (descriptor.segmentIndex !== null && descriptor.segmentSha256 !== null) || descriptor.kind !== "segment" && (descriptor.segmentIndex !== null || descriptor.segmentSha256 !== null)) throw new Error("Replay event does not bind its exact revision/segment cursor.");
    return { ...descriptor, view, viewBody: String(row.body), sha256: String(row.event_sha256) };
  }
  private storedSegment(index: number): CheckedSegment {
    const size = this.db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes,length(body_sha256) AS body_length,length(previous_sha256) AS previous_length,length(segment_sha256) AS hash_length FROM segments WHERE idx=?").get(index);
    if (!size || size.body_length !== 64 || size.previous_length !== 64 || size.hash_length !== 64) throw new Error("Replay committed segment is missing or malformed.");
    const bytes = this.admit(size.bytes), row = this.db.prepare("SELECT * FROM segments WHERE idx=?").get(index)!;
    const eventSequence = integer(row.event_sequence, 1), previousSha256 = sha(row.previous_sha256);
    const expected = hash(JSON.stringify({ index, bodySha256: sha(row.body_sha256), previousSha256, eventSequence }));
    if (integer(row.idx) !== index || expected !== sha(row.segment_sha256)) throw new Error("Replay segment chain checksum changed.");
    const value = checkSegment(this.parsed(row.body, row.body_sha256, bytes));
    if (value.index !== index) throw new Error("Replay segment row and body have different committed cursors.");
    return { value, previousSha256, sha256: expected, eventSequence };
  }
  private segmentEvent(segment: CheckedSegment): CheckedEvent {
    const event = this.event(segment.eventSequence);
    if (event.kind !== "segment" || event.segmentIndex !== segment.value.index || event.segmentSha256 !== segment.sha256 || event.segmentsSha256 !== segment.sha256 || event.view.completedSpans !== segment.value.index + 1) throw new Error("Replay segment has no matching committed event.");
    return event;
  }
  private current(): { head: ReplayJobHead; event: CheckedEvent } {
    this.readOrigin();
    const sizes = this.db.prepare("SELECT length(event_sha256) AS event_length,length(segments_sha256) AS segments_length,length(sha256) AS hash_length FROM head WHERE singleton=1").get();
    if (!sizes || sizes.event_length !== 64 || sizes.segments_length !== 64 || sizes.hash_length !== 64) throw new Error("Replay committed head is missing or malformed; no empty job was substituted.");
    const row = this.db.prepare("SELECT * FROM head WHERE singleton=1").get()!;
    const head: ReplayJobHead = { sequence: integer(row.sequence, 1), sha256: sha(row.event_sha256), revision: Number(row.sequence), segmentCount: integer(row.segment_count), segmentsSha256: sha(row.segments_sha256) };
    if (hash(JSON.stringify(head)) !== sha(row.sha256)) throw new Error("Replay committed head checksum changed.");
    const event = this.event(head.sequence);
    const firstEvent = this.db.prepare("SELECT sequence FROM events ORDER BY sequence LIMIT 1").get()?.sequence;
    const lastEvent = this.db.prepare("SELECT sequence FROM events ORDER BY sequence DESC LIMIT 1").get()?.sequence;
    const firstSegment = this.db.prepare("SELECT idx FROM segments ORDER BY idx LIMIT 1").get()?.idx;
    const lastSegment = this.db.prepare("SELECT idx FROM segments ORDER BY idx DESC LIMIT 1").get()?.idx;
    if (firstEvent !== 1 || lastEvent !== head.sequence || event.sha256 !== head.sha256 || event.segmentsSha256 !== head.segmentsSha256 ||
      event.view.completedSpans !== head.segmentCount || (head.segmentCount ? firstSegment !== 0 || lastSegment !== head.segmentCount - 1 : firstSegment !== undefined || lastSegment !== undefined)) throw new Error("Replay committed head/segment endpoints are inconsistent.");
    if (event.view.id !== this.currentId || event.view.brainId !== this.currentOwner) throw new Error("Replay job belongs to another exact current owner or assembly.");
    if (head.segmentCount) { const segment = this.storedSegment(head.segmentCount - 1); if (segment.sha256 !== head.segmentsSha256) throw new Error("Replay committed segment head changed."); this.segmentEvent(segment); }
    else if (head.segmentsSha256 !== this.originSegmentsSha256) throw new Error("Replay empty segment origin changed.");
    return { head, event };
  }
  head(): ReplayJobHead { return this.transaction(false, () => this.current().head); }
  view(): ImaginationAssembly { return this.transaction(false, () => this.current().event.view); }
  count(): number { return this.head().segmentCount; }
  segment(index: number): ReplayJobSegment | undefined {
    integer(index);
    return this.transaction(false, () => {
      const { head } = this.current(); if (index >= head.segmentCount) return undefined;
      const segment = this.storedSegment(index); this.segmentEvent(segment);
      const previous = index ? this.storedSegment(index - 1).sha256 : this.originSegmentsSha256;
      if (segment.previousSha256 !== previous) throw new Error("Replay segment predecessor changed."); return segment.value;
    });
  }
  /** Bounded rows, one verified original cursor at a time; never a growing array. */
  *segments(): Generator<ReplayJobSegment> {
    const initial = this.head(); let previous = this.originSegmentsSha256;
    for (let index = 0; index < initial.segmentCount; index++) {
      const segment = this.transaction(false, () => {
        this.current(); const value = this.storedSegment(index); this.segmentEvent(value);
        if (value.previousSha256 !== previous) throw new Error("Replay segment chain is incomplete or changed."); return value;
      });
      previous = segment.sha256;
      if (index === initial.segmentCount - 1 && previous !== initial.segmentsSha256) throw new Error("Replay iteration no longer reaches its exact committed cursor.");
      yield segment.value;
    }
  }
  private append(kind: Kind, next: ImaginationAssembly, segment: ReplayJobSegment | null, prior: ReplayJobHead): ImaginationAssembly {
    const encoded = this.json(next), sequence = prior.sequence + 1;
    let segmentsSha256 = prior.segmentsSha256, segmentSha256: string | null = null;
    if (segment) {
      const data = this.json(segment);
      segmentSha256 = hash(JSON.stringify({ index: segment.index, bodySha256: data.sha256, previousSha256: prior.segmentsSha256, eventSequence: sequence }));
      this.db.prepare("INSERT INTO segments VALUES(?,?,?,?,?,?)").run(segment.index, data.body, data.sha256, prior.segmentsSha256, segmentSha256, sequence);
      segmentsSha256 = segmentSha256;
    }
    const descriptor = { sequence, kind, previousSha256: prior.sha256, viewSha256: encoded.sha256,
      segmentIndex: segment?.index ?? null, segmentSha256, segmentsSha256 };
    const eventSha256 = hash(JSON.stringify(descriptor));
    this.db.prepare("INSERT INTO events VALUES(?,?,?,?,?,?,?,?,?)").run(sequence, kind, prior.sha256, encoded.body, encoded.sha256, descriptor.segmentIndex, segmentSha256, segmentsSha256, eventSha256);
    const head: ReplayJobHead = { sequence, sha256: eventSha256, revision: sequence, segmentCount: prior.segmentCount + (segment ? 1 : 0), segmentsSha256 };
    const checkedHead = this.json(head);
    this.db.prepare("INSERT INTO head VALUES(1,?,?,?,?,?) ON CONFLICT(singleton) DO UPDATE SET sequence=excluded.sequence,event_sha256=excluded.event_sha256,segment_count=excluded.segment_count,segments_sha256=excluded.segments_sha256,sha256=excluded.sha256")
      .run(head.sequence, head.sha256, head.segmentCount, head.segmentsSha256, checkedHead.sha256);
    return encoded.value;
  }
  private nextView(next: ImaginationAssembly, expectedRevision: number, head: ReplayJobHead, count: number): ImaginationAssembly {
    integer(expectedRevision, 1); if (expectedRevision === Number.MAX_SAFE_INTEGER) throw new Error("Replay revision is outside the exact SQLite/JSON cursor range.");
    const checked = this.sourceView(this.json(next).value);
    if (head.revision !== expectedRevision) throw new Error("Replay revision changed; refresh before committing this exact cursor.");
    if (checked.revision !== expectedRevision + 1 || checked.id !== this.currentId || checked.brainId !== this.currentOwner || checked.completedSpans !== count) throw new Error("Replay next view changed ownership, revision or committed segment count.");
    return checked;
  }
  saveView(next: ImaginationAssembly, expectedRevision: number): ImaginationAssembly {
    return this.transaction(true, () => { const { head } = this.current(); return this.append("view", this.nextView(next, expectedRevision, head, head.segmentCount), null, head); });
  }
  commitSegment(value: ReplayJobSegment, next: ImaginationAssembly, expectedRevision: number): ImaginationAssembly {
    const segment = checkSegment(value);
    return this.transaction(true, () => {
      const { head, event: current } = this.current();
      if (segment.index < head.segmentCount) {
        const stored = this.storedSegment(segment.index), committed = this.segmentEvent(stored);
        if (JSON.stringify(stored.value) !== JSON.stringify(segment) || expectedRevision !== committed.sequence - 1 || this.json(next).sha256 !== hash(committed.viewBody)) throw new Error("Repeated replay segment commit differs from its original exact bytes or view.");
        return current.view;
      }
      if (segment.index !== head.segmentCount) throw new Error("Replay segment commit cannot skip an uncommitted span.");
      const checked = this.nextView(next, expectedRevision, head, head.segmentCount + 1);
      return this.append("segment", checked, segment, head);
    });
  }
  rekey(newOwner: string, newId = this.pathId): ImaginationAssembly {
    owner(newOwner); id(newId);
    if (newId !== this.pathId) throw new Error("Replay rekey requires the already-copied exact destination directory.");
    const result = this.transaction(true, () => {
      this.validate();
      const { head, event } = this.current();
      if (head.revision === Number.MAX_SAFE_INTEGER) throw new Error("Replay revision is outside the exact SQLite/JSON cursor range.");
      const next: ImaginationAssembly = { ...event.view, id: newId, brainId: newOwner, revision: head.revision + 1, updatedAt: new Date().toISOString(),
        inheritedFrom: { brainId: this.origin.brainId, assemblyId: this.origin.assemblyId } };
      if (active.has(next.state)) next.state = "dormant";
      delete next.currentAttempt; delete next.currentSpan; delete next.mediaUrl; delete next.available;
      return this.append("rekey", this.sourceView(next), null, head);
    });
    this.currentOwner = newOwner; this.currentId = newId; return result;
  }
  /** Full-chain custody/recovery validation is O(rows) I/O but bounded memory. */
  validate(): void {
    this.transaction(false, () => {
      this.verifySchema(); const { head } = this.current();
      let previous = this.originSha256, segmentsSha256 = this.originSegmentsSha256, count = 0, prior: ImaginationAssembly | undefined;
      for (let sequence = 1; sequence <= head.sequence; sequence++) {
        const event = this.event(sequence);
        if (event.previousSha256 !== previous || sequence === 1 && event.kind !== "create" || sequence > 1 && event.kind === "create") throw new Error("Replay event history is missing or changed.");
        if (event.kind === "segment") {
          const segment = this.storedSegment(count);
          if (event.segmentIndex !== count || segment.eventSequence !== sequence || segment.previousSha256 !== segmentsSha256 || event.segmentSha256 !== segment.sha256) throw new Error("Replay segment history is not a contiguous atomic cursor.");
          segmentsSha256 = segment.sha256; count++;
        }
        if (event.view.completedSpans !== count || event.segmentsSha256 !== segmentsSha256) throw new Error("Replay event advanced without its complete committed segment history.");
        if (event.kind === "create" && (event.view.id !== this.origin.assemblyId || event.view.brainId !== this.origin.brainId || event.view.inheritedFrom !== undefined)) throw new Error("Replay original owner event changed.");
        if (prior && event.kind !== "rekey" && (event.view.id !== prior.id || event.view.brainId !== prior.brainId)) throw new Error("Replay owner changed outside an explicit dormant rekey event.");
        if (event.kind === "rekey" && (!prior || event.view.inheritedFrom?.brainId !== this.origin.brainId || event.view.inheritedFrom.assemblyId !== this.origin.assemblyId || active.has(prior.state) && event.view.state !== "dormant" || event.view.currentAttempt !== undefined || event.view.currentSpan !== undefined || event.view.mediaUrl !== undefined || event.view.available !== undefined)) throw new Error("Replay copied owner inherited live execution or lost its source custody.");
        previous = event.sha256; prior = event.view;
      }
      if (previous !== head.sha256 || segmentsSha256 !== head.segmentsSha256 || count !== head.segmentCount ||
        this.db.prepare("SELECT COUNT(*) AS n FROM events").get()?.n !== head.sequence || this.db.prepare("SELECT COUNT(*) AS n FROM segments").get()?.n !== count) throw new Error("Replay committed head has missing or additional history rows.");
    });
  }
  /** Only the explicit copied-directory adapter can open old saved identity at a new path. */
  static async rekeyCopied(root: string, pathId: string, sourceOwner: string, sourceId: string, newOwner: string): Promise<void> {
    const store = await ReplayJobStore.openAt(root, id(pathId), owner(sourceOwner), id(sourceId), {});
    try { store.validate(); store.rekey(owner(newOwner), pathId); store.validate(); } finally { store.close(); }
  }
}

export const rekeyCopiedReplayJobStore = (root: string, pathId: string, sourceOwner: string, sourceId: string, newOwner: string): Promise<void> =>
  ReplayJobStore.rekeyCopied(root, pathId, sourceOwner, sourceId, newOwner);
