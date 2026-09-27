import { createHash, randomBytes } from "node:crypto";
import { open as openFile } from "node:fs/promises";
import { access, copyFile, mkdir, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  BrainActivityLedgerSummary,
  JournalEntry,
  JournalLedgerEntry,
  TrainingSource,
  TrainingSourceLedgerEntry
} from "../shared/types";

const FORMAT = "omni-brain-activity-ledger";
const FORMAT_VERSION = 1;
const JOURNAL_EXPORT_FORMAT = "omni-journal-export";
const SOURCE_EXPORT_FORMAT = "omni-training-source-export";
const ZERO_HASH = "0".repeat(64);
const SAFE_BRAIN_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SAFE_ENTRY_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}$/;
const CURSOR = /^([1-9][0-9]{0,15})\.([a-f0-9]{64})$/;
const JOURNAL_KINDS = new Set([
  "learning",
  "consolidation",
  "tool",
  "fork",
  "reflection",
  "system"
]);
const SOURCE_KINDS = new Set([
  "pdf",
  "text",
  "markdown",
  "code",
  "json",
  "csv",
  "parquet",
  "arrow",
  "archive",
  "sqlite",
  "dataset",
  "image",
  "audio",
  "video",
  "unknown"
]);
const SOURCE_POLICIES = new Set(["encode", "consolidate", "pretrain", "archive"]);

export const ACTIVITY_LEDGER_DIRECTORY = "activity";
export const ACTIVITY_LEDGER_DATABASE = "ledger.sqlite3";

interface JournalRow {
  sequence: number;
  entry_id: string;
  created_at: string;
  journal_kind: JournalEntry["kind"];
  summary: string;
  lineage_marker: "" | "fork" | "import";
  payload_json: string;
  payload_sha256: string;
  previous_sha256: string;
  row_sha256: string;
}

interface SourceRow {
  sequence: number;
  source_id: string;
  imported_at: string;
  source_name: string;
  content_hash: string | null;
  learned_records: number;
  payload_json: string;
  payload_sha256: string;
  previous_sha256: string;
  row_sha256: string;
}

interface CurrentSourceRow extends SourceRow {
  evidence_fingerprint: string;
  bytes: number;
  learned_ideas: number;
  learned_concepts: number;
  learned_synapses: number;
  parameter_steps: number;
  parameters_changed: number;
}

export function trainingSourceEvidenceFingerprint(source: TrainingSource): string {
  if (source.contentHash && /^[a-f0-9]{64}$/i.test(source.contentHash)) {
    return `content:${source.contentHash.toLocaleLowerCase()}`;
  }
  if (source.blobHash && /^[a-f0-9]{64}$/i.test(source.blobHash)) {
    return `blob:${source.blobHash.toLocaleLowerCase()}`;
  }
  return `metadata:${sha256(JSON.stringify({
    name: source.name,
    kind: source.kind,
    bytes: source.bytes,
    provenanceUrl: source.provenanceUrl ?? "",
    importedAt: source.importedAt
  }))}`;
}

export interface ActivityJournalPage {
  entries: JournalLedgerEntry[];
  totalEntries: number;
  nextCursor?: string;
  rowsRead: number;
}

export interface ActivityTrainingSourcePage {
  entries: TrainingSourceLedgerEntry[];
  totalEntries: number;
  nextCursor?: string;
  rowsRead: number;
}

interface PortableCollection<T> {
  format: string;
  formatVersion: 1;
  brainId: string;
  entries: T[];
  contentSha256: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function safeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw new Error(`Brain activity ${label} is invalid.`);
  }
  return Number(value);
}

function optionalString(value: unknown, label: string): string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "string" || value.includes("\0")) {
    throw new Error(`Brain activity ${label} is invalid.`);
  }
  return value;
}

function normalizeJournal(value: unknown): JournalEntry {
  const entry = record(value);
  if (
    !entry ||
    typeof entry.id !== "string" ||
    !SAFE_ENTRY_ID.test(entry.id) ||
    typeof entry.createdAt !== "string" ||
    !Number.isFinite(Date.parse(entry.createdAt)) ||
    typeof entry.kind !== "string" ||
    !JOURNAL_KINDS.has(entry.kind) ||
    typeof entry.summary !== "string" ||
    entry.summary.includes("\0") ||
    (entry.detail !== undefined &&
      (typeof entry.detail !== "string" || entry.detail.includes("\0")))
  ) {
    throw new Error("Brain activity journal entry is invalid.");
  }
  return {
    id: entry.id,
    createdAt: entry.createdAt,
    kind: entry.kind as JournalEntry["kind"],
    summary: entry.summary,
    ...(typeof entry.detail === "string" ? { detail: entry.detail } : {})
  };
}

function normalizeSource(value: unknown): TrainingSource {
  const source = record(value);
  if (
    !source ||
    typeof source.id !== "string" ||
    !SAFE_ENTRY_ID.test(source.id) ||
    typeof source.name !== "string" ||
    source.name.includes("\0") ||
    typeof source.kind !== "string" ||
    !SOURCE_KINDS.has(source.kind) ||
    typeof source.importedAt !== "string" ||
    !Number.isFinite(Date.parse(source.importedAt)) ||
    typeof source.rawTextRetained !== "boolean" ||
    (source.parametersChanged !== undefined && typeof source.parametersChanged !== "boolean") ||
    (source.policy !== undefined &&
      (typeof source.policy !== "string" || !SOURCE_POLICIES.has(source.policy)))
  ) {
    throw new Error("Brain activity training source is invalid.");
  }
  const path = optionalString(source.path, "training source path");
  const rawText = optionalString(source.rawText, "training source raw text");
  const contentHash = optionalString(source.contentHash, "training source content hash");
  const blobHash = optionalString(source.blobHash, "training source blob hash");
  const provenanceUrl = optionalString(
    source.provenanceUrl,
    "training source provenance URL"
  );
  const license = optionalString(source.license, "training source license");
  const licenseUrl = optionalString(source.licenseUrl, "training source license URL");
  return {
    id: source.id,
    name: source.name,
    ...(path !== undefined ? { path } : {}),
    kind: source.kind as TrainingSource["kind"],
    bytes: safeInteger(source.bytes, "training source bytes"),
    learnedIdeas: safeInteger(source.learnedIdeas, "learned ideas"),
    learnedConcepts: safeInteger(source.learnedConcepts, "learned concepts"),
    learnedSynapses: safeInteger(source.learnedSynapses, "learned synapses"),
    ...(source.learnedRecords !== undefined
      ? { learnedRecords: safeInteger(source.learnedRecords, "learned records") }
      : {}),
    ...(source.learnedParameterSteps !== undefined
      ? {
          learnedParameterSteps: safeInteger(
            source.learnedParameterSteps,
            "learned parameter steps"
          )
        }
      : {}),
    ...(typeof source.parametersChanged === "boolean"
      ? { parametersChanged: source.parametersChanged }
      : {}),
    importedAt: source.importedAt,
    rawTextRetained: source.rawTextRetained,
    ...(rawText !== undefined ? { rawText } : {}),
    ...(contentHash !== undefined ? { contentHash } : {}),
    ...(blobHash !== undefined ? { blobHash } : {}),
    ...(typeof source.policy === "string"
      ? { policy: source.policy as TrainingSource["policy"] }
      : {}),
    ...(provenanceUrl !== undefined ? { provenanceUrl } : {}),
    ...(license !== undefined ? { license } : {}),
    ...(licenseUrl !== undefined ? { licenseUrl } : {})
  };
}

function journalRowHash(row: Omit<JournalRow, "payload_json" | "row_sha256">): string {
  return sha256(JSON.stringify({
    sequence: row.sequence,
    entryId: row.entry_id,
    createdAt: row.created_at,
    kind: row.journal_kind,
    payloadSha256: row.payload_sha256,
    previousSha256: row.previous_sha256
  }));
}

function sourceRowHash(row: Omit<SourceRow, "payload_json" | "row_sha256">): string {
  return sha256(JSON.stringify({
    sequence: row.sequence,
    sourceId: row.source_id,
    importedAt: row.imported_at,
    payloadSha256: row.payload_sha256,
    previousSha256: row.previous_sha256
  }));
}

function presentJournal(row: JournalRow): JournalLedgerEntry {
  if (
    sha256(row.payload_json) !== row.payload_sha256 ||
    journalRowHash(row) !== row.row_sha256
  ) {
    throw new Error("Brain activity journal row checksum failed.");
  }
  const entry = normalizeJournal(JSON.parse(row.payload_json) as unknown);
  if (
    entry.id !== row.entry_id ||
    entry.createdAt !== row.created_at ||
    entry.kind !== row.journal_kind ||
    entry.summary !== row.summary
  ) {
    throw new Error("Brain activity journal row identity failed.");
  }
  return {
    sequence: row.sequence,
    payloadSha256: row.payload_sha256,
    rowSha256: row.row_sha256,
    entry
  };
}

function presentSource(row: SourceRow): TrainingSourceLedgerEntry {
  if (
    sha256(row.payload_json) !== row.payload_sha256 ||
    sourceRowHash(row) !== row.row_sha256
  ) {
    throw new Error("Brain activity training-source row checksum failed.");
  }
  const source = normalizeSource(JSON.parse(row.payload_json) as unknown);
  if (
    source.id !== row.source_id ||
    source.importedAt !== row.imported_at ||
    source.name !== row.source_name ||
    (source.contentHash ?? null) !== row.content_hash ||
    (source.learnedRecords ?? 0) !== row.learned_records
  ) {
    throw new Error("Brain activity training-source row identity failed.");
  }
  const projectedFingerprint = (row as Partial<CurrentSourceRow>).evidence_fingerprint;
  if (
    projectedFingerprint !== undefined &&
    projectedFingerprint !== trainingSourceEvidenceFingerprint(source)
  ) {
    throw new Error("Brain activity training-source fingerprint failed.");
  }
  return {
    sequence: row.sequence,
    payloadSha256: row.payload_sha256,
    rowSha256: row.row_sha256,
    source
  };
}

function collectionDigest<T>(body: Omit<PortableCollection<T>, "contentSha256">): string {
  return sha256(JSON.stringify(body));
}

function parseCollection<T>(
  value: unknown,
  format: string,
  expectedBrainId: string,
  normalize: (entry: unknown) => T
): T[] {
  const source = record(value);
  if (
    !source ||
    source.format !== format ||
    source.formatVersion !== 1 ||
    source.brainId !== expectedBrainId ||
    !Array.isArray(source.entries) ||
    typeof source.contentSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(source.contentSha256)
  ) {
    throw new Error("Portable brain activity collection is invalid.");
  }
  const entries = source.entries.map(normalize);
  const body = { format, formatVersion: 1 as const, brainId: expectedBrainId, entries };
  if (collectionDigest(body) !== source.contentSha256) {
    throw new Error("Portable brain activity collection checksum failed.");
  }
  return entries;
}

export function parseJournalExport(value: unknown, brainId: string): JournalEntry[] {
  return parseCollection(value, JOURNAL_EXPORT_FORMAT, brainId, normalizeJournal);
}

export function parseTrainingSourceExport(value: unknown, brainId: string): TrainingSource[] {
  return parseCollection(value, SOURCE_EXPORT_FORMAT, brainId, normalizeSource);
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function checkedAdd(left: number, right: number, label: string): number {
  const value = left + right;
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`Brain activity ${label} exceeds safe integer storage.`);
  }
  return value;
}

function configure(database: DatabaseSync): void {
  database.exec(`
    PRAGMA journal_mode=DELETE;
    PRAGMA synchronous=FULL;
    PRAGMA busy_timeout=5000;
  `);
}

function migrateDerivedProjection(database: DatabaseSync): void {
  const columns = new Set(
    (
      database.prepare("PRAGMA table_info(training_source_current)").all() as unknown as Array<{
        name?: string;
      }>
    ).map((row) => String(row.name ?? ""))
  );
  const indices = new Set(
    (
      database.prepare("PRAGMA index_list(training_source_current)").all() as unknown as Array<{
        name?: string;
      }>
    ).map((row) => String(row.name ?? ""))
  );
  const currentVersion = Number((database.prepare("PRAGMA user_version").get() as {
    user_version?: number;
  } | undefined)?.user_version ?? 0);
  const blankFingerprint = columns.has("evidence_fingerprint")
    ? database.prepare(
        "SELECT 1 present FROM training_source_current " +
        "WHERE evidence_fingerprint='' LIMIT 1"
      ).get()
    : undefined;
  if (
    columns.has("evidence_fingerprint") &&
    indices.has("training_source_current_fingerprint") &&
    !blankFingerprint &&
    currentVersion >= 2
  ) return;
  database.exec("BEGIN IMMEDIATE");
  try {
    // `training_source_current` is a rebuildable projection over the immutable
    // hash chain. Older stable-v1 ledgers predate this lookup column, so add
    // and backfill it transactionally without renumbering or rewriting any
    // version row.
    if (!columns.has("evidence_fingerprint")) {
      database.exec(
        "ALTER TABLE training_source_current " +
          "ADD COLUMN evidence_fingerprint TEXT NOT NULL DEFAULT ''"
      );
    }
    const rows = database.prepare(`
      SELECT sequence,source_id,imported_at,source_name,content_hash,learned_records,
        payload_json,payload_sha256,previous_sha256,row_sha256
      FROM training_source_current
      WHERE evidence_fingerprint=''
      ORDER BY sequence
    `).iterate() as unknown as Iterable<SourceRow>;
    const update = database.prepare(
      "UPDATE training_source_current SET evidence_fingerprint=? " +
      "WHERE source_id=? AND sequence=? AND evidence_fingerprint=''"
    );
    for (const row of rows) {
      const source = presentSource(row).source;
      const result = update.run(
        trainingSourceEvidenceFingerprint(source),
        row.source_id,
        row.sequence
      );
      if (result.changes !== 1) {
        throw new Error("Brain activity fingerprint migration lost its row lock.");
      }
    }
    database.exec(
      "CREATE INDEX IF NOT EXISTS training_source_current_fingerprint " +
        "ON training_source_current(evidence_fingerprint)"
    );
    database.exec("PRAGMA user_version=2");
    database.exec("COMMIT");
  } catch (error) {
    database.exec("ROLLBACK");
    throw error;
  }
}

const INITIAL_META: Array<[string, string]> = [
  ["format", FORMAT],
  ["formatVersion", String(FORMAT_VERSION)],
  ["brainId", ""],
  ["journalCount", "0"],
  ["journalHeadSequence", "0"],
  ["journalHeadSha256", ZERO_HASH],
  ["trainingSourceCount", "0"],
  ["trainingSourceVersionCount", "0"],
  ["trainingSourceHeadSequence", "0"],
  ["trainingSourceHeadSha256", ZERO_HASH],
  ["trainingSourceBytes", "0"],
  ["learnedIdeas", "0"],
  ["learnedConcepts", "0"],
  ["learnedSynapses", "0"],
  ["learnedRecords", "0"],
  ["learnedParameterSteps", "0"],
  ["parametersChangedSources", "0"]
];

function createDatabase(path: string, brainId: string): DatabaseSync {
  const database = new DatabaseSync(path);
  configure(database);
  database.exec(`
    CREATE TABLE meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    CREATE TABLE journal_entries (
      sequence INTEGER PRIMARY KEY,
      entry_id TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL,
      journal_kind TEXT NOT NULL,
      summary TEXT NOT NULL,
      lineage_marker TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      previous_sha256 TEXT NOT NULL,
      row_sha256 TEXT NOT NULL UNIQUE
    );
    CREATE INDEX journal_kind_sequence
      ON journal_entries(journal_kind, sequence DESC);
    CREATE INDEX journal_lineage_sequence
      ON journal_entries(lineage_marker, sequence DESC);
    CREATE TABLE training_source_versions (
      sequence INTEGER PRIMARY KEY,
      source_id TEXT NOT NULL,
      imported_at TEXT NOT NULL,
      source_name TEXT NOT NULL,
      content_hash TEXT,
      learned_records INTEGER NOT NULL,
      payload_json TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      previous_sha256 TEXT NOT NULL,
      row_sha256 TEXT NOT NULL UNIQUE
    );
    CREATE INDEX training_source_versions_id_sequence
      ON training_source_versions(source_id, sequence DESC);
    CREATE TABLE training_source_current (
      source_id TEXT PRIMARY KEY,
      sequence INTEGER NOT NULL UNIQUE,
      imported_at TEXT NOT NULL,
      source_name TEXT NOT NULL,
      content_hash TEXT,
      evidence_fingerprint TEXT NOT NULL,
      learned_records INTEGER NOT NULL,
      bytes INTEGER NOT NULL,
      learned_ideas INTEGER NOT NULL,
      learned_concepts INTEGER NOT NULL,
      learned_synapses INTEGER NOT NULL,
      parameter_steps INTEGER NOT NULL,
      parameters_changed INTEGER NOT NULL,
      payload_json TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      previous_sha256 TEXT NOT NULL,
      row_sha256 TEXT NOT NULL,
      FOREIGN KEY(sequence) REFERENCES training_source_versions(sequence)
    );
    CREATE INDEX training_source_current_content
      ON training_source_current(content_hash, sequence DESC);
    CREATE INDEX training_source_current_fingerprint
      ON training_source_current(evidence_fingerprint);
    CREATE INDEX training_source_current_sequence
      ON training_source_current(sequence DESC);
    CREATE INDEX training_source_current_adaptation
      ON training_source_current(learned_records DESC, imported_at DESC, sequence DESC);
    PRAGMA user_version=2;
  `);
  const insert = database.prepare("INSERT INTO meta(key,value) VALUES(?,?)");
  for (const [key, initial] of INITIAL_META) {
    insert.run(key, key === "brainId" ? brainId : initial);
  }
  return database;
}

function metaMap(database: DatabaseSync): Map<string, string> {
  const rows = database.prepare("SELECT key,value FROM meta").all() as unknown as Array<{
    key: string;
    value: string;
  }>;
  return new Map(rows.map((row) => [row.key, row.value]));
}

function metaInteger(meta: Map<string, string>, key: string): number {
  const value = Number(meta.get(key));
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error("Brain activity ledger summary is invalid.");
  }
  return value;
}

function summary(database: DatabaseSync, brainId: string): BrainActivityLedgerSummary {
  const meta = metaMap(database);
  if (
    meta.get("format") !== FORMAT ||
    meta.get("formatVersion") !== String(FORMAT_VERSION) ||
    meta.get("brainId") !== brainId ||
    !/^[a-f0-9]{64}$/.test(meta.get("journalHeadSha256") ?? "") ||
    !/^[a-f0-9]{64}$/.test(meta.get("trainingSourceHeadSha256") ?? "")
  ) {
    throw new Error("Brain activity ledger identity is invalid.");
  }
  const journalCount = metaInteger(meta, "journalCount");
  const journalHeadSequence = metaInteger(meta, "journalHeadSequence");
  const trainingSourceCount = metaInteger(meta, "trainingSourceCount");
  const trainingSourceVersionCount = metaInteger(meta, "trainingSourceVersionCount");
  const trainingSourceHeadSequence = metaInteger(meta, "trainingSourceHeadSequence");
  if (
    journalCount !== journalHeadSequence ||
    trainingSourceVersionCount !== trainingSourceHeadSequence ||
    trainingSourceCount > trainingSourceVersionCount
  ) {
    throw new Error("Brain activity ledger counts are inconsistent.");
  }
  const top = database.prepare(`
    SELECT source_id,source_name,learned_records
    FROM training_source_current
    WHERE learned_records > 0
    ORDER BY learned_records DESC, imported_at DESC, sequence DESC
    LIMIT 1
  `).get() as { source_id?: string; source_name?: string; learned_records?: number } | undefined;
  return {
    format: FORMAT,
    formatVersion: 1,
    journalCount,
    journalHeadSequence,
    journalHeadSha256: meta.get("journalHeadSha256")!,
    trainingSourceCount,
    trainingSourceVersionCount,
    trainingSourceHeadSequence,
    trainingSourceHeadSha256: meta.get("trainingSourceHeadSha256")!,
    trainingSourceBytes: metaInteger(meta, "trainingSourceBytes"),
    learnedIdeas: metaInteger(meta, "learnedIdeas"),
    learnedConcepts: metaInteger(meta, "learnedConcepts"),
    learnedSynapses: metaInteger(meta, "learnedSynapses"),
    learnedRecords: metaInteger(meta, "learnedRecords"),
    learnedParameterSteps: metaInteger(meta, "learnedParameterSteps"),
    parametersChangedSources: metaInteger(meta, "parametersChangedSources"),
    ...(top?.source_id && top.source_name && Number(top.learned_records) > 0
      ? {
          topAdaptation: {
            sourceId: top.source_id,
            sourceLabel: top.source_name,
            learnedRecords: Number(top.learned_records)
          }
        }
      : {})
  };
}

function openDatabase(path: string, brainId: string): DatabaseSync {
  let database: DatabaseSync | undefined;
  try {
    database = new DatabaseSync(path);
    configure(database);
    migrateDerivedProjection(database);
    summary(database, brainId);
    return database;
  } catch (error) {
    try {
      database?.close();
    } catch {
      // Best effort while unwinding a failed database open.
    }
    throw error;
  }
}

const pathLocks = new Map<string, Promise<void>>();

async function withPathLock<T>(path: string, operation: () => Promise<T>): Promise<T> {
  const previous = pathLocks.get(path) ?? Promise.resolve();
  const result = previous.catch(() => undefined).then(operation);
  const tail = result.then(() => undefined, () => undefined);
  pathLocks.set(path, tail);
  try {
    return await result;
  } finally {
    if (pathLocks.get(path) === tail) pathLocks.delete(path);
  }
}

function cursorValues(
  cursor: string | undefined,
  headSequence: number
): { beforeSequence: number; expectedFirstSha256?: string } {
  if (cursor === undefined) return { beforeSequence: headSequence + 1 };
  const match = CURSOR.exec(cursor);
  const beforeSequence = Number(match?.[1]);
  if (
    !match ||
    !Number.isSafeInteger(beforeSequence) ||
    beforeSequence < 1 ||
    beforeSequence > headSequence
  ) {
    throw new Error("Brain activity cursor is invalid.");
  }
  return { beforeSequence, expectedFirstSha256: match[2] };
}

function normalizedLimit(value: number): number {
  return Math.max(1, Math.min(100, Math.floor(value)));
}

export class BrainActivityLedger {
  private constructor(
    readonly brainDirectory: string,
    readonly brainId: string,
    readonly path: string,
    private readonly database: DatabaseSync
  ) {}

  static directory(brainDirectory: string): string {
    return join(brainDirectory, ACTIVITY_LEDGER_DIRECTORY);
  }

  static databasePath(brainDirectory: string): string {
    return join(this.directory(brainDirectory), ACTIVITY_LEDGER_DATABASE);
  }

  static async open(brainDirectory: string, brainId: string): Promise<BrainActivityLedger> {
    if (!SAFE_BRAIN_ID.test(brainId)) throw new Error("Invalid brain activity identity.");
    const directory = this.directory(brainDirectory);
    const path = this.databasePath(brainDirectory);
    await mkdir(directory, { recursive: true });
    await withPathLock(path, async () => {
      if (!await pathExists(path)) {
        await this.writeAtomic(brainDirectory, brainId, [], [], false);
      }
      const probe = openDatabase(path, brainId);
      probe.close();
    });
    return new BrainActivityLedger(brainDirectory, brainId, path, openDatabase(path, brainId));
  }

  static async clone(
    sourceBrainDirectory: string,
    targetBrainDirectory: string,
    sourceBrainId: string,
    targetBrainId: string
  ): Promise<void> {
    const source = await this.open(sourceBrainDirectory, sourceBrainId);
    try {
      source.integrity();
    } finally {
      source.close();
    }
    const targetDirectory = this.directory(targetBrainDirectory);
    const targetPath = this.databasePath(targetBrainDirectory);
    await mkdir(targetDirectory, { recursive: true });
    await withPathLock(targetPath, async () => {
      if (await pathExists(targetPath)) {
        throw new Error("Target brain activity ledger already exists.");
      }
      const temporary = join(
        targetDirectory,
        `.ledger.${randomBytes(12).toString("hex")}.sqlite3.tmp`
      );
      try {
        await copyFile(this.databasePath(sourceBrainDirectory), temporary);
        const database = openDatabase(temporary, sourceBrainId);
        try {
          database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
        } finally {
          database.close();
        }
        // Windows FlushFileBuffers requires a writable handle for fsync.
        const handle = await openFile(temporary, "r+");
        try {
          await handle.sync();
        } finally {
          await handle.close();
        }
        await rename(temporary, targetPath);
      } finally {
        await Promise.all([
          rm(temporary, { force: true }),
          rm(`${temporary}-journal`, { force: true })
        ]);
      }
    });
    const cloned = await this.open(targetBrainDirectory, targetBrainId);
    try {
      cloned.integrity();
    } finally {
      cloned.close();
    }
  }

  static async replace(
    brainDirectory: string,
    brainId: string,
    journals: JournalEntry[],
    sources: TrainingSource[]
  ): Promise<void> {
    if (!SAFE_BRAIN_ID.test(brainId)) throw new Error("Invalid brain activity identity.");
    await mkdir(this.directory(brainDirectory), { recursive: true });
    const path = this.databasePath(brainDirectory);
    await withPathLock(path, () =>
      this.writeAtomic(brainDirectory, brainId, journals, sources, true)
    );
  }

  static async project(
    sourceBrainDirectory: string,
    targetBrainDirectory: string,
    sourceBrainId: string,
    targetBrainId: string,
    transformJournal: (entry: JournalEntry) => JournalEntry,
    transformSource: (source: TrainingSource) => TrainingSource
  ): Promise<BrainActivityLedgerSummary> {
    const source = await this.open(sourceBrainDirectory, sourceBrainId);
    await this.replace(targetBrainDirectory, targetBrainId, [], []);
    const target = await this.open(targetBrainDirectory, targetBrainId);
    try {
      source.integrity();
      const journalBatch: JournalEntry[] = [];
      for (const entry of source.journals()) {
        journalBatch.push(transformJournal(entry));
        if (journalBatch.length >= 100) {
          target.appendJournals(journalBatch.splice(0));
        }
      }
      if (journalBatch.length) target.appendJournals(journalBatch);
      const sourceBatch: TrainingSource[] = [];
      for (const entry of source.trainingSources()) {
        sourceBatch.push(transformSource(entry));
        if (sourceBatch.length >= 100) {
          target.upsertTrainingSources(sourceBatch.splice(0));
        }
      }
      if (sourceBatch.length) target.upsertTrainingSources(sourceBatch);
      return target.integrity();
    } finally {
      source.close();
      target.close();
    }
  }

  private static async writeAtomic(
    brainDirectory: string,
    brainId: string,
    journals: JournalEntry[],
    sources: TrainingSource[],
    replaceExisting: boolean
  ): Promise<void> {
    const directory = this.directory(brainDirectory);
    const path = this.databasePath(brainDirectory);
    const nonce = randomBytes(12).toString("hex");
    const temporary = join(directory, `.ledger.${nonce}.sqlite3.tmp`);
    const backup = join(directory, `.ledger.${nonce}.sqlite3.bak`);
    let database: DatabaseSync | undefined;
    let movedExisting = false;
    let promoted = false;
    try {
      database = createDatabase(temporary, brainId);
      const ledger = new BrainActivityLedger(brainDirectory, brainId, temporary, database);
      ledger.appendJournals(journals);
      ledger.upsertTrainingSources(sources);
      ledger.integrity();
      ledger.close();
      database = undefined;
      const handle = await openFile(temporary, "r+");
      try {
        await handle.sync();
      } finally {
        await handle.close();
      }
      if (await pathExists(path)) {
        if (!replaceExisting) return;
        await rename(path, backup);
        movedExisting = true;
      }
      try {
        await rename(temporary, path);
        promoted = true;
      } catch (error) {
        if (movedExisting) await rename(backup, path).catch(() => undefined);
        throw error;
      }
      if (movedExisting) {
        await rm(backup, { force: true });
        movedExisting = false;
      }
    } finally {
      try {
        database?.close();
      } catch {
        // Best effort while unwinding a failed database initialization.
      }
      await Promise.all([
        rm(temporary, { force: true }),
        rm(`${temporary}-journal`, { force: true })
      ]);
      if (!promoted && movedExisting && await pathExists(backup)) {
        await rename(backup, path).catch(() => undefined);
      }
    }
  }

  close(): void {
    this.database.close();
  }

  getSummary(): BrainActivityLedgerSummary {
    return summary(this.database, this.brainId);
  }

  appendJournals(values: JournalEntry[]): BrainActivityLedgerSummary {
    if (!values.length) return this.getSummary();
    const entries = values.map(normalizeJournal);
    const current = this.getSummary();
    let sequence = current.journalHeadSequence;
    let previous = current.journalHeadSha256;
    let inserted = 0;
    const existing = this.database.prepare(
      "SELECT payload_sha256 FROM journal_entries WHERE entry_id=?"
    );
    const insert = this.database.prepare(`
      INSERT INTO journal_entries(
        sequence,entry_id,created_at,journal_kind,summary,lineage_marker,
        payload_json,payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?,?,?,?,?)
    `);
    const updateMeta = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const entry of entries) {
        const payloadJson = JSON.stringify(entry);
        const payloadSha256 = sha256(payloadJson);
        const prior = existing.get(entry.id) as { payload_sha256?: string } | undefined;
        if (prior) {
          if (prior.payload_sha256 !== payloadSha256) {
            throw new Error("Brain activity journal idempotency conflict.");
          }
          continue;
        }
        sequence += 1;
        const marker = entry.kind === "fork"
          ? "fork"
          : /^Imported from /i.test(entry.summary)
            ? "import"
            : "";
        const base: Omit<JournalRow, "row_sha256"> = {
          sequence,
          entry_id: entry.id,
          created_at: entry.createdAt,
          journal_kind: entry.kind,
          summary: entry.summary,
          lineage_marker: marker,
          payload_json: payloadJson,
          payload_sha256: payloadSha256,
          previous_sha256: previous
        };
        const digest = journalRowHash(base);
        insert.run(
          sequence,
          entry.id,
          entry.createdAt,
          entry.kind,
          entry.summary,
          marker,
          payloadJson,
          payloadSha256,
          previous,
          digest
        );
        previous = digest;
        inserted += 1;
      }
      if (inserted) {
        updateMeta.run(String(current.journalCount + inserted), "journalCount");
        updateMeta.run(String(sequence), "journalHeadSequence");
        updateMeta.run(previous, "journalHeadSha256");
      }
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.getSummary();
  }

  upsertTrainingSources(values: TrainingSource[]): BrainActivityLedgerSummary {
    if (!values.length) return this.getSummary();
    const sources = values.map(normalizeSource);
    const current = this.getSummary();
    let sequence = current.trainingSourceHeadSequence;
    let previous = current.trainingSourceHeadSha256;
    let sourceCount = current.trainingSourceCount;
    let versionCount = current.trainingSourceVersionCount;
    const totals = {
      trainingSourceBytes: current.trainingSourceBytes,
      learnedIdeas: current.learnedIdeas,
      learnedConcepts: current.learnedConcepts,
      learnedSynapses: current.learnedSynapses,
      learnedRecords: current.learnedRecords,
      learnedParameterSteps: current.learnedParameterSteps,
      parametersChangedSources: current.parametersChangedSources
    };
    const existing = this.database.prepare(
      "SELECT * FROM training_source_current WHERE source_id=?"
    );
    const insertVersion = this.database.prepare(`
      INSERT INTO training_source_versions(
        sequence,source_id,imported_at,source_name,content_hash,learned_records,
        payload_json,payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?,?,?,?,?)
    `);
    const upsertCurrent = this.database.prepare(`
      INSERT INTO training_source_current(
        source_id,sequence,imported_at,source_name,content_hash,evidence_fingerprint,learned_records,
        bytes,learned_ideas,learned_concepts,learned_synapses,parameter_steps,
        parameters_changed,payload_json,payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      ON CONFLICT(source_id) DO UPDATE SET
        sequence=excluded.sequence,
        imported_at=excluded.imported_at,
        source_name=excluded.source_name,
        content_hash=excluded.content_hash,
        evidence_fingerprint=excluded.evidence_fingerprint,
        learned_records=excluded.learned_records,
        bytes=excluded.bytes,
        learned_ideas=excluded.learned_ideas,
        learned_concepts=excluded.learned_concepts,
        learned_synapses=excluded.learned_synapses,
        parameter_steps=excluded.parameter_steps,
        parameters_changed=excluded.parameters_changed,
        payload_json=excluded.payload_json,
        payload_sha256=excluded.payload_sha256,
        previous_sha256=excluded.previous_sha256,
        row_sha256=excluded.row_sha256
    `);
    const updateMeta = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const source of sources) {
        const payloadJson = JSON.stringify(source);
        const payloadSha256 = sha256(payloadJson);
        const prior = existing.get(source.id) as CurrentSourceRow | undefined;
        if (prior?.payload_sha256 === payloadSha256) continue;
        const valuesByField = {
          trainingSourceBytes: source.bytes,
          learnedIdeas: source.learnedIdeas,
          learnedConcepts: source.learnedConcepts,
          learnedSynapses: source.learnedSynapses,
          learnedRecords: source.learnedRecords ?? 0,
          learnedParameterSteps: source.learnedParameterSteps ?? 0,
          parametersChangedSources: source.parametersChanged ? 1 : 0
        };
        const priorByField = {
          trainingSourceBytes: Number(prior?.bytes ?? 0),
          learnedIdeas: Number(prior?.learned_ideas ?? 0),
          learnedConcepts: Number(prior?.learned_concepts ?? 0),
          learnedSynapses: Number(prior?.learned_synapses ?? 0),
          learnedRecords: Number(prior?.learned_records ?? 0),
          learnedParameterSteps: Number(prior?.parameter_steps ?? 0),
          parametersChangedSources: Number(prior?.parameters_changed ?? 0)
        };
        for (const key of Object.keys(totals) as Array<keyof typeof totals>) {
          totals[key] = checkedAdd(
            totals[key] - priorByField[key],
            valuesByField[key],
            key
          );
        }
        sequence += 1;
        versionCount += 1;
        if (!prior) sourceCount += 1;
        const base: Omit<SourceRow, "row_sha256"> = {
          sequence,
          source_id: source.id,
          imported_at: source.importedAt,
          source_name: source.name,
          content_hash: source.contentHash ?? null,
          learned_records: source.learnedRecords ?? 0,
          payload_json: payloadJson,
          payload_sha256: payloadSha256,
          previous_sha256: previous
        };
        const digest = sourceRowHash(base);
        insertVersion.run(
          sequence,
          source.id,
          source.importedAt,
          source.name,
          source.contentHash ?? null,
          source.learnedRecords ?? 0,
          payloadJson,
          payloadSha256,
          previous,
          digest
        );
        upsertCurrent.run(
          source.id,
          sequence,
          source.importedAt,
          source.name,
          source.contentHash ?? null,
          trainingSourceEvidenceFingerprint(source),
          source.learnedRecords ?? 0,
          source.bytes,
          source.learnedIdeas,
          source.learnedConcepts,
          source.learnedSynapses,
          source.learnedParameterSteps ?? 0,
          source.parametersChanged ? 1 : 0,
          payloadJson,
          payloadSha256,
          previous,
          digest
        );
        previous = digest;
      }
      updateMeta.run(String(sourceCount), "trainingSourceCount");
      updateMeta.run(String(versionCount), "trainingSourceVersionCount");
      updateMeta.run(String(sequence), "trainingSourceHeadSequence");
      updateMeta.run(previous, "trainingSourceHeadSha256");
      for (const [key, value] of Object.entries(totals)) {
        updateMeta.run(String(value), key);
      }
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.getSummary();
  }

  journalPage(cursor?: string, limitValue = 80): ActivityJournalPage {
    const state = this.getSummary();
    const limit = normalizedLimit(limitValue);
    const { beforeSequence, expectedFirstSha256 } = cursorValues(
      cursor,
      state.journalHeadSequence
    );
    const rows = this.database.prepare(`
      SELECT * FROM journal_entries
      WHERE sequence < ? ORDER BY sequence DESC LIMIT ?
    `).all(beforeSequence, limit + 1) as unknown as JournalRow[];
    if (cursor === undefined && rows[0] && (
      rows[0].sequence !== state.journalHeadSequence ||
      rows[0].row_sha256 !== state.journalHeadSha256
    )) {
      throw new Error("Brain activity journal head checksum failed.");
    }
    if (expectedFirstSha256 && rows[0]?.row_sha256 !== expectedFirstSha256) {
      throw new Error("Brain activity journal cursor checksum failed.");
    }
    for (let index = 0; index < rows.length - 1; index += 1) {
      if (
        rows[index]!.sequence !== rows[index + 1]!.sequence + 1 ||
        rows[index]!.previous_sha256 !== rows[index + 1]!.row_sha256
      ) {
        throw new Error("Brain activity journal page chain failed.");
      }
    }
    const selected = rows.slice(0, limit);
    const oldest = selected.at(-1);
    return {
      entries: selected.map(presentJournal).reverse(),
      totalEntries: state.journalCount,
      rowsRead: rows.length,
      ...(rows.length > limit && oldest
        ? { nextCursor: `${oldest.sequence}.${oldest.previous_sha256}` }
        : {})
    };
  }

  trainingSourcePage(cursor?: string, limitValue = 80): ActivityTrainingSourcePage {
    const state = this.getSummary();
    const limit = normalizedLimit(limitValue);
    const { beforeSequence, expectedFirstSha256 } = cursorValues(
      cursor,
      state.trainingSourceHeadSequence
    );
    const rows = this.database.prepare(`
      SELECT * FROM training_source_current
      WHERE sequence < ? ORDER BY sequence DESC LIMIT ?
    `).all(beforeSequence, limit + 1) as unknown as CurrentSourceRow[];
    if (cursor === undefined && rows[0] && (
      rows[0].sequence !== state.trainingSourceHeadSequence ||
      rows[0].row_sha256 !== state.trainingSourceHeadSha256
    )) {
      throw new Error("Brain activity training-source head checksum failed.");
    }
    if (expectedFirstSha256 && rows[0]?.row_sha256 !== expectedFirstSha256) {
      throw new Error("Brain activity training-source cursor checksum failed.");
    }
    const selected = rows.slice(0, limit);
    const oldest = selected.at(-1);
    const lookahead = rows[limit];
    return {
      entries: selected.map(presentSource).reverse(),
      totalEntries: state.trainingSourceCount,
      rowsRead: rows.length,
      ...(lookahead && oldest
        ? { nextCursor: `${oldest.sequence}.${lookahead.row_sha256}` }
        : {})
    };
  }

  journalById(id: string): JournalEntry | undefined {
    if (!SAFE_ENTRY_ID.test(id)) throw new Error("Invalid journal entry id.");
    const row = this.database.prepare(
      "SELECT * FROM journal_entries WHERE entry_id=?"
    ).get(id) as JournalRow | undefined;
    return row ? presentJournal(row).entry : undefined;
  }

  trainingSourceById(id: string): TrainingSource | undefined {
    if (!SAFE_ENTRY_ID.test(id)) throw new Error("Invalid training source id.");
    const row = this.database.prepare(
      "SELECT * FROM training_source_current WHERE source_id=?"
    ).get(id) as CurrentSourceRow | undefined;
    return row ? presentSource(row).source : undefined;
  }

  trainingSourceByContentHash(contentHash: string): TrainingSource | undefined {
    if (!/^[a-f0-9]{64}$/i.test(contentHash)) return undefined;
    const row = this.database.prepare(`
      SELECT * FROM training_source_current
      WHERE content_hash=? ORDER BY sequence DESC LIMIT 1
    `).get(contentHash.toLocaleLowerCase()) as CurrentSourceRow | undefined;
    return row ? presentSource(row).source : undefined;
  }

  trainingSourceByFingerprint(fingerprint: string): TrainingSource | undefined {
    if (!/^(?:content|blob|metadata):[a-f0-9]{64}$/i.test(fingerprint)) {
      throw new Error("Invalid training source fingerprint.");
    }
    const row = this.database.prepare(`
      SELECT * FROM training_source_current
      WHERE evidence_fingerprint=? LIMIT 1
    `).get(fingerprint.toLocaleLowerCase()) as CurrentSourceRow | undefined;
    return row ? presentSource(row).source : undefined;
  }

  latestLineageJournal(): JournalEntry | undefined {
    const row = this.database.prepare(`
      SELECT * FROM journal_entries
      WHERE lineage_marker IN ('fork','import')
      ORDER BY sequence DESC LIMIT 1
    `).get() as JournalRow | undefined;
    return row ? presentJournal(row).entry : undefined;
  }

  *journals(): IterableIterator<JournalEntry> {
    const rows = this.database.prepare(
      "SELECT * FROM journal_entries ORDER BY sequence"
    ).iterate() as unknown as Iterable<JournalRow>;
    for (const row of rows) yield presentJournal(row).entry;
  }

  *trainingSources(): IterableIterator<TrainingSource> {
    const rows = this.database.prepare(
      "SELECT * FROM training_source_current ORDER BY sequence"
    ).iterate() as unknown as Iterable<CurrentSourceRow>;
    for (const row of rows) yield presentSource(row).source;
  }

  integrity(): BrainActivityLedgerSummary {
    const quick = this.database.prepare("PRAGMA quick_check").get() as
      { quick_check?: string } | undefined;
    if (quick?.quick_check !== "ok") {
      throw new Error("Brain activity SQLite integrity failed.");
    }
    const journals = this.database.prepare(
      "SELECT * FROM journal_entries ORDER BY sequence"
    ).iterate() as unknown as Iterable<JournalRow>;
    let journalPrevious = ZERO_HASH;
    let journalCount = 0;
    for (const row of journals) {
      journalCount += 1;
      if (row.sequence !== journalCount || row.previous_sha256 !== journalPrevious) {
        throw new Error("Brain activity journal hash chain failed.");
      }
      presentJournal(row);
      journalPrevious = row.row_sha256;
    }
    const versions = this.database.prepare(
      "SELECT * FROM training_source_versions ORDER BY sequence"
    ).iterate() as unknown as Iterable<SourceRow>;
    let sourcePrevious = ZERO_HASH;
    let sourceVersionCount = 0;
    for (const row of versions) {
      sourceVersionCount += 1;
      if (row.sequence !== sourceVersionCount || row.previous_sha256 !== sourcePrevious) {
        throw new Error("Brain activity training-source hash chain failed.");
      }
      presentSource(row);
      sourcePrevious = row.row_sha256;
    }
    const currentRows = this.database.prepare(
      "SELECT * FROM training_source_current ORDER BY sequence"
    ).iterate() as unknown as Iterable<CurrentSourceRow>;
    const totals = {
      trainingSourceBytes: 0,
      learnedIdeas: 0,
      learnedConcepts: 0,
      learnedSynapses: 0,
      learnedRecords: 0,
      learnedParameterSteps: 0,
      parametersChangedSources: 0
    };
    const latestVersion = this.database.prepare(`
      SELECT sequence,row_sha256,payload_sha256
      FROM training_source_versions
      WHERE source_id=? ORDER BY sequence DESC LIMIT 1
    `);
    let sourceCount = 0;
    for (const row of currentRows) {
      sourceCount += 1;
      const version = latestVersion.get(row.source_id) as {
        sequence?: number;
        row_sha256?: string;
        payload_sha256?: string;
      } | undefined;
      if (
        !version ||
        version.sequence !== row.sequence ||
        version.row_sha256 !== row.row_sha256 ||
        version.payload_sha256 !== row.payload_sha256
      ) {
        throw new Error("Brain activity training-source projection failed integrity.");
      }
      presentSource(row);
      totals.trainingSourceBytes = checkedAdd(
        totals.trainingSourceBytes,
        Number(row.bytes),
        "training source bytes"
      );
      totals.learnedIdeas = checkedAdd(totals.learnedIdeas, Number(row.learned_ideas), "ideas");
      totals.learnedConcepts = checkedAdd(
        totals.learnedConcepts,
        Number(row.learned_concepts),
        "concepts"
      );
      totals.learnedSynapses = checkedAdd(
        totals.learnedSynapses,
        Number(row.learned_synapses),
        "synapses"
      );
      totals.learnedRecords = checkedAdd(
        totals.learnedRecords,
        Number(row.learned_records),
        "records"
      );
      totals.learnedParameterSteps = checkedAdd(
        totals.learnedParameterSteps,
        Number(row.parameter_steps),
        "parameter steps"
      );
      totals.parametersChangedSources = checkedAdd(
        totals.parametersChangedSources,
        Number(row.parameters_changed),
        "parameters-changed sources"
      );
    }
    const state = this.getSummary();
    if (
      state.journalCount !== journalCount ||
      state.journalHeadSha256 !== journalPrevious ||
      state.trainingSourceVersionCount !== sourceVersionCount ||
      state.trainingSourceHeadSha256 !== sourcePrevious ||
      state.trainingSourceCount !== sourceCount ||
      (Object.keys(totals) as Array<keyof typeof totals>).some(
        (key) => state[key] !== totals[key]
      )
    ) {
      throw new Error("Brain activity ledger summary checksum failed.");
    }
    return state;
  }
}
