import { createHash, randomBytes } from "node:crypto";
import { open as openFile } from "node:fs/promises";
import { access, mkdir, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { TrainingSource } from "../shared/types";

const FORMAT = "omni-merge-evidence-plan";
const FORMAT_VERSION = 1;
const ZERO_HASH = "0".repeat(64);
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SAFE_TOKEN = /^[a-f0-9]{64}$/;

export interface MergeEvidencePlanIdentity {
  sourceBrainId: string;
  targetBrainId: string;
  sourceUpdatedAt: string;
  targetUpdatedAt: string;
  substrateDigest: string;
}

export interface MergeEvidencePlanEntry {
  sourceSequence: number;
  fingerprint: string;
  targetSourceId: string;
  source: TrainingSource;
}

export interface MergeEvidencePlanRow extends MergeEvidencePlanEntry {
  sequence: number;
  payloadSha256: string;
  rowSha256: string;
}

export interface MergeEvidencePlanSummary extends MergeEvidencePlanIdentity {
  reviewToken?: string;
  descriptorSha256?: string;
  evidenceCount: number;
  evidenceHeadSha256: string;
  cursorSequence: number;
  mergedCount: number;
  state: "building" | "ready" | "running" | "paused" | "complete";
  pauseReason?: string;
  diskFreeBytes?: number;
  diskReserveBytes?: number;
  updatedAt: string;
}

export interface MergeEvidencePlanPage {
  entries: MergeEvidencePlanRow[];
  rowsRead: number;
  totalEntries: number;
  nextSequence?: number;
}

export interface MergeEvidenceResourceCheckpoint {
  diskFreeBytes: number;
  diskReserveBytes: number;
  incomingWriteBytes: number;
  memoryHeadroom: boolean;
}

interface StoredRow {
  sequence: number;
  source_sequence: number;
  fingerprint: string;
  target_source_id: string;
  payload_json: string;
  payload_sha256: string;
  previous_sha256: string;
  row_sha256: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function safeInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 0) {
    throw new Error(`Merge evidence ${label} is invalid.`);
  }
  return Number(value);
}

function validateIdentity(value: MergeEvidencePlanIdentity): MergeEvidencePlanIdentity {
  if (
    !SAFE_ID.test(value.sourceBrainId) ||
    !SAFE_ID.test(value.targetBrainId) ||
    value.sourceBrainId === value.targetBrainId ||
    !Number.isFinite(Date.parse(value.sourceUpdatedAt)) ||
    !Number.isFinite(Date.parse(value.targetUpdatedAt)) ||
    !SAFE_TOKEN.test(value.substrateDigest)
  ) {
    throw new Error("Merge evidence plan identity is invalid.");
  }
  return { ...value };
}

function normalizeSource(value: unknown): TrainingSource {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Merge evidence source payload is invalid.");
  }
  const source = value as TrainingSource;
  if (
    !SAFE_ID.test(source.id) ||
    typeof source.name !== "string" ||
    !source.name ||
    typeof source.kind !== "string" ||
    !Number.isSafeInteger(source.bytes) ||
    source.bytes < 0 ||
    !Number.isSafeInteger(source.learnedIdeas) ||
    source.learnedIdeas < 0 ||
    !Number.isSafeInteger(source.learnedConcepts) ||
    source.learnedConcepts < 0 ||
    !Number.isSafeInteger(source.learnedSynapses) ||
    source.learnedSynapses < 0 ||
    !Number.isFinite(Date.parse(source.importedAt)) ||
    typeof source.rawTextRetained !== "boolean"
  ) {
    throw new Error("Merge evidence source payload is invalid.");
  }
  return JSON.parse(JSON.stringify(source)) as TrainingSource;
}

function normalizeEntry(value: MergeEvidencePlanEntry): MergeEvidencePlanEntry {
  if (
    !Number.isSafeInteger(value.sourceSequence) ||
    value.sourceSequence < 1 ||
    !/^(?:content|blob|metadata):[a-f0-9]{64}$/i.test(value.fingerprint) ||
    !SAFE_ID.test(value.targetSourceId)
  ) {
    throw new Error("Merge evidence row identity is invalid.");
  }
  return {
    sourceSequence: value.sourceSequence,
    fingerprint: value.fingerprint.toLocaleLowerCase(),
    targetSourceId: value.targetSourceId,
    source: normalizeSource(value.source)
  };
}

function rowHash(row: Omit<StoredRow, "payload_json" | "row_sha256">): string {
  return sha256(JSON.stringify({
    sequence: row.sequence,
    sourceSequence: row.source_sequence,
    fingerprint: row.fingerprint,
    targetSourceId: row.target_source_id,
    payloadSha256: row.payload_sha256,
    previousSha256: row.previous_sha256
  }));
}

function presentRow(row: StoredRow): MergeEvidencePlanRow {
  if (
    sha256(row.payload_json) !== row.payload_sha256 ||
    rowHash(row) !== row.row_sha256
  ) {
    throw new Error("Merge evidence row checksum failed.");
  }
  const source = normalizeSource(JSON.parse(row.payload_json) as unknown);
  return {
    sequence: row.sequence,
    sourceSequence: row.source_sequence,
    fingerprint: row.fingerprint,
    targetSourceId: row.target_source_id,
    source,
    payloadSha256: row.payload_sha256,
    rowSha256: row.row_sha256
  };
}

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function configure(database: DatabaseSync): void {
  database.exec(`
    PRAGMA journal_mode=DELETE;
    PRAGMA synchronous=FULL;
    PRAGMA busy_timeout=5000;
  `);
}

function createDatabase(path: string, identity: MergeEvidencePlanIdentity): DatabaseSync {
  const database = new DatabaseSync(path);
  configure(database);
  database.exec(`
    CREATE TABLE meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    CREATE TABLE evidence (
      sequence INTEGER PRIMARY KEY,
      source_sequence INTEGER NOT NULL UNIQUE,
      fingerprint TEXT NOT NULL,
      target_source_id TEXT NOT NULL,
      payload_json TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      previous_sha256 TEXT NOT NULL,
      row_sha256 TEXT NOT NULL UNIQUE
    );
    CREATE INDEX evidence_pending_sequence ON evidence(sequence);
  `);
  const insert = database.prepare("INSERT INTO meta(key,value) VALUES(?,?)");
  const now = new Date().toISOString();
  const values: Array<[string, string]> = [
    ["format", FORMAT],
    ["formatVersion", String(FORMAT_VERSION)],
    ["sourceBrainId", identity.sourceBrainId],
    ["targetBrainId", identity.targetBrainId],
    ["sourceUpdatedAt", identity.sourceUpdatedAt],
    ["targetUpdatedAt", identity.targetUpdatedAt],
    ["substrateDigest", identity.substrateDigest],
    ["reviewToken", ""],
    ["descriptorSha256", ""],
    ["descriptorJson", ""],
    ["evidenceCount", "0"],
    ["evidenceHeadSha256", ZERO_HASH],
    ["cursorSequence", "0"],
    ["mergedCount", "0"],
    ["state", "building"],
    ["pauseReason", ""],
    ["diskFreeBytes", "0"],
    ["diskReserveBytes", "0"],
    ["updatedAt", now]
  ];
  for (const [key, value] of values) insert.run(key, value);
  return database;
}

function meta(database: DatabaseSync): Map<string, string> {
  const rows = database.prepare("SELECT key,value FROM meta").all() as unknown as Array<{
    key: string;
    value: string;
  }>;
  return new Map(rows.map((row) => [row.key, row.value]));
}

function readSummary(database: DatabaseSync): MergeEvidencePlanSummary {
  const values = meta(database);
  if (
    values.get("format") !== FORMAT ||
    values.get("formatVersion") !== String(FORMAT_VERSION)
  ) {
    throw new Error("Merge evidence plan format is invalid.");
  }
  const identity = validateIdentity({
    sourceBrainId: values.get("sourceBrainId") ?? "",
    targetBrainId: values.get("targetBrainId") ?? "",
    sourceUpdatedAt: values.get("sourceUpdatedAt") ?? "",
    targetUpdatedAt: values.get("targetUpdatedAt") ?? "",
    substrateDigest: values.get("substrateDigest") ?? ""
  });
  const evidenceCount = safeInteger(Number(values.get("evidenceCount")), "count");
  const cursorSequence = safeInteger(Number(values.get("cursorSequence")), "cursor");
  const mergedCount = safeInteger(Number(values.get("mergedCount")), "merged count");
  const state = values.get("state");
  const head = values.get("evidenceHeadSha256") ?? "";
  if (
    !SAFE_TOKEN.test(head) ||
    cursorSequence > evidenceCount ||
    mergedCount > cursorSequence ||
    !["building", "ready", "running", "paused", "complete"].includes(state ?? "")
  ) {
    throw new Error("Merge evidence plan cursor is invalid.");
  }
  const reviewToken = values.get("reviewToken") || undefined;
  const descriptorSha256 = values.get("descriptorSha256") || undefined;
  if (
    (reviewToken !== undefined && !SAFE_TOKEN.test(reviewToken)) ||
    (descriptorSha256 !== undefined && !SAFE_TOKEN.test(descriptorSha256)) ||
    (state !== "building" && (!reviewToken || !descriptorSha256))
  ) {
    throw new Error("Merge evidence plan review binding is invalid.");
  }
  return {
    ...identity,
    reviewToken,
    descriptorSha256,
    evidenceCount,
    evidenceHeadSha256: head,
    cursorSequence,
    mergedCount,
    state: state as MergeEvidencePlanSummary["state"],
    ...(values.get("pauseReason") ? { pauseReason: values.get("pauseReason") } : {}),
    ...(Number(values.get("diskFreeBytes")) > 0
      ? { diskFreeBytes: safeInteger(Number(values.get("diskFreeBytes")), "disk free bytes") }
      : {}),
    ...(Number(values.get("diskReserveBytes")) > 0
      ? {
          diskReserveBytes: safeInteger(
            Number(values.get("diskReserveBytes")),
            "disk reserve bytes"
          )
        }
      : {}),
    updatedAt: values.get("updatedAt") ?? ""
  };
}

function openDatabase(path: string): DatabaseSync {
  const database = new DatabaseSync(path);
  try {
    configure(database);
    readSummary(database);
    return database;
  } catch (error) {
    database.close();
    throw error;
  }
}

export class MergeEvidencePlanBuilder {
  private closed = false;

  private constructor(
    readonly directory: string,
    readonly path: string,
    private readonly database: DatabaseSync
  ) {}

  static async begin(
    directory: string,
    identity: MergeEvidencePlanIdentity
  ): Promise<MergeEvidencePlanBuilder> {
    validateIdentity(identity);
    await mkdir(directory, { recursive: true });
    const path = join(
      directory,
      `.merge-evidence.${randomBytes(12).toString("hex")}.sqlite3.tmp`
    );
    return new MergeEvidencePlanBuilder(directory, path, createDatabase(path, identity));
  }

  summary(): MergeEvidencePlanSummary {
    if (this.closed) throw new Error("Merge evidence builder is closed.");
    return readSummary(this.database);
  }

  containsFingerprint(fingerprint: string): boolean {
    if (this.closed) throw new Error("Merge evidence builder is closed.");
    if (!/^(?:content|blob|metadata):[a-f0-9]{64}$/i.test(fingerprint)) {
      throw new Error("Merge evidence fingerprint is invalid.");
    }
    return Boolean(this.database.prepare(
      "SELECT 1 present FROM evidence WHERE fingerprint=?"
    ).get(fingerprint.toLocaleLowerCase()));
  }

  append(entries: MergeEvidencePlanEntry[]): MergeEvidencePlanSummary {
    if (this.closed) throw new Error("Merge evidence builder is closed.");
    if (entries.length > 100) {
      throw new Error("Merge evidence batches must contain at most 100 rows.");
    }
    if (!entries.length) return readSummary(this.database);
    const normalized = entries.map(normalizeEntry);
    const state = readSummary(this.database);
    let sequence = state.evidenceCount;
    let previous = state.evidenceHeadSha256;
    const insert = this.database.prepare(`
      INSERT INTO evidence(
        sequence,source_sequence,fingerprint,target_source_id,payload_json,
        payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?,?,?)
    `);
    const updateMeta = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const entry of normalized) {
        sequence += 1;
        const payloadJson = JSON.stringify(entry.source);
        const payloadSha256 = sha256(payloadJson);
        const base: Omit<StoredRow, "payload_json" | "row_sha256"> = {
          sequence,
          source_sequence: entry.sourceSequence,
          fingerprint: entry.fingerprint,
          target_source_id: entry.targetSourceId,
          payload_sha256: payloadSha256,
          previous_sha256: previous
        };
        const digest = rowHash(base);
        insert.run(
          sequence,
          entry.sourceSequence,
          entry.fingerprint,
          entry.targetSourceId,
          payloadJson,
          payloadSha256,
          previous,
          digest
        );
        previous = digest;
      }
      updateMeta.run(String(sequence), "evidenceCount");
      updateMeta.run(previous, "evidenceHeadSha256");
      updateMeta.run(new Date().toISOString(), "updatedAt");
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return readSummary(this.database);
  }

  async commit(
    reviewToken: string,
    descriptorSha256: string,
    descriptorJson = ""
  ): Promise<string> {
    if (this.closed) throw new Error("Merge evidence builder is closed.");
    if (!SAFE_TOKEN.test(reviewToken) || !SAFE_TOKEN.test(descriptorSha256)) {
      throw new Error("Merge evidence review binding is invalid.");
    }
    if (
      descriptorJson.includes("\0") ||
      Buffer.byteLength(descriptorJson) > 8 * 1024 * 1024 ||
      (descriptorJson && sha256(descriptorJson) !== descriptorSha256)
    ) {
      throw new Error("Merge evidence review descriptor is invalid.");
    }
    const update = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      update.run(reviewToken, "reviewToken");
      update.run(descriptorSha256, "descriptorSha256");
      update.run(descriptorJson, "descriptorJson");
      update.run("ready", "state");
      update.run(new Date().toISOString(), "updatedAt");
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    const built = readSummary(this.database);
    this.database.close();
    this.closed = true;
    const handle = await openFile(this.path, "r+");
    try {
      await handle.sync();
    } finally {
      await handle.close();
    }
    const target = join(this.directory, `${reviewToken}.sqlite3`);
    if (await pathExists(target)) {
      const existing = await MergeEvidencePlan.open(this.directory, reviewToken);
      try {
        const summary = existing.summary();
        if (
          summary.descriptorSha256 !== descriptorSha256 ||
          summary.evidenceHeadSha256 !== built.evidenceHeadSha256 ||
          summary.evidenceCount !== built.evidenceCount ||
          summary.sourceBrainId !== built.sourceBrainId ||
          summary.targetBrainId !== built.targetBrainId ||
          summary.sourceUpdatedAt !== built.sourceUpdatedAt ||
          summary.targetUpdatedAt !== built.targetUpdatedAt ||
          summary.substrateDigest !== built.substrateDigest
        ) {
          throw new Error("Merge evidence review token already names another plan.");
        }
      } finally {
        existing.close();
      }
      await rm(this.path, { force: true });
      return target;
    }
    await rename(this.path, target);
    return target;
  }

  async abort(): Promise<void> {
    if (!this.closed) {
      this.database.close();
      this.closed = true;
    }
    await Promise.all([
      rm(this.path, { force: true }),
      rm(`${this.path}-journal`, { force: true })
    ]);
  }
}

export class MergeEvidencePlan {
  private constructor(
    readonly path: string,
    private readonly database: DatabaseSync
  ) {}

  static async openExisting(
    directory: string,
    reviewToken: string
  ): Promise<MergeEvidencePlan | undefined> {
    if (!SAFE_TOKEN.test(reviewToken)) {
      throw new Error("Merge evidence review token is invalid.");
    }
    if (!await pathExists(join(directory, `${reviewToken}.sqlite3`))) return undefined;
    return this.open(directory, reviewToken);
  }

  static async open(directory: string, reviewToken: string): Promise<MergeEvidencePlan> {
    if (!SAFE_TOKEN.test(reviewToken)) throw new Error("Merge evidence review token is invalid.");
    const path = join(directory, `${reviewToken}.sqlite3`);
    if (!await pathExists(path)) throw new Error("Merge evidence plan was not found.");
    const database = openDatabase(path);
    const summary = readSummary(database);
    if (summary.reviewToken !== reviewToken) {
      database.close();
      throw new Error("Merge evidence plan token does not match its filename.");
    }
    return new MergeEvidencePlan(path, database);
  }

  close(): void {
    this.database.close();
  }

  summary(): MergeEvidencePlanSummary {
    return readSummary(this.database);
  }

  descriptor<T = unknown>(): T | undefined {
    const row = this.database.prepare(
      "SELECT value FROM meta WHERE key='descriptorJson'"
    ).get() as { value?: string } | undefined;
    const value = row?.value ?? "";
    if (!value) return undefined;
    const state = this.summary();
    if (sha256(value) !== state.descriptorSha256) {
      throw new Error("Merge evidence review descriptor checksum failed.");
    }
    try {
      return JSON.parse(value) as T;
    } catch {
      throw new Error("Merge evidence review descriptor is invalid JSON.");
    }
  }

  page(afterSequence = 0, limitValue = 100): MergeEvidencePlanPage {
    if (!Number.isSafeInteger(afterSequence) || afterSequence < 0) {
      throw new Error("Merge evidence page cursor is invalid.");
    }
    const limit = Math.max(1, Math.min(100, Math.floor(limitValue)));
    const state = this.summary();
    const rows = this.database.prepare(`
      SELECT * FROM evidence
      WHERE sequence > ? ORDER BY sequence LIMIT ?
    `).all(afterSequence, limit + 1) as unknown as StoredRow[];
    const selected = rows.slice(0, limit);
    const previous = afterSequence === 0
      ? ZERO_HASH
      : (this.database.prepare(
          "SELECT row_sha256 FROM evidence WHERE sequence=?"
        ).get(afterSequence) as { row_sha256?: string } | undefined)?.row_sha256;
    if (!previous || (selected[0] && selected[0].previous_sha256 !== previous)) {
      throw new Error("Merge evidence page cursor checksum failed.");
    }
    for (let index = 0; index < selected.length; index += 1) {
      const row = selected[index]!;
      if (row.sequence !== afterSequence + index + 1) {
        throw new Error("Merge evidence page sequence is invalid.");
      }
      if (index > 0 && row.previous_sha256 !== selected[index - 1]!.row_sha256) {
        throw new Error("Merge evidence page hash chain failed.");
      }
    }
    return {
      entries: selected.map(presentRow),
      rowsRead: rows.length,
      totalEntries: state.evidenceCount,
      ...(rows.length > limit ? { nextSequence: selected.at(-1)!.sequence } : {})
    };
  }

  pendingPage(limit = 100): MergeEvidencePlanPage {
    return this.page(this.summary().cursorSequence, limit);
  }

  startOrResume(): MergeEvidencePlanSummary {
    const state = this.summary();
    if (state.state === "complete") return state;
    const update = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      update.run("running", "state");
      update.run("", "pauseReason");
      update.run("0", "diskFreeBytes");
      update.run("0", "diskReserveBytes");
      update.run(new Date().toISOString(), "updatedAt");
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.summary();
  }

  recordApplied(upToSequence: number, mergedInBatch: number): MergeEvidencePlanSummary {
    const state = this.summary();
    if (
      !Number.isSafeInteger(upToSequence) ||
      upToSequence < state.cursorSequence ||
      upToSequence > state.evidenceCount ||
      !Number.isSafeInteger(mergedInBatch) ||
      mergedInBatch < 0 ||
      state.mergedCount + mergedInBatch > upToSequence
    ) {
      throw new Error("Merge evidence cursor update is invalid.");
    }
    if (upToSequence === state.cursorSequence) {
      if (mergedInBatch !== 0) throw new Error("Merge evidence cursor replay changed its count.");
      return state;
    }
    const expected = Number((this.database.prepare(`
      SELECT COUNT(*) count FROM evidence
      WHERE sequence > ? AND sequence <= ?
    `).get(state.cursorSequence, upToSequence) as { count: number }).count);
    if (expected !== upToSequence - state.cursorSequence) {
      throw new Error("Merge evidence cursor skipped a row.");
    }
    if (mergedInBatch !== expected) {
      throw new Error("Merge evidence cursor did not account for every row.");
    }
    const update = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      update.run(String(upToSequence), "cursorSequence");
      update.run(String(state.mergedCount + expected), "mergedCount");
      update.run(
        upToSequence === state.evidenceCount ? "complete" : "running",
        "state"
      );
      update.run(new Date().toISOString(), "updatedAt");
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.summary();
  }

  pause(
    reason: string,
    diskFreeBytes?: number,
    diskReserveBytes?: number
  ): MergeEvidencePlanSummary {
    const clean = reason.replace(/\0/g, "").trim().slice(0, 1_000);
    if (!clean) throw new Error("Merge evidence pause reason is required.");
    const update = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      update.run("paused", "state");
      update.run(clean, "pauseReason");
      update.run(String(safeInteger(diskFreeBytes ?? 0, "disk free bytes")), "diskFreeBytes");
      update.run(
        String(safeInteger(diskReserveBytes ?? 0, "disk reserve bytes")),
        "diskReserveBytes"
      );
      update.run(new Date().toISOString(), "updatedAt");
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return this.summary();
  }

  checkpointResource(
    checkpoint: MergeEvidenceResourceCheckpoint
  ): { paused: boolean; summary: MergeEvidencePlanSummary } {
    const diskFreeBytes = safeInteger(checkpoint.diskFreeBytes, "disk free bytes");
    const diskReserveBytes = safeInteger(
      checkpoint.diskReserveBytes,
      "disk reserve bytes"
    );
    const incomingWriteBytes = safeInteger(
      checkpoint.incomingWriteBytes,
      "incoming write bytes"
    );
    if (
      !checkpoint.memoryHeadroom ||
      diskFreeBytes - incomingWriteBytes <= diskReserveBytes
    ) {
      const reason = !checkpoint.memoryHeadroom
        ? "Merge paused at the memory headroom watermark."
        : "Merge paused before entering the physical disk reserve.";
      return {
        paused: true,
        summary: this.pause(reason, diskFreeBytes, diskReserveBytes)
      };
    }
    return { paused: false, summary: this.startOrResume() };
  }

  integrity(): MergeEvidencePlanSummary {
    const quick = this.database.prepare("PRAGMA quick_check").get() as {
      quick_check?: string;
    } | undefined;
    if (quick?.quick_check !== "ok") throw new Error("Merge evidence SQLite integrity failed.");
    this.descriptor();
    const rows = this.database.prepare(
      "SELECT * FROM evidence ORDER BY sequence"
    ).iterate() as unknown as Iterable<StoredRow>;
    let sequence = 0;
    let previous = ZERO_HASH;
    for (const row of rows) {
      sequence += 1;
      if (row.sequence !== sequence || row.previous_sha256 !== previous) {
        throw new Error("Merge evidence hash chain failed.");
      }
      presentRow(row);
      previous = row.row_sha256;
    }
    const state = this.summary();
    if (state.evidenceCount !== sequence || state.evidenceHeadSha256 !== previous) {
      throw new Error("Merge evidence summary checksum failed.");
    }
    return state;
  }
}
