import { createHash, randomBytes } from "node:crypto";
import { open as openFile } from "node:fs/promises";
import { access, mkdir, readFile, rename, rm } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

const ARTIFACT_INDEX_FORMAT = "omni-generated-artifact-index";
const ARTIFACT_LEDGER_FORMAT = "omni-generated-artifact-ledger";
const ARTIFACT_LEDGER_VERSION = 1;
const ZERO_HASH = "0".repeat(64);
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const CURSOR = /^([1-9][0-9]{0,15})\.([a-f0-9]{64})$/;

export const ARTIFACT_INDEX_DATABASE = "index.sqlite3";
export const LEGACY_ARTIFACT_INDEX = "index.json";

export interface PersistedGeneratedArtifact {
  id: string;
  modality: "image" | "audio" | "video";
  mimeType: string;
  sha256: string;
  bytes: number;
  relativePath: string;
  createdAt: string;
  seed?: number;
  initialization?: string;
  qualityNote?: string;
}

export interface PersistedArtifactIndex {
  format: typeof ARTIFACT_INDEX_FORMAT;
  formatVersion: 1;
  brainId: string;
  artifacts: PersistedGeneratedArtifact[];
  contentSha256: string;
}

export interface ArtifactIndexPage {
  artifacts: PersistedGeneratedArtifact[];
  totalArtifacts: number;
  nextCursor?: string;
  /** Exact SQLite rows materialized for this request, including one lookahead row. */
  rowsRead: number;
}

interface ArtifactRow {
  sequence: number;
  artifact_id: string;
  payload_json: string;
  payload_sha256: string;
  previous_sha256: string;
  row_sha256: string;
}

interface ArtifactLedgerSummary {
  totalArtifacts: number;
  headSequence: number;
  headSha256: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function artifactIndexDigest(
  value: Omit<PersistedArtifactIndex, "contentSha256">
): string {
  return sha256(JSON.stringify(value));
}

function safeMimeType(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const normalized = value.toLocaleLowerCase();
  return /^(?:image|audio|video)\/[a-z0-9.+-]{1,80}$/i.test(normalized)
    ? normalized
    : undefined;
}

function normalizeArtifact(value: unknown): PersistedGeneratedArtifact {
  const entry = record(value);
  const mimeType = safeMimeType(entry?.mimeType);
  const sha = entry?.sha256;
  const modality = entry?.modality;
  const relativePath = entry?.relativePath;
  if (
    !entry ||
    typeof entry.id !== "string" ||
    !/^[a-f0-9]{64}$/.test(entry.id) ||
    !["image", "audio", "video"].includes(String(modality)) ||
    !mimeType ||
    typeof sha !== "string" ||
    !/^[a-f0-9]{64}$/.test(sha) ||
    typeof entry.bytes !== "number" ||
    !Number.isSafeInteger(entry.bytes) ||
    entry.bytes <= 0 ||
    typeof relativePath !== "string" ||
    !/^artifacts\/[a-zA-Z0-9][a-zA-Z0-9._-]{0,254}$/.test(relativePath) ||
    typeof entry.createdAt !== "string" ||
    !Number.isFinite(Date.parse(entry.createdAt))
  ) {
    throw new Error("Generated artifact index entry is invalid.");
  }
  return {
    id: entry.id,
    modality: modality as PersistedGeneratedArtifact["modality"],
    mimeType,
    sha256: sha,
    bytes: entry.bytes,
    relativePath,
    createdAt: entry.createdAt,
    ...(typeof entry.seed === "number" && Number.isSafeInteger(entry.seed)
      ? { seed: entry.seed }
      : {}),
    ...(typeof entry.initialization === "string"
      ? { initialization: entry.initialization.slice(0, 240) }
      : {}),
    ...(typeof entry.qualityNote === "string"
      ? { qualityNote: entry.qualityNote.slice(0, 1_000) }
      : {})
  };
}

export function serializeArtifactIndex(
  brainId: string,
  artifacts: PersistedGeneratedArtifact[]
): string {
  const body: Omit<PersistedArtifactIndex, "contentSha256"> = {
    format: ARTIFACT_INDEX_FORMAT,
    formatVersion: 1,
    brainId,
    artifacts: artifacts.map(normalizeArtifact)
  };
  return JSON.stringify({ ...body, contentSha256: artifactIndexDigest(body) }, null, 2);
}

export function parseArtifactIndex(
  value: unknown,
  expectedBrainId: string,
  rekeyBrainId = expectedBrainId
): PersistedArtifactIndex {
  const source = record(value);
  if (
    !source ||
    source.format !== ARTIFACT_INDEX_FORMAT ||
    source.formatVersion !== 1 ||
    source.brainId !== expectedBrainId ||
    !Array.isArray(source.artifacts) ||
    typeof source.contentSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(source.contentSha256)
  ) {
    throw new Error("Generated artifact index is invalid.");
  }
  const artifacts = source.artifacts.map(normalizeArtifact);
  const body: Omit<PersistedArtifactIndex, "contentSha256"> = {
    format: ARTIFACT_INDEX_FORMAT,
    formatVersion: 1,
    brainId: expectedBrainId,
    artifacts
  };
  if (artifactIndexDigest(body) !== source.contentSha256) {
    throw new Error("Generated artifact index checksum failed.");
  }
  const rekeyed = { ...body, brainId: rekeyBrainId };
  return { ...rekeyed, contentSha256: artifactIndexDigest(rekeyed) };
}

function rowHash(row: Omit<ArtifactRow, "row_sha256" | "payload_json">): string {
  return sha256(JSON.stringify({
    sequence: row.sequence,
    artifactId: row.artifact_id,
    payloadSha256: row.payload_sha256,
    previousSha256: row.previous_sha256
  }));
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

function createDatabase(path: string, brainId: string): DatabaseSync {
  const database = new DatabaseSync(path);
  configure(database);
  database.exec(`
    CREATE TABLE meta (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );
    CREATE TABLE artifacts (
      sequence INTEGER PRIMARY KEY,
      artifact_id TEXT NOT NULL UNIQUE,
      payload_json TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL,
      previous_sha256 TEXT NOT NULL,
      row_sha256 TEXT NOT NULL UNIQUE
    );
  `);
  const insert = database.prepare("INSERT INTO meta(key,value) VALUES(?,?)");
  for (const [key, value] of ([
    ["format", ARTIFACT_LEDGER_FORMAT],
    ["formatVersion", String(ARTIFACT_LEDGER_VERSION)],
    ["brainId", brainId],
    ["totalArtifacts", "0"],
    ["headSequence", "0"],
    ["headSha256", ZERO_HASH]
  ] satisfies Array<[string, string]>)) insert.run(key, value);
  return database;
}

function readMeta(database: DatabaseSync): Map<string, string> {
  const rows = database.prepare("SELECT key,value FROM meta").all() as unknown as Array<{
    key: string;
    value: string;
  }>;
  return new Map(rows.map((row) => [row.key, row.value]));
}

function openDatabase(path: string, brainId: string): DatabaseSync {
  let database: DatabaseSync;
  try {
    database = new DatabaseSync(path);
    configure(database);
    const meta = readMeta(database);
    if (
      meta.get("format") !== ARTIFACT_LEDGER_FORMAT ||
      meta.get("formatVersion") !== String(ARTIFACT_LEDGER_VERSION) ||
      meta.get("brainId") !== brainId ||
      !/^[0-9]+$/.test(meta.get("totalArtifacts") ?? "") ||
      !/^[0-9]+$/.test(meta.get("headSequence") ?? "") ||
      !/^[a-f0-9]{64}$/.test(meta.get("headSha256") ?? "")
    ) {
      throw new Error("Generated artifact ledger identity is invalid.");
    }
    return database;
  } catch (error) {
    try {
      database!.close();
    } catch {
      // Best effort for a database that failed while opening.
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

function parseSummary(database: DatabaseSync): ArtifactLedgerSummary {
  const meta = readMeta(database);
  const totalArtifacts = Number(meta.get("totalArtifacts"));
  const headSequence = Number(meta.get("headSequence"));
  const headSha256 = meta.get("headSha256") ?? "";
  if (
    !Number.isSafeInteger(totalArtifacts) ||
    totalArtifacts < 0 ||
    !Number.isSafeInteger(headSequence) ||
    headSequence < 0 ||
    !/^[a-f0-9]{64}$/.test(headSha256) ||
    totalArtifacts !== headSequence
  ) {
    throw new Error("Generated artifact ledger summary is invalid.");
  }
  return { totalArtifacts, headSequence, headSha256 };
}

function presentRow(row: ArtifactRow): PersistedGeneratedArtifact {
  if (
    sha256(row.payload_json) !== row.payload_sha256 ||
    rowHash(row) !== row.row_sha256
  ) {
    throw new Error("Generated artifact ledger row checksum failed.");
  }
  const artifact = normalizeArtifact(JSON.parse(row.payload_json) as unknown);
  if (artifact.id !== row.artifact_id) {
    throw new Error("Generated artifact ledger row identity failed.");
  }
  return artifact;
}

/**
 * Append-only, cursor-indexed artifact metadata. Normal browsing materializes
 * at most one page plus a lookahead row; whole-ledger scans are reserved for
 * explicit integrity/export/clone operations.
 */
export class ArtifactIndexStore {
  private constructor(
    readonly directory: string,
    readonly brainId: string,
    readonly path: string,
    private readonly database: DatabaseSync
  ) {}

  static databasePath(directory: string): string {
    return join(directory, ARTIFACT_INDEX_DATABASE);
  }

  static legacyPath(directory: string): string {
    return join(directory, LEGACY_ARTIFACT_INDEX);
  }

  static async exists(directory: string): Promise<boolean> {
    return (await pathExists(this.databasePath(directory))) ||
      (await pathExists(this.legacyPath(directory)));
  }

  static async openExisting(
    directory: string,
    brainId: string
  ): Promise<ArtifactIndexStore | undefined> {
    if (!await this.exists(directory)) return undefined;
    return this.open(directory, brainId);
  }

  static async open(directory: string, brainId: string): Promise<ArtifactIndexStore> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid generated artifact brain id.");
    await mkdir(directory, { recursive: true });
    const path = this.databasePath(directory);
    const legacy = this.legacyPath(directory);
    await withPathLock(path, async () => {
      if (!await pathExists(path)) {
        let artifacts: PersistedGeneratedArtifact[] = [];
        if (await pathExists(legacy)) {
          artifacts = parseArtifactIndex(
            JSON.parse(await readFile(legacy, "utf8")),
            brainId
          ).artifacts;
        }
        await this.writeAtomic(directory, brainId, artifacts, false);
      }
      const probe = openDatabase(path, brainId);
      probe.close();
      // If a crash left both files after successful promotion, the SQLite
      // ledger is authoritative and the monolithic migration source is stale.
      await rm(legacy, { force: true });
    });
    return new ArtifactIndexStore(directory, brainId, path, openDatabase(path, brainId));
  }

  static async replace(
    directory: string,
    brainId: string,
    artifacts: PersistedGeneratedArtifact[]
  ): Promise<void> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid generated artifact brain id.");
    await mkdir(directory, { recursive: true });
    const path = this.databasePath(directory);
    await withPathLock(path, () => this.writeAtomic(directory, brainId, artifacts, true));
  }

  private static async writeAtomic(
    directory: string,
    brainId: string,
    artifacts: PersistedGeneratedArtifact[],
    replaceExisting: boolean
  ): Promise<void> {
    const path = this.databasePath(directory);
    const temporary = join(
      directory,
      `.index.${randomBytes(12).toString("hex")}.sqlite3.tmp`
    );
    const backup = join(
      directory,
      `.index.${randomBytes(12).toString("hex")}.sqlite3.bak`
    );
    let database: DatabaseSync | undefined;
    let movedExisting = false;
    try {
      database = createDatabase(temporary, brainId);
      const store = new ArtifactIndexStore(directory, brainId, temporary, database);
      store.append(artifacts);
      store.integrity();
      store.close();
      database = undefined;
      const handle = await openFile(temporary, "r");
      try {
        await handle.sync();
      } finally {
        await handle.close();
      }
      if (await pathExists(path)) {
        if (!replaceExisting) {
          await rm(temporary, { force: true });
          return;
        }
        await rename(path, backup);
        movedExisting = true;
      }
      try {
        await rename(temporary, path);
      } catch (error) {
        if (movedExisting) await rename(backup, path).catch(() => undefined);
        throw error;
      }
      if (movedExisting) await rm(backup, { force: true });
      await rm(this.legacyPath(directory), { force: true });
    } finally {
      try {
        database?.close();
      } catch {
        // Best effort while unwinding a failed SQLite initialization.
      }
      await Promise.all([
        rm(temporary, { force: true }),
        rm(`${temporary}-journal`, { force: true }),
        movedExisting && await pathExists(backup)
          ? rename(backup, path).catch(() => undefined)
          : Promise.resolve()
      ]);
    }
  }

  close(): void {
    this.database.close();
  }

  append(values: PersistedGeneratedArtifact[]): ArtifactLedgerSummary {
    if (!values.length) return parseSummary(this.database);
    const normalized = values.map(normalizeArtifact);
    const summary = parseSummary(this.database);
    let sequence = summary.headSequence;
    let previous = summary.headSha256;
    let inserted = 0;
    const existing = this.database.prepare(
      "SELECT payload_sha256 FROM artifacts WHERE artifact_id=?"
    );
    const insert = this.database.prepare(`
      INSERT INTO artifacts(
        sequence,artifact_id,payload_json,payload_sha256,previous_sha256,row_sha256
      ) VALUES(?,?,?,?,?,?)
    `);
    const updateMeta = this.database.prepare("UPDATE meta SET value=? WHERE key=?");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      for (const artifact of normalized) {
        const payloadJson = JSON.stringify(artifact);
        const payloadSha256 = sha256(payloadJson);
        const prior = existing.get(artifact.id) as { payload_sha256?: string } | undefined;
        if (prior) {
          if (prior.payload_sha256 !== payloadSha256) {
            throw new Error("Generated artifact ledger idempotency conflict.");
          }
          continue;
        }
        sequence += 1;
        const base: Omit<ArtifactRow, "row_sha256"> = {
          sequence,
          artifact_id: artifact.id,
          payload_json: payloadJson,
          payload_sha256: payloadSha256,
          previous_sha256: previous
        };
        const digest = rowHash(base);
        insert.run(
          base.sequence,
          base.artifact_id,
          base.payload_json,
          base.payload_sha256,
          base.previous_sha256,
          digest
        );
        previous = digest;
        inserted += 1;
      }
      if (inserted) {
        updateMeta.run(String(summary.totalArtifacts + inserted), "totalArtifacts");
        updateMeta.run(String(sequence), "headSequence");
        updateMeta.run(previous, "headSha256");
      }
      this.database.exec("COMMIT");
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
    return parseSummary(this.database);
  }

  page(cursor?: string, limitValue = 48): ArtifactIndexPage {
    const limit = Math.max(1, Math.min(100, Math.floor(limitValue)));
    const summary = parseSummary(this.database);
    let beforeSequence = summary.headSequence + 1;
    let expectedFirstSha256: string | undefined;
    if (cursor !== undefined) {
      const match = CURSOR.exec(cursor);
      beforeSequence = Number(match?.[1]);
      expectedFirstSha256 = match?.[2];
      if (
        !match ||
        !Number.isSafeInteger(beforeSequence) ||
        beforeSequence < 1 ||
        beforeSequence > summary.headSequence
      ) {
        throw new Error("Generated artifact cursor is invalid.");
      }
    }
    const rows = this.database.prepare(`
      SELECT sequence,artifact_id,payload_json,payload_sha256,previous_sha256,row_sha256
      FROM artifacts
      WHERE sequence < ?
      ORDER BY sequence DESC
      LIMIT ?
    `).all(beforeSequence, limit + 1) as unknown as ArtifactRow[];
    if (cursor === undefined && rows[0] && (
      rows[0].sequence !== summary.headSequence ||
      rows[0].row_sha256 !== summary.headSha256
    )) {
      throw new Error("Generated artifact ledger head checksum failed.");
    }
    if (expectedFirstSha256 !== undefined) {
      if (!rows[0] || rows[0].row_sha256 !== expectedFirstSha256) {
        throw new Error("Generated artifact cursor checksum failed.");
      }
    }
    for (let index = 0; index < rows.length - 1; index += 1) {
      if (
        rows[index]!.sequence !== rows[index + 1]!.sequence + 1 ||
        rows[index]!.previous_sha256 !== rows[index + 1]!.row_sha256
      ) {
        throw new Error("Generated artifact ledger page chain failed.");
      }
    }
    const selected = rows.slice(0, limit);
    const artifacts = selected.map(presentRow);
    const oldest = selected.at(-1);
    return {
      artifacts,
      totalArtifacts: summary.totalArtifacts,
      rowsRead: rows.length,
      ...(rows.length > limit && oldest
        ? { nextCursor: `${oldest.sequence}.${oldest.previous_sha256}` }
        : {})
    };
  }

  integrity(): ArtifactLedgerSummary {
    const quick = this.database.prepare("PRAGMA quick_check").get() as
      { quick_check?: string } | undefined;
    if (quick?.quick_check !== "ok") {
      throw new Error("Generated artifact SQLite integrity failed.");
    }
    const rows = this.database.prepare(`
      SELECT sequence,artifact_id,payload_json,payload_sha256,previous_sha256,row_sha256
      FROM artifacts ORDER BY sequence
    `).all() as unknown as ArtifactRow[];
    let sequence = 0;
    let previous = ZERO_HASH;
    for (const row of rows) {
      sequence += 1;
      if (
        row.sequence !== sequence ||
        row.previous_sha256 !== previous
      ) {
        throw new Error("Generated artifact ledger integrity verification failed.");
      }
      presentRow(row);
      previous = row.row_sha256;
    }
    const summary = parseSummary(this.database);
    if (
      summary.totalArtifacts !== rows.length ||
      summary.headSequence !== sequence ||
      summary.headSha256 !== previous
    ) {
      throw new Error("Generated artifact ledger summary checksum failed.");
    }
    return summary;
  }

  snapshot(rekeyBrainId = this.brainId): PersistedArtifactIndex {
    this.integrity();
    const rows = this.database.prepare(`
      SELECT sequence,artifact_id,payload_json,payload_sha256,previous_sha256,row_sha256
      FROM artifacts ORDER BY sequence
    `).all() as unknown as ArtifactRow[];
    return parseArtifactIndex(
      JSON.parse(serializeArtifactIndex(this.brainId, rows.map(presentRow))),
      this.brainId,
      rekeyBrainId
    );
  }
}
