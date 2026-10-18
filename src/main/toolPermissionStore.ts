import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  ToolPermissionLevel,
  ToolPermissionRecord
} from "../shared/types";

const SAFE_BRAIN_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SAFE_REQUEST_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SAFE_TOOL_ID = /^[a-zA-Z0-9][a-zA-Z0-9._:-]{0,255}$/;
const LEVELS = new Set<ToolPermissionLevel>(["off", "ask", "auto", "full"]);

export interface ToolPermissionSnapshot {
  revision: number;
  records: ToolPermissionRecord[];
}

export interface ToolPermissionMutationOptions {
  requestId: string;
  expectedRevision?: number;
}

interface StoredRequest {
  requestSha256: string;
  resultJson: string;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function labelFor(toolId: string): string {
  return toolId
    .split(".")
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(" ");
}

function requireBrainId(value: string): string {
  if (!SAFE_BRAIN_ID.test(value)) throw new Error("Invalid brain id for host policy.");
  return value;
}

function requireRequestId(value: string): string {
  if (!SAFE_REQUEST_ID.test(value)) throw new Error("Invalid tool-permission request id.");
  return value;
}

function normalizedRecords(
  records: readonly ToolPermissionRecord[],
  revision: number
): ToolPermissionRecord[] {
  const normalized = new Map<string, ToolPermissionRecord>();
  for (const record of records) {
    if (!record || !SAFE_TOOL_ID.test(record.toolId) || !LEVELS.has(record.level)) {
      throw new Error("Invalid tool permission record.");
    }
    const label = typeof record.label === "string" && record.label.trim()
      ? record.label.trim().slice(0, 256)
      : labelFor(record.toolId);
    const updatedAt = typeof record.updatedAt === "string" && Number.isFinite(Date.parse(record.updatedAt))
      ? new Date(record.updatedAt).toISOString()
      : new Date(0).toISOString();
    normalized.set(record.toolId, {
      toolId: record.toolId,
      label,
      level: record.level,
      updatedAt,
      revision
    });
  }
  return [...normalized.values()].sort(
    (left, right) => left.label.localeCompare(right.label) || left.toolId.localeCompare(right.toolId)
  );
}

function parseSnapshot(value: string): ToolPermissionSnapshot {
  const parsed = JSON.parse(value) as ToolPermissionSnapshot;
  if (!Number.isSafeInteger(parsed.revision) || parsed.revision < 1 || !Array.isArray(parsed.records)) {
    throw new Error("Stored tool-permission receipt is invalid.");
  }
  return {
    revision: parsed.revision,
    records: normalizedRecords(parsed.records, parsed.revision).map((record) => {
      const source = parsed.records.find((candidate) => candidate.toolId === record.toolId);
      return { ...record, revision: source?.revision ?? parsed.revision };
    })
  };
}

/**
 * Installation-private, main-process authority for external-action policy.
 *
 * The database deliberately lives outside every brain directory so Auto file
 * access, neural checkpoints, imports, and recovery-point promotion cannot
 * rewrite authority. Request receipts are durable and never pruned: retrying
 * an old lost-ACK request can therefore never become a fresh grant.
 */
export class ToolPermissionStore {
  private readonly directory: string;
  private readonly path: string;

  constructor(repositoryRoot: string) {
    this.directory = join(repositoryRoot, ".host-policy");
    this.path = join(this.directory, "tool-permissions.sqlite3");
  }

  async initialize(): Promise<void> {
    await mkdir(this.directory, { recursive: true, mode: 0o700 });
    const info = await lstat(this.directory);
    if (!info.isDirectory() || info.isSymbolicLink()) {
      throw new Error("Host-policy storage is not a private app-managed directory.");
    }
    try {
      const databaseInfo = await lstat(this.path);
      if (!databaseInfo.isFile() || databaseInfo.isSymbolicLink()) {
        throw new Error("Host-policy database is not a safe regular file.");
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    this.withDatabase(() => undefined);
  }

  private withDatabase<T>(operation: (database: DatabaseSync) => T): T {
    const database = new DatabaseSync(this.path);
    try {
      database.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=5000;");
      database.exec(`
        CREATE TABLE IF NOT EXISTS permission_instances (
          brain_id TEXT PRIMARY KEY,
          revision INTEGER NOT NULL CHECK (revision >= 1),
          updated_at TEXT NOT NULL
        ) STRICT;
        CREATE TABLE IF NOT EXISTS tool_permissions (
          brain_id TEXT NOT NULL,
          tool_id TEXT NOT NULL,
          label TEXT NOT NULL,
          level TEXT NOT NULL CHECK (level IN ('off', 'ask', 'auto', 'full')),
          updated_at TEXT NOT NULL,
          revision INTEGER NOT NULL CHECK (revision >= 1),
          PRIMARY KEY (brain_id, tool_id),
          FOREIGN KEY (brain_id) REFERENCES permission_instances(brain_id) ON DELETE CASCADE
        ) STRICT;
        CREATE TABLE IF NOT EXISTS permission_requests (
          brain_id TEXT NOT NULL,
          request_id TEXT NOT NULL,
          request_sha256 TEXT NOT NULL,
          result_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          PRIMARY KEY (brain_id, request_id),
          FOREIGN KEY (brain_id) REFERENCES permission_instances(brain_id) ON DELETE CASCADE
        ) STRICT;
      `);
      database.exec("PRAGMA foreign_keys=ON;");
      return operation(database);
    } finally {
      database.close();
    }
  }

  private snapshot(database: DatabaseSync, brainId: string): ToolPermissionSnapshot | undefined {
    const instance = database.prepare(
      "SELECT revision FROM permission_instances WHERE brain_id = ?"
    ).get(brainId) as { revision: number } | undefined;
    if (!instance) return undefined;
    const rows = database.prepare(`
      SELECT tool_id AS toolId, label, level, updated_at AS updatedAt, revision
      FROM tool_permissions WHERE brain_id = ? ORDER BY label, tool_id
    `).all(brainId) as unknown as ToolPermissionRecord[];
    return { revision: instance.revision, records: rows.map((record) => ({ ...record })) };
  }

  private initializeInTransaction(
    database: DatabaseSync,
    brainId: string,
    records: readonly ToolPermissionRecord[]
  ): ToolPermissionSnapshot {
    const existing = this.snapshot(database, brainId);
    if (existing) return existing;
    const now = new Date().toISOString();
    const prepared = normalizedRecords(records, 1);
    database.prepare(
      "INSERT INTO permission_instances (brain_id, revision, updated_at) VALUES (?, 1, ?)"
    ).run(brainId, now);
    const insert = database.prepare(`
      INSERT INTO tool_permissions (brain_id, tool_id, label, level, updated_at, revision)
      VALUES (?, ?, ?, ?, ?, 1)
    `);
    for (const record of prepared) {
      insert.run(brainId, record.toolId, record.label, record.level, record.updatedAt);
    }
    return this.snapshot(database, brainId)!;
  }

  async read(
    brainId: string,
    bootstrap: readonly ToolPermissionRecord[]
  ): Promise<ToolPermissionSnapshot> {
    requireBrainId(brainId);
    await this.initialize();
    return this.withDatabase((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const snapshot = this.initializeInTransaction(database, brainId, bootstrap);
        database.exec("COMMIT");
        return snapshot;
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }

  async initializeIdentity(
    brainId: string,
    records: readonly ToolPermissionRecord[]
  ): Promise<ToolPermissionSnapshot> {
    return this.read(brainId, records);
  }

  /**
   * Establish policy at an explicit new-identity trust boundary. This is used
   * for reviewed merge access and safe external-import defaults, never for a
   * snapshot restore or neural checkpoint.
   */
  async replaceIdentity(
    brainId: string,
    records: readonly ToolPermissionRecord[]
  ): Promise<ToolPermissionSnapshot> {
    requireBrainId(brainId);
    await this.initialize();
    return this.withDatabase((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        database.prepare("DELETE FROM permission_instances WHERE brain_id = ?").run(brainId);
        const snapshot = this.initializeInTransaction(database, brainId, records);
        database.exec("COMMIT");
        return snapshot;
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }

  async update(
    brainId: string,
    toolId: string,
    label: string,
    level: ToolPermissionLevel,
    bootstrap: readonly ToolPermissionRecord[],
    options: ToolPermissionMutationOptions
  ): Promise<ToolPermissionSnapshot> {
    requireBrainId(brainId);
    requireRequestId(options.requestId);
    if (!SAFE_TOOL_ID.test(toolId) || !LEVELS.has(level)) {
      throw new Error("Invalid tool permission update.");
    }
    if (options.expectedRevision !== undefined &&
      (!Number.isSafeInteger(options.expectedRevision) || options.expectedRevision < 1)) {
      throw new Error("Invalid expected tool-permission revision.");
    }
    await this.initialize();
    const requestSha256 = sha256(JSON.stringify({ toolId, level, expectedRevision: options.expectedRevision ?? null }));
    return this.withDatabase((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        let snapshot = this.initializeInTransaction(database, brainId, bootstrap);
        const prior = database.prepare(`
          SELECT request_sha256 AS requestSha256, result_json AS resultJson
          FROM permission_requests WHERE brain_id = ? AND request_id = ?
        `).get(brainId, options.requestId) as StoredRequest | undefined;
        if (prior) {
          if (prior.requestSha256 !== requestSha256) {
            throw new Error("Tool-permission request id was already used for different arguments.");
          }
          // Parse the durable result as an integrity check, but expose the
          // current policy. A very old retry is acknowledged without ever
          // replaying its mutation or presenting a superseded grant as live.
          parseSnapshot(prior.resultJson);
          database.exec("COMMIT");
          return snapshot;
        }
        if (options.expectedRevision !== undefined && options.expectedRevision !== snapshot.revision) {
          throw new Error("Tool permissions changed since this settings view was loaded.");
        }
        const existing = snapshot.records.find((record) => record.toolId === toolId);
        if (!existing || existing.level !== level || existing.label !== label) {
          const revision = snapshot.revision + 1;
          const updatedAt = new Date().toISOString();
          database.prepare(`
            INSERT INTO tool_permissions (brain_id, tool_id, label, level, updated_at, revision)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(brain_id, tool_id) DO UPDATE SET
              label = excluded.label,
              level = excluded.level,
              updated_at = excluded.updated_at,
              revision = excluded.revision
          `).run(brainId, toolId, label.slice(0, 256), level, updatedAt, revision);
          database.prepare(
            "UPDATE permission_instances SET revision = ?, updated_at = ? WHERE brain_id = ?"
          ).run(revision, updatedAt, brainId);
          snapshot = this.snapshot(database, brainId)!;
        }
        database.prepare(`
          INSERT INTO permission_requests
            (brain_id, request_id, request_sha256, result_json, created_at)
          VALUES (?, ?, ?, ?, ?)
        `).run(brainId, options.requestId, requestSha256, JSON.stringify(snapshot), new Date().toISOString());
        database.exec("COMMIT");
        return snapshot;
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }

  async ensure(
    brainId: string,
    toolId: string,
    label: string,
    level: ToolPermissionLevel,
    bootstrap: readonly ToolPermissionRecord[],
    requestId: string
  ): Promise<ToolPermissionSnapshot> {
    requireBrainId(brainId);
    requireRequestId(requestId);
    if (!SAFE_TOOL_ID.test(toolId) || !LEVELS.has(level)) {
      throw new Error("Invalid tool permission default.");
    }
    await this.initialize();
    const requestSha256 = sha256(JSON.stringify({ ensure: true, toolId, level }));
    return this.withDatabase((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        let snapshot = this.initializeInTransaction(database, brainId, bootstrap);
        const prior = database.prepare(`
          SELECT request_sha256 AS requestSha256, result_json AS resultJson
          FROM permission_requests WHERE brain_id = ? AND request_id = ?
        `).get(brainId, requestId) as StoredRequest | undefined;
        if (prior) {
          if (prior.requestSha256 !== requestSha256) {
            throw new Error("Tool-permission request id was already used for different arguments.");
          }
          parseSnapshot(prior.resultJson);
          database.exec("COMMIT");
          return snapshot;
        }
        if (!snapshot.records.some((record) => record.toolId === toolId)) {
          const revision = snapshot.revision + 1;
          const updatedAt = new Date().toISOString();
          database.prepare(`
            INSERT INTO tool_permissions (brain_id, tool_id, label, level, updated_at, revision)
            VALUES (?, ?, ?, ?, ?, ?)
          `).run(brainId, toolId, label.slice(0, 256), level, updatedAt, revision);
          database.prepare(
            "UPDATE permission_instances SET revision = ?, updated_at = ? WHERE brain_id = ?"
          ).run(revision, updatedAt, brainId);
          snapshot = this.snapshot(database, brainId)!;
        }
        database.prepare(`
          INSERT INTO permission_requests
            (brain_id, request_id, request_sha256, result_json, created_at)
          VALUES (?, ?, ?, ?, ?)
        `).run(brainId, requestId, requestSha256, JSON.stringify(snapshot), new Date().toISOString());
        database.exec("COMMIT");
        return snapshot;
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }

  async removeIdentity(brainId: string): Promise<void> {
    requireBrainId(brainId);
    await this.initialize();
    this.withDatabase((database) => {
      database.exec("BEGIN IMMEDIATE");
      try {
        database.prepare("DELETE FROM permission_instances WHERE brain_id = ?").run(brainId);
        database.exec("COMMIT");
      } catch (error) {
        database.exec("ROLLBACK");
        throw error;
      }
    });
  }
}
