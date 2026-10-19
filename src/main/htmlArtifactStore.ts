import { createHash, randomUUID } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, mkdir, open, readFile, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  HtmlArtifactContent,
  HtmlArtifactOpenRequest,
  HtmlArtifactProjection,
  HtmlArtifactResult
} from "../shared/htmlArtifacts";
import { HTML_ARTIFACT_RESULT_FORMAT } from "../shared/htmlArtifacts";
import {
  canonicalHtmlArtifactValue,
  htmlArtifactContentSha256,
  normalizeHtmlArtifactContent,
  validateHtmlArtifactContent
} from "./htmlArtifactFormat";
import { sqliteFilename } from "./sqliteFilename";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const HEX32 = /^[a-f0-9]{32}$/;
const SHA256 = /^[a-f0-9]{64}$/;
export const HTML_ARTIFACT_DATABASE = "index.sqlite3";
const leases = new Map<string, Promise<void>>();

async function withArtifactLease<T>(key: string, operation: () => Promise<T>): Promise<T> {
  const prior = leases.get(key) ?? Promise.resolve();
  let release!: () => void;
  const next = new Promise<void>((resolve) => { release = resolve; });
  leases.set(key, next);
  await prior;
  try { return await operation(); }
  finally { release(); if (leases.get(key) === next) leases.delete(key); }
}

export interface HtmlArtifactLineage {
  brainId: string;
  artifactId: string;
  kind: "copy" | "merge";
  recordedAt: string;
}

export interface PersistedHtmlArtifactRow {
  local_id: string;
  brain_id: string;
  original_brain_id: string;
  original_artifact_id: string;
  source_action_id: string;
  content_sha256: string;
  request_sha256: string;
  title: string;
  interactive: number;
  operational_state: "enabled" | "dormant";
  revision: number;
  manifest_sha256: string;
  created_at: string;
  updated_at: string;
  lineage_json: string;
}

function digest(value: string | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

async function exists(path: string): Promise<boolean> {
  try { await lstat(path); return true; }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return false; throw error; }
}

async function fileSha256(path: string): Promise<string> {
  const hash = createHash("sha256");
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest("hex");
}

function requireId(value: string, label: string): string {
  if (!SAFE_ID.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}

function lineage(value: string): HtmlArtifactLineage[] {
  let parsed: unknown;
  try { parsed = JSON.parse(value); } catch { throw new Error("HTML artifact lineage is not JSON."); }
  if (!Array.isArray(parsed) || parsed.some((entry) => {
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) return true;
    const row = entry as Record<string, unknown>;
    return Object.keys(row).sort().join(",") !== "artifactId,brainId,kind,recordedAt" ||
      !SAFE_ID.test(String(row.brainId)) || !SAFE_ID.test(String(row.artifactId)) ||
      !["copy", "merge"].includes(String(row.kind)) ||
      typeof row.recordedAt !== "string" || !Number.isFinite(Date.parse(row.recordedAt));
  })) throw new Error("HTML artifact lineage is invalid.");
  return parsed as HtmlArtifactLineage[];
}

function publicManifest(row: Omit<PersistedHtmlArtifactRow, "manifest_sha256">): Record<string, unknown> {
  return {
    format: "omni-chat-html-artifact-record",
    version: 1,
    localId: row.local_id,
    brainId: row.brain_id,
    originalBrainId: row.original_brain_id,
    originalArtifactId: row.original_artifact_id,
    sourceActionId: row.source_action_id,
    contentSha256: row.content_sha256,
    requestSha256: row.request_sha256,
    title: row.title,
    interactive: row.interactive === 1,
    operationalState: row.operational_state,
    revision: row.revision,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    lineage: JSON.parse(row.lineage_json)
  };
}

export function htmlArtifactManifestSha256(row: Omit<PersistedHtmlArtifactRow, "manifest_sha256">): string {
  return digest(canonicalHtmlArtifactValue(publicManifest(row)));
}

export function validatePersistedHtmlArtifactRow(row: PersistedHtmlArtifactRow, brainId: string): void {
  const withoutManifest = { ...row } as PersistedHtmlArtifactRow;
  delete (withoutManifest as Partial<PersistedHtmlArtifactRow>).manifest_sha256;
  const history = lineage(row.lineage_json);
  if (!HEX32.test(row.local_id) || row.brain_id !== brainId || !SAFE_ID.test(row.original_brain_id) ||
      !SAFE_ID.test(row.original_artifact_id) || !SAFE_ID.test(row.source_action_id) ||
      !SHA256.test(row.content_sha256) || !SHA256.test(row.request_sha256) || !SHA256.test(row.manifest_sha256) ||
      typeof row.title !== "string" || !row.title.trim() || row.title.length > 256 ||
      ![0, 1].includes(row.interactive) || !["enabled", "dormant"].includes(row.operational_state) ||
      !Number.isSafeInteger(row.revision) || row.revision < 1 ||
      !Number.isFinite(Date.parse(row.created_at)) || !Number.isFinite(Date.parse(row.updated_at)) ||
      canonicalHtmlArtifactValue(history) !== row.lineage_json ||
      htmlArtifactManifestSha256(withoutManifest) !== row.manifest_sha256) {
    throw new Error("Saved HTML artifact record failed integrity validation.");
  }
}

function projection(row: PersistedHtmlArtifactRow): HtmlArtifactProjection {
  validatePersistedHtmlArtifactRow(row, row.brain_id);
  return {
    format: "omni-chat-html-artifact-projection",
    version: 1,
    brainId: row.brain_id,
    originalBrainId: row.original_brain_id,
    artifactId: row.original_artifact_id,
    sourceActionId: row.source_action_id,
    contentSha256: row.content_sha256,
    title: row.title,
    interactive: row.interactive === 1,
    operationalState: row.operational_state,
    revision: row.revision,
    createdAt: row.created_at,
    savedArtifact: true,
    untrusted: true
  };
}

async function writeContentAddressed(path: string, bytes: Buffer, sha256: string): Promise<boolean> {
  await mkdir(dirname(path), { recursive: true, mode: 0o700 });
  try {
    const handle = await open(path, "wx", 0o600);
    try { await handle.writeFile(bytes); await handle.sync(); }
    finally { await handle.close(); }
    return true;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== bytes.length || await fileSha256(path) !== sha256) {
      throw new Error("HTML artifact content-addressed blob is corrupt.");
    }
    return false;
  }
}

export class HtmlArtifactStore {
  readonly root: string;
  readonly blobs: string;
  readonly databasePath: string;

  constructor(readonly brainRoot: string, readonly brainId: string) {
    requireId(brainId, "HTML artifact owner");
    this.root = join(brainRoot, "html-artifacts");
    this.blobs = join(this.root, "blobs");
    this.databasePath = join(this.root, HTML_ARTIFACT_DATABASE);
  }

  async initialize(): Promise<void> {
    await mkdir(this.blobs, { recursive: true, mode: 0o700 });
    for (const path of [this.root, this.blobs]) {
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("HTML artifact workspace is unsafe.");
    }
    this.database(() => undefined);
  }

  private database<T>(operation: (database: DatabaseSync) => T): T {
    const database = new DatabaseSync(sqliteFilename(this.databasePath));
    try {
      database.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=30000; PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT;
        CREATE TABLE IF NOT EXISTS artifacts(
          local_id TEXT PRIMARY KEY, brain_id TEXT NOT NULL, original_brain_id TEXT NOT NULL,
          original_artifact_id TEXT NOT NULL, source_action_id TEXT NOT NULL,
          content_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL, title TEXT NOT NULL,
          interactive INTEGER NOT NULL CHECK(interactive IN(0,1)),
          operational_state TEXT NOT NULL CHECK(operational_state IN('enabled','dormant')),
          revision INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL,
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL, lineage_json TEXT NOT NULL,
          UNIQUE(original_brain_id,original_artifact_id,source_action_id)
        ) STRICT;
        CREATE TABLE IF NOT EXISTS requests(
          source_action_id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, local_id TEXT NOT NULL,
          FOREIGN KEY(local_id) REFERENCES artifacts(local_id) ON DELETE RESTRICT
        ) STRICT;
        CREATE INDEX IF NOT EXISTS artifact_source ON artifacts(original_brain_id,original_artifact_id,source_action_id);
        CREATE INDEX IF NOT EXISTS artifact_content ON artifacts(content_sha256,local_id);`);
      database.prepare("INSERT OR IGNORE INTO meta VALUES('format','omni-chat-html-artifacts-1')").run();
      database.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(this.brainId);
      if (database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== this.brainId) {
        throw new Error("HTML artifact workspace belongs to another brain.");
      }
      return operation(database);
    } finally { database.close(); }
  }

  async create(sourceActionId: string, value: unknown): Promise<HtmlArtifactResult> {
    return withArtifactLease(this.root, () => this.createLocked(sourceActionId, value));
  }

  private async createLocked(sourceActionId: string, value: unknown): Promise<HtmlArtifactResult> {
    await this.initialize();
    requireId(sourceActionId, "HTML artifact source action");
    const content = normalizeHtmlArtifactContent(value);
    const contentJson = canonicalHtmlArtifactValue(content);
    const contentSha256 = htmlArtifactContentSha256(content);
    const requestSha256 = digest(canonicalHtmlArtifactValue({ sourceActionId, content }));
    const prior = this.database((database) => database.prepare(`SELECT a.* FROM requests r
      JOIN artifacts a ON a.local_id=r.local_id WHERE r.source_action_id=?`).get(sourceActionId) as unknown as PersistedHtmlArtifactRow | undefined);
    if (prior) {
      if (prior.request_sha256 !== requestSha256 || prior.content_sha256 !== contentSha256) {
        throw new Error("This HTML artifact action was already used with different content.");
      }
      return { format: HTML_ARTIFACT_RESULT_FORMAT, version: 1, artifact: projection(prior) };
    }
    const blobPath = join(this.blobs, `${contentSha256}.json`);
    const createdBlob = await writeContentAddressed(blobPath, Buffer.from(contentJson, "utf8"), contentSha256);
    const now = new Date().toISOString(), localId = randomUUID().replaceAll("-", "");
    const row = {
      local_id: localId,
      brain_id: this.brainId,
      original_brain_id: this.brainId,
      original_artifact_id: localId,
      source_action_id: sourceActionId,
      content_sha256: contentSha256,
      request_sha256: requestSha256,
      title: content.title,
      interactive: content.script ? 1 : 0,
      operational_state: "enabled" as const,
      revision: 1,
      created_at: now,
      updated_at: now,
      lineage_json: "[]"
    };
    const manifestSha256 = htmlArtifactManifestSha256(row);
    try {
      this.database((database) => {
        database.exec("BEGIN IMMEDIATE");
        try {
          const concurrent = database.prepare(`SELECT a.* FROM requests r JOIN artifacts a ON a.local_id=r.local_id
            WHERE r.source_action_id=?`).get(sourceActionId) as unknown as PersistedHtmlArtifactRow | undefined;
          if (concurrent) {
            if (concurrent.request_sha256 !== requestSha256 || concurrent.content_sha256 !== contentSha256) {
              throw new Error("This HTML artifact action was already used with different content.");
            }
            database.exec("COMMIT");
            return;
          }
          database.prepare(`INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)`).run(
            row.local_id,row.brain_id,row.original_brain_id,row.original_artifact_id,row.source_action_id,
            row.content_sha256,row.request_sha256,row.title,row.interactive,row.operational_state,row.revision,
            manifestSha256,row.created_at,row.updated_at,row.lineage_json
          );
          database.prepare("INSERT INTO requests VALUES(?,?,?)").run(sourceActionId, requestSha256, localId);
          database.exec("COMMIT");
        } catch (error) { database.exec("ROLLBACK"); throw error; }
      });
    } catch (error) {
      if (createdBlob) await rm(blobPath, { force: true }).catch(() => undefined);
      throw error;
    }
    const stored = this.database((database) => database.prepare(`SELECT a.* FROM requests r
      JOIN artifacts a ON a.local_id=r.local_id WHERE r.source_action_id=?`).get(sourceActionId) as unknown as PersistedHtmlArtifactRow);
    if (!stored || stored.request_sha256 !== requestSha256 || stored.content_sha256 !== contentSha256) {
      throw new Error("HTML artifact request receipt did not commit its exact content.");
    }
    return { format: HTML_ARTIFACT_RESULT_FORMAT, version: 1, artifact: projection(stored) };
  }

  async find(request: HtmlArtifactOpenRequest): Promise<PersistedHtmlArtifactRow> {
    await this.initialize();
    if (request.brainId !== this.brainId || !SAFE_ID.test(request.originalBrainId) ||
        !SAFE_ID.test(request.artifactId) || !SAFE_ID.test(request.sourceActionId) ||
        !SHA256.test(request.contentSha256)) throw new Error("Invalid HTML artifact identity.");
    const row = this.database((database) => database.prepare(`SELECT * FROM artifacts
      WHERE original_brain_id=? AND original_artifact_id=? AND source_action_id=? AND content_sha256=?`).get(
        request.originalBrainId, request.artifactId, request.sourceActionId, request.contentSha256
      ) as unknown as PersistedHtmlArtifactRow | undefined);
    if (!row) throw new Error("This saved HTML artifact is unavailable for the selected brain.");
    validatePersistedHtmlArtifactRow(row, this.brainId);
    return row;
  }

  async content(row: PersistedHtmlArtifactRow): Promise<HtmlArtifactContent> {
    validatePersistedHtmlArtifactRow(row, this.brainId);
    const path = join(this.blobs, `${row.content_sha256}.json`), info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || await fileSha256(path) !== row.content_sha256) {
      throw new Error("Saved HTML artifact content failed integrity validation.");
    }
    const value = validateHtmlArtifactContent(JSON.parse(await readFile(path, "utf8")));
    if (htmlArtifactContentSha256(value) !== row.content_sha256 || value.title !== row.title ||
        Boolean(value.script) !== (row.interactive === 1)) {
      throw new Error("Saved HTML artifact content differs from its record.");
    }
    return value;
  }

  async describe(request: HtmlArtifactOpenRequest): Promise<HtmlArtifactProjection> {
    return projection(await this.find(request));
  }

  async activate(request: HtmlArtifactOpenRequest & { expectedRevision: number }): Promise<HtmlArtifactProjection> {
    const row = await this.find(request);
    if (!Number.isSafeInteger(request.expectedRevision) || request.expectedRevision < 1 ||
        row.revision !== request.expectedRevision) throw new Error("HTML artifact review is stale; inspect its current revision.");
    if (row.operational_state === "enabled") return projection(row);
    const next = { ...row, operational_state: "enabled" as const, revision: row.revision + 1,
      updated_at: new Date().toISOString() };
    const without = { ...next }; delete (without as Partial<PersistedHtmlArtifactRow>).manifest_sha256;
    next.manifest_sha256 = htmlArtifactManifestSha256(without);
    this.database((database) => {
      const changed = database.prepare(`UPDATE artifacts SET operational_state='enabled',revision=?,updated_at=?,manifest_sha256=?
        WHERE local_id=? AND revision=? AND operational_state='dormant'`).run(
          next.revision, next.updated_at, next.manifest_sha256, row.local_id, row.revision
        ).changes;
      if (changed !== 1) throw new Error("HTML artifact review was superseded; no capability was enabled.");
    });
    return projection(next);
  }

  async rows(): Promise<PersistedHtmlArtifactRow[]> {
    await this.initialize();
    return this.database((database) => database.prepare("SELECT * FROM artifacts ORDER BY local_id").all() as unknown as PersistedHtmlArtifactRow[]);
  }

  static databasePath(brainRoot: string): string {
    return join(brainRoot, "html-artifacts", HTML_ARTIFACT_DATABASE);
  }
}

export function htmlArtifactProjection(row: PersistedHtmlArtifactRow): HtmlArtifactProjection {
  return projection(row);
}

export async function htmlArtifactBlobSha256(path: string): Promise<string> {
  return fileSha256(path);
}

export async function htmlArtifactWorkspaceExists(brainRoot: string): Promise<boolean> {
  return exists(HtmlArtifactStore.databasePath(brainRoot));
}
