import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { lstat, opendir, readFile, rm } from "node:fs/promises";
import { basename, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import {
  HtmlArtifactStore,
  htmlArtifactManifestSha256,
  validatePersistedHtmlArtifactRow,
  type HtmlArtifactLineage,
  type PersistedHtmlArtifactRow
} from "./htmlArtifactStore";
import { canonicalHtmlArtifactValue, htmlArtifactContentSha256, validateHtmlArtifactContent } from "./htmlArtifactFormat";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const EXPECTED_SCHEMA = [
  ["index", "artifact_content", "artifacts", "CREATE INDEX artifact_content ON artifacts(content_sha256,local_id)"],
  ["index", "artifact_source", "artifacts", "CREATE INDEX artifact_source ON artifacts(original_brain_id,original_artifact_id,source_action_id)"],
  ["table", "artifacts", "artifacts", `CREATE TABLE artifacts(
    local_id TEXT PRIMARY KEY, brain_id TEXT NOT NULL, original_brain_id TEXT NOT NULL,
    original_artifact_id TEXT NOT NULL, source_action_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL, request_sha256 TEXT NOT NULL, title TEXT NOT NULL,
    interactive INTEGER NOT NULL CHECK(interactive IN(0,1)),
    operational_state TEXT NOT NULL CHECK(operational_state IN('enabled','dormant')),
    revision INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, lineage_json TEXT NOT NULL,
    UNIQUE(original_brain_id,original_artifact_id,source_action_id)
  ) STRICT`],
  ["table", "meta", "meta", "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL) STRICT"],
  ["table", "requests", "requests", `CREATE TABLE requests(
    source_action_id TEXT PRIMARY KEY, request_sha256 TEXT NOT NULL, local_id TEXT NOT NULL,
    FOREIGN KEY(local_id) REFERENCES artifacts(local_id) ON DELETE RESTRICT
  ) STRICT`]
] as const;

function normalizedSchemaSql(value: unknown): string {
  return typeof value === "string" ? value.replace(/\s+/g, " ").trim() : "";
}

function validateSchema(database: DatabaseSync): void {
  const observed = database.prepare(`SELECT type,name,tbl_name,sql FROM sqlite_schema
    WHERE name NOT GLOB 'sqlite_*' ORDER BY type,name`).all() as Array<Record<string, unknown>>;
  const actual = observed.map((row) => [String(row.type), String(row.name), String(row.tbl_name), normalizedSchemaSql(row.sql)]);
  const expected = EXPECTED_SCHEMA.map(([type, name, table, sql]) => [type, name, table, normalizedSchemaSql(sql)]);
  if (canonicalHtmlArtifactValue(actual) !== canonicalHtmlArtifactValue(expected)) {
    throw new Error("HTML artifact registry contains unsupported executable/schema objects.");
  }
}

function digest(value: string): string {
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

function withoutManifest(row: PersistedHtmlArtifactRow): Omit<PersistedHtmlArtifactRow, "manifest_sha256"> {
  const copy = { ...row };
  delete (copy as Partial<PersistedHtmlArtifactRow>).manifest_sha256;
  return copy;
}

async function validateBlob(path: string, expectedSha256: string): Promise<void> {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || await fileSha256(path) !== expectedSha256) {
    throw new Error("Portable HTML artifact blob failed its content checksum.");
  }
  const content = validateHtmlArtifactContent(JSON.parse(await readFile(path, "utf8")));
  if (htmlArtifactContentSha256(content) !== expectedSha256) {
    throw new Error("Portable HTML artifact blob differs from its canonical content.");
  }
}

export async function validateHtmlArtifactWorkspace(brainRoot: string, brainId: string): Promise<boolean> {
  if (!SAFE_ID.test(brainId)) throw new Error("Invalid HTML artifact workspace owner.");
  const root = join(brainRoot, "html-artifacts"), databasePath = HtmlArtifactStore.databasePath(brainRoot);
  if (!await exists(databasePath)) {
    if (!await exists(root)) return false;
    const directory = await opendir(root);
    for await (const entry of directory) {
      if (!["index.sqlite3-wal", "index.sqlite3-shm", "index.sqlite3-journal"].includes(entry.name)) {
        throw new Error("HTML artifact workspace has no registry.");
      }
    }
    return false;
  }
  const info = await lstat(databasePath);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("HTML artifact registry is unsafe.");
  const rootDirectory = await opendir(root);
  for await (const entry of rootDirectory) {
    if (!["index.sqlite3", "index.sqlite3-wal", "index.sqlite3-shm", "index.sqlite3-journal", "blobs"].includes(entry.name) ||
        entry.isSymbolicLink() || (entry.name === "blobs" ? !entry.isDirectory() : !entry.isFile())) {
      throw new Error("HTML artifact workspace contains an unsupported payload.");
    }
  }
  const database = new DatabaseSync(databasePath, { readOnly: true });
  let rows: PersistedHtmlArtifactRow[];
  try {
    validateSchema(database);
    if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" ||
        database.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-chat-html-artifacts-1" ||
        database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) {
      throw new Error("HTML artifact registry failed integrity validation.");
    }
    rows = database.prepare("SELECT * FROM artifacts ORDER BY local_id").all() as unknown as PersistedHtmlArtifactRow[];
    for (const row of rows) validatePersistedHtmlArtifactRow(row, brainId);
    for (const request of database.prepare("SELECT * FROM requests ORDER BY source_action_id").all() as Array<Record<string, unknown>>) {
      const row = rows.find((candidate) => candidate.local_id === request.local_id);
      if (!row || request.source_action_id !== row.source_action_id || request.request_sha256 !== row.request_sha256 ||
          row.original_brain_id !== brainId || !SHA256.test(String(request.request_sha256))) {
        throw new Error("HTML artifact request receipt is invalid or grants a copied action.");
      }
    }
  } finally { database.close(); }
  const referenced = new Set(rows.map((row) => row.content_sha256));
  const blobs = join(root, "blobs"), blobInfo = await lstat(blobs);
  if (!blobInfo.isDirectory() || blobInfo.isSymbolicLink()) throw new Error("HTML artifact blob directory is unsafe.");
  const observed = new Set<string>(), directory = await opendir(blobs);
  for await (const entry of directory) {
    if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/.test(entry.name)) throw new Error("HTML artifact blob inventory is unsafe.");
    const sha256 = basename(entry.name, ".json");
    if (!referenced.has(sha256)) throw new Error("HTML artifact workspace contains an unreferenced blob.");
    await validateBlob(join(blobs, entry.name), sha256);
    observed.add(sha256);
  }
  if (observed.size !== referenced.size) throw new Error("HTML artifact workspace is missing content bytes.");
  return true;
}

/** New owners retain exact presentation bytes but never inherit bridge authority. */
export async function rekeyHtmlArtifactWorkspace(
  brainRoot: string,
  sourceBrainId: string,
  targetBrainId: string
): Promise<void> {
  if (!await validateHtmlArtifactWorkspace(brainRoot, sourceBrainId)) return;
  if (!SAFE_ID.test(targetBrainId)) throw new Error("Invalid HTML artifact target owner.");
  const database = new DatabaseSync(HtmlArtifactStore.databasePath(brainRoot));
  try {
    database.exec("PRAGMA foreign_keys=ON; BEGIN IMMEDIATE");
    try {
      database.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
      if (sourceBrainId !== targetBrainId) {
        const now = new Date().toISOString();
        for (const row of database.prepare("SELECT * FROM artifacts ORDER BY local_id").all() as unknown as PersistedHtmlArtifactRow[]) {
          const prior = JSON.parse(row.lineage_json) as HtmlArtifactLineage[];
          const next = { ...row, brain_id: targetBrainId, operational_state: "dormant" as const,
            revision: row.revision + 1, updated_at: now,
            lineage_json: canonicalHtmlArtifactValue([...prior, { brainId: sourceBrainId,
              artifactId: row.local_id, kind: "copy", recordedAt: now }]) };
          next.manifest_sha256 = htmlArtifactManifestSha256(withoutManifest(next));
          database.prepare(`UPDATE artifacts SET brain_id=?,operational_state=?,revision=?,updated_at=?,lineage_json=?,manifest_sha256=?
            WHERE local_id=?`).run(next.brain_id,next.operational_state,next.revision,next.updated_at,
              next.lineage_json,next.manifest_sha256,next.local_id);
        }
        database.prepare("DELETE FROM requests").run();
      }
      database.exec("COMMIT");
    } catch (error) { database.exec("ROLLBACK"); throw error; }
  } finally { database.close(); }
  await validateHtmlArtifactWorkspace(brainRoot, targetBrainId);
}

/** Merge preserves visible artifacts and provenance, with every bridge dormant. */
export async function mergeHtmlArtifactWorkspace(
  sourceRoot: string,
  targetRoot: string,
  targetBrainId: string,
  parentBrainId: string,
  reserve?: (bytes: number) => Promise<void>
): Promise<void> {
  if (!await validateHtmlArtifactWorkspace(sourceRoot, targetBrainId)) return;
  const target = new HtmlArtifactStore(targetRoot, targetBrainId);
  await target.initialize();
  const sourceDb = new DatabaseSync(HtmlArtifactStore.databasePath(sourceRoot), { readOnly: true });
  const targetDb = new DatabaseSync(HtmlArtifactStore.databasePath(targetRoot));
  const copied: string[] = [];
  try {
    const rows = sourceDb.prepare("SELECT * FROM artifacts ORDER BY local_id").all() as unknown as PersistedHtmlArtifactRow[];
    for (const row of rows) {
      const destination = join(targetRoot, "html-artifacts", "blobs", `${row.content_sha256}.json`);
      if (await exists(destination)) {
        await validateBlob(destination, row.content_sha256);
        continue;
      }
      const source = join(sourceRoot, "html-artifacts", "blobs", `${row.content_sha256}.json`);
      const bytes = (await lstat(source)).size;
      await reserve?.(bytes);
      await copyMutableFileIsolated(source, destination, { expectedSha256: row.content_sha256 });
      copied.push(row.content_sha256);
    }
    targetDb.exec("PRAGMA foreign_keys=ON; BEGIN IMMEDIATE");
    try {
      const now = new Date().toISOString();
      for (const row of rows) {
        const existing = targetDb.prepare(`SELECT * FROM artifacts WHERE original_brain_id=? AND
          original_artifact_id=? AND source_action_id=?`).get(
            row.original_brain_id,row.original_artifact_id,row.source_action_id
          ) as unknown as PersistedHtmlArtifactRow | undefined;
        if (existing) {
          if (existing.content_sha256 !== row.content_sha256 || existing.request_sha256 !== row.request_sha256) {
            throw new Error("Merged HTML artifact provenance collided with different content.");
          }
          continue;
        }
        const localId = digest(canonicalHtmlArtifactValue({ parentBrainId, originalBrainId: row.original_brain_id,
          artifactId: row.original_artifact_id, sourceActionId: row.source_action_id,
          contentSha256: row.content_sha256 })).slice(0, 32);
        const collision = targetDb.prepare("SELECT content_sha256 FROM artifacts WHERE local_id=?").get(localId) as { content_sha256?: string } | undefined;
        if (collision && collision.content_sha256 !== row.content_sha256) throw new Error("Merged HTML artifact local identity collided.");
        const prior = JSON.parse(row.lineage_json) as HtmlArtifactLineage[];
        const next: PersistedHtmlArtifactRow = { ...row, local_id: localId, brain_id: targetBrainId,
          operational_state: "dormant", revision: row.revision + 1, updated_at: now,
          lineage_json: canonicalHtmlArtifactValue([...prior, { brainId: parentBrainId,
            artifactId: row.local_id, kind: "merge", recordedAt: now }]), manifest_sha256: "" };
        next.manifest_sha256 = htmlArtifactManifestSha256(withoutManifest(next));
        targetDb.prepare("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)").run(
          next.local_id,next.brain_id,next.original_brain_id,next.original_artifact_id,next.source_action_id,
          next.content_sha256,next.request_sha256,next.title,next.interactive,next.operational_state,next.revision,
          next.manifest_sha256,next.created_at,next.updated_at,next.lineage_json
        );
      }
      targetDb.exec("COMMIT");
    } catch (error) { targetDb.exec("ROLLBACK"); throw error; }
  } catch (error) {
    for (const sha256 of copied) await rm(join(targetRoot, "html-artifacts", "blobs", `${sha256}.json`), { force: true }).catch(() => undefined);
    throw error;
  } finally { sourceDb.close(); targetDb.close(); }
  await validateHtmlArtifactWorkspace(targetRoot, targetBrainId);
}

export function safeHtmlArtifactWorkspacePath(value: string): boolean {
  return value === "html-artifacts/index.sqlite3" || /^html-artifacts\/blobs\/[a-f0-9]{64}\.json$/.test(value);
}
