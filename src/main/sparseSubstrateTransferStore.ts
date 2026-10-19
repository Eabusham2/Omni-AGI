import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { SparseSubstrateBundle, SparseSubstrateJob, SparseSubstratePlan } from "../shared/sparseSubstrateTransfer";
import { sqliteFilename } from "./sqliteFilename";

export const SPARSE_TRANSFER_DIRECTORY = "substrate-transfers";
const FORMAT = "omni-sparse-substrate-workspace-1";
const declarations = {
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  artifacts: "CREATE TABLE artifacts(id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL)",
  requests: "CREATE TABLE requests(id TEXT PRIMARY KEY,action TEXT NOT NULL,sha TEXT NOT NULL,result TEXT,state TEXT NOT NULL)",
  jobs: "CREATE TABLE jobs(id TEXT PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,payload TEXT NOT NULL)",
  heads: "CREATE TABLE heads(sha TEXT PRIMARY KEY,source_brain TEXT NOT NULL,payload TEXT NOT NULL)",
  history: "CREATE TABLE history(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,body TEXT NOT NULL)",
};
export function sparseId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid sparse transfer handle.");
  return value;
}
export function sparseHash(value: unknown): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Invalid sparse transfer SHA256.");
  return value;
}
const normalized = (sql: unknown): string => String(sql).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/\s+/g, "").replace(/;$/, "").toUpperCase();
function validateSchema(db: DatabaseSync): void {
  const seen = new Set<string>();
  for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
    if (row.type === "index" && /^sqlite_autoindex_/.test(String(row.name)) && row.sql === null) continue;
    if (row.name === "sqlite_sequence" && row.type === "table" && normalized(row.sql) === "CREATETABLESQLITE_SEQUENCE(NAME,SEQ)") continue;
    const declaration = declarations[row.name as keyof typeof declarations];
    if (row.type !== "table" || !declaration || normalized(row.sql) !== normalized(declaration)) throw new Error("Sparse workspace contains unsupported executable/schema objects.");
    seen.add(String(row.name));
  }
  if (seen.size !== Object.keys(declarations).length || db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Sparse workspace schema/integrity failed.");
}
export function validateSparseJob(value: unknown, brainId: string): SparseSubstrateJob {
  const row = value as SparseSubstrateJob;
  if (!row || row.format !== "omni-sparse-substrate-job" || row.version !== 1 || row.scope !== "native-sparse-assemblies" || row.brainId !== brainId || row.liveAuthorityPersisted !== false ||
      !["queued", "boundary-required", "applying", "complete", "needs-review", "cancelled", "failed", "uncertain", "dormant"].includes(row.state) ||
      !Number.isSafeInteger(row.revision) || row.revision < 1 || !Number.isFinite(Date.parse(row.createdAt)) || !Number.isFinite(Date.parse(row.updatedAt))) throw new Error("Invalid sparse transfer job custody.");
  for (const id of [row.id, row.originalBrainId, row.requestId, row.planId]) sparseId(id);
  sparseHash(row.requestSha256); sparseHash(row.planSha256);
  if (typeof row.authorizationRevision !== "string" || !row.authorizationRevision) throw new Error("Sparse job has no audited authority revision.");
  return row;
}

/** Full raw artifacts/history are retained; pending rows never contain grants. */
export class SparseSubstrateTransferStore {
  private constructor(readonly root: string, readonly brainId: string, readonly db: DatabaseSync) {}
  static async open(root: string, brainId: string): Promise<SparseSubstrateTransferStore> {
    sparseId(brainId); const directory = join(root, SPARSE_TRANSFER_DIRECTORY);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    const folder = await lstat(directory);
    if (!folder.isDirectory() || folder.isSymbolicLink()) throw new Error("Sparse transfer custody requires a real owned directory.");
    const path = join(directory, "workspace.sqlite3");
    const present = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (present && (!present.isFile() || present.isSymbolicLink())) throw new Error("Sparse transfer workspace is unsafe.");
    const db = new DatabaseSync(sqliteFilename(path), { allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;");
      for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
        if (row.type === "index" && /^sqlite_autoindex_/.test(String(row.name)) && row.sql === null) continue;
        if (row.name === "sqlite_sequence" && row.type === "table") continue;
        const declaration = declarations[row.name as keyof typeof declarations];
        if (row.type !== "table" || !declaration || normalized(row.sql) !== normalized(declaration)) throw new Error("Sparse workspace contains unsupported executable/schema objects.");
      }
      for (const sql of Object.values(declarations)) db.exec(sql.replace("CREATE TABLE", "CREATE TABLE IF NOT EXISTS"));
      db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(FORMAT);
      db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT || db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId || db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Sparse workspace identity/integrity failed.");
      return new SparseSubstrateTransferStore(root, brainId, db);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  validate(): void {
    validateSchema(this.db);
    if (this.db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== FORMAT || this.db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== this.brainId ||
        this.db.prepare("SELECT COUNT(*) AS n FROM meta").get()?.n !== 2) throw new Error("Sparse workspace metadata owner/format differs.");
    for (const row of this.db.prepare("SELECT id,kind,payload FROM artifacts").iterate()) {
      if (row.kind !== "bundle" && row.kind !== "plan") throw new Error("Unknown sparse artifact kind.");
      const artifact = this.artifact(String(row.id), row.kind);
      if (artifact.version !== 1 || !Number.isSafeInteger(artifact.bytes) || artifact.bytes < 1 || !Number.isSafeInteger(artifact.rows) || artifact.rows < 0 ||
          createHash("sha256").update(artifact.manifestJson).digest("hex") !== artifact.manifestSha256) throw new Error("Sparse artifact descriptor/manifest hash failed.");
      const manifest = JSON.parse(artifact.manifestJson) as { scope?: unknown; version?: unknown; rows?: unknown };
      if (manifest.scope !== artifact.scope || manifest.version !== 1 || manifest.rows !== artifact.rows) throw new Error("Sparse artifact declared scope/coverage differs.");
    }
    for (const row of this.db.prepare("SELECT id,request_id,payload FROM jobs").iterate()) {
      const job = validateSparseJob(JSON.parse(String(row.payload)), this.brainId);
      if (job.id !== row.id || job.requestId !== row.request_id) throw new Error("Sparse job indexed owner/identity differs.");
    }
    for (const row of this.db.prepare("SELECT id,action,sha,state,result FROM requests").iterate()) {
      sparseId(row.id); sparseHash(row.sha);
      if (!["export", "register-bundle", "prepare-import", "prepare-rollback", "apply"].includes(String(row.action)) || !["prepared", "terminal"].includes(String(row.state)) || row.state === "terminal" && row.result === null) throw new Error("Sparse request state/action is invalid.");
      if (row.result !== null) { const value: unknown = JSON.parse(String(row.result)); if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Sparse request receipt is not typed data."); }
    }
    for (const row of this.db.prepare("SELECT sha,source_brain,payload FROM heads").iterate()) {
      sparseHash(row.sha); sparseId(row.source_brain); const raw = String(row.payload), head = JSON.parse(raw) as Record<string, unknown>;
      if (createHash("sha256").update(raw).digest("hex") !== row.sha || head.brainId !== row.source_brain || head.format !== "omni-sparse-substrate-applied" || head.version !== 1 || head.scope !== "native-sparse-assemblies" ||
          head.weightsAveraged !== false || head.cortexParametersChanged !== false || head.originChanged !== false) throw new Error("Archived sparse native head is invalid.");
    }
    for (const row of this.db.prepare("SELECT job_id,body FROM history ORDER BY sequence").iterate()) {
      const job = validateSparseJob(JSON.parse(String(row.body)), this.brainId);
      if (job.id !== row.job_id) throw new Error("Sparse history indexed job differs.");
    }
  }
  static async openFile(path: string, expectedBrainId: string): Promise<SparseSubstrateTransferStore> {
    sparseId(expectedBrainId); const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink()) throw new Error("Captured sparse workspace is not a regular immutable file.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: true, allowExtension: false });
    try { db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-128; PRAGMA mmap_size=0;");
      const value = new SparseSubstrateTransferStore("", expectedBrainId, db); value.validate(); return value;
    } catch (error) { db.close(); throw error; }
  }
  request(id: string, action: string, sha: string): { state: string; result?: unknown } | undefined {
    const row = this.db.prepare("SELECT action,sha,state,result FROM requests WHERE id=?").get(sparseId(id));
    if (!row) return undefined;
    if (row.action !== action || row.sha !== sparseHash(sha)) throw new Error("Sparse transfer request ID was reused with different operands.");
    return { state: String(row.state), ...(row.result === null ? {} : { result: JSON.parse(String(row.result)) as unknown }) };
  }
  beginRequest(id: string, action: string, sha: string): void { this.db.prepare("INSERT INTO requests VALUES(?,?,?,NULL,'prepared')").run(sparseId(id), action, sparseHash(sha)); }
  finishRequest(id: string, result: unknown): void { this.db.prepare("UPDATE requests SET result=?,state='terminal' WHERE id=?").run(JSON.stringify(result), sparseId(id)); }
  artifact<T extends SparseSubstrateBundle | SparseSubstratePlan>(id: string, kind: "bundle" | "plan"): T {
    const row = this.db.prepare("SELECT kind,payload FROM artifacts WHERE id=?").get(sparseId(id));
    if (!row || row.kind !== kind) throw new Error("Sparse artifact handle is not owned by this instance.");
    const value = JSON.parse(String(row.payload)) as T;
    if (value.id !== id || value.scope !== "native-sparse-assemblies" || !new RegExp(`^${SPARSE_TRANSFER_DIRECTORY}/(?:bundles|plans)/[A-Za-z0-9._-]+\\.sqlite3$`).test(value.relativePath)) throw new Error("Sparse artifact custody identity/path is invalid.");
    sparseHash(value.sha256); sparseHash(value.manifestSha256);
    return value;
  }
  putArtifact(kind: "bundle" | "plan", value: SparseSubstrateBundle | SparseSubstratePlan): void {
    sparseId(value.id); sparseHash(value.sha256); this.db.prepare("INSERT INTO artifacts VALUES(?,?,?)").run(value.id, kind, JSON.stringify(value));
  }
  job(id: string): SparseSubstrateJob {
    const row = this.db.prepare("SELECT payload FROM jobs WHERE id=?").get(sparseId(id));
    if (!row) throw new Error("Sparse transfer job does not exist for this instance.");
    return validateSparseJob(JSON.parse(String(row.payload)), this.brainId);
  }
  jobs(after = "", limit = 50): { entries: SparseSubstrateJob[]; nextAfter?: string } {
    if (after) sparseId(after);
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("Invalid sparse job page size.");
    const rows = this.db.prepare("SELECT id FROM jobs WHERE id>? ORDER BY id LIMIT ?").all(after, limit + 1);
    return { entries: rows.slice(0, limit).map(row => this.job(String(row.id))), ...(rows.length > limit ? { nextAfter: String(rows[limit - 1]!.id) } : {}) };
  }
  createJob(job: SparseSubstrateJob): void {
    validateSparseJob(job, this.brainId);
    this.db.exec("BEGIN IMMEDIATE");
    try { this.db.prepare("INSERT INTO jobs VALUES(?,?,?)").run(job.id, job.requestId, JSON.stringify(job));
      this.db.prepare("INSERT INTO history(job_id,body) VALUES(?,?)").run(job.id, JSON.stringify(job)); this.db.exec("COMMIT"); }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  updateJob(id: string, patch: Partial<Pick<SparseSubstrateJob, "state" | "error" | "receipt">>): SparseSubstrateJob {
    const old = this.job(id), next = validateSparseJob({ ...old, ...patch, updatedAt: new Date().toISOString(), revision: old.revision + 1 }, this.brainId);
    this.db.exec("BEGIN IMMEDIATE");
    try { this.db.prepare("UPDATE jobs SET payload=? WHERE id=?").run(JSON.stringify(next), id);
      this.db.prepare("INSERT INTO history(job_id,body) VALUES(?,?)").run(id, JSON.stringify(next)); this.db.exec("COMMIT"); return next; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  archiveHead(sourceBrainId: string, receiptSha256: string, receiptJson: string): void {
    const prior = this.db.prepare("SELECT source_brain,payload FROM heads WHERE sha=?").get(sparseHash(receiptSha256));
    if (prior && (prior.source_brain !== sourceBrainId || prior.payload !== receiptJson)) throw new Error("Sparse native archival receipt conflicts.");
    this.db.prepare("INSERT OR IGNORE INTO heads VALUES(?,?,?)").run(receiptSha256, sparseId(sourceBrainId), receiptJson);
  }
  /** Called once after process restart or owner adoption; never auto-dispatch. */
  retirePending(): number {
    let changed = 0;
    for (const row of this.db.prepare("SELECT id,payload FROM jobs").iterate()) {
      const job = validateSparseJob(JSON.parse(String(row.payload)), this.brainId);
      if (["queued", "boundary-required", "applying"].includes(job.state)) { this.updateJob(String(row.id), { state: job.state === "applying" ? "uncertain" : "dormant", error: "No live activation/Ask/runtime grant survived the process or ownership boundary; reconcile before a newly reviewed action." }); changed++; }
    }
    return changed;
  }
  static id(): string { return randomUUID(); }
}

export async function validateSparseTransferDatabaseFile(path: string, expectedBrainId: string): Promise<void> {
  const store = await SparseSubstrateTransferStore.openFile(path, expectedBrainId); try { store.validate(); } finally { store.close(); }
}
