import { DatabaseSync } from "node:sqlite";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { LocalTeacherItem, LocalTeacherJob, LocalTeacherSelection } from "../shared/localTeacher";
import type { LocalTeacherSelectedFile } from "./localTeacherFiles";
import { localTeacherCanonical, localTeacherGeneration, localTeacherHash, localTeacherId, localTeacherSha } from "./localTeacherFormat";
import { sqliteFilename } from "./sqliteFilename";

export const LOCAL_TEACHER_WORKSPACE = "local-teachers";
export interface StoredLocalTeacherSelection {
  selection: LocalTeacherSelection;
  runtime: LocalTeacherSelectedFile;
  model: LocalTeacherSelectedFile;
  requestId: string;
  requestSha256: string;
}
export interface LocalTeacherSessionSeal {
  format: "omni-local-teacher-custody-seal"; version: 1; sessionId: string; jobId: string;
  runtimeSha256: string; modelSha256: string; at: string;
  /** Exact opened descriptors/pathnames remained unchanged through shutdown. */
  dependencyCustody: "verified-before-and-after-owned-session";
  childClosed: true;
  receipts: Array<{ index: number; attemptId: string; sha256: string }>;
}
const jobStates = new Set(["queued", "running", "paused", "stopped", "waiting-load", "complete", "failed", "dormant"]);
export function validateLocalTeacherJob(value: unknown, owner: string): LocalTeacherJob {
  const row = value as LocalTeacherJob;
  if (!row || row.format !== "omni-local-teacher-job" || row.version !== 1 || row.brainId !== owner || !jobStates.has(row.state) ||
      !Number.isSafeInteger(row.revision) || row.revision < 1 || !Number.isSafeInteger(row.promptCount) || row.promptCount < 1 ||
      !Number.isSafeInteger(row.generatedCount) || !Number.isSafeInteger(row.learnedCount) || row.learnedCount < 0 ||
      row.generatedCount < row.learnedCount || row.generatedCount > row.promptCount || row.proof !== "attributed-generation-and-native-source-receipts") throw new Error("Saved local-teacher job is invalid.");
  for (const id of [row.id, row.originalBrainId, row.originalJobId, row.requestId, row.teacherId]) localTeacherId(id);
  localTeacherSha(row.requestSha256); localTeacherGeneration(row.generation);
  if (!Number.isSafeInteger(row.epochs) || row.epochs < 1 || !Number.isFinite(Date.parse(row.createdAt)) || !Number.isFinite(Date.parse(row.updatedAt))) throw new Error("Saved teacher epochs/timestamps are invalid.");
  return row;
}
export class LocalTeacherStore {
  private constructor(readonly db: DatabaseSync, readonly root: string, readonly brainId: string) {}
  static async open(root: string, brainId: string, create = true): Promise<LocalTeacherStore | undefined> {
    localTeacherId(brainId); const directory = join(root, LOCAL_TEACHER_WORKSPACE), path = join(directory, "workspace.sqlite3");
    if (create) await mkdir(directory, { recursive: true, mode: 0o700 });
    const present = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!present && !create) return undefined;
    const folder = await lstat(directory);
    if (!folder.isDirectory() || folder.isSymbolicLink() || present && (!present.isFile() || present.isSymbolicLink())) throw new Error("Local-teacher custody requires real owned files/directories.");
    const db = new DatabaseSync(sqliteFilename(path), { readOnly: !create });
    try {
      db.exec("PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (create) {
        db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; " +
          "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL); " +
          "CREATE TABLE IF NOT EXISTS selections(id TEXT PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,request_sha TEXT NOT NULL,payload TEXT NOT NULL); " +
          "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,request_id TEXT NOT NULL UNIQUE,request_sha TEXT NOT NULL,payload TEXT NOT NULL); " +
          "CREATE TABLE IF NOT EXISTS items(job_id TEXT NOT NULL,ordinal INTEGER NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(job_id,ordinal)); " +
          "CREATE TABLE IF NOT EXISTS session_seals(id TEXT NOT NULL,job_id TEXT NOT NULL,sha TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(id,job_id)); " +
          "CREATE TABLE IF NOT EXISTS merge_receipts(parent_id TEXT PRIMARY KEY,source_sha TEXT NOT NULL); " +
          "CREATE TABLE IF NOT EXISTS history(sequence INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,event TEXT NOT NULL,body TEXT NOT NULL);");
        db.prepare("INSERT OR IGNORE INTO meta VALUES('format','omni-local-teacher-workspace-1')").run();
        db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(brainId);
      }
      if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-local-teacher-workspace-1" ||
          db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Local-teacher workspace belongs to a different owner/format.");
      return new LocalTeacherStore(db, root, brainId);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  transaction<T>(work: () => T): T {
    this.db.exec("BEGIN IMMEDIATE"); try { const result = work(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  selection(id: string): StoredLocalTeacherSelection {
    const row = this.db.prepare("SELECT payload FROM selections WHERE id=?").get(localTeacherId(id));
    if (!row) throw new Error("Select or reapprove a current local teacher first.");
    const result = JSON.parse(String(row.payload)) as StoredLocalTeacherSelection;
    if (result.selection.brainId !== this.brainId || result.selection.id !== id || result.selection.currentHostAuthorized !== false) throw new Error("Saved teacher selection contains invalid owner/authority.");
    return result;
  }
  selections(): StoredLocalTeacherSelection[] { return this.db.prepare("SELECT id FROM selections ORDER BY rowid").all().map(row => this.selection(String(row.id))); }
  putSelection(row: StoredLocalTeacherSelection): StoredLocalTeacherSelection {
    if (row.selection.brainId !== this.brainId || row.selection.currentHostAuthorized !== false) throw new Error("Live host authorization must never be persisted or copied.");
    const previous = this.db.prepare("SELECT id,request_sha FROM selections WHERE request_id=?").get(localTeacherId(row.requestId));
    if (previous) { if (previous.request_sha !== row.requestSha256) throw new Error("Teacher selection request changed its exact dependencies."); return this.selection(String(previous.id)); }
    const existing = this.db.prepare("SELECT id FROM selections WHERE id=?").get(row.selection.id);
    if (existing) return this.selection(row.selection.id);
    this.db.prepare("INSERT INTO selections VALUES(?,?,?,?)").run(row.selection.id, row.requestId, row.requestSha256, localTeacherCanonical(row)); return row;
  }
  reapproveSelection(row: StoredLocalTeacherSelection): void {
    const previous = this.selection(row.selection.id);
    if (row.selection.currentHostAuthorized !== false || row.selection.brainId !== this.brainId ||
        previous.selection.runtime.sha256 !== row.runtime.sha256 || previous.selection.model.sha256 !== row.model.sha256 || previous.selection.runner !== row.selection.runner)
      throw new Error("Reapproval cannot rewrite a teacher dependency or copy a live grant.");
    this.transaction(() => {
      this.event(row.selection.id, "local-dependencies-reselected", { previous, selected: row });
      this.db.prepare("UPDATE selections SET payload=? WHERE id=?").run(localTeacherCanonical(row), row.selection.id);
    });
  }
  job(id: string): LocalTeacherJob {
    const row = this.db.prepare("SELECT payload FROM jobs WHERE id=?").get(localTeacherId(id));
    if (!row) throw new Error("Local-teacher job was not found."); return validateLocalTeacherJob(JSON.parse(String(row.payload)), this.brainId);
  }
  jobs(): LocalTeacherJob[] { return this.db.prepare("SELECT id FROM jobs ORDER BY rowid DESC").all().map(row => this.job(String(row.id))); }
  jobByRequest(requestId: string): LocalTeacherJob | undefined {
    const row = this.db.prepare("SELECT id FROM jobs WHERE request_id=?").get(localTeacherId(requestId)); return row ? this.job(String(row.id)) : undefined;
  }
  createJob(job: LocalTeacherJob, prompts: readonly string[]): LocalTeacherJob {
    validateLocalTeacherJob(job, this.brainId);
    return this.transaction(() => {
      const previous = this.jobByRequest(job.requestId);
      if (previous) { if (previous.requestSha256 !== job.requestSha256) throw new Error("Local-teacher retry has different complete prompts/settings."); return previous; }
      this.db.prepare("INSERT INTO jobs VALUES(?,?,?,?)").run(job.id, job.requestId, job.requestSha256, localTeacherCanonical(job));
      const insert = this.db.prepare("INSERT INTO items VALUES(?,?,?)");
      prompts.forEach((prompt, index) => { const item: LocalTeacherItem = { jobId: job.id, index, prompt, inputSha256: localTeacherHash(prompt), state: "pending" }; insert.run(job.id, index, localTeacherCanonical(item)); });
      this.event(job.id, "created", job); return job;
    });
  }
  event(jobId: string, event: string, body: unknown): void {
    this.db.prepare("INSERT INTO history(job_id,event,body) VALUES(?,?,?)").run(jobId, event, localTeacherCanonical({ at: new Date().toISOString(), body }));
  }
  update(jobId: string, patch: Partial<LocalTeacherJob>, event = "updated"): LocalTeacherJob {
    const old = this.job(jobId);
    if (["id", "brainId", "originalBrainId", "originalJobId", "requestId", "requestSha256", "teacherId", "generation", "epochs", "promptCount"].some(key => key in patch)) throw new Error("Teacher job input/source identity is immutable.");
    const next = { ...old, ...patch, revision: old.revision + 1, updatedAt: new Date().toISOString() };
    for (const key of Object.keys(next) as Array<keyof LocalTeacherJob>) if (next[key] === undefined) delete next[key];
    validateLocalTeacherJob(next, this.brainId);
    this.db.prepare("UPDATE jobs SET payload=? WHERE id=?").run(localTeacherCanonical(next), jobId); this.event(jobId, event, next); return next;
  }
  item(jobId: string, index: number): LocalTeacherItem {
    const row = this.db.prepare("SELECT payload FROM items WHERE job_id=? AND ordinal=?").get(localTeacherId(jobId), index);
    if (!row) throw new Error("Teacher prompt/source position was not found.");
    const item = JSON.parse(String(row.payload)) as LocalTeacherItem;
    if (item.jobId !== jobId || item.index !== index || localTeacherHash(item.prompt) !== item.inputSha256 ||
        !["pending", "generating", "generated", "learning", "complete", "interrupted"].includes(item.state)) throw new Error("Saved teacher prompt/source identity is corrupt.");
    return item;
  }
  itemBytes(jobId: string, afterIndex: number, limit: number): number {
    const row = this.db.prepare("SELECT COALESCE(SUM(n),0) AS n FROM (SELECT LENGTH(CAST(payload AS BLOB)) AS n FROM items WHERE job_id=? AND ordinal>? ORDER BY ordinal LIMIT ?)").get(jobId, afterIndex, limit);
    return Number(row?.n ?? 0);
  }
  items(jobId: string, afterIndex = -1, limit = 20): LocalTeacherItem[] {
    return this.db.prepare("SELECT ordinal FROM items WHERE job_id=? AND ordinal>? ORDER BY ordinal LIMIT ?").all(localTeacherId(jobId), afterIndex, limit).map(row => this.item(jobId, Number(row.ordinal)));
  }
  updateItem(jobId: string, index: number, patch: Partial<LocalTeacherItem>): LocalTeacherItem {
    if (["jobId", "index", "prompt", "inputSha256"].some(key => key in patch)) throw new Error("Teacher prompt identity cannot be rewritten.");
    const next = { ...this.item(jobId, index), ...patch };
    for (const key of Object.keys(next) as Array<keyof LocalTeacherItem>) if (next[key] === undefined) delete next[key];
    this.db.prepare("UPDATE items SET payload=? WHERE job_id=? AND ordinal=?").run(localTeacherCanonical(next), jobId, index);
    this.event(jobId, "item-updated", { index, item: next }); return next;
  }
  sealSession(seal: LocalTeacherSessionSeal): void {
    const job = this.job(seal.jobId), teacher = this.selection(job.teacherId).selection;
    if (seal.format !== "omni-local-teacher-custody-seal" || seal.version !== 1 || seal.childClosed !== true ||
        seal.dependencyCustody !== "verified-before-and-after-owned-session" || teacher.runtime.sha256 !== seal.runtimeSha256 || teacher.model.sha256 !== seal.modelSha256)
      throw new Error("Invalid local teacher dependency-custody seal.");
    if (!Array.isArray(seal.receipts) || new Set(seal.receipts.map(row => row.index)).size !== seal.receipts.length || !Number.isFinite(Date.parse(seal.at)))
      throw new Error("Teacher custody positions or timestamp are invalid.");
    localTeacherId(seal.sessionId);
    this.transaction(() => {
      const sha = localTeacherHash(localTeacherCanonical(seal)), existing = this.db.prepare("SELECT sha FROM session_seals WHERE id=? AND job_id=?").get(seal.sessionId, job.id);
      if (existing && existing.sha !== sha) throw new Error("Local teacher session seal changed.");
      for (const row of seal.receipts) {
        const item = this.item(job.id, row.index);
        if (!item.receipt || item.attemptId !== row.attemptId || item.receipt.state !== "complete" || localTeacherHash(localTeacherCanonical(item.receipt)) !== row.sha256)
          throw new Error("Teacher candidate has no exact completed session receipt.");
      }
      this.db.prepare("INSERT OR IGNORE INTO session_seals VALUES(?,?,?,?)").run(seal.sessionId, job.id, sha, localTeacherCanonical(seal));
      for (const row of seal.receipts) this.updateItem(job.id, row.index, { state: "generated" });
      const generatedCount = Number(this.db.prepare("SELECT COUNT(*) AS n FROM items WHERE job_id=? AND json_extract(payload,'$.state') IN ('generated','learning','complete')").get(job.id)?.n ?? 0);
      this.update(job.id, { generatedCount }, "dependency-custody-sealed");
    });
  }
  hasSealedReceipt(jobId: string, item: LocalTeacherItem): boolean {
    if (!item.receipt || !item.attemptId) return false;
    const sha = localTeacherHash(localTeacherCanonical(item.receipt));
    for (const row of this.db.prepare("SELECT payload FROM session_seals WHERE job_id=?").iterate(jobId)) {
      const seal = JSON.parse(String(row.payload)) as LocalTeacherSessionSeal;
      if (seal.receipts.some(value => value.index === item.index && value.attemptId === item.attemptId && value.sha256 === sha)) return true;
    }
    return false;
  }
  recover(): void {
    for (const job of this.jobs()) if (["queued", "running", "waiting-load"].includes(job.state))
      this.update(job.id, { state: "paused", phase: "restart-review", error: "Local teacher paused after restart. Reapprove exact local dependencies before unfinished generation; committed sources/cursors are retained." }, "recovered-without-execution");
  }
}
