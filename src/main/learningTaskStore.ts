import { createHash } from "node:crypto";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { FactCheckClaim, FactCheckEvidence, FactCheckFinding, FactCheckWorkspace, LearningSourceProgress, LearningTask, LearningTaskState, WebLearningPolicy } from "../shared/learning";
import { validateWebLearningPolicy } from "./webLearningPolicy";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
export function learningId(value: unknown, label = "learning identity"): string {
  if (typeof value !== "string" || !ID.test(value)) throw new Error(`Invalid ${label}.`);
  return value;
}
export function learningHash(value: unknown): string {
  const canonical = (item: unknown): unknown => Array.isArray(item) ? item.map(canonical) :
    item && typeof item === "object" ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => [key, canonical(value)])) : item;
  return createHash("sha256").update(JSON.stringify(canonical(value))).digest("hex");
}
export interface StoredLearningSource extends LearningSourceProgress { spec: Record<string, unknown> }
type Row = Record<string, unknown>;
function parse<T>(value: unknown): T { if (typeof value !== "string") throw new Error("Saved learning record is invalid."); return JSON.parse(value) as T; }

/** Operational jobs/evidence only. Neural memory remains in the native worker. */
export class LearningTaskStore {
  private constructor(readonly directory: string, readonly brainId: string, private readonly database: DatabaseSync) {}

  static async open(brainDirectory: string, brainId: string): Promise<LearningTaskStore> {
    learningId(brainId);
    const directory = join(brainDirectory, "learning");
    for (const path of [brainDirectory, directory]) {
      const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
      if (info && (!info.isDirectory() || info.isSymbolicLink())) throw new Error("Unsafe learning workspace directory.");
      if (!info) await mkdir(path, { recursive: false, mode: 0o700 });
    }
    const path = join(directory, "tasks.sqlite3");
    const info = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (info && (!info.isFile() || info.isSymbolicLink())) throw new Error("Unsafe learning workspace database.");
    const database = new DatabaseSync(path);
    try {
      database.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA cache_size=-512;
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY,request_sha TEXT NOT NULL,body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS sources(task_id TEXT NOT NULL,id TEXT NOT NULL,ordinal INTEGER NOT NULL,body TEXT NOT NULL,PRIMARY KEY(task_id,id));
        CREATE TABLE IF NOT EXISTS claims(task_id TEXT NOT NULL,id TEXT NOT NULL,ordinal INTEGER NOT NULL,body TEXT NOT NULL,PRIMARY KEY(task_id,id));
        CREATE TABLE IF NOT EXISTS evidence(task_id TEXT NOT NULL,claim_id TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(task_id,id));
        CREATE TABLE IF NOT EXISTS findings(task_id TEXT NOT NULL,claim_id TEXT NOT NULL,id TEXT NOT NULL,request_sha TEXT NOT NULL,body TEXT NOT NULL,PRIMARY KEY(task_id,id));
        CREATE TABLE IF NOT EXISTS native_turns(turn_id TEXT PRIMARY KEY,task_id TEXT NOT NULL,claim_id TEXT);
        CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,task_id TEXT NOT NULL,kind TEXT NOT NULL,at TEXT NOT NULL,body TEXT NOT NULL);`);
      const owner = database.prepare("SELECT value FROM meta WHERE key='brainId'").get() as Row | undefined;
      if (owner && owner.value !== brainId) throw new Error("Copied learning jobs require identity-safe recovery before execution.");
      database.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('brainId',?)").run(brainId);
      database.prepare("INSERT OR IGNORE INTO meta(key,value) VALUES('format','omni-learning-workspace-1')").run();
      return new LearningTaskStore(directory, brainId, database);
    } catch (error) { database.close(); throw error; }
  }
  close(): void { this.database.close(); }
  private transaction<T>(operation: () => T): T {
    this.database.exec("BEGIN IMMEDIATE");
    try { const value = operation(); this.database.exec("COMMIT"); return value; }
    catch (error) { this.database.exec("ROLLBACK"); throw error; }
  }
  private event(taskId: string, kind: string, body: unknown): void {
    this.database.prepare("INSERT INTO events(task_id,kind,at,body) VALUES(?,?,?,?)").run(taskId, kind, new Date().toISOString(), JSON.stringify(body));
  }
  create(task: LearningTask, sources: StoredLearningSource[], claims: FactCheckClaim[] = []): LearningTask {
    return this.transaction(() => {
      const existing = this.database.prepare("SELECT request_sha FROM tasks WHERE id=?").get(task.id) as Row | undefined;
      if (existing) {
        if (existing.request_sha !== task.requestSha256) throw new Error("This learning request ID already has different contents.");
        return this.get(task.id);
      }
      const { sources: _sources, ...body } = task;
      this.database.prepare("INSERT INTO tasks VALUES(?,?,?)").run(task.id, task.requestSha256, JSON.stringify(body));
      const insert = this.database.prepare("INSERT INTO sources VALUES(?,?,?,?)");
      sources.forEach((source, index) => insert.run(task.id, source.id, index, JSON.stringify(source)));
      const insertClaim = this.database.prepare("INSERT INTO claims VALUES(?,?,?,?)");
      claims.forEach((claim, index) => insertClaim.run(task.id, claim.id, index, JSON.stringify(claim)));
      this.event(task.id, "created", { kind: task.kind, requestSha256: task.requestSha256 });
      return this.get(task.id);
    });
  }
  get(taskId: string): LearningTask {
    const row = this.database.prepare("SELECT body FROM tasks WHERE id=?").get(learningId(taskId)) as Row | undefined;
    if (!row) throw new Error("Learning task not found.");
    const task = parse<LearningTask>(row.body);
    if (task.format !== "omni-learning-task" || task.version !== 1 || task.id !== taskId || task.brainId !== this.brainId) throw new Error("Learning task ownership failed validation.");
    task.policy = validateWebLearningPolicy(task.policy);
    task.sources = this.sources(taskId).map(({ spec: _spec, ...source }) => source);
    return task;
  }
  list(): LearningTask[] {
    return (this.database.prepare("SELECT id FROM tasks ORDER BY rowid DESC").all() as Row[]).map(row => this.get(String(row.id)));
  }
  sources(taskId: string): StoredLearningSource[] {
    return (this.database.prepare("SELECT body FROM sources WHERE task_id=? ORDER BY ordinal").all(learningId(taskId)) as Row[]).map(row => parse<StoredLearningSource>(row.body));
  }
  update(taskId: string, patch: Partial<Omit<LearningTask, "id" | "brainId" | "requestId" | "requestSha256" | "sources" | "format" | "version" | "createdAt">>, eventKind?: string): LearningTask {
    return this.transaction(() => {
      const { sources: _sources, ...current } = this.get(taskId);
      const next = { ...current, ...patch, updatedAt: new Date().toISOString() };
      this.database.prepare("UPDATE tasks SET body=? WHERE id=?").run(JSON.stringify(next), taskId);
      if (eventKind) this.event(taskId, eventKind, patch);
      return this.get(taskId);
    });
  }
  steer(taskId: string, expectedRevision: number, policy: WebLearningPolicy, policyParentTaskId?: string): LearningTask {
    return this.transaction(() => {
      const { sources: _sources, ...current } = this.get(taskId);
      if (current.revision !== expectedRevision || policy.revision !== current.policy.revision + 1) throw new Error("The learning scope changed; refresh it before steering.");
      const next = { ...current, policy, policyParentTaskId, revision: current.revision + 1, updatedAt: policy.changedAt };
      this.database.prepare("UPDATE tasks SET body=? WHERE id=?").run(JSON.stringify(next), taskId);
      this.event(taskId, "user-steering", { previousPolicySha256: current.policy.sha256, policy });
      return this.get(taskId);
    });
  }
  updateSource(taskId: string, sourceId: string, patch: Partial<Omit<StoredLearningSource, "id" | "kind">>): StoredLearningSource {
    return this.transaction(() => {
      const row = this.database.prepare("SELECT body FROM sources WHERE task_id=? AND id=?").get(taskId, sourceId) as Row | undefined;
      if (!row) throw new Error("Learning source not found.");
      const source = { ...parse<StoredLearningSource>(row.body), ...patch };
      this.database.prepare("UPDATE sources SET body=? WHERE task_id=? AND id=?").run(JSON.stringify(source), taskId, sourceId);
      return source;
    });
  }
  claims(taskId: string): FactCheckClaim[] {
    return (this.database.prepare("SELECT body FROM claims WHERE task_id=? ORDER BY ordinal").all(taskId) as Row[]).map(row => {
      const claim = parse<FactCheckClaim>(row.body);
      claim.evidence = (this.database.prepare("SELECT body FROM evidence WHERE task_id=? AND claim_id=? ORDER BY rowid").all(taskId, claim.id) as Row[]).map(row => parse<FactCheckEvidence>(row.body));
      claim.findings = (this.database.prepare("SELECT body FROM findings WHERE task_id=? AND claim_id=? ORDER BY rowid").all(taskId, claim.id) as Row[]).map(row => parse<FactCheckFinding>(row.body));
      return claim;
    });
  }
  workspace(taskId: string): FactCheckWorkspace {
    const task = this.get(taskId);
    if (task.kind !== "fact-check") throw new Error("This is not a fact-check task.");
    return { task, claims: this.claims(taskId), scope: "selected-claims-only", completeKnowledgeInventory: false };
  }
  updateClaim(taskId: string, claimId: string, patch: Partial<Pick<FactCheckClaim, "state" | "questionId" | "answer" | "correctionTaskId" | "reviewError" | "assessmentReceipt">>): void {
    const current = this.claims(taskId).find(claim => claim.id === claimId);
    if (!current) throw new Error("Fact-check claim not found.");
    this.database.prepare("UPDATE claims SET body=? WHERE task_id=? AND id=?").run(JSON.stringify({ ...current, ...patch, evidence: [], findings: [] }), taskId, claimId);
  }
  bindNativeTurn(taskId: string, turnId: string, claimId?: string): void {
    this.get(learningId(taskId)); learningId(turnId, "native learning turn");
    if (claimId !== undefined && !this.claims(taskId).some(claim => claim.id === claimId)) throw new Error("Native learning turn names another claim.");
    const old = this.nativeTurn(turnId);
    if (old) {
      if (old.taskId !== taskId || old.claimId !== claimId) throw new Error("Native learning turn identity changed.");
      return;
    }
    this.database.prepare("INSERT INTO native_turns VALUES(?,?,?)").run(turnId, taskId, claimId ?? null);
  }
  nativeTurn(turnId: string): { taskId: string; claimId?: string } | undefined {
    const row = this.database.prepare("SELECT task_id,claim_id FROM native_turns WHERE turn_id=?").get(learningId(turnId, "native learning turn")) as Row | undefined;
    return row ? { taskId: String(row.task_id), ...(row.claim_id === null ? {} : { claimId: String(row.claim_id) }) } : undefined;
  }
  addEvidence(taskId: string, evidence: FactCheckEvidence): void {
    const existing = this.database.prepare("SELECT body FROM evidence WHERE task_id=? AND id=?").get(taskId, evidence.id) as Row | undefined;
    if (existing) {
      const old = parse<FactCheckEvidence>(existing.body);
      if (old.claimId !== evidence.claimId || old.url !== evidence.url || old.contentSha256 !== evidence.contentSha256 || old.sourceId !== evidence.sourceId || old.capture !== evidence.capture)
        throw new Error("Fact-check evidence identity changed.");
      if (old.toolReceiptId !== evidence.toolReceiptId) this.event(taskId, "repeated-evidence-capture", evidence);
      return; // Keep the first actual capture, not a retry's wall clock.
    }
    this.database.prepare("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?)").run(taskId, evidence.claimId, evidence.id, JSON.stringify(evidence));
  }
  addFinding(taskId: string, finding: FactCheckFinding, requestSha256: string): FactCheckFinding {
    return this.transaction(() => {
      const existing = this.database.prepare("SELECT body,request_sha FROM findings WHERE task_id=? AND id=?").get(taskId, finding.id) as Row | undefined;
      if (existing) {
        if (existing.request_sha !== requestSha256) throw new Error("Fact-check review retry has different contents.");
        return parse<FactCheckFinding>(existing.body);
      }
      this.database.prepare("INSERT INTO findings VALUES(?,?,?,?,?)").run(taskId, finding.claimId, finding.id, requestSha256, JSON.stringify(finding));
      this.event(taskId, "finding-recorded", { findingId: finding.id, claimId: finding.claimId, source: finding.source });
      return finding;
    });
  }
  recover(): LearningTask[] {
    for (const task of this.list()) if (["running", "queued"].includes(task.state)) {
      for (const source of this.sources(task.id)) if (source.state === "running") this.updateSource(task.id, source.id, { state: "paused" });
      this.update(task.id, { state: "paused", phase: "interrupted", label: "Paused after restart · committed source progress retained" }, "interrupted");
    }
    return this.list();
  }
}
