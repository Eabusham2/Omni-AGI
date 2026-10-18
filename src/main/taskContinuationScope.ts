import { createHash } from "node:crypto";
import { lstatSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ChatTaskAttachmentKind } from "../shared/taskAttachments";
import { canonicalTaskAttachmentValue, taskAttachmentManifestSha256, validatedTaskAttachmentSource,
  type PersistedTaskAttachmentRow } from "./taskAttachmentStore";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA = /^[a-f0-9]{64}$/;
const digest = (value: unknown): string => createHash("sha256").update(canonicalTaskAttachmentValue(value)).digest("hex");
const object = (value: unknown): value is Record<string, unknown> => Boolean(value && typeof value === "object" && !Array.isArray(value));

/** Main-only evidence from an actual yielded controller owner, never renderer metadata. */
export interface WarmTaskAttachmentTransfer {
  brainId: string;
  predecessorTurnId: string;
  successorTurnId: string;
  predecessorInputSha256: string;
  successorInputSha256: string;
  attentionEpoch: number;
}

export interface TaskContinuationFileSource {
  brainId: string;
  owningTurnId: string;
  attachmentId: string;
  contentSha256: string;
  manifestSha256: string;
  sourceSha256: string;
  bytes: number;
  mimeType: string;
  kind: ChatTaskAttachmentKind;
  displayName: string;
  sourceJson: string;
  createdAt: string;
  updatedAt: string;
}

/** Immutable original lineage. Copy/adoption never rewrites this record or its SHA. */
export interface TaskContinuationScopeRecord extends WarmTaskAttachmentTransfer {
  format: "omni-task-attachment-continuation";
  version: 1;
  createdAt: string;
  predecessorScopeSha256?: string;
  files: TaskContinuationFileSource[];
}

interface ScopeRow {
  scope_id: string;
  scope_sha256: string;
  owner_brain_id: string;
  successor_turn_id: string;
  record_json: string;
  state: "active" | "dormant" | "retired";
  revision: number;
  activation_sha256: string | null;
}

export interface TaskContinuationFileAlias {
  ordinal: number;
  attachmentId: string;
  owningTurnId: string | null;
  manifestSha256: string;
  sourceSha256: string;
  custody: Array<{ kind: "rekey" | "merge"; parentBrainId?: string; row: PersistedTaskAttachmentRow }>;
}

export interface TaskContinuationScopeSnapshot {
  scopeId: string;
  scopeSha256: string;
  ownerBrainId: string;
  successorTurnId: string;
  state: ScopeRow["state"];
  revision: number;
  record: TaskContinuationScopeRecord;
  aliases: TaskContinuationFileAlias[];
  activation?: TaskContinuationResumeProof;
  guardSha256: string;
}

/** Supplied only after the service authenticates the real retained execution/frontier. */
export interface TaskContinuationResumeProof {
  brainId: string;
  turnId: string;
  inputSha256: string;
  attentionEpoch: number;
  checkpointId: string;
  inherited?: { executionId: string; sourceBrainId: string; originalTurnId: string };
}

function validateTransfer(value: WarmTaskAttachmentTransfer): void {
  if (![value.brainId, value.predecessorTurnId, value.successorTurnId].every(id => typeof id === "string" && ID.test(id)) ||
      value.predecessorTurnId === value.successorTurnId || !SHA.test(value.predecessorInputSha256) ||
      !SHA.test(value.successorInputSha256) || !Number.isSafeInteger(value.attentionEpoch) || value.attentionEpoch < 0) {
    throw new Error("Task continuation lost its exact yielded turn identities.");
  }
}

export function checkedTaskContinuationRecord(value: unknown): TaskContinuationScopeRecord {
  if (!object(value) || value.format !== "omni-task-attachment-continuation" || value.version !== 1 ||
      typeof value.createdAt !== "string" || !Number.isFinite(Date.parse(value.createdAt)) ||
      value.predecessorScopeSha256 !== undefined && !SHA.test(String(value.predecessorScopeSha256)) || !Array.isArray(value.files)) {
    throw new Error("Task continuation record is invalid.");
  }
  validateTransfer(value as unknown as WarmTaskAttachmentTransfer);
  const identities = new Set<string>();
  for (const file of value.files) {
    if (!object(file) || ![file.brainId, file.owningTurnId, file.attachmentId].every(id => typeof id === "string" && ID.test(id)) ||
        ![file.contentSha256, file.manifestSha256, file.sourceSha256].every(hash => typeof hash === "string" && SHA.test(hash)) ||
        !Number.isSafeInteger(file.bytes) || Number(file.bytes) < 1 || typeof file.mimeType !== "string" ||
        !/^[a-z0-9][a-z0-9.+-]{0,80}\/[a-z0-9][a-z0-9.+-]{0,80}$/.test(file.mimeType) ||
        !["file", "image", "audio", "video"].includes(String(file.kind)) || typeof file.sourceJson !== "string" ||
        createHash("sha256").update(file.sourceJson).digest("hex") !== file.sourceSha256 || typeof file.displayName !== "string" ||
        !file.displayName || file.displayName.length > 512 || ![file.createdAt, file.updatedAt].every(time => typeof time === "string" && Number.isFinite(Date.parse(time)))) throw new Error("Task continuation file source is invalid.");
    validatedTaskAttachmentSource(JSON.parse(file.sourceJson));
    if (taskAttachmentManifestSha256({ id: String(file.attachmentId), brain_id: String(file.brainId),
      content_sha256: String(file.contentSha256), bytes: Number(file.bytes), mime_type: file.mimeType,
      kind: file.kind as ChatTaskAttachmentKind, display_name: file.displayName, state: "bound", turn_id: String(file.owningTurnId),
      created_at: String(file.createdAt), updated_at: String(file.updatedAt) }) !== file.manifestSha256) throw new Error("Task continuation original manifest failed validation.");
    const key = `${file.brainId}\0${file.owningTurnId}\0${file.attachmentId}`;
    if (identities.has(key)) throw new Error("Task continuation duplicates an original file binding.");
    identities.add(key);
  }
  return structuredClone(value) as unknown as TaskContinuationScopeRecord;
}

export function hasTaskContinuationScopes(database: DatabaseSync): boolean {
  return Boolean(database.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_continuation_scopes'").get());
}

export function initializeTaskContinuationTables(database: DatabaseSync): void {
  database.exec(`CREATE TABLE IF NOT EXISTS task_continuation_scopes(
    scope_id TEXT PRIMARY KEY, scope_sha256 TEXT NOT NULL, owner_brain_id TEXT NOT NULL, successor_turn_id TEXT NOT NULL,
    record_json TEXT NOT NULL, state TEXT NOT NULL, revision INTEGER NOT NULL, activation_sha256 TEXT,
    UNIQUE(owner_brain_id,successor_turn_id));
    CREATE TABLE IF NOT EXISTS task_continuation_files(
    scope_id TEXT NOT NULL, ordinal INTEGER NOT NULL, attachment_id TEXT NOT NULL,
    owning_turn_id TEXT, manifest_sha256 TEXT NOT NULL, source_sha256 TEXT NOT NULL, custody_json TEXT NOT NULL,
    PRIMARY KEY(scope_id,ordinal), FOREIGN KEY(scope_id) REFERENCES task_continuation_scopes(scope_id),
    FOREIGN KEY(attachment_id) REFERENCES attachments(id));
    CREATE TABLE IF NOT EXISTS task_continuation_resumes(scope_id TEXT NOT NULL,receipt_sha256 TEXT NOT NULL,proof_json TEXT NOT NULL,
    PRIMARY KEY(scope_id,receipt_sha256),FOREIGN KEY(scope_id) REFERENCES task_continuation_scopes(scope_id));`);
}

function attachmentRow(database: DatabaseSync, brainId: string, attachmentId: string): PersistedTaskAttachmentRow {
  const row = database.prepare("SELECT * FROM attachments WHERE id=?").get(attachmentId) as unknown as PersistedTaskAttachmentRow | undefined;
  if (!row || row.brain_id !== brainId || taskAttachmentManifestSha256(row) !== row.manifest_sha256 ||
      createHash("sha256").update(row.source_json).digest("hex") !== row.source_sha256 ||
      !["bound", "retired"].includes(row.state)) throw new Error("Task continuation original attachment is unavailable or changed.");
  validatedTaskAttachmentSource(JSON.parse(row.source_json));
  return row;
}

function originalSource(row: PersistedTaskAttachmentRow): TaskContinuationFileSource {
  if (row.state !== "bound" || !row.turn_id) throw new Error("An unsent attachment cannot enter a continuation scope.");
  return { brainId: row.brain_id, owningTurnId: row.turn_id, attachmentId: row.id,
    contentSha256: row.content_sha256, manifestSha256: row.manifest_sha256, sourceSha256: row.source_sha256,
    bytes: row.bytes, mimeType: row.mime_type, kind: row.kind, displayName: row.display_name,
    sourceJson: row.source_json, createdAt: row.created_at, updatedAt: row.updated_at };
}

function localAlias(row: PersistedTaskAttachmentRow, ordinal: number): TaskContinuationFileAlias {
  return { ordinal, attachmentId: row.id, owningTurnId: row.turn_id,
    manifestSha256: row.manifest_sha256, sourceSha256: row.source_sha256, custody: [] };
}

function validateCustody(file: TaskContinuationFileSource, alias: TaskContinuationFileAlias, actual: PersistedTaskAttachmentRow): void {
  let previous = { id: file.attachmentId, brain_id: file.brainId, turn_id: file.owningTurnId as string | null,
    content_sha256: file.contentSha256, manifest_sha256: file.manifestSha256, source_sha256: file.sourceSha256,
    bytes: file.bytes, mime_type: file.mimeType, kind: file.kind, display_name: file.displayName,
    source_json: file.sourceJson, created_at: file.createdAt, updated_at: file.updatedAt, state: "bound" as "bound" | "retired" };
  if (!Array.isArray(alias.custody)) throw new Error("Task continuation custody is missing.");
  for (const step of alias.custody) {
    if (!object(step) || !["rekey", "merge"].includes(step.kind) || !object(step.row)) throw new Error("Task continuation custody step is invalid.");
    const row = step.row;
    const expectedId = step.kind === "merge" ? digest({ parentBrainId: step.parentBrainId, attachmentId: previous.id }) : previous.id;
    const expectedSourceJson = step.kind === "merge" ? canonicalTaskAttachmentValue({ ...JSON.parse(previous.source_json),
      inheritedFrom: { brainId: step.parentBrainId, attachmentId: previous.id } }) : previous.source_json;
    if (!ID.test(row.brain_id) || step.kind === "merge" && (!step.parentBrainId || !ID.test(step.parentBrainId)) ||
        row.id !== expectedId || row.turn_id !== (step.kind === "merge" ? null : previous.turn_id) ||
        row.state !== (step.kind === "merge" ? "retired" : previous.state) ||
        row.source_json !== expectedSourceJson || createHash("sha256").update(row.source_json).digest("hex") !== row.source_sha256 ||
        ["content_sha256", "bytes", "mime_type", "kind", "display_name", "created_at"].some(key => row[key as keyof PersistedTaskAttachmentRow] !== previous[key as keyof typeof previous]) ||
        !Number.isFinite(Date.parse(row.updated_at)) || taskAttachmentManifestSha256(row) !== row.manifest_sha256) throw new Error("Task continuation custody does not prove this original binding.");
    previous = { ...row, state: row.state as "bound" | "retired" };
  }
  if (["id", "brain_id", "turn_id", "content_sha256", "manifest_sha256", "source_sha256", "bytes", "mime_type", "kind", "display_name", "source_json", "created_at", "updated_at", "state"]
    .some(key => actual[key as keyof PersistedTaskAttachmentRow] !== previous[key as keyof typeof previous])) throw new Error("Task continuation custody differs from its current exact attachment.");
}

export function readTaskContinuationScope(database: DatabaseSync, brainId: string, turnId: string): TaskContinuationScopeSnapshot | undefined {
  if (!hasTaskContinuationScopes(database)) return undefined;
  const row = database.prepare("SELECT * FROM task_continuation_scopes WHERE owner_brain_id=? AND successor_turn_id=?")
    .get(brainId, turnId) as unknown as ScopeRow | undefined;
  if (!row) return undefined;
  const record = checkedTaskContinuationRecord(JSON.parse(row.record_json));
  if (!SHA.test(row.scope_id) || !SHA.test(row.scope_sha256) || digest(record) !== row.scope_sha256 || row.owner_brain_id !== brainId ||
      !ID.test(row.successor_turn_id) || !["active", "dormant", "retired"].includes(row.state) ||
      !Number.isSafeInteger(row.revision) || row.revision < 0) throw new Error("Task continuation registry failed identity validation.");
  const aliases = database.prepare(`SELECT ordinal,attachment_id AS attachmentId,owning_turn_id AS owningTurnId,
    manifest_sha256 AS manifestSha256,source_sha256 AS sourceSha256,custody_json FROM task_continuation_files
    WHERE scope_id=? ORDER BY ordinal`).all(row.scope_id).map(item => {
      const { custody_json, ...alias } = item; return { ...alias, custody: JSON.parse(String(custody_json)) };
    }) as unknown as TaskContinuationFileAlias[];
  if (aliases.length !== record.files.length) throw new Error("Task continuation lost an exact original file reference.");
  const localIds = new Set<string>();
  for (const [ordinal, alias] of aliases.entries()) {
    const file = record.files[ordinal]!, source = attachmentRow(database, brainId, alias.attachmentId);
    if (alias.ordinal !== ordinal || localIds.has(alias.attachmentId) ||
        alias.owningTurnId !== source.turn_id || alias.manifestSha256 !== source.manifest_sha256 ||
        alias.sourceSha256 !== source.source_sha256 || file.contentSha256 !== source.content_sha256 || file.bytes !== source.bytes ||
        file.mimeType !== source.mime_type || file.kind !== source.kind || row.state === "active" && (source.state !== "bound" || !source.turn_id)) {
      throw new Error("Task continuation local alias no longer proves the original file binding.");
    }
    validateCustody(file, alias, source);
    localIds.add(alias.attachmentId);
  }
  let activation: TaskContinuationResumeProof | undefined;
  if (row.activation_sha256 !== null) {
    const receipt = database.prepare("SELECT proof_json FROM task_continuation_resumes WHERE scope_id=? AND receipt_sha256=?").get(row.scope_id, row.activation_sha256);
    if (!receipt || !SHA.test(row.activation_sha256)) throw new Error("Task continuation lost its actual retained execution receipt.");
    activation = JSON.parse(String(receipt.proof_json)) as TaskContinuationResumeProof;
    if (digest(activation) !== row.activation_sha256 || activation.brainId !== brainId || activation.turnId !== turnId ||
        activation.inputSha256 !== record.successorInputSha256 || activation.attentionEpoch !== record.attentionEpoch || !ID.test(activation.checkpointId)) throw new Error("Task continuation activation differs from the real resumed frontier.");
  }
  const snapshot = { scopeId: row.scope_id, scopeSha256: row.scope_sha256, ownerBrainId: brainId, successorTurnId: turnId,
    state: row.state, revision: row.revision, record, aliases, ...(activation ? { activation } : {}) };
  return { ...snapshot, guardSha256: digest(snapshot) };
}

/** Validate optional prefixed tables inside the already-validated portable attachment registry. */
export function validateTaskContinuationScopes(database: DatabaseSync, brainId: string): void {
  if (!hasTaskContinuationScopes(database)) {
    if (database.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='task_continuation_files'").get()) throw new Error("Task continuation registry is incomplete.");
    return;
  }
  for (const row of database.prepare("SELECT owner_brain_id,successor_turn_id FROM task_continuation_scopes").iterate()) {
    if (row.owner_brain_id !== brainId) throw new Error("Task continuation has another current owner.");
    readTaskContinuationScope(database, brainId, String(row.successor_turn_id));
  }
  if (database.prepare(`SELECT 1 FROM task_continuation_files f LEFT JOIN task_continuation_scopes s
    ON f.scope_id=s.scope_id WHERE s.scope_id IS NULL LIMIT 1`).get()) throw new Error("Task continuation has an orphaned source reference.");
  for (const receipt of database.prepare("SELECT scope_id,receipt_sha256,proof_json FROM task_continuation_resumes").iterate()) {
    if (!database.prepare("SELECT 1 FROM task_continuation_scopes WHERE scope_id=?").get(String(receipt.scope_id)) ||
        digest(JSON.parse(String(receipt.proof_json))) !== receipt.receipt_sha256) throw new Error("Task continuation resume provenance is corrupt or orphaned.");
  }
}

/** The registry has no public activation API; only trusted controller/resume composition calls this store. */
export class TaskContinuationScopeStore {
  constructor(private readonly path: string, private readonly brainId: string) {
    if (!ID.test(brainId)) throw new Error("Invalid task continuation current owner.");
  }

  private database<T>(operation: (database: DatabaseSync) => T): T {
    for (const path of [this.path, dirname(this.path)]) {
      const info = lstatSync(path);
      if (info.isSymbolicLink() || path === this.path && !info.isFile() || path !== this.path && !info.isDirectory()) throw new Error("Task continuation registry path is unsafe.");
    }
    const database = new DatabaseSync(this.path);
    try {
      database.exec("PRAGMA foreign_keys=ON; PRAGMA busy_timeout=30000; PRAGMA synchronous=FULL");
      if (database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== this.brainId ||
          database.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== "omni-chat-task-attachments-1") throw new Error("Task continuation lost its current attachment owner.");
      initializeTaskContinuationTables(database);
      return operation(database);
    } finally { database.close(); }
  }

  snapshot(turnId: string): TaskContinuationScopeSnapshot | undefined {
    if (!ID.test(turnId)) throw new Error("Invalid task continuation turn.");
    return this.database(database => readTaskContinuationScope(database, this.brainId, turnId));
  }

  assertCurrent(snapshot: TaskContinuationScopeSnapshot): void {
    const current = this.snapshot(snapshot.successorTurnId);
    if (!current || current.state !== "active" || current.guardSha256 !== snapshot.guardSha256) throw new Error("Task continuation scope changed or was revoked during this read.");
  }

  transfer(request: WarmTaskAttachmentTransfer): TaskContinuationScopeSnapshot | undefined {
    validateTransfer(request);
    if (request.brainId !== this.brainId) throw new Error("Task continuation cannot cross current owners.");
    return this.database(database => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const existing = readTaskContinuationScope(database, this.brainId, request.successorTurnId);
        if (existing) {
          const original = existing.record;
          if (existing.state !== "active" || ["brainId", "predecessorTurnId", "successorTurnId", "predecessorInputSha256", "successorInputSha256", "attentionEpoch"]
            .some(key => original[key as keyof WarmTaskAttachmentTransfer] !== request[key as keyof WarmTaskAttachmentTransfer])) throw new Error("Task continuation successor was reused or needs a genuine resume.");
          database.exec("COMMIT"); return existing;
        }
        const predecessor = readTaskContinuationScope(database, this.brainId, request.predecessorTurnId);
        if (predecessor && (predecessor.state !== "active" || predecessor.record.attentionEpoch !== request.attentionEpoch ||
            predecessor.record.successorInputSha256 !== request.predecessorInputSha256)) throw new Error("The predecessor's task scope is not active at this actual boundary.");
        const files = [...(predecessor?.record.files ?? [])], aliases = [...(predecessor?.aliases ?? [])];
        const selected = database.prepare("SELECT id FROM attachments WHERE state='bound' AND turn_id=? ORDER BY request_id,request_index").all(request.predecessorTurnId);
        for (const item of selected) {
          const source = attachmentRow(database, this.brainId, String(item.id));
          if (aliases.some(alias => alias.attachmentId === source.id)) continue;
          files.push(originalSource(source)); aliases.push(localAlias(source, aliases.length));
        }
        if (!files.length) { database.exec("COMMIT"); return undefined; }
        const record: TaskContinuationScopeRecord = { format: "omni-task-attachment-continuation", version: 1,
          ...request, createdAt: new Date().toISOString(), ...(predecessor ? { predecessorScopeSha256: predecessor.scopeSha256 } : {}), files };
        const scopeSha256 = digest(checkedTaskContinuationRecord(record));
        const scopeId = digest({ scopeSha256, brainId: this.brainId, turnId: request.successorTurnId });
        database.prepare("INSERT INTO task_continuation_scopes VALUES(?,?,?,?,?,?,0,NULL)")
          .run(scopeId, scopeSha256, this.brainId, request.successorTurnId, canonicalTaskAttachmentValue(record), "active");
        const insert = database.prepare("INSERT INTO task_continuation_files VALUES(?,?,?,?,?,?,?)");
        for (const alias of aliases) insert.run(scopeId, alias.ordinal, alias.attachmentId, alias.owningTurnId, alias.manifestSha256, alias.sourceSha256, canonicalTaskAttachmentValue(alias.custody));
        const result = readTaskContinuationScope(database, this.brainId, request.successorTurnId);
        database.exec("COMMIT"); return result;
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  retireExceptEpoch(epoch: number): void {
    if (!Number.isSafeInteger(epoch) || epoch < 0) throw new Error("Invalid actual attention epoch.");
    this.database(database => {
      for (const row of database.prepare("SELECT successor_turn_id FROM task_continuation_scopes WHERE state<>'retired'").all()) {
        const scope = readTaskContinuationScope(database, this.brainId, String(row.successor_turn_id))!;
        if (scope.record.attentionEpoch !== epoch) database.prepare("UPDATE task_continuation_scopes SET state='retired',revision=revision+1 WHERE scope_id=?").run(scope.scopeId);
      }
    });
  }

  activateRetained(proof: TaskContinuationResumeProof): void {
    if (proof.brainId !== this.brainId || !ID.test(proof.turnId) || !SHA.test(proof.inputSha256) || !ID.test(proof.checkpointId) ||
        !Number.isSafeInteger(proof.attentionEpoch) || proof.attentionEpoch < 0) throw new Error("Task scope resume needs the real authenticated native frontier.");
    this.database(database => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const scope = readTaskContinuationScope(database, this.brainId, proof.turnId);
        if (!scope) { database.exec("COMMIT"); return; }
        if (scope.state === "retired" || scope.record.successorInputSha256 !== proof.inputSha256 || scope.record.attentionEpoch !== proof.attentionEpoch ||
            proof.inherited && (!SHA.test(proof.inherited.executionId) || scope.record.brainId !== proof.inherited.sourceBrainId || scope.record.successorTurnId !== proof.inherited.originalTurnId) ||
            scope.aliases.some(alias => !alias.owningTurnId)) throw new Error("The retained execution does not prove this dormant task scope.");
        const receiptSha256 = digest(proof);
        database.prepare("INSERT OR IGNORE INTO task_continuation_resumes VALUES(?,?,?)").run(scope.scopeId, receiptSha256, canonicalTaskAttachmentValue(proof));
        if (scope.state !== "active" || digest(scope.activation ?? null) !== receiptSha256) database.prepare("UPDATE task_continuation_scopes SET state='active',revision=revision+1,activation_sha256=? WHERE scope_id=?").run(receiptSha256, scope.scopeId);
        database.exec("COMMIT");
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }
}

/** Within the caller's atomic registry rekey. Original records remain byte-for-byte unchanged. */
export function rekeyTaskContinuationScopes(database: DatabaseSync, targetBrainId: string): void {
  if (!hasTaskContinuationScopes(database)) return;
  database.prepare("UPDATE task_continuation_scopes SET owner_brain_id=?,state=CASE WHEN state='retired' THEN state ELSE 'dormant' END,revision=revision+1,activation_sha256=NULL").run(targetBrainId);
  for (const alias of database.prepare("SELECT scope_id,ordinal,attachment_id,custody_json FROM task_continuation_files").all()) {
    const source = attachmentRow(database, targetBrainId, String(alias.attachment_id));
    const custody = JSON.parse(String(alias.custody_json)) as TaskContinuationFileAlias["custody"];
    // Same-owner recovery does not rewrite even one original manifest or custody step.
    const existing = database.prepare("SELECT manifest_sha256,source_sha256 FROM task_continuation_files WHERE scope_id=? AND ordinal=?")
      .get(String(alias.scope_id), Number(alias.ordinal));
    if (existing?.manifest_sha256 !== source.manifest_sha256 || existing?.source_sha256 !== source.source_sha256) custody.push({ kind: "rekey", row: source });
    database.prepare("UPDATE task_continuation_files SET owning_turn_id=?,manifest_sha256=?,source_sha256=?,custody_json=? WHERE scope_id=? AND ordinal=?")
      .run(source.turn_id, source.manifest_sha256, source.source_sha256, canonicalTaskAttachmentValue(custody), String(alias.scope_id), Number(alias.ordinal));
  }
}

/** Merge preserves every distinct immutable source scope, with dormant destination aliases only. */
export function mergeTaskContinuationScopes(source: DatabaseSync, target: DatabaseSync, targetBrainId: string,
  parentBrainId: string, turnAliases: ReadonlyMap<string, string> = new Map()): void {
  if (!hasTaskContinuationScopes(source)) return;
  initializeTaskContinuationTables(target);
  for (const old of source.prepare("SELECT * FROM task_continuation_scopes ORDER BY scope_sha256").iterate() as Iterable<unknown>) {
    const row = old as ScopeRow, original = readTaskContinuationScope(source, targetBrainId, row.successor_turn_id)!;
    const turnId = turnAliases.get(original.record.successorTurnId) ?? digest({ parentBrainId, turnId: row.successor_turn_id, scopeSha256: row.scope_sha256 });
    const scopeId = digest({ parentBrainId, sourceScopeId: row.scope_id, scopeSha256: row.scope_sha256, turnId });
    const prior = target.prepare("SELECT * FROM task_continuation_scopes WHERE scope_id=?").get(scopeId) as unknown as ScopeRow | undefined;
    if (prior) {
      if (prior.record_json !== row.record_json) throw new Error("Merged task scope conflicts with its immutable original.");
      if (prior.owner_brain_id !== targetBrainId || prior.successor_turn_id !== turnId) throw new Error("Merged task scope alias conflicts with its source binding.");
      continue; // Only the same immutable original AND the same destination binding are deduplicated.
    }
    target.prepare("INSERT INTO task_continuation_scopes VALUES(?,?,?,?,?,?,?,NULL)")
      .run(scopeId, row.scope_sha256, targetBrainId, turnId, row.record_json, "dormant", row.revision + 1);
    for (const alias of original.aliases) {
      const attachmentId = digest({ parentBrainId, attachmentId: alias.attachmentId });
      const file = attachmentRow(target, targetBrainId, attachmentId);
      target.prepare("INSERT INTO task_continuation_files VALUES(?,?,?,?,?,?,?)")
        .run(scopeId, alias.ordinal, file.id, file.turn_id, file.manifest_sha256, file.source_sha256,
          canonicalTaskAttachmentValue([...alias.custody, { kind: "merge", parentBrainId, row: file }]));
    }
    for (const receipt of source.prepare("SELECT receipt_sha256,proof_json FROM task_continuation_resumes WHERE scope_id=?").all(row.scope_id)) {
      target.prepare("INSERT INTO task_continuation_resumes VALUES(?,?,?)").run(scopeId, String(receipt.receipt_sha256), String(receipt.proof_json));
    }
  }
}
