import { lstatSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { InheritedTaskFileReviewBody, InheritedTaskFileAdoptionReceipt } from "../shared/inheritedTaskFiles";
import { canonicalTaskAttachmentValue } from "./taskAttachmentStore";
import { inheritedTaskFileHash } from "./inheritedTaskFileClosure";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/, SHA = /^[a-f0-9]{64}$/;
interface ReviewRow { id: string; owner_brain_id: string; review_sha256: string; body_json: string;
  fingerprint: string; state: "approved" | "revoked" | "dormant"; receipt_json: string; revision: number; }

export function inheritedTaskReviewFingerprint(body: InheritedTaskFileReviewBody): string {
  const { reviewId: _id, createdAt: _at, ...exact } = body;
  return inheritedTaskFileHash(exact);
}

export function checkedInheritedTaskReview(value: unknown): InheritedTaskFileReviewBody {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Inherited task review is not a typed record.");
  const body = value as InheritedTaskFileReviewBody;
  if (body.format !== "omni-inherited-task-file-review" || body.version !== 1 ||
      ![body.reviewId, body.brainId, body.rootTurnId, body.sourceBrainId, body.sourceTurnId].every(id => typeof id === "string" && ID.test(id)) ||
      ![body.executionId, body.inputSha256, body.ownerBindingSha256, body.sourceClosureSha256].every(hash => typeof hash === "string" && SHA.test(hash)) ||
      !Number.isFinite(Date.parse(body.createdAt)) || ![body.rootAttentionEpoch, body.sourceAttentionEpoch].every(epoch => Number.isSafeInteger(epoch) && epoch >= 0) ||
      !body.frontier || body.frontier.turnId !== body.rootTurnId || body.frontier.inputSha256 !== body.inputSha256 || !ID.test(body.frontier.checkpointId) ||
      body.frontier.exactSupportedResume !== true || body.frontier.phase !== "native-text-token-boundary" ||
      ![body.frontier.lastStreamSequence, body.frontier.sampledTokens, body.frontier.publishedTokens].every(count => Number.isSafeInteger(count) && count >= 0) ||
      body.permission?.toolId !== "task.attachments" || !["off", "ask", "auto", "full"].includes(body.permission.level) || !SHA.test(body.permission.revisionSha256) ||
      !Array.isArray(body.files) || !body.files.length) throw new Error("Inherited task review lost its exact owner/frontier/access binding.");
  const ids = new Set<string>();
  for (const file of body.files) {
    if (!file || !ID.test(file.attachmentId) || ids.has(file.attachmentId) || !file.displayName || file.displayName.length > 512 ||
        !["file", "image", "audio", "video"].includes(file.kind) || !/^[a-z0-9][a-z0-9.+-]{0,80}\/[a-z0-9][a-z0-9.+-]{0,80}$/.test(file.mimeType) ||
        !Number.isSafeInteger(file.bytes) || file.bytes < 1 || ![file.contentSha256, file.manifestSha256, file.sourceSha256].every(hash => SHA.test(hash)) ||
        !file.originalBinding || ![file.originalBinding.brainId, file.originalBinding.turnId, file.originalBinding.attachmentId].every(id => ID.test(id)) ||
        ![file.originalBinding.manifestSha256, file.originalBinding.sourceSha256].every(hash => SHA.test(hash))) throw new Error("Inherited task review file disclosure is invalid.");
    ids.add(file.attachmentId);
  }
  return structuredClone(body);
}

export function hasInheritedTaskFileReviews(database: DatabaseSync): boolean {
  return Boolean(database.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='inherited_task_file_reviews'").get());
}
function initialize(database: DatabaseSync): void {
  database.exec(`CREATE TABLE IF NOT EXISTS inherited_task_file_reviews(id TEXT PRIMARY KEY,owner_brain_id TEXT NOT NULL,
    review_sha256 TEXT NOT NULL,body_json TEXT NOT NULL,fingerprint TEXT NOT NULL,state TEXT NOT NULL,receipt_json TEXT NOT NULL,revision INTEGER NOT NULL);
    CREATE INDEX IF NOT EXISTS inherited_task_file_review_lookup ON inherited_task_file_reviews(owner_brain_id,fingerprint,state);`);
}

function validate(row: ReviewRow, brainId: string): { body: InheritedTaskFileReviewBody; receipt: InheritedTaskFileAdoptionReceipt } {
  const body = checkedInheritedTaskReview(JSON.parse(row.body_json)), receipt = JSON.parse(row.receipt_json) as InheritedTaskFileAdoptionReceipt;
  const { receiptSha256, ...receiptBody } = receipt;
  if (!ID.test(row.id) || row.owner_brain_id !== brainId || !["approved", "revoked", "dormant"].includes(row.state) ||
      !Number.isSafeInteger(row.revision) || row.revision < 0 || inheritedTaskFileHash(body) !== row.review_sha256 ||
      inheritedTaskReviewFingerprint(body) !== row.fingerprint || receipt.format !== "omni-inherited-task-file-adoption" || receipt.version !== 1 ||
      inheritedTaskFileHash(receiptBody) !== receiptSha256 || receipt.reviewId !== body.reviewId || receipt.reviewSha256 !== row.review_sha256 ||
      receipt.brainId !== body.brainId || receipt.executionId !== body.executionId || receipt.rootTurnId !== body.rootTurnId ||
      !Number.isFinite(Date.parse(receipt.approvedAt)) || receipt.permissionsChanged !== false || receipt.sourceBindingsChanged !== false ||
      row.state === "approved" && (body.brainId !== brainId || row.id !== body.reviewId)) throw new Error("Inherited task file adoption receipt is corrupt or has another owner.");
  return { body, receipt };
}

export class InheritedTaskFileReviewStore {
  constructor(private readonly path: string, private readonly brainId: string) {}
  private database<T>(write: boolean, operation: (database: DatabaseSync) => T): T {
    if (!ID.test(this.brainId)) throw new Error("Invalid inherited review owner.");
    for (const path of [dirname(this.path), this.path]) {
      const info = lstatSync(path); if (info.isSymbolicLink() || path === this.path && !info.isFile()) throw new Error("Inherited review registry path is unsafe.");
    }
    const database = new DatabaseSync(this.path, { readOnly: !write });
    try {
      database.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-512; PRAGMA busy_timeout=30000");
      if (database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== this.brainId) throw new Error("Inherited review lost its current registry owner.");
      if (write) { database.exec("PRAGMA synchronous=FULL"); initialize(database); }
      return operation(database);
    } finally { database.close(); }
  }

  approved(body: InheritedTaskFileReviewBody): { body: InheritedTaskFileReviewBody; receipt: InheritedTaskFileAdoptionReceipt } | undefined {
    return this.database(false, database => {
      if (!hasInheritedTaskFileReviews(database)) return undefined;
      const row = database.prepare("SELECT * FROM inherited_task_file_reviews WHERE owner_brain_id=? AND fingerprint=? AND state='approved' ORDER BY id LIMIT 1")
        .get(this.brainId, inheritedTaskReviewFingerprint(body)) as unknown as ReviewRow | undefined;
      return row ? validate(row, this.brainId) : undefined;
    });
  }

  approvedById(reviewId: string, reviewSha256: string): { body: InheritedTaskFileReviewBody; receipt: InheritedTaskFileAdoptionReceipt } | undefined {
    return this.database(false, database => {
      if (!hasInheritedTaskFileReviews(database)) return undefined;
      const row = database.prepare("SELECT * FROM inherited_task_file_reviews WHERE id=? AND owner_brain_id=? AND state='approved' AND review_sha256=?")
        .get(reviewId, this.brainId, reviewSha256) as unknown as ReviewRow | undefined;
      return row ? validate(row, this.brainId) : undefined;
    });
  }

  approve(bodyValue: InheritedTaskFileReviewBody): InheritedTaskFileAdoptionReceipt {
    const body = checkedInheritedTaskReview(bodyValue);
    if (body.brainId !== this.brainId) throw new Error("Inherited review cannot grant another destination access.");
    const approvedAt = new Date().toISOString(), reviewSha256 = inheritedTaskFileHash(body);
    const receiptBody = { format: "omni-inherited-task-file-adoption" as const, version: 1 as const, brainId: this.brainId,
      reviewId: body.reviewId, reviewSha256, executionId: body.executionId, rootTurnId: body.rootTurnId, approvedAt,
      permissionsChanged: false as const, sourceBindingsChanged: false as const };
    const receipt = { ...receiptBody, receiptSha256: inheritedTaskFileHash(receiptBody) };
    return this.database(true, database => {
      database.exec("BEGIN IMMEDIATE");
      try {
        const prior = database.prepare("SELECT * FROM inherited_task_file_reviews WHERE id=?").get(body.reviewId) as unknown as ReviewRow | undefined;
        if (prior) {
          const saved = validate(prior, this.brainId);
          if (prior.state !== "approved" || prior.review_sha256 !== reviewSha256) throw new Error("This exact review is revoked, dormant, or changed; prepare a new review.");
          database.exec("COMMIT"); return saved.receipt;
        }
        database.prepare("INSERT INTO inherited_task_file_reviews VALUES(?,?,?,?,?,'approved',?,0)")
          .run(body.reviewId, this.brainId, reviewSha256, canonicalTaskAttachmentValue(body), inheritedTaskReviewFingerprint(body), canonicalTaskAttachmentValue(receipt));
        database.exec("COMMIT"); return receipt;
      } catch (error) { database.exec("ROLLBACK"); throw error; }
    });
  }

  assertApproved(receipt: InheritedTaskFileAdoptionReceipt): void {
    this.database(false, database => {
      const row = hasInheritedTaskFileReviews(database) ? database.prepare("SELECT * FROM inherited_task_file_reviews WHERE id=? AND owner_brain_id=?").get(receipt.reviewId, this.brainId) as unknown as ReviewRow | undefined : undefined;
      if (!row || row.state !== "approved" || validate(row, this.brainId).receipt.receiptSha256 !== receipt.receiptSha256) throw new Error("Inherited task file adoption was revoked or replaced during this read.");
    });
  }

  revoke(reviewId: string, reviewSha256: string): void {
    this.database(true, database => {
      const row = database.prepare("SELECT * FROM inherited_task_file_reviews WHERE id=?").get(reviewId) as unknown as ReviewRow | undefined;
      if (!row) throw new Error("Unknown inherited task file approval.");
      validate(row, this.brainId);
      if (row.review_sha256 !== reviewSha256) throw new Error("Inherited review hash changed before revocation.");
      database.prepare("UPDATE inherited_task_file_reviews SET state='revoked',revision=revision+1 WHERE id=?").run(reviewId);
    });
  }
}

export function validateInheritedTaskFileReviews(database: DatabaseSync, brainId: string): void {
  if (!hasInheritedTaskFileReviews(database)) return;
  for (const row of database.prepare("SELECT * FROM inherited_task_file_reviews").iterate()) validate(row as unknown as ReviewRow, brainId);
}
export function rekeyInheritedTaskFileReviews(database: DatabaseSync, brainId: string): void {
  if (hasInheritedTaskFileReviews(database)) database.prepare("UPDATE inherited_task_file_reviews SET owner_brain_id=?,state='dormant',revision=revision+1").run(brainId);
}
export function mergeInheritedTaskFileReviews(source: DatabaseSync, target: DatabaseSync, brainId: string, parentId: string): void {
  if (!hasInheritedTaskFileReviews(source)) return; initialize(target);
  for (const row of source.prepare("SELECT * FROM inherited_task_file_reviews").iterate()) {
    const item = row as unknown as ReviewRow; validate(item, brainId);
    const id = inheritedTaskFileHash({ parentId, reviewId: item.id });
    target.prepare("INSERT OR IGNORE INTO inherited_task_file_reviews VALUES(?,?,?,?,?,'dormant',?,?)")
      .run(id, brainId, item.review_sha256, item.body_json, item.fingerprint, item.receipt_json, item.revision + 1);
  }
}
