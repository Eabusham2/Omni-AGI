import { createHash } from "node:crypto";
import { lstat } from "node:fs/promises";
import { freemem } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { getHeapStatistics } from "node:v8";
import type { WebImageDownload, WebImageReference, WebImageSelection } from "../shared/webImages";

const SHA = /^[a-f0-9]{64}$/, ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const TABLE = "CREATE TABLE web_image_references(id TEXT PRIMARY KEY,brain_id TEXT NOT NULL,turn_id TEXT NOT NULL,body TEXT NOT NULL,sha256 TEXT NOT NULL,active INTEGER NOT NULL CHECK(active IN (0,1)))";
function schemaChecked(db: DatabaseSync, create = false): boolean {
  const declaration = db.prepare("SELECT type,length(CAST(sql AS BLOB)) AS bytes FROM sqlite_master WHERE name='web_image_references'").get();
  if (!declaration) { if (create) { db.exec(TABLE); return true; } return false; }
  webImageReadBudget(Number(declaration.bytes));
  if (declaration.type !== "table" || String(db.prepare("SELECT sql FROM sqlite_master WHERE name='web_image_references'").get()?.sql).replace(/\s/g, "") !== TABLE.replace(/\s/g, "")) throw new Error("Web image reference table schema changed.");
  if (db.prepare("SELECT 1 FROM sqlite_master WHERE tbl_name='web_image_references' AND type!='table' AND NOT(type='index' AND name='sqlite_autoindex_web_image_references_1' AND sql IS NULL) LIMIT 1").get()) throw new Error("Web image references refuse executable/unknown schema objects.");
  return true;
}
export function webImageCanonical(value: unknown): string {
  return Array.isArray(value) ? `[${value.map(webImageCanonical).join(",")}]` : value && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => `${JSON.stringify(key)}:${webImageCanonical(child)}`).join(",")}}` : JSON.stringify(value);
}
export const webImageHash = (value: unknown): string => createHash("sha256").update(webImageCanonical(value)).digest("hex");
export function webImageReadBudget(bytes: number): void {
  if (!Number.isSafeInteger(bytes) || bytes < 0 || !Number.isSafeInteger(bytes * 12) || bytes * 12 > Math.min(freemem(), getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) throw new Error("Web image metadata awaits physical read headroom; no references were truncated.");
}
export function webImageUrl(value: unknown, base?: string): string {
  if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error("Web image reference requires an actual HTTPS URL.");
  const url = new URL(value, base);
  if (url.protocol !== "https:" || url.username || url.password) throw new Error("Web image references cannot grant local, credential-bearing or executable URLs.");
  url.hash = ""; return url.toString();
}
export function checkedWebImageReference(value: unknown): WebImageReference {
  const row = value as WebImageReference;
  if (!row || row.format !== "omni-web-image-reference" || row.version !== 1 || !SHA.test(row.id) || !ID.test(row.brainId) || !ID.test(row.turnId) || row.fetched !== false ||
    !row.source || !["search-response", "html-response", "image-response"].includes(row.source.kind) || !SHA.test(row.source.sha256) || !Number.isFinite(Date.parse(row.source.fetchedAt)) ||
    row.label !== undefined && (typeof row.label !== "string" || row.label.includes("\0"))) throw new Error("Web image reference has invalid source attribution.");
  for (const url of [row.url, row.pageUrl, row.source.url]) if (webImageUrl(url) !== url) throw new Error("Web image reference URL changed after observation.");
  const { id: _id, ...body } = row; if (webImageHash(body) !== row.id) throw new Error("Web image reference source checksum changed.");
  return structuredClone(row);
}
export function checkedWebImageDownload(value: unknown): WebImageDownload {
  const row = value as WebImageDownload, reference = checkedWebImageReference(row?.reference);
  if (!ID.test(row.requestId) || row.ownerTurnId !== reference.turnId || !ID.test(row.scopeTurnId) || row.requestedUrl !== reference.url ||
    !Array.isArray(row.redirects) || !row.redirects.length || row.redirects[0] !== row.requestedUrl || row.redirects.at(-1) !== row.finalUrl ||
    !Number.isFinite(Date.parse(row.fetchedAt)) || !SHA.test(row.permissionRevision) || row.learningPolicySha256 !== undefined && !SHA.test(row.learningPolicySha256) ||
    !["image/png", "image/jpeg", "image/webp", "image/gif"].includes(row.mimeType) || !SHA.test(row.sha256) || !Number.isSafeInteger(row.bytes) || row.bytes < 1) throw new Error("Selected web image lost its exact download provenance.");
  for (const url of row.redirects) if (webImageUrl(url) !== url) throw new Error("Selected image redirect attribution changed.");
  if (row.scopeTurnId !== "webimg-" + webImageHash([reference.brainId, row.ownerTurnId, row.requestId, reference.id])) throw new Error("Selected image has another action-owned task scope.");
  return structuredClone(row);
}
export function webImageSelectionIdentity(value: Omit<WebImageSelection, "selectionSha256" | "observation">): string { return webImageHash(value); }
export function checkedWebImageSelection(value: unknown): WebImageSelection {
  const row = value as WebImageSelection, download = checkedWebImageDownload(row?.download), reference = checkedWebImageReference(row?.reference), attachment = row.attachment;
  if (row.format !== "omni-web-image-selection" || row.version !== 1 || row.brainId !== reference.brainId || row.turnId !== reference.turnId || webImageHash(download.reference) !== webImageHash(reference) ||
    !attachment || attachment.brainId !== row.brainId || attachment.turnId !== download.scopeTurnId || !ID.test(attachment.attachmentId) || attachment.kind !== "image" ||
    attachment.contentSha256 !== download.sha256 || attachment.bytes !== download.bytes || attachment.mimeType !== download.mimeType || !SHA.test(attachment.manifestSha256) ||
    attachment.relativePath !== `attachments/blobs/${download.sha256}.data` || typeof attachment.displayName !== "string" ||
    row.observation?.state !== "pending-native-perception" || row.observation.pixelsRead !== false || row.observation.parametersChanged !== false || row.observation.rawSourceTextInjected !== false) throw new Error("Selected web image lost its exact current action/attachment binding.");
  const { selectionSha256, observation: _observation, ...identity } = row;
  if (webImageSelectionIdentity(identity) !== selectionSha256) throw new Error("Selected web image input checksum changed."); return structuredClone(row);
}
export function webImagePermissionRevision(row: { toolId: string; level: string; updatedAt: string; revision?: number }): string {
  return createHash("sha256").update(JSON.stringify({ toolId: "web.fetch", level: row.level, updatedAt: row.updatedAt, revision: row.revision ?? 0 })).digest("hex");
}

function entities(value: string): string {
  return value.replace(/&(?:quot|apos|amp|lt|gt|#\d+|#x[\da-f]+);/gi, entity => {
    const plain: Record<string, string> = { "&quot;": '"', "&apos;": "'", "&amp;": "&", "&lt;": "<", "&gt;": ">" };
    if (plain[entity.toLowerCase()]) return plain[entity.toLowerCase()]!;
    const point = parseInt(entity.slice(entity[2]?.toLowerCase() === "x" ? 3 : 2, -1), entity[2]?.toLowerCase() === "x" ? 16 : 10);
    return point > 0 && point <= 0x10ffff && !(point >= 0xd800 && point <= 0xdfff) ? String.fromCodePoint(point) : entity;
  });
}
export function* htmlImageCandidates(html: string, pageUrl: string): Generator<{ url: string; pageUrl: string; label?: string }> {
  // Non-executing literal metadata extraction, not a browser/DOM or a captioner.
  for (const match of html.matchAll(/<(img|meta|link)\b[^>]*>/gi)) {
    const fields: Record<string, string> = {};
    for (const attr of match[0].matchAll(/([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/g)) fields[attr[1]!.toLowerCase()] = entities(attr[2] ?? attr[3] ?? attr[4] ?? "");
    const kind = match[1]!.toLowerCase();
    const raw = kind === "img" ? fields.src : kind === "meta" && ["og:image", "og:image:url", "twitter:image"].includes((fields.property ?? fields.name ?? "").toLowerCase()) ? fields.content : kind === "link" && fields.rel?.toLowerCase() === "image_src" ? fields.href : undefined;
    try { if (raw) yield { url: webImageUrl(raw, pageUrl), pageUrl, ...(fields.alt ? { label: fields.alt } : {}) }; } catch { /* Unsupported references remain unfetched. */ }
  }
}
export function* searchImageCandidates(results: readonly unknown[]): Generator<{ url: string; pageUrl: string; label?: string }> {
  for (const value of results) {
    if (!value || typeof value !== "object" || Array.isArray(value)) continue;
    const row = value as Record<string, unknown>;
    try {
      const pageUrl = webImageUrl(row.url ?? row.link);
      for (const field of ["img_src", "thumbnail", "thumbnail_src", "image"] as const) {
        if (typeof row[field] !== "string") continue;
        try { yield { url: webImageUrl(row[field], pageUrl), pageUrl, ...(typeof row.title === "string" ? { label: row.title } : {}) }; } catch { /* No data: or provider-supplied local paths. */ }
      }
    } catch { /* A malformed search hit cannot create image authority. */ }
  }
}

/** Small attributable reference rows live in the existing task-file registry. */
export class WebImageReferenceStore {
  private constructor(private readonly db: DatabaseSync, readonly brainId: string, private readonly readOnly: boolean) {}
  static async open(root: string, brainId: string, readOnly = false): Promise<WebImageReferenceStore> {
    if (!ID.test(brainId)) throw new Error("Invalid web image reference owner.");
    for (const path of [root, join(root, "attachments"), join(root, "attachments", "registry.sqlite3")]) {
      const info = await lstat(path); if (info.isSymbolicLink() || !(path.endsWith(".sqlite3") ? info.isFile() : info.isDirectory())) throw new Error("Web image reference custody is unsafe.");
    }
    const db = new DatabaseSync(join(root, "attachments", "registry.sqlite3"), { readOnly, allowExtension: false });
    try {
      db.exec("PRAGMA trusted_schema=OFF; PRAGMA cache_size=-128; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000");
      if (db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Web image registry has another current owner.");
      schemaChecked(db, !readOnly);
      return new WebImageReferenceStore(db, brainId, readOnly);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  private present(): boolean { return !!this.db.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name='web_image_references'").get(); }
  record(turnId: string, source: WebImageReference["source"], candidate: { url: string; pageUrl: string; label?: string }): WebImageReference {
    if (this.readOnly || !ID.test(turnId)) throw new Error("Web image references require their current writable task owner.");
    const body = { format: "omni-web-image-reference" as const, version: 1 as const, brainId: this.brainId, turnId, url: webImageUrl(candidate.url), pageUrl: webImageUrl(candidate.pageUrl),
      ...(candidate.label ? { label: candidate.label } : {}), fetched: false as const, source: { ...source, url: webImageUrl(source.url) } };
    const reference = checkedWebImageReference({ ...body, id: webImageHash(body) }), text = webImageCanonical(reference); webImageReadBudget(Buffer.byteLength(text));
    this.db.prepare("INSERT OR IGNORE INTO web_image_references VALUES(?,?,?,?,?,1)").run(reference.id, this.brainId, turnId, text, webImageHash(reference));
    const saved = this.resolve(turnId, reference.id); if (webImageHash(saved) !== webImageHash(reference)) throw new Error("Web image reference ID is already bound to different exact metadata."); return saved;
  }
  private row(referenceId: string): { reference: WebImageReference; active: boolean } {
    const size = this.db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM web_image_references WHERE id=?").get(referenceId);
    if (!size) throw new Error("Select an image reference actually returned to this task."); webImageReadBudget(Number(size.bytes));
    const row = this.db.prepare("SELECT * FROM web_image_references WHERE id=?").get(referenceId)!;
    if (typeof row.body !== "string" || Buffer.byteLength(row.body) !== size.bytes || createHash("sha256").update(row.body).digest("hex") !== row.sha256) throw new Error("Web image reference bytes changed.");
    const reference = checkedWebImageReference(JSON.parse(row.body));
    if (reference.id !== row.id || reference.brainId !== row.brain_id || reference.turnId !== row.turn_id || ![0, 1].includes(Number(row.active))) throw new Error("Web image reference row lost its owner binding.");
    return { reference, active: row.active === 1 };
  }
  resolve(turnId: string, referenceId: string): WebImageReference {
    if (!ID.test(turnId) || !SHA.test(referenceId) || !this.present()) throw new Error("Select an image reference actually returned to this task.");
    this.db.exec("BEGIN");
    try { const row = this.row(referenceId); if (!row.active || row.reference.brainId !== this.brainId || row.reference.turnId !== turnId) throw new Error("This image reference is dormant or belongs to another task."); this.db.exec("COMMIT"); return row.reference; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  downloadForAttachment(attachmentId: string, requestId: string): WebImageDownload {
    if (!ID.test(attachmentId) || !ID.test(requestId)) throw new Error("Invalid selected image receipt identity.");
    this.db.exec("BEGIN");
    try {
      const size = this.db.prepare("SELECT length(CAST(source_json AS BLOB)) AS bytes FROM attachments WHERE id=? AND request_id=? AND brain_id=?").get(attachmentId, requestId, this.brainId);
      if (!size) throw new Error("Selected image has no current owned attachment receipt."); webImageReadBudget(Number(size.bytes));
      const row = this.db.prepare("SELECT source_json,source_sha256,content_sha256,bytes,mime_type,kind FROM attachments WHERE id=? AND request_id=? AND brain_id=?").get(attachmentId, requestId, this.brainId)!;
      if (typeof row.source_json !== "string" || createHash("sha256").update(row.source_json).digest("hex") !== row.source_sha256) throw new Error("Selected image provenance bytes changed.");
      const source = JSON.parse(row.source_json), download = checkedWebImageDownload(source.webImage);
      if (source.kind !== "web-image" || source.selectedAt !== download.fetchedAt || row.content_sha256 !== download.sha256 || row.bytes !== download.bytes || row.mime_type !== download.mimeType || row.kind !== "image") throw new Error("Selected image download differs from its attachment receipt.");
      this.db.exec("COMMIT"); return download;
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  validate(): void {
    if (!this.present()) return;
    this.db.exec("BEGIN");
    try { for (const row of this.db.prepare("SELECT id FROM web_image_references ORDER BY id").iterate()) { const saved = this.row(String(row.id)); if (saved.active && saved.reference.brainId !== this.brainId) throw new Error("Inherited web image references cannot inherit live task authority."); } this.db.exec("COMMIT"); }
    catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  retire(): void { if (this.readOnly) throw new Error("Read-only image references cannot rekey authority."); if (this.present()) this.db.prepare("UPDATE web_image_references SET active=0").run(); }
}

export async function validateWebImageReferences(root: string, brainId: string): Promise<void> {
  const store = await WebImageReferenceStore.open(root, brainId, true); try { store.validate(); } finally { store.close(); }
}
export function retireWebImageReferences(db: DatabaseSync): void {
  if (schemaChecked(db)) db.prepare("UPDATE web_image_references SET active=0").run();
}
export function mergeWebImageReferences(source: DatabaseSync, target: DatabaseSync): void {
  if (!schemaChecked(source)) return;
  schemaChecked(target, true);
  const insert = target.prepare("INSERT OR IGNORE INTO web_image_references VALUES(?,?,?,?,?,0)");
  for (const row of source.prepare("SELECT id FROM web_image_references ORDER BY id").iterate()) {
    const size = source.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM web_image_references WHERE id=?").get(String(row.id))!;
    webImageReadBudget(Number(size.bytes)); const original = source.prepare("SELECT * FROM web_image_references WHERE id=?").get(String(row.id))!;
    insert.run(String(original.id), String(original.brain_id), String(original.turn_id), String(original.body), String(original.sha256));
    const saved = target.prepare("SELECT body,sha256 FROM web_image_references WHERE id=?").get(String(original.id))!;
    if (saved.body !== original.body || saved.sha256 !== original.sha256) throw new Error("Merged web image provenance conflicts with an existing exact reference.");
  }
}
