import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { lstat, mkdtemp, open, rm } from "node:fs/promises";
import { join } from "node:path";
import type { ChatTaskAttachmentHandle, ChatTaskAttachmentResolved } from "../shared/taskAttachments";
import type { WebImageDownload, WebImageReference, WebImageSelection } from "../shared/webImages";
import type { TaskAttachmentStagePath } from "./taskAttachmentStore";
import type { LearningExecutionContext } from "./learningExecutionContext";
import { decideWebSource, publicationFromHtml, webUrlWithinScope } from "./webLearningPolicy";
import { checkedWebImageDownload, checkedWebImageSelection, WebImageReferenceStore, webImageCanonical, webImageHash, webImageSelectionIdentity, webImageUrl } from "./webImageReferences";

export interface WebImageServiceDependencies {
  root(brainId: string): Promise<string>;
  initializeAttachments(brainId: string): Promise<void>;
  requireLoaded(brainId: string): void | Promise<void>;
  checkResources(brainId: string, bytes: number): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  permission(brainId: string): Promise<{ level: string; revision: string }>;
  learningContext?(brainId: string, turnId: string): Promise<LearningExecutionContext | undefined>;
  fetchPublic(url: URL, init: RequestInit, maximumRedirects: number, hooks: { beforeRequest(url: URL): Promise<void>; afterResponse(url: URL, response: Response): void }): Promise<Response>;
  attachments: {
    requestHandles(brainId: string, requestId: string): Promise<ChatTaskAttachmentHandle[] | undefined>;
    stagePaths(brainId: string, requestId: string, paths: readonly TaskAttachmentStagePath[], selection: "images", signal?: AbortSignal): Promise<ChatTaskAttachmentHandle[]>;
    bindToTurn(brainId: string, turnId: string, ids: readonly string[]): Promise<ChatTaskAttachmentResolved[]>;
    resolveForTurn(brainId: string, turnId: string, attachmentId: string, signal?: AbortSignal): Promise<ChatTaskAttachmentResolved>;
  };
}

/** Typed metadata only. The potentially large image remains in its CAS file. */
export async function readWebImageActionEvidence(path: string, brainId: string, evidenceId: string, executionId: string, expectedSha256: string,
  admit: (bytes: number) => Promise<void>): Promise<WebImageSelection> {
  const before = await lstat(path);
  if (!before.isFile() || before.isSymbolicLink()) throw new Error("Selected image action evidence requires a real owned file.");
  await admit(before.size * 16 + CHUNK * 4);
  const file = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const opened = await file.stat();
    const same = (value: typeof opened): boolean => value.isFile() && value.dev === opened.dev && value.ino === opened.ino && value.size === opened.size && value.mtimeMs === opened.mtimeMs && value.ctimeMs === opened.ctimeMs;
    if (!same(before)) throw new Error("Selected image action evidence was replaced.");
    const bytes = await file.readFile(), checksum = createHash("sha256").update(bytes).digest("hex");
    if (checksum !== expectedSha256 || !same(await file.stat()) || !same(await lstat(path))) throw new Error("Selected image action evidence changed.");
    const body = JSON.parse(bytes.toString("utf8")), selected = checkedWebImageSelection(body.output);
    if (body.format !== "omni-completed-action-result-v1" || body.originBrainId !== brainId || body.evidenceId !== evidenceId || body.executionId !== executionId ||
      body.toolId !== "web.fetch" || body.action !== "inspect-image" || body.executionState !== "complete" || body.outputPresent !== true || body.chatTurnId !== selected.turnId ||
      body.arguments?.referenceId !== selected.reference.id || selected.download.requestId !== webImageHash(["web-image", brainId, selected.turnId, body.actionEventId, selected.reference.id])) throw new Error("Selected image action evidence lost its exact original operands/source binding.");
    return selected;
  } finally { await file.close(); }
}
const CHUNK = 65536, ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
function signature(header: Buffer, mime: string): boolean {
  return mime === "image/png" && header.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])) ||
    mime === "image/jpeg" && header[0] === 255 && header[1] === 216 && header[2] === 255 ||
    mime === "image/webp" && header.subarray(0, 4).toString("ascii") === "RIFF" && header.subarray(8, 12).toString("ascii") === "WEBP" ||
    mime === "image/gif" && ["GIF87a", "GIF89a"].includes(header.subarray(0, 6).toString("ascii"));
}

/** Explicit selected pixels only; no provider thumbnail is fetched on discovery. */
export class WebImageService {
  constructor(private readonly deps: WebImageServiceDependencies) {}
  async register(brainId: string, turnId: string | undefined, source: WebImageReference["source"],
    candidates: Iterable<{ url: string; pageUrl: string; label?: string }>, signal: AbortSignal): Promise<WebImageReference[]> {
    if (!turnId) return [];
    signal.throwIfAborted(); const root = await this.deps.root(brainId); await this.deps.initializeAttachments(brainId);
    const store = await WebImageReferenceStore.open(root, brainId), result: WebImageReference[] = [], seen = new Set<string>(); let bytes = 0;
    try { for (const candidate of candidates) {
      signal.throwIfAborted(); const key = webImageHash([candidate.url, candidate.pageUrl]); if (seen.has(key)) continue;
      bytes += Buffer.byteLength(webImageCanonical(candidate)) + 2048;
      await this.deps.checkResources(brainId, bytes * 12 + CHUNK * 4); await this.deps.reserveWrite(root, bytes + CHUNK);
      const reference = store.record(turnId, source, candidate); result.push(reference); seen.add(key);
    } return result; } finally { store.close(); }
  }
  async inspect(brainId: string, turnId: string | undefined, requestId: string, referenceId: string, signal: AbortSignal,
    requireAuthority: () => Promise<void>, learning?: LearningExecutionContext, maxBytes?: number): Promise<WebImageSelection> {
    if (!turnId || !ID.test(turnId) || !ID.test(requestId)) throw new Error("Selected web images require their exact current chat action scope.");
    if (maxBytes !== undefined && (!Number.isSafeInteger(maxBytes) || maxBytes < 1)) throw new Error("Optional image download stop size must be a positive safe integer.");
    const root = await this.deps.root(brainId); await this.deps.requireLoaded(brainId); await this.deps.initializeAttachments(brainId);
    const references = await WebImageReferenceStore.open(root, brainId);
    let temporary: string | undefined, responseBody: ReadableStream<Uint8Array> | null = null;
    try {
    const reference = references.resolve(turnId, referenceId);
    const stageId = webImageHash(["web-image", brainId, turnId, requestId, referenceId]);
    const scopeTurnId = "webimg-" + webImageHash([brainId, turnId, stageId, referenceId]);
    const authority = await this.deps.permission(brainId), policySha256 = (await learning?.policy())?.sha256;
    if (authority.level === "off") throw new Error("Web image inspection is disabled.");
    const check = async (url?: URL): Promise<void> => {
      signal.throwIfAborted(); await requireAuthority(); await this.deps.requireLoaded(brainId);
      const permission = await this.deps.permission(brainId);
      if (permission.level === "off" || permission.revision !== authority.revision) throw new Error("Web image inspection authority changed before observation.");
      if (references.resolve(turnId, referenceId).id !== reference.id) throw new Error("Selected image reference changed.");
      if (learning) {
        const policy = await learning.policy();
        if (policy.sha256 !== policySha256) throw new Error("Selected image learning policy changed.");
        if (!webUrlWithinScope(new URL(reference.pageUrl), policy.constraints) || url && !webUrlWithinScope(url, policy.constraints)) throw new Error("Selected image is outside this learning task's current source scope.");
        const decision = decideWebSource(reference.pageUrl, new URL(reference.pageUrl), "", publicationFromHtml(""), policy);
        if (decision.decision !== "admitted") throw new Error("Selected image publication/scope is unverified for this task: " + decision.decision + ".");
      }
      await this.deps.checkResources(brainId, CHUNK * 4 + 131072); signal.throwIfAborted();
    };
      await check(new URL(reference.url));
      let handles = await this.deps.attachments.requestHandles(brainId, stageId);
      if (!handles) {
        await this.deps.reserveWrite(root, CHUNK * 4); temporary = await mkdtemp(join(root, ".web-image-"));
        const redirects: string[] = [], requested = new URL(reference.url);
        const response = await this.deps.fetchPublic(requested, { signal, headers: { Accept: "image/png,image/jpeg,image/webp,image/gif" } }, 5, {
          beforeRequest: async url => { await check(url); },
          afterResponse: url => { redirects.push(webImageUrl(url.toString())); }
        });
        responseBody = response.body;
        if (!response.ok) { await response.body?.cancel().catch(() => undefined); throw new Error("Selected image returned HTTP " + response.status + "."); }
        const mime = (response.headers.get("content-type") ?? "").split(";", 1)[0]!.trim().toLowerCase();
        if (!["image/png", "image/jpeg", "image/webp", "image/gif"].includes(mime)) { await response.body?.cancel().catch(() => undefined); throw new Error("Selected web image requires a supported image MIME, not HTML, SVG, a path or a decoder URL."); }
        const declaredHeader = response.headers.get("content-length");
        const declared = declaredHeader === null ? undefined : /^\d+$/.test(declaredHeader) ? Number(declaredHeader) : NaN;
        if (declared !== undefined && (!Number.isSafeInteger(declared) || declared < 1 || maxBytes !== undefined && declared > maxBytes)) { await response.body?.cancel().catch(() => undefined); throw new Error("Selected image has an invalid declared length or exceeds its explicit stop size."); }
        if (declared) await this.deps.reserveWrite(root, declared * 2 + CHUNK * 4);
        if (!response.body) throw new Error("Selected image returned no body.");
        const path = join(temporary, "source.image"), file = await open(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | (constants.O_NOFOLLOW ?? 0), 0o600), reader = response.body.getReader();
        const digest = createHash("sha256"); let bytes = 0, header = Buffer.alloc(0), finished = false;
        try {
          for (;;) {
            await check(new URL(redirects.at(-1) ?? reference.url)); const part = await reader.read(); if (part.done) { finished = true; break; } if (!part.value) continue;
            await this.deps.checkResources(brainId, part.value.byteLength * 2 + CHUNK * 4);
            for (let offset = 0; offset < part.value.byteLength; offset += CHUNK) {
              const piece = part.value.subarray(offset, offset + CHUNK);
              if (!Number.isSafeInteger(bytes + piece.byteLength) || maxBytes !== undefined && bytes + piece.byteLength > maxBytes) throw new Error("Selected image exceeds its exact resource cursor or explicit stop size.");
              await check(); await this.deps.reserveWrite(root, piece.byteLength * 2 + CHUNK * 4);
              if (header.byteLength < 32) header = Buffer.concat([header, Buffer.from(piece.subarray(0, 32 - header.byteLength))]);
              digest.update(piece); let written = 0;
              while (written < piece.byteLength) { const step = await file.write(piece, written, piece.byteLength - written); if (step.bytesWritten < 1) throw new Error("Selected image had a short disk write."); written += step.bytesWritten; }
              bytes += piece.byteLength;
            }
          }
          if (!bytes || declared !== undefined && bytes !== declared || !signature(header, mime)) throw new Error("Selected image MIME/signature or complete declared bytes do not match.");
          if (mime === "image/webp" && header.length >= 12 && header.readUInt32LE(4) + 8 > bytes) throw new Error("Selected WebP container is truncated.");
          await file.sync();
        } finally { if (!finished) await reader.cancel("selected image interrupted").catch(() => undefined); reader.releaseLock(); await file.close(); }
        const download: WebImageDownload = checkedWebImageDownload({ reference, requestId: stageId, ownerTurnId: turnId, scopeTurnId,
          requestedUrl: reference.url, finalUrl: redirects.at(-1) ?? reference.url, redirects: redirects.length ? redirects : [reference.url], fetchedAt: new Date().toISOString(),
          permissionRevision: authority.revision, ...(policySha256 ? { learningPolicySha256: policySha256 } : {}),
          mimeType: mime, sha256: digest.digest("hex"), bytes });
        if (reference.source.kind === "image-response" && download.sha256 !== reference.source.sha256) throw new Error("Previously fetched image bytes changed; select a newly observed source.");
        await check(new URL(download.finalUrl));
        const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink() || info.size !== bytes) throw new Error("Selected image staging source changed.");
        handles = await this.deps.attachments.stagePaths(brainId, stageId, [{ path, displayName: "Selected web image", mimeType: download.mimeType, kind: "image",
          source: { kind: "web-image", selectedAt: download.fetchedAt, webImage: download } }], "images", signal);
      }
      if (handles.length !== 1) throw new Error("Selected image action has a different committed attachment shape.");
      const download = references.downloadForAttachment(handles[0]!.id, stageId);
      if (download.reference.id !== reference.id || download.ownerTurnId !== turnId || download.scopeTurnId !== scopeTurnId || download.permissionRevision !== authority.revision || download.learningPolicySha256 !== policySha256) throw new Error("Selected image request already owns another source or former authority.");
      await check(new URL(download.finalUrl));
      await this.deps.attachments.bindToTurn(brainId, scopeTurnId, [handles[0]!.id]);
      const attachment = await this.deps.attachments.resolveForTurn(brainId, scopeTurnId, handles[0]!.id, signal);
      await check(new URL(download.finalUrl));
      const identity = { format: "omni-web-image-selection" as const, version: 1 as const, brainId, turnId, reference, download, attachment };
      return checkedWebImageSelection({ ...identity, selectionSha256: webImageSelectionIdentity(identity),
        observation: { state: "pending-native-perception", pixelsRead: false, parametersChanged: false, rawSourceTextInjected: false } });
    } finally { references.close(); await responseBody?.cancel("selected image preparation closed").catch(() => undefined); if (temporary) await rm(temporary, { recursive: true, force: true }); }
  }
  async validateObservation(brainId: string, turnId: string, value: unknown, signal?: AbortSignal): Promise<WebImageSelection> {
    const selected = checkedWebImageSelection(value);
    if (selected.brainId !== brainId || selected.turnId !== turnId) throw new Error("Selected web image belongs to another observed action owner.");
    signal?.throwIfAborted(); await this.deps.requireLoaded(brainId); const root = await this.deps.root(brainId);
    const permission = await this.deps.permission(brainId);
    if (permission.level === "off" || permission.revision !== selected.download.permissionRevision) throw new Error("Selected web image is pending renewed current authority, not silently observed.");
    const context = await this.deps.learningContext?.(brainId, turnId), policy = await context?.policy();
    if (policy?.sha256 !== selected.download.learningPolicySha256) throw new Error("Selected web image's original learning scope is no longer current.");
    if (policy && (!webUrlWithinScope(new URL(selected.download.finalUrl), policy.constraints) || !webUrlWithinScope(new URL(selected.reference.pageUrl), policy.constraints))) throw new Error("Selected web image is outside the current learning scope.");
    const store = await WebImageReferenceStore.open(root, brainId, true);
    try {
      if (store.resolve(turnId, selected.reference.id).id !== selected.reference.id || webImageHash(store.downloadForAttachment(selected.attachment.attachmentId, selected.download.requestId)) !== webImageHash(selected.download)) throw new Error("Selected image's source custody changed.");
      const attachment = await this.deps.attachments.resolveForTurn(brainId, selected.download.scopeTurnId, selected.attachment.attachmentId, signal);
      if (webImageHash(attachment) !== webImageHash(selected.attachment)) throw new Error("Selected image attachment was replaced.");
      await this.deps.checkResources(brainId, CHUNK * 4 + 131072); signal?.throwIfAborted(); return selected;
    } finally { store.close(); }
  }
}
