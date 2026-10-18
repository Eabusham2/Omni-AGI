import { createHash, randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import { chmod, lstat, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type {
  ExternalAdviceCancellation,
  ExternalAdviceChannel,
  ExternalAdviceChannelConfiguration,
  ExternalAdviceConfigureInput,
  ExternalAdviceContextItem,
  ExternalAdviceContextSelection,
  ExternalAdviceCredentialSaveInput,
  ExternalAdviceCredentialStatus,
  ExternalAdviceEvent,
  ExternalAdviceHistoryPage,
  ExternalAdviceHistoryQuery,
  ExternalAdviceModelDiscoveryInput,
  ExternalAdviceModelDiscoveryResult,
  ExternalAdviceModelDiscoveryCancellation,
  ExternalAdvicePreview,
  ExternalAdvicePreviewInput,
  ExternalAdviceProvider,
  ExternalAdviceRequest,
  ExternalAdviceReviewReceipt,
  ExternalAdviceSelectedContext,
  ExternalAdviceSettings,
  ExternalAdviceStartInput
} from "../shared/externalAdvice";
import type { ActionReviewInvocation, ActionReviewerApprovalReview } from "../shared/actionReviewer";
import type { SecureSecretStore } from "./secureSecretStore";
import { ExternalAdviceClient, externalAdviceLoopbackEndpoint, type ExternalAdviceClientRequest } from "./externalAdviceClient";
import { apiTeacherProvider, ollamaLoopbackEndpoint, ollamaLocalModelId } from "./apiTeacherProviders";
import { ApiTeacherModelDiscoveryService } from "./apiTeacherModelDiscovery";
import type { ApiTeacherModelDiscoveryRequest } from "../shared/types";

export const EXTERNAL_ADVICE_RELATIVE_PATH = "external-advice/workspace.sqlite3";
export const EXTERNAL_ADVICE_FORMAT = "omni-external-advice-workspace-1";
export const EXTERNAL_ADVICE_REQUEST_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const PROVIDERS: readonly ExternalAdviceProvider[] = ["openai", "anthropic", "gemini", "openrouter", "ollama", "openai-compatible-local"];
const CHANNELS: readonly ExternalAdviceChannel[] = ["assistant", "reviewer"];
const MODEL = /^[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}$/;
const SHA = /^[a-f0-9]{64}$/;
const MAX_TEXT_BYTES = 4 * 1024 * 1024;
const MAX_RECORD_BYTES = 16 * 1024 * 1024;
const MAX_HISTORY_PAGE_BYTES = 64 * 1024 * 1024;
const RESPONSE_BATCH_BYTES = 64 * 1024;
const RESPONSE_BATCH_MS = 500;
const MIN_SQLITE_WRITE_ESTIMATE = 512 * 1024;
const DEVICE_FORMAT = "omni-external-advice-device-credentials-1";
const TERMINAL = new Set(["complete", "cancelled", "failed"]);

const schema = new Map(Object.entries({
  meta: "CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
  settings: "CREATE TABLE settings(channel TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  requests: "CREATE TABLE requests(id TEXT PRIMARY KEY,channel TEXT NOT NULL,state TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)",
  requests_updated: "CREATE INDEX requests_updated ON requests(updated_at,id)",
  inherited_requests: "CREATE TABLE inherited_requests(source_brain_id TEXT NOT NULL,source_request_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_request_id,source_sha256))",
  inherited_settings: "CREATE TABLE inherited_settings(source_brain_id TEXT NOT NULL,channel TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,channel,source_sha256))",
  inherited_meta: "CREATE TABLE inherited_meta(source_brain_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_sha256))",
  reviews: "CREATE TABLE reviews(id TEXT PRIMARY KEY,owner_brain_id TEXT NOT NULL,source_brain_id TEXT NOT NULL,review_id TEXT NOT NULL,finished_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL)"
}));

type AdviceSecrets = Pick<SecureSecretStore, "get" | "set" | "remove" | "has" | "persistence">;

export interface ExternalAdviceDeviceAuthority {
  brainId: string;
  channel: ExternalAdviceChannel;
  enabled: boolean;
  ownerEpoch: string;
  revision: number;
}

export interface ExternalAdviceServiceOptions {
  deviceDirectory: string;
  brainRoot(brainId: string): string | Promise<string>;
  authority(brainId: string, channel: ExternalAdviceChannel): Promise<ExternalAdviceDeviceAuthority>;
  selectContext(
    input: { brainId: string; channel: ExternalAdviceChannel; selection: ExternalAdviceContextSelection },
    signal: AbortSignal
  ): Promise<ExternalAdviceSelectedContext>;
  /** Production injects the shared disk-floor/resource admission. Omission fails closed. */
  admitWrite?(brainId: string, additionalBytes: number, signal: AbortSignal): Promise<void>;
  now?: () => Date;
  uuid?: () => string;
  monotonicNow?: () => number;
  /** Injected HTTP for tests; never contacted until explicit model discovery. */
  discoveryFetch?: typeof fetch;
}

interface DeviceCredentialRecord { id: string; revision: number }
interface DeviceDocument {
  format: typeof DEVICE_FORMAT;
  version: 1;
  revision: number;
  credentials: Partial<Record<ExternalAdviceProvider, DeviceCredentialRecord>>;
}

interface ActiveAdvice {
  brainId: string;
  channel: ExternalAdviceChannel;
  provider: ExternalAdviceProvider;
  controller: AbortController;
  promise: Promise<ExternalAdviceRequest>;
}
interface ActiveAdviceDiscovery {
  brainId: string; channel: ExternalAdviceChannel; requestId: string;
  settingsRevision: number; credentialRevision: number;
  provider?: ApiTeacherModelDiscoveryRequest["provider"];
  wire?: ApiTeacherModelDiscoveryRequest;
  cancelled: boolean;
  promise: Promise<ExternalAdviceModelDiscoveryResult>;
}

export interface ExternalAdviceWorkspaceInventory {
  brainId: string;
  settingsRevision: number;
  settings: Record<ExternalAdviceChannel, ExternalAdviceChannelConfiguration>;
  requests: number;
  inheritedRequests: number;
  inheritedSettings: number;
  inheritedMeta: number;
  reviews: number;
}

export interface ExternalAdviceStoredReviewRow {
  id: string;
  owner_brain_id: string;
  source_brain_id: string;
  review_id: string;
  finished_at: string;
  payload: string;
  sha256: string;
}

export function externalAdviceCanonical(value: unknown): string {
  const visit = (item: unknown): unknown => Array.isArray(item) ? item.map(visit) : item && typeof item === "object"
    ? Object.fromEntries(Object.entries(item as Record<string, unknown>)
      .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
      .map(([key, child]) => [key, visit(child)])) : item;
  return JSON.stringify(visit(value));
}

export function externalAdviceSha256(value: string): string {
  return createHash("sha256").update(value, "utf8").digest("hex");
}

export function externalAdviceRequestId(value: unknown): string {
  if (typeof value !== "string" || !EXTERNAL_ADVICE_REQUEST_ID.test(value)) throw new Error("Invalid external-advice request ID.");
  return value;
}

function brainId(value: unknown): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid external-advice brain owner.");
  return value;
}

function channel(value: unknown): ExternalAdviceChannel {
  if (!CHANNELS.includes(value as ExternalAdviceChannel)) throw new Error("Invalid external-advice channel.");
  return value as ExternalAdviceChannel;
}

function provider(value: unknown): ExternalAdviceProvider {
  if (!PROVIDERS.includes(value as ExternalAdviceProvider)) throw new Error("Invalid external-advice provider.");
  return value as ExternalAdviceProvider;
}

function sha(value: unknown, label: string): string {
  if (typeof value !== "string" || !SHA.test(value)) throw new Error(`${label} is invalid.`);
  return value;
}

function safeTime(value: unknown, label: string): string {
  if (typeof value !== "string" || !Number.isFinite(Date.parse(value))) throw new Error(`${label} is invalid.`);
  return value;
}

function safeText(value: unknown, label: string, maximumBytes = MAX_TEXT_BYTES, allowEmpty = false): string {
  if (typeof value !== "string" || value.includes("\0") || (!allowEmpty && !value.trim()) || Buffer.byteLength(value) > maximumBytes) {
    throw new Error(`${label} is empty or exceeds its transfer budget.`);
  }
  return value;
}

function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} is invalid.`);
  return value as Record<string, unknown>;
}

function known(row: Record<string, unknown>, keys: readonly string[], label: string): void {
  const allowed = new Set(keys);
  if (Object.keys(row).some((key) => !allowed.has(key))) throw new Error(`${label} contains unsupported fields.`);
}

function contextSelection(value: ExternalAdviceContextSelection): ExternalAdviceContextSelection {
  const row = object(value, "External-advice context selection"); known(row, ["includeTask", "includeActivity", "maxCharacters"], "External-advice context selection");
  if (typeof value.includeTask !== "boolean" || typeof value.includeActivity !== "boolean" || !Number.isSafeInteger(value.maxCharacters) ||
      value.maxCharacters < 0 || value.maxCharacters > 1_000_000) {
    throw new Error("External-advice context selection is invalid.");
  }
  return { includeTask: value.includeTask, includeActivity: value.includeActivity, maxCharacters: value.maxCharacters };
}

function configuration(value: ExternalAdviceChannelConfiguration): ExternalAdviceChannelConfiguration {
  const row = object(value, "External-advice configuration");
  known(row, ["enabled", "provider", "model", "endpoint", "maxOutputTokens", "context"], "External-advice configuration");
  if (typeof value.enabled !== "boolean") throw new Error("External-advice configuration is invalid.");
  const selectedProvider = provider(value.provider), selectedModel = value.model;
  if (typeof selectedModel !== "string" || selectedModel.length > 200 || selectedModel && !MODEL.test(selectedModel) || value.enabled && !selectedModel) {
    throw new Error("Enter a valid manual external-advice model ID before enabling this channel.");
  }
  if (!Number.isSafeInteger(value.maxOutputTokens) || value.maxOutputTokens < 1 || value.maxOutputTokens > 65_536) {
    throw new Error("External-advice output tokens must be between 1 and 65,536.");
  }
  let endpoint: string | undefined;
  if (selectedProvider === "ollama") {
    if (selectedModel) ollamaLocalModelId(selectedModel);
    endpoint = ollamaLoopbackEndpoint(value.endpoint);
  } else if (selectedProvider === "openai-compatible-local") {
    if (value.endpoint !== undefined && value.endpoint !== "") endpoint = externalAdviceLoopbackEndpoint(value.endpoint);
    if (value.enabled && !endpoint) throw new Error("Enter an explicit loopback endpoint before enabling local external advice.");
  } else if (value.endpoint !== undefined) throw new Error("Remote provider endpoints are fixed and cannot be overridden.");
  return { enabled: value.enabled, provider: selectedProvider, model: selectedModel,
    ...(endpoint ? { endpoint } : {}), maxOutputTokens: value.maxOutputTokens, context: contextSelection(value.context) };
}

function defaultConfiguration(): ExternalAdviceChannelConfiguration {
  return { enabled: false, provider: "openai", model: "", maxOutputTokens: 2_048,
    context: { includeTask: true, includeActivity: false, maxCharacters: 32_000 } };
}

function schemaSql(value: unknown): string {
  return String(value).replace(/IF\s+NOT\s+EXISTS\s+/gi, "").replace(/\s+|;$/g, "").toUpperCase();
}

function requestIntent(value: ExternalAdviceRequest): Record<string, unknown> {
  return { id: value.id, brainId: value.brainId, channel: value.channel, provider: value.provider, model: value.model,
    ...(value.endpoint ? { endpoint: value.endpoint } : {}), prompt: value.prompt, requestText: value.requestText,
    context: value.context, settingsRevision: value.settingsRevision, authorityRevision: value.authorityRevision,
    authorityOwnerEpoch: value.authorityOwnerEpoch, contextSha256: value.contextSha256, previewSha256: value.previewSha256,
    maxOutputTokens: value.maxOutputTokens };
}

function previewIntent(value: ExternalAdviceRequest, owner: string): Record<string, unknown> {
  return { format: "omni-external-advice-preview", version: 1, brainId: owner, channel: value.channel, provider: value.provider,
    model: value.model, ...(value.endpoint ? { endpoint: value.endpoint } : {}), prompt: value.prompt, requestText: value.requestText,
    context: value.context, settingsRevision: value.settingsRevision, authorityRevision: value.authorityRevision,
    authorityOwnerEpoch: value.authorityOwnerEpoch, contextSha256: value.contextSha256, maxOutputTokens: value.maxOutputTokens,
    advisoryOnly: true, automaticLearning: false };
}

export function externalAdviceInheritedRequestId(sourceBrainId: string, sourceRequestId: string, sourceSha256: string, targetBrainId: string): string {
  return externalAdviceSha256(externalAdviceCanonical({ format: "omni-external-advice-inherited-request-id", version: 1,
    sourceBrainId: brainId(sourceBrainId), sourceRequestId: externalAdviceRequestId(sourceRequestId),
    sourceSha256: sha(sourceSha256, "Inherited external-advice receipt hash"), targetBrainId: brainId(targetBrainId) }));
}

export function externalAdviceInheritedReviewId(sourceBrainId: string, reviewId: string, sourceSha256: string, targetBrainId: string): string {
  return externalAdviceSha256(externalAdviceCanonical({ format: "omni-external-advice-inherited-review-id", version: 1,
    sourceBrainId: brainId(sourceBrainId), reviewId: externalAdviceRequestId(reviewId),
    sourceSha256: sha(sourceSha256, "Inherited action-review receipt hash"), targetBrainId: brainId(targetBrainId) }));
}

function reviewText(value: unknown, label: string, maximum = MAX_TEXT_BYTES, allowEmpty = false): string {
  if (typeof value !== "string" || (!allowEmpty && !value) || Buffer.byteLength(value) > maximum) throw new Error(`${label} is invalid or exceeds its custody budget.`);
  return value;
}

function reviewJson(value: unknown, depth = 0): unknown {
  if (depth > 64) throw new Error("Action-review arguments exceed the supported JSON depth.");
  if (value === null || typeof value === "boolean" || typeof value === "string") return value;
  if (typeof value === "number") { if (!Number.isFinite(value)) throw new Error("Action-review arguments contain a nonfinite number."); return value; }
  if (Array.isArray(value)) return value.map((item) => reviewJson(item, depth + 1));
  const row = object(value, "Action-review arguments");
  return Object.fromEntries(Object.entries(row).map(([key, item]) => [key, reviewJson(item, depth + 1)]));
}

function reviewInvocation(value: unknown, owner: string): ActionReviewInvocation {
  const row = object(value, "Action-review invocation"); known(row, ["brainId", "toolId", "action", "arguments"], "Action-review invocation");
  if (row.brainId !== owner || ![row.toolId, row.action].every((item) => typeof item === "string" && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(item))) {
    throw new Error("Action-review invocation has another owner or invalid operation identity.");
  }
  const args = reviewJson(row.arguments);
  if (!args || typeof args !== "object" || Array.isArray(args)) throw new Error("Action-review invocation arguments must be a JSON object.");
  return { brainId: owner, toolId: String(row.toolId), action: String(row.action), arguments: args as Record<string, unknown> };
}

export function validateExternalAdviceActionReview(value: unknown): ActionReviewerApprovalReview {
  const row = object(value, "Action-review receipt");
  known(row, ["kind", "formatVersion", "reviewId", "brainId", "requestId", "configurationRevision", "ownerRevision", "reviewerId",
    "permissionRevision", "proposalSha256", "proposedInvocation", "resolvedInvocation", "disposition", "startedAt", "finishedAt", "elapsedMs",
    "observedContext", "requestBytes", "responseBytes", "verdict", "summary", "evidence"], "Action-review receipt");
  if (row.kind !== "action-review" || row.formatVersion !== 1) throw new Error("Action-review receipt has another format.");
  const owner = brainId(row.brainId), reviewId = externalAdviceRequestId(row.reviewId), requestId = externalAdviceRequestId(row.requestId);
  for (const key of ["configurationRevision", "ownerRevision", "reviewerId", "permissionRevision"] as const) reviewText(row[key], `Action-review ${key}`, 512);
  const proposalSha256 = sha(row.proposalSha256, "Action-review proposal hash");
  const proposedInvocation = reviewInvocation(row.proposedInvocation, owner);
  const resolvedInvocation = row.resolvedInvocation === undefined ? undefined : reviewInvocation(row.resolvedInvocation, owner);
  if (!['reviewed', 'unavailable', 'unparseable', 'timeout', 'budget-refused'].includes(String(row.disposition)) || !['allow', 'ask'].includes(String(row.verdict))) {
    throw new Error("Action-review disposition or verdict is invalid.");
  }
  const startedAt = safeTime(row.startedAt, "Action-review start time"), finishedAt = safeTime(row.finishedAt, "Action-review finish time");
  if (Date.parse(finishedAt) < Date.parse(startedAt) || typeof row.elapsedMs !== "number" || !Number.isFinite(row.elapsedMs) || row.elapsedMs < 0 ||
      !Number.isSafeInteger(row.requestBytes) || Number(row.requestBytes) < 0 || !Number.isSafeInteger(row.responseBytes) || Number(row.responseBytes) < 0) {
    throw new Error("Action-review timing or byte accounting is invalid.");
  }
  if (!Array.isArray(row.observedContext) || !Array.isArray(row.evidence)) throw new Error("Action-review observed context or evidence is invalid.");
  const source = new Map<string, string>([["action.proposed", JSON.stringify(proposedInvocation)]]);
  if (resolvedInvocation) source.set("action.resolved", JSON.stringify(resolvedInvocation));
  const observedContext = row.observedContext.map((item) => {
    const observed = object(item, "Action-review observed item"); known(observed, ["id", "kind", "text"], "Action-review observed item");
    const id = reviewText(observed.id, "Action-review observed ID", 256);
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(id) || source.has(id) || !["task", "work", "self-report"].includes(String(observed.kind))) {
      throw new Error("Action-review observed item identity is invalid or duplicated.");
    }
    const text = reviewText(observed.text, "Action-review observed text", MAX_TEXT_BYTES, true); source.set(id, text);
    return { id, kind: observed.kind as "task" | "work" | "self-report", text };
  });
  const summary = reviewText(row.summary, "Action-review summary", 512 * 1024);
  const evidence = row.evidence.map((item) => {
    const evidenceRow = object(item, "Action-review evidence"); known(evidenceRow, ["sourceId", "quote"], "Action-review evidence");
    const sourceId = reviewText(evidenceRow.sourceId, "Action-review evidence source", 256), quote = reviewText(evidenceRow.quote, "Action-review evidence quote", 512 * 1024);
    if (!source.get(sourceId)?.includes(quote)) throw new Error("Action-review evidence is not an exact retained quote.");
    return { sourceId, quote };
  });
  if (row.verdict === "allow" && !evidence.length) throw new Error("Action-review allow verdict lacks retained evidence.");
  const receipt: ActionReviewerApprovalReview = { kind: "action-review", formatVersion: 1, reviewId, brainId: owner, requestId,
    configurationRevision: String(row.configurationRevision), ownerRevision: String(row.ownerRevision), reviewerId: String(row.reviewerId),
    permissionRevision: String(row.permissionRevision), proposalSha256, proposedInvocation, ...(resolvedInvocation ? { resolvedInvocation } : {}),
    disposition: row.disposition as ActionReviewerApprovalReview["disposition"], startedAt, finishedAt, elapsedMs: row.elapsedMs,
    observedContext, requestBytes: Number(row.requestBytes), responseBytes: Number(row.responseBytes), verdict: row.verdict as "allow" | "ask", summary, evidence };
  if (Buffer.byteLength(externalAdviceCanonical(receipt)) > MAX_RECORD_BYTES) throw new Error("Action-review receipt exceeds the 16 MiB custody budget.");
  return receipt;
}

export function validateExternalAdviceStoredReview(value: unknown, owner: string): { receipt: ActionReviewerApprovalReview; payload: string; sha256: string } {
  const row = object(value, "Stored action-review row");
  known(row, ["id", "owner_brain_id", "source_brain_id", "review_id", "finished_at", "payload", "sha256"], "Stored action-review row");
  const id = externalAdviceRequestId(row.id), currentOwner = brainId(row.owner_brain_id), sourceOwner = brainId(row.source_brain_id);
  if (currentOwner !== brainId(owner)) throw new Error("Stored action-review row has another workspace owner.");
  const reviewId = externalAdviceRequestId(row.review_id), finishedAt = safeTime(row.finished_at, "Stored action-review finish time");
  const payload = reviewText(row.payload, "Stored action-review payload", MAX_RECORD_BYTES), digest = sha(row.sha256, "Stored action-review hash");
  if (externalAdviceSha256(payload) !== digest || externalAdviceCanonical(JSON.parse(payload)) !== payload) throw new Error("Stored action-review payload checksum failed.");
  const receipt = validateExternalAdviceActionReview(JSON.parse(payload));
  if (receipt.brainId !== sourceOwner || receipt.reviewId !== reviewId || receipt.finishedAt !== finishedAt) throw new Error("Stored action-review index differs from its receipt.");
  const expectedId = currentOwner === sourceOwner ? reviewId : externalAdviceInheritedReviewId(sourceOwner, reviewId, digest, currentOwner);
  if (id !== expectedId) throw new Error("Stored action-review owner/ancestry identity is invalid.");
  return { receipt, payload, sha256: digest };
}

function validatedContext(value: ExternalAdviceSelectedContext, owner: string, selectedChannel: ExternalAdviceChannel): ExternalAdviceSelectedContext {
  const row = object(value, "Selected external-advice context");
  known(row, ["format", "version", "brainId", "channel", "selection", "items", "includedCharacters", "omittedItems", "truncated", "redactedCredentials"], "Selected external-advice context");
  if (value.format !== "omni-external-advice-selected-context" || value.version !== 1 || value.brainId !== owner || value.channel !== selectedChannel ||
      !Array.isArray(value.items) || !Number.isSafeInteger(value.omittedItems) || value.omittedItems < 0 || typeof value.truncated !== "boolean") {
    throw new Error("The selected external-advice context is invalid or belongs to another request.");
  }
  const selection = contextSelection(value.selection), ids = new Set<string>(); let includedCharacters = 0;
  const items: ExternalAdviceContextItem[] = value.items.map((item) => {
    const itemRow = object(item, "External-advice context item");
    known(itemRow, ["id", "kind", "label", "text", "contentSha256", "sourceContentSha256", "createdAt"], "External-advice context item");
    if (!EXTERNAL_ADVICE_REQUEST_ID.test(item.id) || ids.has(item.id) || !["task", "activity"].includes(item.kind) ||
        item.kind === "task" && !selection.includeTask || item.kind === "activity" && !selection.includeActivity) {
      throw new Error("The selected external-advice context contains an invalid or unrequested item.");
    }
    ids.add(item.id); const text = safeText(item.text, "External-advice context item"), label = safeText(item.label, "External-advice context label", 4_096);
    if (externalAdviceSha256(text) !== item.contentSha256) throw new Error("External-advice context content differs from its declared hash.");
    if (item.createdAt !== undefined) safeTime(item.createdAt, "External-advice context timestamp");
    includedCharacters += text.length;
    if (item.sourceContentSha256 !== undefined) sha(item.sourceContentSha256, "External-advice original context hash");
    return { id: item.id, kind: item.kind, label, text, contentSha256: item.contentSha256,
      ...(item.sourceContentSha256 ? { sourceContentSha256: item.sourceContentSha256 } : {}), ...(item.createdAt ? { createdAt: item.createdAt } : {}) };
  });
  if (includedCharacters !== value.includedCharacters || includedCharacters > selection.maxCharacters || value.truncated && value.omittedItems < 1) {
    throw new Error("External-advice context coverage does not match its visible budget report.");
  }
  if (value.redactedCredentials !== undefined && (!Number.isSafeInteger(value.redactedCredentials) || value.redactedCredentials < 0)) {
    throw new Error("External-advice redaction count is invalid.");
  }
  return { format: value.format, version: 1, brainId: owner, channel: selectedChannel, selection, items,
    includedCharacters, omittedItems: value.omittedItems, truncated: value.truncated,
    ...(value.redactedCredentials !== undefined ? { redactedCredentials: value.redactedCredentials } : {}) };
}

export function validateExternalAdviceRequest(value: unknown, owner: string, inheritedSource?: unknown): ExternalAdviceRequest {
  const raw = object(value, "External-advice history row");
  known(raw, ["format", "version", "id", "brainId", "channel", "provider", "model", "endpoint", "state", "prompt", "requestText", "context",
    "response", "responseCredentialRedactions", "error", "createdAt", "updatedAt", "settingsRevision", "authorityRevision", "authorityOwnerEpoch", "contextSha256", "previewSha256",
    "requestSha256", "maxOutputTokens", "providerRequestId", "finishReason", "advisoryOnly", "automaticLearning", "hostActionAuthority",
    "credentialsStoredInBrain", "inheritedFrom"], "External-advice history row");
  const row = value as ExternalAdviceRequest;
  if (row.format !== "omni-external-advice-request" || row.version !== 1 || row.brainId !== owner || !["queued", "streaming", "complete", "cancelled", "failed"].includes(row.state)) {
    throw new Error("External-advice history row has another format, owner, or state.");
  }
  externalAdviceRequestId(row.id); channel(row.channel); provider(row.provider);
  let contextOwner = owner;
  if (row.inheritedFrom) {
    known(object(row.inheritedFrom, "Inherited external-advice ancestry"), ["brainId", "requestId", "sha256"], "Inherited external-advice ancestry");
    brainId(row.inheritedFrom.brainId); externalAdviceRequestId(row.inheritedFrom.requestId); sha(row.inheritedFrom.sha256, "Inherited external-advice receipt hash");
    contextOwner = row.inheritedFrom.brainId;
  }
  if (!MODEL.test(row.model)) throw new Error("External-advice history model is invalid.");
  if (row.provider === "ollama") { ollamaLoopbackEndpoint(row.endpoint); ollamaLocalModelId(row.model); }
  else if (row.provider === "openai-compatible-local") externalAdviceLoopbackEndpoint(row.endpoint); else if (row.endpoint !== undefined) throw new Error("External-advice history has an unexpected remote endpoint.");
  safeText(row.prompt, "External-advice history prompt", 1024 * 1024); safeText(row.requestText, "External-advice history request");
  safeText(row.response, "External-advice history response", MAX_TEXT_BYTES, true);
  if (row.responseCredentialRedactions !== undefined && (!Number.isSafeInteger(row.responseCredentialRedactions) || row.responseCredentialRedactions < 0)) {
    throw new Error("External-advice response redaction count is invalid.");
  }
  validatedContext(row.context, contextOwner, row.channel); safeTime(row.createdAt, "External-advice history creation time"); safeTime(row.updatedAt, "External-advice history update time");
  if (!Number.isSafeInteger(row.settingsRevision) || row.settingsRevision < 1 || !Number.isSafeInteger(row.authorityRevision) || row.authorityRevision < 1 ||
      typeof row.authorityOwnerEpoch !== "string" || !row.authorityOwnerEpoch || row.authorityOwnerEpoch.length > 256 || row.advisoryOnly !== true ||
      row.automaticLearning !== false || row.hostActionAuthority !== false || row.credentialsStoredInBrain !== false) throw new Error("External-advice history authority metadata is invalid.");
  if (!Number.isSafeInteger(row.maxOutputTokens) || row.maxOutputTokens < 1 || row.maxOutputTokens > 65_536) throw new Error("External-advice history output budget is invalid.");
  sha(row.contextSha256, "External-advice context hash"); sha(row.previewSha256, "External-advice preview hash"); sha(row.requestSha256, "External-advice request hash");
  if (externalAdviceSha256(externalAdviceCanonical(row.context)) !== row.contextSha256 ||
      externalAdviceSha256(externalAdviceCanonical(previewIntent(row, contextOwner))) !== row.previewSha256) throw new Error("External-advice history preview or context checksum failed.");
  if (row.error !== undefined) safeText(row.error, "External-advice history error", 4_096);
  if (row.providerRequestId !== undefined) safeText(row.providerRequestId, "External provider request ID", 1_024);
  if (row.finishReason !== undefined) safeText(row.finishReason, "External provider finish reason", 1_024);
  if (!row.inheritedFrom) {
    if (inheritedSource !== undefined) throw new Error("Unexpected inherited external-advice archive was supplied for an original request.");
    if (externalAdviceSha256(externalAdviceCanonical(requestIntent(row))) !== row.requestSha256) throw new Error("External-advice history intent checksum failed.");
  } else {
    if (inheritedSource === undefined) throw new Error("Inherited external-advice history is missing its immutable source archive.");
    const sourceBody = typeof inheritedSource === "string" ? inheritedSource : externalAdviceCanonical(inheritedSource);
    if (externalAdviceSha256(sourceBody) !== row.inheritedFrom.sha256) throw new Error("Inherited external-advice source archive checksum failed.");
    let parsed: unknown; try { parsed = JSON.parse(sourceBody); } catch { throw new Error("Inherited external-advice source archive is invalid JSON."); }
    const source = validateExternalAdviceRequest(parsed, row.inheritedFrom.brainId);
    if (source.inheritedFrom || source.id !== row.inheritedFrom.requestId ||
        row.id !== externalAdviceInheritedRequestId(source.brainId, source.id, row.inheritedFrom.sha256, owner)) {
      throw new Error("Inherited external-advice ancestry is invalid.");
    }
    const expectedState = source.state === "queued" || source.state === "streaming" ? "cancelled" : source.state;
    const normalized = { ...row, id: source.id, brainId: source.brainId, state: source.state } as Record<string, unknown>;
    delete normalized.inheritedFrom;
    if (row.state !== expectedState || externalAdviceCanonical(normalized) !== externalAdviceCanonical(source)) {
      throw new Error("Inherited external-advice presentation rewrote source history beyond owner, identity, dormancy, and ancestry.");
    }
  }
  return JSON.parse(JSON.stringify(row)) as ExternalAdviceRequest;
}

function admitExternalAdviceDatabase(db: DatabaseSync, owner: string, initialize: boolean): void {
  db.exec("PRAGMA trusted_schema=OFF;");
  for (const row of db.prepare("SELECT type,name,sql FROM sqlite_master").iterate()) {
    if (row.type === "index" && /^sqlite_autoindex_[A-Za-z0-9_]+_\d+$/.test(String(row.name)) && row.sql === null) continue;
    const declaration = schema.get(String(row.name));
    if (!declaration || row.type !== (row.name === "requests_updated" ? "index" : "table") || schemaSql(row.sql) !== schemaSql(declaration)) {
      throw new Error("External-advice database contains unsupported executable/schema objects.");
    }
  }
  if (initialize) {
    db.exec(`PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;
      CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS settings(channel TEXT PRIMARY KEY,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,channel TEXT NOT NULL,state TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS requests_updated ON requests(updated_at,id);
      CREATE TABLE IF NOT EXISTS inherited_requests(source_brain_id TEXT NOT NULL,source_request_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_request_id,source_sha256));
      CREATE TABLE IF NOT EXISTS inherited_settings(source_brain_id TEXT NOT NULL,channel TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,channel,source_sha256));
      CREATE TABLE IF NOT EXISTS inherited_meta(source_brain_id TEXT NOT NULL,source_sha256 TEXT NOT NULL,payload TEXT NOT NULL,PRIMARY KEY(source_brain_id,source_sha256));
      CREATE TABLE IF NOT EXISTS reviews(id TEXT PRIMARY KEY,owner_brain_id TEXT NOT NULL,source_brain_id TEXT NOT NULL,review_id TEXT NOT NULL,finished_at TEXT NOT NULL,payload TEXT NOT NULL,sha256 TEXT NOT NULL);`);
    db.prepare("INSERT OR IGNORE INTO meta VALUES('format',?)").run(EXTERNAL_ADVICE_FORMAT);
    db.prepare("INSERT OR IGNORE INTO meta VALUES('brainId',?)").run(owner);
    db.prepare("INSERT OR IGNORE INTO meta VALUES('settingsRevision','1')").run();
    db.prepare("INSERT OR IGNORE INTO meta VALUES('settingsUpdatedAt',?)").run(new Date(0).toISOString());
    for (const selected of CHANNELS) {
      const payload = externalAdviceCanonical(defaultConfiguration());
      db.prepare("INSERT OR IGNORE INTO settings VALUES(?,?,?)").run(selected, payload, externalAdviceSha256(payload));
    }
  }
  const actualObjects = (db.prepare("SELECT name FROM sqlite_master WHERE NOT (type='index' AND name LIKE 'sqlite_autoindex_%') ORDER BY name").all() as Array<{ name: unknown }>)
    .map((row) => String(row.name));
  if (externalAdviceCanonical(actualObjects) !== externalAdviceCanonical([...schema.keys()].sort())) {
    throw new Error("External-advice database schema is incomplete.");
  }
  if (db.prepare("SELECT value FROM meta WHERE key='format'").get()?.value !== EXTERNAL_ADVICE_FORMAT ||
      db.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== owner) throw new Error("External-advice workspace has another format or owner.");
  const metaKeys = (db.prepare("SELECT key FROM meta ORDER BY key").all() as Array<{ key: unknown }>).map((row) => String(row.key));
  if (externalAdviceCanonical(metaKeys) !== externalAdviceCanonical(["brainId", "format", "settingsRevision", "settingsUpdatedAt"])) {
    throw new Error("External-advice workspace metadata contains unsupported fields.");
  }
}

class ExternalAdviceWorkspace {
  private constructor(readonly root: string, readonly brainId: string, readonly db: DatabaseSync) {}
  static async open(root: string, owner: string, readOnly = false): Promise<ExternalAdviceWorkspace> {
    brainId(owner); const directory = join(root, "external-advice"), path = join(root, EXTERNAL_ADVICE_RELATIVE_PATH);
    const rootInfo = await lstat(root); if (!rootInfo.isDirectory() || rootInfo.isSymbolicLink()) throw new Error("External-advice brain workspace must be an owned directory.");
    if (!readOnly) await mkdir(directory, { recursive: true, mode: 0o700 });
    const directoryInfo = await lstat(directory); if (!directoryInfo.isDirectory() || directoryInfo.isSymbolicLink()) throw new Error("External-advice workspace must be a private directory.");
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (file && (!file.isFile() || file.isSymbolicLink())) throw new Error("External-advice workspace database must be a regular file.");
    const db = new DatabaseSync(path, { readOnly, allowExtension: false });
    try {
      if (file) {
        admitExternalAdviceDatabase(db, owner, false);
        if (!readOnly) db.exec("PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL; PRAGMA busy_timeout=2500; PRAGMA cache_size=-256; PRAGMA temp_store=FILE;");
      } else admitExternalAdviceDatabase(db, owner, true);
      return new ExternalAdviceWorkspace(root, owner, db);
    } catch (error) { db.close(); throw error; }
  }
  static async openDatabaseFile(path: string, owner: string): Promise<ExternalAdviceWorkspace> {
    brainId(owner); const file = await lstat(path);
    if (!file.isFile() || file.isSymbolicLink()) throw new Error("External-advice captured database must be a regular file.");
    const db = new DatabaseSync(path, { readOnly: true, allowExtension: false });
    try { admitExternalAdviceDatabase(db, owner, false); return new ExternalAdviceWorkspace(dirname(path), owner, db); }
    catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  transaction<T>(operation: () => T): T { this.db.exec("BEGIN IMMEDIATE"); try { const result = operation(); this.db.exec("COMMIT"); return result; }
    catch (error) { this.db.exec("ROLLBACK"); throw error; } }
  private decoded<T>(payload: unknown, digest: unknown): T {
    if (typeof payload !== "string" || Buffer.byteLength(payload) > MAX_RECORD_BYTES || externalAdviceSha256(payload) !== digest) throw new Error("External-advice workspace checksum or record-size check failed.");
    const parsed = JSON.parse(payload) as T;
    if (externalAdviceCanonical(parsed) !== payload) throw new Error("External-advice workspace payload is not canonical.");
    return parsed;
  }
  settingsRevision(): number { const value = Number(this.db.prepare("SELECT value FROM meta WHERE key='settingsRevision'").get()?.value);
    if (!Number.isSafeInteger(value) || value < 1) throw new Error("External-advice settings revision is invalid."); return value; }
  settingsUpdatedAt(): string { return safeTime(this.db.prepare("SELECT value FROM meta WHERE key='settingsUpdatedAt'").get()?.value, "External-advice settings timestamp"); }
  configuration(selected: ExternalAdviceChannel): ExternalAdviceChannelConfiguration {
    const row = this.db.prepare("SELECT payload,sha256 FROM settings WHERE channel=?").get(channel(selected));
    if (!row) throw new Error("External-advice channel settings are missing."); return configuration(this.decoded(row.payload, row.sha256));
  }
  configure(selected: ExternalAdviceChannel, value: ExternalAdviceChannelConfiguration, expected: number, at: string): number {
    const clean = configuration(value); return this.transaction(() => {
      if (this.settingsRevision() !== expected) throw new Error("External-advice settings changed; review the current revision.");
      const payload = externalAdviceCanonical(clean), next = expected + 1;
      this.db.prepare("UPDATE settings SET payload=?,sha256=? WHERE channel=?").run(payload, externalAdviceSha256(payload), selected);
      this.db.prepare("UPDATE meta SET value=? WHERE key='settingsRevision'").run(String(next));
      this.db.prepare("UPDATE meta SET value=? WHERE key='settingsUpdatedAt'").run(at); return next;
    });
  }
  request(id: string): ExternalAdviceRequest | undefined {
    const row = this.db.prepare("SELECT id,channel,state,created_at,updated_at,payload,sha256 FROM requests WHERE id=?").get(externalAdviceRequestId(id));
    if (!row) return undefined; const value = this.decoded<ExternalAdviceRequest>(row.payload, row.sha256);
    this.assertRequestBinding(row, value);
    return validateExternalAdviceRequest(value, this.brainId, this.inheritedSource(value));
  }
  private assertRequestBinding(row: Record<string, unknown>, value: ExternalAdviceRequest): void {
    if (row.id !== value.id || row.channel !== value.channel || row.state !== value.state || row.created_at !== value.createdAt || row.updated_at !== value.updatedAt) {
      throw new Error("External-advice request index differs from its canonical payload.");
    }
  }
  private inheritedSource(value: ExternalAdviceRequest): string | undefined {
    if (!value.inheritedFrom) return undefined;
    const row = this.db.prepare("SELECT payload FROM inherited_requests WHERE source_brain_id=? AND source_request_id=? AND source_sha256=?")
      .get(value.inheritedFrom.brainId, value.inheritedFrom.requestId, value.inheritedFrom.sha256);
    return row ? String(row.payload) : undefined;
  }
  putRequest(value: ExternalAdviceRequest): void {
    const clean = validateExternalAdviceRequest(value, this.brainId, this.inheritedSource(value)), payload = externalAdviceCanonical(clean), digest = externalAdviceSha256(payload);
    const prior = this.db.prepare("SELECT state,payload FROM requests WHERE id=?").get(clean.id);
    if (prior && TERMINAL.has(String(prior.state)) && prior.payload !== payload) throw new Error("Terminal external-advice history is immutable.");
    this.db.prepare("INSERT INTO requests VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET channel=excluded.channel,state=excluded.state,updated_at=excluded.updated_at,payload=excluded.payload,sha256=excluded.sha256")
      .run(clean.id, clean.channel, clean.state, clean.createdAt, clean.updatedAt, payload, digest);
  }
  putReview(value: ActionReviewerApprovalReview): void {
    const receipt = validateExternalAdviceActionReview(value), payload = externalAdviceCanonical(receipt), digest = externalAdviceSha256(payload);
    const row: ExternalAdviceStoredReviewRow = { id: receipt.reviewId, owner_brain_id: this.brainId, source_brain_id: receipt.brainId,
      review_id: receipt.reviewId, finished_at: receipt.finishedAt, payload, sha256: digest };
    validateExternalAdviceStoredReview(row, this.brainId);
    const prior = this.db.prepare("SELECT payload,sha256 FROM reviews WHERE id=?").get(row.id);
    if (prior) {
      if (prior.payload !== payload || prior.sha256 !== digest) throw new Error("Action-review receipt identity collided with different custody bytes.");
      return;
    }
    this.db.prepare("INSERT INTO reviews VALUES(?,?,?,?,?,?,?)")
      .run(row.id, row.owner_brain_id, row.source_brain_id, row.review_id, row.finished_at, row.payload, row.sha256);
  }
  history(query: ExternalAdviceHistoryQuery = {}): ExternalAdviceHistoryPage {
    const selected = query.channel === undefined ? undefined : channel(query.channel), limit = query.limit ?? 50;
    if (!Number.isSafeInteger(limit) || limit < 1 || limit > 200) throw new Error("External-advice history pages must contain 1 to 200 entries.");
    let cursorTime: string | undefined, cursorId: string | undefined;
    if (query.after !== undefined) {
      cursorId = externalAdviceRequestId(query.after); const cursor = this.db.prepare("SELECT created_at FROM requests WHERE id=?").get(cursorId);
      if (!cursor) throw new Error("External-advice history cursor was not found."); cursorTime = String(cursor.created_at);
    }
    const clauses = ["1=1"], args: Array<string | number> = [];
    if (selected) { clauses.push("channel=?"); args.push(selected); }
    if (cursorTime && cursorId) { clauses.push("(created_at<? OR (created_at=? AND id<?))"); args.push(cursorTime, cursorTime, cursorId); }
    const pageBytes = Number(this.db.prepare(`SELECT COALESCE(SUM(bytes),0) AS bytes FROM (SELECT length(CAST(payload AS BLOB)) AS bytes FROM requests WHERE ${clauses.join(" AND ")} ORDER BY created_at DESC,id DESC LIMIT ?)`)
      .get(...args, limit + 1)?.bytes ?? 0);
    if (!Number.isSafeInteger(pageBytes) || pageBytes > MAX_HISTORY_PAGE_BYTES) throw new Error("External-advice history page exceeds 64 MiB; request a smaller page.");
    const rows = this.db.prepare(`SELECT id,channel,state,created_at,updated_at,payload,sha256 FROM requests WHERE ${clauses.join(" AND ")} ORDER BY created_at DESC,id DESC LIMIT ?`).all(...args, limit + 1);
    const entries = rows.slice(0, limit).map(row => { const value = this.decoded<ExternalAdviceRequest>(row.payload, row.sha256); this.assertRequestBinding(row, value);
      return validateExternalAdviceRequest(value, this.brainId, this.inheritedSource(value)); });
    const totalArgs: Array<string | number> = []; const totalClause = selected ? "WHERE channel=?" : ""; if (selected) totalArgs.push(selected);
    const total = Number(this.db.prepare(`SELECT COUNT(*) AS n FROM requests ${totalClause}`).get(...totalArgs)?.n ?? 0);
    return { entries, total, activeRequestIds: [], ...(rows.length > limit ? { nextAfter: entries.at(-1)!.id } : {}) };
  }
  pendingAfter(after = ""): ExternalAdviceRequest | undefined {
    const cursor = after ? externalAdviceRequestId(after) : "";
    const row = this.db.prepare("SELECT id,channel,state,created_at,updated_at,payload,sha256 FROM requests WHERE id>? AND state IN ('queued','streaming') ORDER BY id LIMIT 1").get(cursor);
    if (!row) return undefined; const value = this.decoded<ExternalAdviceRequest>(row.payload, row.sha256); this.assertRequestBinding(row, value);
    return validateExternalAdviceRequest(value, this.brainId, this.inheritedSource(value));
  }
  inventory(): ExternalAdviceWorkspaceInventory {
    for (const row of this.db.prepare("SELECT source_brain_id,source_request_id,source_sha256,payload FROM inherited_requests").iterate()) {
      brainId(row.source_brain_id); externalAdviceRequestId(row.source_request_id); const digest = sha(row.source_sha256, "Inherited external-advice hash");
      if (externalAdviceSha256(String(row.payload)) !== digest || externalAdviceCanonical(JSON.parse(String(row.payload))) !== String(row.payload)) throw new Error("Inherited external-advice archive checksum failed.");
      validateExternalAdviceRequest(JSON.parse(String(row.payload)), String(row.source_brain_id));
    }
    for (const row of this.db.prepare("SELECT source_brain_id,channel,source_sha256,payload FROM inherited_settings").iterate()) {
      brainId(row.source_brain_id); const selected = channel(row.channel), digest = sha(row.source_sha256, "Inherited external-advice settings hash");
      if (externalAdviceSha256(String(row.payload)) !== digest || externalAdviceCanonical(configuration(JSON.parse(String(row.payload)))) !== String(row.payload)) {
        throw new Error(`Inherited ${selected} external-advice settings checksum failed.`);
      }
    }
    for (const row of this.db.prepare("SELECT source_brain_id,source_sha256,payload FROM inherited_meta").iterate()) {
      const sourceBrainId = brainId(row.source_brain_id), digest = sha(row.source_sha256, "Inherited external-advice metadata hash"), payload = String(row.payload);
      if (externalAdviceSha256(payload) !== digest || externalAdviceCanonical(JSON.parse(payload)) !== payload) throw new Error("Inherited external-advice metadata checksum failed.");
      const value = object(JSON.parse(payload), "Inherited external-advice metadata");
      known(value, ["format", "brainId", "settingsRevision", "settingsUpdatedAt"], "Inherited external-advice metadata");
      if (value.format !== EXTERNAL_ADVICE_FORMAT || value.brainId !== sourceBrainId || !Number.isSafeInteger(value.settingsRevision) || Number(value.settingsRevision) < 1) {
        throw new Error("Inherited external-advice metadata has another owner, format, or revision.");
      }
      safeTime(value.settingsUpdatedAt, "Inherited external-advice settings timestamp");
    }
    for (const row of this.db.prepare("SELECT id,channel,state,created_at,updated_at,payload,sha256 FROM requests").iterate()) { const value = this.decoded<ExternalAdviceRequest>(row.payload, row.sha256); this.assertRequestBinding(row, value);
      validateExternalAdviceRequest(value, this.brainId, this.inheritedSource(value)); }
    for (const row of this.db.prepare("SELECT id,owner_brain_id,source_brain_id,review_id,finished_at,payload,sha256 FROM reviews").iterate()) {
      validateExternalAdviceStoredReview(row, this.brainId);
    }
    return { brainId: this.brainId, settingsRevision: this.settingsRevision(),
      settings: { assistant: this.configuration("assistant"), reviewer: this.configuration("reviewer") },
      requests: Number(this.db.prepare("SELECT COUNT(*) AS n FROM requests").get()?.n ?? 0),
      inheritedRequests: Number(this.db.prepare("SELECT COUNT(*) AS n FROM inherited_requests").get()?.n ?? 0),
      inheritedSettings: Number(this.db.prepare("SELECT COUNT(*) AS n FROM inherited_settings").get()?.n ?? 0),
      inheritedMeta: Number(this.db.prepare("SELECT COUNT(*) AS n FROM inherited_meta").get()?.n ?? 0),
      reviews: Number(this.db.prepare("SELECT COUNT(*) AS n FROM reviews").get()?.n ?? 0) };
  }
}

export async function validateExternalAdviceWorkspace(root: string, owner: string): Promise<ExternalAdviceWorkspaceInventory> {
  const store = await ExternalAdviceWorkspace.open(root, owner, true); try { return store.inventory(); } finally { store.close(); }
}

/** Validates an already-consistent captured SQLite file without copying it. */
export async function validateExternalAdviceDatabaseFile(path: string, owner: string): Promise<ExternalAdviceWorkspaceInventory> {
  const store = await ExternalAdviceWorkspace.openDatabaseFile(path, owner); try { return store.inventory(); } finally { store.close(); }
}

function authority(value: ExternalAdviceDeviceAuthority, owner: string, selectedChannel: ExternalAdviceChannel): ExternalAdviceDeviceAuthority {
  if (!value || value.brainId !== owner || value.channel !== selectedChannel || typeof value.enabled !== "boolean" ||
      typeof value.ownerEpoch !== "string" || !value.ownerEpoch || value.ownerEpoch.length > 256 || !Number.isSafeInteger(value.revision) || value.revision < 1) {
    throw new Error("External-advice device authority is invalid or belongs to another instance.");
  }
  return value;
}

function cleanRemoteError(value: unknown): string {
  return (value instanceof Error ? value.message : String(value)).replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]").replace(/[\r\n\0]+/g, " ").slice(0, 1_000);
}

/** Persistent advisory history plus current, installation-owned provider authority. */
export class ExternalAdviceService extends EventEmitter {
  private readonly active = new Map<string, ActiveAdvice>();
  private readonly initializing = new Set<string>();
  private readonly retirements = new Map<string, Promise<{ requestIds: string[]; error: string } | undefined>>();
  private readonly now: () => Date;
  private readonly uuid: () => string;
  private readonly monotonicNow: () => number;
  private deviceQueue: Promise<void> = Promise.resolve();
  private credentialEpoch = new Map<ExternalAdviceProvider, number>();
  private readonly discoveries = new Map<string, ActiveAdviceDiscovery>();
  private readonly modelDiscovery: ApiTeacherModelDiscoveryService;
  private discoveryClosed = false;

  constructor(
    private readonly client: Pick<ExternalAdviceClient, "stream">,
    private readonly secrets: AdviceSecrets,
    private readonly options: ExternalAdviceServiceOptions
  ) {
    super(); this.now = options.now ?? (() => new Date()); this.uuid = options.uuid ?? randomUUID;
    this.monotonicNow = options.monotonicNow ?? (() => performance.now());
    this.modelDiscovery = new ApiTeacherModelDiscoveryService({ get: async id => {
      if (!id.startsWith("teacher:")) throw new Error("Invalid advice discovery credential alias.");
      const selected = apiTeacherProvider(id.slice("teacher:".length));
      return (await this.credential(selected)).value;
    } }, options.discoveryFetch ?? fetch);
  }

  private activeKey(owner: string, id: string): string { return `${brainId(owner)}:${externalAdviceRequestId(id)}`; }
  private async root(owner: string): Promise<string> { return this.options.brainRoot(brainId(owner)); }
  private persistenceSignal(): AbortSignal { return new AbortController().signal; }
  private sqliteWriteEstimate(payloadBytes: number): number {
    const estimate = payloadBytes * 4 + 256 * 1024;
    if (!Number.isSafeInteger(payloadBytes) || payloadBytes < 0 || !Number.isSafeInteger(estimate)) throw new Error("External-advice write estimate is invalid.");
    return Math.max(MIN_SQLITE_WRITE_ESTIMATE, estimate);
  }
  private async admitWrite(owner: string, payloadBytes: number, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted(); const admit = this.options.admitWrite;
    if (!admit) throw new Error("External-advice disk-write admission is not configured; nothing was persisted or dispatched.");
    await admit(brainId(owner), this.sqliteWriteEstimate(payloadBytes), signal); signal.throwIfAborted();
  }
  private async store(owner: string, signal: AbortSignal = this.persistenceSignal()): Promise<ExternalAdviceWorkspace> {
    owner = brainId(owner); const root = await this.root(owner), path = join(root, EXTERNAL_ADVICE_RELATIVE_PATH);
    const file = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!file) await this.admitWrite(owner, 256 * 1024, signal);
    return ExternalAdviceWorkspace.open(root, owner);
  }
  private timestamp(): string { return this.now().toISOString(); }
  private devicePath(): string { return join(this.options.deviceDirectory, "external-advice-device.json"); }

  private async readDevice(): Promise<DeviceDocument> {
    let parsed: unknown;
    try {
      const info = await lstat(this.devicePath()); if (!info.isFile() || info.isSymbolicLink()) throw new Error("External-advice device credential map must be a private regular file.");
      parsed = JSON.parse(await readFile(this.devicePath(), "utf8"));
    }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return { format: DEVICE_FORMAT, version: 1, revision: 0, credentials: {} }; throw error; }
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("External-advice device credential map is invalid.");
    const row = parsed as Partial<DeviceDocument>;
    known(parsed as Record<string, unknown>, ["format", "version", "revision", "credentials"], "External-advice device credential map");
    if (row.format !== DEVICE_FORMAT || row.version !== 1 || !Number.isSafeInteger(row.revision) || Number(row.revision) < 0 || !row.credentials || typeof row.credentials !== "object" || Array.isArray(row.credentials)) {
      throw new Error("External-advice device credential map is invalid.");
    }
    const credentials: DeviceDocument["credentials"] = {};
    for (const [key, item] of Object.entries(row.credentials)) {
      const selected = provider(key); if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("External-advice device credential map is invalid.");
      const value = item as DeviceCredentialRecord;
      known(item as unknown as Record<string, unknown>, ["id", "revision"], "External-advice device credential record");
      if (typeof value.id !== "string" || !/^external-advice:[A-Za-z0-9-]{36}$/.test(value.id) || !Number.isSafeInteger(value.revision) || value.revision < 1) throw new Error("External-advice device credential map is invalid.");
      credentials[selected] = { id: value.id, revision: value.revision };
    }
    return { format: DEVICE_FORMAT, version: 1, revision: Number(row.revision), credentials };
  }

  private async writeDevice(value: DeviceDocument): Promise<void> {
    await mkdir(this.options.deviceDirectory, { recursive: true, mode: 0o700 });
    const directory = await lstat(this.options.deviceDirectory); if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("External-advice device directory must be private and local.");
    const existing = await lstat(this.devicePath()).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (existing && (!existing.isFile() || existing.isSymbolicLink())) throw new Error("External-advice device credential map must be a private regular file.");
    const temporary = `${this.devicePath()}.${this.uuid()}.tmp`;
    try { await writeFile(temporary, externalAdviceCanonical(value), { encoding: "utf8", mode: 0o600, flag: "wx" }); await rename(temporary, this.devicePath()); await chmod(this.devicePath(), 0o600).catch(() => undefined); }
    finally { await rm(temporary, { force: true }).catch(() => undefined); }
  }

  private deviceMutation<T>(operation: () => Promise<T>): Promise<T> {
    const result = this.deviceQueue.then(operation, operation); this.deviceQueue = result.then(() => undefined, () => undefined); return result;
  }

  private abortProvider(selected: ExternalAdviceProvider, reason: string): void {
    for (const active of this.active.values()) if (active.provider === selected) active.controller.abort(new Error(reason));
    if (selected !== "openai-compatible-local") this.modelDiscovery.credentialChanged(selected);
    for (const active of this.discoveries.values()) if (!active.provider || active.provider === selected) this.cancelDiscovery(active);
  }

  private abortChannel(owner: string, selected: ExternalAdviceChannel, reason: string): void {
    for (const active of this.active.values()) if (active.brainId === owner && active.channel === selected) active.controller.abort(new Error(reason));
    for (const active of this.discoveries.values()) if (active.brainId === owner && active.channel === selected) this.cancelDiscovery(active);
  }

  private abortBrain(owner: string, reason: string): void {
    for (const active of this.active.values()) if (active.brainId === owner) active.controller.abort(new Error(reason));
    for (const active of this.discoveries.values()) if (active.brainId === owner) this.cancelDiscovery(active);
  }

  private async knownCredentials(): Promise<string[]> {
    await this.deviceQueue; const document = await this.readDevice(), values: string[] = [];
    for (const record of Object.values(document.credentials)) {
      if (!record) continue; const value = await this.secrets.get(record.id); if (value) values.push(value);
    }
    return [...new Set(values)].sort((left, right) => right.length - left.length);
  }

  private redact(value: string, credentials: readonly string[]): { text: string; count: number } {
    let text = value, count = 0;
    for (const credential of credentials) {
      if (!credential) continue; const pieces = text.split(credential); if (pieces.length > 1) { count += pieces.length - 1; text = pieces.join("[REDACTED]"); }
    }
    text = text.replace(/\b(?:sk-[A-Za-z0-9_-]{12,}|AIza[A-Za-z0-9_-]{16,})\b/g, () => { count += 1; return "[REDACTED]"; });
    return { text, count };
  }

  private redactReviewJson(value: unknown, credentials: readonly string[]): unknown {
    if (typeof value === "string") return this.redact(value, credentials).text;
    if (Array.isArray(value)) return value.map((item) => this.redactReviewJson(item, credentials));
    if (!value || typeof value !== "object") return value;
    const entries: Array<[string, unknown]> = [], keys = new Set<string>();
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      const redactedKey = this.redact(key, credentials).text;
      if (keys.has(redactedKey)) throw new Error("Credential redaction would collide action-review argument keys.");
      keys.add(redactedKey); entries.push([redactedKey, this.redactReviewJson(item, credentials)]);
    }
    return Object.fromEntries(entries);
  }

  private redactedActionReview(value: ActionReviewerApprovalReview, credentials: readonly string[]): ActionReviewerApprovalReview {
    const receipt = validateExternalAdviceActionReview(value);
    const protectedStrings = [receipt.reviewId, receipt.brainId, receipt.requestId, receipt.configurationRevision, receipt.ownerRevision,
      receipt.reviewerId, receipt.permissionRevision, receipt.proposalSha256, receipt.proposedInvocation.brainId, receipt.proposedInvocation.toolId,
      receipt.proposedInvocation.action, receipt.resolvedInvocation?.brainId, receipt.resolvedInvocation?.toolId, receipt.resolvedInvocation?.action,
      receipt.disposition, receipt.startedAt, receipt.finishedAt, receipt.verdict,
      ...receipt.observedContext.flatMap((item) => [item.id, item.kind]), ...receipt.evidence.map((item) => item.sourceId)]
      .filter((item): item is string => typeof item === "string");
    if (credentials.some((credential) => credential && protectedStrings.some((item) => item.includes(credential)))) {
      throw new Error("An action-review identity field contains a configured credential and cannot be persisted safely.");
    }
    const invocation = (item: ActionReviewInvocation): ActionReviewInvocation => ({ ...item,
      arguments: this.redactReviewJson(item.arguments, credentials) as Record<string, unknown> });
    return validateExternalAdviceActionReview({ ...receipt,
      proposedInvocation: invocation(receipt.proposedInvocation),
      ...(receipt.resolvedInvocation ? { resolvedInvocation: invocation(receipt.resolvedInvocation) } : {}),
      summary: this.redact(receipt.summary, credentials).text,
      observedContext: receipt.observedContext.map((item) => ({ ...item, text: this.redact(item.text, credentials).text })),
      evidence: receipt.evidence.map((item) => ({ ...item, quote: this.redact(item.quote, credentials).text })) });
  }

  async credentialStatus(): Promise<ExternalAdviceCredentialStatus[]> {
    await this.deviceQueue; const document = await this.readDevice();
    return Promise.all(PROVIDERS.map(async (selected) => {
      const record = document.credentials[selected], configured = Boolean(record && await this.secrets.has(record.id));
      return { provider: selected, configured, required: selected !== "openai-compatible-local" && selected !== "ollama",
        persistence: configured ? this.secrets.persistence() : "none", revision: record?.revision ?? this.credentialEpoch.get(selected) ?? 0 };
    }));
  }

  async saveCredential(input: ExternalAdviceCredentialSaveInput): Promise<ExternalAdviceCredentialStatus[]> {
    const selected = provider(input?.provider); safeText(input?.credential, "External-advice credential", 64_000);
    if (selected === "ollama") throw new Error("Local Ollama uses no API key; configure its local model and endpoint instead.");
    await this.deviceMutation(async () => {
      const document = await this.readDevice(), previous = document.credentials[selected], id = `external-advice:${this.uuid()}`;
      await this.secrets.set(id, input.credential);
      try {
        const revision = (previous?.revision ?? 0) + 1;
        document.credentials[selected] = { id, revision }; document.revision += 1; await this.writeDevice(document);
        this.credentialEpoch.set(selected, revision); this.abortProvider(selected, "The provider credential changed during external advice.");
      } catch (error) { await this.secrets.remove(id).catch(() => undefined); throw error; }
      if (previous) await this.secrets.remove(previous.id).catch(() => undefined);
    });
    return this.credentialStatus();
  }

  async removeCredential(value: ExternalAdviceProvider): Promise<ExternalAdviceCredentialStatus[]> {
    const selected = provider(value);
    await this.deviceMutation(async () => {
      const document = await this.readDevice(), previous = document.credentials[selected];
      if (previous) { delete document.credentials[selected]; document.revision += 1; await this.writeDevice(document); await this.secrets.remove(previous.id).catch(() => undefined); }
      const revision = (previous?.revision ?? this.credentialEpoch.get(selected) ?? 0) + 1; this.credentialEpoch.set(selected, revision);
      this.abortProvider(selected, "The provider credential was removed during external advice.");
    });
    return this.credentialStatus();
  }

  private discoveryKey(owner: string, selected: ExternalAdviceChannel, id: string): string { return `${brainId(owner)}:${selected}:${externalAdviceRequestId(id)}`; }
  private cancelDiscovery(active: ActiveAdviceDiscovery): void {
    active.cancelled = true;
    if (active.wire) this.modelDiscovery.cancel(active.wire);
  }
  discoverModels(owner: string, input: ExternalAdviceModelDiscoveryInput): Promise<ExternalAdviceModelDiscoveryResult> {
    if (this.discoveryClosed) return Promise.reject(new Error("Advice model discovery service is closed."));
    owner = brainId(owner); const selected = channel(input?.channel), id = externalAdviceRequestId(input?.requestId);
    known(object(input, "Advice model discovery"), ["channel", "requestId", "expectedSettingsRevision", "expectedCredentialRevision"], "Advice model discovery");
    if (!Number.isSafeInteger(input.expectedSettingsRevision) || input.expectedSettingsRevision < 1 ||
        !Number.isSafeInteger(input.expectedCredentialRevision) || input.expectedCredentialRevision < 0) throw new Error("Advice discovery requires current settings/credential revisions.");
    const key = this.discoveryKey(owner, selected, id), existing = this.discoveries.get(key);
    if (existing) {
      if (existing.settingsRevision !== input.expectedSettingsRevision || existing.credentialRevision !== input.expectedCredentialRevision) throw new Error("Advice discovery request belongs to another settings or credential revision.");
      return existing.promise;
    }
    const active: ActiveAdviceDiscovery = { brainId: owner, channel: selected, requestId: id, settingsRevision: input.expectedSettingsRevision,
      credentialRevision: input.expectedCredentialRevision, cancelled: false, promise: undefined! };
    // Register before any callback/await: a queued Stop cannot be overwritten
    // by late saved-config or credential lookup completion.
    active.promise = Promise.resolve().then(() => this.runModelDiscovery(active)).finally(() => {
      if (this.discoveries.get(key) === active) this.discoveries.delete(key);
    });
    this.discoveries.set(key, active); return active.promise;
  }
  cancelModelDiscovery(owner: string, input: Pick<ExternalAdviceModelDiscoveryInput, "channel" | "requestId">): ExternalAdviceModelDiscoveryCancellation {
    owner = brainId(owner); const selected = channel(input?.channel), id = externalAdviceRequestId(input?.requestId);
    known(object(input, "Advice discovery Stop"), ["channel", "requestId"], "Advice discovery Stop");
    const active = this.discoveries.get(this.discoveryKey(owner, selected, id));
    if (active) this.cancelDiscovery(active);
    return { brainId: owner, channel: selected, requestId: id, cancelled: Boolean(active) };
  }
  private async runModelDiscovery(active: ActiveAdviceDiscovery): Promise<ExternalAdviceModelDiscoveryResult> {
    const current = () => { if (active.cancelled) throw new Error("Advice model discovery was cancelled or retired."); };
    current(); let config: ExternalAdviceChannelConfiguration;
    const store = await this.store(active.brainId);
    try {
      current();
      if (store.settingsRevision() !== active.settingsRevision) throw new Error("Advice discovery settings revision is stale.");
      config = store.configuration(active.channel);
    } finally { store.close(); }
    if (config.provider === "openai-compatible-local") throw new Error("This generic local provider uses manual model IDs; no catalog endpoint is assumed.");
    active.provider = config.provider;
    const credential = await this.credential(config.provider); current();
    if (credential.revision !== active.credentialRevision) throw new Error("Advice discovery credential revision is stale.");
    active.wire = { provider: config.provider, requestId: externalAdviceSha256(this.discoveryKey(active.brainId, active.channel, active.requestId)),
      ...(config.provider === "ollama" ? { endpoint: ollamaLoopbackEndpoint(config.endpoint) } : {}) };
    const result = await this.modelDiscovery.discover(active.wire); current();
    const checked = await this.store(active.brainId);
    try {
      current(); const now = checked.configuration(active.channel);
      if (checked.settingsRevision() !== active.settingsRevision || now.provider !== config.provider || now.endpoint !== config.endpoint) throw new Error("Advice discovery owner/provider/endpoint settings changed before publication.");
    } finally { checked.close(); }
    const fresh = await this.credential(config.provider); current();
    if (fresh.revision !== active.credentialRevision) throw new Error("Advice discovery credential changed before publication.");
    return { ...result, requestId: active.requestId, brainId: active.brainId, channel: active.channel,
      settingsRevision: active.settingsRevision, credentialRevision: active.credentialRevision,
      ...(config.provider === "ollama" ? { endpoint: config.endpoint } : {}) };
  }

  /** Durable, advisory-only ActionReviewer history. It grants nothing and teaches nothing. */
  async recordReview(receipt: ActionReviewerApprovalReview, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted(); const original = validateExternalAdviceActionReview(receipt);
    await this.deviceMutation(async () => {
      signal.throwIfAborted(); const document = await this.readDevice(), credentials: string[] = [];
      for (const item of Object.values(document.credentials)) {
        if (!item) continue; const value = await this.secrets.get(item.id); if (value) credentials.push(value);
      }
      const redacted = this.redactedActionReview(original, [...new Set(credentials)].sort((left, right) => right.length - left.length));
      const payloadBytes = Buffer.byteLength(externalAdviceCanonical(redacted));
      await this.admitWrite(redacted.brainId, payloadBytes, signal);
      signal.throwIfAborted(); const store = await this.store(redacted.brainId, signal);
      try { store.putReview(redacted); } finally { store.close(); }
      signal.throwIfAborted();
    });
  }

  private async authority(owner: string, selected: ExternalAdviceChannel): Promise<ExternalAdviceDeviceAuthority> {
    return authority(await this.options.authority(owner, selected), owner, selected);
  }

  async settings(owner: string): Promise<ExternalAdviceSettings> {
    owner = brainId(owner); const store = await this.store(owner);
    try {
      const [assistantAuthority, reviewerAuthority] = await Promise.all([this.authority(owner, "assistant"), this.authority(owner, "reviewer")]);
      return { format: "omni-external-advice-settings", version: 1, brainId: owner, revision: store.settingsRevision(), updatedAt: store.settingsUpdatedAt(),
        assistant: store.configuration("assistant"), reviewer: store.configuration("reviewer"),
        deviceAuthority: { assistant: assistantAuthority.enabled, reviewer: reviewerAuthority.enabled },
        deviceAuthorityRequired: true, credentialsStoredInBrain: false };
    } finally { store.close(); }
  }

  async configure(owner: string, input: ExternalAdviceConfigureInput): Promise<ExternalAdviceSettings> {
    owner = brainId(owner); const selected = channel(input?.channel), clean = configuration(input?.configuration);
    if (!Number.isSafeInteger(input?.expectedRevision) || input.expectedRevision < 1) throw new Error("External-advice expected revision is invalid.");
    const signal = this.persistenceSignal(); await this.admitWrite(owner, Buffer.byteLength(externalAdviceCanonical(clean)), signal);
    // The portable settings revision covers both channels. Conservatively stop
    // every request for this brain rather than let the other channel wait under
    // a superseded global revision.
    this.abortBrain(owner, "External-advice settings changed during this request.");
    const store = await this.store(owner, signal); try { store.configure(selected, clean, input.expectedRevision, this.timestamp()); } finally { store.close(); }
    return this.settings(owner);
  }

  private async previewWithAuthority(owner: string, input: ExternalAdvicePreviewInput, signal: AbortSignal): Promise<ExternalAdvicePreview> {
    owner = brainId(owner); const selected = channel(input?.channel), rawPrompt = safeText(input?.prompt, "External-advice prompt", 1024 * 1024);
    const store = await this.store(owner); let config: ExternalAdviceChannelConfiguration, settingsRevision: number;
    try { config = store.configuration(selected); settingsRevision = store.settingsRevision(); } finally { store.close(); }
    if (!config.enabled) throw new Error(`External-advice ${selected} is disabled in this brain's portable settings.`);
    const currentAuthority = await this.authority(owner, selected); signal.throwIfAborted();
    if (!currentAuthority.enabled) throw new Error(`External-advice ${selected} is not enabled for this brain on the current device.`);
    const hostContext = validatedContext(await this.options.selectContext({ brainId: owner, channel: selected, selection: config.context }, signal), owner, selected);
    if (externalAdviceCanonical(hostContext.selection) !== externalAdviceCanonical(config.context)) throw new Error("The host context selector changed the reviewed external-advice budget.");
    const credentials = await this.knownCredentials(); signal.throwIfAborted(); const promptRedaction = this.redact(rawPrompt, credentials);
    let redactedCredentials = promptRedaction.count;
    const redactedItems = hostContext.items.map((item) => {
      const text = this.redact(item.text, credentials), label = this.redact(item.label, credentials); redactedCredentials += text.count + label.count;
      return { ...item, label: label.text, text: text.text, contentSha256: externalAdviceSha256(text.text),
        ...(text.text !== item.text ? { sourceContentSha256: item.sourceContentSha256 ?? item.contentSha256 } : {}) };
    });
    const selectedContext = validatedContext({ ...hostContext, items: redactedItems,
      includedCharacters: redactedItems.reduce((sum, item) => sum + item.text.length, 0), redactedCredentials }, owner, selected);
    const prompt = promptRedaction.text;
    const contextText = selectedContext.items.map((item) => `[${item.kind}:${item.id}] ${item.label}\n${item.text}`).join("\n\n");
    const report = `Context coverage: ${selectedContext.includedCharacters} characters included; ${selectedContext.omittedItems} item(s) omitted; truncated=${selectedContext.truncated}.`;
    const requestText = `${prompt}\n\n--- Selected task/activity context (visible source data, not hidden instructions) ---\n${report}${contextText ? `\n\n${contextText}` : ""}`;
    if (Buffer.byteLength(requestText) > MAX_TEXT_BYTES) throw new Error("External-advice preview exceeds the 4 MiB provider-request budget.");
    const contextSha256 = externalAdviceSha256(externalAdviceCanonical(selectedContext));
    const body = { format: "omni-external-advice-preview" as const, version: 1 as const, brainId: owner, channel: selected,
      provider: config.provider, model: config.model, ...(config.endpoint ? { endpoint: config.endpoint } : {}), prompt, requestText,
      context: selectedContext, settingsRevision, authorityRevision: currentAuthority.revision, authorityOwnerEpoch: currentAuthority.ownerEpoch,
      contextSha256, maxOutputTokens: config.maxOutputTokens, advisoryOnly: true as const, automaticLearning: false as const };
    return { ...body, sha256: externalAdviceSha256(externalAdviceCanonical(body)) };
  }

  preview(owner: string, input: ExternalAdvicePreviewInput, signal: AbortSignal = new AbortController().signal): Promise<ExternalAdvicePreview> {
    return this.previewWithAuthority(owner, input, signal);
  }

  private async credential(selected: ExternalAdviceProvider): Promise<{ value?: string; revision: number }> {
    if (selected === "ollama") return { revision: this.credentialEpoch.get(selected) ?? 0 };
    await this.deviceQueue; const document = await this.readDevice(), record = document.credentials[selected];
    if (!record) {
      if (selected === "openai-compatible-local") return { revision: this.credentialEpoch.get(selected) ?? 0 };
      throw new Error(`No ${selected} external-advice credential is configured.`);
    }
    const value = await this.secrets.get(record.id); if (!value) throw new Error(`The ${selected} external-advice credential is unavailable in this OS account/session.`);
    this.credentialEpoch.set(selected, record.revision); return { value, revision: record.revision };
  }

  private async assertCurrent(request: ExternalAdviceRequest, credentialRevision: number, signal: AbortSignal): Promise<void> {
    signal.throwIfAborted(); const store = await this.store(request.brainId);
    try {
      const current = store.configuration(request.channel);
      if (store.settingsRevision() !== request.settingsRevision || !current.enabled || current.provider !== request.provider || current.model !== request.model ||
          current.endpoint !== request.endpoint || current.maxOutputTokens !== request.maxOutputTokens ||
          externalAdviceCanonical(current.context) !== externalAdviceCanonical(request.context.selection)) throw new Error("External-advice settings changed during this request.");
    } finally { store.close(); }
    const currentAuthority = await this.authority(request.brainId, request.channel); signal.throwIfAborted();
    if (!currentAuthority.enabled || currentAuthority.revision !== request.authorityRevision || currentAuthority.ownerEpoch !== request.authorityOwnerEpoch) {
      throw new Error("External-advice device authority changed during this request.");
    }
    if ((this.credentialEpoch.get(request.provider) ?? credentialRevision) !== credentialRevision) throw new Error("External-advice credential changed during this request.");
  }

  private async persistRequest(request: ExternalAdviceRequest, signal: AbortSignal): Promise<void> {
    const payload = externalAdviceCanonical(request);
    await this.admitWrite(request.brainId, Buffer.byteLength(payload), signal);
    const store = await this.store(request.brainId, signal); try { store.putRequest(request); } finally { store.close(); }
  }

  private publish(request: ExternalAdviceRequest, persistedCharacters = request.response.length): void {
    const key = this.activeKey(request.brainId, request.id);
    const event: ExternalAdviceEvent = { brainId: request.brainId, request: JSON.parse(JSON.stringify(request)),
      active: ["queued", "streaming"].includes(request.state) && (this.active.has(key) || this.initializing.has(key)) };
    if (request.state === "streaming" || persistedCharacters !== request.response.length) {
      event.persistence = { receivedCharacters: request.response.length, persistedCharacters };
    }
    this.emit("event", event);
  }

  private async begin(preview: ExternalAdvicePreview, id: string, externalSignal?: AbortSignal): Promise<{ request: ExternalAdviceRequest; promise: Promise<ExternalAdviceRequest> }> {
    id = externalAdviceRequestId(id); const key = this.activeKey(preview.brainId, id);
    if (this.active.has(key) || this.initializing.has(key)) throw new Error("This external-advice request is already active.");
    this.initializing.add(key);
    try {
    const createdAt = this.timestamp();
    const seed = { format: "omni-external-advice-request" as const, version: 1 as const, id, brainId: preview.brainId, channel: preview.channel,
      provider: preview.provider, model: preview.model, ...(preview.endpoint ? { endpoint: preview.endpoint } : {}), state: "queued" as const,
      prompt: preview.prompt, requestText: preview.requestText, context: preview.context, response: "", createdAt, updatedAt: createdAt,
      settingsRevision: preview.settingsRevision, authorityRevision: preview.authorityRevision, authorityOwnerEpoch: preview.authorityOwnerEpoch,
      contextSha256: preview.contextSha256, previewSha256: preview.sha256, advisoryOnly: true as const, automaticLearning: false as const,
      hostActionAuthority: false as const, credentialsStoredInBrain: false as const, maxOutputTokens: preview.maxOutputTokens };
    const request: ExternalAdviceRequest = { ...seed, requestSha256: externalAdviceSha256(externalAdviceCanonical(requestIntent({ ...seed, requestSha256: "0".repeat(64) }))) };
    const persistenceSignal = this.persistenceSignal(), store = await this.store(preview.brainId, persistenceSignal);
    try {
      let previous = store.request(id); if (previous) {
        if (previous.requestSha256 !== request.requestSha256) throw new Error("External-advice request ID was reused with different operands.");
        if (TERMINAL.has(previous.state)) return { request: previous, promise: Promise.resolve(previous) };
        throw new Error("External-advice request is already pending recovery and cannot be replayed.");
      }
      await this.admitWrite(preview.brainId, Buffer.byteLength(externalAdviceCanonical(request)), persistenceSignal);
      previous = store.request(id); if (previous) throw new Error("External-advice request ownership changed during write admission; review current history.");
      store.putRequest(request);
    } finally { store.close(); }
    const controller = new AbortController();
    const onExternalAbort = () => controller.abort(externalSignal?.reason ?? new Error("External-advice request was cancelled."));
    if (externalSignal) { if (externalSignal.aborted) onExternalAbort(); else externalSignal.addEventListener("abort", onExternalAbort, { once: true }); }
    const promise = this.run(request, controller).finally(() => {
      this.active.delete(key); externalSignal?.removeEventListener("abort", onExternalAbort);
    });
    this.active.set(key, { brainId: preview.brainId, channel: preview.channel, provider: preview.provider, controller, promise });
    this.publish(request); return { request: JSON.parse(JSON.stringify(request)), promise };
    } finally { this.initializing.delete(key); }
  }

  private async run(request: ExternalAdviceRequest, controller: AbortController): Promise<ExternalAdviceRequest> {
    let responseCredentials: string[] = [], secretTail = "", receivedResponse = request.response;
    let responseRedactions = request.responseCredentialRedactions ?? 0, lastCheckpoint = this.monotonicNow(), forcedFailure: unknown;
    const projected = (): ExternalAdviceRequest => ({ ...request, response: receivedResponse, updatedAt: this.timestamp(),
      ...(responseRedactions > 0 ? { responseCredentialRedactions: responseRedactions } : {}) });
    const receive = async (delta: string | undefined, flush: boolean, persist: boolean, signal: AbortSignal): Promise<void> => {
      if (delta) secretTail += delta;
      if (secretTail) {
        const redacted = this.redact(secretTail, responseCredentials); responseRedactions += redacted.count;
        const retain = flush ? 0 : Math.max(128, ...responseCredentials.map((value) => value.length - 1));
        const boundary = Math.max(0, redacted.text.length - retain), safe = redacted.text.slice(0, boundary);
        secretTail = redacted.text.slice(boundary); receivedResponse += safe;
      }
      const unpersisted = receivedResponse.slice(request.response.length), unpersistedBytes = Buffer.byteLength(unpersisted);
      if (!unpersistedBytes) return;
      const now = this.monotonicNow(), due = flush || unpersistedBytes >= RESPONSE_BATCH_BYTES || now - lastCheckpoint >= RESPONSE_BATCH_MS;
      if (!persist || !due) { this.publish(projected(), request.response.length); return; }
      const candidate = projected(); await this.persistRequest(candidate, signal); Object.assign(request, candidate); lastCheckpoint = now;
      this.publish(request);
    };
    try {
      const credential = await this.credential(request.provider); responseCredentials = await this.knownCredentials();
      await this.assertCurrent(request, credential.revision, controller.signal);
      const streaming: ExternalAdviceRequest = { ...request, state: "streaming", updatedAt: this.timestamp() };
      await this.persistRequest(streaming, controller.signal); Object.assign(request, streaming); lastCheckpoint = this.monotonicNow(); this.publish(request);
      const clientRequest: ExternalAdviceClientRequest = { provider: request.provider, model: request.model, ...(request.endpoint ? { endpoint: request.endpoint } : {}),
        text: request.requestText, maxOutputTokens: request.maxOutputTokens, ...(credential.value ? { credential: credential.value } : {}) };
      const result = await this.client.stream(clientRequest, async (delta) => {
        const arrivedBeforeAbort = !controller.signal.aborted; if (arrivedBeforeAbort) secretTail += delta;
        try { await this.assertCurrent(request, credential.revision, controller.signal); await receive(undefined, false, true, controller.signal); }
        catch (error) { if (!controller.signal.aborted) { forcedFailure = error; controller.abort(error); } throw error; }
      }, controller.signal);
      await this.assertCurrent(request, credential.revision, controller.signal); await receive(undefined, true, false, controller.signal);
      if (receivedResponse !== this.redact(result.text, responseCredentials).text) throw new Error("External-advice streamed text differs from its redacted terminal result.");
      const complete: ExternalAdviceRequest = { ...projected(), state: "complete",
        ...(result.providerRequestId ? { providerRequestId: result.providerRequestId } : {}),
        ...(result.finishReason ? { finishReason: result.finishReason } : {}) };
      await this.persistRequest(complete, this.persistenceSignal()); Object.assign(request, complete); this.publish(request);
    } catch (error) {
      await receive(undefined, true, false, this.persistenceSignal()).catch(() => undefined);
      const credentials = await this.knownCredentials().catch(() => [] as string[]);
      const state = controller.signal.aborted && forcedFailure === undefined ? "cancelled" as const : "failed" as const;
      const detail = this.redact(cleanRemoteError(forcedFailure ?? error), credentials).text;
      const terminal: ExternalAdviceRequest = { ...projected(), state, error: detail };
      try { await this.persistRequest(terminal, this.persistenceSignal()); Object.assign(request, terminal); this.publish(request); }
      catch (storageError) {
        const storageDetail = this.redact(cleanRemoteError(storageError), credentials).text;
        const partial: ExternalAdviceRequest = { ...request, state, updatedAt: this.timestamp(),
          error: `${detail} Portable history retains only the last admitted response prefix: ${storageDetail}`.slice(0, 4_096) };
        try { await this.persistRequest(partial, this.persistenceSignal()); Object.assign(request, partial); this.publish(request); }
        catch { this.publish({ ...terminal, error: `${detail} Portable history could not record the terminal state because storage admission was refused.`.slice(0, 4_096) }, request.response.length); return terminal; }
      }
    }
    return JSON.parse(JSON.stringify(request)) as ExternalAdviceRequest;
  }

  async start(owner: string, input: ExternalAdviceStartInput): Promise<ExternalAdviceRequest> {
    owner = brainId(owner); if (input?.channel !== "assistant") throw new Error("Public external-advice requests are assistant-only.");
    if (!Number.isSafeInteger(input.expectedSettingsRevision) || input.expectedSettingsRevision < 1) throw new Error("External-advice expected settings revision is invalid.");
    const preview = await this.previewWithAuthority(owner, input, new AbortController().signal);
    if (preview.settingsRevision !== input.expectedSettingsRevision || preview.sha256 !== sha(input.previewSha256, "External-advice preview hash")) {
      throw new Error("External-advice preview changed; review the current context and settings.");
    }
    return (await this.begin(preview, input.requestId)).request;
  }

  async completeReview(owner: string, text: string, signal: AbortSignal): Promise<ExternalAdviceReviewReceipt> {
    owner = brainId(owner); const preview = await this.previewWithAuthority(owner, { channel: "reviewer", prompt: safeText(text, "External review proposal", 1024 * 1024) }, signal);
    const started = await this.begin(preview, this.uuid(), signal), request = await started.promise;
    if (request.state !== "complete") throw new Error(request.error ?? `External review ended ${request.state}.`);
    return { request, text: request.response, advisoryOnly: true, hostActionAuthority: false };
  }

  private activeIds(owner: string): string[] {
    return [...this.active.entries()].filter(([, entry]) => entry.brainId === owner).map(([key]) => key.slice(owner.length + 1)).sort();
  }

  private retireOrphanedRequests(owner: string): Promise<{ requestIds: string[]; error: string } | undefined> {
    owner = brainId(owner); const current = this.retirements.get(owner); if (current) return current;
    const operation = this.retireOrphanedRequestsNow(owner).finally(() => { if (this.retirements.get(owner) === operation) this.retirements.delete(owner); });
    this.retirements.set(owner, operation); return operation;
  }

  private async retireOrphanedRequestsNow(owner: string): Promise<{ requestIds: string[]; error: string } | undefined> {
    let after = "";
    for (;;) {
      const store = await this.store(owner); let pending: ExternalAdviceRequest | undefined;
      try { pending = store.pendingAfter(after); } finally { store.close(); }
      if (!pending) return;
      after = pending.id; const key = this.activeKey(owner, pending.id);
      if (this.active.has(key) || this.initializing.has(key)) continue;
      const retired: ExternalAdviceRequest = { ...pending, state: "cancelled", updatedAt: this.timestamp(),
        error: "The previous external-advice process ended before a terminal receipt. Its admitted response prefix is preserved and was not replayed." };
      try { await this.persistRequest(retired, this.persistenceSignal()); this.publish(retired); }
      catch (error) { return { requestIds: [pending.id], error: cleanRemoteError(error) }; }
    }
  }

  async cancel(owner: string, requestId: string): Promise<ExternalAdviceCancellation> {
    const key = this.activeKey(owner, requestId), active = this.active.get(key);
    if (active && active.brainId === owner) {
      active.controller.abort(new Error("External-advice request was cancelled.")); await active.promise;
      return { brainId: owner, requestId, cancelled: true };
    }
    if (this.initializing.has(key)) return { brainId: owner, requestId, cancelled: false };
    const store = await this.store(owner); let pending: ExternalAdviceRequest | undefined;
    try { pending = store.request(requestId); } finally { store.close(); }
    if (!pending || !["queued", "streaming"].includes(pending.state)) return { brainId: owner, requestId, cancelled: false };
    const retired: ExternalAdviceRequest = { ...pending, state: "cancelled", updatedAt: this.timestamp(),
      error: "This prior-session external-advice request was cancelled without replaying its provider call." };
    await this.persistRequest(retired, this.persistenceSignal()); this.publish(retired);
    return { brainId: owner, requestId, cancelled: true };
  }

  async history(owner: string, query: ExternalAdviceHistoryQuery = {}): Promise<ExternalAdviceHistoryPage> {
    owner = brainId(owner); const retirementBlocked = await this.retireOrphanedRequests(owner);
    const store = await this.store(owner); try { const page = store.history(query); return { ...page, activeRequestIds: this.activeIds(owner), ...(retirementBlocked ? { retirementBlocked } : {}) }; }
    finally { store.close(); }
  }

  async suspend(owner: string): Promise<void> {
    owner = brainId(owner); const active = [...this.active.values()].filter((entry) => entry.brainId === owner);
    const discoveries = [...this.discoveries.values()].filter(entry => entry.brainId === owner);
    for (const entry of discoveries) this.cancelDiscovery(entry);
    for (const entry of active) entry.controller.abort(new Error("External advice was suspended with this brain."));
    await Promise.allSettled(active.map((entry) => entry.promise));
    await Promise.allSettled(discoveries.map(entry => entry.promise));
    await this.retireOrphanedRequests(owner);
  }

  async authorityChanged(owner: string, selected: ExternalAdviceChannel): Promise<void> {
    owner = brainId(owner); selected = channel(selected); const active = [...this.active.values()].filter((entry) => entry.brainId === owner && entry.channel === selected);
    this.abortChannel(owner, selected, "External-advice device authority changed.");
    for (const entry of active) entry.controller.abort(new Error("External-advice device authority changed."));
    await Promise.allSettled(active.map((entry) => entry.promise));
  }

  onEvent(listener: (event: ExternalAdviceEvent) => void): () => void {
    this.on("event", listener); return () => { this.off("event", listener); };
  }

  close(): void {
    this.discoveryClosed = true;
    for (const active of this.active.values()) active.controller.abort(new Error("External-advice service closed."));
    for (const active of this.discoveries.values()) this.cancelDiscovery(active);
    this.modelDiscovery.close();
  }
}
