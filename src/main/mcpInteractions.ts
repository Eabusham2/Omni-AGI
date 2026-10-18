import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, readFile, readdir, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { McpElicitationResult, McpInteractionEvent, McpInteractionRecord, McpInteractionResponse, McpRequestOrigin, McpAppPermissionReview } from "../shared/mcpInteractions";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingInput, NativeMcpSamplingResult } from "../shared/nativeMcpSampling";
import { nativeMcpSamplingInputJson, nativeMcpSamplingInputSha256, nativeMcpSamplingMediaSourceSha256 } from "./nativeMcpSampling";
import { McpHostRequestError, mcpElicitation, mcpFormValues, mcpSamplingRequest, mcpSamplingMediaUnchanged } from "./mcpInteractionProtocol";
import type { McpAppDataReview } from "../shared/mcpAppTaskData";
import { mcpAppTaskData } from "./mcpAppTaskData";
import type { McpRootDisclosureReview } from "../shared/mcpHostServices";

interface ReviewedResponse extends McpInteractionResponse { approvedSampling?: ApprovedNativeMcpSamplingInput; }
interface PendingReview {
  record: McpInteractionRecord;
  signal: AbortSignal;
  resolve(response: ReviewedResponse): void;
  reject(error: unknown): void;
  cleanup(): void;
  settling?: Promise<McpInteractionRecord>;
}
export interface McpInteractionOptions {
  /** Private installation state, outside brains/exports/learning journals. */
  directory: string;
  requireAccess(origin: McpRequestOrigin): void;
  admit(bytes: number): Promise<void>;
  openBrowser(url: string): Promise<void>;
  sampleNative?(input: ApprovedNativeMcpSamplingInput, signal: AbortSignal): Promise<NativeMcpSamplingResult>;
}
const UUID = /^[a-f0-9]{8}-(?:[a-f0-9]{4}-){3}[a-f0-9]{12}$/i;
const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const digest = (value: unknown): string => createHash("sha256").update(JSON.stringify(value)).digest("hex");

/** Reviewed RPC requests are not task-choice assumptions or native memories. */
export class McpInteractionService {
  private readonly pending = new Map<string, PendingReview>();
  private readonly listeners = new Set<(event: McpInteractionEvent) => void>();
  private readonly urls = new Map<string, McpInteractionRecord & { kind: "url" }>();
  constructor(private readonly options: McpInteractionOptions) {}

  subscribe(listener: (event: McpInteractionEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  private emit(record: McpInteractionRecord): void {
    for (const listener of this.listeners) { try { listener({ record: structuredClone(record) }); } catch { /* observation is not authority */ } }
  }
  private path(id: string): string { if (!UUID.test(id)) throw new Error("Invalid MCP interaction identity."); return join(this.options.directory, `${id}.json`); }
  private async save(record: McpInteractionRecord): Promise<void> {
    const content = JSON.stringify(record);
    await this.options.admit(Buffer.byteLength(content) * 3);
    await mkdir(this.options.directory, { recursive: true, mode: 0o700 });
    const temporary = join(this.options.directory, `${record.id}.${randomUUID()}.tmp`);
    try { await writeFile(temporary, content, { flag: "wx", mode: 0o600 }); await rename(temporary, this.path(record.id)); }
    finally { await rm(temporary, { force: true }).catch(() => undefined); }
  }
  async initialize(): Promise<void> {
    await mkdir(this.options.directory, { recursive: true, mode: 0o700 });
    for (const name of await readdir(this.options.directory)) {
      if (!name.endsWith(".json") || !UUID.test(name.slice(0, -5))) continue;
      const path = this.path(name.slice(0, -5)); const info = await lstat(path);
      if (!info.isFile() || info.isSymbolicLink()) throw new Error("Invalid private MCP interaction state.");
      await this.options.admit(info.size * 3);
      const record = JSON.parse(await readFile(path, "utf8")) as McpInteractionRecord;
      if (record.format !== "omni-mcp-interaction" || record.version !== 1 || record.id !== name.slice(0, -5)) throw new Error("Invalid private MCP interaction receipt.");
      if (record.status === "pending") {
        record.status = "cancelled"; record.updatedAt = new Date().toISOString();
        record.reason = "The application closed before review. The remote request was not replayed.";
        await this.save(record);
      }
    }
  }
  async list(brainId: string): Promise<McpInteractionRecord[]> {
    if (!SAFE_ID.test(brainId)) throw new Error("Invalid MCP interaction instance.");
    return [...this.pending.values()].filter(entry => entry.record.origin.brainId === brainId).map(entry => structuredClone(entry.record));
  }

  private async review(origin: McpRequestOrigin,
    body: Omit<Extract<McpInteractionRecord, { kind: "form" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status"> |
      Omit<Extract<McpInteractionRecord, { kind: "url" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status"> |
      Omit<Extract<McpInteractionRecord, { kind: "sampling-input" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status"> |
      Omit<Extract<McpInteractionRecord, { kind: "sampling-output" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status"> |
      Omit<Extract<McpInteractionRecord, { kind: "app-permission" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status"> |
      Omit<Extract<McpInteractionRecord, { kind: "app-data" }>, "format" | "version" | "id" | "origin" | "createdAt" | "updatedAt" | "status">,
    signal: AbortSignal): Promise<ReviewedResponse> {
    signal.throwIfAborted(); this.options.requireAccess(origin);
    const timestamp = new Date().toISOString();
    const record = { format: "omni-mcp-interaction", version: 1, id: randomUUID(), origin: structuredClone(origin),
      createdAt: timestamp, updatedAt: timestamp, status: "pending", ...structuredClone(body) } as McpInteractionRecord;
    await this.save(record); signal.throwIfAborted(); this.options.requireAccess(origin);
    let resolve!: PendingReview["resolve"], reject!: PendingReview["reject"];
    const answer = new Promise<ReviewedResponse>((yes, no) => { resolve = yes; reject = no; });
    const abort = (): void => {
      const current = this.pending.get(record.id);
      if (!current) return;
      if (current.settling) {
        void current.settling.finally(() => { if (this.pending.has(record.id)) abort(); }).catch(() => undefined); return;
      }
      void this.respond({ brainId: origin.brainId, id: record.id, action: "cancel" }).catch(reject);
    };
    const pending: PendingReview = { record, signal, resolve, reject, cleanup: () => signal.removeEventListener("abort", abort) };
    this.pending.set(record.id, pending); signal.addEventListener("abort", abort, { once: true });
    this.emit(record); if (signal.aborted) abort();
    return answer;
  }

  async respond(response: McpInteractionResponse): Promise<McpInteractionRecord> {
    if (!response || !UUID.test(response.id) || !SAFE_ID.test(response.brainId) || !["accept", "decline", "cancel"].includes(response.action) ||
      Object.keys(response).some(key => !["brainId", "id", "action", "values", "sampling", "editedText"].includes(key))) throw new Error("Invalid MCP review response.");
    const entry = this.pending.get(response.id);
    if (!entry || entry.record.origin.brainId !== response.brainId) throw new Error("This MCP request is no longer awaiting review for that instance.");
    if (entry.settling) return entry.settling;
    const operation = this.settle(entry, response);
    entry.settling = operation;
    try { return await operation; }
    finally { if (entry.settling === operation) entry.settling = undefined; }
  }

  private async settle(entry: PendingReview, response: McpInteractionResponse): Promise<McpInteractionRecord> {
    const record = entry.record;
    const reviewed: ReviewedResponse = { brainId: response.brainId, id: response.id, action: response.action };
    if (response.action !== "accept" && (response.values !== undefined || response.sampling !== undefined || response.editedText !== undefined)) throw new Error("Decline/cancel never sends form data or generated text.");
    if (response.action === "accept") {
      entry.signal.throwIfAborted(); this.options.requireAccess(record.origin);
      if (record.kind === "form") {
        if (response.sampling !== undefined || response.editedText !== undefined) throw new Error("Invalid form review data.");
        reviewed.values = await mcpFormValues(record.request.requestedSchema, response.values ?? {});
      } else if (record.kind === "sampling-input") {
        if (response.values !== undefined || response.editedText !== undefined) throw new Error("Invalid sampling input review data.");
        const edits = response.sampling ?? {};
        if (Object.keys(edits).some(key => !["messages", "maxTokens", "temperature", "stopSequences"].includes(key))) throw new Error("Sampling review cannot change the native request's owner or add unreviewed fields.");
        const input: NativeMcpSamplingInput = { ...record.request, ...edits };
        nativeMcpSamplingInputJson(input);
        if (!mcpSamplingMediaUnchanged(record.request, input)) throw new Error("Review can edit task text, but cannot replace, remove, or reorder the submitted media. Request a new review for different media.");
        reviewed.approvedSampling = { ...input, inputSha256: nativeMcpSamplingInputSha256(input) };
      } else if (record.kind === "sampling-output") {
        if (response.values !== undefined || response.sampling !== undefined) throw new Error("Invalid sampling output review data.");
        const text = response.editedText ?? record.request.content.text;
        if (typeof text !== "string" || text.includes("\0")) throw new Error("Invalid reviewed sampling text.");
        reviewed.editedText = text;
      } else if (record.kind === "app-permission" || record.kind === "app-data") {
        if (response.values !== undefined || response.sampling !== undefined || response.editedText !== undefined) throw new Error("App authority review cannot submit hidden form or sampling data.");
      } else {
        if (response.values !== undefined || response.sampling !== undefined || response.editedText !== undefined) throw new Error("URL consent never collects credentials or form data.");
        if (!record.navigation) {
          record.navigation = { state: "dispatching", at: new Date().toISOString() };
          await this.save(record); entry.signal.throwIfAborted(); this.options.requireAccess(record.origin);
          try { await this.options.openBrowser(record.request.url); record.navigation.state = "opened"; }
          catch { record.navigation.state = "failed"; }
        }
        if (record.navigation.state !== "opened") { reviewed.action = "cancel"; record.reason = "The browser navigation did not have a confirmed result. It was not repeated."; }
      }
      entry.signal.throwIfAborted(); this.options.requireAccess(record.origin);
    }
    record.status = reviewed.action === "accept" ? "accepted" : reviewed.action === "decline" ? "declined" : "cancelled";
    record.updatedAt = new Date().toISOString();
    if (reviewed.values || reviewed.approvedSampling || reviewed.editedText !== undefined) record.submittedContentSha256 = digest(reviewed.values ?? reviewed.approvedSampling ?? reviewed.editedText);
    if (record.kind === "url" && reviewed.action === "accept") {
      record.outOfBandStatus = "unknown";
      this.urls.set(`${record.origin.brainId}:${record.origin.serverId}:${record.origin.connectionEpoch}:${record.request.elicitationId}`, record);
    }
    await this.save(record);
    this.pending.delete(record.id); entry.cleanup(); this.emit(record);
    if (entry.signal.aborted) entry.reject(entry.signal.reason); else entry.resolve(reviewed);
    return structuredClone(record);
  }

  async elicit(origin: McpRequestOrigin, params: unknown, signal: AbortSignal): Promise<McpElicitationResult> {
    const request = mcpElicitation(params);
    const result = await this.review(origin, { kind: request.mode, request } as Parameters<McpInteractionService["review"]>[1], signal);
    return result.action === "accept" ? { action: "accept", ...(request.mode === "form" ? { content: result.values! } : {}) } : { action: result.action };
  }
  /** Main-owned, one-request authority only; not a server-created grant API. */
  async confirmApp(origin: McpRequestOrigin, request: McpAppPermissionReview, signal: AbortSignal): Promise<boolean> {
    if (!["tool", "network", "link", "roots"].includes(request.category) || typeof request.message !== "string") throw new Error("Invalid main-owned App authority review.");
    if (request.category === "link") mcpElicitation({ mode: "url", message: request.message, url: request.url, elicitationId: "main-app-link" });
    if (request.category === "tool" && (typeof request.toolId !== "string" || !request.arguments || typeof request.arguments !== "object" || Array.isArray(request.arguments)) ||
      request.category === "network" && !request.csp || request.category === "roots" && (!request.arguments || request.arguments.disclosureOnly !== true || request.arguments.filesystemSandbox !== false ||
        !Array.isArray(request.arguments.roots))) throw new Error("The App authority review lacks its exact action or origin declaration.");
    const reviewed = await this.review(origin, { kind: "app-permission", request }, signal);
    return reviewed.action === "accept";
  }
  /** Main-selected paths and current session only; no server-created selection. */
  async confirmRoots(origin: McpRequestOrigin, request: McpRootDisclosureReview, signal: AbortSignal): Promise<boolean> {
    if (!request || request.disclosureOnly !== true || request.filesystemSandbox !== false || !/^[a-f0-9]{64}$/.test(request.selectionSha256) ||
        !Number.isSafeInteger(request.revision) || request.revision < 1 || typeof request.policyRevision !== "string" || !request.policyRevision || !Array.isArray(request.roots) ||
        request.roots.some(root => typeof root?.uri !== "string" || !root.uri.startsWith("file://") || /[\r\n\0]/.test(root.uri))) throw new Error("Invalid main-owned filesystem root disclosure review.");
    return this.confirmApp(origin, { category: "roots", message: "Share these explicitly selected filesystem root names and URIs with this exact connection until it ends or access changes? This is disclosure only: it does not sandbox a local server or grant file/tool access.",
      toolId: "system.files", arguments: structuredClone(request) as unknown as Record<string, unknown> }, signal);
  }
  /** Task data consent is not a grant, hidden prompt, or forged user message. */
  async reviewAppData(origin: McpRequestOrigin, request: McpAppDataReview, signal: AbortSignal): Promise<{ accepted: boolean; reviewId: string }> {
    if (request.view.brainId !== origin.brainId || request.view.serverId !== origin.serverId || request.view.untrusted !== true ||
      !UUID.test(request.view.id) || !/^[a-f0-9]{64}$/.test(request.captureSha256) || request.untrustedTaskData !== true || request.hiddenPromptUsed !== false ||
      typeof request.message !== "string") throw new Error("App task review lost its main-owned source identity.");
    const data = mcpAppTaskData(request.data.method, request.data.params);
    if (request.dataSha256 !== digest(data)) throw new Error("App task review content changed before consent.");
    const reviewed = await this.review(origin, { kind: "app-data", request: { ...request, data } }, signal);
    return { accepted: reviewed.action === "accept", reviewId: reviewed.id };
  }
  async completeElicitation(origin: McpRequestOrigin, elicitationId: string): Promise<void> {
    const key = `${origin.brainId}:${origin.serverId}:${origin.connectionEpoch}:${elicitationId}`;
    const record = this.urls.get(key); if (!record || record.outOfBandStatus === "complete") return;
    record.outOfBandStatus = "complete"; record.updatedAt = new Date().toISOString();
    await this.save(record); this.emit(record); this.urls.delete(key);
  }

  async sample(origin: McpRequestOrigin, params: unknown, signal: AbortSignal): Promise<Record<string, unknown>> {
    if (!this.options.sampleNative) throw new McpHostRequestError(-32601, "Native same-brain sampling is unavailable.");
    const requestId = randomUUID(); const parsed = mcpSamplingRequest(params, origin, requestId);
    const approved = await this.review(origin, { kind: "sampling-input", ...parsed }, signal);
    if (approved.action !== "accept" || !approved.approvedSampling) throw new McpHostRequestError(-1, "The user declined native sampling.");
    signal.throwIfAborted(); this.options.requireAccess(origin);
    const result = await this.options.sampleNative(approved.approvedSampling, signal);
    if (result.brainId !== origin.brainId || result.requestId !== approved.approvedSampling.requestId || result.inputSha256 !== approved.approvedSampling.inputSha256 ||
      result.model !== `OmniCortex/${origin.brainId}` || result.role !== "assistant" || result.content?.type !== "text" || typeof result.content.text !== "string" ||
      result.privateWorkspace !== true || result.parametersMutated !== false) throw new McpHostRequestError(-32603, "Sampling did not return the reviewed same-brain private inference receipt.");
    if (parsed.media?.length && (result.mediaInput?.length !== parsed.media.length || parsed.media.some(expected => !result.mediaInput?.some(receipt =>
      receipt.messageIndex === expected.messageIndex && receipt.blockIndex === expected.blockIndex && receipt.kind === expected.kind && receipt.mimeType === expected.mimeType &&
      receipt.sourceBytes === expected.decodedBytes && receipt.sourceSha256 === (() => { const content = parsed.request.messages[expected.messageIndex]!.content;
        return nativeMcpSamplingMediaSourceSha256((Array.isArray(content) ? content : [content])[expected.blockIndex]!); })() && Number.isSafeInteger(receipt.decodedWindows) && receipt.decodedWindows > 0 &&
      receipt.wholeInputTraversed === true && receipt.nativeSensory === true)))) throw new McpHostRequestError(-32603, "Sampling did not acknowledge every reviewed media input through this brain's native sensory regions.");
    const reviewed = await this.review(origin, { kind: "sampling-output", request: result }, signal);
    if (reviewed.action !== "accept") throw new McpHostRequestError(-1, "The user withheld the sampled response.");
    return { role: result.role, content: { type: "text", text: reviewed.editedText! }, model: result.model, stopReason: result.stopReason,
      _meta: { "omni/review": { inputSha256: result.inputSha256, sameBrain: true, privateWorkspace: true,
        userEdited: reviewed.editedText !== result.content.text, systemPromptOmitted: parsed.systemPromptOmitted !== undefined } } };
  }
  close(): void {
    for (const entry of this.pending.values()) { entry.cleanup(); entry.reject(new Error("MCP interaction host closed; the request was not replayed.")); }
    this.pending.clear(); this.urls.clear(); this.listeners.clear();
  }
}
