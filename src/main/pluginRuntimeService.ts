import { EventEmitter } from "node:events";
import { randomBytes, randomUUID } from "node:crypto";
import type { ValidateFunction } from "ajv";
import type { PluginService } from "./pluginService";
import type { PluginRuntimeActivationRequest, PluginRuntimeApi, PluginRuntimeContribution, PluginRuntimeEvent, PluginRuntimeLaunch,
  PluginRuntimeModuleDefinition, PluginRuntimeMountedContribution, PluginRuntimePendingApproval, PluginRuntimeProcess,
  PluginRuntimeParserStep, PluginRuntimeParserStepResult, PluginRuntimeTrainingParserActivationRequest,
  PluginRuntimeProcessFactory, PluginRuntimePublishedTool, PluginRuntimeResourceLease, PluginRuntimeReviewRequest,
  PluginRuntimeRpcReply, PluginRuntimeRpcRequest, PluginRuntimeSchemaLearning, PluginRuntimeSchemaReceipt, PluginRuntimeSession,
  PluginRuntimeToolDefinition, PluginRuntimeToolExecutor } from "../shared/pluginRuntime";
import type { PluginRuntimeShutdownAck, PluginRuntimeShutdownDiagnostic } from "../shared/pluginRuntime";
import { PLUGIN_RUNTIME_SCHEME } from "../shared/pluginRuntime";
import { PluginRuntimeStore } from "./pluginRuntimeStore";
import { openPluginRuntimeSource } from "./pluginRuntimePortability";
import { compileRuntimeActionInput, runtimeHash, runtimeId, runtimeJson, runtimeKeys, runtimeObject, validateRuntimeContribution, validateRuntimeTool } from "./pluginRuntimeValidation";
import { pluginRuntimeDocument } from "./pluginRuntimeBootstrap";
import { PLUGIN_RUNTIME_PARSER_ABI, PLUGIN_TRAINING_PARSER_CAPABILITY,
  type PluginRuntimeMediaOutputDeclaration, type PluginRuntimeMediaType } from "../shared/plugins";
import { runtimeModuleSha256 } from "./pluginRuntimeValidation";
import { normalizeSourceMediaType } from "../shared/sourceMediaType";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";
import type { ToolInvocation } from "../shared/types";

export interface PluginRuntimeServiceOptions {
  brainRoot(brainId: string): string | Promise<string>;
  tools: PluginRuntimeToolExecutor;
  processFactory: PluginRuntimeProcessFactory;
  admit(input: { brainId: string; pluginId: string; moduleId: string; contentSha256: string; scriptBytes: number }): Promise<PluginRuntimeResourceLease>;
  /** Exact training-parser session admission; production must not substitute a neural-load check. */
  admitTraining?(input: { brainId: string; pluginId: string; moduleId: string; contentSha256: string; scriptBytes: number }): Promise<PluginRuntimeResourceLease>;
  checkDisk?(brainId: string, bytes: number): Promise<void>;
  /** Admission precedes parsed imported data; this is transient transfer RAM. */
  admitRead?(brainId: string, bytes: number): Promise<{ release(): void | Promise<void> }>;
  publishTools(brainId: string, definitions: PluginRuntimePublishedTool[]): Promise<void>;
  retireTools(brainId: string, toolIds: string[]): Promise<void>;
  isBrainLoaded?(brainId: string): boolean;
  learnSchemas?(job: PluginRuntimeSchemaLearning, signal: AbortSignal): Promise<PluginRuntimeSchemaReceipt>;
  moduleReadyTimeoutMs?: number;
  moduleShutdownTimeoutMs?: number;
}
interface Live {
  ownerKey: string;
  session: PluginRuntimeSession; module: PluginRuntimeModuleDefinition; nonce: string; store: PluginRuntimeStore;
  lease: PluginRuntimeResourceLease; controller: AbortController; process?: PluginRuntimeProcess; timer?: ReturnType<typeof setInterval>;
  inflight: Map<string, { hash: string; promise: Promise<PluginRuntimeRpcReply> }>; busy: boolean;
  toolInflight: Map<string, { hash: string; promise: Promise<unknown> }>;
  parserInflight: Map<string, { hash: string; promise: Promise<PluginRuntimeParserStepResult> }>;
  reviewInflight: Map<string, { hash: string; promise: Promise<PluginRuntimeRpcReply> }>;
  hostRequestIds: Set<string>; checking?: boolean;
  contributions: Map<string, PluginRuntimeMountedContribution>; tools: Map<string, PluginRuntimePublishedTool>;
  calls: Map<string, { resolve(value: unknown): void; reject(error: Error): void; hostRequestId: string }>;
  parserCalls: Map<string, { maxOutputBytes: number; resolve(value: PluginRuntimeParserStepResult): void; reject(error: Error): void }>;
  ready: { promise: Promise<void>; resolve(): void; reject(error: Error): void };
  parserReady?: { promise: Promise<void>; resolve(): void; reject(error: Error): void };
  validators: Map<string, Map<string, ValidateFunction<unknown>>>;
  shutdown?: { id: string; nonce: string; acknowledgement?: PluginRuntimeShutdownAck; resolve(ack: PluginRuntimeShutdownAck): void };
}
interface Pending { live: Live; id: string; review: PluginRuntimePendingApproval; token: string; invocation: import("../shared/types").ToolInvocation }
const reply = (id: string, state: PluginRuntimeRpcReply["state"], values: Omit<PluginRuntimeRpcReply, "format" | "version" | "id" | "state"> = {}): PluginRuntimeRpcReply =>
  ({ format: "omni-plugin-runtime-reply", version: 1, id, state, ...values });
const key = (brain: string, plugin: string, module: string) => runtimeJson([brain, plugin, module]);
export function pluginRuntimeHostRequestId(session: PluginRuntimeSession, rpcId: string): string {
  runtimeId(rpcId);
  const identity = session.purpose === "training-parser" && session.parser
    ? ["training-parser", session.brainId, session.pluginId, session.contentSha256, session.moduleId, session.parser.moduleSha256,
      session.parser.formatId, session.parser.transactionKey, session.parser.sourceSha256, rpcId]
    : ["interactive", session.sessionId, rpcId];
  return `plugin-runtime-${runtimeHash(runtimeJson(identity))}`;
}

export interface PluginRuntimeOriginSnapshot {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  moduleId: string;
  sessionId: string;
  catalogRevision: number;
  instanceRevision: number;
  state: PluginRuntimeSession["state"];
  toolId: string;
  action: string;
  argumentsSha256: string;
  capabilityEnabled: boolean;
}

/** Create a live, non-serializable guard for work which can outlive execute(). */
export function createPluginRuntimeOriginAuthority(input: {
  session: PluginRuntimeSession;
  invocation: ToolInvocation;
  signal: AbortSignal;
  observe(): Promise<PluginRuntimeOriginSnapshot>;
}): HostActionOriginAuthority {
  const expected = {
    brainId: input.session.brainId,
    pluginId: input.session.pluginId,
    contentSha256: input.session.contentSha256,
    moduleId: input.session.moduleId,
    sessionId: input.session.sessionId,
    catalogRevision: input.session.catalogRevision,
    instanceRevision: input.session.instanceRevision,
    toolId: input.invocation.toolId,
    action: input.invocation.action,
    argumentsSha256: runtimeHash(runtimeJson(input.invocation.arguments))
  };
  const revision = runtimeHash(runtimeJson({ format: "omni-plugin-runtime-origin-authority", version: 1, ...expected }));
  return { revision, signal: input.signal, assertCurrent: async () => {
    input.signal.throwIfAborted();
    const observed = await input.observe();
    input.signal.throwIfAborted();
    const actual = { brainId: observed.brainId, pluginId: observed.pluginId, contentSha256: observed.contentSha256,
      moduleId: observed.moduleId, sessionId: observed.sessionId, catalogRevision: observed.catalogRevision,
      instanceRevision: observed.instanceRevision, toolId: observed.toolId, action: observed.action,
      argumentsSha256: observed.argumentsSha256 };
    if (runtimeJson(actual) !== runtimeJson(expected) || observed.state !== "active" || observed.capabilityEnabled !== true) {
      throw new Error("Runtime origin authority is stale; review the current package, capability, and session before a new effect.");
    }
    input.signal.throwIfAborted();
  } };
}
export function validatePluginRuntimeParserStepResult(value: unknown, maximum: number, output?: PluginRuntimeMediaOutputDeclaration): PluginRuntimeParserStepResult {
  const row = runtimeObject(value);
  if (!output) {
    runtimeKeys(row, ["state", "outputBase64"]);
    if (!["need-input", "output", "done"].includes(String(row.state))) throw new Error("Runtime parser returned an invalid JSONL state.");
  } else {
    if (!["need-input", "asset-start", "asset-chunk", "asset-end", "done"].includes(String(row.state))) throw new Error("Runtime media parser returned an invalid state.");
  }
  if (row.state === "output") {
    if (typeof row.outputBase64 !== "string" || !row.outputBase64.length || row.outputBase64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(row.outputBase64)) throw new Error("Runtime parser returned invalid output bytes.");
    const bytes = Buffer.from(row.outputBase64, "base64");
    if (bytes.toString("base64") !== row.outputBase64 || bytes.length > maximum) throw new Error("Runtime parser output exceeds its admitted byte page.");
    return { state: "output", outputBase64: row.outputBase64 };
  }
  if (row.state === "asset-start") {
    if (!output) throw new Error("Runtime JSONL parser cannot emit typed media assets.");
    runtimeKeys(row, ["state", "assetId", "kind", "mimeType"]);
    const assetId = runtimeId(row.assetId), mimeType = normalizeSourceMediaType(row.mimeType), kind = String(row.kind);
    if (!mimeType || !output.mediaTypes.includes(mimeType as PluginRuntimeMediaType) || !["image", "audio", "video"].includes(kind) || !mimeType.startsWith(`${kind}/`)) {
      throw new Error("Runtime media parser asset is outside its reviewed kind/MIME declaration.");
    }
    return { state: "asset-start", assetId, kind: kind as "image" | "audio" | "video", mimeType };
  }
  if (row.state === "asset-chunk") {
    runtimeKeys(row, ["state", "assetId", "offset", "bytesBase64"]);
    const assetId = runtimeId(row.assetId), offset = Number(row.offset);
    if (!Number.isSafeInteger(offset) || offset < 0 || typeof row.bytesBase64 !== "string" || !row.bytesBase64.length ||
        row.bytesBase64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(row.bytesBase64)) throw new Error("Runtime media parser returned an invalid asset page.");
    const bytes = Buffer.from(row.bytesBase64, "base64");
    if (bytes.toString("base64") !== row.bytesBase64 || bytes.length > maximum) throw new Error("Runtime media parser asset page exceeds its admitted envelope.");
    return { state: "asset-chunk", assetId, offset, bytesBase64: row.bytesBase64 };
  }
  if (row.state === "asset-end") {
    runtimeKeys(row, ["state", "assetId", "bytes", "sha256"]);
    const assetId = runtimeId(row.assetId), bytes = Number(row.bytes);
    if (!Number.isSafeInteger(bytes) || bytes < 1 || typeof row.sha256 !== "string" || !/^[a-f0-9]{64}$/.test(row.sha256)) {
      throw new Error("Runtime media parser returned an invalid asset closure.");
    }
    return { state: "asset-end", assetId, bytes, sha256: row.sha256 };
  }
  runtimeKeys(row, ["state"]);
  return { state: row.state as "need-input" | "done" };
}

/** General module computation is isolated by the supplied Electron process factory. */
export class PluginRuntimeService extends EventEmitter implements PluginRuntimeApi {
  private readonly live = new Map<string, Live>();
  private readonly owners = new Map<string, Live>();
  private readonly starting = new Map<string, { hash: string; request: PluginRuntimeActivationRequest | PluginRuntimeTrainingParserActivationRequest; promise: Promise<PluginRuntimeSession> }>();
  private readonly selecting = new Set<string>();
  private readonly pending = new Map<string, Pending>();
  private readonly learning = new Map<string, Promise<void>>();
  private readonly stopping = new Map<string, { brainId: string; value: Live; promise: Promise<PluginRuntimeSession> }>();
  private readonly unsubscribe: () => void;
  constructor(private readonly plugins: PluginService, private readonly options: PluginRuntimeServiceOptions) {
    super(); if (!Number.isSafeInteger(options.moduleReadyTimeoutMs ?? 15000) || (options.moduleReadyTimeoutMs ?? 15000) < 1 || (options.moduleReadyTimeoutMs ?? 15000) > 60000) throw new Error("Invalid bounded module-ready timeout.");
    if (!Number.isSafeInteger(options.moduleShutdownTimeoutMs ?? 100) || (options.moduleShutdownTimeoutMs ?? 100) < 1 || (options.moduleShutdownTimeoutMs ?? 100) > 1000) throw new Error("Invalid bounded module-shutdown timeout.");
    this.unsubscribe = plugins.onEvent(event => { if (event.type === "semantic-learning") return;
      for (const value of this.live.values()) if (value.session.pluginId === event.pluginId && (event.type !== "instance" || event.brainId === value.session.brainId))
        void this.stop(value.session.brainId, value.session.sessionId).catch(() => undefined); });
  }
  onEvent(listener: (event: PluginRuntimeEvent) => void): () => void { this.on("runtime", listener); return () => { this.off("runtime", listener); }; }
  private publish(value: Live, kind: PluginRuntimeEvent["kind"]): void { this.emit("runtime", { brainId: value.session.brainId, kind, session: structuredClone(value.session) } satisfies PluginRuntimeEvent); }
  private assertNotStopping(value: Live): void { if (value.session.state === "stopping") throw new Error("Runtime is stopping; no new work is admitted."); }
  private async authority(value: Live): Promise<void> {
    value.controller.signal.throwIfAborted();
    this.assertNotStopping(value);
    if (this.live.get(value.session.sessionId) !== value) throw new Error("Stale runtime owner.");
    const entry = (await this.plugins.list()).find(entry => entry.pluginId === value.session.pluginId), access = await this.plugins.instance(value.session.brainId, value.session.pluginId);
    value.controller.signal.throwIfAborted();
    if (value.session.state === "stopping" || this.live.get(value.session.sessionId) !== value || !entry?.enabled || entry.activeContentSha256 !== value.session.contentSha256 ||
        entry.revision !== value.session.catalogRevision || !access?.enabled || access.contentSha256 !== value.session.contentSha256 || access.revision !== value.session.instanceRevision)
      throw new Error("Runtime package/owner/access changed; explicit activation is required again.");
    await value.lease.check(value.process?.processId); value.controller.signal.throwIfAborted();
    this.assertNotStopping(value);
  }
  async activate(request: PluginRuntimeActivationRequest): Promise<PluginRuntimeSession> {
    runtimeId(request.brainId); runtimeId(request.requestId); const ownerKey = runtimeJson([request.brainId, request.pluginId, request.moduleId, "interactive"]), hash = runtimeHash(runtimeJson(request));
    if (this.selecting.has(key(request.brainId, request.contentSha256, request.moduleId))) throw new Error("Inherited state selection is in progress; activate after it commits.");
    const starting = this.starting.get(ownerKey); if (starting) { if (starting.hash !== hash) throw new Error("Concurrent runtime activation changed reviewed operands."); return structuredClone(await starting.promise); }
    const existing = this.owners.get(ownerKey); if (existing) { if (existing.session.contentSha256 !== request.contentSha256 || existing.session.catalogRevision !== request.expectedCatalogRevision || existing.session.instanceRevision !== request.expectedInstanceRevision) throw new Error("Activation request names another reviewed runtime generation."); await this.authority(existing); return structuredClone(existing.session); }
    const operation = Promise.resolve().then(() => this.activateOwned(request, ownerKey)), record = { hash, request: structuredClone(request), promise: operation }; this.starting.set(ownerKey, record);
    try { return structuredClone(await operation); } finally { if (this.starting.get(ownerKey) === record) this.starting.delete(ownerKey); }
  }
  async activateTrainingParser(request: PluginRuntimeTrainingParserActivationRequest): Promise<{ session: PluginRuntimeSession; maxMessageBytes: number }> {
    runtimeId(request.brainId); runtimeId(request.requestId); runtimeId(request.pluginId); runtimeId(request.moduleId); runtimeId(request.formatId);
    if (!/^[a-f0-9]{64}$/.test(request.contentSha256) || !/^[a-f0-9]{64}$/.test(request.moduleSha256) ||
        !/^[a-f0-9]{64}$/.test(request.transactionKey) || !/^[a-f0-9]{64}$/.test(request.sourceSha256) ||
        !Number.isSafeInteger(request.sourceBytes) || request.sourceBytes < 0 ||
        request.inputMediaType !== undefined && normalizeSourceMediaType(request.inputMediaType) !== request.inputMediaType ||
        request.output !== undefined && (request.output.kind !== "media-assets" || !Array.isArray(request.output.mediaTypes) || !request.output.mediaTypes.length)) {
      throw new Error("Runtime parser activation identity is invalid.");
    }
    const ownerKey = runtimeJson([request.brainId, request.pluginId, request.moduleId, "training-parser", request.transactionKey]);
    const hash = runtimeHash(runtimeJson(request)), starting = this.starting.get(ownerKey);
    if (starting) { if (starting.hash !== hash) throw new Error("Concurrent runtime parser activation changed reviewed operands.");
      const session = await starting.promise, live = this.live.get(session.sessionId); if (!live) throw new Error("Runtime parser activation retired before acknowledgment.");
      return { session: structuredClone(session), maxMessageBytes: live.lease.maxMessageBytes }; }
    const existing = this.owners.get(ownerKey);
    if (existing) {
      const sameOutput = existing.session.parser?.output === undefined && request.output === undefined ||
        existing.session.parser?.output !== undefined && request.output !== undefined &&
          runtimeJson(existing.session.parser.output) === runtimeJson(request.output);
      if (existing.session.parser?.formatId !== request.formatId || existing.session.parser.transactionKey !== request.transactionKey ||
          existing.session.parser.sourceSha256 !== request.sourceSha256 || existing.session.parser.sourceBytes !== request.sourceBytes ||
          existing.session.parser.moduleSha256 !== request.moduleSha256 || existing.session.parser.inputMediaType !== request.inputMediaType ||
          !sameOutput ||
          existing.session.catalogRevision !== request.expectedCatalogRevision || existing.session.instanceRevision !== request.expectedInstanceRevision) {
        throw new Error("Runtime parser activation names another exact source or reviewed generation.");
      }
      await this.authority(existing); return { session: structuredClone(existing.session), maxMessageBytes: existing.lease.maxMessageBytes };
    }
    const operation = Promise.resolve().then(() => this.activateOwned(request, ownerKey)), record = { hash, request: structuredClone(request), promise: operation };
    this.starting.set(ownerKey, record);
    try { const session = await operation, live = this.live.get(session.sessionId); if (!live) throw new Error("Runtime parser activation retired before publication.");
      return { session: structuredClone(session), maxMessageBytes: live.lease.maxMessageBytes }; }
    finally { if (this.starting.get(ownerKey) === record) this.starting.delete(ownerKey); }
  }
  private async activateOwned(request: PluginRuntimeActivationRequest | PluginRuntimeTrainingParserActivationRequest, ownerKey: string): Promise<PluginRuntimeSession> {
    const parserRequest = "formatId" in request ? request : undefined;
    const entry = (await this.plugins.list()).find(value => value.pluginId === request.pluginId), access = await this.plugins.instance(request.brainId, request.pluginId);
    if (!entry?.enabled || entry.activeContentSha256 !== request.contentSha256 || entry.revision !== request.expectedCatalogRevision ||
        !access?.enabled || access.contentSha256 !== request.contentSha256 || access.revision !== request.expectedInstanceRevision) throw new Error("Review the exact enabled runtime package and instance version first.");
    const packageValue = await this.plugins.package(request.pluginId, request.contentSha256), module = packageValue.manifest.runtimeModules?.find(value => value.id === request.moduleId);
    if (!module) throw new Error("This exact package has no reviewed runtime module.");
    if (parserRequest) {
      const format = packageValue.manifest.trainingFormats.find(value => value.id === parserRequest.formatId);
      const capability = access.capabilities.find(value => value.toolId === PLUGIN_TRAINING_PARSER_CAPABILITY);
      if (!format || format.parser === "declaration-only" || format.parser.kind !== "runtime-stream" ||
          format.parser.moduleId !== module.id || parserRequest.moduleSha256 !== runtimeModuleSha256(module) ||
          !capability?.enabled || !capability.actions.includes(format.id)) {
        throw new Error("Runtime parser does not match its exact reviewed format/module capability.");
      }
      const declaredOutput = format.parser.output;
      if (declaredOutput === "jsonl" ? parserRequest.output !== undefined : runtimeJson(parserRequest.output) !== runtimeJson(declaredOutput)) {
        throw new Error("Runtime parser output protocol does not match its exact reviewed format declaration.");
      }
    }
    const admit = parserRequest ? this.options.admitTraining : this.options.admit;
    if (!admit) throw new Error("Runtime parser resource admission is unavailable without a neural load.");
    const lease = await admit({ ...request, scriptBytes: Buffer.byteLength(module.script) });
    if (!Number.isSafeInteger(lease.maxMessageBytes) || lease.maxMessageBytes < 4096) { await lease.release(); throw new Error("Runtime transfer admission is unavailable."); }
    let store: PluginRuntimeStore | undefined, value: Live | undefined;
    try {
      await lease.check(); const root = await this.options.brainRoot(request.brainId);
      store = await PluginRuntimeStore.open(root, request.brainId, { create: true, checkDisk: bytes => this.options.checkDisk?.(request.brainId, bytes) ?? Promise.resolve() });
      if (!store) throw new Error("Runtime custody could not be created.");
      if (!parserRequest && store.conflictCount(request.contentSha256, module.id)) throw new Error("This module has inherited state conflicts. Select exact parent values before activation.");
      const session: PluginRuntimeSession = { format: "omni-plugin-runtime-session", version: 1, brainId: request.brainId, pluginId: request.pluginId,
        contentSha256: request.contentSha256, moduleId: module.id, sessionId: randomUUID(), catalogRevision: entry.revision, instanceRevision: access.revision,
        state: "starting", startedAt: new Date().toISOString(), ...(parserRequest ? { purpose: "training-parser" as const,
          parser: { formatId: parserRequest.formatId, transactionKey: parserRequest.transactionKey, sourceSha256: parserRequest.sourceSha256,
            sourceBytes: parserRequest.sourceBytes, moduleSha256: parserRequest.moduleSha256,
            ...(parserRequest.inputMediaType ? { inputMediaType: parserRequest.inputMediaType } : {}),
            ...(parserRequest.output ? { output: structuredClone(parserRequest.output) } : {}) } } : {}) };
      let resolveReady!: () => void, rejectReady!: (error: Error) => void; const readyPromise = new Promise<void>((resolve, reject) => { resolveReady = resolve; rejectReady = reject; }); void readyPromise.catch(() => undefined);
      let parserReady: Live["parserReady"];
      if (parserRequest) { let resolve!: () => void, reject!: (error: Error) => void; const promise = new Promise<void>((ok, fail) => { resolve = ok; reject = fail; }); void promise.catch(() => undefined); parserReady = { promise, resolve, reject }; }
      value = { ownerKey, session, module, nonce: randomBytes(32).toString("hex"), store, lease, controller: new AbortController(), inflight: new Map(), toolInflight: new Map(), parserInflight: new Map(), reviewInflight: new Map(), hostRequestIds: new Set(), busy: false,
        contributions: new Map(), tools: new Map(), calls: new Map(), parserCalls: new Map(), validators: new Map(), ready: { promise: readyPromise, resolve: resolveReady, reject: rejectReady }, ...(parserReady ? { parserReady } : {}) };
      this.live.set(session.sessionId, value); this.owners.set(ownerKey, value); await store.session(session); await this.authority(value);
      const owned = value;
      // Startup is not an unmonitored gap: the family check works before the
      // factory exposes its exact child PID, and abort closes pending launch.
      value.timer = setInterval(() => { if (owned.checking) return; owned.checking = true;
        void this.authority(owned).catch(error => this.fail(owned, error instanceof Error ? error.message : String(error))).finally(() => { owned.checking = false; }); }, 1000);
      value.timer.unref?.();
      const process = await this.options.processFactory.launch({ session: structuredClone(session), nonce: value.nonce, maxMessageBytes: lease.maxMessageBytes,
        url: `${PLUGIN_RUNTIME_SCHEME}://${session.sessionId}/index.html` }, {
        signal: value.controller.signal,
        receive: (senderId, url, message) => this.receive(senderId, url, message),
        gone: reason => { void this.fail(owned, reason); }
      });
      if (value.controller.signal.aborted || this.live.get(session.sessionId) !== value) { await process.terminate(); throw new Error("Runtime stopped while its private process was starting."); }
      value.process = process;
      await this.authority(value); await lease.check(value.process.processId);
      const readyTimeout = setTimeout(() => { const failure = new Error("Module source did not acknowledge startup before its bounded deadline."); owned.ready.reject(failure); owned.parserReady?.reject(failure); }, this.options.moduleReadyTimeoutMs ?? 15000);
      try { await value.ready.promise; if (value.parserReady) await value.parserReady.promise; } finally { clearTimeout(readyTimeout); }
      await this.authority(value);
      value.session = { ...session, state: "active", processId: value.process.processId }; await store.session(value.session);
      value.process.send({ kind: "activate", id: randomUUID() }); this.publish(value, "session");
      return value.session;
    } catch (error) { if (value) await this.fail(value, error instanceof Error ? error.message : String(error)); else { store?.close(); await lease.release(); } throw error; }
  }
  async document(url: string): Promise<Response> {
    let parsed: URL; try { parsed = new URL(url); } catch { return new Response("Unknown runtime", { status: 404 }); }
    const value = this.live.get(parsed.hostname); if (parsed.protocol !== `${PLUGIN_RUNTIME_SCHEME}:` || parsed.pathname !== "/index.html" || !value || url !== `${PLUGIN_RUNTIME_SCHEME}://${value.session.sessionId}/index.html`) return new Response("Unknown runtime", { status: 404 });
    try { await this.authority(value); return pluginRuntimeDocument({ session: value.session, nonce: value.nonce, maxMessageBytes: value.lease.maxMessageBytes, url }, value.module); }
    catch { return new Response("Runtime retired", { status: 410 }); }
  }
  async receive(senderId: number, frameUrl: string, message: unknown): Promise<PluginRuntimeRpcReply> {
    const row = runtimeObject(message); runtimeKeys(row, ["format", "version", "sessionId", "nonce", "id", "operation", "arguments"]);
    const id = runtimeId(row.id), value = this.live.get(String(row.sessionId));
    if (!value || row.format !== "omni-plugin-runtime-rpc" || row.version !== 1 || row.nonce !== value.nonce) throw new Error("Runtime transport owner is stale or foreign.");
    // A starting renderer can deliver ready before factory.launch resolves.
    // No effect/state/registration is admitted until its actual sender is bound.
    if (!value.process) return reply(id, "backpressure");
    if (value.process.senderId !== senderId || frameUrl !== `${PLUGIN_RUNTIME_SCHEME}://${value.session.sessionId}/index.html`) throw new Error("Runtime request came from a foreign renderer/frame.");
    const input = { ...row, arguments: runtimeObject(row.arguments) } as unknown as PluginRuntimeRpcRequest;
    if (Buffer.byteLength(runtimeJson(input)) > value.lease.maxMessageBytes) throw new Error("Runtime transfer exceeds its physically admitted frame; use state/receipt pages.");
    if (input.operation === "shutdown-ack") return this.acceptShutdownAck(value, input);
    if (value.session.state === "stopping") return reply(id, "failed", { error: "Runtime is stopping; no new work is admitted." });
    const requestHash = runtimeHash(runtimeJson({ operation: input.operation, arguments: input.arguments }));
    const prior = value.inflight.get(id); if (prior) {
      if (prior.hash !== requestHash) throw new Error("Runtime retry changed actual operands."); return prior.promise; }
    if (value.busy && input.operation !== "tool-result") return reply(id, "backpressure");
    const operation = Promise.resolve().then(() => this.rpc(value, input)), record = { hash: requestHash, promise: operation }; value.inflight.set(id, record); if (input.operation !== "tool-result") value.busy = true;
    try { return await operation; } finally { if (value.inflight.get(id) === record) value.inflight.delete(id); if (input.operation !== "tool-result") value.busy = false; }
  }
  private async rpc(value: Live, input: PluginRuntimeRpcRequest): Promise<PluginRuntimeRpcReply> {
    const id = input.id; await this.authority(value);
    const prior = await value.store.beginRpc(value.session.sessionId, id, input.operation, input.arguments, value.lease.maxMessageBytes);
    if (prior?.reply) return this.bounded(value, id, prior.reply);
    if (prior) return reply(id, "uncertain", { error: "This committed request has no terminal acknowledgement; it will not be replayed." });
    let result: PluginRuntimeRpcReply;
    try {
      await this.authority(value);
      const args = input.arguments;
      if (input.operation === "ready") {
        runtimeKeys(args, ["ok", "error"]); if (typeof args.ok !== "boolean") throw new Error("Module startup acknowledgement is malformed.");
        result = args.ok ? reply(id, "complete") : reply(id, "failed", { error: String(args.error ?? "Module source initialization failed.") });
        await value.store.markRpc(value.session.sessionId, id, args.ok ? "complete" : "failed", result); await this.authority(value);
        if (args.ok) value.ready.resolve(); else value.ready.reject(new Error(result.error)); return result;
      }
      else if (input.operation === "parser-ready") {
        runtimeKeys(args, ["abi", "formatId"]);
        if (value.session.purpose !== "training-parser" || !value.parserReady || args.abi !== PLUGIN_RUNTIME_PARSER_ABI || args.formatId !== value.session.parser?.formatId) throw new Error("Parser registration does not match this exact training session.");
        result = reply(id, "complete"); await value.store.markRpc(value.session.sessionId, id, "complete", result); await this.authority(value); value.parserReady.resolve(); return result;
      }
      else if (input.operation === "call") return await this.hostCall(value, id, args);
      else if (input.operation === "state.get") { if (value.session.purpose === "training-parser") throw new Error("Training-parser retries use fresh in-session state and transaction-stable host receipts."); result = reply(id, "complete", { value: value.store.valuePage(value.session.contentSha256, value.session.moduleId, runtimeId(args.key), Math.max(1, value.lease.maxMessageBytes - 1024)) }); }
      else if (input.operation === "state.read") { if (value.session.purpose === "training-parser") throw new Error("Training-parser retries use fresh in-session state and transaction-stable host receipts."); result = reply(id, "complete", { value: value.store.valueBytes(value.session.contentSha256, value.session.moduleId, runtimeId(args.key), Number(args.offset), Math.min(Number(args.length), Math.max(1, Math.floor(value.lease.maxMessageBytes / 2)))) }); }
      else if (input.operation === "state.put") { if (value.session.purpose === "training-parser") throw new Error("Training-parser retries use fresh in-session state and transaction-stable host receipts."); result = reply(id, "complete", { value: { revision: await value.store.put(value.session.contentSha256, value.session.moduleId, runtimeId(args.key), args.value, Number(args.expectedRevision)) } }); }
      else if (input.operation === "register-tool") { if (value.session.purpose === "training-parser") throw new Error("Training-parser sessions cannot publish general tools."); await this.registerTool(value, validateRuntimeTool(args.definition)); result = reply(id, "complete"); }
      else if (input.operation === "ui.put") { if (value.session.purpose === "training-parser") throw new Error("Training-parser sessions cannot mount Studio UI."); const contribution = validateRuntimeContribution(args.contribution, value.module), previous = value.contributions.get(contribution.id);
        const mounted = { ...contribution, brainId: value.session.brainId, pluginId: value.session.pluginId, contentSha256: value.session.contentSha256,
          moduleId: value.session.moduleId, sessionId: value.session.sessionId, revision: (previous?.revision ?? 0) + 1 };
        await value.store.record("ui-contribution", mounted); await this.authority(value); value.contributions.set(contribution.id, mounted); this.publish(value, "contributions"); result = reply(id, "complete"); }
      else if (input.operation === "ui.remove") { if (value.session.purpose === "training-parser") throw new Error("Training-parser sessions cannot mount Studio UI."); const removed = runtimeId(args.id); await value.store.record("ui-removed", { id: removed }); await this.authority(value); value.contributions.delete(removed); this.publish(value, "contributions"); result = reply(id, "complete"); }
      else if (input.operation === "receipt.read") result = reply(id, "complete", { value: value.store.receipt(value.session.sessionId, runtimeId(args.id), Number(args.offset), Math.min(Number(args.length), Math.max(1, Math.floor(value.lease.maxMessageBytes / 2)))) });
      else if (input.operation === "module-error") { await value.store.record("module-callback-error", { phase: runtimeId(args.phase), error: String(args.error), ...(args.eventId === undefined ? {} : { eventId: runtimeId(args.eventId) }) }); result = reply(id, "complete"); }
      else if (input.operation === "tool-result") { const called = value.calls.get(runtimeId(args.id)); if (!called) throw new Error("No matching current host tool invocation.");
        await value.store.record("module-tool-result", { id: args.id, ...(args.error === undefined ? { value: args.value } : { error: String(args.error) }) }); await this.authority(value);
        value.calls.delete(String(args.id)); if (args.error === undefined) called.resolve(args.value); else called.reject(new Error(String(args.error))); result = reply(id, "complete"); }
      else if (input.operation === "parser-result") { runtimeKeys(args, ["id", "value", "error"]); const target = runtimeId(args.id), called = value.parserCalls.get(target);
        if (!called || value.session.purpose !== "training-parser") throw new Error("No matching current runtime parser step.");
        value.parserCalls.delete(target);
        if (args.error !== undefined) { const failure = new Error(String(args.error)); await value.store.markRpc(value.session.sessionId, target, "failed", reply(target, "failed", { error: failure.message })); called.reject(failure); }
        else { const parsed = validatePluginRuntimeParserStepResult(args.value, called.maxOutputBytes, value.session.parser?.output); await value.store.markRpc(value.session.sessionId, target, "complete", reply(target, "complete", { value: parsed })); called.resolve(parsed); }
        result = reply(id, "complete"); }
      else throw new Error("Unknown runtime operation.");
      await value.store.markRpc(value.session.sessionId, id, "complete", result); return this.bounded(value, id, result);
    } catch (error) { if (input.operation === "call" && value.store.rpc(value.session.sessionId, id)?.state === "dispatched")
        return reply(id, "uncertain", { error: "The host attempt has no terminal runtime acknowledgement; its ToolExecutor receipt remains authoritative and the effect will not be repeated." });
      result = reply(id, "failed", { error: error instanceof Error ? error.message : String(error) });
      await value.store.markRpc(value.session.sessionId, id, "failed", result); return result; }
  }
  private bounded(value: Live, id: string, result: PluginRuntimeRpcReply): PluginRuntimeRpcReply {
    const json = runtimeJson(result), bytes = Buffer.byteLength(json);
    return bytes <= value.lease.maxMessageBytes ? result : reply(id, result.state, { receipt: { id, bytes, sha256: runtimeHash(json) } });
  }
  private async hostCall(value: Live, id: string, args: Record<string, unknown>): Promise<PluginRuntimeRpcReply> {
    if (value.session.state !== "active") throw new Error("Module host effects require completed startup; use onActivate.");
    const toolId = String(args.toolId), action = runtimeId(args.action), actualArgs = runtimeObject(args.arguments);
    if (!/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(toolId)) throw new Error("Invalid declared host tool ID.");
    if (toolId.startsWith("extension.") || toolId.startsWith("plugin.")) throw new Error("Recursive extension effects are not declared host capabilities.");
    const access = await this.plugins.instance(value.session.brainId, value.session.pluginId);
    if (!access?.capabilities.some(capability => capability.enabled && capability.toolId === toolId && capability.actions.includes(action))) throw new Error("This runtime host capability is not explicitly enabled.");
    await this.authority(value);
    const hostId = pluginRuntimeHostRequestId(value.session, id);
    await value.store.markRpc(value.session.sessionId, id, "dispatched"); await this.authority(value);
    const invocation = { brainId: value.session.brainId, toolId, action, arguments: actualArgs };
    value.hostRequestIds.add(hostId);
    const originAuthority = this.originAuthority(value, invocation);
    await originAuthority.assertCurrent();
    const result = await this.options.tools.execute(invocation, undefined, hostId, undefined, { actionEventId: hostId, originAuthority });
    const { approvalToken: _privateAuthority, ...historical } = result;
    await value.store.record("actual-host-tool-receipt", JSON.parse(JSON.stringify(historical)));
    if (result.state === "approval-required") {
      if (!result.approvalToken) throw new Error("Host returned an incomplete approval receipt.");
      try { await this.authority(value); } catch (error) { this.options.tools.invalidateApproval(value.session.brainId, result.approvalToken); throw error; }
      const review: PluginRuntimePendingApproval = { brainId: value.session.brainId, sessionId: value.session.sessionId, rpcId: id, reviewId: randomUUID(),
        pluginId: value.session.pluginId, contentSha256: value.session.contentSha256, moduleId: value.session.moduleId, invocation,
        executionId: result.id, ...(result.output === undefined ? {} : { details: result.output }) };
      this.pending.set(`${value.session.sessionId}/${id}`, { live: value, id, review, token: result.approvalToken, invocation });
      const response = reply(id, "approval-required"); await value.store.markRpc(value.session.sessionId, id, "awaiting-review", response); this.publish(value, "approval"); return response;
    }
    const response = reply(id, result.state === "complete" ? "complete" : "failed", { ...(result.output === undefined ? {} : { value: result.output }), ...(result.error ? { error: result.error } : {}) });
    await value.store.markRpc(value.session.sessionId, id, "complete", response); value.hostRequestIds.delete(hostId); return this.bounded(value, id, response);
  }
  async approvals(brainId: string): Promise<PluginRuntimePendingApproval[]> { return [...this.pending.values()].filter(value => value.review.brainId === brainId).map(value => structuredClone(value.review)); }
  async review(input: PluginRuntimeReviewRequest): Promise<PluginRuntimeRpcReply> {
    const value = this.live.get(input.sessionId);
    if (!value || value.session.brainId !== input.brainId) throw new Error("Runtime review owner is stale.");
    const hash = runtimeHash(runtimeJson(input)), prior = value.reviewInflight.get(input.rpcId);
    if (prior) { if (prior.hash !== hash) throw new Error("Concurrent runtime review changed decision/identity."); return prior.promise; }
    const operation = Promise.resolve().then(() => this.reviewOwned(input)), record = { hash, promise: operation };
    value.reviewInflight.set(input.rpcId, record); try { return await operation; } finally { if (value.reviewInflight.get(input.rpcId) === record) value.reviewInflight.delete(input.rpcId); }
  }
  private async reviewOwned(input: PluginRuntimeReviewRequest): Promise<PluginRuntimeRpcReply> {
    const pendingKey = `${input.sessionId}/${input.rpcId}`, pending = this.pending.get(pendingKey);
    if (!pending || pending.review.brainId !== input.brainId || pending.review.reviewId !== input.reviewId || !["approve", "deny"].includes(input.decision)) throw new Error("Review is stale or belongs to another runtime request.");
    const value = pending.live; await this.authority(value); this.pending.delete(pendingKey);
    let response: PluginRuntimeRpcReply;
    if (input.decision === "deny") { this.options.tools.invalidateApproval(input.brainId, pending.token); response = reply(input.rpcId, "failed", { error: "The user denied this exact runtime effect." }); }
    else { await value.store.markRpc(input.sessionId, input.rpcId, "dispatched"); await this.authority(value);
      const hostId = pluginRuntimeHostRequestId(value.session, input.rpcId), originAuthority = this.originAuthority(value, pending.invocation);
      await originAuthority.assertCurrent();
      const result = await this.options.tools.execute({ ...pending.invocation, approvalToken: pending.token }, undefined,
        hostId, undefined, { actionEventId: hostId, originAuthority });
      if (result.state === "approval-required") { if (!result.approvalToken) throw new Error("Renewed runtime review is incomplete.");
        try { await this.authority(value); } catch (error) { this.options.tools.invalidateApproval(input.brainId, result.approvalToken); throw error; }
        const next = { ...pending, token: result.approvalToken, review: { ...pending.review, reviewId: randomUUID(), executionId: result.id } };
        this.pending.set(pendingKey, next); await value.store.markRpc(input.sessionId, input.rpcId, "awaiting-review", reply(input.rpcId, "approval-required")); this.publish(value, "approval"); return reply(input.rpcId, "approval-required"); }
      const { approvalToken: _neverPortable, ...historical } = result; await value.store.record("actual-reviewed-host-tool-receipt", JSON.parse(JSON.stringify(historical)));
      response = reply(input.rpcId, result.state === "complete" ? "complete" : "failed", { ...(result.output === undefined ? {} : { value: result.output }), ...(result.error ? { error: result.error } : {}) }); }
    await value.store.markRpc(input.sessionId, input.rpcId, "complete", response);
    value.hostRequestIds.delete(pluginRuntimeHostRequestId(value.session, input.rpcId));
    if (this.live.get(input.sessionId) === value && !value.controller.signal.aborted) value.process?.send({ kind: "rpc-reply", id: input.rpcId, reply: this.bounded(value, input.rpcId, response) });
    this.publish(value, "approval"); return this.bounded(value, input.rpcId, response);
  }
  private originAuthority(value: Live, invocation: ToolInvocation): HostActionOriginAuthority {
    return createPluginRuntimeOriginAuthority({ session: structuredClone(value.session), invocation: structuredClone(invocation),
      signal: value.controller.signal, observe: async () => {
        await this.authority(value);
        const access = await this.plugins.instance(value.session.brainId, value.session.pluginId);
        value.controller.signal.throwIfAborted();
        const capabilityEnabled = access?.contentSha256 === value.session.contentSha256 && access.revision === value.session.instanceRevision &&
          access.capabilities.some(capability => capability.enabled && capability.toolId === invocation.toolId && capability.actions.includes(invocation.action));
        await this.authority(value);
        value.controller.signal.throwIfAborted();
        return { brainId: value.session.brainId, pluginId: value.session.pluginId, contentSha256: value.session.contentSha256,
          moduleId: value.session.moduleId, sessionId: value.session.sessionId, catalogRevision: value.session.catalogRevision,
          instanceRevision: value.session.instanceRevision, state: value.session.state, toolId: invocation.toolId, action: invocation.action,
          argumentsSha256: runtimeHash(runtimeJson(invocation.arguments)), capabilityEnabled };
      } });
  }
  private async registerTool(value: Live, definition: PluginRuntimeToolDefinition): Promise<void> {
    if (!value.module.dynamicTools) throw new Error("This module did not declare dynamic tool registration.");
    const validators = new Map(definition.actions.map(action => [action.id, compileRuntimeActionInput(action.inputSchema)]));
    const source = { brainId: value.session.brainId, pluginId: value.session.pluginId, moduleId: value.session.moduleId, contentSha256: value.session.contentSha256, definition },
      registrationSha256 = runtimeHash(runtimeJson(source)), toolId = `extension.${registrationSha256}`;
    const text = runtimeJson({ format: "omni-plugin-runtime-public-tool", version: 1, pluginId: source.pluginId, moduleId: source.moduleId,
      contentSha256: source.contentSha256, toolId, definition }), sourceSha256 = runtimeHash(text);
    const published: PluginRuntimePublishedTool = { toolId, definition, sourceSha256,
      schema: { id: toolId, actions: definition.actions.map(action => action.id), ...(definition.actions.length === 1 ? { inputSchema: definition.actions[0]!.inputSchema } : {}), pluginRuntimeSourceSha256: sourceSha256 } };
    await value.store.record("tool-registration", published);
    const prior = [...value.tools.values()].filter(tool => tool.definition.id === definition.id && tool.toolId !== toolId);
    if (prior.length) await this.options.retireTools(value.session.brainId, prior.map(tool => tool.toolId));
    await this.authority(value); try { await this.options.publishTools(value.session.brainId, [published]); await this.authority(value); }
    catch (error) { await this.options.retireTools(value.session.brainId, [toolId]); throw error; }
    for (const tool of prior) { value.tools.delete(tool.toolId); value.validators.delete(tool.toolId); } value.tools.set(toolId, published); value.validators.set(toolId, validators);
    await value.store.queueSchema({ brainId: source.brainId, pluginId: source.pluginId, moduleId: source.moduleId, contentSha256: source.contentSha256, sourceSha256, text }); this.publish(value, "tools");
    void this.flushSchemaLearning(value.session.brainId).catch(() => undefined);
  }
  async flushSchemaLearning(brainId: string): Promise<void> {
    const existing = this.learning.get(brainId); if (existing) return existing;
    const operation = Promise.resolve().then(() => this.learnOwnedSchemas(brainId)); this.learning.set(brainId, operation);
    try { await operation; } finally { if (this.learning.get(brainId) === operation) this.learning.delete(brainId); }
  }
  private async learnOwnedSchemas(brainId: string): Promise<void> {
    if (!this.options.isBrainLoaded?.(brainId) || !this.options.learnSchemas) return;
    for (const value of this.live.values()) if (value.session.brainId === brainId) for (const job of value.store.pendingSchemas()) {
      if (job.brainId !== brainId || job.contentSha256 !== value.session.contentSha256 || job.moduleId !== value.session.moduleId) continue;
      await this.authority(value); const result = await this.options.learnSchemas(job, value.controller.signal);
      await this.authority(value); if (result.sourceSha256 !== job.sourceSha256) throw new Error("Runtime public schema learning returned an unrelated receipt.");
      await value.store.completeSchema(job.sourceSha256, result.receipt);
    }
  }
  hasTool(toolId: string): boolean { return [...this.live.values()].some(value => value.session.state === "active" && value.tools.has(toolId)); }
  describeTool(toolId: string): PluginRuntimePublishedTool | undefined { for (const value of this.live.values()) { const tool = value.tools.get(toolId); if (tool) return structuredClone(tool); } return undefined; }
  async executeTool(brainId: string, toolId: string, action: string, args: Record<string, unknown>, hostRequestId: string, signal: AbortSignal): Promise<unknown> {
    runtimeId(hostRequestId); const value = [...this.live.values()].find(value => value.session.brainId === brainId && value.tools.has(toolId));
    if (!value || value.session.state !== "active" || !value.tools.get(toolId)?.definition.actions.some(item => item.id === action)) throw new Error("Runtime tool is not currently registered for this owner/action.");
    const body = runtimeJson(args); if (Buffer.byteLength(body) > value.lease.maxMessageBytes - 1024) throw new Error("Module tool operands exceed the admitted frame; use managed file/state pages.");
    const actualArgs = JSON.parse(body) as Record<string, unknown>, validate = value.validators.get(toolId)?.get(action); if (!validate || !validate(actualArgs)) throw new Error("Actual runtime tool arguments do not match its exact registered action schema.");
    const hash = runtimeHash(runtimeJson({ toolId, action, arguments: actualArgs })), prior = value.toolInflight.get(hostRequestId);
    if (prior) { if (prior.hash !== hash) throw new Error("Host module invocation ID changed actual operands."); return prior.promise; }
    const operation = Promise.resolve().then(() => this.executeOwnedTool(value, toolId, action, actualArgs, hostRequestId, signal)), record = { hash, promise: operation };
    value.toolInflight.set(hostRequestId, record); try { return await operation; } finally { if (value.toolInflight.get(hostRequestId) === record) value.toolInflight.delete(hostRequestId); }
  }
  async parserStep(brainId: string, sessionId: string, step: PluginRuntimeParserStep, signal: AbortSignal): Promise<PluginRuntimeParserStepResult> {
    runtimeId(step.id); const value = this.live.get(sessionId);
    if (!value || value.session.brainId !== brainId || value.session.purpose !== "training-parser" || value.session.state !== "active") throw new Error("Runtime parser session is not active for this owner.");
    if (!["push", "drain"].includes(step.operation) || !Number.isSafeInteger(step.sequence) || step.sequence < 0 ||
        !Number.isSafeInteger(step.maxOutputBytes) || step.maxOutputBytes < 1 || step.maxOutputBytes > Math.floor(value.lease.maxMessageBytes / 2)) throw new Error("Invalid bounded runtime parser step.");
    if (step.operation === "push") {
      if (typeof step.inputBase64 !== "string" || step.inputBase64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(step.inputBase64) || typeof step.final !== "boolean" ||
          Buffer.from(step.inputBase64, "base64").toString("base64") !== step.inputBase64) throw new Error("Runtime parser push contains invalid source bytes.");
    } else if (step.inputBase64 !== undefined || step.final !== undefined) throw new Error("Runtime parser drain cannot carry another source page.");
    const body = runtimeJson(step); if (Buffer.byteLength(body) > value.lease.maxMessageBytes - 1024) throw new Error("Runtime parser source page exceeds its admitted frame.");
    const hash = runtimeHash(body), prior = value.parserInflight.get(step.id);
    if (prior) { if (prior.hash !== hash) throw new Error("Concurrent runtime parser step changed its bytes."); return prior.promise; }
    const operation = this.executeParserStep(value, step, signal), record = { hash, promise: operation }; value.parserInflight.set(step.id, record);
    try { return await operation; } finally { if (value.parserInflight.get(step.id) === record) value.parserInflight.delete(step.id); }
  }
  private async executeParserStep(value: Live, step: PluginRuntimeParserStep, signal: AbortSignal): Promise<PluginRuntimeParserStepResult> {
    await this.authority(value); signal.throwIfAborted();
    const prior = await value.store.beginRpc(value.session.sessionId, step.id, "invoke-parser-step", step as unknown as Record<string, unknown>, value.lease.maxMessageBytes);
    if (prior?.reply) { if (prior.reply.state !== "complete") throw new Error(prior.reply.error ?? "Retained runtime parser step failed."); return validatePluginRuntimeParserStepResult(prior.reply.value, step.maxOutputBytes, value.session.parser?.output); }
    if (prior) throw new Error("This runtime parser step was already dispatched; its mutable parser state cannot be replayed.");
    await value.store.markRpc(value.session.sessionId, step.id, "dispatched"); await this.authority(value);
    return new Promise<PluginRuntimeParserStepResult>((resolve, reject) => {
      const abort = () => { value.parserCalls.delete(step.id); reject(signal.reason instanceof Error ? signal.reason : new Error("Runtime parser step cancelled.")); };
      signal.addEventListener("abort", abort, { once: true });
      value.parserCalls.set(step.id, { maxOutputBytes: step.maxOutputBytes,
        resolve: result => { signal.removeEventListener("abort", abort); resolve(result); },
        reject: error => { signal.removeEventListener("abort", abort); reject(error); } });
      if (signal.aborted) { abort(); return; }
      value.process?.send({ kind: "parser-step", id: step.id, parserStep: structuredClone(step) });
    });
  }
  private async executeOwnedTool(value: Live, toolId: string, action: string, args: Record<string, unknown>, hostRequestId: string, signal: AbortSignal): Promise<unknown> {
    await this.authority(value); signal.throwIfAborted(); const id = runtimeHash(runtimeJson([value.session.sessionId, hostRequestId]));
    const prior = await value.store.beginRpc(value.session.sessionId, id, "invoke-module-tool", { toolId, action, arguments: args }, value.lease.maxMessageBytes);
    if (prior?.reply) { if (prior.reply.state !== "complete") throw new Error(prior.reply.error ?? "Retained module call failed."); return prior.reply.value; }
    if (prior) throw new Error("This module invocation was already dispatched; no lost-ACK effect replay is allowed.");
    await value.store.markRpc(value.session.sessionId, id, "dispatched"); await this.authority(value);
    return new Promise<unknown>((resolve, reject) => {
      const abort = () => { value.calls.delete(id); reject(new Error("Runtime tool invocation stopped; its completed host receipts remain retained.")); };
      signal.addEventListener("abort", abort, { once: true });
      value.calls.set(id, { hostRequestId, resolve: result => { signal.removeEventListener("abort", abort); void value.store.markRpc(value.session.sessionId, id, "complete", reply(id, "complete", { value: result })).then(() => resolve(result), reject); },
        reject: error => { signal.removeEventListener("abort", abort); void value.store.markRpc(value.session.sessionId, id, "failed", reply(id, "failed", { error: error.message })).then(() => reject(error), reject); } });
      if (signal.aborted) { abort(); return; }
      value.process?.send({ kind: "invoke-tool", id, toolId: value.tools.get(toolId)!.definition.id, action, data: args });
    });
  }
  private async offerEvent(value: Live, event: { topic: string; eventId: string; data?: unknown }): Promise<void> {
    await this.authority(value); const prior = await value.store.beginRpc(value.session.sessionId, event.eventId, "offer-event", { topic: event.topic, ...(event.data === undefined ? {} : { data: event.data }) });
    if (prior) return; await value.store.markRpc(value.session.sessionId, event.eventId, "dispatched"); await this.authority(value);
    value.process?.send({ kind: "event", id: event.eventId, topic: event.topic, ...(event.data === undefined ? {} : { data: event.data }) });
    await value.store.markRpc(value.session.sessionId, event.eventId, "complete", reply(event.eventId, "complete", { value: { offered: true, handled: false } }));
  }
  async dispatchEvent(brainId: string, event: { topic: string; eventId: string; data?: unknown }): Promise<void> {
    runtimeId(event.topic); runtimeId(event.eventId); for (const value of this.live.values()) if (value.session.brainId === brainId && value.session.purpose !== "training-parser" && value.module.events.includes(event.topic)) {
      await this.offerEvent(value, event); }
  }
  async contributions(brainId: string): Promise<PluginRuntimeMountedContribution[]> { const result: PluginRuntimeMountedContribution[] = [];
    for (const value of this.live.values()) if (value.session.brainId === brainId && value.session.purpose !== "training-parser" && value.session.state === "active") { await this.authority(value); for (const item of value.contributions.values()) result.push(structuredClone(item)); } return result; }
  async invokeContribution(brainId: string, sessionId: string, contributionId: string, eventId: string, data?: unknown): Promise<void> {
    const value = this.live.get(sessionId), item = value?.contributions.get(contributionId); if (!value || value.session.brainId !== brainId || !item?.event) throw new Error("Runtime UI callback is stale or has no registered handler.");
    await this.offerEvent(value, { topic: `ui:${item.event}`, eventId: runtimeId(eventId), ...(data === undefined ? {} : { data }) });
  }
  async list(brainId: string, page: { afterRowid?: number; maxBytes?: number } = {}): Promise<import("../shared/pluginRuntime").PluginRuntimeSessionPage> {
    const maximum = page.maxBytes ?? 262144; return this.admittedRead(brainId, maximum * 12 + 65536, async () => {
      const store = await PluginRuntimeStore.open(await this.options.brainRoot(brainId), brainId, { readOnly: true }); if (!store) return { sessions: [], total: 0 };
      try { const result = store.sessionsPage(page.afterRowid, maximum); return { ...result, sessions: result.sessions.map(session => {
        // Revoking a renderer is not the end of its pending cancellation and
        // custody flush. Keep that exact closing session visible, without
        // putting it back in the executable live/session registry.
        const current = this.live.get(session.sessionId) ?? this.stopping.get(session.sessionId)?.value;
        return current?.session.brainId === brainId ? structuredClone(current.session) : { ...session, brainId, state: "dormant" as const };
      }) }; } finally { store.close(); } });
  }
  private async reader(brainId: string, sourceSha256?: string): Promise<PluginRuntimeStore | undefined> { const root = await this.options.brainRoot(runtimeId(brainId));
    return sourceSha256 ? openPluginRuntimeSource(root, brainId, sourceSha256) : PluginRuntimeStore.open(root, brainId, { readOnly: true }); }
  private async admittedRead<T>(brainId: string, bytes: number, read: () => T | Promise<T>): Promise<T> {
    if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Invalid runtime read footprint.");
    if (!this.options.admitRead) throw new Error("Runtime data read awaits the shared transfer RAM envelope.");
    const lease = await this.options.admitRead(brainId, bytes); try { return await read(); } finally { await lease.release(); }
  }
  admitParserRead(brainId: string, bytes: number): Promise<{ release(): void | Promise<void> }> {
    runtimeId(brainId); if (!Number.isSafeInteger(bytes) || bytes < 0 || !this.options.admitRead) throw new Error("Runtime parser cache read awaits the shared transfer RAM envelope.");
    return this.options.admitRead(brainId, bytes);
  }
  async history(brainId: string, page: { sourceSha256?: string; afterSequence?: number; maxBytes?: number } = {}): Promise<import("../shared/pluginRuntime").PluginRuntimeHistoryPage> {
    const maximum = page.maxBytes ?? 262144; return this.admittedRead(brainId, maximum * 12 + 65536, async () => { const store = await this.reader(brainId, page.sourceSha256); if (!store) return { events: [], total: 0 }; try { return store.history(page.afterSequence, maximum); } finally { try { await store.verifyRead(); } finally { store.close(); } } });
  }
  async readHistoryEvent(brainId: string, sequence: number, offset: number, length: number, sourceSha256?: string): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }> {
    return this.admittedRead(brainId, length * 3 + 65536, async () => { const store = await this.reader(brainId, sourceSha256); if (!store) throw new Error("Runtime history is unavailable."); try { return store.historyBytes(sequence, offset, length); } finally { try { await store.verifyRead(); } finally { store.close(); } } });
  }
  async stateSources(brainId: string, page: { afterSha256?: string; limit?: number } = {}): Promise<import("../shared/pluginRuntime").PluginRuntimeStateSourcePage> {
    const store = await this.reader(brainId); if (!store) return { sources: [], total: 0 }; try { return store.sources(page.afterSha256, page.limit); } finally { store.close(); }
  }
  async stateConflicts(brainId: string, page: { afterKey?: string; limit?: number } = {}): Promise<import("../shared/pluginRuntime").PluginRuntimeStateConflictPage> {
    const store = await this.reader(brainId); if (!store) return { conflicts: [], total: 0 }; try { return store.conflicts(page.afterKey, page.limit); } finally { store.close(); }
  }
  async readSourceState(brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, stateKey: string, maximum = 262144): Promise<import("../shared/pluginRuntime").PluginRuntimeStateValue> {
    return this.admittedRead(brainId, maximum * 12 + 65536, async () => { const store = await this.reader(brainId, sourceSha256); if (!store) throw new Error("Inherited runtime state is unavailable."); try { return store.valuePage(pluginHash, runtimeId(moduleId), runtimeId(stateKey), maximum); } finally { try { await store.verifyRead(); } finally { store.close(); } } });
  }
  async readSourceStateBytes(brainId: string, sourceSha256: string, pluginHash: string, moduleId: string, stateKey: string, offset: number, length: number): Promise<{ text: string; bytes: number; sha256: string; nextOffset?: number }> {
    return this.admittedRead(brainId, length * 3 + 65536, async () => { const store = await this.reader(brainId, sourceSha256); if (!store) throw new Error("Inherited runtime state is unavailable."); try { return store.valueBytes(pluginHash, runtimeId(moduleId), runtimeId(stateKey), offset, length); } finally { try { await store.verifyRead(); } finally { store.close(); } } });
  }
  async selectSourceState(request: import("../shared/pluginRuntime").PluginRuntimeStateSelection): Promise<void> {
    // This method is exposed only on the human Studio API, not module RPC/tools.
    const selection = key(request.brainId, request.pluginHash, request.moduleId);
    if (this.selecting.has(selection) || [...this.starting.values()].some(value => value.request.brainId === request.brainId && value.request.contentSha256 === request.pluginHash && value.request.moduleId === request.moduleId) || [...this.live.values()].some(value => value.session.brainId === request.brainId && value.session.contentSha256 === request.pluginHash && value.session.moduleId === request.moduleId)) throw new Error("Stop this module before selecting inherited state.");
    this.selecting.add(selection); try { await this.selectOwnedSourceState(request); } finally { this.selecting.delete(selection); }
  }
  private async selectOwnedSourceState(request: import("../shared/pluginRuntime").PluginRuntimeStateSelection): Promise<void> {
    const source = await this.reader(request.brainId, request.sourceSha256); if (!source) throw new Error("Inherited runtime state is missing.");
    try { const info = source.valueInfo(request.pluginHash, request.moduleId, request.key); if (!info.revision) throw new Error("This source has no matching exact module value.");
      await this.admittedRead(request.brainId, info.bytes * 12 + 65536, async () => { const selected = source.value(request.pluginHash, request.moduleId, request.key, Math.max(1, info.bytes)); await source.verifyRead();
        const store = await PluginRuntimeStore.open(await this.options.brainRoot(request.brainId), request.brainId, { checkDisk: bytes => this.options.checkDisk?.(request.brainId, bytes) ?? Promise.resolve() }); if (!store) throw new Error("Runtime custody is missing.");
        try { await store.selectValue(request.pluginHash, request.moduleId, request.key, selected.value, request.expectedRevision, request.sourceSha256); } finally { store.close(); } });
    } finally { source.close(); }
  }
  async stop(brainId: string, sessionId: string): Promise<PluginRuntimeSession> { const pending = this.stopping.get(sessionId); if (pending) { if (pending.brainId !== brainId) throw new Error("Runtime stop belongs to another owner."); return pending.promise; }
    const value = this.live.get(sessionId);
    if (!value || value.session.brainId !== brainId) throw new Error("Runtime session is not active for this owner.");
    const operation = this.finish(value, "stopped"), record = { brainId, value, promise: operation }; this.stopping.set(sessionId, record);
    try { return await operation; } finally { if (this.stopping.get(sessionId) === record) this.stopping.delete(sessionId); } }
  private async fail(value: Live, error: string): Promise<void> {
    // The existing Stop owns its error. A fire-and-forget renderer-gone callback
    // must not mirror a rejected persistence promise as an unhandled failure.
    if (this.stopping.has(value.session.sessionId)) return;
    if (this.live.get(value.session.sessionId) === value) {
    const operation = this.finish(value, "failed", error), record = { brainId: value.session.brainId, value, promise: operation }; this.stopping.set(value.session.sessionId, record);
    try { await operation; } finally { if (this.stopping.get(value.session.sessionId) === record) this.stopping.delete(value.session.sessionId); } } }
  private acceptShutdownAck(value: Live, input: PluginRuntimeRpcRequest): PluginRuntimeRpcReply {
    value.controller.signal.throwIfAborted(); const pending = value.shutdown, args = input.arguments;
    runtimeKeys(args, ["shutdownNonce", "callbacks", "completed", "failed", "errors"]);
    if (value.session.state !== "stopping" || !pending || input.id !== pending.id || args.shutdownNonce !== pending.nonce) throw new Error("Shutdown acknowledgement does not match the current main-owned challenge.");
    if (![args.callbacks, args.completed, args.failed].every(count => Number.isSafeInteger(count) && Number(count) >= 0) ||
        Number(args.completed) + Number(args.failed) !== args.callbacks || !Array.isArray(args.errors) || args.errors.length !== args.failed) throw new Error("Shutdown callback counts are invalid.");
    const seen = new Set<number>();
    const errors = args.errors.map(value => { const row = runtimeObject(value); runtimeKeys(row, ["callback", "message"]);
      if (!Number.isSafeInteger(row.callback) || Number(row.callback) < 0 || Number(row.callback) >= Number(args.callbacks) || seen.has(Number(row.callback)) || typeof row.message !== "string") throw new Error("Shutdown callback error is malformed.");
      seen.add(Number(row.callback)); return { callback: Number(row.callback), message: row.message }; });
    const acknowledgement = { callbacks: Number(args.callbacks), completed: Number(args.completed), failed: Number(args.failed), errors };
    if (pending.acknowledgement && runtimeJson(pending.acknowledgement) !== runtimeJson(acknowledgement)) throw new Error("Shutdown retry changed its reported cleanup.");
    pending.acknowledgement = acknowledgement; pending.resolve(acknowledgement);
    return reply(input.id, "complete");
  }
  private async requestShutdown(value: Live): Promise<PluginRuntimeShutdownDiagnostic> {
    const id = randomUUID(), deadlineMs = this.options.moduleShutdownTimeoutMs ?? 100, started = performance.now();
    const base = { format: "omni-plugin-runtime-shutdown-diagnostic" as const, version: 1 as const, sessionId: value.session.sessionId, requestId: id, deadlineMs };
    if (!value.process) return { ...base, state: "not-started", elapsedMs: performance.now() - started };
    let timer: ReturnType<typeof setTimeout> | undefined;
    const ack = new Promise<PluginRuntimeShutdownAck | undefined>(resolve => {
      value.shutdown = { id, nonce: randomBytes(32).toString("hex"), resolve };
      timer = setTimeout(() => resolve(undefined), deadlineMs);
    });
    try {
      value.process.send({ kind: "deactivate", id, shutdown: { format: "omni-plugin-runtime-shutdown", version: 1, sessionId: value.session.sessionId, nonce: value.shutdown!.nonce } });
      const acknowledgement = await ack;
      return { ...base, state: acknowledgement ? "acknowledged" : "timeout", elapsedMs: performance.now() - started, ...(acknowledgement ? { acknowledgement } : {}) };
    } catch (failure) { return { ...base, state: "send-failed", elapsedMs: performance.now() - started, error: failure instanceof Error ? failure.message : String(failure) }; }
    finally { if (timer) clearTimeout(timer); value.shutdown = undefined; }
  }
  private async finish(value: Live, state: "stopped" | "failed", error?: string): Promise<PluginRuntimeSession> {
    if (value.controller.signal.aborted) return structuredClone(value.session);
    value.session = { ...value.session, state: "stopping" }; if (value.timer) clearInterval(value.timer);
    value.ready.reject(new Error(error ?? "Runtime stopped before module readiness."));
    let diagnostic: PluginRuntimeShutdownDiagnostic | undefined, failure: unknown, termination: Promise<void> | undefined;
    const cancellations: Array<Promise<number>> = [];
    const retainFailure = (reason: unknown): void => { failure ??= reason; };
    try {
      try { this.publish(value, "session"); } catch (reason) { retainFailure(reason); }
      for (const [id, pending] of this.pending) if (pending.live === value) {
        try { this.options.tools.invalidateApproval(value.session.brainId, pending.token); } catch (reason) { retainFailure(reason); }
        this.pending.delete(id);
      }
      for (const call of value.calls.values()) call.reject(new Error("Runtime session stopped.")); value.calls.clear();
      for (const call of value.parserCalls.values()) call.reject(new Error("Runtime parser session stopped.")); value.parserCalls.clear();
      value.parserReady?.reject(new Error("Runtime parser stopped before registration completed."));
      // Begin cancellation before grace: an already queued host dispatch must
      // not gain a new 100ms opportunity to start after Stop was requested.
      for (const requestId of value.hostRequestIds) {
        try { const cancel = this.options.tools.cancelAndWait(value.session.brainId, requestId); void cancel.catch(() => undefined); cancellations.push(cancel); }
        catch (reason) { retainFailure(reason); }
      }
      try { diagnostic = await this.requestShutdown(value); }
      finally {
        // No persistence/host wait may keep an unresponsive renderer alive.
        value.controller.abort(new Error(error ?? "Runtime stopped."));
        this.live.delete(value.session.sessionId); this.owners.delete(value.ownerKey);
        try { termination = value.process?.terminate(); void termination?.catch(() => undefined); } catch (reason) { retainFailure(reason); }
      }
      for (const result of await Promise.allSettled(cancellations)) if (result.status === "rejected") retainFailure(result.reason);
      await this.options.retireTools(value.session.brainId, [...value.tools.keys()]); value.tools.clear(); value.validators.clear(); value.contributions.clear();
      await Promise.allSettled([...value.inflight.values()].map(record => record.promise));
      await Promise.allSettled([...value.toolInflight.values()].map(record => record.promise));
      await Promise.allSettled([...value.parserInflight.values()].map(record => record.promise));
      await Promise.allSettled([...value.reviewInflight.values()].map(record => record.promise));
      const learning = this.learning.get(value.session.brainId); if (learning) await Promise.allSettled([learning]);
      if (diagnostic) await value.store.record("runtime-shutdown", diagnostic);
    } catch (reason) { retainFailure(reason); }
    finally {
      // Also run if preparing/sending/persisting the diagnostic throws.
      if (!value.controller.signal.aborted) value.controller.abort(new Error(error ?? "Runtime stopped."));
      if (this.live.get(value.session.sessionId) === value) this.live.delete(value.session.sessionId);
      if (this.owners.get(value.ownerKey) === value) this.owners.delete(value.ownerKey);
      try { if (!termination && value.process) termination = value.process.terminate(); await termination; } catch (reason) { retainFailure(reason); }
      value.session = { ...value.session, state: failure ? "failed" : state, finishedAt: new Date().toISOString(),
        ...(failure || error ? { error: failure instanceof Error ? failure.message : failure ? String(failure) : error } : {}) };
      try { await value.store.session(value.session); } catch (reason) { retainFailure(reason); }
      finally { try { value.store.close(); } catch (reason) { retainFailure(reason); } finally { try { await value.lease.release(); } catch (reason) { retainFailure(reason); } } }
    }
    this.publish(value, "session"); this.publish(value, "contributions"); if (failure) throw failure; return structuredClone(value.session);
  }
  private async settleStops(brainId?: string): Promise<void> {
    const pending = [...this.stopping.values()].filter(value => brainId === undefined || value.brainId === brainId).map(value => value.promise);
    for (const value of this.live.values()) if (brainId === undefined || value.session.brainId === brainId)
      pending.push(this.stop(value.session.brainId, value.session.sessionId));
    const failure = (await Promise.allSettled(pending)).find((value): value is PromiseRejectedResult => value.status === "rejected");
    if (failure) throw failure.reason;
  }
  async stopOwner(brainId: string): Promise<void> { await this.settleStops(brainId); }
  async dispose(): Promise<void> { this.unsubscribe(); await this.settleStops(); }
}
