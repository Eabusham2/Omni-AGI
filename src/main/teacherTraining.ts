import { createHash, randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import type {
  ApiTeacherCredentialRequest,
  ApiTeacherDurableJobDetails,
  ApiTeacherModelDiscoveryCancellation,
  ApiTeacherModelDiscoveryRequest,
  ApiTeacherModelDiscoveryResult,
  ApiTeacherProvider,
  ApiTeacherProviderStatus,
  ApiTeacherTrainingRequest,
  RuntimeJob,
  RuntimeJobEvent
} from "../shared/types";
import type { BrainService } from "./brainService";
import type { SecureSecretStore } from "./secureSecretStore";
import type { HostActionOriginAuthority } from "../shared/hostActionAuthority";
import { ApiTeacherModelDiscoveryService } from "./apiTeacherModelDiscovery";
import { apiTeacherLessonModalities, type ApiTeacherMediaLesson, type ApiTeacherMediaCoverage } from "../shared/apiTeacherMedia";
import { ApiTeacherMediaResourcePause, checkedApiTeacherMediaCoverage, readApiTeacherResponseEnvelope,
  storeApiTeacherMediaLesson, storeApiTeacherResponseEnvelope, type ApiTeacherMediaIo } from "./apiTeacherMedia";
import { apiTeacherMediaEndpoint, apiTeacherMediaParts, apiTeacherMediaRequest } from "./apiTeacherMediaProviders";
import { ApiTeacherStore, apiTeacherCanonical, apiTeacherId, apiTeacherSha256,
  type ApiTeacherStoredItem, type ApiTeacherStoredJob } from "./apiTeacherStore";
import {
  API_TEACHER_ENDPOINTS,
  API_TEACHER_PROVIDERS,
  apiTeacherCredentialRequired,
  apiTeacherHttpRequest,
  apiTeacherProvider,
  apiTeacherResponseText,
  ollamaLocalModelId,
  type ApiTeacherHttpRequest
} from "./apiTeacherProviders";

type FetchLike = typeof fetch;

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function requireProvider(value: unknown): ApiTeacherProvider {
  return apiTeacherProvider(value);
}

function requireModel(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/.test(value)
  ) {
    throw new Error("Invalid API teacher model name.");
  }
  return value;
}

function credentialId(provider: ApiTeacherProvider): string {
  return `teacher:${provider}`;
}

function cleanRemoteError(value: string): string {
  return value
    .replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]")
    .replace(/[\r\n\0]+/g, " ")
    .slice(0, 1_000);
}

function redactKnownSecret(value: string, secret?: string): string {
  const clean = cleanRemoteError(value);
  return secret ? clean.split(secret).join("[REDACTED]") : clean;
}

function redactTeacherContent(value: string, secrets: readonly string[]): string {
  let result = value.replace(/\bsk-[A-Za-z0-9_-]{12,}\b/g, "[REDACTED]")
    .replace(/\bAIza[A-Za-z0-9_-]{16,}\b/g, "[REDACTED]");
  for (const secret of [...new Set(secrets.filter(Boolean))].sort((a, b) => b.length - a.length))
    result = result.split(secret).join("[REDACTED]");
  return result;
}

function credentialFreeRequestHash(request: ApiTeacherHttpRequest, attemptId: string, deliveryId: string): string {
  const protectedHeaders = new Set(["authorization", "x-api-key", "x-goog-api-key"]);
  const headers = Object.fromEntries(Object.entries(request.headers).sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([name, value]) => [name, protectedHeaders.has(name.toLowerCase()) ? "[DEVICE-CREDENTIAL]" : value]));
  return apiTeacherSha256(apiTeacherCanonical({ method: "POST", url: request.url, headers, body: request.body, attemptId, deliveryId }));
}

class UncertainTeacherDispatchError extends Error {
  constructor(message: string) { super(message); this.name = "UncertainTeacherDispatchError"; }
}

export interface ApiTeacherTrainingOptions {
  brainRoot(brainId: string): Promise<string>;
  admitWrite(brainId: string, additionalBytes: number, signal?: AbortSignal): Promise<void>;
  admitMetadata?(brainId: string, bytes: number, purpose: string, signal?: AbortSignal): Promise<void> | void;
}

interface ActiveTeacherJob {
  brainId: string;
  provider: ApiTeacherProvider;
  controller: AbortController;
  requested?: "paused" | "stopped";
  settled?: Promise<void>;
  /** Runtime-only tool ownership; never part of the stored request or export. */
  authority?: HostActionOriginAuthority;
  releaseAuthority?(): void;
}

async function readBoundedBody(response: Response): Promise<Uint8Array> {
  const declared = Number(response.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > 16 * 1024 * 1024) {
    throw new Error("API teacher response exceeded 16 MiB.");
  }
  if (!response.body) return new Uint8Array();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    if (!value) continue;
    total += value.byteLength;
    if (total > 16 * 1024 * 1024) {
      await reader.cancel().catch(() => undefined);
      throw new Error("API teacher response exceeded 16 MiB.");
    }
    chunks.push(value);
  }
  const bytes = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return bytes;
}

async function responseJson(response: Response): Promise<Record<string, unknown>> {
  const bytes = await readBoundedBody(response);
  const text = new TextDecoder().decode(bytes);
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    throw new Error(`API teacher returned invalid JSON (HTTP ${response.status}).`);
  }
  if (!response.ok) {
    const record = typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : {};
    const error = typeof record.error === "object" && record.error !== null
      ? record.error as Record<string, unknown>
      : {};
    const message = String(error.message ?? record.message ?? `HTTP ${response.status}`);
    throw new Error(`API teacher request failed: ${cleanRemoteError(message)}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("API teacher returned an invalid response object.");
  }
  return parsed as Record<string, unknown>;
}

export class ApiTeacherTrainingService extends EventEmitter {
  private readonly active = new Map<string, ActiveTeacherJob>();
  private readonly knownOwners = new Map<string, string>();
  private readonly recovered = new Set<string>();
  private readonly modelDiscovery: ApiTeacherModelDiscoveryService;

  constructor(
    private readonly brain: Pick<BrainService, "learnStructuredExperience" | "learnTeacherMediaExperience" | "preflightStart">,
    private readonly secrets: SecureSecretStore,
    private readonly options: ApiTeacherTrainingOptions,
    private readonly fetcher: FetchLike = fetch
  ) {
    super();
    this.modelDiscovery = new ApiTeacherModelDiscoveryService(this.secrets, this.fetcher);
  }

  private publish(job: RuntimeJob): void { this.emit("event", { job: { ...job } } satisfies RuntimeJobEvent); }
  private mediaIo(brainId: string, signal?: AbortSignal): ApiTeacherMediaIo {
    return { signal, admitMetadata: (bytes, purpose, checkedSignal) => this.options.admitMetadata?.(brainId, bytes, purpose, checkedSignal),
      checkDisk: (bytes, checkedSignal) => this.options.admitWrite(brainId, bytes, checkedSignal) };
  }

  private async store(brainId: string, signal?: AbortSignal): Promise<ApiTeacherStore> {
    const root = await this.options.brainRoot(apiTeacherId(brainId, "API teacher brain ID")); signal?.throwIfAborted();
    const store = await ApiTeacherStore.open(root, brainId, this.options.admitWrite, { signal });
    if (!store) throw new Error("API teacher workspace is unavailable.");
    if (!this.recovered.has(brainId)) { await store.recover(signal); this.recovered.add(brainId); }
    return store;
  }

  private async knownSecrets(): Promise<string[]> {
    return (await Promise.all(API_TEACHER_PROVIDERS.filter(apiTeacherCredentialRequired)
      .map(provider => this.secrets.get(credentialId(provider))))).filter((value): value is string => Boolean(value));
  }

  async status(): Promise<ApiTeacherProviderStatus[]> {
    return Promise.all(API_TEACHER_PROVIDERS.map(async (provider) => {
      const required = apiTeacherCredentialRequired(provider), key = required ? await this.secrets.get(credentialId(provider)) : undefined;
      return {
        provider,
        configured: required ? Boolean(key) : true,
        persistence: key ? this.secrets.persistence() : "none",
        required,
        ...(key ? { keyHint: `••••${key.slice(-4)}` } : {})
      } satisfies ApiTeacherProviderStatus;
    }));
  }

  async saveCredential(request: ApiTeacherCredentialRequest): Promise<ApiTeacherProviderStatus[]> {
    const provider = requireProvider(request?.provider);
    if (!apiTeacherCredentialRequired(provider)) throw new Error("Local Ollama uses no API key. Start or discover it explicitly on this device.");
    if (typeof request.apiKey !== "string") throw new Error("API credential is required.");
    await this.retireProvider(provider, "API teacher credential changed while this job was active.");
    await this.secrets.set(credentialId(provider), request.apiKey);
    this.modelDiscovery.credentialChanged(provider);
    return this.status();
  }

  async removeCredential(provider: ApiTeacherProvider): Promise<ApiTeacherProviderStatus[]> {
    const selected = requireProvider(provider);
    await this.retireProvider(selected, "API teacher credential was removed while this job was active.");
    await this.secrets.remove(credentialId(selected));
    this.modelDiscovery.credentialChanged(selected);
    return this.status();
  }

  discoverModels(request: ApiTeacherModelDiscoveryRequest): Promise<ApiTeacherModelDiscoveryResult> {
    return this.modelDiscovery.discover(request);
  }

  cancelModelDiscovery(request: ApiTeacherModelDiscoveryRequest): ApiTeacherModelDiscoveryCancellation {
    return this.modelDiscovery.cancel(request);
  }

  closeModelDiscovery(): void { this.modelDiscovery.close(); }

  private async retireProvider(provider: ApiTeacherProvider, reason: string): Promise<void> {
    const work = [...this.active.values()].filter(value => value.provider === provider);
    for (const value of work) {
      value.requested = "paused";
      value.controller.abort(new Error(reason));
    }
    await Promise.all(work.flatMap(value => value.settled ? [value.settled] : []));
  }

  async close(): Promise<void> {
    this.modelDiscovery.close();
    const work = [...this.active.values()];
    for (const value of work) {
      value.requested = "paused";
      value.controller.abort(new Error("API teacher service is closing."));
    }
    await Promise.all(work.flatMap(value => value.settled ? [value.settled] : []));
  }

  async list(brainId?: string): Promise<RuntimeJob[]> {
    if (!brainId) return Promise.all([...new Set(this.knownOwners.values())].map(async owner => this.list(owner))).then(rows => rows.flat()
      .sort((left, right) => right.createdAt.localeCompare(left.createdAt)));
    const store = await this.store(brainId);
    try { const jobs = store.jobs(); jobs.forEach(job => this.knownOwners.set(job.id, brainId)); return jobs.map(job => store.toRuntimeJob(job)); }
    finally { store.close(); }
  }

  async details(brainId: string, jobId: string): Promise<ApiTeacherDurableJobDetails> {
    const store = await this.store(brainId);
    try {
      const job = store.job(jobId); this.knownOwners.set(job.id, brainId);
      return { format: "omni-api-teacher-job-details", version: 1, brainId, jobId: job.id,
        originalBrainId: job.originalBrainId, originalJobId: job.originalJobId, requestId: job.requestId,
        requestSha256: job.requestSha256, provider: job.provider, model: job.model, state: job.state,
        revision: job.revision, nextIndex: job.nextIndex, promptCount: job.promptCount, learned: job.learned,
        failed: job.failed, skipped: job.skipped, active: this.active.get(job.id)?.brainId === brainId,
        createdAt: job.createdAt, updatedAt: job.updatedAt, items: store.items(job.id).map(item => structuredClone(item)),
        ...(job.error ? { error: job.error } : {}) };
    } finally { store.close(); }
  }

  async cancel(jobId: string): Promise<RuntimeJob> {
    const owner = this.active.get(jobId)?.brainId ?? this.knownOwners.get(jobId);
    if (!owner) throw new Error("List this brain's durable API teacher jobs before stopping one after restart.");
    return this.stop(owner, jobId);
  }

  async pause(brainId: string, jobId: string): Promise<RuntimeJob> { return this.interrupt(brainId, jobId, "paused"); }
  async stop(brainId: string, jobId: string): Promise<RuntimeJob> { return this.interrupt(brainId, jobId, "stopped"); }
  private async interrupt(brainId: string, jobId: string, disposition: "paused" | "stopped"): Promise<RuntimeJob> {
    const active = this.active.get(jobId); if (active?.brainId === brainId) { active.requested = disposition;
      active.controller.abort(new Error(`API teacher job ${disposition}.`)); await active.settled; }
    const store = await this.store(brainId);
    try {
      const job = store.job(jobId); this.knownOwners.set(job.id, brainId);
      if (["complete", "failed", "stopped", "dormant", "paused", "review-required"].includes(job.state)) return store.toRuntimeJob(job);
      const uncertain = store.items(job.id).some(item =>
        item.state === "dispatching" || item.state === "uncertain-review" || item.state === "response-review" || item.state === "learning-uncertain-review");
      const updated = await store.updateJob(job.id, { state: uncertain ? "review-required" : disposition,
        error: uncertain ? "Provider dispatch may have occurred; it was not repeated. Review this exact item before resuming." : undefined }, disposition);
      const projected = store.toRuntimeJob(updated); this.publish(projected); return projected;
    } finally { store.close(); }
  }

  /** Stop provider/learning work before Unload, recovery, copy, or deletion. */
  async suspend(brainId: string): Promise<void> {
    const work = [...this.active.entries()].filter(([, value]) => value.brainId === brainId);
    for (const [, value] of work) {
      value.requested = "paused";
      value.controller.abort(new Error("API teacher work suspended at the instance lifecycle boundary."));
    }
    await Promise.all(work.flatMap(([, value]) => value.settled ? [value.settled] : []));
  }

  /** Tool-policy retirement pauses only tool-owned jobs, not explicit panel work. */
  invalidateToolAuthority(brainId: string): void {
    for (const active of this.active.values()) if (active.brainId === brainId && active.authority) {
      active.requested ??= "paused";
      active.controller.abort(new Error("API teacher tool authority changed; review before resuming."));
    }
  }

  private async requireToolAuthority(active: ActiveTeacherJob | undefined, controller: AbortController): Promise<void> {
    controller.signal.throwIfAborted();
    if (!active?.authority) return;
    try {
      await active.authority.assertCurrent(); controller.signal.throwIfAborted();
      await this.brain.preflightStart(active.brainId);
      await active.authority.assertCurrent(); controller.signal.throwIfAborted();
    } catch (error) {
      active.requested ??= "paused"; controller.abort(error); throw error;
    }
  }

  private launch(jobId: string, brainId: string, provider: ApiTeacherProvider, authority?: HostActionOriginAuthority): void {
    const controller = new AbortController(), active: ActiveTeacherJob = { brainId, provider, controller, ...(authority ? { authority } : {}) };
    if (authority) {
      const abort = (): void => { active.requested ??= "paused"; controller.abort(authority.signal.reason); };
      authority.signal.addEventListener("abort", abort, { once: true });
      active.releaseAuthority = () => authority.signal.removeEventListener("abort", abort);
      if (authority.signal.aborted) abort();
    }
    this.active.set(jobId, active);
    active.settled = this.run(jobId, brainId, controller);
    void active.settled;
  }

  async resume(brainId: string, jobId: string, authority?: HostActionOriginAuthority): Promise<RuntimeJob> {
    authority?.signal.throwIfAborted(); await authority?.assertCurrent();
    if (this.active.has(jobId)) throw new Error("This API teacher job is already active.");
    const store = await this.store(brainId);
    let projected: RuntimeJob, provider!: ApiTeacherProvider;
    try {
      const job = store.job(jobId); if (job.state === "dormant") throw new Error("Copied API teacher history needs a new explicit job and current device credential consent.");
      if (store.items(job.id).some(item => ["uncertain-review", "response-review", "learning-uncertain-review", "dispatching", "learning"].includes(item.state)))
        throw new Error("Resolve the uncertain provider or neural-learning item before resuming; it was not automatically repeated.");
      if (["complete", "stopped"].includes(job.state)) throw new Error("This API teacher job is terminal.");
      authority?.signal.throwIfAborted(); await authority?.assertCurrent();
      const updated = await store.updateJob(job.id, { state: "queued", error: undefined }, "explicit-resume");
      provider = job.provider; projected = store.toRuntimeJob(updated); this.publish(projected);
    } finally { store.close(); }
    this.launch(jobId, brainId, provider, authority); return projected;
  }

  async resolveUncertain(brainId: string, jobId: string, index: number, decision: "retry" | "skip"): Promise<RuntimeJob> {
    const store = await this.store(brainId);
    try {
      const job = store.job(jobId), item = store.item(jobId, index);
      if (item.state !== "uncertain-review" && item.state !== "response-review" && item.state !== "learning-uncertain-review" && item.state !== "learning")
        throw new Error("This API teacher item has no uncertain provider or neural-learning effect to review.");
      const neural = item.state === "learning-uncertain-review" || item.state === "learning";
      const received = item.state === "response-review";
      await store.updateItem(jobId, index, decision === "retry"
        ? received ? { state: "response-staged", error: undefined } : neural ? { state: "response-committed", error: undefined }
          : { state: "pending", attemptId: undefined, requestSha256: undefined, error: undefined }
        : { state: "skipped", error: neural
          ? "User skipped an uncertain neural-learning acknowledgement; no neural update was replayed."
          : "User skipped an uncertain provider request; no retry was sent." }, `uncertain-${neural ? "learning-" : "provider-"}${decision}`);
      const items = store.items(jobId), failed = items.filter(value => value.state === "failed").length,
        skipped = items.filter(value => value.state === "skipped").length;
      const updated = await store.updateJob(jobId, { state: "paused", failed, skipped,
        nextIndex: items.find(value => !["complete", "failed", "skipped"].includes(value.state))?.index ?? job.promptCount,
        error: undefined }, "uncertain-reviewed");
      const projected = store.toRuntimeJob(updated); this.publish(projected); return projected;
    } finally { store.close(); }
  }

  async start(request: ApiTeacherTrainingRequest, authority?: HostActionOriginAuthority): Promise<RuntimeJob> {
    authority?.signal.throwIfAborted(); await authority?.assertCurrent();
    const provider = requireProvider(request?.provider);
    const model = provider === "ollama" ? ollamaLocalModelId(request?.model) : requireModel(request?.model);
    if (typeof request.brainId !== "string" || !request.brainId) {
      throw new Error("A brain is required for API teacher training.");
    }
    if (!Array.isArray(request.prompts) || request.prompts.length === 0) {
      throw new Error("Add at least one explicit learning question.");
    }
    const secrets = await this.knownSecrets();
    const prompts = request.prompts.map((prompt) => {
      if (typeof prompt !== "string") throw new Error("Teacher prompts must be text.");
      const clean = redactTeacherContent(prompt.replace(/\0/g, "").trim(), secrets);
      if (!clean || Buffer.byteLength(clean) > 1_000_000) {
        throw new Error("A teacher prompt is empty or exceeds 1 MB.");
      }
      return clean;
    });
    const maxOutputTokens = request.maxOutputTokens ?? 2_048;
    const responseModalities = request.responseModalities === undefined ? undefined : apiTeacherLessonModalities(request.responseModalities);
    if (!Number.isSafeInteger(maxOutputTokens) || maxOutputTokens < 1 || maxOutputTokens > 65_536) {
      throw new Error("API teacher output tokens must be between 1 and 65,536.");
    }
    const requestId = request.requestId === undefined ? randomUUID() : apiTeacherId(request.requestId, "API teacher request ID");
    const requestSha256 = apiTeacherSha256(apiTeacherCanonical({ brainId: request.brainId, provider, model, prompts, maxOutputTokens,
      ...(responseModalities ? { responseModalities } : {}) }));
    const now = new Date().toISOString(), id = randomUUID();
    const job: ApiTeacherStoredJob = { format: "omni-api-teacher-job", version: 1, id, brainId: request.brainId,
      originalBrainId: request.brainId, originalJobId: id, requestId, requestSha256, provider, model, maxOutputTokens,
      promptCount: prompts.length, learned: 0, failed: 0, skipped: 0, nextIndex: 0, state: "queued", revision: 1,
      createdAt: now, updatedAt: now, ...(responseModalities ? { responseModalities } : {}) };
    const items: ApiTeacherStoredItem[] = prompts.map((prompt, index) => ({ format: "omni-api-teacher-item", version: 1,
      jobId: id, index, prompt, promptSha256: apiTeacherSha256(prompt), deliveryId: apiTeacherSha256(`${id}\0${index}`), state: "pending" }));
    const store = await this.store(request.brainId);
    let result: { job: ApiTeacherStoredJob; created: boolean };
    try {
      const previous = store.jobByRequest(requestId);
      if (!previous && store.jobs().some(value => ["queued", "running"].includes(value.state)))
        throw new Error("This brain already has an active API teacher job.");
      authority?.signal.throwIfAborted(); await authority?.assertCurrent();
      result = await store.createJob(job, items);
      this.knownOwners.set(result.job.id, request.brainId);
      const projected = store.toRuntimeJob(result.job); this.publish(projected);
      if (!result.created) return projected;
    } finally { store.close(); }
    this.launch(result.job.id, request.brainId, provider, authority);
    return { id: result.job.id, brainId: result.job.brainId, kind: "teacher", state: "queued", progress: 0,
      label: `Preparing ${provider} teacher training`, createdAt: result.job.createdAt, updatedAt: result.job.updatedAt };
  }

  private async requestTeacher(
    provider: ApiTeacherProvider,
    request: ApiTeacherHttpRequest,
    apiKey: string,
    signal: AbortSignal
  ): Promise<string> {
    const requestSignal = AbortSignal.any([signal, AbortSignal.timeout(120_000)]);
    let response: Response;
    try {
      response = await this.fetcher(request.url, {
        method: "POST", headers: request.headers, body: JSON.stringify(request.body), redirect: "error", signal: requestSignal
      });
    } catch (error) {
      if (signal.aborted) throw error;
      throw new UncertainTeacherDispatchError(`Provider dispatch ended without a response: ${redactKnownSecret(error instanceof Error ? error.message : String(error), apiKey)}`);
    }
    if (response.redirected || response.type === "opaqueredirect") throw new Error("API teacher refused an authentication redirect.");
    if (response.url) {
      const actual = new URL(response.url), expected = new URL(request.url);
      if (actual.origin !== expected.origin || actual.pathname !== expected.pathname)
        throw new Error("API teacher refused an unexpected response destination.");
    }
    const text = apiTeacherResponseText(provider, await responseJson(response));
    if (!text) throw new Error("API teacher returned no learnable text.");
    return text;
  }

  private async requestTeacherMedia(brainId: string, request: ApiTeacherHttpRequest, secrets: readonly string[], signal: AbortSignal) {
    let response: Response;
    try { response = await this.fetcher(request.url, { method: "POST", headers: request.headers, body: JSON.stringify(request.body),
      redirect: "error", signal: AbortSignal.any([signal, AbortSignal.timeout(120_000)]) }); }
    catch { throw new UncertainTeacherDispatchError("Media provider dispatch ended without a proven response; it was not repeated."); }
    if (response.redirected || response.type === "opaqueredirect" || response.url && response.url !== request.url)
      throw new UncertainTeacherDispatchError("Teacher media response destination was not its exact reviewed provider endpoint.");
    if (!response.ok) throw new Error(`Teacher media provider returned HTTP ${response.status}; no media was learned.`);
    const root = await this.options.brainRoot(brainId);
    // Stream cancellation still interrupts incomplete fetch bodies. Once the
    // complete envelope is received, its durable ACK must survive a late Stop.
    try { return await storeApiTeacherResponseEnvelope(root, response, secrets, this.mediaIo(brainId)); }
    catch (error) { throw new UncertainTeacherDispatchError(`Media response could not enter safe custody: ${redactTeacherContent(error instanceof Error ? error.message : String(error), secrets)}`); }
  }

  private async run(jobId: string, brainId: string, controller: AbortController): Promise<void> {
    let store: ApiTeacherStore | undefined, apiKey = "", secrets: string[] = [];
    const active = this.active.get(jobId);
    try {
      store = await this.store(brainId, controller.signal);
      let job = store.job(jobId);
      if (job.state === "dormant") throw new Error("Copied API teacher history cannot execute without an explicit new job.");
      if (active?.authority) await this.requireToolAuthority(active, controller);
      else await this.brain.preflightStart(brainId);
      secrets = await this.knownSecrets(); controller.signal.throwIfAborted();
      apiKey = apiTeacherCredentialRequired(job.provider) ? await this.secrets.get(credentialId(job.provider)) ?? "" : "";
      if (!job.responseModalities?.some(kind => kind !== "text") && apiTeacherCredentialRequired(job.provider) && !apiKey)
        throw new Error(`No ${job.provider} API credential is configured on this device.`);
      job = await store.updateJob(job.id, { state: "running", error: undefined }, "running", controller.signal);
      this.publish(store.toRuntimeJob(job));
      for (let index = job.nextIndex; index < job.promptCount; index += 1) {
        controller.signal.throwIfAborted(); let item = store.item(job.id, index);
        const media = job.responseModalities?.some(kind => kind !== "text") === true;
        if (["complete", "failed", "skipped"].includes(item.state)) continue;
        if (item.state === "uncertain-review" || item.state === "response-review" || item.state === "learning-uncertain-review" || item.state === "dispatching") {
          job = await store.updateJob(job.id, { state: "review-required", nextIndex: index,
            error: item.state === "learning-uncertain-review"
              ? "Neural learning acknowledgement is uncertain; it was not replayed without explicit review."
              : "Provider dispatch outcome is uncertain; explicit review is required before any retry." }, "waiting-uncertain-review");
          this.publish(store.toRuntimeJob(job)); return;
        }
        if (item.state === "pending") {
          if (apiTeacherCredentialRequired(job.provider) && !apiKey) throw new Error(`No ${job.provider} API credential is configured on this device.`);
          const attemptId = randomUUID(), input = { model: job.model, prompt: item.prompt,
            maxOutputTokens: job.maxOutputTokens, ...(apiTeacherCredentialRequired(job.provider) ? { credential: apiKey } : {}) },
            request = media ? apiTeacherMediaRequest(job.provider, input, job.responseModalities!) : apiTeacherHttpRequest(job.provider, input);
          const requestSha256 = credentialFreeRequestHash(request, attemptId, item.deliveryId);
          await this.requireToolAuthority(active, controller);
          item = await store.updateItem(job.id, index, { state: "dispatching", attemptId, requestSha256, error: undefined },
            "provider-dispatch-started", controller.signal);
          let providerStarted = false;
          try {
            await this.requireToolAuthority(active, controller);
            providerStarted = true;
            if (media) {
              const responseEnvelope = await this.requestTeacherMedia(brainId, request, secrets, controller.signal);
              item = await store.updateItem(job.id, index, { state: "response-staged", rawResponseSha256: responseEnvelope.sha256,
                responseEnvelope, error: undefined }, "provider-media-response-staged");
            } else {
              const raw = await this.requestTeacher(job.provider, request, apiKey, controller.signal);
              const response = redactTeacherContent(raw, secrets).trim();
              if (!response) throw new Error("API teacher response contained no credential-safe learnable text.");
              item = await store.updateItem(job.id, index, { state: "response-committed", rawResponseSha256: sha256(raw),
                learnedResponseSha256: sha256(response), response, error: undefined }, "provider-response-committed", controller.signal);
            }
          } catch (error) {
            if (!providerStarted) {
              // The current authority/resource guard refused before the call.
              // Preserve the pending lesson, not a fictional unknown remote effect.
              await store.updateItem(job.id, index, { state: "pending", attemptId: undefined,
                requestSha256: undefined, error: undefined }, "provider-not-dispatched");
              throw error;
            }
            const requested = active?.requested;
            if (error instanceof UncertainTeacherDispatchError || controller.signal.aborted) {
              item = await store.updateItem(job.id, index, { state: "uncertain-review",
                error: redactTeacherContent(error instanceof Error ? error.message : String(error), secrets) }, "provider-dispatch-uncertain");
              job = await store.updateJob(job.id, { state: "review-required", nextIndex: index,
                error: requested ? `${requested === "stopped" ? "Stop" : "Pause"} acknowledged; provider outcome remains uncertain and was not retried.` :
                  "Provider outcome remains uncertain and was not retried." }, "paused-uncertain-provider");
              this.publish(store.toRuntimeJob(job)); return;
            }
            item = await store.updateItem(job.id, index, { state: "failed",
              error: redactTeacherContent(error instanceof Error ? error.message : String(error), secrets) }, "provider-response-failed");
            const items = store.items(job.id), failed = items.filter(value => value.state === "failed").length;
            job = await store.updateJob(job.id, { failed, nextIndex: index + 1 }, "item-failed");
            this.publish(store.toRuntimeJob(job)); continue;
          }
        }
        controller.signal.throwIfAborted(); item = store.item(job.id, index);
        if (media && item.state === "response-staged") {
          try {
            const root = await this.options.brainRoot(brainId), io = this.mediaIo(brainId, controller.signal);
            const body = await readApiTeacherResponseEnvelope(root, item.responseEnvelope!, io);
            const parts = apiTeacherMediaParts(job.provider, body, job.responseModalities!).map(part => part.type === "text"
              ? { ...part, text: redactTeacherContent(part.text, secrets) } : part);
            const mediaLesson = await storeApiTeacherMediaLesson(root, { provider: job.provider, sourceEndpoint: apiTeacherMediaEndpoint(job.provider, job.responseModalities!),
              model: job.model, deliveryId: item.deliveryId, prompt: item.prompt, responseModalities: job.responseModalities!, parts }, io);
            item = await store.updateItem(job.id, index, { state: "response-committed", mediaLesson,
              learnedResponseSha256: apiTeacherSha256(apiTeacherCanonical(mediaLesson.parts)),
              response: parts.filter(part => part.type === "text").map(part => part.text).join("\n"), error: undefined }, "typed-media-response-committed", controller.signal);
          } catch (error) {
            if (controller.signal.aborted || error instanceof ApiTeacherMediaResourcePause) throw error;
            await store.updateItem(job.id, index, { state: "response-review", error: redactTeacherContent(error instanceof Error ? error.message : String(error), secrets) }, "typed-response-refused");
            const held = await store.updateJob(job.id, { state: "review-required", nextIndex: index,
              error: "Received media response requires explicit review; no provider re-dispatch or neural learning occurred." }, "typed-response-review");
            this.publish(store.toRuntimeJob(held)); return;
          }
        }
        if (item.state !== "response-committed" && item.state !== "learning")
          throw new Error("API teacher item lost its committed response boundary.");
        const trajectory = JSON.stringify({ format: "omni-api-teacher-trajectory", version: 1,
          deliveryId: item.deliveryId, teacher: { provider: job.provider, model: job.model },
          objective: "corpus response imitation without preference optimization", input: item.prompt, output: item.response,
          hiddenBehavioralPrompt: false, rewardModel: false, rlhf: false });
        const experienceSha256 = media ? item.mediaLesson!.manifestSha256 : sha256(trajectory);
        if (item.experienceSha256 && item.experienceSha256 !== experienceSha256)
          throw new Error("API teacher committed learning payload changed after response custody.");
        await this.requireToolAuthority(active, controller);
        item = await store.updateItem(job.id, index, { state: "learning", experienceSha256, error: undefined },
          "learning-started", controller.signal);
        let learned: Awaited<ReturnType<BrainService["learnStructuredExperience"]>>, mediaCoverage: ApiTeacherMediaCoverage | undefined;
        try {
          if (media) {
            const result = await this.brain.learnTeacherMediaExperience(brainId, { lesson: item.mediaLesson!, prompt: item.prompt }, controller.signal);
            mediaCoverage = checkedApiTeacherMediaCoverage(result.mediaCoverage, item.mediaLesson!); learned = result;
          } else learned = await this.brain.learnStructuredExperience(brainId, {
            content: trajectory, name: `${job.provider}:${job.model}:${item.promptSha256.slice(0, 12)}`,
            sourceLabel: `${job.provider} API teacher trajectory`, provenanceUrl: API_TEACHER_ENDPOINTS[job.provider],
            license: "User-authorized API response training", licenseUrl: API_TEACHER_ENDPOINTS[job.provider]
          }, controller.signal);
          if (learned.brain.id !== brainId || learned.source.contentHash !== experienceSha256 || !item.learnedResponseSha256)
            throw new Error("API teacher neural learning receipt differs from its exact owner/content.");
        } catch (error) {
          item = await store.updateItem(job.id, index, { state: "learning-uncertain-review",
            error: redactTeacherContent(`Neural learning acknowledgement is uncertain and was not replayed: ${error instanceof Error ? error.message : String(error)}`, secrets) },
          "learning-acknowledgement-uncertain");
          job = await store.updateJob(job.id, { state: "review-required", nextIndex: index,
            error: "Neural learning may have committed, but its exact host receipt was not proven. Review before any retry." },
          "paused-uncertain-learning");
          this.publish(store.toRuntimeJob(job)); return;
        }
        item = await store.updateItem(job.id, index, { state: "complete", receipt: { brainId,
          sourceId: learned.source.id, contentHash: experienceSha256, learnedResponseSha256: item.learnedResponseSha256,
          deliveryId: item.deliveryId, learnedIdeas: learned.source.learnedIdeas, learnedConcepts: learned.source.learnedConcepts,
          learnedSynapses: learned.source.learnedSynapses, learnedParameterSteps: learned.source.learnedParameterSteps ?? 0,
          parametersChanged: learned.source.parametersChanged === true, ...(mediaCoverage ? { mediaCoverage } : {}) }, error: undefined }, "learning-committed", media ? undefined : controller.signal);
        const items = store.items(job.id), learnedCount = items.filter(value => value.state === "complete").length,
          failed = items.filter(value => value.state === "failed").length, skipped = items.filter(value => value.state === "skipped").length;
        job = await store.updateJob(job.id, { learned: learnedCount, failed, skipped, nextIndex: index + 1 }, "item-complete", media ? undefined : controller.signal);
        this.publish(store.toRuntimeJob(job));
      }
      job = store.job(jobId);
      const terminal = job.learned > 0 ? "complete" : "failed";
      job = await store.updateJob(job.id, { state: terminal,
        error: terminal === "failed" ? "No API teacher trajectories were learned." : undefined, nextIndex: job.promptCount }, "finished");
      this.publish(store.toRuntimeJob(job));
    } catch (error) {
      if (!store) return;
      try {
        const job = store.job(jobId), requested = active?.requested;
        if (["complete", "failed", "stopped", "dormant", "review-required"].includes(job.state)) return;
        const next = await store.updateJob(job.id, { state: requested ?? (controller.signal.aborted || error instanceof ApiTeacherMediaResourcePause ? "paused" : "failed"),
          error: redactTeacherContent(error instanceof Error ? error.message : String(error), secrets.length ? secrets : [apiKey]) }, "run-ended");
        this.publish(store.toRuntimeJob(next));
      } catch { /* Preserve the first durable/storage failure; no provider request is replayed. */ }
    } finally {
      active?.releaseAuthority?.();
      store?.close(); if (this.active.get(jobId) === active) this.active.delete(jobId);
    }
  }
}
