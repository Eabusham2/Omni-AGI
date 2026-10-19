import { createHash, randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { lstat, mkdir, open, rename, rm } from "node:fs/promises";
import { once } from "node:events";
import { basename, join } from "node:path";
import type { DatasetManifest, IngestResult, TrainingCoverage, WebCrawlRequest, WebCrawlResult, ToolExecutionResult, ToolInvocation, ChatResult } from "../shared/types";
import type { InstanceResumeAttachment } from "../shared/instanceLifecycleApi";
import type { FactCheckClaim, FactCheckRequest, FactCheckReviewRequest, FactCheckWorkspace, LearningInput,
  LearningTask, LearningTaskEvent, LearningTaskSteerRequest, SpecificLearnRequest, WebSourceProvenance } from "../shared/learning";
import type { UserQuestionRecord } from "../shared/userQuestions";
import type { UserQuestionService } from "./userQuestions";
import { userQuestionRequestSha256 } from "./userQuestions";
import { hashFile } from "./dataIngestion";
import { LearningTaskStore, learningHash, learningId, type StoredLearningSource } from "./learningTaskStore";
import { decideWebSource, makeWebLearningPolicy, normalizeLearningUrl, normalizeWebLearningConstraints, publicationFromHtml, webUrlWithinScope, type WebCrawlControl } from "./webLearningPolicy";
import { declaredLearningCapabilityPack, LEARNING_CAPABILITY_PACK, learningCapabilityPackText } from "./learningCapabilityPack";
import type { LearningExecutionContext } from "./learningExecutionContext";
import { committedNativeLearningTask, factCheckNativeTask } from "./learningNativeTask";
import { InstanceContinuationDetachedError } from "./instanceContinuationDetached";

export interface LearningSourceControl {
  jobId: string;
  signal: AbortSignal;
  cancelled(): boolean;
  progress(value: number, message: string, coverage?: TrainingCoverage): void;
}
export interface LearningCoordinatorOptions {
  brainDirectory(brainId: string): string | Promise<string>;
  /** Explicit user requests must not implicitly load another neural identity. */
  requireLoaded(brainId: string, retainedTurnId?: string): void;
  authorize?(brainId: string, kind: LearningTask["kind"]): Promise<void>;
  reserveWrite(directory: string, bytes: number): Promise<void>;
  manifest(brainId: string, manifestId: string): Promise<DatasetManifest>;
  prepare(brainId: string, paths: string[], signal: AbortSignal): Promise<DatasetManifest>;
  ingest(brainId: string, manifestId: string, epochs: number, control: LearningSourceControl): Promise<{ coverage: TrainingCoverage; paused: boolean; pauseReason?: string }>;
  crawl(request: WebCrawlRequest, control: LearningSourceControl, policy: WebCrawlControl): Promise<WebCrawlResult>;
  questions: UserQuestionService;
  /** Same native brain, current user-authorized task; no hidden behavior prompt. */
  assess?(workspace: FactCheckWorkspace, claim: FactCheckClaim, signal: AbortSignal): Promise<Omit<FactCheckReviewRequest, "brainId" | "taskId" | "claimId" | "requestId"> | undefined>;
  followInstructions?(task: LearningTask, signal: AbortSignal): Promise<{ turnId: string; policySha256?: string }>;
  teachCapabilities?(brainId: string, pack: ReturnType<typeof declaredLearningCapabilityPack>, signal: AbortSignal): Promise<NonNullable<LearningTask["capabilityPack"]>["receipt"]>;
  onError?(error: unknown): void;
}
interface RunningTask { brainId: string; taskId: string; controller: AbortController; pauseAtBoundary: boolean; desired: "paused" | "stopped"; rerun: boolean; operation: Promise<void>;
  phase: "sources" | "native-task" | "capability-learning"; unloaded: boolean; detached?: boolean; resumeGate?: Promise<void>; releaseResume?: () => void }
function object(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`Invalid ${label}.`);
  return value as Record<string, unknown>;
}
function text(value: unknown, label: string, empty = false): string {
  if (typeof value !== "string" || (!empty && !value.trim()) || value.includes("\0")) throw new Error(`Enter ${label}.`);
  for (const point of value) if (point.length === 1 && point.charCodeAt(0) >= 0xd800 && point.charCodeAt(0) <= 0xdfff) throw new Error(`${label} contains invalid Unicode.`);
  return value;
}
function sourceInput(value: unknown): LearningInput {
  const raw = object(value, "learning source");
  const allowed = raw.kind === "text" ? ["kind", "text", "name"] : raw.kind === "dataset" ? ["kind", "manifestId"] :
    raw.kind === "web" ? ["kind", "url", "followLinks", "followExternalLinks", "respectRobots", "quarantine", "maxPages", "maxDepth"] : [];
  if (!allowed.length || Object.keys(raw).some(key => !allowed.includes(key))) throw new Error("Learning sources are text, selected dataset manifests, or web URLs; raw file paths are not grants.");
  if (raw.kind === "text") return { kind: "text", text: text(raw.text, "text to learn"), ...(raw.name === undefined ? {} : { name: text(raw.name, "source name") }) };
  if (raw.kind === "dataset") return { kind: "dataset", manifestId: learningId(raw.manifestId, "selected dataset manifest") };
  for (const field of ["followLinks", "followExternalLinks", "respectRobots", "quarantine"]) if (raw[field] !== undefined && typeof raw[field] !== "boolean") throw new Error("Invalid web learning switch.");
  for (const field of ["maxPages", "maxDepth"]) if (raw[field] !== undefined && (!Number.isSafeInteger(raw[field]) || Number(raw[field]) < (field === "maxDepth" ? 0 : 1))) throw new Error("Invalid optional web stop condition.");
  return { ...raw, kind: "web", url: normalizeLearningUrl(raw.url) } as Extract<LearningInput, { kind: "web" }>;
}

/** Durable orchestration; real source cursors and neural receipts own progress. */
export class LearningCoordinator {
  private readonly stores = new Map<string, Promise<LearningTaskStore>>();
  private readonly running = new Map<string, RunningTask>();
  private readonly queued = new Map<string, { brainId: string; taskId: string }>();
  private readonly listeners = new Set<(event: LearningTaskEvent) => void>();
  private readonly unsubscribe: () => void;
  private closed = false;
  constructor(private readonly options: LearningCoordinatorOptions) {
    this.unsubscribe = options.questions.subscribe(event => {
      if (event.type === "resolved" || event.type === "cancelled") void this.onAnswer(event.question).catch(error => options.onError?.(error));
    });
  }
  subscribe(listener: (event: LearningTaskEvent) => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  private emit(task: LearningTask, type: LearningTaskEvent["type"] = "updated"): LearningTask {
    for (const listener of this.listeners) { try { listener({ type, task: structuredClone(task) }); } catch (error) { this.options.onError?.(error); } }
    return task;
  }
  private store(brainId: string): Promise<LearningTaskStore> {
    if (this.closed) throw new Error("Learning coordinator is closed.");
    learningId(brainId);
    let store = this.stores.get(brainId);
    if (!store) {
      store = Promise.resolve(this.options.brainDirectory(brainId)).then(path => LearningTaskStore.open(path, brainId));
      this.stores.set(brainId, store);
      void store.catch(() => { this.stores.delete(brainId); });
    }
    return store;
  }
  async get(brainId: string, taskId: string): Promise<LearningTask> { return (await this.store(brainId)).get(taskId); }
  async list(brainId: string): Promise<LearningTask[]> { return (await this.store(brainId)).list(); }
  async factCheck(brainId: string, taskId: string): Promise<FactCheckWorkspace> { return (await this.store(brainId)).workspace(taskId); }
  async recover(brainId: string): Promise<LearningTask[]> {
    const store = await this.store(brainId);
    store.recover();
    // A question may resolve while this coordinator is closed. Reconcile its
    // durable receipt without starting a worker or repeating the question.
    for (const task of store.list()) if (task.kind === "fact-check") {
      for (const claim of store.claims(task.id)) {
        const declaration = claim.findings.at(-1);
        if (!claim.questionId && declaration?.question) {
          const question = await this.options.questions.request({ ...declaration.question, brainId, requestId: declaration.id,
            source: { kind: "fact-check", actionId: task.id } });
          claim.questionId = question.id;
          store.updateClaim(task.id, claim.id, { questionId: question.id, state: question.status === "pending" ? "waiting-answer" : "reviewable" });
        }
        if (!claim.questionId) continue;
        const question = await this.options.questions.get(brainId, claim.questionId);
        if (question.resolution && !question.suspended && question.resolution.id !== claim.answer?.id) {
          store.updateClaim(task.id, claim.id, { answer: question.resolution, state: "reviewable" });
          store.update(task.id, { state: "paused", phase: "answer-recovered", label: "Saved answer available · resume to continue checking" });
        }
      }
    }
    return store.list();
  }
  isLearning(brainId: string): boolean { return [...this.running.values()].some(task => task.brainId === brainId); }
  async installCapabilities(brainId: string): Promise<LearningTask> {
    const content = learningCapabilityPackText();
    this.options.requireLoaded(brainId);
    const pack = declaredLearningCapabilityPack();
    return this.create(brainId, learningHash({ capabilityPack: LEARNING_CAPABILITY_PACK.id, content, declaredPack: pack.sha256 }), "specific-learn",
      [{ kind: "text", text: content, name: `Learning capabilities · ${LEARNING_CAPABILITY_PACK.id}` }], "", {}, 1, [], undefined, { id: pack.id, sha256: pack.sha256 });
  }

  async bindNativeTurn(brainId: string, taskId: string, turnId: string, claimId?: string, inputSha256?: string): Promise<void> {
    const store = await this.store(brainId);
    store.bindNativeTurn(taskId, turnId, claimId);
    if (inputSha256 !== undefined) {
      if (!/^[a-f0-9]{64}$/.test(inputSha256)) throw new Error("Native learning input hash is invalid.");
      const old = store.get(taskId).retainedNativeTurn;
      if (old && (old.turnId !== turnId || old.inputSha256 !== inputSha256)) throw new Error("Resume the retained native task before admitting another one.");
      store.update(taskId, { retainedNativeTurn: { turnId, inputSha256, policySha256: old?.policySha256 ?? store.get(taskId).policy.sha256 } });
    }
  }
  async recordNativeAssessment(brainId: string, taskId: string, claimId: string,
    receipt: { turnId: string; policySha256: string; committedAt: string }): Promise<void> {
    const store = await this.store(brainId), binding = store.nativeTurn(receipt.turnId), task = store.get(taskId);
    if (binding?.taskId !== taskId || binding.claimId !== claimId || !/^[a-f0-9]{64}$/.test(receipt.policySha256) ||
        !Number.isFinite(Date.parse(receipt.committedAt))) throw new Error("Native assessment receipt belongs to another task or claim.");
    const claim = store.claims(taskId).find(row => row.id === claimId)!;
    store.updateClaim(taskId, claimId, { assessmentReceipt: receipt,
      ...(task.policy.sha256 !== receipt.policySha256 && claim.state !== "waiting-answer" ? { state: "reviewable" as const } : {}) });
    if (task.retainedNativeTurn?.turnId === receipt.turnId) store.update(taskId, { retainedNativeTurn: undefined });
  }
  async executionContext(brainId: string, turnId: string): Promise<LearningExecutionContext | undefined> {
    const store = await this.store(brainId), binding = store.nativeTurn(turnId);
    if (!binding) return undefined;
    return { brainId, taskId: binding.taskId, claimId: binding.claimId, turnId, originalTaskId: store.get(binding.taskId).inheritedFrom?.taskId,
      policy: async () => {
        const task = store.get(binding.taskId);
        await this.options.authorize?.(brainId, task.kind);
        if (["paused", "stopped", "failed", "complete"].includes(task.state)) throw new Error("This learning task is not running; resume it before another external action.");
        return task.policy;
      }, observe: async (invocation, result) => this.observeNativeEvidence(store, binding, invocation, result) };
  }
  private observeNativeEvidence(store: LearningTaskStore, binding: { taskId: string; claimId?: string },
    invocation: ToolInvocation, result: ToolExecutionResult): void {
    if (!binding.claimId || result.state !== "complete" || result.toolId !== invocation.toolId || result.action !== invocation.action ||
        !["web.fetch", "web.search"].includes(invocation.toolId)) return;
    const task = store.get(binding.taskId), claim = store.claims(task.id).find(row => row.id === binding.claimId);
    if (!claim) throw new Error("The captured web result has no bound fact-check claim.");
    const output = object(result.output, "captured web outcome");
    const candidates = invocation.toolId === "web.fetch" ? [output] : Array.isArray(output.results) ? output.results : [];
    for (const value of candidates) {
      if (!value || typeof value !== "object" || Array.isArray(value)) continue;
      const row = value as Record<string, unknown>;
      const rawUrl = invocation.toolId === "web.fetch" ? row.finalUrl : row.url ?? row.link;
      let url: string; try { url = normalizeLearningUrl(rawUrl); } catch { continue; }
      const publication = invocation.toolId === "web.fetch" && typeof row.content === "string" && String(row.contentType).includes("html")
        ? publicationFromHtml(row.content) : publicationFromHtml("");
      const provenance = decideWebSource(url, new URL(url), String(row.contentType ?? ""), publication, task.policy, result.finishedAt);
      // Search hits with unknown dates are navigational references only. They
      // cannot establish a dated evidence capture or a definite finding.
      if (provenance.decision !== "admitted") continue;
      const hash = invocation.toolId === "web.fetch" ? row.sha256 : learningHash(row);
      if (typeof hash !== "string" || !/^[a-f0-9]{64}$/.test(hash)) throw new Error("Captured web result lacks an actual content hash.");
      store.addEvidence(task.id, { id: learningHash({ claimId: claim.id, capture: invocation.toolId, url, hash }), claimId: claim.id,
        url, requestedUrl: invocation.toolId === "web.fetch" ? String(invocation.arguments.url) : url,
        contentSha256: hash, toolReceiptId: result.id, toolId: invocation.toolId as "web.fetch" | "web.search",
        capture: invocation.toolId === "web.fetch" ? "fetched-content" : "search-result", publication,
        fetchedAt: result.finishedAt ?? result.startedAt, policyRevision: task.policy.revision, policySha256: task.policy.sha256, relation: "unreviewed" });
    }
    this.emit(store.get(task.id));
  }

  async start(request: SpecificLearnRequest, inheritedTaskId?: string): Promise<LearningTask> {
    const raw = object(request, "learning request");
    const brainId = learningId(raw.brainId); this.options.requireLoaded(brainId);
    await this.options.authorize?.(brainId, "specific-learn");
    if (!Array.isArray(raw.inputs) || !raw.inputs.length) throw new Error("Select text, files/folders, or a web source to learn.");
    if (raw.quarantine !== undefined && typeof raw.quarantine !== "boolean") throw new Error("Web quarantine must be a boolean.");
    const inputs = raw.inputs.map(sourceInput).map(input => input.kind === "web" && input.quarantine === undefined && raw.quarantine !== undefined
      ? { ...input, quarantine: raw.quarantine as boolean } : input);
    if (raw.quarantine !== undefined && !inputs.some(input => input.kind === "web")) throw new Error("Quarantine applies to web sources only.");
    const instructions = text(raw.instructions ?? "", "learning instructions", true);
    const requestedConstraints = normalizeWebLearningConstraints(raw.constraints);
    let constraints = requestedConstraints;
    if (inheritedTaskId) {
      const parent = (await this.store(brainId)).get(learningId(inheritedTaskId));
      if (["paused", "stopped", "failed", "complete"].includes(parent.state)) throw new Error("Resume the parent learning task before starting a dependent source.");
      constraints = parent.policy.constraints;
      for (const input of inputs) if (input.kind === "web" && !webUrlWithinScope(new URL(input.url), constraints)) throw new Error("The new web source is outside the current learning task's scope.");
    }
    const epochs = raw.epochs ?? 1;
    if (!Number.isSafeInteger(epochs) || Number(epochs) < 1) throw new Error("Learning epochs must be a positive integer.");
    const requestId = raw.requestId === undefined ? randomUUID() : learningId(raw.requestId, "learning request");
    return this.create(brainId, requestId, "specific-learn", inputs, instructions, constraints, Number(epochs), [], inheritedTaskId, undefined, requestedConstraints);
  }
  async startFactCheck(request: FactCheckRequest): Promise<FactCheckWorkspace> {
    const raw = object(request, "fact-check request");
    const brainId = learningId(raw.brainId); this.options.requireLoaded(brainId);
    await this.options.authorize?.(brainId, "fact-check");
    if (!Array.isArray(raw.claims) || !raw.claims.length) throw new Error("Select statements to check; distributed weights are not an exact inventory of all facts.");
    const claims = raw.claims.map(value => {
      const claim = object(value, "selected claim");
      if (claim.sources !== undefined && !Array.isArray(claim.sources)) throw new Error("Claim sources must be web URLs.");
      return { text: text(claim.text, "a selected claim"), sources: [...new Set((claim.sources as unknown[] | undefined ?? []).map(normalizeLearningUrl))] };
    });
    const requestId = raw.requestId === undefined ? randomUUID() : learningId(raw.requestId, "fact-check request");
    const inputs: LearningInput[] = [...new Set(claims.flatMap(claim => claim.sources))].map(url => ({ kind: "web", url, followLinks: false }));
    const task = await this.create(brainId, requestId, "fact-check", inputs, text(raw.instructions ?? "", "fact-check instructions", true), normalizeWebLearningConstraints(raw.constraints), 1, claims);
    return this.factCheck(brainId, task.id);
  }
  private async create(brainId: string, requestId: string, kind: LearningTask["kind"], inputs: LearningInput[], instructions: string,
    constraints: import("../shared/learning").WebLearningConstraints, epochs: number, rawClaims: Array<{ text: string; sources: string[] }>, policyParentTaskId?: string,
    capabilityPack?: LearningTask["capabilityPack"], identityConstraints?: import("../shared/learning").WebLearningConstraints): Promise<LearningTask> {
    const store = await this.store(brainId);
    const id = learningHash({ brainId, requestId });
    const fingerprintInputs = inputs.map(input => input.kind === "text" ? { ...input, text: undefined, textSha256: createHash("sha256").update(input.text).digest("hex") } : input);
    const requestSha256 = learningHash({ kind, inputs: fingerprintInputs, instructions, constraints: identityConstraints ?? constraints, epochs, claims: rawClaims, policyParentTaskId, capabilityPack });
    const old = store.list().find(task => task.id === id || task.requestId === requestId);
    if (old) { if (old.requestSha256 !== requestSha256) throw new Error("Learning request retry has different contents."); return old; }
    const sources: StoredLearningSource[] = [];
    for (const [index, input] of inputs.entries()) {
      const sourceId = learningHash({ id, index, input: fingerprintInputs[index] });
      if (input.kind === "text") {
        const relativePath = `inputs/${sourceId}.txt`;
        await this.writeText(store.directory, relativePath, input.text);
        sources.push({ id: sourceId, kind: "text", label: input.name ?? "Pasted text", state: "pending",
          spec: { kind: "text", relativePath, contentSha256: createHash("sha256").update(input.text).digest("hex") } });
      } else if (input.kind === "dataset") {
        const manifest = await this.options.manifest(brainId, input.manifestId);
        if (manifest.brainId !== brainId) throw new Error("Dataset manifest belongs to another instance.");
        const names = manifest.roots.map(path => basename(path)).filter(Boolean);
        const label = `${names.slice(0, 3).join(", ")}${names.length > 3 ? ` and ${names.length - 3} more` : ""} · ${manifest.discoveredFiles.toLocaleString()} files`;
        sources.push({ id: sourceId, kind: "dataset", label, state: "pending", manifestId: input.manifestId,
          spec: { kind: "dataset", manifestId: input.manifestId, manifestSha256: manifest.manifestHash } });
      } else sources.push({ id: sourceId, kind: "web", label: input.url, state: "pending", crawlId: sourceId.slice(0, 32), spec: { ...input } });
    }
    const createdAt = new Date().toISOString();
    const task: LearningTask = { format: "omni-learning-task", version: 1, id, brainId, requestId, requestSha256, kind,
      state: "queued", createdAt, updatedAt: createdAt, revision: 1, policy: makeWebLearningPolicy(constraints, instructions, 1, createdAt),
      ...(policyParentTaskId ? { policyParentTaskId } : {}), ...(capabilityPack ? { capabilityPack } : {}),
      sources: [], epochs, pendingQuestions: [], traversalComplete: false, phase: "queued", label: kind === "fact-check" ? "Selected claims queued for checking" : "Selected sources queued for learning" };
    if (policyParentTaskId) task.policy = makeWebLearningPolicy(store.get(policyParentTaskId).policy.constraints, instructions, 1, createdAt);
    const claims: FactCheckClaim[] = rawClaims.map((claim, index) => ({ ...claim, id: learningHash({ id, index, claim }), state: "pending", evidence: [], findings: [] }));
    const result = this.emit(store.create(task, sources, claims), "created");
    this.launch(brainId, id); return result;
  }
  private async writeText(directory: string, relativePath: string, value: string): Promise<void> {
    await this.options.reserveWrite(directory, Buffer.byteLength(value));
    const inputs = join(directory, "inputs"); await mkdir(inputs, { recursive: true, mode: 0o700 });
    if ((await lstat(inputs)).isSymbolicLink()) throw new Error("Unsafe learning input directory.");
    const path = join(directory, relativePath), expected = createHash("sha256").update(value).digest("hex");
    const existing = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (existing) { if (!existing.isFile() || existing.isSymbolicLink() || await hashFile(path) !== expected) throw new Error("Learning text spool changed."); return; }
    const temporary = `${path}.${randomUUID()}.next`, output = createWriteStream(temporary, { flags: "wx", mode: 0o600 });
    try {
      for (let start = 0; start < value.length;) {
        let end = Math.min(value.length, start + 65536);
        if (end < value.length && value.charCodeAt(end - 1) >= 0xd800 && value.charCodeAt(end - 1) <= 0xdbff) end++;
        if (!output.write(value.slice(start, end), "utf8")) await once(output, "drain");
        start = end;
      }
      const closed = once(output, "close"); output.end(); await closed;
      const staged = await open(temporary, "r+");
      try { await staged.sync(); } finally { await staged.close(); }
      await rename(temporary, path);
    } catch (error) { output.destroy(); await rm(temporary, { force: true }); throw error; }
  }

  private launch(brainId: string, taskId: string): void {
    const key = `${brainId}:${taskId}`;
    if (this.closed) return;
    const existing = this.running.get(key);
    if (existing) { existing.rerun = true; return; }
    if (this.isLearning(brainId)) { this.queued.set(key, { brainId, taskId }); return; }
    this.queued.delete(key);
    const active: RunningTask = { brainId, taskId, controller: new AbortController(), pauseAtBoundary: false, desired: "paused", rerun: false, operation: Promise.resolve(), phase: "sources", unloaded: false };
    this.running.set(key, active);
    active.operation = this.drive(active).catch(error => this.options.onError?.(error)).finally(async () => {
      if (this.running.get(key) === active) this.running.delete(key);
      if (this.closed) return;
      if (active.rerun && !active.pauseAtBoundary && !active.controller.signal.aborted) {
        const store = await this.store(brainId), task = store.get(taskId);
        if (!["paused", "stopped", "failed"].includes(task.state)) {
          store.update(taskId, { state: "queued" }); this.queued.set(key, { brainId, taskId });
        }
      }
      const next = [...this.queued.entries()].find(([, item]) => item.brainId === brainId);
      if (next) { this.queued.delete(next[0]); this.launch(next[1].brainId, next[1].taskId); }
    });
  }
  private async drive(active: RunningTask): Promise<void> {
    const store = await this.store(active.brainId), signal = active.controller.signal;
    const stopped = (): boolean => signal.aborted || active.pauseAtBoundary;
    try {
      this.options.requireLoaded(active.brainId, store.get(active.taskId).retainedNativeTurn?.turnId);
      if (!store.get(active.taskId).capabilityPack) await this.options.authorize?.(active.brainId, store.get(active.taskId).kind);
      this.emit(store.update(active.taskId, { state: "running", phase: "sources", pausedForUnload: undefined, error: undefined, label: "Processing selected sources" }, "started"));
      for (let source of store.sources(active.taskId)) {
        if (stopped()) break;
        if (source.state === "complete") continue;
        this.options.requireLoaded(active.brainId);
        source = store.updateSource(active.taskId, source.id, { state: "running", error: undefined });
        this.emit(store.update(active.taskId, { currentSource: source.id,
          label: `${source.kind === "web" && source.spec.quarantine === true ? "Quarantining" : "Learning"} ${source.label}`, progress: 0 }));
        let lastUpdate = 0;
        const control: LearningSourceControl = { signal, jobId: learningHash({ taskId: active.taskId, sourceId: source.id }),
          cancelled: stopped, progress: (progress, label, coverage) => {
            if (Date.now() - lastUpdate < 200 && progress < 1) return;
            lastUpdate = Date.now();
            if (coverage) store.updateSource(active.taskId, source.id, { coverage });
            this.emit(store.update(active.taskId, { progress: Math.max(0, Math.min(0.99, progress)), label }));
          } };
        try {
          if (source.kind === "web") {
            const spec = sourceInput(source.spec) as Extract<LearningInput, { kind: "web" }>;
            const result = await this.options.crawl({ brainId: active.brainId, url: spec.url, crawlId: source.crawlId,
              // The existing Specific Learn default remains direct learning;
              // the separate direct-crawler API keeps its own quarantine default.
              policy: "encode", quarantine: spec.quarantine ?? false, resume: true, maxPages: spec.followLinks === true ? spec.maxPages : 1,
              maxDepth: spec.followLinks === true ? spec.maxDepth : 0, followExternalLinks: spec.followExternalLinks,
              respectRobots: spec.respectRobots ?? true }, control, {
                policy: async () => { await this.options.authorize?.(active.brainId, store.get(active.taskId).kind); return store.get(active.taskId).policy; },
                onCoverage: coverage => { store.updateSource(active.taskId, source.id, { coverage }); },
                onSource: async (provenance, result) => this.recordWebEvidence(store, active.taskId, provenance, result)
              });
            source = store.updateSource(active.taskId, source.id, { coverage: result.coverage,
              ...(spec.quarantine === true ? { quarantined: true } : {}),
              state: result.coverage.complete && !result.stopped ? "complete" : "paused",
              result: { crawlId: result.crawlId, visited: result.visited, skipped: result.skipped, resultLog: result.resultLog, warnings: result.warnings } });
          } else {
            let manifestId = source.manifestId;
            if (!manifestId) {
              const relativePath = source.spec.relativePath;
              if (typeof relativePath !== "string" || !/^inputs\/[a-f0-9]{64}\.txt$/.test(relativePath)) throw new Error("Learning source spool is invalid.");
              const path = join(store.directory, relativePath);
              if ((await lstat(path)).isSymbolicLink() || await hashFile(path) !== source.spec.contentSha256) throw new Error("Learning text changed before resume.");
              const manifest = await this.options.prepare(active.brainId, [path], signal);
              manifestId = manifest.id;
              // Commit the source's manifest ID before any neural call, so a
              // lost acknowledgement never starts another source cursor.
              source = store.updateSource(active.taskId, source.id, { manifestId, spec: { ...source.spec, manifestSha256: manifest.manifestHash } });
            }
            const manifest = await this.options.manifest(active.brainId, manifestId);
            if (manifest.brainId !== active.brainId || manifest.manifestHash !== source.spec.manifestSha256) throw new Error("Selected source manifest changed before resume.");
            const result = await this.options.ingest(active.brainId, manifestId, store.get(active.taskId).epochs, control);
            source = store.updateSource(active.taskId, source.id, { coverage: result.coverage,
              state: result.coverage.complete && !result.paused ? "complete" : "paused", error: result.pauseReason });
          }
        } catch (error) {
          source = store.updateSource(active.taskId, source.id, { state: stopped() ? "paused" : "failed", error: error instanceof Error ? error.message : String(error) });
          if (stopped()) break;
        }
        this.emit(store.get(active.taskId));
        // The paused source owns an exact native cursor. Do not begin another
        // source and conceal that incomplete traversal behind aggregate 100%.
        if (source.state === "paused") break;
      }
      if (stopped()) {
        this.emit(store.update(active.taskId, { state: active.desired, phase: "safe-boundary", label: active.desired === "stopped" ? "Stopped · committed learning retained" : "Paused · exact source progress retained" }, active.desired));
        return;
      }
      const capabilityTask = store.get(active.taskId);
      if (capabilityTask.capabilityPack && !capabilityTask.capabilityPack.receipt && capabilityTask.sources.every(source => source.state === "complete")) {
        if (!this.options.teachCapabilities) throw new Error("Declared native tool teaching is unavailable; text exposure alone is not action-head training.");
        const pack = declaredLearningCapabilityPack();
        if (pack.id !== capabilityTask.capabilityPack.id || pack.sha256 !== capabilityTask.capabilityPack.sha256) throw new Error("Saved supplementary teaching pack differs from the installed declaration.");
        this.emit(store.update(capabilityTask.id, { phase: "capability-learning", label: "Learning declared tool actions in the same brain" }));
        active.phase = "capability-learning";
        const receipt = await this.options.teachCapabilities(active.brainId, pack, signal);
        if (this.closed) return;
        active.phase = "sources";
        signal.throwIfAborted();
        if (!receipt || typeof receipt.candidateAccepted !== "boolean" || !Number.isSafeInteger(receipt.recordsVisited) ||
            !Number.isSafeInteger(receipt.parameterSteps) || !/^[a-f0-9]{64}$/.test(receipt.sha256)) throw new Error("Native tool teaching has no committed receipt.");
        store.update(capabilityTask.id, { capabilityPack: { ...capabilityTask.capabilityPack, receipt } });
      }
      if (store.get(active.taskId).kind === "fact-check") await this.assessClaims(store, active);
      else if (this.options.followInstructions && store.get(active.taskId).policy.instructions.trim() && store.sources(active.taskId).every(source => source.state === "complete")) {
        const task = store.get(active.taskId);
        if (task.instructionReceipt?.policySha256 !== task.policy.sha256) {
          this.emit(store.update(task.id, { phase: "instructions", label: "Processing your learning instructions in the same brain" }));
          active.phase = "native-task";
          const result = await this.options.followInstructions(task, signal);
          await this.afterNativeTask(active);
          if (this.closed) return;
          signal.throwIfAborted();
          const admittedPolicy = result.policySha256 ?? task.retainedNativeTurn?.policySha256 ?? task.policy.sha256;
          store.update(task.id, { retainedNativeTurn: undefined, instructionReceipt: { turnId: result.turnId, policySha256: admittedPolicy, committedAt: new Date().toISOString() } });
          if (admittedPolicy !== store.get(task.id).policy.sha256) active.rerun = true;
        }
      }
      if (stopped()) {
        this.emit(store.update(active.taskId, { state: active.desired, phase: "safe-boundary", label: "Paused · committed learning and source progress retained" }, active.desired));
        return;
      }
      this.finish(store, active.taskId);
    } catch (error) {
      if (this.closed) return; // A durably parked turn may settle after worker teardown.
      if (typeof error === "object" && error !== null && "code" in error && error.code === "INSTANCE_CONTINUATION_DETACHED") {
        active.pauseAtBoundary = true; active.unloaded = true;
        this.emit(store.update(active.taskId, { state: "paused", pausedForUnload: true, phase: "host-detached",
          label: "Saved native task detached for recovery · no work was cancelled" }, "host-detached"));
        return;
      }
      this.emit(store.update(active.taskId, { state: stopped() ? active.desired : "failed", phase: stopped() ? "safe-boundary" : "attention-needed",
        label: stopped() ? "Work paused at a retained boundary" : "Learning needs attention; committed progress retained", error: error instanceof Error ? error.message : String(error) }, "attention"));
    }
  }
  private async afterNativeTask(active: RunningTask): Promise<void> {
    // Native suspension already committed the frontier. Do not complete the
    // host pipeline while the user still has this identity unloaded.
    if (active.unloaded && !this.closed && !active.controller.signal.aborted) await active.resumeGate;
    if (active.detached) throw new InstanceContinuationDetachedError(active.brainId, (await this.get(active.brainId, active.taskId)).retainedNativeTurn?.turnId ?? active.taskId);
    active.phase = "sources";
    active.controller.signal.throwIfAborted();
  }
  private recordWebEvidence(store: LearningTaskStore, taskId: string, provenance: WebSourceProvenance, result?: IngestResult): void {
    if (!result || provenance.decision !== "admitted") return;
    for (const claim of store.claims(taskId)) if (claim.sources.includes(provenance.requestedUrl) || claim.sources.includes(provenance.finalUrl)) {
      const id = learningHash({ claimId: claim.id, url: provenance.finalUrl, hash: result.source.contentHash });
      store.addEvidence(taskId, { id, claimId: claim.id, url: provenance.finalUrl, sourceId: result.source.id,
        contentSha256: result.source.contentHash, publication: provenance.publication, fetchedAt: provenance.fetchedAt ?? provenance.checkedAt,
        policyRevision: provenance.policyRevision, policySha256: provenance.policySha256, capture: "learned-source", relation: "unreviewed" });
      store.updateClaim(taskId, claim.id, { state: "reviewable" });
    }
  }
  private async assessClaims(store: LearningTaskStore, active: RunningTask): Promise<void> {
    for (const claim of store.claims(active.taskId)) {
      if (active.controller.signal.aborted || active.pauseAtBoundary) return;
      if (["reviewed", "waiting-answer"].includes(claim.state)) continue;
      const workspace = store.workspace(active.taskId), nativeTask = factCheckNativeTask(workspace, claim);
      // An unresolved already-committed assessment stays unresolved. Resume
      // must not replay an identical chat merely to make the bar move.
      if (claim.assessmentReceipt?.turnId === nativeTask.turnId && claim.assessmentReceipt.policySha256 === workspace.task.policy.sha256) continue;
      if (!this.options.assess) { store.updateClaim(active.taskId, claim.id, { state: "reviewable" }); continue; }
      let proposal;
      try {
        active.phase = "native-task";
        proposal = await this.options.assess(workspace, claim, active.controller.signal);
        await this.afterNativeTask(active);
        if (this.closed) return;
      }
      catch (error) {
        if (this.closed) return;
        if (typeof error === "object" && error !== null && "code" in error && error.code === "INSTANCE_CONTINUATION_DETACHED") throw error;
        if (active.controller.signal.aborted) throw error;
        active.phase = "sources";
        store.updateClaim(active.taskId, claim.id, { state: "reviewable", reviewError: error instanceof Error ? error.message : String(error) });
        continue; // A failed claim must not silently stop independent work.
      }
      if (proposal) await this.review({ ...proposal, brainId: active.brainId, taskId: active.taskId, claimId: claim.id,
        requestId: learningHash({ taskId: active.taskId, claimId: claim.id, revision: store.get(active.taskId).revision, answer: claim.answer?.id }) }, "model");
      else {
        const current = store.claims(active.taskId).find(row => row.id === claim.id)!;
        if (!["reviewed", "waiting-answer"].includes(current.state)) store.updateClaim(active.taskId, claim.id, { state: "reviewable" });
      }
    }
  }
  private finish(store: LearningTaskStore, taskId: string): LearningTask {
    const task = store.get(taskId), claims = task.kind === "fact-check" ? store.claims(taskId) : [];
    const quarantined = task.sources.some(source => source.quarantined === true);
    const failed = task.sources.some(source => source.state === "failed"), paused = task.sources.some(source => source.state === "paused");
    const pendingQuestions = claims.filter(claim => claim.state === "waiting-answer").flatMap(claim => claim.questionId ? [claim.questionId] : []);
    const awaiting = claims.some(claim => claim.state !== "reviewed");
    const currentInstructions = task.kind === "fact-check" || !this.options.followInstructions || !task.policy.instructions.trim() || task.instructionReceipt?.policySha256 === task.policy.sha256;
    const complete = !failed && !paused && !awaiting && currentInstructions && task.sources.every(source => source.state === "complete");
    return this.emit(store.update(taskId, { state: complete ? "complete" : failed ? "failed" : paused ? "paused" : "waiting",
      pendingQuestions, traversalComplete: complete, progress: complete ? 1 : undefined, currentSource: undefined,
      phase: complete ? "complete" : "attention-needed", label: complete ? (task.kind === "fact-check" ? "Selected claims reviewed · findings remain evidence-based self-reports" :
        quarantined ? "Traversal complete · quarantined web sources requested no new learning" :
        task.capabilityPack?.receipt?.candidateAccepted === false ? "Declared teaching examined · candidate declined to preserve current abilities" : "All selected source traversal committed") :
        failed ? "Some selected sources failed; completed receipts retained" : paused ? "Source progress paused" : "Evidence processed · selected claims still need review" }, "settled"));
  }
  async review(request: FactCheckReviewRequest, source: "user" | "model" = "user"): Promise<FactCheckWorkspace> {
    const store = await this.store(learningId(request.brainId)), workspace = store.workspace(learningId(request.taskId));
    const claim = workspace.claims.find(claim => claim.id === request.claimId);
    if (!claim || !["supported", "contradicted", "mixed", "undetermined"].includes(request.conclusion)) throw new Error("Invalid selected fact-check claim or conclusion.");
    const explanation = text(request.explanation, "a finding explanation");
    if (!Array.isArray(request.evidenceIds) || new Set(request.evidenceIds).size !== request.evidenceIds.length || request.evidenceIds.some(id => !claim.evidence.some(evidence => evidence.id === id))) throw new Error("A finding must reference this claim's actual evidence.");
    if (request.conclusion !== "undetermined" && !request.evidenceIds.length) throw new Error("A definite proposed conclusion requires captured evidence.");
    for (const id of request.evidenceIds) {
      const evidence = claim.evidence.find(row => row.id === id)!;
      const scoped = decideWebSource(evidence.requestedUrl ?? evidence.url, new URL(evidence.url), "",
        evidence.publication ?? publicationFromHtml(""), workspace.task.policy, evidence.fetchedAt);
      if (scoped.decision !== "admitted") throw new Error(`This finding references evidence outside the current scope: ${scoped.decision}.`);
    }
    const findingId = learningHash({ taskId: request.taskId, requestId: learningId(request.requestId) });
    if (request.question) userQuestionRequestSha256({ ...request.question, brainId: request.brainId, requestId: findingId,
      source: { kind: "fact-check", actionId: request.taskId } });
    const finding = store.addFinding(request.taskId, { id: findingId, claimId: claim.id,
      conclusion: request.conclusion, explanation, evidenceIds: [...request.evidenceIds], source, createdAt: new Date().toISOString(), verifiedTruth: false,
      ...(request.proposedCorrection === undefined ? {} : { proposedCorrection: text(request.proposedCorrection, "a proposed correction") }),
      ...(request.question ? { question: request.question } : {}) }, learningHash({ ...request, source }));
    if (request.question) {
      const question = await this.options.questions.request({ ...request.question, brainId: request.brainId, requestId: finding.id,
        source: { kind: "fact-check", actionId: request.taskId } });
      store.updateClaim(request.taskId, claim.id, { state: question.status === "pending" ? "waiting-answer" : "reviewable", questionId: question.id, answer: question.resolution });
    } else store.updateClaim(request.taskId, claim.id, { state: "reviewed" });
    if (!this.running.has(`${request.brainId}:${request.taskId}`)) this.finish(store, request.taskId);
    else this.emit(store.get(request.taskId));
    return store.workspace(request.taskId);
  }
  private async onAnswer(question: UserQuestionRecord): Promise<void> {
    if (question.source?.kind !== "fact-check" || !question.source.actionId || !question.resolution || question.suspended) return;
    const store = await this.store(question.brainId);
    const task = store.get(question.source.actionId);
    for (const claim of store.claims(task.id)) if (claim.questionId === question.id) store.updateClaim(task.id, claim.id, { answer: question.resolution, state: "reviewable" });
    if (["running", "queued", "waiting"].includes(task.state)) {
      this.emit(store.update(task.id, { state: "queued", label: "Question answered · continuing independent checks" }));
      this.launch(question.brainId, task.id);
    } else this.emit(store.get(task.id));
  }
  async acceptCorrection(brainId: string, taskId: string, claimId: string, findingId: string): Promise<LearningTask> {
    const store = await this.store(brainId), workspace = store.workspace(taskId), claim = workspace.claims.find(claim => claim.id === claimId);
    const finding = claim?.findings.find(finding => finding.id === findingId);
    if (!claim || !finding?.proposedCorrection) throw new Error("Select an actual proposed correction.");
    const result = await this.start({ brainId, requestId: learningHash({ taskId, claimId, findingId, operation: "accepted-correction" }),
      inputs: [{ kind: "text", name: "Reviewed correction experience", text: JSON.stringify({ source: "user-accepted-correction",
        originalClaim: claim.text, proposedCorrection: finding.proposedCorrection, evidence: claim.evidence.filter(row => finding.evidenceIds.includes(row.id)),
        verifiedUniversalTruth: false }) }] });
    store.updateClaim(taskId, claimId, { correctionTaskId: result.id }); this.emit(store.get(taskId)); return result;
  }
  async pause(brainId: string, taskId: string, wait = true): Promise<LearningTask> { return this.interrupt(brainId, taskId, "paused", wait); }
  async stop(brainId: string, taskId: string, wait = true): Promise<LearningTask> { return this.interrupt(brainId, taskId, "stopped", wait); }
  private async interrupt(brainId: string, taskId: string, desired: "paused" | "stopped", wait: boolean): Promise<LearningTask> {
    const store = await this.store(brainId), task = store.get(taskId), active = this.running.get(`${brainId}:${taskId}`);
    this.queued.delete(`${brainId}:${taskId}`);
    if (task.state === "complete") return task;
    if (active) {
      active.unloaded = false; active.releaseResume?.();
      active.desired = desired; active.controller.abort(new Error(`Learning task ${desired}`));
      if (!wait) return this.emit(store.update(taskId, { phase: `${desired}-requested`, label: `${desired === "paused" ? "Pause" : "Stop"} requested · waiting for the committed boundary` }, "interruption-requested"));
      await active.operation;
    }
    return this.emit(store.update(taskId, { state: desired, pausedForUnload: undefined, label: desired === "stopped" ? "Stopped · committed learning retained" : "Paused · exact source progress retained", phase: "safe-boundary" }, desired));
  }
  async resume(brainId: string, taskId: string): Promise<LearningTask> {
    const store = await this.store(brainId), task = store.get(taskId);
    this.options.requireLoaded(brainId, task.retainedNativeTurn?.turnId);
    if (!task.capabilityPack) await this.options.authorize?.(brainId, task.kind);
    if (task.resumeBlocked) throw new Error(task.resumeBlocked.reason === "retained-native-frontier-unavailable"
      ? "This task's exact inherited native frontier is unavailable; its inputs and progress remain retained, but it will not be silently restarted."
      : "This task belongs to a different retained neural generation. Restore its matching learned checkpoint before resuming; completed records will not be silently restarted or skipped.");
    if (task.state === "complete" || this.running.has(`${brainId}:${taskId}`)) return task;
    const queued = this.emit(store.update(taskId, { state: "queued", error: undefined, label: "Resuming retained task progress" }, "resumed"));
    this.launch(brainId, taskId); return queued;
  }
  async steer(request: LearningTaskSteerRequest, fromModel = false): Promise<LearningTask> {
    const store = await this.store(request.brainId), task = store.get(request.taskId);
    if (["complete", "stopped"].includes(task.state)) throw new Error("Resume stopped work or start another task before steering.");
    const parentId = fromModel ? task.policyParentTaskId : undefined;
    const constraints = parentId ? store.get(parentId).policy.constraints : request.constraints ?? task.policy.constraints;
    const next = makeWebLearningPolicy(constraints,
      request.instructions === undefined ? task.policy.instructions : text(request.instructions, "learning instructions", true), task.policy.revision + 1);
    const changed = store.steer(task.id, request.expectedRevision, next, parentId);
    store.update(task.id, { traversalComplete: false });
    // Scope exclusions are reopened by the durable crawler under the new
    // revision. Completed neural records remain deduplicated by its receipts.
    for (const source of store.sources(task.id)) if (source.kind === "web" && source.state === "complete") store.updateSource(task.id, source.id, { state: "pending" });
    if (task.kind === "fact-check") for (const claim of store.claims(task.id)) if (claim.state === "reviewed") store.updateClaim(task.id, claim.id, { state: "reviewable" });
    const active = this.running.get(`${request.brainId}:${task.id}`);
    if (active) active.rerun = true;
    // Generated children cannot evade later user steering by retaining an old
    // scope. Explicit user steering of a child detaches it from that parent.
    const frontier = [changed.id], seen = new Set(frontier);
    for (let index = 0; index < frontier.length; index++) for (const child of store.list().filter(row => row.policyParentTaskId === frontier[index])) {
      if (seen.has(child.id)) throw new Error("Learning task scope inheritance is cyclic.");
      seen.add(child.id); frontier.push(child.id);
      const parent = store.get(child.policyParentTaskId!);
      store.steer(child.id, child.revision, makeWebLearningPolicy(parent.policy.constraints, child.policy.instructions, child.policy.revision + 1), child.policyParentTaskId);
      for (const source of store.sources(child.id)) if (source.kind === "web" && source.state === "complete") store.updateSource(child.id, source.id, { state: "pending" });
      store.update(child.id, { traversalComplete: false });
      const running = this.running.get(`${request.brainId}:${child.id}`); if (running) running.rerun = true;
      this.emit(store.get(child.id));
    }
    return this.emit(store.get(changed.id));
  }
  /** Unload waits at a named source boundary; it is not cancellation. */
  async pauseAtBoundary(brainId: string): Promise<void> {
    const store = await this.store(brainId);
    for (const [key, task] of this.queued) if (task.brainId === brainId) {
      this.queued.delete(key); this.emit(store.update(task.taskId, { state: "paused", pausedForUnload: true, phase: "unloaded", label: "Paused for unload" }));
    }
    const active = [...this.running.values()].filter(task => task.brainId === brainId);
    for (const task of active) {
      task.pauseAtBoundary = true; task.desired = "paused"; task.unloaded = true;
      this.emit(store.update(task.taskId, { state: "paused", pausedForUnload: true, phase: "unloaded", label: task.phase === "native-task" ? "Paused at the saved native turn frontier" : "Pausing at the committed source boundary" }));
      if (task.phase === "native-task" && !task.resumeGate) task.resumeGate = new Promise(resolve => { task.releaseResume = resolve; });
    }
    // Waiting for a parked actions.send here deadlocks Unload: that same
    // promise deliberately waits for the next explicit Load.
    await Promise.all(active.filter(task => task.phase !== "native-task").map(task => task.operation));
  }
  async resumeAfterLoad(brainId: string): Promise<void> {
    if (this.closed) return;
    const store = await this.store(brainId);
    for (const task of store.list()) if (task.pausedForUnload && !task.resumeBlocked) {
      const active = this.running.get(`${brainId}:${task.id}`);
      this.emit(store.update(task.id, { state: active ? "running" : "queued", pausedForUnload: undefined, error: undefined,
        label: "Continuing the retained learning task after Load" }, "loaded"));
      if (active) { active.unloaded = false; active.pauseAtBoundary = false; active.releaseResume?.(); active.resumeGate = undefined; active.releaseResume = undefined; }
      else this.launch(brainId, task.id);
    }
  }
  /** Direct lifecycle Resume also resumes its bound operational task, once. */
  async resumeNativeContinuation(brainId: string, attachment: InstanceResumeAttachment,
    run: (signal: AbortSignal) => Promise<ChatResult>): Promise<ChatResult> {
    const store = await this.store(brainId), turnId = attachment.frontier.turnId, binding = store.nativeTurn(turnId);
    if (!binding) return run(new AbortController().signal);
    const task = store.get(binding.taskId), key = `${brainId}:${task.id}`;
    if (attachment.brainId !== brainId || attachment.hostAttached || task.resumeBlocked || task.state === "complete" || this.isLearning(brainId) ||
        task.retainedNativeTurn?.turnId !== turnId || task.retainedNativeTurn.inputSha256 !== createHash("sha256").update(attachment.input).digest("hex"))
      throw new Error("This native task cannot be resumed under a different or already-active learning owner.");
    this.options.requireLoaded(brainId, turnId); await this.options.authorize?.(brainId, task.kind);
    const active: RunningTask = { brainId, taskId: task.id, controller: new AbortController(), pauseAtBoundary: false,
      desired: "paused", rerun: false, operation: Promise.resolve(), phase: "native-task", unloaded: false };
    this.running.set(key, active);
    this.emit(store.update(task.id, { state: "running", pausedForUnload: undefined, error: undefined, phase: "native-resume", label: "Resuming the exact retained task turn" }, "native-resumed"));
    const operation = (async () => {
      try {
        const result = await run(active.controller.signal); await this.afterNativeTask(active);
        if (this.closed) return result;
        const receipt = committedNativeLearningTask(result, { turnId, input: attachment.input });
        const policySha256 = task.retainedNativeTurn?.policySha256 ?? task.policy.sha256;
        if (binding.claimId) await this.recordNativeAssessment(brainId, task.id, binding.claimId, { ...receipt, policySha256 });
        else store.update(task.id, { retainedNativeTurn: undefined, instructionReceipt: { ...receipt, policySha256 } });
        this.finish(store, task.id);
        if (policySha256 !== store.get(task.id).policy.sha256) this.queued.set(key, { brainId, taskId: task.id });
        return result;
      } catch (error) {
        if (!this.closed) {
          const detached = typeof error === "object" && error !== null && "code" in error && error.code === "INSTANCE_CONTINUATION_DETACHED";
          this.emit(store.update(task.id, { state: detached || active.unloaded || active.controller.signal.aborted ? active.desired : "failed",
            pausedForUnload: detached || active.unloaded ? true : undefined, phase: detached ? "host-detached" : "attention-needed",
            label: detached ? "Saved task detached for recovery · no work was cancelled" : "Native task progress retained", error: detached ? undefined : error instanceof Error ? error.message : String(error) }));
        }
        throw error;
      } finally {
        if (this.running.get(key) === active) this.running.delete(key);
        if (!this.closed) {
          const next = [...this.queued.entries()].find(([, row]) => row.brainId === brainId);
          if (next) { this.queued.delete(next[0]); this.launch(brainId, next[1].taskId); }
        }
      }
    })();
    active.operation = operation.then(() => undefined, () => undefined);
    return operation;
  }
  /** Quiesce storage users before a recovery replaces their owned ledgers. */
  async forget(brainId: string): Promise<void> {
    if (this.isLearning(brainId)) throw new Error("Unload the retained learning task before replacing its workspace.");
    for (const [key, task] of this.queued) if (task.brainId === brainId) this.queued.delete(key);
    const store = this.stores.get(brainId); this.stores.delete(brainId); if (store) (await store).close();
  }
  async detachAndForget(brainId: string): Promise<void> {
    const active = [...this.running.values()].filter(task => task.brainId === brainId);
    if (active.some(task => !task.unloaded)) throw new Error("A learning workspace cannot be replaced before acknowledged Unload.");
    for (const task of active) { task.detached = true; task.releaseResume?.(); }
    // BrainService has retired the host attachment, not the durable native
    // turn. Its rejected promise settles these drivers without cancelling it.
    await Promise.all(active.map(task => task.operation));
    await this.forget(brainId);
  }
  async close(): Promise<void> {
    this.unsubscribe();
    this.queued.clear();
    const awaited: Promise<void>[] = [];
    for (const task of this.running.values()) {
      if (task.unloaded && task.phase === "native-task") continue; // Saved frontier, not Cancel.
      task.pauseAtBoundary = true; task.controller.abort(new Error("Application closed; source progress retained")); awaited.push(task.operation);
    }
    await Promise.all(awaited);
    this.closed = true;
    for (const task of this.running.values()) task.releaseResume?.();
    for (const store of this.stores.values()) (await store).close();
    this.stores.clear(); this.listeners.clear();
  }
}
