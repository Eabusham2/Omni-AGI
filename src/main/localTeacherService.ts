import { randomUUID } from "node:crypto";
import { open, realpath } from "node:fs/promises";
import { join, basename } from "node:path";
import type { LearningTask } from "../shared/learning";
import type { DatasetManifest, ToolPermissionLevel } from "../shared/types";
import type { LocalTeacherEvent, LocalTeacherItem, LocalTeacherJob,
  LocalTeacherPlan, LocalTeacherResultReceipt, LocalTeacherSelectRequest, LocalTeacherSelection,
  LocalTeacherSelectionProgress, LocalTeacherStartRequest } from "../shared/localTeacher";
import type { LearningCoordinator } from "./learningCoordinator";
import { localTeacherCanonical, localTeacherHash, localTeacherId, localTeacherRequest } from "./localTeacherFormat";
import { hashLocalTeacherFile, inspectSelectedLocalTeacherHeader, localTeacherFileBinding, LocalTeacherFileVerifier } from "./localTeacherFiles";
import { estimateLocalTeacherReservation, localTeacherPlan, type LocalTeacherMeasurements } from "./localTeacherPlan";
import { LocalTeacherStore, type LocalTeacherSessionSeal, type StoredLocalTeacherSelection } from "./localTeacherStore";
import { LOCAL_TEACHER_ENV_POLICY_SHA256, localTeacherBaseArguments, localTeacherProbeFlags, readBoundedTeacherText, startLocalTeacherChild } from "./localTeacherRunner";
import { localTeacherAttemptId, localTeacherDirectory, writeTeacherDialogueSource, writeTeacherReceipt, writeTeacherText } from "./localTeacherArtifacts";
import { localTeacherCompletionBody, LocalTeacherCompletionError, openLocalTeacherServer } from "./localTeacherServer";
import { verifyLocalTeacherTcpOwner } from "./localTeacherTcpOwner";
import { registerLocalTeacherBoundary, requireLocalTeacherMutation } from "./localTeacherWorkspaceBoundary";
import { recoverLocalTeacherSeals } from "./localTeacherRecovery";

interface Permission { level: ToolPermissionLevel; revision: string }
interface Grant { brainId: string; teacherId: string; runtimeSha256: string; modelSha256: string; permissionRevision: string }
interface Active { brainId: string; jobId: string; controller: AbortController; custody: AbortController; startedChild: boolean;
  desired?: "paused" | "stopped"; operation: Promise<void> }
export interface LocalTeacherServiceOptions {
  brainDirectory(brainId: string): Promise<string>;
  currentPermission(brainId: string): Promise<Permission>;
  /** Main-owned dialogs. Selection expressly permits a bounded --help invocation. */
  selectFiles(request: LocalTeacherSelectRequest): Promise<{ runtimePath: string; modelPath: string } | null>;
  requireLoaded(brainId: string): void | Promise<void>;
  isLoaded(brainId: string): boolean | Promise<boolean>;
  measurements(brainId: string, childPids?: readonly number[]): Promise<LocalTeacherMeasurements>;
  checkMetadata(brainId: string, bytes: number, childPids?: readonly number[]): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  withBrainWrite<T>(brainId: string, work: () => Promise<T>, signal?: AbortSignal): Promise<T>;
  withHostCompute<T>(context: { requestId: string; brainId: string; jobId: string; label: string }, signal: AbortSignal, work: (signal: AbortSignal) => Promise<T>): Promise<T>;
  previewDataset(brainId: string, paths: string[], signal: AbortSignal): Promise<DatasetManifest>;
  learning: Pick<LearningCoordinator, "start" | "list" | "get" | "pause" | "resume" | "stop" | "subscribe">;
  onError?(error: unknown, brainId?: string): void;
}
const failure = (error: unknown): string => error instanceof Error ? error.message : String(error);
const copiedAuthority = "Copied/recovered teacher jobs are dormant. Select the exact local dependencies and explicitly start a new job; no host execution grant was imported.";

/** Local teacher is an external source producer, never the Omni neural runtime. */
export class LocalTeacherService {
  private readonly stores = new Map<string, Promise<LocalTeacherStore>>();
  private readonly boundaryReleases = new Map<string, () => void>();
  private readonly grants = new Map<string, Grant>();
  private readonly active = new Map<string, Active>();
  private readonly selectionsInFlight = new Map<string, AbortController>();
  private readonly selectionSettled = new Map<string, Promise<void>>();
  private readonly progress = new Map<string, LocalTeacherSelectionProgress>();
  private readonly progressListeners = new Set<(event: LocalTeacherSelectionProgress) => void>();
  private readonly listeners = new Set<(event: LocalTeacherEvent) => void>();
  private readonly verifier = new LocalTeacherFileVerifier();
  private closed = false;
  constructor(private readonly options: LocalTeacherServiceOptions) {}
  subscribe(listener: (event: LocalTeacherEvent) => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  subscribeSelection(listener: (event: LocalTeacherSelectionProgress) => void): () => void { this.progressListeners.add(listener); return () => this.progressListeners.delete(listener); }
  private emit(job: LocalTeacherJob): LocalTeacherJob {
    for (const listener of this.listeners) { try { listener({ brainId: job.brainId, job: structuredClone(job) }); } catch (error) { this.options.onError?.(error, job.brainId); } } return job;
  }
  private publish(event: LocalTeacherSelectionProgress): void {
    this.progress.set(`${event.brainId}:${event.requestId}`, event);
    // Only bounded completion metadata is retained here; authoritative sources live on disk.
    while (this.progress.size > 32) { const key = [...this.progress.keys()].find(id => !this.selectionsInFlight.has(id)); if (!key) break; this.progress.delete(key); }
    for (const listener of this.progressListeners) { try { listener(structuredClone(event)); } catch (error) { this.options.onError?.(error, event.brainId); } }
  }
  private store(brainId: string): Promise<LocalTeacherStore> {
    if (this.closed) throw new Error("Local teacher service is closed."); localTeacherId(brainId);
    let store = this.stores.get(brainId);
    if (!store) { store = this.options.brainDirectory(brainId).then(root => {
      requireLocalTeacherMutation(root);
      this.boundaryReleases.set(brainId, registerLocalTeacherBoundary(root, () => [...this.active.values()].some(row => row.brainId === brainId) ||
        [...this.selectionsInFlight.keys()].some(key => key.startsWith(`${brainId}:`))));
      return LocalTeacherStore.open(root, brainId);
    }).then(value => value!); this.stores.set(brainId, store);
      void store.catch(() => this.stores.delete(brainId)); } return store;
  }
  private key(brainId: string, id: string): string { return `${localTeacherId(brainId)}:${localTeacherId(id)}`; }
  private async permission(brainId: string): Promise<Permission> {
    const permission = await this.options.currentPermission(brainId);
    if (permission.level === "off") throw new Error("Local teacher execution is disabled for this instance."); return permission;
  }
  private async requireGrant(row: StoredLocalTeacherSelection): Promise<void> {
    if (this.closed) throw new Error("Local teacher service is closed.");
    const permission = await this.permission(row.selection.brainId), grant = this.grants.get(this.key(row.selection.brainId, row.selection.id));
    if (!grant || grant.permissionRevision !== permission.revision || grant.runtimeSha256 !== row.runtime.sha256 || grant.modelSha256 !== row.model.sha256)
      throw new Error("Reapprove this exact local runtime/GGUF selection under the current permission policy; no copied/old host grant was used.");
    await this.options.brainDirectory(row.selection.brainId);
  }
  async selectionProgress(brainId: string, requestId: string): Promise<LocalTeacherSelectionProgress | null> { await this.options.brainDirectory(brainId); return structuredClone(this.progress.get(this.key(brainId, requestId)) ?? null); }
  async cancelSelection(brainId: string, requestId: string): Promise<boolean> { await this.options.brainDirectory(brainId); const controller = this.selectionsInFlight.get(this.key(brainId, requestId));
    if (!controller) return false; controller.abort(new Error("Teacher selection cancelled; no model was loaded.")); return true; }
  async select(request: LocalTeacherSelectRequest): Promise<LocalTeacherSelection | null> {
    if (!request || Object.keys(request).some(key => !["brainId", "requestId", "runner", "licenseDeclaration", "teacherId"].includes(key)) ||
      !["llama-server", "llama-completion", "llama-cli"].includes(request.runner)) throw new Error("Choose a supported explicit local teacher runner.");
    const brainId = localTeacherId(request.brainId), requestId = localTeacherId(request.requestId), key = this.key(brainId, requestId);
    if (this.selectionsInFlight.has(key)) throw new Error("This teacher selection is already being prepared.");
    if (request.licenseDeclaration !== undefined && typeof request.licenseDeclaration !== "string") throw new Error("License declaration must be explicit text, not verified permission.");
    const permission = await this.permission(brainId), store = await this.store(brainId), controller = new AbortController(), signal = controller.signal;
    requireLocalTeacherMutation(store.root); let selectionDone!: () => void;
    this.selectionSettled.set(key, new Promise<void>(resolve => { selectionDone = resolve; }));
    this.selectionsInFlight.set(key, controller);
    try {
      const files = await this.options.selectFiles(request); signal.throwIfAborted(); if (!files) return null;
      // Resolve only the explicitly selected dependency. This permits common
      // installed executable symlinks without following arbitrary registry paths.
      const runtimePath = await realpath(files.runtimePath), modelPath = await realpath(files.modelPath);
      if (basename(runtimePath).replace(/\.exe$/i, "") !== request.runner) throw new Error("Select the installed executable matching the chosen llama.cpp runner.");
      const admit = (bytes: number) => this.options.checkMetadata(brainId, bytes);
      const report = (phase: LocalTeacherSelectionProgress["phase"], name: string) => (bytes: number, totalBytes: number) => this.publish({ brainId, requestId, phase, fileName: name, bytes, totalBytes });
      const runtime = await hashLocalTeacherFile(runtimePath, signal, admit, report("hashing-runtime", basename(runtimePath)));
      const model = await hashLocalTeacherFile(modelPath, signal, admit, report("hashing-model", basename(modelPath)));
      this.publish({ brainId, requestId, phase: "reading-header", fileName: model.displayName, bytes: model.bytes, totalBytes: model.bytes });
      const requireSelected = async (): Promise<void> => { signal.throwIfAborted(); const current = await this.permission(brainId);
        if (current.revision !== permission.revision) throw new Error("Local teacher permission changed during selection."); await this.options.brainDirectory(brainId); };
      await requireSelected(); const header = await inspectSelectedLocalTeacherHeader(model, signal); await requireSelected();
      this.publish({ brainId, requestId, phase: "probing-runtime", fileName: runtime.displayName, bytes: 0, totalBytes: runtime.bytes });
      const probeId = localTeacherHash(`${brainId}\0${requestId}\0${randomUUID()}`);
      const probe = await this.options.withHostCompute({ brainId, jobId: probeId, requestId, label: "Inspect selected local teacher --help (no model)" }, signal,
        slotSignal => this.verifier.withVerifiedFile(localTeacherFileBinding(brainId, runtime), requireSelected, async () => {
          await this.options.reserveWrite(store.root, 2 * 1024 * 1024);
          const child = await startLocalTeacherChild({ root: store.root, relativeDirectory: `local-teachers/probes/${probeId}`, executable: runtime.path,
            arguments: ["--help"], signal: AbortSignal.any([signal, slotSignal]), maxLogBytes: 1024 * 1024, timeoutMs: 10000, requireCurrent: requireSelected,
            checkResources: pids => this.options.checkMetadata(brainId, 256 * 1024, pids) });
          return child.closed;
        }, signal));
      if (probe.exitCode !== 0 || probe.error || probe.stdout.coverage !== "complete" || probe.stderr.coverage !== "complete") throw new Error(probe.error ?? "Selected runtime --help failed; no model was loaded.");
      const help = await readBoundedTeacherText(join(store.root, probe.stdout.relativePath), 1024 * 1024) + "\n" + await readBoundedTeacherText(join(store.root, probe.stderr.relativePath), 1024 * 1024);
      const flags = localTeacherProbeFlags(help, request.runner), id = request.teacherId ? localTeacherId(request.teacherId) : localTeacherHash(`${brainId}\0${requestId}`);
      const existing = request.teacherId ? store.selection(id) : undefined;
      if (existing && (existing.runtime.sha256 !== runtime.sha256 || existing.model.sha256 !== model.sha256 || existing.selection.runner !== request.runner)) throw new Error("Reapproval chose different dependencies. Create a separate teacher selection instead of rewriting its identity.");
      const selection: LocalTeacherSelection = existing?.selection ?? { format: "omni-local-gguf-teacher", version: 1, id, brainId, originalBrainId: brainId,
        originalSelectionId: id, createdAt: new Date().toISOString(), runner: request.runner,
        runtime: { displayName: runtime.displayName, sha256: runtime.sha256, bytes: runtime.bytes, helpSha256: localTeacherHash(help), rawOfflineSupported: flags.supported, missingFlags: flags.missingFlags },
        model: { displayName: model.displayName, sha256: model.sha256, bytes: model.bytes, header },
        license: { declaration: request.licenseDeclaration ?? header.declaredLicense ?? null,
          source: request.licenseDeclaration !== undefined ? "user-declared" : header.declaredLicense ? "gguf-metadata" : "not-declared", verified: false }, currentHostAuthorized: false };
      const row: StoredLocalTeacherSelection = { selection, runtime, model, requestId, requestSha256: localTeacherHash(localTeacherCanonical({ runner: request.runner,
        runtimeSha256: runtime.sha256, modelSha256: model.sha256, licenseDeclaration: request.licenseDeclaration ?? null })) };
      if (existing) store.reapproveSelection(row); else store.putSelection(row);
      await requireSelected(); this.grants.set(this.key(brainId, id), { brainId, teacherId: id, runtimeSha256: runtime.sha256, modelSha256: model.sha256, permissionRevision: permission.revision });
      this.publish({ brainId, requestId, phase: "complete", bytes: model.bytes, totalBytes: model.bytes }); return { ...selection, currentHostAuthorized: true };
    } catch (error) { this.publish({ brainId, requestId, phase: signal.aborted ? "cancelled" : "failed", bytes: 0, totalBytes: 0, error: failure(error) }); throw error; }
    finally { this.selectionsInFlight.delete(key); this.selectionSettled.delete(key); selectionDone(); }
  }
  async selections(brainId: string): Promise<LocalTeacherSelection[]> { const store = await this.store(brainId), permission = await this.options.currentPermission(brainId);
    return store.selections().map(row => ({ ...row.selection, currentHostAuthorized: this.grants.get(this.key(brainId, row.selection.id))?.permissionRevision === permission.revision && permission.level !== "off" })); }
  async plan(request: { brainId: string; teacherId: string; prompts?: string[] }): Promise<LocalTeacherPlan> {
    const row = (await this.store(request.brainId)).selection(localTeacherId(request.teacherId));
    if (request.prompts) for (const prompt of request.prompts) { if (typeof prompt !== "string") throw new Error("Prompt planning requires actual complete strings."); }
    return localTeacherPlan(row.selection, await this.options.measurements(request.brainId));
  }
  async jobs(brainId: string): Promise<LocalTeacherJob[]> { return (await this.store(brainId)).jobs(); }
  async get(brainId: string, jobId: string): Promise<LocalTeacherJob> { return (await this.store(brainId)).job(localTeacherId(jobId)); }
  async items(brainId: string, jobId: string, afterIndex = -1, limit = 20): Promise<LocalTeacherItem[]> {
    if (!Number.isSafeInteger(afterIndex) || afterIndex < -1 || !Number.isSafeInteger(limit) || limit < 1) throw new Error("Choose an exact teacher-source page.");
    const store = await this.store(brainId); store.job(jobId); await this.options.checkMetadata(brainId, store.itemBytes(jobId, afterIndex, limit) * 8 + 4096); return store.items(jobId, afterIndex, limit);
  }
  async start(value: LocalTeacherStartRequest): Promise<LocalTeacherJob> {
    const { request, sha256, promptBytes } = localTeacherRequest(value), store = await this.store(request.brainId), existing = store.jobByRequest(request.requestId);
    if (existing) { if (existing.requestSha256 !== sha256) throw new Error("Local teacher request changed its complete prompts/settings."); return existing; }
    const teacher = store.selection(request.teacherId); await this.requireGrant(teacher);
    if (!teacher.selection.runtime.rawOfflineSupported) throw new Error("The selected runner does not support the verified raw/offline route.");
    if (!Number.isSafeInteger(request.prompts.length * (request.epochs ?? 1))) throw new Error("Requested complete native file visits exceed supported exact accounting.");
    await this.options.checkMetadata(request.brainId, promptBytes * 8 + request.prompts.length * 4096);
    await this.resourceCheck(request.brainId, [], estimateLocalTeacherReservation(teacher.selection, request.generation));
    await this.options.reserveWrite(store.root, promptBytes * 6 + request.prompts.length * (request.generation.maxOutputBytes * 8 + 16384));
    const id = localTeacherHash(`${request.brainId}\0${request.requestId}`), now = new Date().toISOString();
    requireLocalTeacherMutation(store.root);
    const job = store.createJob({ format: "omni-local-teacher-job", version: 1, id, brainId: request.brainId, originalBrainId: request.brainId,
      originalJobId: id, requestId: request.requestId, requestSha256: sha256, teacherId: request.teacherId, generation: request.generation, epochs: request.epochs ?? 1,
      state: "queued", revision: 1, createdAt: now, updatedAt: now, promptCount: request.prompts.length, generatedCount: 0, learnedCount: 0,
      phase: "waiting-for-native-compute-boundary", proof: "attributed-generation-and-native-source-receipts" }, request.prompts);
    this.emit(job); this.launch(job); return job;
  }
  private async resourceCheck(brainId: string, pids: readonly number[], reservationBytes = 0): Promise<void> {
    const measured = await this.options.measurements(brainId, pids);
    if (Object.values(measured).some(value => !Number.isSafeInteger(value) || value < 0) || measured.logicalCpuCount < 1)
      throw new Error("Actual app/native/teacher residency is unknown; execution waits rather than guesses fit.");
    if (measured.appFamilyResidentBytes + reservationBytes > measured.selectedRamCeilingBytes || reservationBytes > measured.physicalFreeBytes * .85)
      throw new Error("Local teacher and current app/native owners exceed the reviewed RAM envelope. Unload explicitly or adjust allocation; no native owner was killed.");
    await this.options.checkMetadata(brainId, 65536, pids);
  }
  private launch(job: LocalTeacherJob): void {
    const key = this.key(job.brainId, job.id); if (this.active.has(key)) return;
    const active: Active = { brainId: job.brainId, jobId: job.id, controller: new AbortController(), custody: new AbortController(), startedChild: false, operation: Promise.resolve() };
    this.active.set(key, active);
    active.operation = this.run(active).catch(async error => {
      if (this.closed) return; const store = await this.store(job.brainId), current = store.job(job.id);
      if (current.state !== "complete") this.emit(store.update(job.id, { state: active.desired ?? "failed", phase: active.desired ? "safe-boundary" : "attention-needed", activeIndex: undefined, error: active.desired ? undefined : failure(error) }));
    }).finally(() => this.active.delete(key));
  }
  private async run(active: Active): Promise<void> {
    const store = await this.store(active.brainId); let job = store.job(active.jobId), row = store.selection(job.teacherId);
    if (job.state === "dormant") throw new Error(copiedAuthority);
    active.controller.signal.throwIfAborted();
    this.emit(store.update(job.id, { state: "running", error: undefined, phase: "preparing-local-teacher" }));
    if (job.generatedCount < job.promptCount) {
      await this.requireGrant(row);
      // Acquire the ordinary brain writer first. Reversing this order could
      // deadlock an earlier writer waiting to enter the native dispatch queue.
      await this.options.withBrainWrite(job.brainId, () => this.options.withHostCompute({ brainId: job.brainId, jobId: job.id,
        requestId: job.requestId, label: "Generate selected local GGUF teacher sources" }, active.controller.signal, async slotSignal => {
        const runSignal = AbortSignal.any([active.controller.signal, slotSignal]);
        const candidates: LocalTeacherSessionSeal["receipts"] = [], sessionId = localTeacherHash(`${job.originalJobId}\0${randomUUID()}`);
        let generationError: unknown;
        const authority = () => this.requireGrant(row);
        await this.resourceCheck(job.brainId, [], estimateLocalTeacherReservation(row.selection, job.generation));
        await this.verifier.withVerifiedFile(localTeacherFileBinding(job.brainId, row.runtime), authority, () =>
          this.verifier.withVerifiedFile(localTeacherFileBinding(job.brainId, row.model), authority, async () => {
            let session: Awaited<ReturnType<typeof openLocalTeacherServer>> | undefined;
            try {
              runSignal.throwIfAborted(); active.startedChild = true;
              if (row.selection.runner === "llama-server") session = await openLocalTeacherServer({ root: store.root, sessionId, executable: row.runtime.path, modelPath: row.model.path,
                generation: job.generation, signal: runSignal, requireCurrent: authority, verifyTcpOwner: verifyLocalTeacherTcpOwner,
                reserveLogWrite: bytes => this.options.reserveWrite(store.root, bytes),
                checkResources: async (pids, extra = 0) => { await this.resourceCheck(job.brainId, pids); if (extra) await this.options.checkMetadata(job.brainId, extra, pids); },
                onPhase: phase => this.emit(store.update(job.id, { phase })) });
              for (let index = 0; index < job.promptCount; index++) {
                runSignal.throwIfAborted(); const item = store.item(job.id, index);
                if (["generated", "learning", "complete"].includes(item.state)) { if (!store.hasSealedReceipt(job.id, item)) throw new Error("Generated source lacks its dependency-custody seal."); continue; }
                const attemptId = localTeacherAttemptId(job, index); store.updateItem(job.id, index, { state: "generating", attemptId, receipt: undefined, source: undefined });
                this.emit(store.update(job.id, { activeIndex: index, phase: "generating-attributed-source" }));
                const receipt = session ? await this.serverLesson(store.root, job, item, row, attemptId, session, runSignal)
                  : await this.consoleLesson(store.root, job, item, row, attemptId, runSignal);
                await writeTeacherReceipt(store.root, `local-teachers/attempts/${attemptId}/result.json`, receipt);
                store.updateItem(job.id, index, { receipt });
                if (receipt.state !== "complete") throw new Error(receipt.error ?? "Local teacher did not complete this source.");
                candidates.push({ index, attemptId, sha256: localTeacherHash(localTeacherCanonical(receipt)) });
              }
            } catch (error) { generationError = error; }
            finally {
              if (session) { const shutdown = await session.close(); await writeTeacherReceipt(store.root, `local-teachers/attempts/${sessionId}/shutdown.json`, shutdown); }
            }
          }, active.custody.signal), active.custody.signal);
        // Only after both opened dependency closures have passed their final
        // identity/authority checks can complete HTTP/process output be learned.
        const seal: LocalTeacherSessionSeal = { format: "omni-local-teacher-custody-seal", version: 1, sessionId, jobId: job.id, runtimeSha256: row.runtime.sha256,
          modelSha256: row.model.sha256, at: new Date().toISOString(), dependencyCustody: "verified-before-and-after-owned-session", childClosed: true, receipts: candidates };
        await localTeacherDirectory(store.root, `local-teachers/attempts/${sessionId}`);
        await writeTeacherReceipt(store.root, `local-teachers/attempts/${sessionId}/custody.json`, seal); store.sealSession(seal);
        if (generationError && !active.controller.signal.aborted) throw generationError;
      }), active.controller.signal);
    }
    active.controller.signal.throwIfAborted(); job = store.job(job.id);
    // Both hardware reservation and writer are released BEFORE ordinary source
    // ingestion. Its normal manifest/cursor owns learning and deduplication.
    const paths: string[] = [];
    for (let index = 0; index < job.promptCount; index++) {
      const item = store.item(job.id, index); if (!store.hasSealedReceipt(job.id, item) || !item.receipt) throw new Error("This source is not an exact sealed teacher lesson.");
      const source = item.source ?? await writeTeacherDialogueSource(store.root, job, item, row.selection, item.receipt, active.controller.signal);
      store.updateItem(job.id, index, { source }); paths.push(join(store.root, source.relativePath));
    }
    if (!(await this.options.isLoaded(job.brainId))) { this.emit(store.update(job.id, { state: "waiting-load", phase: "waiting-for-explicit-load", activeIndex: undefined })); return; }
    await this.options.requireLoaded(job.brainId); active.controller.signal.throwIfAborted();
    if (!job.manifestId) { const preview = await this.options.previewDataset(job.brainId, paths, active.controller.signal);
      if (!preview.id || !preview.manifestHash || preview.brainId !== job.brainId || preview.discoveredFiles !== job.promptCount)
        throw new Error("Generated source manifest did not include every selected complete lesson; nothing was silently omitted.");
      job = store.update(job.id, { manifestId: preview.id, manifestSha256: preview.manifestHash,
        learningRequestId: localTeacherHash(`${job.originalJobId}\0native-source-learning`) }, "source-manifest-bound"); }
    const requestId = job.learningRequestId!;
    let task = job.learningTaskId ? await this.options.learning.get(job.brainId, job.learningTaskId)
      : (await this.options.learning.list(job.brainId)).find(value => value.requestId === requestId);
    if (!task) task = await this.options.learning.start({ brainId: job.brainId, requestId, inputs: [{ kind: "dataset", manifestId: job.manifestId! }], epochs: job.epochs });
    job = store.update(job.id, { learningTaskId: task.id, phase: "ordinary-native-source-learning", state: "running" }, "native-task-bound");
    if (["paused", "stopped", "failed"].includes(task.state)) task = await this.options.learning.resume(job.brainId, task.id);
    for (let index = 0; index < job.promptCount; index++) if (store.item(job.id, index).state !== "complete") store.updateItem(job.id, index, { state: "learning" });
    task = await this.waitLearning(job.brainId, task.id, active.controller.signal, progress => {
      const coverage = progress.sources.find(source => source.manifestId === job.manifestId)?.coverage;
      if (coverage) this.emit(store.update(job.id, { coverage, learnedCount: Math.min(job.promptCount, coverage.processedFiles) }, "native-cursor-progress"));
    });
    const coverage = task.sources.find(source => source.manifestId === job.manifestId)?.coverage;
    if (task.state !== "complete" || !task.traversalComplete || !coverage || coverage.rejectedRecords || coverage.rejectedFiles || coverage.errorCount ||
      coverage.processedFiles !== job.promptCount * job.epochs || coverage.discoveredFiles !== job.promptCount * job.epochs || !coverage.complete ||
      coverage.requestedEpochs !== job.epochs || coverage.completedEpochs !== job.epochs)
      throw new Error(task.error ?? "Native learning has not committed full teacher-source traversal; completed receipts remain retained.");
    for (let index = 0; index < job.promptCount; index++) store.updateItem(job.id, index, { state: "complete" });
    this.emit(store.update(job.id, { state: "complete", phase: "complete", learnedCount: job.promptCount, coverage, activeIndex: undefined, error: undefined }, "full-source-traversal-committed"));
  }
  private async serverLesson(root: string, job: LocalTeacherJob, item: LocalTeacherItem, row: StoredLocalTeacherSelection, attemptId: string,
    session: Awaited<ReturnType<typeof openLocalTeacherServer>>, signal: AbortSignal): Promise<LocalTeacherResultReceipt> {
    const directory = await localTeacherDirectory(root, `local-teachers/attempts/${attemptId}`);
    const request = await open(join(directory, "request.json"), "wx", 0o600);
    try { await writeTeacherText(request, localTeacherCompletionBody(item.prompt, job.generation), false, signal); await request.sync(); } finally { await request.close(); }
    let result: Awaited<ReturnType<typeof session.complete>>, error: string | undefined;
    try { result = await session.complete(item.prompt, attemptId, signal); }
    catch (failure) { if (!(failure instanceof LocalTeacherCompletionError)) throw failure; result = { ...failure.partial, final: {} }; error = failure.message; }
    const empty = { relativePath: `local-teachers/attempts/${attemptId}/empty.stderr`, sha256: localTeacherHash(""), bytes: 0 };
    const stderr = await open(join(root, empty.relativePath), "wx", 0o600); try { await stderr.sync(); } finally { await stderr.close(); }
    return { format: "omni-local-teacher-result", version: 1, attemptId, jobId: job.id, index: item.index, inputSha256: item.inputSha256,
      runtimeSha256: row.runtime.sha256, modelSha256: row.model.sha256, argvSha256: session.argvSha256, environmentPolicySha256: LOCAL_TEACHER_ENV_POLICY_SHA256,
      exitCode: null, ...(error ? {} : { completionProof: "http-terminal" as const }), state: error ? signal.aborted ? "interrupted" : "failed" : "complete", stdout: result.content, stderr: empty, rawTransport: result.transport, sessionId: session.sessionId,
      elapsedNanoseconds: result.elapsedNanoseconds, outputChannel: "structured-completion", transportDecorationsPreserved: false, hiddenSystemPrompt: false, foreignWeightsImported: false,
      ...(error ? { error } : {}) };
  }
  private async consoleLesson(root: string, job: LocalTeacherJob, item: LocalTeacherItem, row: StoredLocalTeacherSelection, attemptId: string, signal: AbortSignal): Promise<LocalTeacherResultReceipt> {
    const directory = await localTeacherDirectory(root, `local-teachers/attempts/${attemptId}`), promptPath = join(directory, "prompt.txt"), file = await open(promptPath, "wx", 0o600);
    try { await writeTeacherText(file, item.prompt, false, signal); await file.sync(); } finally { await file.close(); }
    const args = [...localTeacherBaseArguments(row.model.path, job.generation), "--no-conversation", "--no-display-prompt", "--simple-io", "--no-escape",
      "--file", promptPath, "--predict", String(job.generation.maxOutputTokens), "--temp", String(job.generation.temperature), "--seed", String(job.generation.seed), "--color", "off", "--verbosity", "0"];
    const child = await startLocalTeacherChild({ root, relativeDirectory: `local-teachers/attempts/${attemptId}/console`, executable: row.runtime.path, arguments: args,
      signal, maxLogBytes: job.generation.maxOutputBytes, timeoutMs: job.generation.timeoutMs, requireCurrent: () => this.requireGrant(row), checkResources: pids => this.resourceCheck(job.brainId, pids) });
    const result = await child.closed;
    return { format: "omni-local-teacher-result", version: 1, attemptId, jobId: job.id, index: item.index, inputSha256: item.inputSha256,
      runtimeSha256: row.runtime.sha256, modelSha256: row.model.sha256, argvSha256: localTeacherHash(localTeacherCanonical(args)), environmentPolicySha256: LOCAL_TEACHER_ENV_POLICY_SHA256,
      exitCode: result.exitCode, completionProof: "process-exit", state: !result.error && result.exitCode === 0 && result.stdout.coverage === "complete" ? "complete" : signal.aborted ? "interrupted" : "failed",
      stdout: result.stdout, stderr: result.stderr, elapsedNanoseconds: result.elapsedNanoseconds, outputChannel: "teacher-console", transportDecorationsPreserved: true,
      hiddenSystemPrompt: false, foreignWeightsImported: false, ...(result.error ? { error: result.error } : {}) };
  }
  private async waitLearning(brainId: string, taskId: string, signal: AbortSignal, progress: (task: LearningTask) => void): Promise<LearningTask> {
    const settled = (task: LearningTask) => ["complete", "paused", "stopped", "failed", "waiting"].includes(task.state);
    signal.throwIfAborted(); const current = await this.options.learning.get(brainId, taskId); progress(current); if (settled(current)) return current;
    return new Promise((resolve, reject) => {
      const finish = (task: LearningTask) => { if (task.brainId !== brainId || task.id !== taskId) return; progress(task); if (!settled(task)) return; unsubscribe(); signal.removeEventListener("abort", abort); resolve(task); };
      const unsubscribe = this.options.learning.subscribe(event => finish(event.task));
      const abort = () => { unsubscribe(); signal.removeEventListener("abort", abort); reject(signal.reason); };
      signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
      else void this.options.learning.get(brainId, taskId).then(finish, error => { unsubscribe(); signal.removeEventListener("abort", abort); reject(error); });
    });
  }
  private async interrupt(brainId: string, jobId: string, desired: "paused" | "stopped"): Promise<LocalTeacherJob> {
    const store = await this.store(brainId), job = store.job(localTeacherId(jobId)); if (job.state === "complete" || job.state === "dormant") return job;
    const active = this.active.get(this.key(brainId, jobId)); if (active) { active.desired = desired; active.controller.abort(new Error(`Local teacher ${desired}; committed source progress is retained.`));
      if (!active.startedChild) active.custody.abort(active.controller.signal.reason);
      await active.operation; }
    const current = store.job(jobId);
    if (current.learningTaskId) await this.options.learning[desired === "paused" ? "pause" : "stop"](brainId, current.learningTaskId);
    return this.emit(store.update(jobId, { state: desired, phase: "safe-boundary", activeIndex: undefined, error: undefined }, desired));
  }
  pause(brainId: string, jobId: string): Promise<LocalTeacherJob> { return this.interrupt(brainId, jobId, "paused"); }
  stop(brainId: string, jobId: string): Promise<LocalTeacherJob> { return this.interrupt(brainId, jobId, "stopped"); }
  async resume(brainId: string, jobId: string): Promise<LocalTeacherJob> { const store = await this.store(brainId), job = store.job(localTeacherId(jobId));
    if (job.state === "dormant") throw new Error(copiedAuthority); if (job.state === "complete") return job;
    if (this.active.has(this.key(brainId, jobId))) return job;
    if (job.generatedCount < job.promptCount) await this.requireGrant(store.selection(job.teacherId)); else await this.permission(brainId);
    requireLocalTeacherMutation(store.root);
    this.emit(store.update(jobId, { state: "queued", phase: "resuming-exact-source-progress", error: undefined })); this.launch(store.job(jobId)); return store.job(jobId); }
  async resumeAfterLoad(brainId: string): Promise<void> { const store = await this.store(brainId); for (const job of store.jobs()) if (job.state === "waiting-load") await this.resume(brainId, job.id); }
  async recover(brainId: string): Promise<void> { const store = await this.store(brainId); store.recover();
    await recoverLocalTeacherSeals(store, bytes => this.options.checkMetadata(brainId, bytes)); }
  async forget(brainId: string): Promise<void> {
    for (const [key, controller] of this.selectionsInFlight) if (key.startsWith(`${brainId}:`)) controller.abort(new Error("Teacher selection cancelled for owner lifecycle change."));
    await Promise.all([...this.selectionSettled.entries()].filter(([key]) => key.startsWith(`${brainId}:`)).map(([, promise]) => promise));
    for (const active of [...this.active.values()]) if (active.brainId === brainId) await this.pause(brainId, active.jobId);
    for (const key of [...this.grants.keys()]) if (key.startsWith(`${brainId}:`)) this.grants.delete(key);
    const store = this.stores.get(brainId); this.stores.delete(brainId); if (store) (await store).close();
    this.boundaryReleases.get(brainId)?.(); this.boundaryReleases.delete(brainId);
  }
  permissionChanged(brainId: string): void { for (const key of [...this.grants.keys()]) if (key.startsWith(`${brainId}:`)) this.grants.delete(key);
    for (const active of this.active.values()) if (active.brainId === brainId) { active.custody.abort(new Error("Teacher policy changed.")); active.controller.abort(new Error("Teacher policy changed; no more sources admitted.")); } }
  async close(): Promise<void> { for (const brainId of [...this.stores.keys()]) await this.forget(brainId); this.closed = true; this.listeners.clear(); this.progressListeners.clear(); }
}
export type LocalTeacherToolService = Pick<LocalTeacherService, "selections" | "plan" | "start" | "jobs" | "get" | "items" | "pause" | "resume" | "stop">;
