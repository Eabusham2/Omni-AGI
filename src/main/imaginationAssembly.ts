import { createHash, randomUUID } from "node:crypto";
import { lstat, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ImaginationAssembly, ImaginationAssemblyEvent, ImaginationAssemblyManifestSummary, ImaginationAssemblyPlan, ImaginationAssemblyRequest, ImaginationAssemblyPlanRequest, ImaginationPlanEvent } from "../shared/imaginationAssembly";
import type { ImaginationCapturedFragment, ImaginationRecording } from "../shared/imaginationCapture";
import { bundledVideoRuntimeManifest, videoRuntimeArtifactHash } from "./videoRuntimeManifest";
import type { PreparedVideoRuntime, VideoRuntimeProgress } from "./videoRuntimeProvisioner";
import { copyMutableFileIsolated } from "./mutableFileIsolation";
import { createReplayRunner, inspectReplaySource, joinReplaySpans, renderReplaySpan, replayFileHash, requireReplayEncoders, type InspectedReplaySource, type ReplayRunner } from "./imaginationReplayCodec";
import { buildReplaySpool, ReplaySpool, verifyReplaySpool, type ReplaySpoolSeal } from "./imaginationReplaySpool";
import { readReplay, registerReplayWorkspaceBoundary, replayHash, replayId, replayJobIds, REPLAY_ROOT, writeReplayState, type ReplayIntent, type ReplayStoredState } from "./imaginationAssemblyWorkspace";
import { ReplayJobStore } from "./imaginationReplayJobStore";
import { readReplayPlanRequest, saveReplayPlanRequest, type ReplayPlanRequestReceipt } from "./imaginationPlanRequests";
import { sqliteFilename } from "./sqliteFilename";

export interface ImaginationAssemblyDependencies {
  brainDirectory(id: string): string;
  snapshotDatabase(brainId: string, recordingId: string, expectedRevision: number, destination: string): Promise<ImaginationRecording>;
  prepareRuntime(signal: AbortSignal, progress: (value: VideoRuntimeProgress) => void): Promise<PreparedVideoRuntime>;
  checkResources(brainId: string, bytes: number, pids: readonly number[]): Promise<void>;
  reserve(root: string, bytes: number): Promise<void>;
  mediaUrl(brainId: string, sha256: string, mimeType: string): Promise<string>;
  /** Measured host presentation refresh, not a neural benchmark or product ceiling. */
  presentationRefreshHz?(): number | undefined;
  runner?(deps: Parameters<typeof createReplayRunner>[0]): Promise<ReplayRunner>;
}
interface PreparedPlan { brainId: string; manifest: ImaginationAssemblyManifestSummary; public: ImaginationAssemblyPlan; runtime?: PreparedVideoRuntime; directory: string; seal: ReplaySpoolSeal }
interface PlanOperation { brainId:string;requestId:string;fingerprint:string;controller:AbortController;promise:Promise<PreparedPlan>;receipt?:ReplayPlanRequestReceipt;persist:Promise<void>;lastProgress:number }
interface ActiveAssembly { brainId: string; id: string; controller: AbortController; pause: boolean; started: boolean; promise: Promise<void> }
const label = "Delivered timeline replay — presentation timing, not native synchronization" as const;
const positiveId = (value: string) => { if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$/.test(value)) throw new Error("Invalid replay owner or request identity."); return value; };
const hexHash = (body: string) => createHash("sha256").update(body).digest("hex");

/** Main-only file transformation. No neural Load, model, input admission or tool replay. */
export class ImaginationAssemblyService {
  private readonly plans = new Map<string, PreparedPlan>();
  private readonly active = new Map<string, ActiveAssembly>();
  private readonly listeners = new Set<(event: ImaginationAssemblyEvent) => void>();
  private readonly boundaries = new Map<string, () => void>();
  private readonly blocked = new Set<string>();
  private readonly admissions = new Map<string, { brainId: string; fingerprint: string; controller: AbortController; promise: Promise<ImaginationAssembly> }>();
  private readonly preparingPlans = new Map<string,PlanOperation>();
  private readonly planListeners=new Set<(event:ImaginationPlanEvent)=>void>();
  private readonly snapshotHolds=new Set<string>();
  private queue: Promise<void> = Promise.resolve();
  private disposed = false;
  constructor(private readonly deps: ImaginationAssemblyDependencies) {}
  subscribe(listener: (event: ImaginationAssemblyEvent) => void): () => void { this.listeners.add(listener); return () => { this.listeners.delete(listener); }; }
  subscribePlans(listener:(event:ImaginationPlanEvent)=>void):()=>void{this.planListeners.add(listener);return()=>{this.planListeners.delete(listener);};}
  private emit(view: ImaginationAssembly): void { for (const listener of this.listeners) try { listener({ brainId: view.brainId, assemblyId: view.id, revision: view.revision, state: view.state }); } catch { /* Display observers do not own media work. */ } }
  private async root(id: string): Promise<string> {
    positiveId(id); const root = this.deps.brainDirectory(id);
    for (const path of [root, join(root, "brain.json")]) { const info = await lstat(path); if (info.isSymbolicLink() || (path === root ? !info.isDirectory() : !info.isFile())) throw new Error("Replay requires its existing real managed owner."); }
    if (!this.boundaries.has(id)) this.boundaries.set(id, registerReplayWorkspaceBoundary(root, async work => {
      this.snapshotHolds.add(id);
      let running:ActiveAssembly[]=[];
      try {
      await Promise.all([...this.admissions.values()].filter(row => row.brainId === id).map(row => row.promise.catch(() => undefined)));
      await Promise.all([...this.preparingPlans.values()].filter(row => row.brainId === id).map(row => row.promise.catch(() => undefined)));
      running = [...this.active.values()].filter(job => job.brainId === id); for (const job of running) job.pause = true;
      await Promise.all(running.map(job => job.promise));
      return await work(); } finally {
        this.snapshotHolds.delete(id);
        if (!this.disposed && !this.blocked.has(id)) for (const job of running) if (!job.controller.signal.aborted) { const { state } = await readReplay(root, job.id, id); if (state.view.state === "dormant") this.start(id, job.id); }
      }
    }));
    return root;
  }
  private async checkedSource(root: string, fragment: Omit<ImaginationCapturedFragment, "mediaUrl" | "available">): Promise<string> {
    if (!/^[a-f0-9]{64}$/.test(fragment.sha256) || !Number.isSafeInteger(fragment.bytes) || fragment.bytes < 1) throw new Error("Recorded replay source has an invalid exact hash or byte count.");
    for (const parent of [join(root, "imagination"), join(root, "imagination", "blobs")]) { const info = await lstat(parent); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Replay sources require owned real archive directories."); }
    const path = join(root, "imagination", "blobs", `${fragment.sha256}.data`), info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size !== fragment.bytes || await replayFileHash(path) !== fragment.sha256) throw new Error("An immutable recorded source changed before replay."); return path;
  }
  private async prepared(request: Omit<ImaginationAssemblyRequest, "requestId">, signal: AbortSignal, progress:(patch:Partial<ImaginationPlanEvent>)=>void=()=>undefined): Promise<PreparedPlan> {
    signal.throwIfAborted();
    const root = await this.root(request.brainId); signal.throwIfAborted();
    const stage = await mkdtemp(join(root, ".replay-plan-")); let published = false;
    try {
      signal.throwIfAborted();
      progress({state:"snapshotting"});
      const recording = await this.deps.snapshotDatabase(request.brainId, positiveId(request.recordingId), request.expectedRevision, join(stage, "source.sqlite3"));
      signal.throwIfAborted();
      const sourceCopy = new DatabaseSync(sqliteFilename(join(stage, "source.sqlite3"))); try { sourceCopy.exec("PRAGMA journal_mode=DELETE"); } finally { sourceCopy.close(); }
      progress({state:"runtime"}); const runtime = await this.deps.prepareRuntime(signal, value => progress({state:"runtime",runtime:value})), run = await (this.deps.runner ?? createReplayRunner)({ runtime, estimatedPeakBytes: 65536, checkResources: (bytes, pids) => this.deps.checkResources(request.brainId, bytes, pids), reserve: bytes => this.deps.reserve(root, bytes) });
      const built = await buildReplaySpool({ directory: stage, recording, options: request.options, refreshHz: this.deps.presentationRefreshHz?.(), signal, progress: (state,completed,total)=>progress({state,completedFragments:completed,totalFragments:total}),
        inspect: async fragment => { await this.deps.checkResources(request.brainId, (fragment.width ?? 2) * (fragment.height ?? 2) * 32 + 65536, []); return inspectReplaySource(run, await this.checkedSource(root, fragment), fragment.mimeType, fragment.mediaTimeline, stage, signal); },
        admit: bytes => this.deps.checkResources(request.brainId, bytes, []), reserve: bytes => this.deps.reserve(root, bytes) });
      const id = replayHash(built.seal), directory = join(root, REPLAY_ROOT, "plans", id); await mkdir(join(root, REPLAY_ROOT, "plans"), { recursive: true, mode: 0o700 });
      if (await lstat(directory).catch(error => { if (error.code === "ENOENT") return undefined; throw error; })) await verifyReplaySpool(directory, built.seal);
      else { await rename(stage, directory); published = true; }
      const manifest = built.summary, plan: ImaginationAssemblyPlan = { id, outputKind: manifest.outputKind, options: manifest.options, durationMs: manifest.durationMs, totalSpans: manifest.spanCount, sourceFragments: manifest.fragmentCount, estimatedPeakBytes: built.estimatedPeakBytes,
        automatic: request.options?.width === undefined || request.options.height === undefined || request.options.fps === undefined, timingIssueCount: built.timingIssueCount, quantizationMs: manifest.presentation.maximumQuantizationMs, captureGapCount: manifest.captureGapCount };
      const prepared = { brainId: request.brainId, manifest, public: plan, runtime, directory, seal: built.seal }; this.rememberPlan(prepared); return prepared;
    } finally { if (!published) await rm(stage, { recursive: true, force: true }); }
  }
  private rememberPlan(plan: PreparedPlan): void {
    this.plans.set(plan.public.id, plan);
    while (this.plans.size > 16) this.plans.delete(this.plans.keys().next().value!);
  }
  private planProgress(entry:PlanOperation,root:string,patch:Partial<ImaginationPlanEvent>):void{
    if(!entry.receipt||entry.controller.signal.aborted&&patch.state!=="cancelling"&&patch.state!=="cancelled"&&patch.state!=="failed")return;
    const now=Date.now(),phase=patch.state??entry.receipt.event.state,complete=patch.completedFragments!==undefined&&patch.completedFragments===patch.totalFragments||patch.runtime?.completedBytes!==undefined&&patch.runtime.completedBytes===patch.runtime.totalBytes,runtimeTransition=patch.runtime&&patch.runtime.state!==entry.receipt.event.runtime?.state;if(phase===entry.receipt.event.state&&phase!=="ready"&&!complete&&!runtimeTransition&&now-entry.lastProgress<250)return;entry.lastProgress=now;
    const event:ImaginationPlanEvent={...entry.receipt.event,...patch,brainId:entry.brainId,requestId:entry.requestId,sequence:entry.receipt.event.sequence+1};entry.receipt={...entry.receipt,event,updatedAt:new Date().toISOString()};
    const receipt=structuredClone(entry.receipt);entry.persist=entry.persist.then(()=>saveReplayPlanRequest(root,receipt,bytes=>this.deps.reserve(root,bytes)));void entry.persist.catch(error=>entry.controller.abort(error));
    for(const listener of this.planListeners)try{listener(event);}catch{}
  }
  async plan(request:ImaginationAssemblyPlanRequest):Promise<ImaginationAssemblyPlan>{
    positiveId(request.requestId);const key=`${positiveId(request.brainId)}:${request.requestId}`,fingerprint=replayHash(request),prior=this.preparingPlans.get(key);
    if(prior){if(prior.fingerprint!==fingerprint)throw new Error("Preparation ID is already bound to different exact inputs.");return(await prior.promise).public;}
    if(this.disposed||this.blocked.has(request.brainId)||this.snapshotHolds.has(request.brainId))throw new Error("Replay preparation is currently suspended or checkpointing.");
    const entry:PlanOperation={brainId:request.brainId,requestId:request.requestId,fingerprint,controller:new AbortController(),promise:Promise.resolve(undefined as never),persist:Promise.resolve(),lastProgress:0};
    entry.promise=(async()=>{const root=await this.root(request.brainId),old=await readReplayPlanRequest(root,request.brainId,request.requestId);
      if(old){if(old.requestSha256!==fingerprint)throw new Error("Preparation ID has different saved source inputs.");if(old.event.state==="ready"&&old.result){await verifyReplaySpool(join(root,REPLAY_ROOT,"plans",old.result.plan.id),old.result.seal);const result:PreparedPlan={brainId:request.brainId,manifest:old.result.manifest,public:old.result.plan,directory:join(root,REPLAY_ROOT,"plans",old.result.plan.id),seal:old.result.seal};this.rememberPlan(result);return result;}throw new Error("This preparation already ended or was interrupted; retain its receipt and start a new explicit request.");}
      const at=new Date().toISOString();entry.receipt={format:"omni-replay-preparation",version:1,brainId:request.brainId,originBrainId:request.brainId,requestId:request.requestId,request:structuredClone(request),requestSha256:fingerprint,event:{brainId:request.brainId,requestId:request.requestId,sequence:1,state:"snapshotting"},createdAt:at,updatedAt:at};await saveReplayPlanRequest(root,entry.receipt,bytes=>this.deps.reserve(root,bytes));
      try{const result=await this.prepared(request,entry.controller.signal,patch=>this.planProgress(entry,root,patch));entry.controller.signal.throwIfAborted();entry.receipt={...entry.receipt!,result:{plan:result.public,manifest:result.manifest,seal:result.seal}};this.planProgress(entry,root,{state:"ready"});await entry.persist;return result;}
      catch(error){this.planProgress(entry,root,{state:entry.controller.signal.aborted?"cancelled":"failed",error:(error instanceof Error?error.message:String(error)).replaceAll(root,"[managed instance]")});await entry.persist.catch(()=>undefined);throw error;}
    })();this.preparingPlans.set(key,entry);try{return(await entry.promise).public;}finally{if(this.preparingPlans.get(key)===entry)this.preparingPlans.delete(key);}
  }
  async cancelPlan(request:{brainId:string;requestId:string}):Promise<{brainId:string;requestId:string;state:ImaginationPlanEvent["state"]|"absent"}>{
    positiveId(request.requestId);const key=`${positiveId(request.brainId)}:${request.requestId}`,entry=this.preparingPlans.get(key),root=await this.root(request.brainId);
    if(entry){entry.controller.abort(new Error("This exact replay preparation was cancelled; recorded source remains unchanged."));this.planProgress(entry,root,{state:"cancelling"});await entry.promise.catch(()=>undefined);await entry.persist.catch(()=>undefined);}
    const receipt=await readReplayPlanRequest(root,request.brainId,request.requestId);return{brainId:request.brainId,requestId:request.requestId,state:receipt?.event.state??"absent"};
  }
  async assemble(request: ImaginationAssemblyRequest): Promise<ImaginationAssembly> {
    const key = `${positiveId(request.brainId)}:${positiveId(request.requestId)}`, fingerprint = replayHash(request), prior = this.admissions.get(key);
    if (prior) { if (prior.fingerprint !== fingerprint) throw new Error("Concurrent replay request ID has different exact inputs."); return prior.promise; }
    const controller = new AbortController(), promise = this.createAssembly(request, controller.signal); this.admissions.set(key, { brainId: request.brainId, fingerprint, controller, promise });
    try { return await promise; } finally { if (this.admissions.get(key)?.promise === promise) this.admissions.delete(key); }
  }
  private async createAssembly(request: ImaginationAssemblyRequest, signal: AbortSignal): Promise<ImaginationAssembly> {
    if (this.disposed || this.blocked.has(request.brainId)||this.snapshotHolds.has(request.brainId)) throw new Error("Replay owner is currently suspended or closing; no new encoding was started.");
    const root = await this.root(request.brainId); positiveId(request.requestId);
    const fingerprint = replayHash({ brainId: request.brainId, recordingId: request.recordingId, expectedRevision: request.expectedRevision, requestId: request.requestId, planId: request.planId ?? null, options: request.options ?? null });
    for await (const id of replayJobIds(root)) { const { intent, state } = await readReplay(root, id, request.brainId); if (intent.requestId === request.requestId) { if (intent.requestSha256 !== fingerprint) throw new Error("A replay request ID is already bound to different exact inputs."); return this.project(state.view); } }
    const prepared = request.planId ? this.plans.get(request.planId) : await this.prepared(request, signal); signal.throwIfAborted();
    if (!prepared || prepared.brainId !== request.brainId || prepared.manifest.recording.id !== request.recordingId || prepared.manifest.recording.revision !== request.expectedRevision) throw new Error("The measured replay plan expired or belongs to another exact recording; plan again.");
    if (request.planId && request.options && JSON.stringify({ ...prepared.manifest.options, ...request.options }) !== JSON.stringify(prepared.manifest.options)) throw new Error("Exact options disagree with the selected measured plan; no override was silently ignored.");
    const id = randomUUID().replaceAll("-", ""), base = join(root, REPLAY_ROOT, "jobs", id); await this.deps.reserve(root, Buffer.byteLength(JSON.stringify(prepared.manifest)) + 65536); await mkdir(join(root, REPLAY_ROOT, "jobs"), { recursive: true, mode: 0o700 }); await mkdir(base, { mode: 0o700 });
    const intent: ReplayIntent = { format: "omni-replay-intent", version: 1, id, brainId: request.brainId, requestId: request.requestId, requestSha256: fingerprint, manifestSha256: prepared.manifest.manifestSha256, manifest: prepared.manifest, spoolId: prepared.public.id, spoolSeal: prepared.seal };
    const at = new Date().toISOString(), view: ImaginationAssembly = { format: "omni-imagination-assembly", version: 1, id, brainId: request.brainId, requestId: request.requestId, revision: 1, recordingId: request.recordingId, recordingRevision: request.expectedRevision,
      state: "queued", createdAt: at, updatedAt: at, completedSpans: 0, totalSpans: intent.manifest.spanCount, durationMs: intent.manifest.durationMs, manifestSha256: intent.manifestSha256, label };
    try { await writeFile(join(base, "intent.json"), JSON.stringify(intent), { flag: "wx", mode: 0o600 }); await writeReplayState(root, { view }); }
    catch (error) { await rm(base, { recursive: true, force: true }); throw error; }
    this.emit(view); this.start(request.brainId, id, prepared); return view;
  }
  private async save(root: string, state: ReplayStoredState): Promise<void> { state.view.revision++; state.view.updatedAt = new Date().toISOString(); await this.deps.reserve(root, Buffer.byteLength(JSON.stringify(state)) + 65536); await writeReplayState(root, state); this.emit(state.view); }
  private start(brainId: string, id: string, prepared?: PreparedPlan): void {
    if (this.active.has(`${brainId}:${id}`)) return; const job: ActiveAssembly = { brainId, id, controller: new AbortController(), pause: false, started: false, promise: Promise.resolve() };
    const work = this.queue.then(() => this.run(job, prepared)); this.queue = work.catch(() => undefined); job.promise = work.finally(() => { if (this.active.get(`${brainId}:${id}`) === job) this.active.delete(`${brainId}:${id}`); });
    this.active.set(`${brainId}:${id}`, job); void job.promise.catch(() => undefined);
  }
  private async run(job: ActiveAssembly, prepared?: PreparedPlan): Promise<void> {
    const root = await this.root(job.brainId), { intent, state } = await readReplay(root, job.id, job.brainId), signal = job.controller.signal, manifest = intent.manifest; job.started = true;
    const attempt = randomUUID().replaceAll("-", ""), work = join(root, REPLAY_ROOT, "jobs", job.id, "work", attempt), audioOnly = manifest.outputKind === "audio-only";
    let spool: ReplaySpool | undefined;
    try {
      if (job.pause || this.blocked.has(job.brainId)) { state.view.state = "dormant"; await this.save(root, state); return; }
      signal.throwIfAborted(); state.view.state = "preparing"; await this.save(root, state); await mkdir(work, { recursive: true, mode: 0o700 });
      const directory = join(root, REPLAY_ROOT, "plans", intent.spoolId); await verifyReplaySpool(directory, intent.spoolSeal); spool = new ReplaySpool(directory, manifest);
      const runtime = prepared?.runtime ?? await this.deps.prepareRuntime(signal, () => undefined), estimate = audioOnly ? manifest.trackCount * 262144 + 65536 : manifest.options.width * manifest.options.height * 32 * (manifest.trackCount + 2) + manifest.trackCount * 16384;
      const run = await (this.deps.runner ?? createReplayRunner)({ runtime, estimatedPeakBytes: estimate, checkResources: (bytes, pids) => this.deps.checkResources(job.brainId, bytes, pids), reserve: bytes => this.deps.reserve(root, bytes) });
      const encoders = await requireReplayEncoders(run, work, signal, audioOnly), sourceDirectory = runtime.sourceDirectory;
      if (runtime.state === "ready" && (!runtime.artifactId || !runtime.artifactSha256 || !runtime.binarySha256 || !sourceDirectory)) throw new Error("Reviewed runtime omitted its immutable binary/source identity.");
      const binarySha256 = runtime.binarySha256 ?? await replayFileHash(runtime.executablePath), pins = bundledVideoRuntimeManifest().artifacts.find(row => row.id === runtime.artifactId);
      if (runtime.state === "ready" && (!pins || videoRuntimeArtifactHash(pins) !== runtime.artifactSha256 || pins.binary.sha256 !== binarySha256 || pins.target !== runtime.target)) throw new Error("Selected runtime no longer matches its exact reviewed source/license artifact.");
      const codec = { state: runtime.state, ...(runtime.artifactId ? { artifactId: runtime.artifactId, artifactSha256: runtime.artifactSha256 } : {}), binarySha256, distributionLicense: runtime.state === "ready" ? pins?.license ?? "unidentified-reviewed-artifact" : "external-host-runtime-not-distribution-reviewed", encoders };
      state.view.codec = codec;
      for (const fragment of spool.fragments()) { signal.throwIfAborted(); await this.checkedSource(root, fragment); }
      const prior = await ReplayJobStore.open(root, job.id, job.brainId); try { prior.validate(); for (const segment of prior.segments()) { const oldPath = join(root, REPLAY_ROOT, "jobs", job.id, segment.path); if (await replayFileHash(oldPath) !== segment.sha256) throw new Error("Retained committed replay segment changed.");
        await this.deps.reserve(root, segment.bytes); await copyMutableFileIsolated(oldPath, join(work, `segment-${segment.index}.${audioOnly ? "wav" : "mkv"}`), { expectedSha256: segment.sha256 }); } } finally { prior.close(); }
      for (let i = state.view.completedSpans; i < manifest.spanCount; i++) {
        if (job.pause || this.blocked.has(job.brainId)) { state.view.state = "dormant"; delete state.view.currentAttempt; delete state.view.currentSpan; await this.save(root, state); return; }
        signal.throwIfAborted(); state.view.state = "encoding"; state.view.currentAttempt = attempt; state.view.currentSpan = i; await this.save(root, state);
        const span = spool.span(i), sources = new Map<string, InspectedReplaySource>(); for (const selected of span.states) if (selected.visual !== "none" || selected.audio !== "none") sources.set(selected.fragmentId, spool.source(root, selected.fragmentId));
        const path = join(work, `segment-${i}.${audioOnly ? "wav" : "mkv"}`); await renderReplaySpan(run, manifest, span, sources, path, work, signal);
        const info = await lstat(path); if (!info.isFile() || info.isSymbolicLink() || info.size < 1) throw new Error("Codec did not create a real replay segment.");
        const segment = { index: i, path: `work/${attempt}/segment-${i}.${audioOnly ? "wav" : "mkv"}`, sha256: await replayFileHash(path), bytes: info.size }, revision = state.view.revision;
        state.view.completedSpans = i + 1; state.view.revision++; state.view.updatedAt = new Date().toISOString(); delete state.view.currentAttempt; delete state.view.currentSpan;
        await this.deps.reserve(root, 65536); const cursor = await ReplayJobStore.open(root, job.id, job.brainId); try { state.view = cursor.commitSegment(segment, state.view, revision); } catch (error) { state.view = cursor.view(); throw error; } finally { cursor.close(); } this.emit(state.view);
      }
      if (job.pause || this.blocked.has(job.brainId)) { state.view.state = "dormant"; await this.save(root, state); return; }
      const output = join(work, `output.${audioOnly ? "wav" : "mp4"}`); await joinReplaySpans(run, manifest.spanCount, work, output, signal, audioOnly, bytes => this.deps.reserve(root, bytes),{fps:manifest.options.fps,durationMs:manifest.durationMs,spanDuration:index=>{const span=spool!.span(index);return span.endMs-span.startMs;}});
      const measured = await inspectReplaySource(run, output, audioOnly ? "audio/wav" : "video/mp4", undefined, work, signal); signal.throwIfAborted();
      if (!audioOnly && measured.frames !== Math.round(manifest.durationMs * manifest.options.fps / 1000) || Math.abs(measured.durationMs - manifest.durationMs) > Math.max(audioOnly ? 1 : 2000 / manifest.options.fps, 2048 * 1000 / 48000)) throw new Error(`Codec output did not retain the declared presentation: ${measured.frames} frames/${measured.durationMs} ms; expected ${Math.round(manifest.durationMs * manifest.options.fps / 1000)} frames/${manifest.durationMs} ms.`);
      for (const fragment of spool.fragments()) { signal.throwIfAborted(); await this.checkedSource(root, fragment); }
      const info = await lstat(output), sha = await replayFileHash(output), blobs = join(root, REPLAY_ROOT, "blobs"); await mkdir(blobs, { recursive: true, mode: 0o700 }); await this.deps.reserve(root, info.size + 65536);
      await copyMutableFileIsolated(output, join(blobs, `${sha}.data`), { expectedSha256: sha });
      const receipt = { format: "omni-delivered-replay-receipt", version: 1, manifestSha256: intent.manifestSha256, sourceSpoolSeal: intent.spoolSeal, sourceFragmentCount: manifest.fragmentCount, artifactSha256: sha, bytes: info.size, codec, measuredOutput: { durationMs: measured.durationMs, frames: measured.frames, hasAudio: measured.hasAudio, width: measured.width, height: measured.height } };
      const body = JSON.stringify(receipt), receiptSha256 = hexHash(body); await this.deps.reserve(root, Buffer.byteLength(body)); await writeFile(join(blobs, `${receiptSha256}.json`), body, { flag: "wx", mode: 0o600 }).catch(async error => { if (error.code !== "EEXIST" || await replayFileHash(join(blobs, `${receiptSha256}.json`)) !== receiptSha256) throw error; });
      state.view.state = "complete"; state.view.artifactSha256 = sha; state.view.receiptSha256 = receiptSha256; state.view.bytes = info.size; state.view.mimeType = audioOnly ? "audio/wav" : "video/mp4"; delete state.view.error; await this.save(root, state);
    } catch (error) { state.view.state = signal.aborted ? "cancelled" : "failed"; state.view.error = (error instanceof Error ? error.message : String(error)).replaceAll(root, "[managed instance]"); await this.save(root, state); }
    finally { spool?.close(); }
  }
  private async project(view: ImaginationAssembly): Promise<ImaginationAssembly> { if (!view.artifactSha256 || !view.mimeType) return structuredClone(view);
    try { return { ...view, mediaUrl: await this.deps.mediaUrl(view.brainId, view.artifactSha256, view.mimeType), available: true }; } catch { return { ...view, available: false }; } }
  async get(brainId: string, id: string): Promise<ImaginationAssembly> { const root = await this.root(brainId), { state } = await readReplay(root, replayId(id), brainId);
    if (["queued", "preparing", "encoding"].includes(state.view.state) && !this.active.has(`${brainId}:${id}`)) { state.view.state = "dormant"; await this.save(root, state); }
    return this.project(state.view); }
  async list(brainId: string, query: { afterId?: string; limit?: number; recordingId?: string } = {}) { const root = await this.root(brainId), limit = query.limit ?? 20; if (!Number.isSafeInteger(limit) || limit < 1 || limit > 100) throw new Error("Invalid replay page size."); if (query.afterId) replayId(query.afterId); if (query.recordingId) positiveId(query.recordingId);
    const selected: string[] = []; for await (const id of replayJobIds(root)) { if (query.afterId && id <= query.afterId) continue; if (query.recordingId) { const { state } = await readReplay(root, id, brainId); if ((state.view.currentRecordingId ?? state.view.recordingId) !== query.recordingId) continue; } selected.push(id); selected.sort(); if (selected.length > limit) selected.pop(); }
    const entries = []; for (const id of selected) entries.push(await this.get(brainId, id)); return { entries, ...(selected.length === limit ? { nextAfterId: selected.at(-1) } : {}) }; }
  async manifest(brainId: string, id: string): Promise<ImaginationAssemblyManifestSummary> { const root = await this.root(brainId); return (await readReplay(root, replayId(id), brainId)).intent.manifest; }
  async manifestPage(brainId: string, id: string, query: { section: "fragments" | "tracks" | "spans"; afterIndex?: number; limit?: number }) { const root = await this.root(brainId), { intent } = await readReplay(root, replayId(id), brainId), directory = join(root, REPLAY_ROOT, "plans", intent.spoolId);
    await verifyReplaySpool(directory, intent.spoolSeal); const spool = new ReplaySpool(directory, intent.manifest); try { return spool.page(query.section, query.afterIndex, query.limit); } finally { spool.close(); } }
  async cancel(request: { brainId: string; assemblyId: string; expectedRevision: number }): Promise<ImaginationAssembly> { const root = await this.root(request.brainId), { state } = await readReplay(root, replayId(request.assemblyId), request.brainId);
    if (request.expectedRevision !== state.view.revision) throw new Error("Replay changed before cancellation; refresh its actual revision."); const active = this.active.get(`${request.brainId}:${request.assemblyId}`);
    if (active) { active.controller.abort(new Error("This exact replay assembly was cancelled; committed source and segments remain.")); await active.promise; }
    else if (!['complete','cancelled'].includes(state.view.state)) { state.view.state = "cancelled"; await this.save(root, state); } return this.get(request.brainId, request.assemblyId); }
  async resume(request: { brainId: string; assemblyId: string; expectedRevision: number }): Promise<ImaginationAssembly> { if (this.disposed||this.blocked.has(request.brainId)||this.snapshotHolds.has(request.brainId)) throw new Error("Replay owner is closing, checkpointing or being recovered."); const root = await this.root(request.brainId), { state } = await readReplay(root, replayId(request.assemblyId), request.brainId);
    if (state.view.state !== "dormant" || state.view.revision !== request.expectedRevision || this.active.has(`${request.brainId}:${request.assemblyId}`)) throw new Error("Only this exact dormant replay can be explicitly resumed.");
    state.view.state = "queued"; await this.save(root, state); this.start(request.brainId, request.assemblyId); return state.view; }
  async release(brainId: string, keepBlocked = true): Promise<void> { this.blocked.add(brainId); for (const row of [...this.preparingPlans.values(), ...this.admissions.values()]) if (row.brainId === brainId) row.controller.abort(new Error("This owner suspended codec preparation; no neural work was admitted."));
    await Promise.all([...this.preparingPlans.values(), ...this.admissions.values()].filter(row => row.brainId === brainId).map(row => row.promise.catch(() => undefined)));
    for (const job of this.active.values()) if (job.brainId === brainId) job.pause = true; await Promise.all([...this.active.values()].filter(job => job.brainId === brainId).map(job => job.promise));
    this.boundaries.get(brainId)?.(); this.boundaries.delete(brainId); for (const [key, value] of this.plans) if (value.brainId === brainId) this.plans.delete(key); if (!keepBlocked && !this.disposed) this.blocked.delete(brainId); }
  activate(brainId: string): void { this.blocked.delete(brainId); }
  async dispose(): Promise<void> { this.disposed = true; for (const id of new Set([...this.boundaries.keys(),...this.active.values()].map(row=>typeof row==="string"?row:row.brainId))) await this.release(id); for (const unregister of this.boundaries.values()) unregister(); this.boundaries.clear(); this.plans.clear(); }
}
