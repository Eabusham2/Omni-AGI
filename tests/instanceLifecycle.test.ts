import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { InstanceLifecycleController } from "../src/main/instanceLifecycleController";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { ChatActionController } from "../src/main/chatActionController";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { NativeInstanceStatus } from "../src/shared/instanceLifecycleApi";
import type { InstanceResumeFrontier } from "../src/shared/instanceLifecycle";
import { DEFAULT_CONFIG, type ChatResult, type StructuredAction } from "../src/shared/types";

const roots: string[] = [];
const digest = (text: string) => createHash("sha256").update(text).digest("hex");
const frontier: InstanceResumeFrontier = { checkpointId: "c".repeat(64), turnId: "turn-fixture", inputSha256: digest("input"),
  lastStreamSequence: 4, sampledTokens: 2, publishedTokens: 1, phase: "native-text-token-boundary", exactSupportedResume: true };
async function root() { const path = await mkdtemp(join(tmpdir(), "omni-lifecycle-host-")); roots.push(path); return path; }
afterEach(async () => { for (const path of roots.splice(0)) await rm(path, { recursive: true, force: true }); });
async function controllerFixture() {
  const path = await root();
  let native: NativeInstanceStatus = { brainId: "fixture", learnedRuntimeResident: false, minimumContextTokens: 8, reconnect: [] };
  let allowed = true;
  const inspect = vi.fn(async () => structuredClone(native));
  const load = vi.fn(async () => { native.learnedRuntimeResident = true; });
  const suspend = vi.fn(async () => { native = { ...native, learnedRuntimeResident: false, unloaded: true, resume: frontier }; return structuredClone(native); });
  const dependencies = { inspect, load, suspend, fit: vi.fn(async () => ({ brainId: "fixture", checkedAt: new Date().toISOString(), allowed,
    minimumContextTokens: 8, blockers: allowed ? [] : ["Current pressure"] })) };
  return { path, dependencies, native, load, suspend, inspect, setAllowed: (value: boolean) => { allowed = value; },
    controller: new InstanceLifecycleController(path, dependencies) };
}

describe("explicit instance lifecycle host contract", () => {
  it("defaults Auto-load off and persists only the local preference without loading", async () => {
    const f = await controllerFixture();
    expect(await f.controller.status("fixture")).toMatchObject({ phase: "unloaded", autoLoad: false });
    await f.controller.setAutoLoad("fixture", true);
    const restarted = new InstanceLifecycleController(f.path, f.dependencies);
    expect(await restarted.status("fixture")).toMatchObject({ phase: "unloaded", autoLoad: true });
    expect(f.load).not.toHaveBeenCalled(); expect(() => restarted.requireLoaded("fixture")).toThrow(/Load/);
  });
  it("keeps rejected fit recoverable and never treats it as loaded", async () => {
    const f = await controllerFixture(); f.setAllowed(false);
    expect(await f.controller.load("fixture")).toMatchObject({ phase: "unloaded", error: { code: "RESOURCE_FIT", recoverable: true } });
    expect(f.load).not.toHaveBeenCalled();
    f.setAllowed(true); expect(await f.controller.load("fixture")).toMatchObject({ phase: "loaded" });
    expect(f.controller.isLoaded("fixture")).toBe(true);
  });
  it("preserves an unsupported historical frontier without loading or pretending resource changes can repair its runtime seal", async () => {
    const f = await controllerFixture();
    f.native.resume = frontier; f.native.resumeCompatibility = { compatible: false, reason: "The old frontier has no exact runtime seal." };
    expect(await f.controller.status("fixture")).toMatchObject({ phase: "suspended", resumeCompatibility: { compatible: false } });
    expect(await f.controller.load("fixture")).toMatchObject({ phase: "error", error: { code: "LOAD_FAILED", message: "The old frontier has no exact runtime seal." } });
    expect(f.load).not.toHaveBeenCalled(); expect(() => f.controller.requireLoaded("fixture", frontier.turnId)).toThrow(/runtime seal/);
  });
  it("parks the same waiter over Unload and wakes it only after Load acknowledgement", async () => {
    const f = await controllerFixture(); await f.controller.load("fixture");
    expect(await f.controller.unload("fixture")).toMatchObject({ phase: "suspended", learnedRuntimeResident: false, resume: frontier });
    let continued = false; const pending = f.controller.waitUntilLoaded("fixture", undefined, frontier.turnId).then(() => { continued = true; });
    await Promise.resolve(); expect(continued).toBe(false);
    await f.controller.load("fixture", frontier.checkpointId); await pending; expect(continued).toBe(true);
    expect(f.suspend).toHaveBeenCalledOnce();
  });
  it("reserves the loaded frontier for its original turn before allowing other queued learning", async () => {
    const f = await controllerFixture(); await f.controller.load("fixture"); await f.controller.unload("fixture");
    let otherRan = false;
    const other = f.controller.waitUntilLoaded("fixture").then(() => { otherRan = true; });
    const exact = f.controller.waitUntilLoaded("fixture", undefined, frontier.turnId);
    await f.controller.load("fixture", frontier.checkpointId); await exact;
    expect(otherRan).toBe(false); expect(() => f.controller.requireLoaded("fixture")).toThrow(/saved turn/);
    f.controller.completed("fixture"); await other; expect(otherRan).toBe(true);
  });
  it("keeps merged inherited frontiers executable without releasing unrelated root mutation", async () => {
    const f = await controllerFixture(), resumeBackground = vi.fn(), releaseDispatch = vi.fn();
    const controller = new InstanceLifecycleController(f.path, { ...f.dependencies, resumeBackground, releaseDispatch });
    f.native.inheritedExecutions = { format: "omni-connected-execution-queue", pending: 1, rootMutationHeld: true,
      activeExecutionId: null, minimumContextTokens: 12, entries: [{ id: "a".repeat(64), turnId: "ancestral-turn",
        parentId: "parent", status: "queued", backend: "cpu", publishedText: "Retained prefix" }] };
    expect(await controller.status("fixture")).toMatchObject({ phase: "suspended", inheritedExecutions: { pending: 1 } });
    await controller.load("fixture");
    expect(controller.canRun("fixture")).toBe(false);
    expect(controller.canRun("fixture", "another-turn")).toBe(false);
    expect(controller.canRun("fixture", "ancestral-turn")).toBe(true);
    expect(resumeBackground).not.toHaveBeenCalled(); expect(releaseDispatch).not.toHaveBeenCalled();
    let rootReady = false; const pending = controller.waitUntilLoaded("fixture").then(() => { rootReady = true; });
    await Promise.resolve(); expect(rootReady).toBe(false);
    f.native.inheritedExecutions.entries[0]!.status = "complete";
    f.native.inheritedExecutions.pending = 0; f.native.inheritedExecutions.rootMutationHeld = false;
    await controller.status("fixture"); await pending;
    expect(rootReady).toBe(true); expect(resumeBackground).toHaveBeenCalledOnce(); expect(releaseDispatch).toHaveBeenCalledOnce();
  });
  it("does not report a warm unsupported boundary as unloaded", async () => {
    const f = await controllerFixture(); await f.controller.load("fixture");
    f.suspend.mockImplementation(async () => ({ ...f.native, learnedRuntimeResident: true, unloaded: false,
      waitingFor: { kind: "external-effect", label: "Reconnect the current handle explicitly" } }));
    expect(await f.controller.unload("fixture")).toMatchObject({ phase: "suspending", learnedRuntimeResident: true });
    expect(f.controller.isLoaded("fixture")).toBe(false);
  });
  it("invalidates a worker-confirmed released owner without inventing an Unload checkpoint or automatic reload", async () => {
    const f = await controllerFixture(); await f.controller.load("fixture"); f.native.learnedRuntimeResident = false;
    expect(await f.controller.confirmedNativeOwnerRelease("fixture")).toMatchObject({ phase: "unloaded", learnedRuntimeResident: false,
      error: { code: "NATIVE_OWNER_RELEASED", recoverable: true } });
    expect(f.controller.isLoaded("fixture")).toBe(false); expect(() => f.controller.requireLoaded("fixture")).toThrow("Load");
    expect(f.load).toHaveBeenCalledOnce(); expect(f.suspend).not.toHaveBeenCalled();
    await f.controller.load("fixture"); expect(f.controller.isLoaded("fixture")).toBe(true);
  });
  it("preserves order for overlapping explicit load and unload and rejects stale checkpoints", async () => {
    const f = await controllerFixture(); let release!: () => void;
    f.load.mockImplementation(async () => { await new Promise<void>((resolve) => { release = resolve; }); f.native.learnedRuntimeResident = true; });
    const loading = f.controller.load("fixture"); await vi.waitFor(() => expect(f.load).toHaveBeenCalledOnce());
    const unloading = f.controller.unload("fixture"); release(); await loading;
    expect((await unloading).phase).toBe("suspended");
    expect(await f.controller.load("fixture", "other-checkpoint")).toMatchObject({ phase: "error", error: { code: "LOAD_FAILED" } });
  });
  it("Stop's abort rejects a parked waiter without pretending that Unload cancelled it", async () => {
    const f = await controllerFixture(), abort = new AbortController();
    const waiting = f.controller.waitUntilLoaded("fixture", abort.signal);
    abort.abort(new Error("Explicit Stop")); await expect(waiting).rejects.toThrow("Explicit Stop");
    expect(f.suspend).not.toHaveBeenCalled();
  });
  it("does not let a stale status read overwrite a newer acknowledged load", async () => {
    const f = await controllerFixture(); await f.controller.status("fixture");
    let release!: (value: NativeInstanceStatus) => void;
    f.inspect.mockImplementationOnce(() => new Promise((resolve) => { release = resolve; }));
    const stale = f.controller.status("fixture"); await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    await f.controller.load("fixture");
    release({ brainId: "fixture", learnedRuntimeResident: false, minimumContextTokens: 8, reconnect: [] });
    expect((await stale).phase).toBe("loaded"); expect(f.controller.isLoaded("fixture")).toBe(true);
  });
  it.each([false, true])("keeps non-initialization jobs queued without loading and handles explicit cancel=%s locally", async (cancel) => {
    const f = await controllerFixture();
    const learn = vi.fn(async () => ({ state: "complete", coverage: { complete: true } }));
    const service = { instanceLifecycle: f.controller, waitUntilInstanceLoaded: f.controller.waitUntilLoaded.bind(f.controller),
      resumeFullMergeIngestion: learn } as unknown as BrainService;
    const engine = Object.assign(new EventEmitter(), { cancelRequest: vi.fn(), request: vi.fn() });
    const jobs = new RuntimeJobManager(service, engine as unknown as EngineSupervisor);
    const job = jobs.resumeFullMergeIngestion({ brainId: "fixture", pendingId: "d".repeat(64), sourceName: "owned-source" }, "/fixture/source.jsonl");
    expect(jobs.list("fixture")[0]).toMatchObject({ state: "queued", label: "Waiting for Load or the retained turn's completion" });
    expect(learn).not.toHaveBeenCalled(); expect(engine.request).not.toHaveBeenCalled();
    if (cancel) {
      expect((await jobs.cancel(job.id)).state).toBe("cancelled");
      expect(engine.cancelRequest).not.toHaveBeenCalled(); expect(engine.request).not.toHaveBeenCalled();
    } else {
      await f.controller.load("fixture");
      await vi.waitFor(() => expect(jobs.list("fixture")[0]?.state).toBe("complete"));
      expect(learn).toHaveBeenCalledOnce();
    }
  });
});

describe("same-turn BrainService continuation with a protocol-only worker", () => {
  it("keeps copied native turn IDs independent across instance controllers", async () => {
    const release = new Map<string, () => void>();
    const service = { chat: vi.fn(async (brainId: string) => {
      await new Promise<void>(resolve => { release.set(brainId, resolve); });
      return { brain: { messages: [] }, humanMessage: { id: "human", role: "human", content: "input", createdAt: "now" },
        brainMessage: { id: "reply", role: "brain", content: brainId, createdAt: "now" }, trace: { id: "trace" } } as unknown as ChatResult;
    }) };
    const controller = new ChatActionController(service, { execute: vi.fn(), cancel: vi.fn(() => 0) }, { start: vi.fn() });
    const original = controller.send("original", "input", undefined, "retained-native-turn");
    const copy = controller.send("copy", "input", undefined, "retained-native-turn");
    await vi.waitFor(() => expect(service.chat).toHaveBeenCalledTimes(2));
    expect(controller.cancel("unrelated", "retained-native-turn")).toBe(0);
    release.get("copy")!(); expect((await copy).brainMessage.content).toBe("copy");
    release.get("original")!(); expect((await original).brainMessage.content).toBe("original");
  });
  it.each(["resume", "stop", "detach"] as const)("gates implicit load, releases the writer and preserves Resume/Stop/recovery-detach=%s", async (mode) => {
    const path = await root(), original = new BrainRepository(join(path, "brains")); await original.initialize();
    let brain = await original.create({ ...DEFAULT_CONFIG, name: "Tiny lifecycle fixture", onlineLearning: false });
    brain = { ...brain, readiness: { ...brain.readiness, state: "ready" } };
    const repository = { root: original.root, brainDirectory: (id: string) => original.brainDirectory(id),
      get: vi.fn(async () => structuredClone(brain)), prepareConfig: (config: typeof brain.config) => original.prepareConfig(config),
      save: vi.fn(async (next: typeof brain) => { brain = structuredClone(next); return structuredClone(brain); }) } as unknown as BrainRepository;
    const native: NativeInstanceStatus = { brainId: brain.id, learnedRuntimeResident: false, minimumContextTokens: 8, reconnect: [] };
    let suspendFirst!: () => void; const first = new Promise<void>((resolve) => { suspendFirst = resolve; });
    const requests: string[] = [], tokens: string[] = [];
    const engine = {
      on: vi.fn(), instanceStatus: vi.fn(async () => structuredClone(native)), claimForeground: vi.fn(async () => undefined),
      suspendChat: vi.fn(async () => { native.resume = frontier; suspendFirst(); return { requested: true }; }),
      request: vi.fn(async (method: string) => { requests.push(method);
        if (method === "load") native.learnedRuntimeResident = true;
        if (method === "suspend_instance") { native.learnedRuntimeResident = false; return { ...native, unloaded: true }; }
        if (method === "stop_suspended_chat") { native.resume = undefined; return { stopped: true, brainId: brain.id, turnId: frontier.turnId, checkpointId: frontier.checkpointId }; }
        if (method === "chat_receipt") return { format: "omni-chat-turn-receipt-query", formatVersion: 1, brainId: brain.id,
          turnId: frontier.turnId, inputSha256: digest("input"), inferenceCount: 0, committed: false, turnCommitted: false };
        return {}; }),
      requestStream: vi.fn(async (method: string, params: Record<string, unknown>, onEvent: (event: unknown) => void,
        _timeout: number, _signal: AbortSignal, _stream: string, _priority: string, context: { onTransition(event: unknown): void }) => {
        context.onTransition({ method, state: "running", requestId: frontier.turnId, owner: "chat", label: "Chat", brainId: brain.id, turnId: frontier.turnId });
        onEvent({ type: "chat-token", brainId: brain.id, streamId: frontier.turnId, sequence: method === "chat" ? 4 : 5, data: { delta: method === "chat" ? "A" : "B" } });
        if (method === "chat") { await first; return { suspended: true, turnCommitted: false, brainId: brain.id,
          turnId: frontier.turnId, inputSha256: digest("input"), resume: frontier }; }
        expect(params).not.toHaveProperty("input"); expect(params.checkpointId).toBe(frontier.checkpointId);
        native.resume = undefined; return { text: "AB" };
      })
    };
    const service = new BrainService(repository, engine as unknown as EngineSupervisor);
    const releaseShutdown = service.beginRuntimeShutdown();
    expect(() => service.loadInstance(brain.id)).toThrow(/checkpointing before exit/);
    await expect(service.preflightStart(brain.id)).rejects.toThrow(/checkpointing before exit/);
    releaseShutdown();
    await expect(service.chat(brain.id, "input")).rejects.toThrow(/Load this instance/);
    expect(engine.request).not.toHaveBeenCalled();
    expect((await service.loadInstance(brain.id)).phase).toBe("loaded");
    let settled = false;
    const abort = new AbortController();
    const turn = service.chat(brain.id, "input", abort.signal, (event) => { if (event.type === "chat-token") tokens.push(event.delta); }, frontier.turnId).then((result) => { settled = true; return result; });
    await vi.waitFor(() => expect(engine.requestStream).toHaveBeenCalledOnce());
    expect((await service.unloadInstance(brain.id)).phase).toBe("suspended");
    expect(settled).toBe(false);
    await service.updateConfig(brain.id, { ...brain.config, name: "Offline changed", systemRamMode: "manual", systemRamSharePercent: 44 });
    expect(requests).toEqual(["load", "suspend_instance"]);
    await expect(service.updateConfig(brain.id, { ...brain.config, contextWindowTokens: 7 })).rejects.toThrow(/eight|frontier/);
    if (mode === "detach") {
      const detached = expect(turn).rejects.toMatchObject({ code: "INSTANCE_CONTINUATION_DETACHED", brainId: brain.id, turnId: frontier.turnId });
      await service.detachInstanceHostContinuations(brain.id); await detached;
      expect(requests).not.toContain("stop_suspended_chat"); expect(engine.requestStream).toHaveBeenCalledOnce();
      expect(native.resume).toEqual(frontier); expect(native.learnedRuntimeResident).toBe(false);
      await service.loadInstance(brain.id, frontier.checkpointId);
      expect(engine.requestStream).toHaveBeenCalledOnce(); // No retired attachment wakes and replays after Load.
      return;
    }
    if (mode === "stop") {
      const stopped = expect(turn).rejects.toThrow(/aborted/);
      abort.abort(); await stopped;
      expect(requests).toContain("stop_suspended_chat");
      expect(native.resume).toBeUndefined(); expect(engine.requestStream).toHaveBeenCalledOnce();
      expect((await service.instanceStatus(brain.id)).resume).toBeUndefined();
      return;
    }
    expect((await service.loadInstance(brain.id, frontier.checkpointId)).phase).toBe("loaded");
    const result = await turn;
    expect(engine.requestStream.mock.calls.map(([method]) => method)).toEqual(["chat", "resume_chat"]);
    expect(tokens).toEqual(["A", "B"]); expect(result.brainMessage.content).toBe("AB");
    expect(result.brain.messages.filter((message) => message.role === "human")).toHaveLength(1);
    expect(result.brain.config.name).toBe("Offline changed");
  });
  it("cold reattachment suppresses prior external proposals and skips capability reconnection", async () => {
    const action: StructuredAction = { kind: "tool", source: "brain", actionId: "a".repeat(32), toolId: "web.search", action: "search", arguments: { query: "already handled" } };
    const result = { brain: { messages: [] }, humanMessage: { id: "human", role: "human", content: "input", createdAt: "now" },
      brainMessage: { id: "reply", role: "brain", content: "AB", createdAt: "now" }, trace: { id: "trace" }, proposedActions: [action] } as unknown as ChatResult;
    const service = { requireInstanceLoaded: vi.fn(), instanceResumePresentation: vi.fn(async () => ({ brainId: "fixture", frontier,
      input: "input", publishedText: "A", hostAttached: false, priorActions: [action] })), chat: vi.fn(async () => result) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) }, controller = new ChatActionController(service, tools, { start: vi.fn() });
    const prepare = vi.fn(async () => []); controller.setCapabilityPreparation(prepare);
    await controller.resumeInstanceTurn("fixture", frontier.checkpointId);
    expect(service.chat).toHaveBeenCalledWith("fixture", "input", expect.any(AbortSignal), expect.any(Function), frontier.turnId,
      undefined, undefined, undefined, frontier);
    expect(tools.execute).not.toHaveBeenCalled(); expect(tools.cancel).not.toHaveBeenCalled(); expect(prepare).not.toHaveBeenCalled();
  });
  it("lets explicit Stop settle a host effect parked behind Unload without executing it", async () => {
    const action: StructuredAction = { kind: "tool", source: "brain", actionId: "b".repeat(32), toolId: "web.search", action: "search", arguments: { query: "pending only" } };
    const result = { brain: { messages: [] }, humanMessage: { id: "human", role: "human", content: "input", createdAt: "now" },
      brainMessage: { id: "reply", role: "brain", content: "A", createdAt: "now" }, trace: { id: "trace" }, proposedActions: [action] } as unknown as ChatResult;
    const service = { chat: vi.fn(async () => result), waitUntilInstanceLoaded: vi.fn((_id: string, signal?: AbortSignal) =>
      new Promise<void>((_resolve, reject) => { signal?.addEventListener("abort", () => reject(signal.reason), { once: true }); })) };
    const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) }, controller = new ChatActionController(service, tools, { start: vi.fn() });
    const pending = controller.send("fixture", "input", undefined, "parked-host-effect");
    const rejected = pending.catch((error: unknown) => error);
    await vi.waitFor(() => expect(service.waitUntilInstanceLoaded).toHaveBeenCalledOnce());
    await expect(controller.cancelAndWait("fixture", "parked-host-effect")).resolves.toBe(1);
    await rejected; expect(tools.execute).not.toHaveBeenCalled();
  });
});
