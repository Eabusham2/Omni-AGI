import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainWriteCoordinator } from "../src/main/brainWriteCoordinator";
import {
  NativeCandidateToolBoundaryError, NativeCandidateToolService,
  type NativeCandidateToolDependencies
} from "../src/main/nativeCandidateToolService";
import type { AuthorizedNativeCandidateOperation, EvolutionRunRecord } from "../src/main/evolutionController";
import { ToolExecutor } from "../src/main/toolExecutor";
import { nativeCandidateToolRequest, type NativeCandidateToolJob } from "../src/shared/nativeCandidateTool";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";
import type { RecursiveEvolutionReassessment, ToolInvocation } from "../src/shared/types";

const brainId = "native-tool-fixture";
const request = { candidateKind: "neural", objective: "Improve the selected evidence", texts: [" first\n", "second"], recursive: false };
function deferred<T>() { let resolve!: (value: T) => void; let reject!: (reason: unknown) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; }
async function eventually(check: () => boolean): Promise<void> {
  for (let index = 0; index < 80; index++) { if (check()) return; await new Promise<void>(resolve => setImmediate(resolve)); }
  throw new Error("Expected bounded fake-host state did not settle.");
}
function fakeRun(id = "real-controller-run", state: EvolutionRunRecord["state"] = "awaiting-review"): EvolutionRunRecord {
  return { id, brainId, state, recursive: false, generation: 0, candidateIds: ["real-controller-candidate"], objective: request.objective,
    hypothesis: "Synthetic test controller outcome", limitations: [], evaluatorVersion: 1, evaluatorPolicySha256: "a".repeat(64),
    createdAt: "2026-10-15T12:00:00Z", updatedAt: "2026-10-15T12:00:00Z" };
}
function fixture() {
  const jobs = new Map<string, NativeCandidateToolJob>(), runs = new Map<string, EvolutionRunRecord>();
  const coordinator = new BrainWriteCoordinator(), notices: NativeCandidateToolJob[] = [];
  const origin = new AbortController(); let current = true, loaded = true, writerOwned = false;
  const authority: HostActionOriginAuthority = { revision: "exact-private-module-lease", signal: origin.signal,
    assertCurrent: vi.fn(async () => { if (!current) throw new Error("Exact module disabled"); }) };
  const controller: NativeCandidateToolDependencies["controller"] = {
    nativeToolJob: vi.fn(async (_brain, id) => { const job = jobs.get(id); if (!job) throw new Error("Missing fake job"); return structuredClone(job); }),
    nativeToolJobForRequest: vi.fn(async (owner, id) => structuredClone([...jobs.values()].find(job => job.brainId === owner && job.requestId === id))),
    nativeToolRun: vi.fn(async (_brain, id) => structuredClone(runs.get(id))),
    putNativeToolJob: vi.fn(async (_brain, job) => { jobs.set(job.id, structuredClone(job)); return structuredClone(job); }),
    updateNativeToolJob: vi.fn(async (_brain, id, patch) => { const prior = jobs.get(id); if (!prior) throw new Error("Missing fake job");
      const next = { ...prior, ...structuredClone(patch), revision: prior.revision + 1 }; jobs.set(id, next); return structuredClone(next); }),
    startAuthorized: vi.fn(async (_request, operation) => { await operation.authority.assertCurrent();
      const run = fakeRun(); await operation.onRunCreated?.(run.id, run.candidateIds[0]!); runs.set(run.id, run); return run; }),
    deliverNativeToolReassessment: vi.fn(async () => undefined),
    stop: vi.fn(async (_brain, runId, scope) => { await scope?.assertCurrent(); const run = fakeRun(runId, "stopped"); runs.set(runId, run); return run; })
  };
  const dependencies: NativeCandidateToolDependencies = {
    controller, deferWrite: (owner, task, signal) => coordinator.run(owner, async () => {
      writerOwned = true; try { return await task(); } finally { writerOwned = false; }
    }, signal),
    assertOwner: vi.fn(async () => undefined), assertLoadedBoundary: vi.fn(async () => { if (!loaded) throw new NativeCandidateToolBoundaryError("waiting-load", "Instance is explicitly unloaded"); }),
    checkResources: vi.fn(async () => undefined), changed: job => { notices.push(job); }
  };
  const service = new NativeCandidateToolService(dependencies);
  return { service, controller, dependencies, jobs, runs, notices, coordinator, authority, origin,
    setCurrent: (value: boolean) => { current = value; }, setLoaded: (value: boolean) => { loaded = value; }, writerOwned: () => writerOwned,
    send: (action: string, args: Record<string, unknown>, id: string) => service.execute(brainId, action, args, id, new AbortController().signal, authority) };
}
const active: NativeCandidateToolService[] = [];
afterEach(async () => { await Promise.all(active.splice(0).map(service => service.shutdown())); });
function tracked() { const value = fixture(); active.push(value.service); return value; }

describe("native candidate strict operands and source routing (no model)", () => {
  it("preserves exact native evidence and typed architecture operands, rejecting incompatible or missing fields", () => {
    expect(nativeCandidateToolRequest(brainId, request)).toMatchObject({ texts: [" first\n", "second"], recursive: false });
    const geometry = { token: [{ path: "/selected/tokens", records: 3 }], tool: [{ path: "/selected/tools", records: 4 }],
      modality: [{ path: "/selected/audio.wav", kind: "audio", conditionText: "Exact held out words" }] };
    expect(nativeCandidateToolRequest(brainId, { candidateKind: "architecture", objective: "geometry", architectureChange: { mutation: "resize-width", dModel: 16, nHeads: 4 }, geometryHoldouts: geometry })).toMatchObject({ geometryHoldouts: geometry });
    for (const args of [{ ...request, sourceEdits: [] }, { ...request, tests: ["unit"] }, { ...request, candidateKind: "mystery" },
      { candidateKind: "architecture", objective: "geometry" }, { candidateKind: "architecture", objective: "geometry", addExperts: 1, architectureChange: { mutation: "grow-experts", addExperts: 2 } },
      { candidateKind: "architecture", objective: "geometry", addExperts: 1, geometryHoldouts: { ...geometry, token: [{ path: "/selected/tokens" }] } },
      { ...request, provenance: { silentlyDropped: undefined } }]) expect(() => nativeCandidateToolRequest(brainId, args)).toThrow();
  });

  it("routes only explicit native kinds to the typed service and preserves Git source/default routes", async () => {
    const tools = Object.create(ToolExecutor.prototype) as ToolExecutor;
    // Representative queued receipt only: this case tests dispatch, not proposal execution.
    const nativeResult: NativeCandidateToolJob = { format: "omni-native-candidate-tool-job", version: 1,
      id: "route-only-job", brainId, requestId: "route-request", requestSha256: "a".repeat(64), candidateKind: "neural",
      arguments: request, request: nativeCandidateToolRequest(brainId, request), authorizationRevision: "route-only-fixture",
      state: "queued", createdAt: "2026-10-15T12:00:00Z", updatedAt: "2026-10-15T12:00:00Z", revision: 1, liveAuthorityPersisted: false };
    const native = vi.fn(async (..._args: unknown[]) => structuredClone(nativeResult)), git = vi.fn(async (..._args: unknown[]) => ({ kind: "git" }));
    tools.setNativeCandidateToolService({ execute: native });
    type Dispatch = { dispatch(invocation: ToolInvocation, ...args: unknown[]): Promise<unknown>; sourceEvolution: typeof git };
    const privateTools = tools as unknown as Dispatch; privateTools.sourceEvolution = git;
    const signal = new AbortController().signal, authority: HostActionOriginAuthority = { revision: "route", signal, assertCurrent: async () => undefined };
    const invoke = (action: string, args: Record<string, unknown>) => privateTools.dispatch({ brainId, toolId: "source.self-modify", action, arguments: args },
      signal, "route-request", undefined, "full", "host-event", undefined, "actual-chat-turn", undefined, undefined, authority);
    for (const candidateKind of ["neural", "data", "substrate", "architecture"]) expect(await invoke("propose", { candidateKind, objective: "native" })).toEqual(nativeResult);
    await invoke("native-status", { jobId: "job" }); await invoke("native-stop", { jobId: "job" });
    expect(native).toHaveBeenCalledTimes(6); expect(native.mock.calls[0]?.[5]).toBe(authority); expect(git).not.toHaveBeenCalled();
    await invoke("propose", { objective: "source", sourceEdits: [] }); await invoke("propose", { candidateKind: "source", objective: "source" });
    expect(git).toHaveBeenCalledTimes(2);
    await expect(invoke("propose", { candidateKind: "unsupported" })).rejects.toThrow("cannot fall through");
    (tools as unknown as { nativeCandidateToolService?: unknown }).nativeCandidateToolService = undefined;
    await expect(invoke("propose", request)).rejects.toThrow("cannot fall back"); expect(git).toHaveBeenCalledTimes(2);
  });
});

describe("detached native candidate jobs (fake controller/FIFO only)", () => {
  it("returns a durable queued handle without waiting for the chat writer, with exact idempotent operands and real run ID", async () => {
    const f = tracked(), gate = deferred<void>(), acquired = deferred<void>();
    const foreground = f.coordinator.run(brainId, async () => { acquired.resolve(); await gate.promise; }); await acquired.promise;
    const job = await f.send("propose", request, "once"); expect(job.state).toBe("queued"); expect(f.controller.startAuthorized).not.toHaveBeenCalled();
    expect(await f.send("propose", request, "once")).toMatchObject({ id: job.id });
    await expect(f.send("propose", { ...request, texts: ["changed"] }, "once")).rejects.toThrow("changed operands");
    gate.resolve(); await foreground; await eventually(() => f.jobs.get(job.id)?.state === "complete");
    expect(f.jobs.get(job.id)).toMatchObject({ runId: "real-controller-run", candidateId: "real-controller-candidate", liveAuthorityPersisted: false, arguments: request, run: { state: "awaiting-review" } });
    const called = vi.mocked(f.controller.startAuthorized).mock.calls[0]!;
    expect(called[0].texts).toEqual(request.texts); expect(called[0].provenance).toMatchObject({ source: "permissioned-native-candidate-tool" });
    expect(f.controller.startAuthorized).toHaveBeenCalledTimes(1);
  });

  it("binds concurrent duplicate IDs before admission and rejects changed operands rather than returning the wrong job", async () => {
    const f = tracked(), gate = deferred<void>(); vi.mocked(f.controller.nativeToolJobForRequest).mockImplementation(async () => { await gate.promise; return undefined; });
    const first = f.send("propose", request, "concurrent");
    await eventually(() => vi.mocked(f.controller.nativeToolJobForRequest).mock.calls.length === 1);
    await expect(f.send("propose", { ...request, objective: "different" }, "concurrent")).rejects.toThrow("concurrent native request changed operands");
    gate.resolve(); await first;
  });

  it("rechecks the live module lease on FIFO acquisition and prevents work after revocation", async () => {
    const f = tracked(), gate = deferred<void>(), acquired = deferred<void>();
    const foreground = f.coordinator.run(brainId, async () => { acquired.resolve(); await gate.promise; }); await acquired.promise;
    const job = await f.send("propose", request, "revoked"); f.setCurrent(false); gate.resolve(); await foreground;
    await eventually(() => f.jobs.get(job.id)?.state === "failed"); expect(f.controller.startAuthorized).not.toHaveBeenCalled();
    expect(f.jobs.get(job.id)?.error).toContain("Exact module disabled");
  });

  it("holds an unloaded job until an explicit boundary event without implicit Load or a persisted grant", async () => {
    const f = tracked(); f.setLoaded(false);
    const job = await f.send("propose", request, "unloaded"); await eventually(() => f.jobs.get(job.id)?.state === "waiting-load");
    expect(f.controller.startAuthorized).not.toHaveBeenCalled(); f.setLoaded(true); await f.service.boundaryAvailable(brainId);
    await eventually(() => f.jobs.get(job.id)?.state === "complete"); expect(f.controller.startAuthorized).toHaveBeenCalledTimes(1);
  });

  it("restarted/copied pending job evidence is dormant and never reconstructs runtime authority", async () => {
    const f = tracked(); f.setLoaded(false);
    const job = await f.send("propose", request, "saved"); await eventually(() => f.jobs.get(job.id)?.state === "waiting-load");
    const restarted = new NativeCandidateToolService(f.dependencies); active.push(restarted);
    expect(await restarted.execute(brainId, "propose", request, "saved", new AbortController().signal, f.authority)).toMatchObject({ id: job.id, state: "dormant" });
    await restarted.boundaryAvailable(brainId); expect(f.controller.startAuthorized).not.toHaveBeenCalled();
  });

  it("module Stop propagates to the same deferred native operation signal and preserves unknown outcome honestly", async () => {
    const f = tracked(), entered = deferred<HostActionOriginAuthority>();
    vi.mocked(f.controller.startAuthorized).mockImplementation(async (_request, operation) => {
      await operation.onRunCreated?.("real-controller-run", "real-controller-candidate"); entered.resolve(operation.authority);
      return new Promise((_resolve, reject) => operation.authority.signal.addEventListener("abort", () => reject(operation.authority.signal.reason), { once: true }));
    });
    const job = await f.send("propose", request, "origin-stop"); const running = await entered.promise;
    f.origin.abort(new Error("Module stopped")); await eventually(() => f.jobs.get(job.id)?.state === "uncertain");
    expect(running.signal.aborted).toBe(true); expect(f.jobs.get(job.id)).toMatchObject({ runId: "real-controller-run", state: "uncertain" });
    expect(f.controller.startAuthorized).toHaveBeenCalledTimes(1);
  });

  it("explicit native Stop uses only the controller-created current run, never arbitrary or copied run IDs", async () => {
    const f = tracked(), entered = deferred<void>();
    vi.mocked(f.controller.startAuthorized).mockImplementation(async (_request, operation) => {
      await operation.onRunCreated?.("real-controller-run", "real-controller-candidate"); entered.resolve();
      return new Promise((_resolve, reject) => operation.authority.signal.addEventListener("abort", () => reject(operation.authority.signal.reason), { once: true }));
    });
    const job = await f.send("propose", request, "requested-stop"); await entered.promise;
    await f.send("native-stop", { jobId: job.id }, "stop-action"); await eventually(() => f.jobs.get(job.id)?.state === "cancelled");
    expect(f.controller.stop).toHaveBeenCalledWith(brainId, "real-controller-run", expect.any(Object));
    await expect(f.send("native-stop", { jobId: job.id, runId: "unrelated" }, "forged-stop")).rejects.toThrow("only an exact jobId");
  });

  it("reassessment is delivered through the normal writer only after the candidate FIFO releases", async () => {
    const f = tracked(); let reassessmentInsideWriter = false;
    vi.mocked(f.controller.startAuthorized).mockImplementation(async (_request, operation: AuthorizedNativeCandidateOperation) => {
      expect(f.writerOwned()).toBe(true); const run = fakeRun(undefined, "promoted"); run.recursive = true;
      await operation.onRunCreated?.(run.id, run.candidateIds[0]!); f.runs.set(run.id, run);
      await operation.deferReassessment?.({ parentCandidateId: run.candidateIds[0]! } as RecursiveEvolutionReassessment); return run;
    });
    vi.mocked(f.controller.deliverNativeToolReassessment).mockImplementation(async (_brain, _candidate, context) => {
      reassessmentInsideWriter = f.writerOwned();
      await f.coordinator.run(brainId, async () => { await context.assertCurrent(); expect(context.loadedOnly).toBe(true); });
    });
    const job = await f.send("propose", { ...request, recursive: true }, "nonreentrant");
    await eventually(() => f.jobs.get(job.id)?.reassessmentPending === false);
    expect(reassessmentInsideWriter).toBe(false); expect(f.controller.deliverNativeToolReassessment).toHaveBeenCalledTimes(1);
    expect(f.jobs.get(job.id)).toMatchObject({ state: "complete", run: { state: "promoted" } });
  });

  it("late source revocation does not erase a real promoted run; its unlearned reassessment remains explicit", async () => {
    const f = tracked();
    vi.mocked(f.controller.startAuthorized).mockImplementation(async (_request, operation) => {
      const run = fakeRun(undefined, "promoted"); run.recursive = true; f.runs.set(run.id, run);
      await operation.onRunCreated?.(run.id, run.candidateIds[0]!);
      await operation.deferReassessment?.({ parentCandidateId: run.candidateIds[0]! } as RecursiveEvolutionReassessment);
      f.setCurrent(false); return run;
    });
    vi.mocked(f.controller.deliverNativeToolReassessment).mockImplementation(async (_brain, _candidate, context) => { await context.assertCurrent(); });
    const job = await f.send("propose", { ...request, recursive: true }, "late-revoke");
    await eventually(() => Boolean(f.jobs.get(job.id)?.reassessmentError));
    expect(f.jobs.get(job.id)).toMatchObject({ state: "complete", run: { state: "promoted" }, reassessmentPending: true, reassessmentError: "Exact module disabled" });
  });
});
