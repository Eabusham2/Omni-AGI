import { mkdir, mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EvolutionController, type EvolutionToolExecutor } from "../src/main/evolutionController";
import type { BrainRepository } from "../src/main/brainRepository";
import type { EngineRequestContext, EngineSupervisor } from "../src/main/engineSupervisor";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";
import type { EvolutionStartRequest, ToolPermissionLevel } from "../src/shared/types";

const brainId = "synthetic-controller-owner";
const sha = (char: string) => char.repeat(64);
const syntheticCandidate = "b".repeat(32);
const requiredChecks = ["integrity", "architectureCompatible", "resources", "ternaryCoverage", "objectiveNonRegression", "capabilityRetention",
  "neuralRetention", "changed", "heldOutToken", "heldOutModality", "heldOutTool", "heldOutResources", "statisticallySupportedImprovement"];
const request: EvolutionStartRequest = { brainId, candidateKind: "neural", objective: "Synthetic no-model routing test", texts: [" first \n", "", "second"],
  epochs: 2, learningRate: 0.0001, latentReplay: false, objectives: ["retain exact objective text "], recursive: false };
const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function fixture(permission: ToolPermissionLevel, loadedTransport = true) {
  const directory = await mkdtemp(join(tmpdir(), "omni-native-candidate-controller-")); roots.push(directory);
  const root = join(directory, brainId); await mkdir(root);
  const origin = new AbortController(); let current = true;
  const authority: HostActionOriginAuthority = { revision: "current-private-runtime-lease", signal: origin.signal,
    assertCurrent: vi.fn(async () => { if (!current) throw new Error("Module lease revoked"); }) };
  const repository = { brainDirectory: () => root, get: vi.fn(async () => ({ id: brainId, concepts: {}, journal: [], updatedAt: "2026-10-15T12:00:00Z",
    toolPermissions: [{ toolId: "source.self-modify", level: permission }] })), recentConversationEvidence: vi.fn(async () => []) } as unknown as BrainRepository;
  const tools = { execute: vi.fn(), cancel: vi.fn(() => 0) } as unknown as EvolutionToolExecutor;
  const proposal = { id: syntheticCandidate, status: "ready", benchmarkSha256: sha("c"), baselineManifestSha256: sha("d"),
    parentParameterChecksum: sha("e"), parentStateChecksum: sha("f"), candidateParameterChecksum: sha("1"), candidateStateChecksum: sha("2"), candidateDiffSha256: sha("3"),
    sourceIds: [], objectives: [] };
  const evaluated = { candidateId: syntheticCandidate, status: "evaluated", passed: true, checks: Object.fromEntries(requiredChecks.map(name => [name, true])),
    benchmarkSha256: sha("c"), evaluationSha256: sha("4"), failures: [] };
  const promoted = { promoted: true, candidateId: syntheticCandidate, stateChecksumBefore: sha("f"), stateChecksumAfter: sha("2"), rollbackAvailable: true };
  const requestNative = vi.fn(async (method: string, _params: Record<string, unknown>, _timeout: number, _signal: AbortSignal, context: EngineRequestContext) => {
    await context.beforeDispatch?.();
    if (method === "evolution.propose") return proposal;
    if (method === "evolution.evaluate") return evaluated;
    if (method === "evolution.promote") return promoted;
    throw new Error(`Unscoped/unsupported synthetic RPC ${method}`);
  });
  const genericRequest = vi.fn(async () => { throw new Error("Generic request could silently start a worker"); });
  const controller = new EvolutionController(repository, tools, { request: genericRequest,
    ...(loadedTransport ? { requestLoadedNativeCandidate: requestNative } : {}) } as unknown as Pick<EngineSupervisor, "request">);
  return { root, controller, requestNative, genericRequest, proposal, evaluated, promoted, authority, origin, tools, setCurrent: (value: boolean) => { current = value; },
    archive: async () => JSON.parse(await readFile(join(root, "evolution", "archive.json"), "utf8")) as { runs: unknown[]; candidates: Array<Record<string, unknown>> } };
}

describe("authorized native candidate Controller hooks (synthetic RPC only)", () => {
  it("retains real run/candidate IDs, exact declared native arrays and scoped RPC without source commands or list-triggered Load", async () => {
    const f = await fixture("ask"), ids = vi.fn();
    const run = await f.controller.startAuthorized(request, { authority: f.authority, onRunCreated: ids });
    expect(run).toMatchObject({ state: "experimenting", candidateKind: "neural" });
    expect(ids).toHaveBeenCalledWith(run.id, run.candidateIds[0]);
    expect(f.requestNative.mock.calls.map(call => call[0])).toEqual(["evolution.propose"]);
    expect(f.requestNative.mock.calls[0]?.[1]).toMatchObject({ jobId: run.id, texts: request.texts, objectives: request.objectives, epochs: 2, learningRate: 0.0001 });
    expect(f.requestNative.mock.calls[0]?.[3]).toBe(f.origin.signal);
    expect(f.requestNative.mock.calls[0]?.[4]).toMatchObject({ requestId: run.id, owner: "evolution", brainId });
    expect(f.tools.execute).not.toHaveBeenCalled();
    expect(f.genericRequest).not.toHaveBeenCalled();
    expect(await f.controller.nativeToolRun(brainId, run.id)).toMatchObject({ id: run.id });
    expect(f.requestNative).toHaveBeenCalledTimes(1);
  });

  it("rechecks exact live authority after evaluation before automatic promotion", async () => {
    const f = await fixture("auto");
    f.requestNative.mockImplementation(async method => {
      if (method === "evolution.propose") return f.proposal;
      if (method === "evolution.evaluate") { f.setCurrent(false); return f.evaluated; }
      throw new Error(`Unexpected RPC ${method}`);
    });
    const run = await f.controller.startAuthorized(request, { authority: f.authority });
    expect(run).toMatchObject({ state: "failed", error: "Module lease revoked" });
    expect(f.requestNative.mock.calls.map(call => call[0])).toEqual(["evolution.propose", "evolution.evaluate"]);
    expect(f.requestNative.mock.calls[1]?.[3]).toBe(f.origin.signal);
    expect(f.requestNative.mock.calls[1]?.[4]).toMatchObject({ requestId: `${run.id}.evaluate` });
    expect((await f.archive()).candidates[0]).toMatchObject({ evaluations: [{ passed: true }], state: "failed" });
  });

  it("keeps an actual promoted synthetic receipt after late revoke and persists exact pending ordinary reassessment/error", async () => {
    const f = await fixture("full"), deferredEvents = vi.fn(), learned = vi.fn(async () => undefined);
    f.controller.setRecursiveReassessmentHandler(learned);
    f.requestNative.mockImplementation(async method => {
      if (method === "evolution.propose") return f.proposal;
      if (method === "evolution.evaluate") return f.evaluated;
      if (method === "evolution.promote") { f.setCurrent(false); return f.promoted; }
      throw new Error(`Unexpected RPC ${method}`);
    });
    const run = await f.controller.startAuthorized({ ...request, recursive: true }, { authority: f.authority, deferReassessment: deferredEvents });
    expect(run.state).toBe("promoted"); expect(learned).not.toHaveBeenCalled(); expect(deferredEvents).toHaveBeenCalledTimes(1);
    const candidateId = run.candidateIds[0]!;
    let candidate = (await f.archive()).candidates[0]!;
    expect(candidate).toMatchObject({ state: "promoted", pendingReassessment: { format: "omni-observed-evolution-reassessment", brainId,
      parentCandidateId: candidateId, workerCandidateId: syntheticCandidate, configuration: { ...request, recursive: true } } });
    await expect(f.controller.deliverNativeToolReassessment(brainId, candidateId, { loadedOnly: true, signal: f.origin.signal, assertCurrent: () => f.authority.assertCurrent() })).rejects.toThrow("Module lease revoked");
    candidate = (await f.archive()).candidates[0]!;
    expect(candidate).toMatchObject({ state: "promoted", recursiveError: "Observed reassessment remains pending: Module lease revoked", pendingReassessment: { parentCandidateId: candidateId } });
    expect(learned).not.toHaveBeenCalled();
    expect(f.requestNative.mock.calls.map(call => call[0])).toEqual(["evolution.propose", "evolution.evaluate", "evolution.promote"]);
  });

  it("fails honestly without a loaded-only transport rather than falling back to generic cold-start requests", async () => {
    const f = await fixture("ask", false);
    const run = await f.controller.startAuthorized(request, { authority: f.authority });
    expect(run).toMatchObject({ state: "failed", error: "The loaded-only native candidate transport is unavailable; no worker will be started." });
    expect(f.genericRequest).not.toHaveBeenCalled(); expect(f.requestNative).not.toHaveBeenCalled();
  });
});
