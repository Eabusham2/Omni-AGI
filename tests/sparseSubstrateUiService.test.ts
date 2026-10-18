import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SparseSubstrateTransferService, SparseTransferBoundaryError,
  type SparseSubstrateNativeBridge } from "../src/main/sparseSubstrateTransferService";
import { SparseSubstrateTransferStore } from "../src/main/sparseSubstrateTransferStore";
import { SPARSE_SUBSTRATE_SCOPE, type SparseSubstrateJob, type SparseSubstratePlan,
  type SparseSubstrateUiEvent } from "../src/shared/sparseSubstrateTransfer";

const roots: string[] = [];
const digest = (value: string): string => createHash("sha256").update(value).digest("hex");
function canonical(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(",")}}`;
}
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-sparse-ui-")); roots.push(root);
  const brainId = "brain-a", planId = "plan-a", bundleSha256 = "b".repeat(64), planSha256 = "c".repeat(64);
  const manifestJson = JSON.stringify({ format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
    rows: 2, fingerprintConflicts: 1, unprovenRemovalRows: 0 });
  const plan: SparseSubstratePlan = { format: "omni-sparse-substrate-plan", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
    id: planId, brainId, bundleId: "bundle-a", bundleSha256, mode: "replace-exact", operation: "import",
    sha256: planSha256, bytes: 128, manifestJson, manifestSha256: digest(manifestJson), relativePath: "substrate-transfers/plans/plan-a.sqlite3",
    rows: 2, additions: 1, replacements: 1, identical: 0, conflicts: 0, targetGenerationSha256: "d".repeat(64),
    selectedPreimagesSha256: "e".repeat(64), changesSha256: "f".repeat(64) };
  const store = await SparseSubstrateTransferStore.open(root, brainId); const now = "2026-10-15T00:00:00.000Z";
  store.putArtifact("plan", plan);
  const dormantSeed: SparseSubstrateJob = { format: "omni-sparse-substrate-job", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
    id: "seed-job", brainId, originalBrainId: brainId, requestId: "seed-request", requestSha256: "1".repeat(64),
    planId, planSha256, authorizationRevision: "private-authority-revision", state: "boundary-required",
    createdAt: now, updatedAt: now, revision: 1, liveAuthorityPersisted: false };
  store.createJob(dormantSeed); store.close();

  let loaded = false;
  const queue: Array<() => Promise<void>> = [];
  const native: SparseSubstrateNativeBridge = {
    inspect: async id => ({ format: "omni-sparse-substrate-transfer", version: 1, scope: SPARSE_SUBSTRATE_SCOPE,
      stateScope: "committed", brainId: id, generationSha256: "9".repeat(64), rows: [], hasMore: false }),
    export: async () => { throw new Error("unused"); }, validateBundle: async () => { throw new Error("unused"); },
    prepare: async () => { throw new Error("unused"); }, prepareRollback: async () => { throw new Error("unused"); },
    head: async () => null,
    apply: async (id, _path, input) => {
      const body = { format: "omni-sparse-substrate-applied" as const, version: 1 as const, scope: SPARSE_SUBSTRATE_SCOPE,
        brainId: id, jobId: input.jobId, requestSha256: input.requestSha256, planSha256: input.planSha256,
        bundleSha256, selectedPreimagesSha256: plan.selectedPreimagesSha256, changesSha256: plan.changesSha256,
        appliedRows: 2, committedAt: "2026-10-15T00:00:01.000Z", generationSha256: "8".repeat(64), duplicate: false,
        weightsAveraged: false as const, cortexParametersChanged: false as const, originChanged: false as const };
      const receiptJson = canonical(body); return { ...body, receiptJson, receiptSha256: digest(receiptJson) };
    }
  };
  const service = new SparseSubstrateTransferService({ brainDirectory: async () => root,
    assertOwner: async id => { if (id !== brainId) throw new Error("foreign owner"); }, checkResources: async () => undefined,
    checkDisk: async () => undefined, native,
    deferWrite: (_id, operation, signal) => new Promise((resolve, reject) => { queue.push(async () => {
      try { resolve(await operation()); } catch (error) { reject(error); } }); signal.addEventListener("abort", () => reject(signal.reason), { once: true }); }),
    assertLoadedBoundary: async () => { if (!loaded) throw new SparseTransferBoundaryError("load explicitly", "boundary-required"); } });
  const authorityController = new AbortController(), authority = { revision: "runtime-revision", signal: authorityController.signal, assertCurrent: async () => undefined };
  return { root, brainId, plan, service, queue, authority, setLoaded: (value: boolean) => { loaded = value; } };
}

describe("sparse substrate typed UI service", () => {
  it("projects owner-safe jobs/plans and trusted Stop without exposing raw authority or receipt JSON", async () => {
    const value = await fixture(), events: SparseSubstrateUiEvent[] = [];
    value.service.onUiEvent(event => { throw new Error(`observer cannot mutate ${event.job.id}`); });
    value.service.onUiEvent(event => { events.push(event); });
    const page = await value.service.uiJobs(value.brainId);
    expect(page).toMatchObject({ total: 1, entries: [{ id: "seed-job", brainId: value.brainId, state: "dormant", phase: "historical",
      liveAuthorityAvailable: false }] });
    expect("authorizationRevision" in page.entries[0]!).toBe(false);
    expect(events.at(-1)).toMatchObject({ kind: "job", job: { id: "seed-job", state: "dormant", revision: 2 } });
    const plan = await value.service.uiPlan(value.brainId, value.plan.id);
    expect(plan).toMatchObject({ id: value.plan.id, mode: "replace-exact", operation: "import", rows: 2,
      additions: 1, replacements: 1, conflicts: 0, fingerprintConflicts: 1, unprovenRemovalRows: 0 });
    expect("manifestJson" in plan).toBe(false); expect("relativePath" in plan).toBe(false);
    const stopped = await value.service.userStop(value.brainId, "seed-job");
    expect(stopped).toMatchObject({ state: "cancelled", phase: "terminal" });
    await expect(value.service.userStop("other-brain", "seed-job")).rejects.toThrow(/foreign owner/i);
  });

  it("publishes queued, boundary, applying and terminal revisions while listener failures remain inert", async () => {
    const value = await fixture(); await value.service.uiJobs(value.brainId);
    const events: Array<Extract<SparseSubstrateUiEvent, { kind: "job" }>> = [];
    value.service.onUiEvent(() => { throw new Error("presentation failure"); });
    value.service.onUiEvent(event => { events.push(event); });
    const job = await value.service.execute(value.brainId, "apply", { planId: value.plan.id, reviewSha256: value.plan.sha256 },
      "apply-request", new AbortController().signal, value.authority,
      { chatTurnId: "turn-a", actionEventId: "action-a" }) as SparseSubstrateJob;
    expect(events.at(-1)).toMatchObject({ job: { id: job.id, state: "queued", revision: 1,
      chatTurnId: "turn-a", actionEventId: "action-a" } });
    await value.queue.shift()!();
    await vi.waitFor(async () => expect((await value.service.uiJob(value.brainId, job.id)).state).toBe("boundary-required"));
    expect(events.some(event => event.job.id === job.id && event.job.state === "boundary-required")).toBe(true);
    value.setLoaded(true); await value.service.boundaryAvailable(value.brainId); await value.queue.shift()!();
    await vi.waitFor(async () => expect((await value.service.uiJob(value.brainId, job.id)).state).toBe("complete"));
    expect(events.filter(event => event.job.id === job.id).map(event => event.job.state)).toEqual([
      "queued", "boundary-required", "applying", "complete"
    ]);
    const complete = await value.service.uiJob(value.brainId, job.id);
    expect(complete.receipt).toMatchObject({ receiptSha256: expect.stringMatching(/^[a-f0-9]{64}$/), appliedRows: 2 });
    expect(JSON.stringify(complete)).not.toContain("receiptJson");
  });
});
