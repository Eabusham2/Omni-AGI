import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SparseSubstrateTransferService, SparseTransferBoundaryError, type SparseSubstrateNativeBridge,
  type SparseSubstrateNativeHead, type SparseSubstrateTransferDependencies } from "../src/main/sparseSubstrateTransferService";
import { SparseSubstrateTransferStore, validateSparseTransferDatabaseFile } from "../src/main/sparseSubstrateTransferStore";
import type { SparseSubstrateBundle, SparseSubstrateJob, SparseSubstrateNativeReceipt, SparseSubstratePlan } from "../src/shared/sparseSubstrateTransfer";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";

const roots: string[] = [], services: SparseSubstrateTransferService[] = [];
afterEach(async () => { await Promise.all(services.splice(0).map(service => service.shutdown())); await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const sha = (value: string | Buffer): string => createHash("sha256").update(value).digest("hex");
const hashA = "a".repeat(64), hashB = "b".repeat(64), hashC = "c".repeat(64), brainId = "sparse-owner";

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-sparse-host-contract-")); roots.push(root);
  const controller = new AbortController(), queue: Array<() => Promise<void>> = [];
  let authorized = true, loaded = true, applies = 0, exports = 0, head: SparseSubstrateNativeHead | null = null;
  const authority: HostActionOriginAuthority = { revision: "current-permission-and-module-revision", signal: controller.signal,
    assertCurrent: async () => { if (!authorized) throw new Error("exact runtime/permission revision revoked"); } };
  const manifest = { format: "omni-sparse-substrate-transfer", version: 1, scope: "native-sparse-assemblies", sourceBrainId: brainId,
    sourceGenerationSha256: hashA, sourceMetadataSha256: hashB, rows: 1, cortexStateIncluded: false, sourceAuthorityIncluded: false };
  const descriptor = (body: Buffer, value: object) => ({ manifestJson: JSON.stringify(value), manifestSha256: sha(JSON.stringify(value)), sha256: sha(body), bytes: body.length, rows: 1 });
  const native: SparseSubstrateNativeBridge = {
    inspect: async id => ({ format: "omni-sparse-substrate-transfer", version: 1, scope: "native-sparse-assemblies", stateScope: "committed", brainId: id, generationSha256: hashA, rows: [], hasMore: false }),
    export: async (_id, _selection, destination) => { exports++; const body = Buffer.from("explicit synthetic packed artifact, not a learned state"); await writeFile(destination, body); return descriptor(body, manifest); },
    validateBundle: async (_id, path, expectedSha) => { const body = await readFile(path); if (sha(body) !== expectedSha) throw new Error("selected source changed"); return descriptor(body, manifest); },
    prepare: async (_id, _bundle, bundleSha, destination, mode) => { const value = { format: "omni-sparse-substrate-plan", version: 1, scope: "native-sparse-assemblies", brainId,
      bundleSha256: bundleSha, mode, rows: 1, additions: 0, replacements: 1, identical: 0, conflicts: 0,
      targetGenerationSha256: hashA, selectedPreimagesSha256: hashB, changesSha256: hashC };
      const body = Buffer.from(JSON.stringify(value)); await writeFile(destination, body); return { ...value, ...descriptor(body, value) }; },
    prepareRollback: async (_id, _plan, _planSha, receiptJson, receiptSha256, destination) => { const receipt = JSON.parse(receiptJson) as SparseSubstrateNativeReceipt;
      const value = { format: "omni-sparse-substrate-plan", version: 1, scope: "native-sparse-assemblies", brainId,
        bundleSha256: receipt.bundleSha256, mode: "replace-exact", operation: "rollback", rollbackOf: { receiptSha256 },
        rows: 1, additions: 0, replacements: 1, identical: 0, conflicts: 0, targetGenerationSha256: hashC,
        selectedPreimagesSha256: hashC, changesSha256: hashB };
      const body = Buffer.from(JSON.stringify(value)); await writeFile(destination, body); return { ...value, ...descriptor(body, value) }; },
    head: async () => head,
    apply: async (_id, planPath, input, signal) => { signal.throwIfAborted(); applies++; const plan = JSON.parse((await readFile(planPath)).toString()) as SparseSubstratePlan;
      const body = { format: "omni-sparse-substrate-applied", version: 1, scope: "native-sparse-assemblies", brainId,
        jobId: input.jobId, requestSha256: input.requestSha256, planSha256: input.planSha256, bundleSha256: plan.bundleSha256,
        selectedPreimagesSha256: plan.selectedPreimagesSha256, changesSha256: plan.changesSha256,
        appliedRows: 1, committedAt: new Date().toISOString(), weightsAveraged: false, cortexParametersChanged: false, originChanged: false };
      const receiptJson = JSON.stringify(body); head = { receiptJson, receiptSha256: sha(receiptJson), generationSha256: hashC };
      return { ...body, ...head, duplicate: false } as SparseSubstrateNativeReceipt; }
  };
  const dependencies: SparseSubstrateTransferDependencies = { brainDirectory: async () => root, assertOwner: async id => { expect(id).toBe(brainId); },
    checkResources: async () => undefined, checkDisk: async () => undefined, native,
    assertLoadedBoundary: async () => { if (!loaded) throw new SparseTransferBoundaryError("no implicit Load", "boundary-required"); },
    deferWrite: async (_id, operation, signal) => new Promise((resolve, reject) => {
      const abort = (): void => reject(signal.reason); signal.addEventListener("abort", abort, { once: true });
      queue.push(async () => { try { signal.throwIfAborted(); resolve(await operation()); } catch (error) { reject(error); } finally { signal.removeEventListener("abort", abort); } });
      if (signal.aborted) abort();
    }) };
  const service = new SparseSubstrateTransferService(dependencies); services.push(service);
  const execute = (action: string, args: Record<string, unknown>, id: string, context?: { chatTurnId?: string }) => service.execute(brainId, action, args, id, new AbortController().signal, authority, context);
  const prepare = async () => {
    const bundle = await execute("export", { selection: { all: true } }, "export") as SparseSubstrateBundle;
    const plan = await execute("prepare-import", { bundleId: bundle.id, mode: "replace-exact" }, "preview") as SparseSubstratePlan;
    return { bundle, plan };
  };
  const status = async (id: string) => { const store = await SparseSubstrateTransferStore.open(root, brainId); try { return store.job(id); } finally { store.close(); } };
  return { root, service, dependencies, native, execute, prepare, status, authority, controller, queue,
    applies: () => applies, exports: () => exports, revoke: () => { authorized = false; }, setLoaded: (value: boolean) => { loaded = value; } };
}

describe("native sparse transfer detached host contracts (synthetic native bridge)", () => {
  it("returns a durable job before the busy writer, then applies once and archives exact receipt", async () => {
    const f = await fixture(), { plan } = await f.prepare();
    const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "apply") as SparseSubstrateJob;
    expect(job.state).toBe("queued"); expect(f.applies()).toBe(0); expect(f.queue).toHaveLength(1);
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("complete"));
    const repeated = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "apply") as SparseSubstrateJob;
    expect(repeated.receipt?.planSha256).toBe(plan.sha256); expect(f.applies()).toBe(1);
    await validateSparseTransferDatabaseFile(join(f.root, "substrate-transfers", "workspace.sqlite3"), brainId);
  });

  it("rejects concurrent same request ID with changed action/operands", async () => {
    const f = await fixture(); let release!: () => void; const gate = new Promise<void>(resolve => { release = resolve; });
    const original = f.native.export; let markEntered!: () => void;
    const entered = new Promise<void>(resolve => { markEntered = resolve; });
    f.native.export = async (...args) => { markEntered(); await gate; return original(...args); };
    const first = f.execute("export", { selection: { all: true } }, "same-request");
    const settled = first.then(() => ({ kind: "complete" as const }), error => ({ kind: "failed" as const, error }));
    try {
      const boundary = await Promise.race([entered.then(() => ({ kind: "entered" as const })), settled]);
      if (boundary.kind === "failed") throw boundary.error;
      if (boundary.kind !== "entered") throw new Error("The first export completed before reaching its held native entry.");
      await expect(f.execute("export", { selection: { neurons: ["different"] } }, "same-request")).rejects.toThrow(/changed reviewed operands/);
      await expect(f.execute("prepare-import", { bundleId: "different", mode: "append" }, "same-request")).rejects.toThrow(/changed reviewed operands/);
    } finally {
      release();
      await first;
    }
    expect(f.exports()).toBe(1);
  });

  it("revokes a queued origin before dispatch without a permission grant or import", async () => {
    const f = await fixture(), { plan } = await f.prepare();
    const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "revoked") as SparseSubstrateJob;
    f.revoke(); await f.service.authorityChanged();
    expect(f.applies()).toBe(0); expect(["cancelled", "needs-review"]).toContain((await f.status(job.id)).state);
    const store = await SparseSubstrateTransferStore.open(f.root, brainId);
    try { const raw = String(store.db.prepare("SELECT payload FROM jobs WHERE id=?").get(job.id)?.payload);
      expect(raw).not.toContain("assertCurrent"); expect(raw).not.toContain("approvalToken"); expect(JSON.parse(raw).liveAuthorityPersisted).toBe(false); }
    finally { store.close(); }
  });

  it("waits for explicit Load notification without hidden loading or tight polling", async () => {
    const f = await fixture(), { plan } = await f.prepare(); f.setLoaded(false);
    const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "wait-loaded") as SparseSubstrateJob;
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("boundary-required"));
    expect(f.applies()).toBe(0); expect(f.queue).toHaveLength(0);
    f.setLoaded(true); await f.service.boundaryAvailable(brainId); expect(f.queue).toHaveLength(1);
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("complete")); expect(f.applies()).toBe(1);
  });

  it("turns restarted pending authority dormant and never redispatches it", async () => {
    const f = await fixture(), { plan } = await f.prepare(); const now = new Date().toISOString(), store = await SparseSubstrateTransferStore.open(f.root, brainId);
    const job: SparseSubstrateJob = { format: "omni-sparse-substrate-job", version: 1, scope: "native-sparse-assemblies", id: "old-job", brainId,
      originalBrainId: brainId, requestId: "old-request", requestSha256: hashA, planId: plan.id, planSha256: plan.sha256, authorizationRevision: "historical-not-authority",
      state: "queued", createdAt: now, updatedAt: now, revision: 1, liveAuthorityPersisted: false };
    try { store.createJob(job); } finally { store.close(); }
    const recovered = new SparseSubstrateTransferService(f.dependencies); services.push(recovered);
    const current = await recovered.execute(brainId, "status", { jobId: job.id }, "status", new AbortController().signal, f.authority) as SparseSubstrateJob;
    expect(current.state).toBe("dormant"); await recovered.boundaryAvailable(brainId); expect(f.queue).toHaveLength(0); expect(f.applies()).toBe(0);
  });

  it("registers only a currently scoped selected bundle and preserves unverified source attribution", async () => {
    const f = await fixture(), sourcePath = join(f.root, "selected-source"), body = Buffer.from("explicit synthetic selected native-shape source"); await writeFile(sourcePath, body);
    let selectedCalls = 0;
    f.dependencies.selectedBundle = async (owner, turn, attachment) => { expect([owner, turn, attachment]).toEqual([brainId, "actual-turn", "selected-file"]); selectedCalls++;
      return { path: sourcePath, sha256: sha(body), bytes: body.length }; };
    await expect(f.execute("register-bundle", { attachmentId: "selected-file" }, "no-task")).rejects.toThrow(/actual main-owned/);
    const bundle = await f.execute("register-bundle", { attachmentId: "selected-file" }, "selected", { chatTurnId: "actual-turn" }) as SparseSubstrateBundle;
    expect(bundle.sourceBinding).toBe("selected-native-shape-source-unverified"); expect(selectedCalls).toBe(2);
    const page = await f.execute("read", { bundleId: bundle.id, offset: 0, maxBytes: 9 }, "read") as { dataBase64: string; nextOffset: number };
    expect(Buffer.from(page.dataBase64, "base64").equals(body.subarray(0, 9))).toBe(true); expect(page.nextOffset).toBe(9);
  });

  it("makes a reviewed inverse preview rather than directly undoing completed state", async () => {
    const f = await fixture(), { plan } = await f.prepare(); const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "original") as SparseSubstrateJob;
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("complete"));
    const inverse = await f.execute("prepare-rollback", { jobId: job.id }, "inverse-preview") as SparseSubstratePlan;
    expect(inverse.operation).toBe("rollback"); expect(inverse.conflicts).toBe(0); expect(f.applies()).toBe(1);
    const inverseJob = await f.execute("apply", { planId: inverse.id, reviewSha256: inverse.sha256 }, "inverse-apply") as SparseSubstrateJob;
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(inverseJob.id)).state).toBe("complete")); expect(f.applies()).toBe(2);
  });

  it("retains uncertain published outcomes without replay when acknowledgement fails", async () => {
    const f = await fixture(), { plan } = await f.prepare();
    f.native.apply = async () => { throw new SparseTransferBoundaryError("published but worker released", "reload-required", null); };
    const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "lost-ack") as SparseSubstrateJob;
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("uncertain"));
    const repeated = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "lost-ack") as SparseSubstrateJob;
    expect(repeated.state).toBe("uncertain"); expect(f.queue).toHaveLength(0);
  });

  it("verifies immutable bytes once across pages and re-verifies after exact identity changes", async () => {
    const f = await fixture(), { bundle } = await f.prepare(); let verifications = 0;
    f.dependencies.checkResources = async (_id, _bytes, purpose) => { if (purpose === "verify exact sparse bundle bytes") verifications++; };
    await f.execute("read", { bundleId: bundle.id, offset: 0, maxBytes: 8 }, "page-one");
    await f.execute("read", { bundleId: bundle.id, offset: 8, maxBytes: 8 }, "page-two");
    expect(verifications).toBe(1);
    const path = join(f.root, bundle.relativePath), original = await readFile(path), changed = Buffer.from(original); changed[0] = changed[0]! ^ 1;
    await writeFile(path, changed);
    await expect(f.execute("read", { bundleId: bundle.id, offset: 0 }, "page-changed")).rejects.toThrow(/content hash changed/); expect(verifications).toBe(2);
    await writeFile(path, original); await f.execute("read", { bundleId: bundle.id, offset: 0 }, "page-restored"); expect(verifications).toBe(3);
  });

  it("preserves an exact native cancelled-unapplied proof instead of inventing uncertain mutation", async () => {
    const f = await fixture(), { plan } = await f.prepare();
    f.native.apply = async () => { throw new SparseTransferBoundaryError("exact native pre-mutation safe boundary", "cancelled-unapplied", false); };
    const job = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "safe-cancel") as SparseSubstrateJob;
    await f.queue.shift()!(); await vi.waitFor(async () => expect((await f.status(job.id)).state).toBe("cancelled"));
    const repeated = await f.execute("apply", { planId: plan.id, reviewSha256: plan.sha256 }, "safe-cancel") as SparseSubstrateJob;
    expect(repeated.state).toBe("cancelled"); expect(f.queue).toHaveLength(0); expect(f.applies()).toBe(0);
  });
});
