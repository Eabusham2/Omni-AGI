/** Actual small SQLite/filesystem custody; receipts are labeled synthetic data. */
import { afterEach, describe, expect, it } from "vitest";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import type { LocalTeacherJob, LocalTeacherResultReceipt, LocalTeacherSelection } from "../src/shared/localTeacher";
import { LocalTeacherStore, type LocalTeacherSessionSeal } from "../src/main/localTeacherStore";
import { hashLocalTeacherFile } from "../src/main/localTeacherFiles";
import { localTeacherCanonical, localTeacherHash } from "../src/main/localTeacherFormat";
import { localTeacherDirectory, writeTeacherDialogueSource, writeTeacherReceipt } from "../src/main/localTeacherArtifacts";
import { mergeLocalTeacherWorkspace, validateLocalTeacherWorkspace } from "../src/main/localTeacherPortability";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { localTeacherPlan } from "../src/main/localTeacherPlan";
import { registerLocalTeacherBoundary, withLocalTeacherSnapshot } from "../src/main/localTeacherWorkspaceBoundary";
import { recoverLocalTeacherSeals } from "../src/main/localTeacherRecovery";
const roots: string[] = [], sourceId = "77777777-7777-4777-8777-777777777777", targetId = "66666666-6666-4666-8666-666666666666";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
const temp = async () => { const root = await mkdtemp(join(tmpdir(), "omni-local-teacher-fixture-")); roots.push(root); return root; };
const generation = { contextTokens: 32, maxOutputTokens: 8, threads: 1, temperature: .7, seed: 1, timeoutMs: 0, maxOutputBytes: 4096 };
async function fixture() {
  const root = await temp(), dependencies = await temp();
  await writeFile(join(dependencies, "llama-server"), "synthetic fixture executable identity: NOT RUN"); await writeFile(join(dependencies, "tiny.gguf"), "synthetic GGUF bytes: NOT INFERRED");
  const runtime = await hashLocalTeacherFile(join(dependencies, "llama-server"), new AbortController().signal, async () => undefined),
    model = await hashLocalTeacherFile(join(dependencies, "tiny.gguf"), new AbortController().signal, async () => undefined), teacherId = localTeacherHash("synthetic teacher"), jobId = localTeacherHash("synthetic job");
  const selection: LocalTeacherSelection = { format: "omni-local-gguf-teacher", version: 1, id: teacherId, brainId: sourceId, originalBrainId: sourceId, originalSelectionId: teacherId,
    createdAt: new Date().toISOString(), runner: "llama-server", currentHostAuthorized: false,
    runtime: { displayName: runtime.displayName, sha256: runtime.sha256, bytes: runtime.bytes, helpSha256: localTeacherHash("synthetic --help"), rawOfflineSupported: true, missingFlags: [] },
    model: { displayName: model.displayName, sha256: model.sha256, bytes: model.bytes, header: { version: 3, tensorCount: "0", metadataCount: "0", metadataEnd: 24,
      architecture: "llama", geometry: { "llama.context_length": 128, "llama.embedding_length": 16, "llama.block_count": 2, "llama.attention.head_count": 2 }, scope: "header-metadata-only", tensorPayloadValidated: false } },
    license: { declaration: "synthetic unverified declaration", source: "user-declared", verified: false } };
  const store = (await LocalTeacherStore.open(root, sourceId))!;
  store.putSelection({ selection, runtime, model, requestId: "synthetic-selection", requestSha256: localTeacherHash("synthetic selection") });
  const now = new Date().toISOString(), job: LocalTeacherJob = { format: "omni-local-teacher-job", version: 1, id: jobId, brainId: sourceId, originalBrainId: sourceId,
    originalJobId: jobId, requestId: "synthetic-request", requestSha256: localTeacherHash("synthetic complete request"), teacherId, generation, epochs: 1, state: "paused",
    revision: 1, createdAt: now, updatedAt: now, promptCount: 1, generatedCount: 0, learnedCount: 0, phase: "synthetic-owned-custody", proof: "attributed-generation-and-native-source-receipts" };
  store.createJob(job, ["  Full private multiline prompt\nsecond line 🌷  "]);
  return { root, store, job, selection };
}
async function response(f: Awaited<ReturnType<typeof fixture>>, sealed = true) {
  const attemptId = localTeacherHash("synthetic attempt"), sessionId = localTeacherHash("synthetic session"), directory = await localTeacherDirectory(f.root, `local-teachers/attempts/${attemptId}`),
    content = "Synthetic teacher response\nexact output 🌷\nNOT a model-quality proof";
  await writeFile(join(directory, "content.bin"), content); await writeFile(join(directory, "empty.stderr"), "");
  const receipt: LocalTeacherResultReceipt = { format: "omni-local-teacher-result", version: 1, attemptId, jobId: f.job.id, index: 0,
    inputSha256: f.store.item(f.job.id, 0).inputSha256, runtimeSha256: f.selection.runtime.sha256, modelSha256: f.selection.model.sha256,
    argvSha256: localTeacherHash("synthetic argv"), environmentPolicySha256: localTeacherHash("synthetic environment policy"), exitCode: null, completionProof: "http-terminal", state: "complete",
    stdout: { relativePath: `local-teachers/attempts/${attemptId}/content.bin`, sha256: localTeacherHash(content), bytes: Buffer.byteLength(content) },
    stderr: { relativePath: `local-teachers/attempts/${attemptId}/empty.stderr`, sha256: localTeacherHash(""), bytes: 0 }, elapsedNanoseconds: "1", outputChannel: "structured-completion",
    transportDecorationsPreserved: false, sessionId, hiddenSystemPrompt: false, foreignWeightsImported: false };
  f.store.updateItem(f.job.id, 0, { state: "generating", attemptId, receipt });
  await writeTeacherReceipt(f.root, `local-teachers/attempts/${attemptId}/result.json`, receipt);
  const seal: LocalTeacherSessionSeal = { format: "omni-local-teacher-custody-seal", version: 1, sessionId, jobId: f.job.id,
    runtimeSha256: f.selection.runtime.sha256, modelSha256: f.selection.model.sha256, at: new Date().toISOString(),
    dependencyCustody: "verified-before-and-after-owned-session", childClosed: true,
    receipts: [{ index: 0, attemptId, sha256: localTeacherHash(localTeacherCanonical(receipt)) }] };
  if (sealed) { await localTeacherDirectory(f.root, `local-teachers/attempts/${sessionId}`); await writeTeacherReceipt(f.root, `local-teachers/attempts/${sessionId}/custody.json`, seal); f.store.sealSession(seal); }
  return { receipt, seal, content };
}

describe("local teacher durable source/copy contract", () => {
  it("reconciles a lost SQL ACK from the exact post-shutdown seal without regenerating or learning", async () => {
    const f = await fixture(); try { const r = await response(f, false); await localTeacherDirectory(f.root, `local-teachers/attempts/${r.seal.sessionId}`);
      await writeTeacherReceipt(f.root, `local-teachers/attempts/${r.seal.sessionId}/custody.json`, r.seal); f.store.update(f.job.id, { state: "running" });
      f.store.recover(); await recoverLocalTeacherSeals(f.store, async () => undefined); expect(f.store.job(f.job.id).state).toBe("paused");
      expect(f.store.job(f.job.id).generatedCount).toBe(1); expect(f.store.job(f.job.id).learnedCount).toBe(0);
      const events = Number(f.store.db.prepare("SELECT COUNT(*) AS n FROM history").get()?.n); await recoverLocalTeacherSeals(f.store, async () => undefined);
      expect(Number(f.store.db.prepare("SELECT COUNT(*) AS n FROM history").get()?.n)).toBe(events); }
    finally { f.store.close(); }
  });
  it("does not call an HTTP terminal candidate generated until exact dependency custody is sealed", async () => {
    const f = await fixture(); try { const r = await response(f, false);
      expect(f.store.job(f.job.id).generatedCount).toBe(0); expect(f.store.hasSealedReceipt(f.job.id, f.store.item(f.job.id, 0))).toBe(false);
      f.store.sealSession(r.seal); expect(f.store.job(f.job.id).generatedCount).toBe(1); expect(f.store.item(f.job.id, 0).state).toBe("generated"); }
    finally { f.store.close(); }
  });
  it("writes exact full typed dialogue/provenance without stripping output or fabricating a training license", async () => {
    const f = await fixture(); try { const r = await response(f), item = f.store.item(f.job.id, 0), source = await writeTeacherDialogueSource(f.root, f.store.job(f.job.id), item, f.selection, r.receipt, new AbortController().signal);
      f.store.updateItem(f.job.id, 0, { source }); const text = await readFile(join(f.root, source.relativePath), "utf8"), record = JSON.parse(text);
      expect(record.messages[0].content).toBe(item.prompt); expect(record.messages[1].content).toBe(r.content); expect(record.teacherProvenance.license.verified).toBe(false);
      expect(record.teacherProvenance.foreignWeightsImported).toBe(false); expect(record.teacherProvenance.verifiedTruth).toBe(false);
      await validateLocalTeacherWorkspace(f.root, sourceId); }
    finally { f.store.close(); }
  });
  it("retains restart progress without executing or accepting a changed prompt retry", async () => {
    const f = await fixture(); try { f.store.update(f.job.id, { state: "running" }); f.store.recover(); expect(f.store.job(f.job.id).state).toBe("paused");
      expect(f.store.createJob(f.job, [f.store.item(f.job.id, 0).prompt]).id).toBe(f.job.id);
      expect(() => f.store.createJob({ ...f.job, requestSha256: localTeacherHash("changed full prompt") }, ["changed"])).toThrow(/different complete/);
      expect(f.store.item(f.job.id, 0).prompt).toContain("second line 🌷"); }
    finally { f.store.close(); }
  });
  it("archives/forks/imports all exact response/job bytes while pending jobs become dormant and dependencies remain external", async () => {
    const f = await fixture(), packageRoot = await temp(), target = await temp(); const r = await response(f);
    f.store.close(); const archive = await snapshotLearningWorkspace(f.root, packageRoot, sourceId); expect(archive).toBeDefined();
    await adoptLearningWorkspace(archive!, target, targetId); await validateLocalTeacherWorkspace(target, targetId);
    const copied = (await LocalTeacherStore.open(target, targetId, false))!; try {
      expect(copied.job(f.job.id).state).toBe("dormant"); expect(copied.selection(f.selection.id).selection.currentHostAuthorized).toBe(false);
      expect(copied.item(f.job.id, 0).inputSha256).toBe(r.receipt.inputSha256); expect(copied.item(f.job.id, 0).receipt).toEqual(r.receipt);
      expect((await readFile(join(target, r.receipt.stdout.relativePath))).toString()).toBe(r.content);
      expect(archive!.files.has("llama-server")).toBe(false); expect(archive!.files.has("tiny.gguf")).toBe(false); }
    finally { copied.close(); }
  });
  it("full merge/recovery retain original custody once and keep operational jobs dormant", async () => {
    const f = await fixture(), stage = await temp(), target = await temp(), packageRoot = await temp(); await response(f); f.store.close();
    const archive = await snapshotLearningWorkspace(f.root, packageRoot, sourceId); await adoptLearningWorkspace(archive!, stage, targetId);
    await mergeLocalTeacherWorkspace(stage, target, targetId, sourceId, async () => undefined);
    const first = (await LocalTeacherStore.open(target, targetId, false))!, count = Number(first.db.prepare("SELECT COUNT(*) AS n FROM history").get()?.n); first.close();
    await mergeLocalTeacherWorkspace(stage, target, targetId, sourceId, async () => undefined);
    const merged = (await LocalTeacherStore.open(target, targetId, false))!; try { expect(merged.jobs()).toHaveLength(1); expect(merged.jobs()[0]!.state).toBe("dormant");
      expect(Number(merged.db.prepare("SELECT COUNT(*) AS n FROM history").get()?.n)).toBe(count);
      expect(merged.item(merged.jobs()[0]!.id, 0).receipt?.jobId).toBe(f.job.id); expect(merged.hasSealedReceipt(merged.jobs()[0]!.id, merged.item(merged.jobs()[0]!.id, 0))).toBe(true); }
    finally { merged.close(); } await validateLocalTeacherWorkspace(target, targetId);
  });
  it("rejects corrupt custody sources and never snapshots an active writer or auto-cancels it", async () => {
    const f = await fixture(); const r = await response(f); f.store.close();
    const unregister = registerLocalTeacherBoundary(f.root, () => true); try { await expect(withLocalTeacherSnapshot(f.root, async () => undefined)).rejects.toThrow(/Pause active/); } finally { unregister(); }
    await writeFile(join(f.root, r.receipt.stdout.relativePath), "corrupt replacement"); await expect(validateLocalTeacherWorkspace(f.root, sourceId)).rejects.toThrow(/hash\/count changed/);
  });
  it("derives recommendations from current family/physical measurements and treats header geometry as unverified", async () => {
    const f = await fixture(); try { const measured = { appFamilyResidentBytes: 8 * 1024 * 1024, selectedRamCeilingBytes: 256 * 1024 * 1024, physicalFreeBytes: 256 * 1024 * 1024, logicalCpuCount: 8 };
      const plan = localTeacherPlan(f.selection, measured); expect(plan.recommendation?.contextTokens).toBe(128); expect(plan.measured).toEqual(measured); expect(plan.declared.tensorPayloadValidated).toBe(false);
      expect(localTeacherPlan({ ...f.selection, model: { ...f.selection.model, header: { ...f.selection.model.header, geometry: {} } } }, measured).fit).toBe("unknown");
      expect(localTeacherPlan(f.selection, { ...measured, appFamilyResidentBytes: measured.selectedRamCeilingBytes }).fit).toBe("blocked"); }
    finally { f.store.close(); }
  });
});
