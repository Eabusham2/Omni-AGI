import { createHash } from "node:crypto";
import { lstat, mkdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { copyFullMergeBlob, type FullMergeAssemblyInput } from "./fullBrainMerge";
import { CompletedActionEvidenceStore, actionEvidenceFileHash, readSavedActionEvidenceJob,
  readUnmanifestedActionEvidence, type CompletedActionEvidenceJob } from "./completedActionEvidence";

const hash = (value: string): string => createHash("sha256").update(value).digest("hex");
const learningUuid = (value: string): string => {
  const digest = hash(value);
  return `${digest.slice(0, 8)}-${digest.slice(8, 12)}-8${digest.slice(13, 16)}-${((parseInt(digest[16]!, 16) & 3) | 8).toString(16)}${digest.slice(17, 20)}-${digest.slice(20, 32)}`;
};
const admit = async (bytes: number): Promise<void> => {
  const reserve = 64 * 1024 * 1024;
  if (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed - reserve < bytes || freemem() - reserve < bytes) {
    throw new Error("Merged completed-result metadata awaits physical transfer RAM.");
  }
};

/** Rehomes learning work only. No tool journal, approvals, or executable effects are adopted. */
export async function adoptFullMergeActionEvidence(input: FullMergeAssemblyInput): Promise<void> {
  const engine = join(input.targetDirectory, "engine");
  const metadata = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
  const regions = new Map<string, string>((metadata.connected_regions?.parents ?? []).map((parent: Record<string, unknown>) =>
    [String(parent.parentId), String(parent.regionId ?? parent.parentId)]));
  const queue: Array<Record<string, any>> = metadata.connected_pending_ingestions ?? [];
  const store = new CompletedActionEvidenceStore(engine, input.targetBrainId, async (bytes, _directory, writing = true) => {
    input.signal?.throwIfAborted(); await admit(bytes * 6); if (writing) await input.reserve(bytes, "merged completed-result learning manifest");
  });
  for (const parent of input.parents) {
    const sourceEngine = join(input.targetDirectory, "merge", "parents", parent.brainId, "engine");
    const sourceMetadata = JSON.parse(await readFile(join(sourceEngine, "brain.json"), "utf8"));
    for (const file of input.files(parent.brainId)) {
      const jobMatch = /^engine\/action-result-learning\/jobs\/([a-f0-9]{64})\.json$/.exec(file.path);
      if (!jobMatch) continue;
      const job = await readSavedActionEvidenceJob(join(sourceEngine, "action-result-learning", "jobs", `${jobMatch[1]}.json`));
      const evidence = await lstat(join(sourceEngine, "action-result-learning", "evidence", `${jobMatch[1]}.jsonl`));
      if (job.brainId !== parent.brainId || job.evidenceId !== jobMatch[1] || !evidence.isFile() || evidence.isSymbolicLink()) {
        throw new Error("A parent result-learning job lost its exact captured evidence.");
      }
    }
    for (const file of input.files(parent.brainId)) {
    const match = /^engine\/action-result-learning\/evidence\/([a-f0-9]{64})\.jsonl$/.exec(file.path);
    if (!match) continue;
    input.signal?.throwIfAborted();
    const oldId = match[1]!;
    const evidence = join(sourceEngine, "action-result-learning", "evidence", `${oldId}.jsonl`);
    const jobPath = join(sourceEngine, "action-result-learning", "jobs", `${oldId}.json`);
    let original: CompletedActionEvidenceJob;
    try { original = await readSavedActionEvidenceJob(jobPath); }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
      original = await readUnmanifestedActionEvidence(evidence, parent.brainId, oldId, admit);
    }
    const earlierSource = original.provenance.mergeSource as Record<string, unknown> | undefined;
    const capturedId = String(earlierSource?.capturedEvidenceId ?? earlierSource?.evidenceId ?? oldId);
    const captured = await readUnmanifestedActionEvidence(evidence, parent.brainId, capturedId, admit);
    if (original.brainId !== parent.brainId || original.evidenceId !== oldId ||
      original.executionId !== captured.executionId || original.evidenceSha256 !== file.sha256 || captured.evidenceSha256 !== file.sha256 ||
      ["actionEventId", "toolId", "action", "completedAt", "executionState"].some((key) => original.provenance[key] !== captured.provenance[key])) {
      throw new Error("Merged action-result learning lost its exact original outcome.");
    }
    const originalEvidenceId = original.mergeContinuation?.originalEvidenceId ?? oldId;
    const matching = queue.filter((item) => item.parentId === parent.brainId &&
      item.checkpoint?.transactionId === originalEvidenceId && item.checkpoint?.contentHash === original.evidenceSha256 &&
      item.checkpoint?.sourceNameHash === hash(`action-result-${originalEvidenceId}`));
    if (matching.length > 1) throw new Error("A completed-result continuation has ambiguous retained native cursors.");
    const pending = matching[0];
    if (original.mergeContinuation && !original.receipt && !pending) throw new Error("A pending merged result lost its native continuation.");
    const id = hash(JSON.stringify(["merged-action-result-v1", regions.get(parent.brainId) ?? parent.brainId, oldId, file.sha256]));
    const prior = await store.read(id);
    if (prior && (prior.evidenceSha256 !== original.evidenceSha256 || prior.executionId !== original.executionId)) throw new Error("A merged result-learning identity changed.");
    if (prior && await actionEvidenceFileHash(store.evidencePath(id)) !== prior.evidenceSha256) throw new Error("A staged merged outcome changed before resumption.");
    if (prior?.state === "complete" || prior && original.state !== "complete") continue;
    await mkdir(join(engine, "action-result-learning", "evidence"), { recursive: true });
    await input.reserve(file.bytes, "private merged completed-result evidence");
    await copyFullMergeBlob(evidence, store.evidencePath(id), original.evidenceSha256, input.signal ?? new AbortController().signal);
    const job: CompletedActionEvidenceJob = { ...original, brainId: input.targetBrainId, evidenceId: id,
      provenance: { ...original.provenance, mergeSource: { parentId: parent.brainId, evidenceId: oldId,
        capturedEvidenceId: capturedId, evidenceSha256: original.evidenceSha256, generationSha256: parent.generationSha256,
        ...(original.route ? { originalRouteEventId: original.route.eventId } : {}),
        ...(original.receipt ? { originalReceipt: original.receipt } : {}) } },
      ...(original.receipt ? { receipt: { ...original.receipt, brainId: input.targetBrainId, evidenceId: id } } : {}),
      ...(original.route ? { route: { ...original.route,
        eventId: learningUuid(JSON.stringify(["merged-route-learning-v1", regions.get(parent.brainId) ?? parent.brainId, original.route.eventId, original.route])) } } : {}),
    };
    delete job.mergeContinuation;
    if (pending && !job.receipt) job.mergeContinuation = { pendingId: String(pending.id), originalEvidenceId, parentId: parent.brainId };
    if (!job.receipt && !pending) {
      const completed = (sourceMetadata.completed_ingestions ?? []).find((receipt: Record<string, any>) =>
        receipt.transactionId === originalEvidenceId && receipt.contentHash === original.evidenceSha256 &&
        receipt.sourceNameHash === hash(`action-result-${originalEvidenceId}`));
      if (completed) {
        if (completed.format !== "omni-completed-ingestion" || completed.formatVersion !== 1 ||
          completed.policy !== "pretrain" || completed.epoch !== 0 || completed.coverage?.complete !== true ||
          !/^[a-f0-9]{64}$/.test(String(completed.sourceIdentity)) || !/^[a-f0-9]{64}$/.test(String(completed.parameterChecksumAfter))) {
          throw new Error("A parent action-result completion has no authenticated native receipt.");
        }
        // A lost Electron acknowledgement is not permission to learn the
        // already-committed source again. Its ancestor neural state is retained.
        job.receipt = { brainId: input.targetBrainId, evidenceId: id, executionId: original.executionId,
          evidenceSha256: original.evidenceSha256, committed: true, processed: false, duplicate: true,
          sourceIdentity: completed.sourceIdentity };
        job.provenance.mergeNativeCompletion = completed;
        if (job.routeComplete) job.state = "complete";
      }
    }
    await store.save(job);
    }
  }
}
