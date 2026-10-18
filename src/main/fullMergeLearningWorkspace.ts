import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { FullMergeAssemblyInput } from "./fullBrainMerge";
import type { LearningTask } from "../shared/learning";
import { LearningTaskStore, learningHash } from "./learningTaskStore";
import { DatasetManifestStore } from "./dataIngestion";
import { adoptLearningWorkspace, learningWorkspaceArchive } from "./learningWorkspacePortability";
import { openLearningRelocations } from "./learningSourceRelocation";
import { copyMutableFileIsolated, writeMutableFileIsolated } from "./mutableFileIsolation";
import { userQuestionRequestSha256, validateUserQuestionRecord } from "./userQuestions";
import { mergeClockWorkspace } from "./clockPortability";
import { mergeVoiceWorkspace } from "./voicePreferencesStore";
import { mergeTaskAttachmentWorkspace } from "./taskAttachmentPortability";
import { mergeFileHandoffCustody } from "./fileToolHandoffCustody";
import { mergeAutomationWorkspace } from "./automationPortability";
import { mergeAssemblyWorkspace } from "./assemblyPortability";
import { mergeSteerDirectionWorkspace, mergedSteerDirectionProjection } from "./steerDirectionPortability";
import { mergeHtmlArtifactWorkspace } from "./htmlArtifactPortability";
import { mergeImaginationWorkspace } from "./imaginationCaptureWorkspace";
import { mergeEnvironmentWorkspace } from "./environmentPortability";
import { mergeClipboardWorkspace } from "./clipboardPortability";
import { mergeReplayWorkspace } from "./imaginationAssemblyWorkspace";
import { mergeLocalTeacherWorkspace } from "./localTeacherPortability";
import { mergeExternalAdviceWorkspace } from "./externalAdvicePortability";
import { mergeApiTeacherWorkspace } from "./apiTeacherPortability";
import { mergePluginRuntimeWorkspace } from "./pluginRuntimePortability";
import { mergePanelReceiptWorkspace } from "./pluginPanelReceiptPortability";
import { mergeSparseSubstrateTransferWorkspace } from "./sparseSubstrateTransferPortability";

const alias = (parent: string, kind: string, id: string): string => learningHash({ parent, kind, id });

/** Executable operational projection; exact originals remain in parent archives. */
export async function adoptFullMergeLearningWorkspace(input: FullMergeAssemblyInput): Promise<void> {
  const root = input.targetDirectory, owner = input.targetBrainId;
  const metadata = JSON.parse(await readFile(join(root, "engine", "brain.json"), "utf8"));
  const nativeEntries: Array<{ parentId: string; originalTurnId: string; rootTurnId: string; id: string }> = metadata.connected_executions?.entries ?? [];
  const targetStore = await LearningTaskStore.open(root, owner), targetDatasets = new DatasetManifestStore(() => root);
  const targetRelocations = await openLearningRelocations(root, owner, true);
  try {
    for (const parent of input.parents) {
      input.signal?.throwIfAborted();
      const parentRoot = await input.materializeParent(parent.brainId), archive = await learningWorkspaceArchive(join(parentRoot, "portable-learning-workspace"), parent.brainId);
      if (!archive) throw new Error("The complete merge parent is missing its learning workspace snapshot.");
      const stage = await mkdtemp(join(root, ".learning-parent-"));
      try {
        await adoptLearningWorkspace(archive, stage, owner, { signal: input.signal, targetCorePath: join(parentRoot, "engine", "core.safetensors"),
          checkDisk: bytes => input.reserve(bytes, "retained parent learning workspace"),
          admitMetadata: async (bytes, purpose, signal) => {
            input.signal?.throwIfAborted(); signal?.throwIfAborted(); await input.admit?.(bytes, purpose);
            input.signal?.throwIfAborted(); signal?.throwIfAborted();
          } });
        const automationAliases = await mergeAutomationWorkspace(stage, root, owner, parent.brainId);
        await mergeEnvironmentWorkspace(stage, root, owner, owner, parent.brainId,
          bytes => input.reserve(bytes, "retained dormant sandbox/VM workspace"));
        await mergeAssemblyWorkspace(stage, root, parent.brainId, bytes => input.reserve(bytes, "retained native-code artifacts"));
        await mergeClockWorkspace(stage, root, owner, parent.brainId, new Map(nativeEntries.filter(entry => entry.parentId === parent.brainId).map(entry => [entry.originalTurnId, entry.rootTurnId])), automationAliases);
        await mergeVoiceWorkspace(stage, root, owner, parent.brainId);
        await mergeClipboardWorkspace(stage, root, owner, parent.brainId, bytes => input.reserve(bytes, "complete inherited instance clipboard"));
        await mergeExternalAdviceWorkspace(stage, root, owner, parent.brainId, bytes => input.reserve(bytes, "complete dormant external-advice/reviewer history"), input.signal);
        await mergeApiTeacherWorkspace(stage, root, owner, parent.brainId,
          bytes => input.reserve(bytes, "complete dormant API teacher history"), input.signal,
          async (bytes, purpose, signal) => {
            input.signal?.throwIfAborted(); signal?.throwIfAborted(); await input.admit?.(bytes, purpose);
            input.signal?.throwIfAborted(); signal?.throwIfAborted();
          });
        await mergePluginRuntimeWorkspace(stage, root, owner, parent.brainId, { signal: input.signal, checkDisk: bytes => input.reserve(bytes, "complete dormant feature-module state/history") });
        await mergePanelReceiptWorkspace(stage, root, owner, { signal: input.signal, checkDisk: bytes => input.reserve(bytes, "complete dormant scripted-panel replay history") });
        await mergeSparseSubstrateTransferWorkspace(stage, root, owner, parent.brainId,
          { signal: input.signal, reserveWrite: bytes => input.reserve(bytes, "complete dormant native sparse-transfer history") });
        await mergeTaskAttachmentWorkspace(stage, root, owner, parent.brainId,
          bytes => input.reserve(bytes, "retained parent task attachment"),
          new Map(nativeEntries.filter(entry => entry.parentId === parent.brainId).map(entry => [entry.originalTurnId, entry.rootTurnId])));
        await mergeFileHandoffCustody(root, owner, stage, owner, { signal: input.signal,
          reserve: async (_root, bytes) => { await input.reserve(bytes, "retained exact task-file handoffs"); } });
        await mergeHtmlArtifactWorkspace(stage, root, owner, parent.brainId,
          bytes => input.reserve(bytes, "retained parent HTML artifact"));
        const recordingAliases = await mergeImaginationWorkspace(stage, root, owner, parent.brainId, bytes => input.reserve(bytes, "complete imagination recording and live-window custody"));
        await mergeReplayWorkspace(stage, root, owner, bytes => input.reserve(bytes, "complete delivered replay and dormant encoding custody"), recordingAliases);
        await mergeSteerDirectionWorkspace(stage, root, owner, parent.brainId,
          record => mergedSteerDirectionProjection(root, owner, parent.brainId, record,
            new Map(nativeEntries.filter(entry => entry.parentId === parent.brainId).map(entry => [entry.originalTurnId, entry.rootTurnId]))),
          "merged-history", bytes => input.reserve(bytes, "retained steering presentation history"));
        const sourceStore = await LearningTaskStore.open(stage, owner), sourceDatasets = new DatasetManifestStore(() => stage);
        const sourceRelocations = await openLearningRelocations(stage, owner);
        const raw = new DatabaseSync(join(stage, "learning", "tasks.sqlite3"), { readOnly: true });
        try {
          const manifestAliases = new Map<string, string>(), questionAliases = new Map<string, string>();
          const taskAliases = new Map(sourceStore.list().map(task => [task.id, alias(parent.brainId, "task", task.id)]));
          const index = new DatabaseSync(join(archive.root, "inventory.sqlite3"), { readOnly: true });
          try {
            for (const row of index.prepare("SELECT path FROM files WHERE path GLOB 'datasets/*/manifest.json'").iterate()) {
              const match = /^datasets\/([^/]+)\/manifest.json$/.exec(String(row.path)); if (!match) continue;
              const old = await sourceDatasets.manifest(owner, match[1]!), id = alias(parent.brainId, "manifest", old.id);
              manifestAliases.set(old.id, id);
              const directory = join(root, "datasets", id); await mkdir(directory, { recursive: true });
              await copyMutableFileIsolated(join(stage, "datasets", old.id, old.entryFile), join(directory, old.entryFile), { expectedSha256: old.manifestHash });
              const manifest = { ...old, id, brainId: owner, portableSources: true as const,
                continuationIdentity: { manifestId: old.continuationIdentity?.manifestId ?? old.id, sourceBrainId: parent.brainId } };
              await writeMutableFileIsolated(join(directory, "manifest.json"), Buffer.from(JSON.stringify(manifest)));
              const progress = await sourceDatasets.progress(owner, old.id);
              const receipt = progress.lastEntryReceipt ? { ...progress.lastEntryReceipt, manifestId: id,
                continuationManifestId: manifest.continuationIdentity.manifestId,
                ...(progress.lastEntryReceipt.coverage ? { coverage: { ...progress.lastEntryReceipt.coverage, manifestId: id } } : {}) } : undefined;
              await targetDatasets.saveProgress(owner, { ...progress.cursor, manifestId: id, ...(progress.cursor.state === "running" ? { state: "paused" as const } : {}) },
                { ...progress.coverage, manifestId: id }, receipt);
              for await (const entry of sourceDatasets.entries(owner, old.id, 0, false)) {
                const binding = sourceRelocations?.source(old.id, entry.index); if (!binding) continue;
                await input.reserve(binding.bytes, "retained exact learning source");
                await copyMutableFileIsolated(join(stage, ...binding.relativePath.split("/")), join(root, ...binding.relativePath.split("/")), { expectedSha256: binding.contentSha256 });
                targetRelocations!.add(id, entry.index, { ...binding, parentId: parent.brainId, originalManifestId: manifest.continuationIdentity.manifestId });
              }
            }
            for (const row of index.prepare("SELECT path FROM files WHERE path GLOB 'questions/records/*.json'").iterate()) {
              const old = validateUserQuestionRecord(JSON.parse(await readFile(join(stage, ...String(row.path).split("/")), "utf8")), owner);
              const requestId = alias(parent.brainId, "question-request", old.requestId), id = learningHash({ requestId }); questionAliases.set(old.id, id);
              const next = { ...old, requestId, id,
                ...(old.source ? { source: { ...old.source, ...(old.source.actionId && taskAliases.has(old.source.actionId) ? { actionId: taskAliases.get(old.source.actionId)! } : {}) } } : {}) };
              next.requestSha256 = userQuestionRequestSha256(next); validateUserQuestionRecord(next, owner, id);
              await writeMutableFileIsolated(join(root, "questions", "records", `${id}.json`), Buffer.from(JSON.stringify(next)));
            }
            for (const row of index.prepare("SELECT path FROM files WHERE path GLOB 'learning/inputs/*.txt'").iterate())
              await copyMutableFileIsolated(join(stage, ...String(row.path).split("/")), join(root, ...String(row.path).split("/")));
          } finally { index.close(); }
          await mergeLocalTeacherWorkspace(stage, root, owner, parent.brainId, bytes => input.reserve(bytes, "retained local teacher data/history"), manifestAliases, taskAliases);
          for (const old of sourceStore.list()) {
            const id = taskAliases.get(old.id)!;
            const native = old.retainedNativeTurn && nativeEntries.find(entry => entry.parentId === parent.brainId && entry.originalTurnId === old.retainedNativeTurn!.turnId);
            const task: LearningTask = { ...old, id, brainId: owner, requestId: alias(parent.brainId, "request", old.requestId),
              pendingQuestions: old.pendingQuestions.map(id => questionAliases.get(id) ?? id),
              ...(old.policyParentTaskId ? { policyParentTaskId: taskAliases.get(old.policyParentTaskId) ?? old.policyParentTaskId } : {}),
              inheritedFrom: { parentId: parent.brainId, taskId: old.id, requestId: old.requestId, requestSha256: old.requestSha256,
                ...(old.retainedNativeTurn ? { originalTurnId: old.retainedNativeTurn.turnId } : {}), ...(native ? { executionId: native.id } : {}) },
              ...(native && old.retainedNativeTurn ? { retainedNativeTurn: { ...old.retainedNativeTurn, turnId: native.rootTurnId } } : {}) };
            if (old.retainedNativeTurn && !native && old.state !== "complete") {
              task.resumeBlocked = { reason: "retained-native-frontier-unavailable", sourceCoreSha256: archive.nativeCoreSha256 ?? "0".repeat(64) };
              task.state = "paused"; task.phase = "retained-native-frontier-unavailable"; task.label = "Retained task needs its exact ancestral native frontier";
            }
            const sources = sourceStore.sources(old.id).map(source => ({ ...source,
              ...(source.manifestId ? { manifestId: manifestAliases.get(source.manifestId)! } : {}),
              ...(source.coverage && source.manifestId ? { coverage: { ...source.coverage, manifestId: manifestAliases.get(source.manifestId)! } } : {}),
              ...(source.crawlId ? { crawlId: alias(parent.brainId, "crawl", source.crawlId) } : {}) }));
            const claims = sourceStore.claims(old.id).map(claim => ({ ...claim, evidence: [], findings: [],
              ...(claim.questionId ? { questionId: questionAliases.get(claim.questionId) ?? claim.questionId } : {}),
              ...(claim.correctionTaskId ? { correctionTaskId: taskAliases.get(claim.correctionTaskId) ?? claim.correctionTaskId } : {}) }));
            targetStore.create(task, sources, claims);
            for (const claim of sourceStore.claims(old.id)) {
              for (const evidence of claim.evidence) targetStore.addEvidence(id, evidence);
              for (const finding of claim.findings) {
                const sha = raw.prepare("SELECT request_sha FROM findings WHERE task_id=? AND id=?").get(old.id, finding.id)?.request_sha;
                if (typeof sha !== "string") throw new Error("Retained finding lost its source request receipt.");
                targetStore.addFinding(id, finding, sha);
              }
            }
            for (const turn of raw.prepare("SELECT turn_id,claim_id FROM native_turns WHERE task_id=?").iterate(old.id)) {
              const matched = nativeEntries.find(entry => entry.parentId === parent.brainId && entry.originalTurnId === turn.turn_id);
              targetStore.bindNativeTurn(id, matched?.rootTurnId ?? alias(parent.brainId, "native-turn", String(turn.turn_id)), turn.claim_id === null ? undefined : String(turn.claim_id));
            }
            for (const source of sourceStore.sources(old.id)) if (source.crawlId) {
              const newId = alias(parent.brainId, "crawl", source.crawlId), destination = join(root, "datasets", "crawls", `${newId}.sqlite3`);
              await copyMutableFileIsolated(join(stage, "datasets", "crawls", `${source.crawlId}.sqlite3`), destination);
              const index = new DatabaseSync(join(archive.root, "inventory.sqlite3"), { readOnly: true });
              try {
                for (const row of index.prepare("SELECT path,body FROM generic_sources").iterate()) {
                  const binding = JSON.parse(String(row.body)), oldPath = String(row.path);
                  if (oldPath.split("/").length > 3 && !oldPath.startsWith(`datasets/web-cache/${source.crawlId}/`)) continue;
                  const relativePath = `datasets/web-cache/${newId}/${basename(oldPath)}`, leaseName = `${basename(oldPath).split(".")[0]}.lease.json`;
                  await copyMutableFileIsolated(join(stage, ...oldPath.split("/")), join(root, ...relativePath.split("/")), { expectedSha256: binding.contentSha256 });
                  await copyMutableFileIsolated(join(stage, dirname(oldPath), leaseName), join(root, "datasets", "web-cache", newId, leaseName));
                  targetRelocations!.addGeneric({ ...binding, parentId: parent.brainId, relativePath });
                }
              } finally { index.close(); }
            }
          }
        } finally { raw.close(); sourceStore.close(); sourceRelocations?.close(); }
      } finally { await rm(stage, { recursive: true, force: true }); }
    }
  } finally { targetStore.close(); targetRelocations?.close(); }
}
