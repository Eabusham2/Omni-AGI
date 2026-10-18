/** ONE opt-in native host path. The external runner owns the hard resource/watchdog cap. */
import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { EngineSupervisor, ENGINE_REQUEST_NO_DEADLINE } from "../src/main/engineSupervisor";
import { ResourcePlanner } from "../src/main/resourcePlanner";
import { SharedResourceRegistry } from "../src/main/sharedResourceRegistry";
import { DEFAULT_CONFIG } from "../src/shared/types";
import { selectedInheritedSourceInventory } from "./currentInheritedSourceInventory";

const identity = vi.hoisted(() => ({ next: undefined as string | undefined }));
vi.mock("node:crypto", async original => {
  const crypto = await original<typeof import("node:crypto")>();
  return { ...crypto, randomUUID: () => { const next = identity.next; identity.next = undefined; return next ?? crypto.randomUUID(); } };
});
const native = process.env.OMNI_CURRENT_INHERITED_HOST_PROOF === "1" ? it : it.skip;
type Capture = { format: string; version: number; executionId: string; rootTurnId: string; checkpointId: string; publishedText: string; sourceUnchanged: boolean; parameterChecksum: string };
type Invariants = { pid: number; resident: boolean; ownerReleased: boolean; leaseReleased: boolean; humanCount: number;
  counters: { inference_count: number; training_steps: number }; activityClock: { paused: boolean }; queue: { entries: Array<{ status: string; id: string }> }; parameterChecksum?: string; resourceStatus?: { diskReserveBytes: number } };

native("uses actual host/native capture, inherited pause/cold resume and alias-aware lost-ACK reconciliation", async () => {
  const root = process.env.OMNI_CURRENT_INHERITED_HOST_ROOT, source = process.env.OMNI_CURRENT_INHERITED_SOURCE;
  if (!root || !source || !process.env.OMNI_CURRENT_INHERITED_EXTERNAL_CAP) throw new Error("Use the dedicated externally capped runner; no uncapped neural fixture entry.");
  const started = performance.now(), notes: unknown[] = [];
  const note = (stage: string, fields: Record<string, unknown> = {}) => {
    const value = { stage, elapsedSeconds: (performance.now() - started) / 1000, ...fields }; notes.push(value);
    process.stdout.write(`CURRENT_INHERITED_PROOF ${JSON.stringify(value)}\n`);
  };
  const repository = new BrainRepository(root), registry = new SharedResourceRegistry(join(root, "shared-quota.sqlite3"));
  const engine = new EngineSupervisor({ appPath: process.cwd(), workerPath: join(process.cwd(), "engine/tests/current_inherited_host_worker.py"),
    pythonCommand: join(process.cwd(), ".venv/bin/python"), sharedResourceLedgerPath: registry.ledgerPath });
  const service = new BrainService(repository, engine, new ResourcePlanner(root), undefined, registry);
  const params = { brainId: "joined", storagePath: join(root, "joined") };
  let response: ReturnType<BrainService["resumeInheritedTurn"]> | undefined;
  try {
    // This is sixteen selected files, not a claim of full-parent payload coverage.
    const originalHashes = new Map(Object.entries(await selectedInheritedSourceInventory(source)));
    const originals = [...originalHashes.keys()];
    const saved = JSON.parse(await readFile(join(source, "merged/engine/origin/brain.json"), "utf8")) as { config: { working_memory_slots: number; memory_resident_items: number; max_seq_len: number } };
    identity.next = "joined";
    await repository.create({ ...DEFAULT_CONFIG, name: "Current inherited fixture", idleCognition: false, onlineLearning: false,
      contextWindowTokens: saved.config.max_seq_len, workingMemorySlots: saved.config.working_memory_slots,
      memoryResidentItems: saved.config.memory_resident_items, storagePoolBytes: 0 });
    note("host-document-created-no-neural-build");
    const prepared = await engine.request<{ modelInitialized: false; root: string }>("fixture_prepare_current", {}, ENGINE_REQUEST_NO_DEADLINE);
    expect(prepared.modelInitialized).toBe(false); expect(prepared.root).toBe(params.storagePath);
    note("immutable-origin-private-clone-ready", { workerPid: engine.pid });
    // The old queue is archival, not executable. This first real Worker Load
    // supplies learned owners for a NEW capture, not a historical cursor restore.
    await engine.request("load", params, ENGINE_REQUEST_NO_DEADLINE);
    note("initial-worker-owned-cold-load", { workerPid: engine.pid });
    const capture = await engine.request<Capture>("fixture_capture_current", params, ENGINE_REQUEST_NO_DEADLINE);
    expect(capture.format).toBe("omni-fresh-connected-execution-fixture"); expect(capture.version).toBe(1); expect(capture.sourceUnchanged).toBe(true);
    note("new-authentic-current-frontier", { checkpointId: capture.checkpointId, prefix: capture.publishedText, workerPid: engine.pid });
    const reference = await engine.request<{ checkpointId: string; text: string; publishedText: string; finalizationSkipped: boolean; originalSourceUnchanged: boolean }>("fixture_reference_current", params, ENGINE_REQUEST_NO_DEADLINE);
    expect(reference.checkpointId).toBe(capture.checkpointId); expect(reference.publishedText).toBe(capture.publishedText);
    expect(reference.finalizationSkipped).toBe(true); expect(reference.originalSourceUnchanged).toBe(true);
    note("independent-reference-before-real-resume");
    const admitted = await service.loadInstance("joined"); expect(admitted.phase).toBe("loaded");
    const before = await engine.request<Invariants>("fixture_invariants", params, ENGINE_REQUEST_NO_DEADLINE);
    expect(before.parameterChecksum).toBe(capture.parameterChecksum); expect(before.resourceStatus!.diskReserveBytes).toBeGreaterThanOrEqual(20 * 1024 ** 3);
    await engine.request("fixture_arm_pause", params, ENGINE_REQUEST_NO_DEADLINE);
    let prefix = capture.publishedText, settled = false, unload: ReturnType<BrainService["unloadInstance"]> | undefined;
    const sequences: number[] = [];
    response = service.resumeInheritedTurn("joined", capture.executionId, undefined, event => {
      if ("sequence" in event && typeof event.sequence === "number" && event.sequence > 0) sequences.push(event.sequence);
      if (event.type === "chat-token") {
        prefix += event.delta;
        if (event.delta && !unload) unload = service.unloadInstance("joined");
      }
    }).finally(() => { settled = true; });
    void response.catch(() => undefined);
    await vi.waitFor(() => expect(unload).toBeDefined(), { timeout: 110_000, interval: 20 });
    expect((await unload!).phase).toBe("suspended"); expect(settled).toBe(false);
    const paused = await engine.request<Invariants>("fixture_invariants", params, ENGINE_REQUEST_NO_DEADLINE);
    expect(paused.resident).toBe(false); expect(paused.ownerReleased).toBe(true); expect(paused.leaseReleased).toBe(true);
    expect(paused.counters.inference_count).toBe(before.counters.inference_count); expect(paused.activityClock.paused).toBe(true);
    const published = await service.instanceInheritedResumePresentation("joined", capture.executionId); expect(published.publishedText).toBe(prefix);
    const firstPid = engine.pid; await engine.stop(); expect(engine.pid).toBeUndefined();
    note("actual-unload-and-first-process-exit", { firstPid, checkpointId: published.frontier.checkpointId });
    await engine.request("fixture_arm_lost_ack", {}, ENGINE_REQUEST_NO_DEADLINE);
    const cold = await service.instanceStatus("joined"); expect(cold.learnedRuntimeResident).toBe(false);
    expect((await service.loadInstance("joined")).phase).toBe("loaded");
    note("actual-host-cold-load", { workerPid: engine.pid }); expect(engine.pid).not.toBe(firstPid);
    const result = await response;
    expect(result.brainMessage.content).toBe(reference.text); expect(prefix).toBe(reference.text);
    expect(result.brain.counters.inferenceCount).toBe(before.counters.inference_count + 1);
    const after = await engine.request<Invariants>("fixture_invariants", params, ENGINE_REQUEST_NO_DEADLINE);
    expect(after.counters.inference_count).toBe(before.counters.inference_count + 1); expect(after.humanCount).toBe(before.humanCount);
    expect(after.counters.training_steps).toBe(before.counters.training_steps);
    expect(after.queue.entries.find(entry => entry.id === capture.executionId)?.status).toBe("complete");
    expect(JSON.parse(await readFile(join(root, "lost-final-ack.json"), "utf8"))).toMatchObject({ executionId: capture.executionId, turnCommitted: true });
    const eventsBeforeRetry = sequences.length;
    expect((await service.resumeInheritedTurn("joined", capture.executionId)).brainMessage.id).toBe(result.brainMessage.id);
    expect(sequences.length).toBe(eventsBeforeRetry); expect(sequences).toEqual([...new Set(sequences)].sort((a, b) => a - b));
    expect(await selectedInheritedSourceInventory(source)).toEqual(Object.fromEntries(originalHashes));
    await service.unloadInstance("joined");
    note("current-host-worker-proof-complete", { selectedOriginalFilesUnchanged: originals.length, lostAckReconciled: true, completedExternalEffects: 0 });
    await writeFile(join(root, "host-proof-receipt.json"), JSON.stringify({ format: "omni-current-inherited-host-proof", version: 1,
      capture, reference, before, after, selectedSourceFileHashes: Object.fromEntries(originalHashes), notes, completedExternalEffects: 0,
      forbiddenOperations: ["Worker.create", "AdaptiveBrain._train_ground_up_curriculum", "AdaptiveBrain.train", "AdaptiveBrain.ingest", "slow replay"],
      limitations: ["Talk-only structural action control", "No real completed external effect in retained source", "Not trained language quality or cross-hardware proof"] }, null, 2), { flag: "wx", mode: 0o600 });
  } finally { await engine.stop(); registry.close(); }
}, 120_000);
