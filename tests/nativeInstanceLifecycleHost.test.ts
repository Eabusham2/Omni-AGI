import { createHash } from "node:crypto";
import { cp, mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { expect, it, vi } from "vitest";

const id = vi.hoisted(() => ({ next: undefined as string | undefined }));
vi.mock("node:crypto", async (original) => {
  const crypto = await original<typeof import("node:crypto")>();
  return { ...crypto, randomUUID: () => { const value = id.next; id.next = undefined; return value ?? crypto.randomUUID(); } };
});

import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";
import { SharedResourceRegistry } from "../src/main/sharedResourceRegistry";
import { DEFAULT_CONFIG } from "../src/shared/types";

const native = process.env.OMNI_NATIVE_LIFECYCLE_HOST === "1" ? it : it.skip;
const hash = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");

native("production host parks one native turn, ports the paused identity and resumes without repeat effects", async () => {
  const source = process.env.OMNI_NATIVE_LIFECYCLE_FIXTURE;
  if (!source) throw new Error("An existing genuine initialized native fixture is required; this proof never creates one.");
  const saved = JSON.parse(await readFile(join(source, "engine/brain.json"), "utf8"));
  const sourceBefore = hash(await readFile(join(source, "engine/brain.json")));
  const root = await mkdtemp(join(tmpdir(), "omni-native-lifecycle-host-"));
  vi.stubEnv("OMNI_NATIVE_LIFECYCLE_HOST_ROOT", root);
  const repository = new BrainRepository(root);
  const registry = new SharedResourceRegistry(join(root, "fixture-quota.sqlite3"));
  const planner = new ResourcePlanner(root, { readResources: async () => ({ totalMemoryBytes: 16 * GIB,
    availableMemoryBytes: 12 * GIB, diskTotalBytes: 512 * GIB, diskFreeBytes: 80 * GIB }),
    benchmark: async () => ({ measuredAt: "2026-10-11T00:00:00.000Z", sampleBytes: 1024 * 1024,
      memoryBytesPerSecond: 12 * GIB, storageBytesPerSecond: GIB, cacheHit: true }) });
  const engine = new EngineSupervisor({ appPath: process.cwd(), workerPath: join(process.cwd(), "engine/tests/native_lifecycle_host_worker.py"),
    pythonCommand: process.env.OMNI_PYTHON ?? join(process.cwd(), ".venv/bin/python"), sharedResourceLedgerPath: registry.ledgerPath });
  const service = new BrainService(repository, engine, planner, undefined, registry);
  let success = false;
  try {
    id.next = saved.brain_id;
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: saved.name, idleCognition: false, onlineLearning: false,
      workingMemorySlots: saved.config.working_memory_slots, memoryResidentItems: saved.config.memory_resident_items,
      contextWindowTokens: saved.config.max_seq_len, storagePoolBytes: 20 * GIB });
    expect(brain.id).toBe(saved.brain_id);
    await cp(join(source, "engine"), join(root, brain.id, "engine"), { recursive: true });
    await vi.waitFor(async () => expect((await service.instanceStatus(brain.id)).phase).toBe("unloaded"), { timeout: 10_000, interval: 1_000 });
    const initialLoad = await service.loadInstance(brain.id);
    if (initialLoad.phase !== "loaded") throw new Error(`Initial native host Load: ${JSON.stringify(initialLoad)}`);
    let unload: ReturnType<BrainService["unloadInstance"]> | undefined;
    let settled = false; let prefix = "";
    const turnId = "production-host-retained-turn";
    const response = service.chat(brain.id, "Hello", undefined, (event) => {
      if (event.type === "chat-token") {
        prefix += event.delta;
        if (event.delta && !unload) unload = service.unloadInstance(brain.id);
      }
    }, turnId, 7).finally(() => { settled = true; });
    void response.catch(() => undefined); // A failed later gate still cleans up the parked original.
    await vi.waitFor(() => expect(unload).toBeDefined(), { timeout: 150_000, interval: 250 });
    const suspended = await unload!;
    expect(suspended.phase).toBe("suspended"); expect(suspended.learnedRuntimeResident).toBe(false);
    expect(settled).toBe(false); expect(prefix).not.toBe("");
    const presentation = await service.instanceResumePresentation(brain.id);
    expect(presentation?.publishedText).toBe(prefix); expect(presentation?.hostAttached).toBe(true);
    const snapshot = await repository.snapshot(brain.id, "Exact paused native turn");
    const fork = await repository.fork(brain.id, "Paused native copy");
    const forkStatus = await service.instanceStatus(fork.id);
    expect(forkStatus.resume?.turnId).toBe(turnId);
    expect((await service.loadInstance(fork.id, forkStatus.resume!.checkpointId)).phase).toBe("loaded");
    const forkResult = await service.chat(fork.id, "Hello", undefined, undefined, turnId, undefined, undefined, undefined, forkStatus.resume);
    await service.unloadInstance(fork.id);
    const bundle = join(root, "paused-native.omni");
    await repository.exportBundle(brain.id, bundle, "private-archive");
    const imported = await repository.importBundle(bundle);
    const importedState = await service.instanceStatus(imported.id);
    expect(importedState.resume?.turnId).toBe(turnId);
    expect((await service.loadInstance(imported.id, importedState.resume!.checkpointId)).phase).toBe("loaded");
    const importedResult = await service.chat(imported.id, "Hello", undefined, undefined, turnId, undefined, undefined, undefined, importedState.resume);
    await service.unloadInstance(imported.id);
    expect(importedResult.brainMessage.content).toBe(forkResult.brainMessage.content);
    expect((await service.loadInstance(brain.id, suspended.resume!.checkpointId)).phase).toBe("loaded");
    const completed = await response;
    expect(completed.brainMessage.content).toBe(forkResult.brainMessage.content);
    expect(prefix).toBe(completed.brainMessage.content);
    await service.unloadInstance(brain.id);
    await repository.restoreSnapshot(brain.id, snapshot.id);
    const restored = await service.instanceStatus(brain.id);
    expect(restored.resume?.turnId).toBe(turnId);
    expect((await service.loadInstance(brain.id, restored.resume!.checkpointId)).phase).toBe("loaded");
    const recovered = await service.chat(brain.id, "Hello", undefined, undefined, turnId, undefined, undefined, undefined, restored.resume);
    expect(recovered.brainMessage.content).toBe(completed.brainMessage.content);
    expect(hash(await readFile(join(source, "engine/brain.json")))).toBe(sourceBefore);
    success = true;
  } catch (error) {
    console.error("Native lifecycle host proof failure detail", error && typeof error === "object" && "data" in error ? error.data : error);
    throw error;
  } finally {
    await engine.stop(); registry.close(); vi.unstubAllEnvs();
    if (!success && process.env.OMNI_RETAIN_NATIVE_LIFECYCLE_HOST === "1") console.error(`Retained owned lifecycle proof fixture: ${root}`);
    else await rm(root, { recursive: true, force: true, maxRetries: 5 });
  }
}, 420_000);
