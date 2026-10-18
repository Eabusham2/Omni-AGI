import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { EngineSupervisor } from "../src/main/engineSupervisor";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";
import { SharedResourceRegistry } from "../src/main/sharedResourceRegistry";
import { DEFAULT_CONFIG } from "../src/shared/types";

const native = process.env.OMNI_FULL_MERGE_NATIVE === "1" ? it : it.skip;

native("runs actual native checkpoint, connected assembly, host adoption, verification and publication through BrainService", async () => {
  const root = await mkdtemp(join(tmpdir(), "omni-native-host-"));
  const workerPath = join(process.cwd(), "engine", "tests", "full_merge_host_worker_fixture.py");
  const python = process.env.OMNI_PYTHON ?? "python3";
  vi.stubEnv("OMNI_NATIVE_MERGE_FIXTURE_ROOT", root);
  const repository = new BrainRepository(root);
  const registry = new SharedResourceRegistry(join(root, "fixture-quota.sqlite3"));
  const planner = new ResourcePlanner(root, { readResources: async () => ({ totalMemoryBytes: 8 * GIB,
    availableMemoryBytes: 6 * GIB, diskTotalBytes: 128 * GIB, diskFreeBytes: 100 * GIB }),
    benchmark: async () => ({ measuredAt: "2026-10-01T00:00:00.000Z", sampleBytes: 16 * 1024 * 1024,
      memoryBytesPerSecond: 12 * GIB, storageBytesPerSecond: GIB, cacheHit: true }) });
  const engine = new EngineSupervisor({ appPath: process.cwd(), workerPath, pythonCommand: python,
    sharedResourceLedgerPath: registry.ledgerPath });
  try {
    const ids = [];
    for (const name of ["Left", "Right"]) {
      const brain = await repository.create({ ...DEFAULT_CONFIG, name, idleCognition: false, workingMemorySlots: 4,
        memoryResidentItems: 4, contextWindowTokens: 16, storagePoolBytes: 0 });
      await promisify(execFile)(python, [workerPath, "create", brain.id, name], {
        env: { ...process.env, OMNI_NATIVE_MERGE_FIXTURE_ROOT: root }, maxBuffer: 1024 * 1024
      });
      brain.messages = [{ id: brain.id + "-message", role: "human", content: name + " saved native experience", createdAt: "2026-10-01T00:00:00.000Z" }];
      await repository.save(brain);
      ids.push(brain.id);
    }
    const service = new BrainService(repository, engine, planner, undefined, registry);
    service.configureFullMergeAccess(async () => undefined); // isolated fixture catalog has no connections
    const done = await service.startFullMerge({ operationId: "native-host-fixture", hardwareTier: "micro", parentIds: ids, name: "Connected",
      config: repository.prepareConfig({ ...DEFAULT_CONFIG, name: "Connected", idleCognition: false,
        workingMemorySlots: 8, memoryResidentItems: 8, contextWindowTokens: 32, storagePoolBytes: 0 }),
      accessPolicy: { credentials: "none", mcp: "disabled", toolPermissions: [] } });
    expect(done.state).toBe("complete");
    const merged = await repository.get(done.targetBrainId);
    expect(merged.provenance?.originKind).toBe("merged-native");
    expect(merged.conversation?.messageCount).toBe(2);
    const metadata = JSON.parse(await readFile(join(root, merged.id, "engine", "brain.json"), "utf8"));
    expect(metadata.connected_regions.parents).toHaveLength(2);
    expect(metadata.connected_runtime_adoption.historyInjectedAsPrompt).toBe(false);
    expect(metadata.ground_up_training_manifest).toBeNull();
    expect((await service.resumeFullMerge("native-host-fixture")).targetBrainId).toBe(merged.id);
    for (const id of ids) expect((await repository.get(id)).id).toBe(id);
    const sourceMetadata = await readFile(join(root, merged.id, "engine", "brain.json"));
    const sourceParents = await Promise.all(ids.map(id => readFile(join(root, merged.id, "merge", "parents", id, "engine", "brain.json"))));
    const load = async (id: string, directory: string): Promise<string> => {
      const result = await engine.request<{ brainId: string; parameterChecksum: string }>("load", { brainId: id, storagePath: directory }, 120_000);
      expect(result.brainId).toBe(id);
      await engine.request("unload", { brainId: id, storagePath: directory });
      return result.parameterChecksum;
    };
    const baseline = await load(merged.id, repository.brainDirectory(merged.id));
    const recovery = await repository.snapshot(merged.id, "native connected portability boundary");
    const fork = await repository.fork(merged.id, "Connected native fork");
    expect(await load(fork.id, repository.brainDirectory(fork.id))).toBe(baseline);
    for (const mode of ["current", "private-archive", "referenced", "origin"] as const) {
      const bundle = join(root, `connected-${mode}.omni`);
      await repository.exportBundle(merged.id, bundle, mode);
      const receiving = mode === "referenced" ? repository : new BrainRepository(join(root, `import-${mode}`));
      await receiving.initialize();
      const imported = await receiving.importBundle(bundle);
      expect(await load(imported.id, receiving.brainDirectory(imported.id))).toBe(baseline);
      // Host recovery consumes its opaque package and installs real native
      // dependency paths before the worker's fresh load callback runs.
      await receiving.restoreSnapshot(imported.id, recovery.id, undefined, async () => {
        expect(await load(imported.id, receiving.brainDirectory(imported.id))).toBe(baseline);
      });
      expect((await receiving.get(imported.id)).provenance?.originKind).toBe("merged-native");
    }
    expect(await readFile(join(root, merged.id, "engine", "brain.json"))).toEqual(sourceMetadata);
    for (const [index, id] of ids.entries()) {
      expect(await readFile(join(root, merged.id, "merge", "parents", id, "engine", "brain.json"))).toEqual(sourceParents[index]);
    }
  } finally {
    await engine.stop(); registry.close(); vi.unstubAllEnvs();
    await rm(root, { recursive: true, force: true, maxRetries: 5 });
  }
}, 420_000);
