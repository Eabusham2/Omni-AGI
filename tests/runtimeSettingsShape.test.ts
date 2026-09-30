/** File/protocol fixtures only; no worker, model, application or UI run. */
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { nativeCoreInventory, sealNativeArchitecture } from "../src/main/nativeCoreInventory";
import { legacyNativeArchitectureDescriptor } from "../src/main/omniArchitectureProfile";
import { GIB, ResourcePlanner } from "../src/main/resourcePlanner";
import { requireWorkingMemoryPlanRequest } from "../src/main/resourcePlanRequest";
import { savedRuntimeShape } from "../src/main/savedRuntimeShape";
import { DEFAULT_CONFIG, type WorkingMemoryResourcePlan } from "../src/shared/types";
import { configWithResourceEnvelope } from "../src/renderer/src/resourceEnvelope";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-runtime-geometry-")); roots.push(root);
  const repository = new BrainRepository(join(root, "brains")); await repository.initialize();
  const original = legacyNativeArchitectureDescriptor("micro", 128);
  const shape = { ...original.shape, layers: original.shape.layers + 1 };
  const payload = { ...original, shape, inventory: nativeCoreInventory(shape), evolutionLineage: {
    format: "omni-compatible-architecture-lineage", formatVersion: 1,
    rootArchitectureSha256: original.sha256, parentArchitectureSha256: original.sha256,
    rootShape: original.shape, mutations: [{ mutation: "grow-depth", addLayers: 1 }],
    normalizationAndHeadGeometryChanged: false, qualityVerified: false
  }};
  const current = sealNativeArchitecture(payload);
  const brain = await repository.create({ ...DEFAULT_CONFIG, workingMemorySlots: 128, nativeArchitecture: original });
  const engine = join(repository.brainDirectory(brain.id), "engine"); await mkdir(engine, { recursive: true });
  const metadata = { brain_id: brain.id, config: { hardware_tier: "micro", d_model: shape.dModel, n_layers: shape.layers,
    working_memory_slots: 128, native_architecture: current } };
  await writeFile(join(engine, "brain.json"), JSON.stringify(metadata));
  await writeFile(join(engine, "core.safetensors"), Buffer.alloc(1024));
  await writeFile(join(engine, "plasticity.safetensors"), Buffer.alloc(1024));
  const planner = new ResourcePlanner(repository.root, {
    readResources: async () => ({ totalMemoryBytes: 32 * GIB, availableMemoryBytes: 24 * GIB, diskTotalBytes: 500 * GIB, diskFreeBytes: 300 * GIB }),
    benchmark: async () => ({ measuredAt: "2026-09-30T00:00:00Z", sampleBytes: 4096, memoryBytesPerSecond: 20 * GIB, storageBytesPerSecond: GIB, cacheHit: true })
  });
  const request = vi.fn(async () => ({}));
  const service = new BrainService(repository, { request, tryRequest: vi.fn(async () => undefined) } as unknown as EngineSupervisor, planner);
  return { root, repository, brain, original, current, engine, metadata, planner, request, service };
}

describe("runtime settings preserve saved learned geometry", () => {
  it("resolves Auto and Extended for the saved brain, not a fresh detected-tier model", async () => {
    const value = await fixture();
    for (const mode of ["auto", "extended"] as const) {
      const plan = await value.service.planWorkingMemory({ mode, brainId: value.brain.id, hardwareTier: "workstation" });
      expect(plan.allowed).toBe(true);
      expect(plan.hardwareTier).toBe("micro");
      expect(plan.selectedItems).toBe(128);
      expect(plan.context.evidence.estimatedKvActivationBytesPerToken).toBe(8 * value.current.shape.dModel * value.current.shape.layers);
      expect(plan.context.evidence.workspaceResidentBytes).toBeGreaterThanOrEqual(4 * value.current.shape.workspaceLatents * (2 * value.current.shape.dModel + 1));
      expect(value.request).not.toHaveBeenCalled();
    }
  });

  it("applies context, RAM and pool changes while preserving current and original hashes", async () => {
    const value = await fixture();
    const updated = await value.service.updateConfig(value.brain.id, {
      ...value.brain.config, workingMemorySlots: 65_536, nativeArchitecture: value.original,
      workingMemoryMode: "extended", extendedWorkingMemory: true, contextWindowTokens: 512,
      systemRamMode: "manual", systemRamSharePercent: 54, storagePoolMode: "manual", storagePoolBytes: 2 * GIB
    });
    expect(updated.config).toMatchObject({ workingMemorySlots: 128, contextWindowTokens: 512,
      workingMemoryMode: "extended", systemRamSharePercent: 54, storagePoolBytes: 2 * GIB,
      nativeArchitecture: { sha256: value.current.sha256 } });
    expect(value.request).toHaveBeenCalledWith("update_config", expect.objectContaining({ config: expect.objectContaining({
      workingMemorySlots: 128, nativeArchitecture: { ...value.current }, contextWindowTokens: 512
    }) }), 300_000);
    const origin = JSON.parse(await readFile(join(value.repository.brainDirectory(value.brain.id), "origin.json"), "utf8"));
    expect(origin.config.nativeArchitecture.sha256).toBe(value.original.sha256);
    expect(origin.config.workingMemorySlots).toBe(128);
  });

  it("rejects wrong saved ownership before any worker write", async () => {
    const value = await fixture();
    await writeFile(join(value.engine, "brain.json"), JSON.stringify({ ...value.metadata, brain_id: "different-brain" }));
    await expect(value.service.updateConfig(value.brain.id, value.brain.config)).rejects.toThrow(/different brain/);
    expect(value.request).not.toHaveBeenCalled();
  });

  it("keeps a descriptor-free legacy shape without synthesizing a new descriptor", async () => {
    const value = await fixture();
    await writeFile(join(value.engine, "brain.json"), JSON.stringify({ brain_id: value.brain.id, config: { working_memory_slots: 77 } }));
    const result = await savedRuntimeShape(value.repository.brainDirectory(value.brain.id), value.brain.id, value.brain.config);
    expect(result).toEqual({ workingMemorySlots: 77, nativeArchitecture: undefined });
  });

  it("Device envelope treats planned item changes as advisory and retains the descriptor", () => {
    const original = legacyNativeArchitectureDescriptor("micro", 128);
    const config = { ...DEFAULT_CONFIG, workingMemorySlots: 128, nativeArchitecture: original };
    const plan = { mode: "extended", selectedItems: 65_536, context: { selectedTokens: 512, evidence: { contextOffloadBudgetBytes: 4096 } },
      resources: { configuredMemorySpillBytes: 123, storagePoolMode: "manual", sharedStoragePoolBytes: GIB },
      offload: { residentMemoryItems: 10_000, estimatedSlowdownPercent: 3, benchmark: { storageBytesPerSecond: GIB } } } as WorkingMemoryResourcePlan;
    expect(configWithResourceEnvelope(config, plan, "manual", 54)).toMatchObject({ workingMemorySlots: 128,
      nativeArchitecture: original, contextWindowTokens: 512, contextOffloadBudgetBytes: 4096, memoryResidentItems: 128 });
    expect(requireWorkingMemoryPlanRequest({ mode: "auto", brainId: "saved.brain-1" }).brainId).toBe("saved.brain-1");
  });
});
