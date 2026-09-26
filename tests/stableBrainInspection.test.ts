import { readFileSync } from "node:fs";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import {
  BrainService,
  LARGE_FOREGROUND_LOAD_TIMEOUT_MS
} from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import type { ResourcePlanner } from "../src/main/resourcePlanner";
import {
  DEFAULT_CONFIG,
  type SubstratePage,
  type WorkspaceSnapshot
} from "../src/shared/types";

describe("stable brain inspection service", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-inspection-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  async function writeNovaSizedSubstrate(brainId: string): Promise<void> {
    const engineDirectory = join(repository.brainDirectory(brainId), "engine");
    await mkdir(engineDirectory, { recursive: true });
    await writeFile(
      join(engineDirectory, "brain.json"),
      JSON.stringify({
        substrate: {
          persistence: {
            shardCount: 9_456,
            counts: {
              neurons: 14_941,
              assemblies: 117,
              synapses: 4_811_811
            }
          }
        }
      }),
      "utf8"
    );
  }

  it("forwards an uncapped cursor query to the authoritative neural worker", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Inspection brain"
    });
    await writeNovaSizedSubstrate(brain.id);
    const page: SubstratePage = {
      brainId: brain.id,
      queriedAt: new Date().toISOString(),
      revision: "live-revision-1",
      entity: "neurons",
      zoom: 1,
      totals: { neurons: 14_941, assemblies: 117, synapses: 4_811_811 },
      matched: 14_941,
      hasMore: true,
      nextCursor: "opaque-cursor",
      clusters: [],
      neurons: [],
      assemblies: [],
      synapses: []
    };
    const request = vi.fn(async () => page);
    const claimForeground = vi.fn(async () => undefined);
    const service = new BrainService(
      repository,
      { request, claimForeground } as unknown as EngineSupervisor
    );

    await expect(
      service.querySubstrate(brain.id, {
        entity: "neurons",
        pageSize: 32,
        region: " semantic ",
        search: " memory ",
        zoom: 4
      })
    ).resolves.toBe(page);
    expect(request).toHaveBeenCalledWith(
      "query_substrate",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: repository.brainDirectory(brain.id),
        query: {
          entity: "neurons",
          cursor: undefined,
          connectedTo: "",
          pageSize: 32,
          region: "semantic",
          search: "memory",
          zoom: 1
        }
      },
      LARGE_FOREGROUND_LOAD_TIMEOUT_MS
    );
    await expect(
      service.querySubstrate(brain.id, { pageSize: 5_001 })
    ).resolves.toBe(page);
    expect(request).toHaveBeenLastCalledWith(
      "query_substrate",
      expect.objectContaining({
        query: expect.objectContaining({ pageSize: 5_001 })
      }),
      LARGE_FOREGROUND_LOAD_TIMEOUT_MS
    );
    await expect(
      service.querySubstrate(brain.id, { entity: "assemblies", connectedTo: "node-a" })
    ).rejects.toThrow(/synapse queries/i);
    expect(request).toHaveBeenCalledTimes(2);
    expect(claimForeground).toHaveBeenCalledTimes(3);
  });

  it("combines load-free parameter accounting and completed dataset coverage", async () => {
    const persistedSubstrateOverview = vi.fn(async () => ({
      brainId: "summary-brain",
      revision: "a".repeat(64),
      source: "validated-persisted-substrate" as const,
      totals: { neurons: 10, assemblies: 2, synapses: 30 },
      parameterAccounting: {
        mutableDenseParameters: 120,
        substrateDynamicSparseSynapses: 30,
        dynamicSparseSynapses: 30,
        totalNeuralParameters: 150,
        countingRule: "non-overlapping fixture"
      }
    }));
    const service = new BrainService(
      {
        brainDirectory: vi.fn(() => "/fixture/summary-brain"),
        persistedSubstrateOverview
      } as unknown as BrainRepository,
      {} as EngineSupervisor
    );
    vi.spyOn(service.datasets, "latestCompletedCoverage").mockResolvedValue({
      brainId: "summary-brain",
      manifestId: "manifest-one",
      manifestHash: "b".repeat(64),
      source: "validated-dataset-progress",
      discoveredFiles: 1,
      processedFiles: 1,
      rejectedFiles: 0,
      discoveredRecords: 21_990,
      processedRecords: 21_990,
      rejectedRecords: 0,
      complete: true,
      updatedAt: "2026-09-06T07:12:58.788Z"
    });

    await expect(service.persistedSubstrateOverview("summary-brain")).resolves.toMatchObject({
      parameterAccounting: { totalNeuralParameters: 150 },
      latestCompletedCoverage: {
        processedRecords: 21_990,
        discoveredRecords: 21_990,
        complete: true
      }
    });
  });

  it("renders authoritative nodes and their paged live pathways without the compatibility mirror", () => {
    const renderer = readFileSync(
      resolve(process.cwd(), "src/renderer/src/App.tsx"),
      "utf8"
    );

    expect(renderer).toContain("const visibleGraphNodes = useMemo<SubstrateMapNode[]>");
    expect(renderer).toContain("...(pathwayPage?.synapses ?? [])");
    expect(renderer).toContain(
      "selectedMapNode?.clusterCount || selectedMapNode?.activityOnly"
    );
    expect(renderer).toContain("A live STDP/activation update invalidates continuation cursors");
    expect(renderer).toContain("Latest change on page");
    expect(renderer.match(/persistedSubstrateOverview\(brain\.id\)/g)).toHaveLength(3);
    expect(renderer).toContain("Loading detailed map…");
    expect(renderer).toContain("persistedOverview.totals.neurons");
    expect(renderer).toContain("overview.parameterAccounting");
    expect(renderer).toContain("overview.latestCompletedCoverage");
    expect(renderer).toContain("Completed dataset traversal");
    expect(renderer).toContain("brain.substrateTotals?.assemblies ?? brain.concepts");
    expect(renderer).toContain("persistedConnections: brain.substrateTotals?.synapses");
    expect(renderer).toContain("mirroredConnections: brain.synapses");
    expect(renderer).not.toContain("mirrored assemblies");
    expect(renderer).toContain(
      'typeof window.omni.brain.onStorageOperation !== "function"'
    );
    expect(renderer).toContain("storage-operation-card");
    expect(renderer).toContain("Cancel {storageOperation.kind}");
    expect(renderer).toContain("}, [brain.id, brain.updatedAt]);");
    expect(renderer).not.toContain("[brain.id, brain.updatedAt, sending]");
  });

  it("reads the committed workspace without loading the neural worker or preflighting writes", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Workspace brain"
    });
    await writeNovaSizedSubstrate(brain.id);
    const committedWorkspace: Omit<WorkspaceSnapshot, "runtimeCard"> = {
      brainId: brain.id,
      queriedAt: new Date().toISOString(),
      contextWindow: {
        capacityTokens: 2048,
        tokenCount: 18,
        tokenHash: "abc",
        sensorySlots: 1,
        extended: true,
        updatedAt: new Date().toISOString()
      },
      latentWorkspace: {
        capacity: 32,
        occupancy: 1,
        items: [{ salience: 0.9, rehearsals: 2 }],
        evictions: 0,
        rehearsals: 1
      },
      liquidState: { dimensions: 64, mean: 0.1, norm: 1.2 },
      hiddenBehavioralPrompt: false,
      rawLongTermTextInjected: false
    };
    const runtimeCard = {
      pretrained_text_cortex: {
        loaded: true,
        resources: {
          residency: "tiered-ram-disk",
          diskPagingPerStep: true,
          estimatedSlowdownPercent: 78,
          actualPlacement: {
            reportedByBackend: true,
            tiers: { cpu: 2, disk: 1 }
          }
        }
      },
      workspace: committedWorkspace
    };
    await writeFile(
      join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify({
        brain_id: brain.id,
        runtime_card: runtimeCard
      }),
      "utf8"
    );
    const request = vi.fn(async () => ({}));
    const claimForeground = vi.fn(async () => undefined);
    const plan = vi.fn(async () => {
      throw new Error("workspace inspection must not run startup admission");
    });
    const service = new BrainService(
      repository,
      { request, claimForeground } as unknown as EngineSupervisor,
      { plan } as unknown as ResourcePlanner
    );

    await expect(service.workspace(brain.id)).resolves.toEqual({
      ...committedWorkspace,
      runtimeCard
    });
    expect(request).not.toHaveBeenCalled();
    expect(claimForeground).not.toHaveBeenCalled();
    expect(plan).not.toHaveBeenCalled();
  });

  it("reads modality readiness without foregrounding or loading the brain", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Modality inspection brain"
    });
    await writeNovaSizedSubstrate(brain.id);
    await writeFile(
      join(repository.brainDirectory(brain.id), "engine", "brain.json"),
      JSON.stringify({
        brain_id: brain.id,
        config: {
          hardware_tier: "gpu",
          vision_enabled: true,
          image_enabled: true,
          audio_enabled: true,
          video_enabled: true
        },
        modality_training: { vision: 1, image: 1, audio: 0, video: 1 },
        installed_modality_packs: []
      }),
      "utf8"
    );
    const request = vi.fn(async () => ({}));
    const claimForeground = vi.fn(async () => undefined);
    const service = new BrainService(
      repository,
      { request, claimForeground } as unknown as EngineSupervisor
    );

    await expect(service.modalityCapabilities(brain.id)).resolves.toMatchObject({
      brainId: brain.id,
      hardwareTier: "gpu",
      imagePerception: true,
      imageGeneration: true,
      audioPerception: false,
      audioGeneration: false,
      videoPerception: true,
      videoGeneration: true,
      synchronizedVideoAudioGeneration: false
    });
    expect(request).not.toHaveBeenCalled();
    expect(claimForeground).not.toHaveBeenCalled();
  });
});
