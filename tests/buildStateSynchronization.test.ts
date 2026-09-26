import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainService } from "../src/main/brainService";
import {
  brainMetrics,
  BrainRepository
} from "../src/main/brainRepository";
import type { InitialFoundationPlan } from "../src/main/buildInitializationPlan";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { GIB, ResourcePlanner } from "../src/main/resourcePlanner";
import { substrateOverviewCount } from "../src/renderer/src/uiPresentation";
import { DEFAULT_CONFIG } from "../src/shared/types";

const foundation: InitialFoundationPlan = {
  hardwareTier: "micro",
  modalities: [],
  origin: "ground-up",
};

function testResourcePlanner(root: string): ResourcePlanner {
  return new ResourcePlanner(root, {
    readResources: async () => ({
      totalMemoryBytes: 16 * GIB,
      availableMemoryBytes: 13 * GIB,
      diskTotalBytes: 500 * GIB,
      diskFreeBytes: 240 * GIB
    }),
    benchmark: async () => ({
      measuredAt: "2026-09-07T00:00:00.000Z",
      sampleBytes: 16 * 1024 * 1024,
      memoryBytesPerSecond: 12 * GIB,
      storageBytesPerSecond: 734_003_201,
      cacheHit: false
    })
  });
}

function workerSummary(
  brainId: string,
  plasticityEvents: number,
  overrides: Record<string, unknown> = {}
): Record<string, unknown> {
  return {
    brainId,
    metrics: {
      plasticityEvents,
      trainingSources: 0,
      counters: {
        inference_count: 3,
        consolidation_cycles: 2
      }
    },
    ...overrides
  };
}

describe("Build worker-to-desktop state synchronization", () => {
  const roots: string[] = [];

  afterEach(async () => {
    await Promise.all(
      roots.splice(0).map((root) =>
        rm(root, { recursive: true, force: true })
      )
    );
  });

  it("refreshes durable counters when a failed foundation retry succeeds", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-build-sync-retry-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const created = await repository.create(
      { ...DEFAULT_CONFIG, name: "Retry synchronization" },
      {
        initializing: true,
        recovery: { foundation, resources: [] }
      }
    );
    await repository.failInitialization(
      created.id,
      "foundation",
      new Error("fixture resource pause")
    );
    await repository.retryInitialization(created.id);

    const request = vi.fn(async (_method: string, params: Record<string, unknown>) =>
      workerSummary(String(params.brainId), 126_176)
    );
    const service = new BrainService(
      repository,
      { request } as unknown as EngineSupervisor
    );
    const resumed = await service.resumeFoundation(created.id, foundation);
    expect(resumed.readiness.state).toBe("initializing");
    expect(resumed.counters).toEqual({
      plasticityEvents: 126_176,
      inferenceCount: 3,
      consolidationCycles: 2
    });

    const completed = await repository.completeInitialization(created.id);
    const [summary] = await repository.list();
    expect(completed.readiness.state).toBe("ready");
    expect(completed.counters.plasticityEvents).toBe(126_176);
    expect(summary).toMatchObject({
      id: created.id,
      neuralUpdates: 126_176,
      inferenceCount: 3,
      trainingSources: 0
    });
    expect(substrateOverviewCount({
      plasticityEvents: completed.counters.plasticityEvents
    })).toBe(1);
  });

  it("commits initial-data sources and the final absolute worker counters", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-build-sync-data-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const sourcePath = join(
      root,
      "validation-00000-of-00001-869c898b519ad725.parquet"
    );
    const sourceText = "One tiny story.\nA second tiny story.\n";
    await writeFile(sourcePath, sourceText, "utf8");
    const request = vi.fn(async (
      method: string,
      params: Record<string, unknown>
    ) => {
      const brainId = String(params.brainId);
      if (method === "create") return workerSummary(brainId, 126_176);
      if (method !== "ingest") throw new Error(`Unexpected method ${method}`);
      const coverage = {
        discoveredFiles: 1,
        completedFiles: 1,
        processedFiles: 1,
        rejectedFiles: 0,
        discoveredRecords: 21_990,
        processedRecords: 21_990,
        rejectedRecords: 0,
        processedBytes: Buffer.byteLength(sourceText),
        shards: 1,
        modalityCounts: { parquet: 21_990 },
        errors: [],
        errorCount: 0,
        errorsTruncated: false,
        complete: true
      };
      return workerSummary(brainId, 17_585_752, {
        source: {
          kind: "parquet",
          learned_ideas: 1,
          learned_concepts: 11_950,
          synaptic_update_events: 51_405_966,
          parameter_update_steps: 2_472,
          parameter_checksum_changed: true,
          warnings: [],
          coverage
        },
        coverage,
        parameterChecksumBefore: "a".repeat(64),
        parameterChecksumAfter: "b".repeat(64)
      });
    });
    const service = new BrainService(
      repository,
      {
        request,
        tryRequest: vi.fn(async () => undefined)
      } as unknown as EngineSupervisor,
      testResourcePlanner(repository.root)
    );
    const building = await service.create({
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Initial data synchronization" }
    });
    expect(building.counters.plasticityEvents).toBe(126_176);

    const manifest = await service.previewDataset(building.id, [sourcePath]);
    const run = await service.ingestManifest(
      building.id,
      manifest.id,
      "pretrain"
    );
    expect(run.coverage).toMatchObject({
      complete: true,
      processedRecords: 21_990,
      processedBytes: Buffer.byteLength(sourceText)
    });
    const completed = await repository.completeInitialization(building.id);
    expect(completed.trainingSources).toHaveLength(1);
    expect(completed.trainingSources[0]).toMatchObject({
      name: "validation-00000-of-00001-869c898b519ad725.parquet",
      kind: "parquet",
      learnedIdeas: 1,
      learnedConcepts: 11_950,
      learnedSynapses: 51_405_966,
      learnedParameterSteps: 2_472,
      parametersChanged: true,
      rawTextRetained: false,
      policy: "pretrain"
    });
    expect(completed.counters.plasticityEvents).toBe(17_585_752);
    expect(brainMetrics(completed)).toMatchObject({
      trainingSources: 1,
      plasticityEvents: 17_585_752,
      inferenceCount: 3
    });
    await expect(repository.list()).resolves.toEqual([
      expect.objectContaining({
        id: building.id,
        neuralUpdates: 17_585_752,
        trainingSources: 1
      })
    ]);
    expect(substrateOverviewCount({
      plasticityEvents: completed.counters.plasticityEvents
    })).toBe(1);
  });
});
