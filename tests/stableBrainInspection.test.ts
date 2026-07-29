import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
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

  it("forwards a bounded cursor query to the authoritative neural worker", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Inspection brain"
    });
    const page: SubstratePage = {
      brainId: brain.id,
      queriedAt: new Date().toISOString(),
      entity: "neurons",
      zoom: 1,
      totals: { neurons: 2, assemblies: 1, synapses: 1 },
      matched: 2,
      hasMore: true,
      nextCursor: "opaque-cursor",
      clusters: [],
      neurons: [],
      assemblies: [],
      synapses: []
    };
    const request = vi.fn(async () => page);
    const service = new BrainService(
      repository,
      { request } as unknown as EngineSupervisor
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
          pageSize: 32,
          region: "semantic",
          search: "memory",
          zoom: 1
        }
      },
      30_000
    );
    await expect(
      service.querySubstrate(brain.id, { pageSize: 5_001 })
    ).rejects.toThrow(/page size/i);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("returns the worker-owned transient workspace rather than the desktop mirror", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Workspace brain"
    });
    const snapshot: WorkspaceSnapshot = {
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
    const request = vi.fn(async () => snapshot);
    const service = new BrainService(
      repository,
      { request } as unknown as EngineSupervisor
    );

    await expect(service.workspace(brain.id)).resolves.toBe(snapshot);
    expect(request).toHaveBeenCalledWith(
      "workspace",
      {
        brainId: brain.id,
        config: brain.config,
        storagePath: repository.brainDirectory(brain.id)
      },
      30_000
    );
  });
});
