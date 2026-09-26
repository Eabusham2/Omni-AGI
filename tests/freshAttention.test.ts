import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import {
  ENGINE_REQUEST_NO_DEADLINE,
  type EngineSupervisor
} from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG, type FreshAttentionResult } from "../src/shared/types";

const checksum = "a".repeat(64);
const fastChecksum = "b".repeat(64);

function receipt(brainId: string, operationId: string): FreshAttentionResult {
  return {
    format: "omni-fresh-attention-boundary",
    formatVersion: 1,
    brainId,
    committed: true,
    idempotent: true,
    boundary: {
      format: "omni-fresh-attention-boundary",
      formatVersion: 1,
      operationId,
      epoch: 2,
      createdAt: "2026-09-07T07:40:00.000Z",
      messagesPreserved: 8,
      tracesPreserved: 4,
      synapsesPreserved: 120,
      replayEntries: 6,
      parameterChecksum: checksum,
      fastSynapseChecksum: fastChecksum,
      substrateContentSha256: "c".repeat(64),
      cleared: { recentTokens: 32 }
    },
    parameterChecksum: checksum,
    fastSynapseChecksum: fastChecksum,
    substrateContentSha256: "c".repeat(64),
    messagesPreserved: 8,
    tracesPreserved: 4,
    synapsesPreserved: 120,
    replayEntries: 6,
    pagedCleanupPending: false,
    rawPriorDialogueEligible: false
  };
}

describe("fresh attention service boundary", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-fresh-attention-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("retries the same operation identity after a lost commit acknowledgement", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Fresh attention fixture"
    });
    const claimForeground = vi.fn().mockResolvedValue(undefined);
    const request = vi
      .fn()
      .mockRejectedValueOnce(new Error("worker stopped after commit"))
      .mockResolvedValueOnce(receipt(brain.id, "fresh-operation"));
    const service = new BrainService(
      repository,
      { claimForeground, request } as unknown as EngineSupervisor
    );

    await expect(
      service.startFreshAttention(brain.id, "fresh-operation")
    ).resolves.toMatchObject({
      committed: true,
      idempotent: true,
      rawPriorDialogueEligible: false,
      boundary: { operationId: "fresh-operation", epoch: 2 }
    });

    expect(claimForeground).toHaveBeenCalledOnce();
    expect(request).toHaveBeenCalledTimes(2);
    for (const call of request.mock.calls) {
      expect(call).toEqual([
        "fresh_attention",
        expect.objectContaining({
          brainId: brain.id,
          operationId: "fresh-operation"
        }),
        ENGINE_REQUEST_NO_DEADLINE
      ]);
    }
  });

  it("rejects a receipt whose durable checksums diverge from its boundary", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Fresh attention invalid receipt"
    });
    const invalid = receipt(brain.id, "fresh-invalid");
    invalid.parameterChecksum = "d".repeat(64);
    const service = new BrainService(
      repository,
      {
        claimForeground: vi.fn().mockResolvedValue(undefined),
        request: vi.fn().mockResolvedValue(invalid)
      } as unknown as EngineSupervisor
    );

    await expect(
      service.startFreshAttention(brain.id, "fresh-invalid")
    ).rejects.toThrow("invalid fresh-attention receipt");
  });

  it("keeps the confirmed operation inside collapsed Research diagnostics", async () => {
    const source = await readFile(
      join(process.cwd(), "src/renderer/src/App.tsx"),
      "utf8"
    );
    const start = source.indexOf("function DeviceRuntimeWorkspace(");
    const end = source.indexOf("function TrainingTelemetryMetrics(", start);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(end).toBeGreaterThan(start);
    const workspace = source.slice(start, end);
    const outside = source.slice(0, start) + source.slice(end);

    expect(workspace).toContain(
      '<details className="research-diagnostics resource-settings-diagnostics">'
    );
    expect(workspace).not.toContain(
      '<details open className="research-diagnostics resource-settings-diagnostics">'
    );
    expect(workspace).toContain("Start fresh attention?\\n\\n");
    expect(workspace).toContain("keeps visible chat history");
    expect(workspace).toContain("learned weights and connections");
    expect(workspace).toContain("can re-enter only when you explicitly use the brain.history tool");
    expect(workspace).toContain("window.omni.brain.freshAttention(brain.id)");
    expect(outside).not.toContain("Start fresh attention?");
  });
});
