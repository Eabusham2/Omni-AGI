import { createHash, randomUUID } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrainService,
  normalizeChatEngineEvent,
  normalizeModalityGenerateRequest,
  normalizeModalityPreview
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { GIB, ResourcePlanner } from "../src/main/resourcePlanner";
import {
  DEFAULT_CONFIG,
  type TrainingSource
} from "../src/shared/types";

function emptySafetensors(label: string): Buffer {
  const metadata = JSON.stringify({ __metadata__: { fixture: label } });
  const header = Buffer.from(metadata.padEnd(128, " "));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  return Buffer.concat([prefix, header]);
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

// This 10k-row stress fixture exceeded 300s on Windows x64 CI. Keep it
// runnable manually and on non-Windows CI while Windows CI runs smaller merges.
const itLargeMerge = process.platform === "win32" && process.env.CI === "true"
  ? it.skip
  : it;

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

describe("streaming media preview validation", () => {
  it("accepts bounded media previews and rejects executable or mismatched data URLs", () => {
    expect(normalizeModalityPreview({
      revision: 3,
      progress: 1.7,
      statusLabel: "Decoding\0 now",
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,cHJldmlldw==",
      artifactPath: "artifacts/\0preview.png"
    })).toEqual({
      schemaVersion: 1,
      revision: 3,
      progress: 1,
      statusLabel: "Decoding now",
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,cHJldmlldw==",
      path: undefined,
      artifactPath: "artifacts/preview.png"
    });
    expect(normalizeModalityPreview({
      schemaVersion: 1,
      revision: 6,
      progress: 0.6,
      statusLabel: "Neural codec waveform 128/256 samples",
      mimeType: "audio/wav",
      dataUrl: "data:audio/wav;base64,UklGRg==",
      modality: "audio",
      stage: "codec-waveform",
      completedUnits: 2,
      totalUnits: 3,
      sampleCount: 128,
      totalSamples: 256,
      durationMs: 8,
      sampleRate: 32000,
      hardwareTier: "micro",
      cadence: "hardware-aware-bounded-synchronous",
      producer: "same-brain-decoder",
      payloadSha256: "a".repeat(64),
      actualDecoderOutput: true,
      spatialResolutionReduced: false,
      ideaSource: "active-working-memory",
      activeAssemblyCount: 2,
      promptProvided: false,
      hardwareScaled: true,
      modelDefinedMaximum: null,
      trained: false,
      trainingState: "untrained-diagnostic",
      semanticQualityClaimed: false,
      partialCoverage: true,
      coveredFraction: 0.5
    })).toMatchObject({
      schemaVersion: 1,
      revision: 6,
      modality: "audio",
      stage: "codec-waveform",
      sampleCount: 128,
      totalSamples: 256,
      sampleRate: 32000,
      producer: "same-brain-decoder",
      payloadSha256: "a".repeat(64),
      actualDecoderOutput: true,
      spatialResolutionReduced: false,
      ideaSource: "active-working-memory",
      hardwareScaled: true,
      modelDefinedMaximum: null,
      trained: false,
      trainingState: "untrained-diagnostic",
      semanticQualityClaimed: false,
      partialCoverage: true,
      coveredFraction: 0.5
    });
    expect(normalizeModalityPreview({
      schemaVersion: 2,
      revision: 7,
      statusLabel: "future schema"
    })).toBeUndefined();
    expect(normalizeModalityPreview({
      revision: 4,
      mimeType: "image/png",
      dataUrl: "javascript:alert(1)"
    })).toBeUndefined();
    expect(normalizeModalityPreview({
      revision: 5,
      mimeType: "video/mp4",
      dataUrl: "data:image/png;base64,cHJldmlldw=="
    })).toBeUndefined();
  });

  it("retains a bounded six-second WAV preview above the old 64 KiB ceiling", () => {
    const dataUrl = `data:audio/wav;base64,${Buffer.alloc(183_852).toString("base64")}`;
    const normalized = normalizeModalityPreview({
      revision: 2,
      mimeType: "audio/wav",
      dataUrl,
      statusLabel: "Neural codec waveform 91904/91904 samples"
    });
    expect(dataUrl.length).toBeGreaterThan(96 * 1024);
    expect(normalized?.dataUrl).toBe(dataUrl);
  });
});

describe("hardware-scaled media request validation", () => {
  it("preserves exact requested fields without an arbitrary model-size clamp", () => {
    expect(normalizeModalityGenerateRequest({
      brainId: "brain-media",
      modality: "video",
      prompt: "wide exact timeline",
      settings: {
        outputMode: "exact",
        width: 12_001,
        height: 7_003,
        durationMs: 90_000.5,
        sampleRate: 96_000,
        fps: 240,
        includeAudio: true
      }
    })).toMatchObject({
      settings: {
        outputMode: "exact",
        width: 12_001,
        height: 7_003,
        durationMs: 90_000.5,
        sampleRate: 96_000,
        fps: 240,
        includeAudio: true
      }
    });
  });

  it("defaults provided settings to Auto and rejects malformed exact/container fields", () => {
    expect(normalizeModalityGenerateRequest({
      brainId: "brain-media",
      modality: "audio",
      settings: { durationMs: 2_500 }
    }).settings).toEqual({ outputMode: "auto", durationMs: 2_500 });
    expect(() => normalizeModalityGenerateRequest({
      brainId: "brain-media",
      modality: "audio",
      settings: { outputMode: "exact", sampleRate: 16_000 }
    })).toThrow(/dimensions or duration/i);
    expect(() => normalizeModalityGenerateRequest({
      brainId: "brain-media",
      modality: "video",
      settings: { outputMode: "exact", durationMs: 1_000, fps: 65_536 }
    })).toThrow(/video fps/i);
  });
});

describe("post-reply chat phase validation", () => {
  const validPhase = {
    type: "chat-phase",
    brainId: "brain-phase",
    streamId: "turn-phase",
    sequence: 7,
    data: {
      phase: "reply-complete-learning",
      replyComplete: true,
      turnCommitted: false,
      learning: true,
      saving: true
    }
  } as const;

  it("flattens only the exact provisional learning-and-save event", () => {
    expect(normalizeChatEngineEvent(validPhase, "brain-phase")).toEqual({
      type: "chat-phase",
      sequence: 7,
      phase: "reply-complete-learning",
      replyComplete: true,
      turnCommitted: false,
      learning: true,
      saving: true
    });

    expect(normalizeChatEngineEvent({
      ...validPhase,
      data: { ...validPhase.data, turnCommitted: true }
    }, "brain-phase")).toBeUndefined();
    expect(normalizeChatEngineEvent({
      ...validPhase,
      data: { ...validPhase.data, saving: false }
    }, "brain-phase")).toBeUndefined();
    expect(normalizeChatEngineEvent(validPhase, "another-brain")).toBeUndefined();
  });
});

describe("BrainService ground-up creation", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;
  let tryRequest: ReturnType<typeof vi.fn>;
  let request: ReturnType<typeof vi.fn>;
  let service: BrainService;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-service-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
    tryRequest = vi.fn(async () => undefined);
    request = vi.fn(async () => ({
      runtimeCard: {
        origin_kind: "ground-up",
        pretrained: false,
        hidden_behavioral_prompt: false,
        reward_model: false,
        rlhf: false
      }
    }));
    service = new BrainService(
      repository,
      { tryRequest, request } as unknown as EngineSupervisor,
      testResourcePlanner(repository.root)
    );
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("builds a ground-up OmniCortex with no external foundation", async () => {
    const built = await service.create({
      hardwareTier: "micro",
      modalities: ["vision", "image", "audio", "video"],
      config: { ...DEFAULT_CONFIG, name: "Ground-up mind" }
    });

    expect(request).toHaveBeenCalledTimes(1);
    expect(built.config.workingMemorySlots).toBe(8_192);
    expect(request).toHaveBeenCalledWith(
      "create",
      expect.objectContaining({
        brainId: built.id,
        origin: "ground-up",
        hardwareTier: "micro",
        config: expect.objectContaining({ workingMemorySlots: 8_192 }),
        modalities: ["vision", "image", "audio", "video"],
        storagePath: repository.brainDirectory(built.id)
      }),
      300_000
    );
    expect(built.provenance).toEqual({ originKind: "ground-up" });
    expect(built.toolPermissions?.map(({ toolId }) => toolId)).toEqual(
      expect.arrayContaining(["system.files", "system.shell"])
    );
    expect(built.toolPermissions?.some(({ toolId }) => toolId.startsWith("windows.")))
      .toBe(false);
    expect(built.journal?.at(-1)?.summary).toContain("native core: locally initialized");
    expect(built.journal?.at(-1)?.summary).not.toContain("external foundation");
    expect(tryRequest).not.toHaveBeenCalled();
  });

  it("learns visible action evidence without adding a synthetic conversation turn", async () => {
    const createdAt = "2026-09-13T12:00:00.000Z";
    let brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Structured action learner"
    });
    brain.messages = [
      {
        id: "human-visible",
        role: "human",
        content: "Inspect the project files.",
        createdAt
      },
      {
        id: "brain-visible",
        role: "brain",
        content: "I inspected the visible files.",
        createdAt
      }
    ];
    brain.traces = [{
      id: "trace-visible",
      createdAt,
      input: "Inspect the project files.",
      seed: 7,
      runtime: "adaptive-core",
      activatedConcepts: [],
      recalledIdeas: [],
      driveScores: { novelty: 0, coherence: 1, curiosity: 0 },
      branches: 1,
      selectedBranch: 1,
      steps: [],
      note: "Visible committed turn"
    }];
    await repository.save(brain);
    brain = await repository.get(brain.id);
    const messagesBefore = structuredClone(brain.messages);
    const tracesBefore = structuredClone(brain.traces);

    const learned = await service.learnStructuredExperience(brain.id, {
      content:
        "[Visible structured action result]\ntool: system.files\naction: list\nresult:\n{\"entries\":[\"README.md\"]}",
      name: "Chat system.files result",
      sourceLabel: "chat visible action evidence",
      license: "Locally observed tool result"
    });

    expect(request).toHaveBeenLastCalledWith(
      "ingest",
      expect.objectContaining({
        brainId: brain.id,
        kind: "text",
        policy: "pretrain",
        text: expect.stringContaining("[Visible structured action result]")
      }),
      86_400_000,
      undefined
    );
    expect(learned.brain.messages).toEqual(messagesBefore);
    expect(learned.source).toMatchObject({
      name: "Chat system.files result",
      rawTextRetained: false,
      policy: "pretrain",
      license: "Locally observed tool result"
    });
    expect(learned.brain.journal?.at(-1)).toMatchObject({
      kind: "learning",
      summary: "Learned chat visible action evidence into neural state."
    });
    expect(learned.brain.journal?.at(-1)?.detail).toContain("hiddenPrompt=false");

    const reloaded = await repository.get(brain.id);
    expect(reloaded.messages).toEqual(messagesBefore);
    expect(reloaded.traces).toEqual(tracesBefore);
  });

  it("requires a hash-bound neural checkpoint before publishing one recovery point", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Recovery checkpoint gate"
    });
    const failedOperation = randomUUID();
    request.mockRejectedValueOnce(new Error("checkpoint flush failed"));
    await expect(
      service.createRecoveryPoint(
        brain.id,
        "must not publish",
        failedOperation
      )
    ).rejects.toThrow(/checkpoint flush failed/i);
    await expect(repository.listSnapshots(brain.id)).resolves.toEqual([]);

    const operationId = randomUUID();
    const parameterChecksum = "a".repeat(64);
    const substrateContentSha256 = "b".repeat(64);
    const mutableStateContentSha256 = "c".repeat(64);
    const packedContentSha256 = "d".repeat(64);
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engineDirectory, "packed-ternary"), { recursive: true });
    const packedBytes = Buffer.from(JSON.stringify({
      contentSha256: packedContentSha256
    }));
    const metadataBytes = Buffer.from(JSON.stringify({
      brain_id: brain.id,
      substrate: { persistence: { contentSha256: substrateContentSha256 } },
      mutable_state: { contentSha256: mutableStateContentSha256 },
      packed_ternary_manifest: {
        contentSha256: packedContentSha256,
        parameterChecksum
      }
    }));
    await Promise.all([
      writeFile(join(engineDirectory, "brain.json"), metadataBytes),
      writeFile(join(engineDirectory, "packed-ternary", "manifest.json"), packedBytes)
    ]);
    const mismatchedOperation = randomUUID();
    request.mockResolvedValueOnce({
      format: "omni-neural-checkpoint",
      formatVersion: 1,
      brainId: brain.id,
      operationId: mismatchedOperation,
      committed: true,
      createdAt: "2026-09-12T00:00:00.000Z",
      parameterChecksum,
      metadataSha256: "f".repeat(64),
      substrateContentSha256,
      mutableStateContentSha256,
      packedManifestSha256: sha256(packedBytes),
      packedContentSha256,
      snapshotCreated: false
    });
    await expect(
      service.createRecoveryPoint(
        brain.id,
        "mismatched files",
        mismatchedOperation
      )
    ).rejects.toThrow(/does not match its committed files/i);
    await expect(repository.listSnapshots(brain.id)).resolves.toEqual([]);

    request.mockResolvedValueOnce({
      format: "omni-neural-checkpoint",
      formatVersion: 1,
      brainId: brain.id,
      operationId,
      committed: true,
      createdAt: "2026-09-12T00:00:00.000Z",
      parameterChecksum,
      metadataSha256: sha256(metadataBytes),
      substrateContentSha256,
      mutableStateContentSha256,
      packedManifestSha256: sha256(packedBytes),
      packedContentSha256,
      snapshotCreated: false
    });
    const repositorySnapshot = vi.spyOn(repository, "snapshot").mockImplementation(
      async (brainId, label, _operation, prepare) => {
        await prepare?.();
        return {
          id: "host-recovery-point",
          brainId,
          label: label ?? "Recovery point",
          createdAt: "2026-09-12T00:00:01.000Z",
          checksum: "e".repeat(64),
          metrics: {
            concepts: 0,
            synapses: 0,
            activeSynapses: 0,
            ideas: 0,
            messages: 0,
            trainingSources: 0,
            averageStability: 0,
            plasticityEvents: 0,
            inferenceCount: 0,
            estimatedBytes: 1
          }
        };
      }
    );
    const snapshot = await service.createRecoveryPoint(
      brain.id,
      "single host recovery point",
      operationId
    );
    expect(snapshot.label).toBe("single host recovery point");
    expect(repositorySnapshot).toHaveBeenCalledOnce();
    await expect(
      stat(join(repository.brainDirectory(brain.id), "engine", "snapshots"))
    ).rejects.toMatchObject({ code: "ENOENT" });
    expect(request).toHaveBeenLastCalledWith(
      "checkpoint",
      expect.objectContaining({ brainId: brain.id, operationId }),
      0,
      undefined,
      "foreground",
      expect.objectContaining({ requestId: operationId, brainId: brain.id })
    );
    expect(tryRequest).not.toHaveBeenCalled();
  });

  it("fails closed before persistence when live resource preflight is unavailable", async () => {
    const unplanned = new BrainService(
      repository,
      { tryRequest, request } as unknown as EngineSupervisor
    );

    await expect(unplanned.create({
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Missing resource preflight" }
    })).rejects.toThrow(/requires live resource preflight/i);

    expect(request).not.toHaveBeenCalled();
    await expect(repository.list()).resolves.toEqual([]);
  });

  it("routes existing-instance selectors away from new creation", async () => {
    await expect(
      service.create({
        origin: "starter",
        foundationModelId: "falcon-e-1b-base-f4001b8",
        foundationRiskAcknowledged: false,
        config: { ...DEFAULT_CONFIG, name: "Unacknowledged research base" }
      } as unknown as Parameters<typeof service.create>[0])
    ).rejects.toThrow(/native OmniCortex origin/i);

    await expect(service.create({
      origin: "ground-up",
      foundationModelId: "falcon-e-1b-base-f4001b8",
      foundationRiskAcknowledged: true,
      config: { ...DEFAULT_CONFIG, name: "No attached foundation" }
    } as unknown as Parameters<typeof service.create>[0])).rejects.toThrow(/native OmniCortex origin/i);

    await expect(service.create({
      starterUrl: "https://catalog.example/legacy.omni",
      config: { ...DEFAULT_CONFIG, name: "Use import" }
    } as unknown as Parameters<typeof service.create>[0])).rejects.toThrow(/native OmniCortex origin/i);

    expect(request).not.toHaveBeenCalled();
  });

  it("ignores the legacy foundation environment override for new Build", async () => {
    vi.stubEnv("OMNI_FOUNDATION_MODEL_ID", "falcon-e-3b-base-ad18b07");
    try {
      const built = await service.create({
        config: { ...DEFAULT_CONFIG, name: "Environment-safe ground-up mind" }
      });
      expect(request).toHaveBeenLastCalledWith(
        "create",
        expect.objectContaining({
          brainId: built.id,
          origin: "ground-up"
        }),
        300_000
      );
      expect(request.mock.calls.some(([method]) => method === "foundation.list"))
        .toBe(false);
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("defaults the stable public API to mandatory ground-up creation", async () => {
    const groundUp = await service.create({
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Implicit ground-up" }
    });
    expect(request).toHaveBeenLastCalledWith(
      "create",
      expect.objectContaining({
        brainId: groundUp.id,
        origin: "ground-up"
      }),
      300_000
    );

    await expect(service.create({
      origin: "blank",
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Rejected legacy blank" }
    } as unknown as Parameters<typeof service.create>[0])).rejects.toThrow(/native OmniCortex origin/i);
  });

  it("projects legacy Windows grants onto platform-neutral system capabilities", async () => {
    let legacy = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Imported legacy permissions"
    });
    legacy.toolPermissions = [
      {
        toolId: "windows.files",
        label: "Windows files",
        level: "full",
        updatedAt: legacy.createdAt
      },
      {
        toolId: "windows.powershell",
        label: "PowerShell",
        level: "off",
        updatedAt: legacy.createdAt
      }
    ];
    legacy = await repository.save(legacy);

    const projected = await service.listToolPermissions(legacy.id);
    expect(projected).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ toolId: "system.files", level: "full" }),
        expect.objectContaining({ toolId: "system.shell", level: "off" })
      ])
    );
    expect(projected.some(({ toolId }) => toolId.startsWith("windows."))).toBe(false);
    const updated = await service.setToolPermission(
      legacy.id,
      "windows.powershell",
      "ask"
    );
    expect(updated).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ toolId: "system.shell", level: "ask" })
      ])
    );
    expect(updated.some(({ toolId }) => toolId.startsWith("windows."))).toBe(false);
  });

  it("keeps legacy checkpoint handling on the explicit import path", async () => {
    const importSpy = vi.spyOn(service, "importUrl");
    await expect(service.create({
      origin: "starter",
      starterUrl: "https://catalog.example/pretrained.omni",
      config: { ...DEFAULT_CONFIG, name: "Must import instead" }
    } as unknown as Parameters<typeof service.create>[0])).rejects.toThrow(/native OmniCortex origin/i);
    expect(importSpy).not.toHaveBeenCalled();
    expect(request).not.toHaveBeenCalled();
  });
});

describe("BrainService reviewed subagent overlay merges", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;
  let request: ReturnType<typeof vi.fn>;
  let service: BrainService;
  let sourceRevision: number;
  let targetRevision: number;
  let overlayMerged: boolean;
  let authoritativeAdditions: {
    neurons: number;
    assemblies: number;
    synapses: number;
    replayExamples: number;
  };

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-merge-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
    sourceRevision = 0;
    targetRevision = 0;
    overlayMerged = false;
    authoritativeAdditions = {
      neurons: 0,
      assemblies: 0,
      synapses: 0,
      replayExamples: 1
    };
    request = vi.fn(async (
      method: string,
      params: Record<string, unknown>
    ) => {
      const stateKey = `${sourceRevision}:${targetRevision}:${overlayMerged}`;
      const sourceCounts = {
        neurons: 20,
        assemblies: 8,
        synapses: 40,
        replayExamples: 6
      };
      const additions = overlayMerged
        ? { neurons: 0, assemblies: 0, synapses: 0, replayExamples: 0 }
        : authoritativeAdditions;
      const preview = {
        schemaVersion: 1,
        engineSchemaVersion: 1,
        sourceBrainId: params.sourceBrainId,
        targetBrainId: params.targetBrainId,
        digest: sha256(`digest:${stateKey}`),
        sourceStateSha256: sha256(`source-state:${stateKey}`),
        targetStateSha256: sha256(
          `target-state:${targetRevision}:${overlayMerged}`
        ),
        sourceParameterSha256: sha256(`source-parameters:${sourceRevision}`),
        targetParameterSha256: sha256(
          `target-parameters:${targetRevision}:${overlayMerged}`
        ),
        sourceConfigSha256: sha256("source-config"),
        targetConfigSha256: sha256("target-config"),
        sourceCounts,
        targetCounts: {
          neurons: sourceCounts.neurons - additions.neurons,
          assemblies: sourceCounts.assemblies - additions.assemblies,
          synapses: sourceCounts.synapses - additions.synapses,
          replayExamples:
            sourceCounts.replayExamples - additions.replayExamples
        },
        additions,
        duplicates: {
          neurons: sourceCounts.neurons - additions.neurons,
          assemblies: sourceCounts.assemblies - additions.assemblies,
          synapses: sourceCounts.synapses - additions.synapses,
          replayExamples:
            sourceCounts.replayExamples - additions.replayExamples
        },
        divergent: { neurons: 0, assemblies: 0, synapses: 0 },
        weightsAveraged: false
      };
      if (method === "preview_overlay") return preview;
      if (method === "merge_overlay") {
        if (params.expectedPreviewDigest !== preview.digest) {
          throw new Error("authoritative overlay changed after review");
        }
        overlayMerged = true;
        return {
          weightsAveraged: false,
          replayExamples: additions.replayExamples,
          reviewedDigest: preview.digest
        };
      }
      throw new Error(`Unexpected worker method ${method}.`);
    });
    service = new BrainService(
      repository,
      { request } as unknown as EngineSupervisor
    );
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("binds previewed ideas, evidence, and branch artifacts to a stale-safe hash", async () => {
    authoritativeAdditions = {
      neurons: 1,
      assemblies: 1,
      synapses: 0,
      replayExamples: 1
    };
    const now = new Date().toISOString();
    let target = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Merge target",
      memoryRecipe: "total-recall",
      retainSourceText: true
    });
    target.concepts.shared = {
      id: "shared",
      label: "shared",
      activation: 0.2,
      importance: 0.4,
      uncertainty: 0.3,
      exposures: 2,
      createdAt: now,
      lastActivatedAt: now,
      aliases: []
    };
    target = await repository.save(target);
    const source = await repository.fork(target.id, "Evidence branch");
    const branch = await repository.get(source.id);
    branch.concepts.shared!.exposures = 99;
    branch.concepts.novel = {
      id: "novel",
      label: "novel",
      activation: 0.8,
      importance: 0.7,
      uncertainty: 0.1,
      exposures: 1,
      createdAt: now,
      lastActivatedAt: now,
      aliases: []
    };
    branch.ideas.push({
      id: randomUUID(),
      statement: "A reviewed branch-local finding.",
      fingerprint: "reviewed-finding",
      conceptIds: ["novel"],
      kind: "knowledge",
      source: "document",
      confidence: 0.8,
      importance: 0.7,
      rehearsals: 1,
      createdAt: now
    });
    const evidenceBytes = Buffer.from("branch evidence bytes");
    const evidenceHash = await repository.storeBlob(evidenceBytes);
    branch.trainingSources.push({
      id: randomUUID(),
      name: "evidence.txt",
      path: join(temporaryRoot, "outside-source.txt"),
      kind: "text",
      bytes: evidenceBytes.byteLength,
      learnedIdeas: 1,
      learnedConcepts: 1,
      learnedSynapses: 0,
      importedAt: now,
      rawTextRetained: true,
      rawText: evidenceBytes.toString("utf8"),
      contentHash: evidenceHash,
      blobHash: evidenceHash,
      policy: "archive",
      license: "Fixture"
    });
    await repository.save(branch);

    const sourceArtifact = join(repository.brainDirectory(source.id), "artifacts", "report.txt");
    const engineArtifact = join(
      repository.brainDirectory(source.id),
      "engine",
      "artifacts",
      "image.bin"
    );
    await Promise.all([
      mkdir(join(repository.brainDirectory(source.id), "artifacts"), { recursive: true }),
      mkdir(join(repository.brainDirectory(source.id), "engine", "artifacts"), {
        recursive: true
      }),
      mkdir(join(repository.brainDirectory(target.id), "engine"), { recursive: true })
    ]);
    await Promise.all([
      writeFile(sourceArtifact, "first report"),
      writeFile(engineArtifact, Buffer.from([1, 2, 3, 4])),
      writeFile(
        join(repository.brainDirectory(target.id), "engine", "core.safetensors"),
        emptySafetensors("target-core")
      ),
      writeFile(
        join(repository.brainDirectory(source.id), "engine", "core.safetensors"),
        emptySafetensors("source-core")
      )
    ]);
    const targetCoreBefore = await readFile(
      join(repository.brainDirectory(target.id), "engine", "core.safetensors")
    );

    const workerStalePreview = await service.previewMerge(source.id, target.id);
    sourceRevision += 1;
    await expect(
      service.merge(source.id, target.id, workerStalePreview.reviewToken)
    ).rejects.toThrow(/preview is stale/i);
    expect(
      request.mock.calls.some(([method]) => method === "merge_overlay")
    ).toBe(false);

    const targetStalePreview = await service.previewMerge(source.id, target.id);
    targetRevision += 1;
    await expect(
      service.merge(source.id, target.id, targetStalePreview.reviewToken)
    ).rejects.toThrow(/preview is stale/i);
    expect(
      request.mock.calls.some(([method]) => method === "merge_overlay")
    ).toBe(false);

    const stalePreview = await service.previewMerge(source.id, target.id);
    expect(stalePreview).toMatchObject({
      newConcepts: 1,
      newIdeas: 1,
      newEvidence: 1,
      newFiles: 3,
      duplicateFiles: 0,
      skippedFiles: 0
    });
    expect(stalePreview.reviewToken).toMatch(/^[a-f0-9]{64}$/);
    expect(stalePreview.conflicts.join(" ")).toMatch(/target versions will be preserved/i);
    expect(stalePreview.files).toHaveLength(3);
    expect(
      stalePreview.files.every(
        (file) =>
          file.destinationPath.includes(file.sha256) &&
          !file.destinationPath.includes(source.id)
      )
    ).toBe(true);

    await writeFile(sourceArtifact, "changed after review");
    await expect(
      service.merge(source.id, target.id, stalePreview.reviewToken)
    ).rejects.toThrow(/preview is stale/i);
    expect(
      request.mock.calls.some(([method]) => method === "merge_overlay")
    ).toBe(false);

    const reviewed = await service.previewMerge(source.id, target.id);
    const merged = await service.merge(source.id, target.id, reviewed.reviewToken);
    expect(request).toHaveBeenCalledWith(
      "merge_overlay",
      expect.objectContaining({
        sourceBrainId: source.id,
        targetBrainId: target.id,
        expectedPreviewDigest: reviewed.substrate.digest
      }),
      600_000
    );
    expect(reviewed.substrate).toMatchObject({
      sourceStateSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      targetStateSha256: expect.stringMatching(/^[a-f0-9]{64}$/),
      additions: {
        neurons: 1,
        assemblies: 1,
        synapses: 0,
        replayExamples: 1
      },
      weightsAveraged: false
    });
    expect(merged.concepts.shared!.exposures).toBe(2);
    expect(merged.concepts.novel).toBeDefined();
    expect(merged.ideas.some((idea) => idea.fingerprint === "reviewed-finding")).toBe(true);
    expect(merged.trainingSources).toHaveLength(1);
    expect(merged.trainingSources[0]?.blobHash).toBe(evidenceHash);
    expect(merged.trainingSources[0]?.path).toContain(repository.brainDirectory(target.id));
    expect(await readFile(merged.trainingSources[0]!.path!, "utf8")).toBe(
      evidenceBytes.toString("utf8")
    );
    for (const file of reviewed.files) {
      expect(
        await readFile(
          join(repository.brainDirectory(target.id), ...file.destinationPath.split("/"))
        )
      ).toEqual(
        file.kind === "evidence"
          ? evidenceBytes
          : file.sourcePath.endsWith("report.txt")
            ? Buffer.from("changed after review")
            : Buffer.from([1, 2, 3, 4])
      );
    }
    expect(
      await readFile(
        join(
          repository.brainDirectory(target.id),
          "artifacts",
          "merge-manifests",
          `${reviewed.reviewToken}.json`
        ),
        "utf8"
      )
    ).toContain('"wholeModelWeightsAveraged": false');
    expect(
      await readFile(join(repository.brainDirectory(target.id), "engine", "core.safetensors"))
    ).toEqual(targetCoreBefore);

    const duplicatePreview = await service.previewMerge(source.id, target.id);
    expect(duplicatePreview).toMatchObject({
      newConcepts: 0,
      newIdeas: 0,
      newEvidence: 0,
      duplicateEvidence: 1,
      newFiles: 0,
      duplicateFiles: 2
    });
  });

  itLargeMerge("streams all 10k+ novel evidence rows through the resumable merge plan", async () => {
    const target = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large merge target"
    });
    const source = await repository.fork(target.id, "Large merge source");
    const branch = await repository.get(source.id);
    const count = 10_050;
    branch.trainingSources = Array.from({ length: count }, (_, index): TrainingSource => ({
      id: `large-source-${String(index).padStart(6, "0")}`,
      name: `source ${index}`,
      kind: "text",
      bytes: index + 1,
      learnedIdeas: index % 5,
      learnedConcepts: index % 7,
      learnedSynapses: index % 11,
      importedAt: new Date(1_700_000_000_000 + index).toISOString(),
      rawTextRetained: false,
      contentHash: sha256(`large-source-${index}`),
      policy: "pretrain"
    }));
    await repository.save(branch);

    const preview = await service.previewMerge(source.id, target.id);
    expect(preview).toMatchObject({ newEvidence: count, duplicateEvidence: 0 });
    const merged = await service.merge(source.id, target.id, preview.reviewToken);
    expect(merged.activity).toMatchObject({ trainingSourceCount: count });
    expect(merged.trainingSources).toHaveLength(100);

    const hashes = new Set<string>();
    let cursor: string | undefined;
    do {
      const page = await repository.trainingSourcePage(target.id, cursor, 100);
      page.entries.forEach((entry) => hashes.add(entry.source.contentHash!));
      cursor = page.nextCursor;
    } while (cursor);
    expect(hashes.size).toBe(count);
    expect(await service.previewMerge(source.id, target.id)).toMatchObject({
      newEvidence: 0,
      duplicateEvidence: count
    });
  }, 300_000);

  it("keeps Synapses Only evidence metadata but does not copy source bytes", async () => {
    const target = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Synapses only",
      memoryRecipe: "synapses-only",
      retainSourceText: false
    });
    const source = await repository.fork(target.id, "Source");
    const branch = await repository.get(source.id);
    const bytes = Buffer.from("must not persist as source material");
    const blobHash = await repository.storeBlob(bytes);
    branch.trainingSources.push({
      id: randomUUID(),
      name: "private.txt",
      kind: "text",
      bytes: bytes.byteLength,
      learnedIdeas: 1,
      learnedConcepts: 2,
      learnedSynapses: 3,
      importedAt: new Date().toISOString(),
      rawTextRetained: true,
      rawText: bytes.toString("utf8"),
      contentHash: blobHash,
      blobHash,
      policy: "archive"
    });
    await repository.save(branch);

    const preview = await service.previewMerge(source.id, target.id);
    expect(preview).toMatchObject({ newEvidence: 1, newFiles: 0 });
    const merged = await service.merge(source.id, target.id, preview.reviewToken);
    expect(merged.trainingSources[0]).toMatchObject({
      name: "private.txt",
      rawTextRetained: false
    });
    expect(merged.trainingSources[0]?.rawText).toBeUndefined();
    expect(merged.trainingSources[0]?.blobHash).toBeUndefined();
    expect(merged.trainingSources[0]?.path).toBeUndefined();
  });
});
