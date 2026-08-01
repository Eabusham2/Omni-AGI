import { createHash, randomUUID } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrainService,
  normalizeModalityPreview
} from "../src/main/brainService";
import { BrainRepository } from "../src/main/brainRepository";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  DEFAULT_CONFIG,
  type BrainConfig,
  type BrainDocument
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
      revision: 3,
      progress: 1,
      statusLabel: "Decoding now",
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,cHJldmlldw==",
      path: undefined,
      artifactPath: "artifacts/preview.png"
    });
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
});

describe("BrainService starter checkpoints", () => {
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
        origin_kind: "starter",
        pretrained: true,
        hidden_behavioral_prompt: false,
        reward_model: false,
        rlhf: false
      }
    }));
    service = new BrainService(
      repository,
      { tryRequest, request } as unknown as EngineSupervisor
    );
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  async function importedBrain(
    config: BrainConfig = { ...DEFAULT_CONFIG, name: "Imported starter" }
  ): Promise<BrainDocument> {
    return repository.create(config);
  }

  it("builds the bundled trained starter when no catalog URL is supplied", async () => {
    const built = await service.create({
      origin: "starter",
      starterUrl: "",
      hardwareTier: "micro",
      modalities: ["vision", "image", "audio", "video"],
      config: { ...DEFAULT_CONFIG, name: "Bundled starter" }
    });

    expect(request).toHaveBeenCalledTimes(1);
    expect(built.config.workingMemorySlots).toBe(128);
    expect(request).toHaveBeenCalledWith(
      "create",
      expect.objectContaining({
        brainId: built.id,
        origin: "starter",
        hardwareTier: "micro",
        config: expect.objectContaining({ workingMemorySlots: 128 }),
        modalities: ["vision", "image", "audio", "video"],
        storagePath: repository.brainDirectory(built.id)
      }),
      300_000
    );
    expect(tryRequest).not.toHaveBeenCalled();
  });

  it("defaults the stable public API to Starter while preserving explicit Blank Brain", async () => {
    const starter = await service.create({
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Implicit starter" }
    });
    expect(request).toHaveBeenLastCalledWith(
      "create",
      expect.objectContaining({
        brainId: starter.id,
        origin: "starter"
      }),
      300_000
    );

    const blank = await service.create({
      origin: "blank",
      hardwareTier: "micro",
      config: { ...DEFAULT_CONFIG, name: "Explicit blank" }
    });
    expect(request).toHaveBeenLastCalledWith(
      "create",
      expect.objectContaining({
        brainId: blank.id,
        origin: "blank"
      }),
      60_000
    );
  });

  it("loads and shape-safely updates a materialized starter without randomizing it", async () => {
    const imported = await importedBrain({
      ...DEFAULT_CONFIG,
      name: "Pretrained checkpoint",
      workingMemorySlots: 320,
      extendedWorkingMemory: false
    });
    const engineDirectory = join(repository.brainDirectory(imported.id), "engine");
    const core = emptySafetensors("pretrained-core");
    const plasticity = emptySafetensors("pretrained-plasticity");
    await mkdir(engineDirectory, { recursive: true });
    await Promise.all([
      writeFile(
        join(engineDirectory, "brain.json"),
        JSON.stringify({
          schema_version: 1,
          format: "omni-cortex-engine",
          release_format: "stable-1.0",
          brain_id: imported.id,
          name: imported.name,
          config: {
            d_model: 64,
            n_layers: 2,
            max_seq_len: 1536,
            working_memory_slots: 320,
            initial_neuron_budget: 4_096
          }
        })
      ),
      writeFile(join(engineDirectory, "core.safetensors"), core),
      writeFile(join(engineDirectory, "plasticity.safetensors"), plasticity)
    ]);
    vi.spyOn(service, "importUrl").mockResolvedValue(imported);

    const built = await service.create({
      origin: "starter",
      starterUrl: "https://catalog.example/pretrained.omni",
      hardwareTier: "workstation",
      modalities: ["vision", "image"],
      config: {
        ...DEFAULT_CONFIG,
        name: "Adapted checkpoint",
        // Builder shape requests cannot resize an imported checkpoint.
        workingMemorySlots: 48,
        extendedWorkingMemory: true,
        // Simulate a pre-v1 caller. Stable normalization must discard these
        // beta behavior and architecture controls.
        initialNeuronBudget: 98_304,
        noise: 0.23
      } as BrainConfig & { initialNeuronBudget: number; noise: number }
    });

    expect(built.id).toBe(imported.id);
    expect(built.config).toMatchObject({
      name: "Adapted checkpoint",
      workingMemorySlots: 320,
      extendedWorkingMemory: false
    });
    expect(built.config).not.toHaveProperty("initialNeuronBudget");
    expect(built.config).not.toHaveProperty("noise");
    expect(tryRequest.mock.calls.map(([method]) => method)).toEqual([
      "unload",
      "load",
      "update_config"
    ]);
    expect(tryRequest.mock.calls.some(([method]) => method === "create")).toBe(false);

    const storagePath = repository.brainDirectory(imported.id);
    expect(tryRequest.mock.calls[1]).toEqual([
      "load",
      {
        brainId: imported.id,
        config: built.config,
        storagePath
      },
      300_000
    ]);
    expect(tryRequest.mock.calls[2]).toEqual([
      "update_config",
      {
        brainId: imported.id,
        config: built.config,
        storagePath
      },
      300_000
    ]);

    const [savedCore, savedPlasticity] = await Promise.all([
      readFile(join(engineDirectory, "core.safetensors")),
      readFile(join(engineDirectory, "plasticity.safetensors"))
    ]);
    expect(sha256(savedCore)).toBe(sha256(core));
    expect(sha256(savedPlasticity)).toBe(sha256(plasticity));
  });

  it("rejects and removes a starter import with no materialized engine state", async () => {
    const imported = await importedBrain();
    vi.spyOn(service, "importUrl").mockResolvedValue(imported);

    await expect(
      service.create({
        origin: "starter",
        starterUrl: "https://catalog.example/unmaterialized.omni",
        config: { ...DEFAULT_CONFIG, name: "Must not build" }
      })
    ).rejects.toThrow(/no materialized OmniCortex checkpoint/i);

    expect(tryRequest).not.toHaveBeenCalled();
    await expect(repository.get(imported.id)).rejects.toThrow();
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
