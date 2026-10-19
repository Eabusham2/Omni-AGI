import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  declaredDocumentationCurriculumPack,
  type DocumentationCurriculumPack
} from "../src/main/documentationCurriculumPack";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import { BuildInitializationPlanStore } from "../src/main/buildInitializationPlan";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  DEFAULT_CONFIG,
  type IngestResult,
  type TrainingSource
} from "../src/shared/types";

const sha256 = (value: string): string =>
  createHash("sha256").update(value, "utf8").digest("hex");

describe("versioned ordinary documentation curriculum", () => {
  const roots: string[] = [];
  afterEach(async () => {
    await Promise.all(
      roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))
    );
  });

  it("binds the public manuals without action targets, persona text, or fake outcomes", () => {
    const pack = declaredDocumentationCurriculumPack();
    expect(pack).toMatchObject({
      format: "omni-ordinary-document-curriculum",
      version: 23,
      id: "omni-v2-user-documentation-23",
      ordinaryNeuralLearning: true,
      hiddenPrompt: false,
      behavioralPersona: false,
      preferenceLabels: false,
      rewardModel: false,
      executedHostActions: false
    });
    expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(pack.records.map((record) => record.id)).toEqual([
      "user-guide",
      "integrations",
      "plugins",
      "html-artifacts",
      "executable-training-parsers",
      "managed-environments",
      "clipboard",
      "api-teacher-model-discovery"
    ]);
    for (const record of pack.records) {
      expect(record.contentSha256).toBe(sha256(record.content));
      expect(record.bytes).toBe(Buffer.byteLength(record.content));
      expect(record.content).toContain("Omni AGI Studio");
      expect(record).not.toHaveProperty("toolId");
      expect(record).not.toHaveProperty("action");
      expect(record).not.toHaveProperty("successfulOutcome");
    }
    expect(declaredDocumentationCurriculumPack()).toEqual(pack);
  });

  it("commits exact per-content receipts and safely deduplicates a lost acknowledgement", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-doc-curriculum-"));
    roots.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create(
      { ...DEFAULT_CONFIG, name: "Documentation fixture" },
      { initializing: true }
    );
    const plans = new BuildInitializationPlanStore(repository);
    await plans.create(brain.id, {
      hardwareTier: "micro",
      modalities: [],
      origin: "ground-up"
    }, []);
    await plans.foundationReady(brain.id);

    const service = new BrainService(
      repository,
      new EventEmitter() as EngineSupervisor
    );
    vi.spyOn(service, "resumeFoundation").mockImplementation(async () =>
      repository.get(brain.id)
    );
    const committed = new Map<string, TrainingSource>();
    vi.spyOn(repository, "trainingSourceByContentHash").mockImplementation(
      async (_brainId, contentHash) => committed.get(contentHash)
    );
    const ordinaryLearn = vi.fn(async (
      brainId: string,
      experience: {
        content: string;
        name: string;
        provenanceUrl?: string;
        license?: string;
      }
    ): Promise<IngestResult> => {
      const contentHash = sha256(experience.content);
      const existing = committed.get(contentHash);
      if (existing) {
        return {
          brain: await repository.get(brainId),
          source: existing,
          warnings: ["This neural experience was already learned."]
        };
      }
      const source: TrainingSource = {
        id: `documentation-${committed.size + 1}`,
        name: experience.name,
        kind: "text",
        bytes: Buffer.byteLength(experience.content),
        learnedIdeas: 1,
        learnedConcepts: 1,
        learnedSynapses: 1,
        learnedParameterSteps: 1,
        parametersChanged: true,
        importedAt: new Date().toISOString(),
        rawTextRetained: false,
        contentHash,
        policy: "pretrain",
        provenanceUrl: experience.provenanceUrl,
        license: experience.license
      };
      committed.set(contentHash, source);
      const current = await repository.get(brainId);
      current.trainingSources.push(source);
      return { brain: await repository.save(current), source, warnings: [] };
    });
    Object.defineProperty(service, "learnStructuredExperienceUnlocked", {
      configurable: true,
      value: ordinaryLearn
    });

    const first = await service.teachInitialDocumentation(brain.id);
    const retried = await service.teachInitialDocumentation(brain.id);
    const currentPack = declaredDocumentationCurriculumPack();
    expect(first).toMatchObject({ complete: true, packVersion: currentPack.version });
    expect(first.records).toHaveLength(currentPack.records.length);
    expect(first.records.every((record) => !record.duplicate)).toBe(true);
    expect(retried.records.map((record) => record.trainingSourceId)).toEqual(
      first.records.map((record) => record.trainingSourceId)
    );
    expect(retried.records.every((record) => record.duplicate)).toBe(true);
    expect(committed.size).toBe(currentPack.records.length);
    const saved = await repository.get(brain.id);
    expect(
      saved.journal?.filter((entry) =>
        entry.detail?.includes(`sha256=${first.packSha256}`)
      )
    ).toHaveLength(1);

    const tampered = structuredClone(
      declaredDocumentationCurriculumPack()
    ) as DocumentationCurriculumPack;
    tampered.records[0]!.content += "tampered";
    await expect(
      service.teachInitialDocumentation(brain.id, tampered)
    ).rejects.toThrow(/declaration is invalid/i);
    expect(committed.size).toBe(currentPack.records.length);
  });
});
