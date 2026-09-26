import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
  BrainRepository,
  assertNativeOmniEngineState
} from "../src/main/brainRepository";
import { DEFAULT_CONFIG } from "../src/shared/types";

describe("persisted brain origin provenance", () => {
  let temporaryRoot = "";

  afterEach(async () => {
    if (temporaryRoot) await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("accepts only materialized ground-up engine state at the bundle boundary", () => {
    const native = {
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      config: { origin_kind: "ground-up" },
      runtime_card: {
        origin_kind: "ground-up",
        pretrained: false,
        baseFrozen: false
      },
      packed_ternary_manifest: { baseFrozen: false, pretrainedTextCortex: null }
    };
    expect(() => assertNativeOmniEngineState(native, "Origin")).not.toThrow();
    for (const altered of [
      { ...native, config: { ...native.config, origin_kind: "starter" } },
      { ...native, config: { ...native.config, foundation_model_id: "external" } },
      { ...native, config: { ...native.config, foundationModelId: null } },
      { ...native, runtime_card: { ...native.runtime_card, pretrained: true } },
      { ...native, runtime_card: { ...native.runtime_card, baseFrozen: true } },
      { ...native, runtime_card: { ...native.runtime_card, pretrained_text_cortex: null } },
      { ...native, starter_training_manifest: { id: "starter" } },
      { ...native, starter_training_manifest: null },
      { ...native, neural_sequence_memory: { answerTable: {} } },
      { ...native, messages: [{ content: "private" }] },
      { ...native, traces: [{ content: "private" }] },
      { ...native, packed_ternary_manifest: { baseFrozen: true } },
      { ...native, packed_ternary_manifest: { baseFrozen: false, foundationModelId: null } }
    ]) {
      expect(() => assertNativeOmniEngineState(altered, "Origin"))
        .toThrow(/locally initialized OmniCortex origin/i);
    }
  });

  it("records new ground-up creation without an external foundation", async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-ground-up-provenance-"));
    const repository = new BrainRepository(join(temporaryRoot, "brains"));
    const brain = await repository.create(
      { ...DEFAULT_CONFIG, name: "Ground-up origin" },
      {
        initializing: true,
        recovery: {
          foundation: {
            hardwareTier: "micro",
            modalities: [],
            origin: "ground-up"
          },
          resources: []
        }
      }
    );

    expect(brain.provenance).toEqual({ originKind: "ground-up" });
    expect((await repository.list())[0]?.provenance).toEqual({
      originKind: "ground-up"
    });
  });

  it("rejects an existing frozen foundation instead of listing it as a mind", async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-hybrid-provenance-"));
    const repository = new BrainRepository(join(temporaryRoot, "brains"));
    let brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Imported hybrid"
    });
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engineDirectory, { recursive: true });
    await writeFile(join(engineDirectory, "brain.json"), JSON.stringify({
      brain_id: brain.id,
      config: { origin_kind: "starter", foundation_model_id: "falcon-e-3b-base-ad18b07" },
      runtime_card: {
        origin_kind: "starter",
        pretrained_text_cortex: {
          id: "falcon-e-3b-base-ad18b07",
          repository: "tiiuae/Falcon-E-3B-Base",
          adapter: { baseFrozen: true }
        }
      }
    }));
    brain.trainingSources.push({
      id: "source-tinystories",
      name: "TinyStories.parquet",
      kind: "parquet",
      bytes: 9_989_127,
      learnedIdeas: 100,
      learnedConcepts: 100,
      learnedSynapses: 100,
      learnedRecords: 21_990,
      importedAt: "2026-09-01T00:00:00.000Z",
      rawTextRetained: false
    });
    await repository.save(brain);

    await expect(repository.get(brain.id)).rejects.toThrow(/non-native origin/i);
    await expect(repository.list()).resolves.toEqual([]);
  });

  it("rejects frozen-foundation evidence despite stale ground-up provenance", async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-conflicting-provenance-"));
    const repository = new BrainRepository(join(temporaryRoot, "brains"));
    let brain = await repository.create(
      { ...DEFAULT_CONFIG, name: "Mislabeled imported hybrid" },
      {
        initializing: true,
        recovery: {
          foundation: {
            hardwareTier: "micro",
            modalities: [],
            origin: "ground-up"
          },
          resources: []
        }
      }
    );
    brain.provenance = {
      originKind: "ground-up",
      randomInitialization: { algorithm: "omni-random-v1", seed: 42 }
    };
    brain = await repository.save(brain);
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engineDirectory, { recursive: true });
    await writeFile(join(engineDirectory, "brain.json"), JSON.stringify({
      brain_id: brain.id,
      config: {
        origin_kind: "ground-up",
        foundation_model_id: "falcon-e-3b-base-ad18b07"
      },
      runtime_card: {
        origin_kind: "ground-up",
        foundationModelId: "falcon-e-3b-base-ad18b07",
        pretrained: true,
        baseFrozen: true,
        pretrained_text_cortex: {
          repository: "tiiuae/Falcon-E-3B-Base"
        }
      }
    }));

    await expect(repository.get(brain.id)).rejects.toThrow(/non-native origin/i);
  });

  it("rejects contradictory pretrained ground-up metadata without a model id", async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-pretrained-ground-up-"));
    const repository = new BrainRepository(join(temporaryRoot, "brains"));
    let brain = await repository.create(
      { ...DEFAULT_CONFIG, name: "Contradictory pretrained origin" },
      {
        initializing: true,
        recovery: {
          foundation: {
            hardwareTier: "micro",
            modalities: [],
            origin: "ground-up"
          },
          resources: []
        }
      }
    );
    brain.provenance = {
      originKind: "ground-up",
      randomInitialization: { algorithm: "omni-random-v1", seed: 7 }
    };
    brain = await repository.save(brain);
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engineDirectory, { recursive: true });
    await writeFile(join(engineDirectory, "brain.json"), JSON.stringify({
      brain_id: brain.id,
      config: { origin_kind: "ground-up" },
      runtime_card: {
        origin_kind: "ground-up",
        pretrained: true,
        baseFrozen: true
      }
    }));

    await expect(repository.get(brain.id)).rejects.toThrow(/non-native origin/i);
  });
});
