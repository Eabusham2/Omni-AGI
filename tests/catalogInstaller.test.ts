import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { strToU8, zipSync } from "fflate";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  listInstalledPacks,
  recordInstalledPack,
  stageModalityPack,
  validateBuildRecipe,
  validateModalityPack
} from "../src/main/catalogInstaller";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { DEFAULT_CONFIG, type ModalityKind } from "../src/shared/types";

function sha256(value: Buffer | Uint8Array): string {
  return createHash("sha256").update(value).digest("hex");
}

function modalityTensors(modality: ModalityKind, tensorName = "weight"): Buffer {
  const data = Buffer.alloc(4);
  data.writeFloatLE(0.25);
  const header = Buffer.from(
    JSON.stringify({
      [`modalities.${modality}.${tensorName}`]: {
        dtype: "F32",
        shape: [1],
        data_offsets: [0, 4]
      }
    })
  );
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  return Buffer.concat([prefix, header, data]);
}

function packFixture(
  modality: ModalityKind = "image",
  overrides: Record<string, unknown> = {},
  extraFiles: Record<string, Uint8Array> = {}
): Buffer {
  const tensor = modalityTensors(modality);
  const modelCard = Buffer.from("# Tiny fixture\n\nA test-only modality pack.\n");
  const manifest = {
    format: "omni-modality-pack",
    formatVersion: 1,
    architecture: "OmniCortex",
    architectureSchemaVersion: 1,
    pack: {
      id: `fixture-${modality}`,
      name: `Fixture ${modality}`,
      version: "1.0.0",
      modalities: [modality]
    },
    compatibility: {
      dModel: 64,
      modalityChannels: 16,
      imageSize: 16,
      audioSamples: 256,
      videoFrames: 4
    },
    licenseLedger: {
      license: "MIT",
      provenanceUrl: "https://example.com/model-card"
    },
    files: {
      "model-card.md": { sha256: sha256(modelCard), bytes: modelCard.byteLength },
      "tensors/modality.safetensors": {
        sha256: sha256(tensor),
        bytes: tensor.byteLength
      }
    },
    ...overrides
  };
  const manifestBytes = Buffer.from(JSON.stringify(manifest));
  const checksums = Buffer.from(
    [
      `${sha256(manifestBytes)}  manifest.json`,
      `${sha256(modelCard)}  model-card.md`,
      `${sha256(tensor)}  tensors/modality.safetensors`
    ].join("\n") + "\n"
  );
  return Buffer.from(
    zipSync(
      {
        "manifest.json": new Uint8Array(manifestBytes),
        "model-card.md": new Uint8Array(modelCard),
        "checksums.sha256": new Uint8Array(checksums),
        "tensors/modality.safetensors": new Uint8Array(tensor),
        ...extraFiles
      },
      { level: 0 }
    )
  );
}

describe("declarative catalog installers", () => {
  const temporary: string[] = [];

  afterEach(async () => {
    await Promise.all(
      temporary.splice(0).map((path) => rm(path, { recursive: true, force: true }))
    );
  });

  it("maps the bundled recipes into complete build requests without executing code", async () => {
    const micro = validateBuildRecipe(
      await readFile(join(process.cwd(), "catalog", "recipes", "whole-brain-micro.json")),
      "catalog:whole-brain-micro"
    );
    expect(micro).toMatchObject({
      id: "whole-brain-micro",
      origin: "blank",
      hardwareTier: "micro",
      modalities: ["vision", "image", "audio", "video"],
      config: {
        preset: "whole-brain",
        ternaryWeights: true,
        spikingDynamics: true,
        stdpPlasticity: true,
        liquidDynamics: true,
        vectorSymbolicMemory: true,
        memoryRecipe: "human-consolidation"
      }
    });
    expect(micro.toolPermissions.every((entry) => entry.level === "ask")).toBe(true);

    const plastic = validateBuildRecipe(
      await readFile(join(process.cwd(), "catalog", "recipes", "plastic-synapse-lab.json"))
    );
    expect(plastic.config).toMatchObject({
      preset: "neuromorphic",
      initialNeuronBudget: 512,
      growthPolicy: "elastic",
      memoryRecipe: "synapses-only",
      retainSourceText: false
    });
    expect(plastic.modalities).toEqual([]);
  });

  it("rejects executable hooks, unknown fields, insecure starters, and oversized JSON", () => {
    const base = {
      schemaVersion: 1,
      id: "unsafe",
      name: "Unsafe",
      description: "Must not load",
      origin: "blank",
      hardwareProfile: "micro",
      architecture: { preset: "whole-brain" },
      memoryRecipe: "human-consolidation",
      toolPermission: "ask"
    };
    expect(() =>
      validateBuildRecipe(Buffer.from(JSON.stringify({ ...base, setup: "powershell evil.ps1" })))
    ).toThrow(/unsupported field setup/i);
    expect(() =>
      validateBuildRecipe(
        Buffer.from(
          JSON.stringify({
            ...base,
            origin: "starter",
            starterUrl: "http://example.com/brain.omni"
          })
        )
      )
    ).toThrow(/must use HTTPS/i);
    expect(() => validateBuildRecipe(Buffer.alloc(1024 * 1024 + 1, 0x20))).toThrow(
      /no larger than 1 MB/i
    );
  });

  it("validates, stages, and records a namespaced safe-tensor modality pack", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-pack-test-"));
    temporary.push(root);
    const fixture = packFixture("image");
    const validated = validateModalityPack(fixture, "fixture.omnipack");
    expect(validated.manifest.pack).toEqual({
      id: "fixture-image",
      name: "Fixture image",
      version: "1.0.0",
      modalities: ["image"]
    });
    const staged = await stageModalityPack(root, fixture, "fixture.omnipack");
    expect(await readFile(staged.packPath)).toEqual(modalityTensors("image"));
    await recordInstalledPack(root, staged.result);
    expect(await listInstalledPacks(root)).toEqual([staged.result]);
  });

  it("rejects extra ZIP content, checksum tampering, and tensors outside pack namespaces", () => {
    expect(() =>
      validateModalityPack(packFixture("image", {}, { "setup.ps1": strToU8("exit 0") }))
    ).toThrow(/unsupported entry|directory is invalid/i);

    const fixture = packFixture("image");
    const markerIndex = fixture.indexOf(Buffer.from("Tiny fixture"));
    expect(markerIndex).toBeGreaterThanOrEqual(0);
    fixture[markerIndex] = (fixture[markerIndex] ?? 0) ^ 1;
    expect(() => validateModalityPack(fixture)).toThrow();

    const badTensor = modalityTensors("audio");
    const modelCard = Buffer.from("# Namespaces\n");
    const manifest = {
      format: "omni-modality-pack",
      formatVersion: 1,
      architecture: "OmniCortex",
      architectureSchemaVersion: 1,
      pack: { id: "wrong", name: "Wrong", version: "1", modalities: ["image"] },
      compatibility: {
        dModel: 64,
        modalityChannels: 16,
        imageSize: 16,
        audioSamples: 256,
        videoFrames: 4
      },
      licenseLedger: { license: "MIT" },
      files: {
        "model-card.md": { sha256: sha256(modelCard), bytes: modelCard.byteLength },
        "tensors/modality.safetensors": {
          sha256: sha256(badTensor),
          bytes: badTensor.byteLength
        }
      }
    };
    const manifestBytes = Buffer.from(JSON.stringify(manifest));
    const checksums = Buffer.from(
      `${sha256(manifestBytes)}  manifest.json\n${sha256(modelCard)}  model-card.md\n${sha256(
        badTensor
      )}  tensors/modality.safetensors\n`
    );
    const wrongNamespace = Buffer.from(
      zipSync({
        "manifest.json": new Uint8Array(manifestBytes),
        "model-card.md": new Uint8Array(modelCard),
        "checksums.sha256": new Uint8Array(checksums),
        "tensors/modality.safetensors": new Uint8Array(badTensor)
      })
    );
    expect(() => validateModalityPack(wrongNamespace)).toThrow(/outside the declared/i);
  });

  it("applies a validated pack through the supervised worker and records it only on success", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-pack-service-"));
    temporary.push(root);
    const repository = new BrainRepository(join(root, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Pack target" });
    const request = vi.fn(async () => ({ installed: true }));
    const service = new BrainService(
      repository,
      { request } as unknown as EngineSupervisor
    );
    const fixture = packFixture("audio");
    const result = await service.installModalityPackBuffer(
      brain.id,
      fixture,
      "audio.omnipack"
    );
    expect(result.modalities).toEqual(["audio"]);
    expect(request).toHaveBeenCalledWith(
      "install_modality_pack",
      expect.objectContaining({
        brainId: brain.id,
        storagePath: repository.brainDirectory(brain.id),
        manifest: expect.objectContaining({ format: "omni-modality-pack" })
      }),
      300_000
    );
    expect(await service.listModalityPacks(brain.id)).toEqual([result]);

    request.mockRejectedValueOnce(new Error("shape mismatch"));
    await expect(
      service.installModalityPackBuffer(brain.id, packFixture("video"), "video.omnipack")
    ).rejects.toThrow(/shape mismatch/i);
    expect((await service.listModalityPacks(brain.id)).map((pack) => pack.id)).toEqual([
      "fixture-audio"
    ]);
  });
});
