import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { strFromU8, unzipSync, zipSync } from "fflate";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrainRepository,
  SUBSTRATE_STORE_FORMAT
} from "../src/main/brainRepository";
import { DEFAULT_CONFIG } from "../src/shared/types";

const OFFICIAL_STARTER_MANIFEST_SHA256 =
  "40091bacb930e5632d564e620e15cd68643073f7b87b73d05338bca30af916d4";

function emptySafetensors(): Buffer {
  const header = Buffer.from(JSON.stringify({ __metadata__: { test: "true" } }).padEnd(128, " "));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  return Buffer.concat([prefix, header]);
}

function byteTensorSafetensors(bytes: number): Buffer {
  const descriptor = JSON.stringify({
    weights: { dtype: "U8", shape: [bytes], data_offsets: [0, bytes] }
  });
  const header = Buffer.from(descriptor.padEnd(Math.ceil(descriptor.length / 8) * 8, " "));
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(BigInt(header.byteLength));
  const data = Buffer.allocUnsafe(bytes);
  let state = 0x6d2b79f5;
  for (let index = 0; index < data.length; index += 1) {
    state = Math.imul(state ^ (state >>> 15), 1 | state);
    state ^= state + Math.imul(state ^ (state >>> 7), 61 | state);
    data[index] = (state ^ (state >>> 14)) & 0xff;
  }
  return Buffer.concat([prefix, header, data]);
}

function digest(value: Uint8Array | Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
}

function originChecksumForDocument(value: Record<string, unknown>): string {
  return digest(JSON.stringify({ ...value, originChecksum: undefined }));
}

function refreshArchiveIntegrity(entries: Record<string, Uint8Array>): void {
  const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as {
    files: Record<string, { sha256: string; bytes: number }>;
  };
  for (const [path, contents] of Object.entries(entries)) {
    if (path === "manifest.json" || path === "checksums.sha256") continue;
    manifest.files[path] = {
      sha256: digest(contents),
      bytes: contents.byteLength
    };
  }
  entries["manifest.json"] = Buffer.from(JSON.stringify(manifest, null, 2));
  entries["checksums.sha256"] = Buffer.from(
    Object.entries(entries)
      .filter(([path]) => path !== "checksums.sha256")
      .map(([path, contents]) => `${digest(contents)}  ${path}`)
      .sort()
      .join("\n") + "\n"
  );
}

async function writePackedTernaryFixture(directory: string): Promise<void> {
  const shard = Buffer.from([0b01010101]);
  const shardHash = digest(shard);
  const tensorHash = digest(
    Buffer.concat([
      Buffer.from(canonicalJson({ dtype: "int8", shape: [1] })),
      Buffer.from([0]),
      Buffer.from([0])
    ])
  );
  const manifestBody = {
    format: "omni-packed-ternary",
    formatVersion: 1,
    architecture: "OmniCortex",
    encoding: {
      bitsPerValue: 2,
      byteOrder: "four-values-lsb-first",
      codes: { "-1": 0, "0": 1, "+1": 2 },
      reservedCode: 3,
      paddingValue: 0
    },
    coverage: {
      eligibleTensorCount: 1,
      eligibleTensorNames: ["fixture.projection.weight"],
      complete: true
    },
    tensors: [
      {
        name: "fixture.projection.weight",
        kind: "projection",
        shape: [1],
        dtype: "int8",
        sourceDtype: "float32",
        scale: 1,
        numel: 1,
        shard: `ternary-00000-${shardHash.slice(0, 16)}.bin`,
        byteOffset: 0,
        byteLength: shard.byteLength,
        packedSha256: shardHash,
        tensorSha256: tensorHash
      }
    ],
    shards: [
      {
        file: `ternary-00000-${shardHash.slice(0, 16)}.bin`,
        byteLength: shard.byteLength,
        sha256: shardHash
      }
    ],
    metadata: { fixture: true }
  };
  // Python's canonical encoder preserves this field's float identity as 1.0;
  // the TypeScript verifier must not collapse it and reject a genuine pack.
  const canonicalBody = canonicalJson(manifestBody).replace(
    '"scale":1,',
    '"scale":1.0,'
  );
  const manifest = {
    ...manifestBody,
    contentSha256: digest(canonicalBody)
  };
  const manifestBytes = Buffer.from(
    canonicalJson(manifest).replace('"scale":1,', '"scale":1.0,')
  );
  await mkdir(directory, { recursive: true });
  await Promise.all([
    writeFile(join(directory, "manifest.json"), manifestBytes),
    writeFile(join(directory, "manifest.sha256"), `${digest(manifestBytes)}\n`),
    writeFile(
      join(directory, `ternary-00000-${shardHash.slice(0, 16)}.bin`),
      shard
    )
  ]);
}

async function writeSubstrateFixture(engineDirectory: string): Promise<Record<string, unknown>> {
  const store = join(engineDirectory, "substrate");
  const record = {
    kind: "neurons",
    ids: ["neuron-fixture"],
    records: [{ id: "neuron-fixture", region: "cortical" }],
    vectorIds: []
  };
  const recordBytes = Buffer.from(canonicalJson(record));
  const recordHash = digest(recordBytes);
  const generationBody = {
    format: "omni-substrate-shards",
    formatVersion: 1,
    schema: 1,
    dimensions: 16,
    seed: 7,
    growthEvents: 1,
    growthPauses: 0,
    recordsPerShard: 512,
    counts: { neurons: 1, assemblies: 0, synapses: 0 },
    shards: [
      {
        kind: "neurons",
        bucket: "a",
        part: 0,
        count: 1,
        records: {
          path: `blobs/${recordHash}.json`,
          sha256: recordHash,
          bytes: recordBytes.byteLength
        },
        tensors: null
      }
    ]
  };
  const contentHash = digest(canonicalJson(generationBody));
  const generation = { ...generationBody, contentSha256: contentHash };
  const generationBytes = Buffer.from(canonicalJson(generation));
  const generationHash = digest(generationBytes);
  const generationRelative = `generations/${contentHash}/manifest.json`;
  const pointer = {
    format: "omni-substrate-shards",
    formatVersion: 1,
    activeGeneration: contentHash,
    generationManifest: generationRelative,
    generationManifestSha256: generationHash,
    counts: generationBody.counts,
    shardCount: 1,
    contentSha256: contentHash
  };
  await Promise.all([
    mkdir(join(store, "blobs"), { recursive: true }),
    mkdir(join(store, "generations", contentHash), { recursive: true })
  ]);
  await Promise.all([
    writeFile(join(store, "blobs", `${recordHash}.json`), recordBytes),
    writeFile(join(store, ...generationRelative.split("/")), generationBytes),
    writeFile(join(store, "manifest.json"), canonicalJson(pointer))
  ]);
  return pointer;
}

async function writeOriginProvenanceFixture(
  engineDirectory: string
): Promise<Buffer> {
  const origin = join(engineDirectory, "origin");
  const [metadataBytes, core, plasticity, packedManifest] = await Promise.all([
    readFile(join(origin, "brain.json")),
    readFile(join(origin, "core.safetensors")),
    readFile(join(origin, "plasticity.safetensors")),
    readFile(join(origin, "packed-ternary", "manifest.json"))
  ]);
  const metadata = JSON.parse(metadataBytes.toString("utf8")) as {
    brain_id: string;
    starter_training_manifest: {
      id: string;
      sha256: string;
      trainedParameterChecksum: string;
    };
    substrate: { persistence: { contentSha256: string } };
  };
  const payload = {
    format: "omni-bundled-origin-provenance-1",
    originBrainId: metadata.brain_id,
    starterId: metadata.starter_training_manifest.id,
    starterManifestSha256: metadata.starter_training_manifest.sha256,
    originParameterChecksum:
      metadata.starter_training_manifest.trainedParameterChecksum,
    coreSha256: digest(core),
    plasticitySha256: digest(plasticity),
    brainMetadataSha256: digest(metadataBytes),
    substrateContentSha256:
      metadata.substrate.persistence.contentSha256,
    packedManifestSha256: digest(packedManifest)
  };
  const bytes = Buffer.from(
    canonicalJson({
      ...payload,
      contentSha256: digest(canonicalJson(payload))
    })
  );
  await writeFile(join(origin, "provenance.json"), bytes);
  return bytes;
}

describe("BrainRepository lifecycle", () => {
  let temporaryRoot: string;
  let repository: BrainRepository;

  beforeEach(async () => {
    temporaryRoot = await mkdtemp(join(tmpdir(), "omni-repository-test-"));
    repository = new BrainRepository(join(temporaryRoot, "brains"));
    await repository.initialize();
  });

  afterEach(async () => {
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("shares the authoritative substrate-store format with the Python worker", async () => {
    const pythonSource = await readFile(
      join(process.cwd(), "engine", "omni_core", "vsa.py"),
      "utf8"
    );
    expect(pythonSource).toMatch(
      new RegExp(
        `_SUBSTRATE_STORE_FORMAT\\s*=\\s*["']${SUBSTRATE_STORE_FORMAT}["']`
      )
    );
    expect(pythonSource).toMatch(/hexdigest\(\)\[:1\]/);
  });

  it("persists only stable v1 choices and discards beta behavior controls", async () => {
    const legacyInput = {
      ...DEFAULT_CONFIG,
      name: "Unconfigured mind",
      curiosityDrive: 1,
      noveltyDrive: 0,
      noise: 0.99,
      parallelThoughts: 64,
      maxConcepts: 16,
      maxSynapses: 16,
      growthPolicy: "fixed",
      ternaryWeights: false
    } as typeof DEFAULT_CONFIG & Record<string, unknown>;
    const brain = await repository.create(legacyInput);
    const stored = JSON.parse(
      await readFile(join(repository.brainDirectory(brain.id), "brain.json"), "utf8")
    ) as { config: Record<string, unknown> };

    expect(brain.config).toEqual({
      ...DEFAULT_CONFIG,
      name: "Unconfigured mind"
    });
    for (const key of [
      "curiosityDrive",
      "noveltyDrive",
      "noise",
      "parallelThoughts",
      "maxConcepts",
      "maxSynapses",
      "growthPolicy",
      "ternaryWeights"
    ]) {
      expect(stored.config).not.toHaveProperty(key);
    }
  });

  it("idempotently promotes concurrent immutable blobs and rejects a corrupt winner", async () => {
    const source = join(temporaryRoot, "shared-tensor.safetensors");
    const contents = byteTensorSafetensors(64 * 1024);
    const expected = digest(contents);
    await writeFile(source, contents);

    const promoted = await Promise.all(
      Array.from({ length: 24 }, () => repository.storeFileAsBlob(source))
    );

    expect(new Set(promoted)).toEqual(new Set([expected]));
    await expect(repository.getBlob(expected)).resolves.toEqual(contents);
    expect((await readdir(join(repository.root, ".blobs"))).sort()).toEqual([
      expected
    ]);

    const corrupt = Buffer.from("corrupt blob winner");
    await writeFile(join(repository.root, ".blobs", expected), corrupt);
    await expect(repository.storeFileAsBlob(source)).rejects.toThrow(
      /blob checksum failed/i
    );
    await expect(readFile(join(repository.root, ".blobs", expected))).resolves.toEqual(
      corrupt
    );
    expect((await readdir(join(repository.root, ".blobs"))).sort()).toEqual([
      expected
    ]);
  });

  it("atomically promotes concurrent in-memory blobs and validates every winner", async () => {
    // The race is in promotion, not payload size. Keep enough simultaneous
    // writers to exercise the collision while leaving headroom for Vitest's
    // other filesystem-heavy suites on slower CI runners.
    const contents = byteTensorSafetensors(64 * 1024);
    const expected = digest(contents);
    const promoted = await Promise.all(
      Array.from({ length: 12 }, () => repository.storeBlob(contents))
    );

    expect(new Set(promoted)).toEqual(new Set([expected]));
    await expect(repository.getBlob(expected)).resolves.toEqual(contents);
    expect((await readdir(join(repository.root, ".blobs"))).sort()).toEqual([
      expected
    ]);

    const corrupt = Buffer.from("corrupt in-memory blob winner");
    await writeFile(join(repository.root, ".blobs", expected), corrupt);
    await expect(repository.storeBlob(contents)).rejects.toThrow(
      /blob checksum failed/i
    );
    await expect(readFile(join(repository.root, ".blobs", expected))).resolves.toEqual(
      corrupt
    );
  }, 15_000);

  it("preserves an imported checkpoint workspace above the former product cap", async () => {
    const recordedSlots = 8_192;
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large imported workspace",
      workingMemorySlots: recordedSlots
    });
    const bundle = join(temporaryRoot, "large-workspace.omni");

    await repository.exportBundle(source.id, bundle, "current");
    const imported = await repository.importBundle(bundle);

    expect(source.config.workingMemorySlots).toBe(recordedSlots);
    expect(imported.config.workingMemorySlots).toBe(recordedSlots);
    expect((await repository.get(imported.id)).config.workingMemorySlots).toBe(
      recordedSlots
    );
  });

  it("atomically reserves unique identities for concurrent imports of one bundle", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Concurrent import source"
    });
    const bundle = join(temporaryRoot, "concurrent-import.omni");
    await repository.exportBundle(source.id, bundle, "current");
    const bundleBytes = await readFile(bundle);

    const destination = new BrainRepository(
      join(temporaryRoot, "concurrent-import-destination")
    );
    await destination.initialize();
    const imported = await Promise.all(
      Array.from({ length: 6 }, (_, index) =>
        destination.importBundleBuffer(
          bundleBytes,
          `concurrent-${index}.omni`
        )
      )
    );

    expect(new Set(imported.map((brain) => brain.id)).size).toBe(6);
    expect(imported.filter((brain) => brain.id === source.id)).toHaveLength(1);
    await Promise.all(
      imported.map(async (brain) => {
        await expect(destination.get(brain.id)).resolves.toMatchObject({
          id: brain.id,
          originChecksum: brain.originChecksum
        });
        const origin = JSON.parse(
          await readFile(
            join(destination.brainDirectory(brain.id), "origin.json"),
            "utf8"
          )
        ) as Record<string, unknown>;
        expect(brain.originChecksum).toBe(originChecksumForDocument(origin));
      })
    );
  });

  it("creates an immutable origin and copy-on-write neural fork", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Ada" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        name: brain.name,
        config: {},
        expert_count: 0,
        substrate: { persistence: substratePointer }
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);

    const fork = await repository.fork(brain.id, "Ada branch");
    const forkEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(fork.id), "engine", "brain.json"), "utf8")
    ) as { brain_id: string; name: string };

    expect(fork.lineage.parentId).toBe(brain.id);
    await expect(
      readFile(
        join(
          repository.brainDirectory(fork.id),
          "engine",
          "substrate",
          "manifest.json"
        ),
        "utf8"
      )
    ).resolves.toBe(canonicalJson(substratePointer));
    expect(fork.lineage.rootId).toBe(brain.id);
    expect(forkEngine.brain_id).toBe(fork.id);
    expect(forkEngine.name).toBe("Ada branch");
    await expect(
      readFile(join(repository.brainDirectory(fork.id), "engine", "origin", "core.safetensors"))
    ).resolves.toBeInstanceOf(Buffer);

    const visibleBeforeFailure = (await repository.list()).map((item) => item.id).sort();
    const cloneFailure = vi
      .spyOn(repository, "storeFileAsBlob")
      .mockRejectedValueOnce(new Error("simulated clone staging failure"));
    try {
      await expect(repository.fork(brain.id, "Broken fork")).rejects.toThrow(
        /simulated clone staging failure/
      );
    } finally {
      cloneFailure.mockRestore();
    }
    expect((await repository.list()).map((item) => item.id).sort()).toEqual(
      visibleBeforeFailure
    );
  });

  it("duplicates every brain as an independent copy-on-write identity", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Original" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        name: brain.name,
        config: {}
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
    await writePackedTernaryFixture(join(engine, "packed-ternary"));

    const duplicate = await repository.duplicate(brain.id);
    const metadata = JSON.parse(
      await readFile(
        join(repository.brainDirectory(duplicate.id), "engine", "brain.json"),
        "utf8"
      )
    ) as { brain_id: string };

    expect(duplicate).toMatchObject({
      name: "Original copy",
      lineage: {
        parentId: brain.id,
        rootId: brain.id,
        generation: 1
      }
    });
    expect(metadata.brain_id).toBe(duplicate.id);
    expect(duplicate.journal?.at(-1)?.summary).toMatch(
      /Duplicated.*copy-on-write/i
    );
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      id: brain.id,
      name: "Original"
    });
    await Promise.all([
      expect(
        readFile(
          join(
            repository.brainDirectory(duplicate.id),
            "engine",
            "packed-ternary",
            "manifest.json"
          )
        )
      ).resolves.toBeInstanceOf(Buffer),
      expect(
        readFile(
          join(
            repository.brainDirectory(duplicate.id),
            "engine",
            "origin",
            "packed-ternary",
            "manifest.json"
          )
        )
      ).resolves.toBeInstanceOf(Buffer)
    ]);
  });

  it("preserves immutable starter provenance through duplicate and .omni round trips", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Verified starter"
    });
    const learnedSourceBytes = Buffer.from("knowledge learned after the immutable origin");
    const learnedSourceHash = digest(learnedSourceBytes);
    brain.trainingSources.push({
      id: randomUUID(),
      name: "learned-after-origin.txt",
      kind: "text",
      bytes: learnedSourceBytes.byteLength,
      learnedIdeas: 8,
      learnedConcepts: 5,
      learnedSynapses: 21,
      importedAt: new Date().toISOString(),
      rawTextRetained: false,
      contentHash: learnedSourceHash,
      policy: "consolidate"
    });
    brain.messages.push({
      id: randomUUID(),
      role: "human",
      content: "This turn exists only in mutable learned state.",
      createdAt: new Date().toISOString()
    });
    brain.counters.plasticityEvents += 21;
    await repository.save(brain);
    const engine = join(repository.brainDirectory(brain.id), "engine");
    const origin = join(engine, "origin");
    await Promise.all([
      mkdir(engine, { recursive: true }),
      mkdir(origin, { recursive: true })
    ]);
    const [currentSubstrate, originSubstrate] = await Promise.all([
      writeSubstrateFixture(engine),
      writeSubstrateFixture(origin)
    ]);
    const starterManifest = {
      id: "omni-starter-bundled-1",
      sha256: OFFICIAL_STARTER_MANIFEST_SHA256,
      trainedParameterChecksum: "b".repeat(64)
    };
    const currentState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: { name: brain.name, origin_kind: "starter" },
      expert_count: 0,
      starter_training_manifest: starterManifest,
      substrate: {
        schema: 1,
        dimensions: 16,
        seed: 7,
        persistence: currentSubstrate
      },
      marker: "mutable-current"
    };
    const originState = {
      ...currentState,
      substrate: {
        ...currentState.substrate,
        persistence: originSubstrate
      },
      marker: "immutable-origin"
    };
    const currentCore = byteTensorSafetensors(32);
    const currentPlasticity = byteTensorSafetensors(24);
    const originCore = byteTensorSafetensors(16);
    const originPlasticity = byteTensorSafetensors(8);
    await Promise.all([
      writeFile(join(engine, "brain.json"), canonicalJson(currentState)),
      writeFile(join(engine, "core.safetensors"), currentCore),
      writeFile(join(engine, "plasticity.safetensors"), currentPlasticity),
      writeFile(join(origin, "brain.json"), canonicalJson(originState)),
      writeFile(join(origin, "core.safetensors"), originCore),
      writeFile(join(origin, "plasticity.safetensors"), originPlasticity),
      writePackedTernaryFixture(join(engine, "packed-ternary")),
      writePackedTernaryFixture(join(origin, "packed-ternary"))
    ]);
    const provenance = await writeOriginProvenanceFixture(engine);
    const originMetadata = await readFile(join(origin, "brain.json"));
    const uiOrigin = await readFile(
      join(repository.brainDirectory(brain.id), "origin.json")
    );

    const duplicate = await repository.duplicate(brain.id);
    const fork = await repository.fork(brain.id, "Verified starter branch");
    const duplicateEngine = join(
      repository.brainDirectory(duplicate.id),
      "engine"
    );
    const duplicateCurrent = JSON.parse(
      await readFile(join(duplicateEngine, "brain.json"), "utf8")
    ) as { brain_id: string; marker: string };
    expect(duplicateCurrent).toMatchObject({
      brain_id: duplicate.id,
      marker: "mutable-current"
    });
    expect(duplicate.originChecksum).toBe(brain.originChecksum);
    expect(fork.originChecksum).toBe(brain.originChecksum);
    expect(duplicate.trainingSources).toHaveLength(1);
    expect(fork.trainingSources).toHaveLength(1);
    await expect(
      readFile(join(repository.brainDirectory(duplicate.id), "origin.json"))
    ).resolves.toEqual(uiOrigin);
    await expect(
      readFile(join(repository.brainDirectory(fork.id), "origin.json"))
    ).resolves.toEqual(uiOrigin);
    await expect(
      readFile(join(duplicateEngine, "origin", "brain.json"))
    ).resolves.toEqual(originMetadata);
    await expect(
      readFile(join(duplicateEngine, "origin", "provenance.json"))
    ).resolves.toEqual(provenance);
    await expect(
      readFile(join(duplicateEngine, "origin", "core.safetensors"))
    ).resolves.toEqual(originCore);
    await expect(
      readFile(join(duplicateEngine, "core.safetensors"))
    ).resolves.toEqual(currentCore);

    const bundle = join(temporaryRoot, "verified-starter.omni");
    await repository.exportBundle(brain.id, bundle, "current");
    const archive = unzipSync(new Uint8Array(await readFile(bundle)));
    expect(Buffer.from(archive["origin/provenance.json"]!)).toEqual(
      provenance
    );
    expect(Buffer.from(archive["origin/state/engine.json"]!)).toEqual(
      originMetadata
    );

    const imported = await repository.importBundle(bundle);
    const importedEngine = join(
      repository.brainDirectory(imported.id),
      "engine"
    );
    const importedCurrent = JSON.parse(
      await readFile(join(importedEngine, "brain.json"), "utf8")
    ) as { brain_id: string };
    expect(importedCurrent.brain_id).toBe(imported.id);
    await expect(
      readFile(join(importedEngine, "origin", "brain.json"))
    ).resolves.toEqual(originMetadata);
    await expect(
      readFile(join(importedEngine, "origin", "provenance.json"))
    ).resolves.toEqual(provenance);
    await expect(
      readFile(join(importedEngine, "origin", "core.safetensors"))
    ).resolves.toEqual(originCore);

    const originBundle = join(temporaryRoot, "verified-starter-origin.omni");
    await repository.exportBundle(duplicate.id, originBundle, "origin");
    const originArchive = unzipSync(new Uint8Array(await readFile(originBundle)));
    const exportedOriginBrain = JSON.parse(
      strFromU8(originArchive["state/brain.json"]!)
    ) as { id: string; name: string; trainingSources: unknown[]; messages: unknown[] };
    const exportedImmutableBrain = JSON.parse(
      strFromU8(originArchive["origin/state/brain.json"]!)
    ) as typeof exportedOriginBrain & { originChecksum: string };
    const exportedCurrentBrain = JSON.parse(
      strFromU8(originArchive["state/brain.json"]!)
    ) as typeof exportedImmutableBrain;
    expect(exportedOriginBrain).toMatchObject({
      id: brain.id,
      name: brain.name,
      trainingSources: [],
      messages: []
    });
    expect(exportedImmutableBrain).toEqual(exportedOriginBrain);
    expect(exportedImmutableBrain.originChecksum).toBe(
      originChecksumForDocument(exportedImmutableBrain)
    );
    expect(exportedCurrentBrain.originChecksum).toBe(
      exportedImmutableBrain.originChecksum
    );
    expect(exportedImmutableBrain.originChecksum).not.toBe(
      brain.originChecksum
    );
    expect(
      JSON.parse(strFromU8(originArchive["state/engine.json"]!))
    ).toMatchObject({ marker: "immutable-origin" });

    const originImported = await repository.importBundle(originBundle);
    expect(originImported.trainingSources).toHaveLength(0);
    expect(originImported.messages).toHaveLength(0);
    expect(originImported.originChecksum).toBe(
      exportedImmutableBrain.originChecksum
    );
    await expect(
      readFile(join(repository.brainDirectory(originImported.id), "origin.json"))
    ).resolves.toEqual(Buffer.from(originArchive["origin/state/brain.json"]!));
    await expect(
      readFile(
        join(
          repository.brainDirectory(originImported.id),
          "engine",
          "origin",
          "brain.json"
        )
      )
    ).resolves.toEqual(originMetadata);

    const badChecksumArchive = unzipSync(
      new Uint8Array(await readFile(bundle))
    );
    const badChecksumOrigin = JSON.parse(
      strFromU8(badChecksumArchive["origin/state/brain.json"]!)
    ) as Record<string, unknown>;
    const badChecksumCurrent = JSON.parse(
      strFromU8(badChecksumArchive["state/brain.json"]!)
    ) as Record<string, unknown>;
    badChecksumOrigin.originChecksum = "0".repeat(64);
    badChecksumCurrent.originChecksum = "0".repeat(64);
    badChecksumArchive["origin/state/brain.json"] = Buffer.from(
      JSON.stringify(badChecksumOrigin, null, 2)
    );
    badChecksumArchive["state/brain.json"] = Buffer.from(
      JSON.stringify(badChecksumCurrent, null, 2)
    );
    refreshArchiveIntegrity(badChecksumArchive);
    await expect(
      repository.importBundleBuffer(
        Buffer.from(zipSync(badChecksumArchive)),
        "bad-origin-checksum.omni"
      )
    ).rejects.toThrow(/origin checksum does not match/i);

    const mismatchedOriginArchive = unzipSync(
      new Uint8Array(await readFile(bundle))
    );
    const mismatchedOrigin = JSON.parse(
      strFromU8(mismatchedOriginArchive["origin/state/brain.json"]!)
    ) as Record<string, unknown> & { config: Record<string, unknown> };
    const mismatchedCurrent = JSON.parse(
      strFromU8(mismatchedOriginArchive["state/brain.json"]!)
    ) as Record<string, unknown>;
    mismatchedOrigin.name = "A different UI origin";
    mismatchedOrigin.config.name = mismatchedOrigin.name;
    mismatchedOrigin.originChecksum = originChecksumForDocument(
      mismatchedOrigin
    );
    mismatchedCurrent.originChecksum = mismatchedOrigin.originChecksum;
    mismatchedOriginArchive["origin/state/brain.json"] = Buffer.from(
      JSON.stringify(mismatchedOrigin, null, 2)
    );
    mismatchedOriginArchive["state/brain.json"] = Buffer.from(
      JSON.stringify(mismatchedCurrent, null, 2)
    );
    refreshArchiveIntegrity(mismatchedOriginArchive);
    await expect(
      repository.importBundleBuffer(
        Buffer.from(zipSync(mismatchedOriginArchive)),
        "mismatched-origin-identities.omni"
      )
    ).rejects.toThrow(/UI origin does not match the immutable neural origin/i);

    const forgedProvenance = JSON.parse(provenance.toString("utf8")) as Record<
      string,
      unknown
    >;
    delete forgedProvenance.contentSha256;
    forgedProvenance.coreSha256 = "c".repeat(64);
    forgedProvenance.contentSha256 = digest(
      canonicalJson(forgedProvenance)
    );
    const forgedBytes = Buffer.from(canonicalJson(forgedProvenance));
    archive["origin/provenance.json"] = forgedBytes;
    const forgedManifest = JSON.parse(
      strFromU8(archive["manifest.json"]!)
    ) as {
      files: Record<string, { sha256: string; bytes: number }>;
    };
    forgedManifest.files["origin/provenance.json"] = {
      sha256: digest(forgedBytes),
      bytes: forgedBytes.byteLength
    };
    archive["manifest.json"] = Buffer.from(
      JSON.stringify(forgedManifest, null, 2)
    );
    archive["checksums.sha256"] = Buffer.from(
      Object.entries(archive)
        .filter(([path]) => path !== "checksums.sha256")
        .map(([path, contents]) => `${digest(contents)}  ${path}`)
        .sort()
        .join("\n") + "\n"
    );
    await expect(
      repository.importBundleBuffer(
        Buffer.from(zipSync(archive)),
        "forged-origin.omni"
      )
    ).rejects.toThrow(/provenance does not match its neural state/);

    const provenancePath = join(origin, "provenance.json");
    const unexpectedProvenance = JSON.parse(
      provenance.toString("utf8")
    ) as Record<string, unknown>;
    delete unexpectedProvenance.contentSha256;
    unexpectedProvenance.unexpectedField = "not allowed";
    unexpectedProvenance.contentSha256 = digest(
      canonicalJson(unexpectedProvenance)
    );
    await writeFile(provenancePath, canonicalJson(unexpectedProvenance));
    await expect(
      repository.exportBundle(
        brain.id,
        join(temporaryRoot, "unexpected-provenance.omni"),
        "current"
      )
    ).rejects.toThrow(/provenance does not match its neural state/);

    const secretProvenance = JSON.parse(
      provenance.toString("utf8")
    ) as Record<string, unknown>;
    delete secretProvenance.contentSha256;
    secretProvenance.binaryLookingPadding = "�".repeat(4_000);
    secretProvenance.note = `api_key=sk-${"z".repeat(32)}`;
    secretProvenance.contentSha256 = digest(canonicalJson(secretProvenance));
    await writeFile(provenancePath, canonicalJson(secretProvenance));
    await expect(
      repository.exportBundle(
        brain.id,
        join(temporaryRoot, "secret-provenance.omni"),
        "current"
      )
    ).rejects.toThrow(/appears to contain credentials/);

    await rm(provenancePath);
    await expect(
      repository.exportBundle(
        brain.id,
        join(temporaryRoot, "missing-provenance.omni"),
        "current"
      )
    ).rejects.toThrow(/missing immutable-origin provenance/);

    const wrongStarterState = JSON.parse(
      originMetadata.toString("utf8")
    ) as {
      starter_training_manifest: { sha256: string };
    };
    wrongStarterState.starter_training_manifest.sha256 = "a".repeat(64);
    await writeFile(join(origin, "brain.json"), canonicalJson(wrongStarterState));
    await writeOriginProvenanceFixture(engine);
    await expect(
      repository.exportBundle(
        brain.id,
        join(temporaryRoot, "wrong-starter.omni"),
        "current"
      )
    ).rejects.toThrow(/official Omni Starter manifest/);

    await Promise.all([
      writeFile(join(origin, "brain.json"), originMetadata),
      writeFile(provenancePath, provenance)
    ]);
  }, 30_000);

  it("enumerates only app-managed beta directories and deletes them only after confirmation", async () => {
    const stable = await repository.create({ ...DEFAULT_CONFIG, name: "Stable mind" });
    const betaId = "managed-beta";
    const betaDirectory = repository.brainDirectory(betaId);
    await mkdir(betaDirectory, { recursive: true });
    await writeFile(
      join(betaDirectory, "brain.json"),
      JSON.stringify({
        schemaVersion: 1,
        id: betaId,
        name: "Old beta mind"
      })
    );
    const externalBundle = join(temporaryRoot, "external-beta.omni");
    await writeFile(externalBundle, "external beta file");

    const candidates = await repository.enumerateManagedBetaBrains();
    expect(candidates).toEqual([
      expect.objectContaining({
        id: betaId,
        name: "Old beta mind",
        path: betaDirectory,
        reason: "beta-document"
      })
    ]);
    await expect(
      repository.deleteManagedBetaBrains([betaId], false)
    ).rejects.toThrow(/explicit confirmation/i);
    await expect(readFile(join(betaDirectory, "brain.json"), "utf8")).resolves.toContain(
      "Old beta mind"
    );

    await expect(
      repository.deleteManagedBetaBrains([betaId], true)
    ).resolves.toEqual([betaId]);
    await expect(stat(betaDirectory)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(repository.get(stable.id)).resolves.toMatchObject({ id: stable.id });
    await expect(readFile(externalBundle, "utf8")).resolves.toBe("external beta file");

    expect(await repository.betaReviewComplete()).toBe(false);
    await repository.completeBetaReview("deleted", [betaId]);
    expect(await repository.betaReviewComplete()).toBe(true);
  });

  it("rejects beta local documents and beta materialized engine exports", async () => {
    const betaId = "beta-local";
    const betaDirectory = repository.brainDirectory(betaId);
    await mkdir(betaDirectory, { recursive: true });
    await writeFile(
      join(betaDirectory, "brain.json"),
      JSON.stringify({
        schemaVersion: 1,
        id: betaId,
        name: "Beta local",
        config: DEFAULT_CONFIG
      })
    );
    await expect(repository.get(betaId)).rejects.toThrow(/beta brain/i);

    const stable = await repository.create({ ...DEFAULT_CONFIG, name: "Stable shell" });
    const engine = join(repository.brainDirectory(stable.id), "engine");
    await mkdir(engine, { recursive: true });
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        brain_id: stable.id
      })
    );
    await expect(
      repository.exportBundle(stable.id, join(temporaryRoot, "beta-engine.omni"))
    ).rejects.toThrow(/beta engine/i);
  });

  it("snapshots and restores both inspectable and neural state", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Snapshot mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        marker: "before",
        substrate: { persistence: substratePointer }
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
    await writePackedTernaryFixture(join(engine, "packed-ternary"));
    const packedBefore = await readFile(
      join(engine, "packed-ternary", "manifest.json")
    );
    const snapshot = await repository.snapshot(brain.id, "before mutation");

    const mutated = await repository.get(brain.id);
    mutated.name = "Mutated";
    mutated.config.name = "Mutated";
    await repository.save(mutated);
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        marker: "after"
      })
    );
    await writeFile(join(engine, "packed-ternary", "manifest.json"), "{}");
    await rm(join(engine, "substrate"), { recursive: true, force: true });

    const stagedFailure = vi
      .spyOn(repository, "storeFileAsBlob")
      .mockRejectedValueOnce(new Error("simulated snapshot staging failure"));
    try {
      await expect(repository.restoreSnapshot(brain.id, snapshot.id)).rejects.toThrow(
        /simulated snapshot staging failure/
      );
    } finally {
      stagedFailure.mockRestore();
    }
    await expect(repository.get(brain.id)).resolves.toMatchObject({ name: "Mutated" });
    await expect(
      readFile(join(engine, "brain.json"), "utf8")
    ).resolves.toContain('"marker":"after"');

    const restored = await repository.restoreSnapshot(brain.id, snapshot.id);
    const engineState = JSON.parse(await readFile(join(engine, "brain.json"), "utf8")) as {
      marker: string;
    };
    expect(restored.name).toBe("Snapshot mind");
    expect(engineState.marker).toBe("before");
    expect(snapshot.engineChecksum).toMatch(/^[a-f0-9]{64}$/);
    await expect(
      readFile(join(engine, "packed-ternary", "manifest.json"))
    ).resolves.toEqual(packedBefore);
    await expect(
      readFile(join(engine, "substrate", "manifest.json"), "utf8")
    ).resolves.toBe(canonicalJson(substratePointer));
  });

  it("round-trips a checksum-verified ZIP and omits private sources by default", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Portable mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    await writeSubstrateFixture(join(engine, "origin"));
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: { name: brain.name },
      expert_count: 0,
      training_sources: [],
      substrate: {
        schema: 1,
        dimensions: 16,
        seed: 7,
        persistence: substratePointer
      }
    };
    await Promise.all([
      writeFile(join(engine, "brain.json"), JSON.stringify(engineState)),
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors()),
      writeFile(join(engine, "origin", "brain.json"), JSON.stringify(engineState)),
      writeFile(join(engine, "origin", "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "origin", "plasticity.safetensors"), emptySafetensors())
    ]);
    await Promise.all([
      writePackedTernaryFixture(join(engine, "packed-ternary")),
      writePackedTernaryFixture(join(engine, "origin", "packed-ternary"))
    ]);
    const sourceBytes = Buffer.from("private source material");
    const copiedCredential = `sk-${"a".repeat(32)}`;
    brain.messages.push({
      id: randomUUID(),
      role: "human",
      content: `Do not share ${copiedCredential}`,
      createdAt: new Date().toISOString()
    });
    brain.journal?.push({
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      kind: "system",
      summary: `credential=${copiedCredential}`
    });
    const blobHash = await repository.storeBlob(sourceBytes);
    brain.trainingSources.push({
      id: randomUUID(),
      name: "private.txt",
      path: "C:\\private\\private.txt",
      kind: "text",
      bytes: sourceBytes.byteLength,
      learnedIdeas: 1,
      learnedConcepts: 2,
      learnedSynapses: 2,
      importedAt: new Date().toISOString(),
      rawTextRetained: true,
      rawText: sourceBytes.toString("utf8"),
      contentHash: blobHash,
      blobHash,
      policy: "archive"
    });
    await repository.save(brain);

    const portablePath = join(temporaryRoot, "portable.omni");
    await repository.exportBundle(brain.id, portablePath, "current");
    const entries = unzipSync(new Uint8Array(await readFile(portablePath)));
    const portableState = JSON.parse(strFromU8(entries["state/brain.json"]!)) as {
      trainingSources: Array<Record<string, unknown>>;
    };
    expect(Object.keys(entries)).toEqual(
      expect.arrayContaining([
        "manifest.json",
        "checksums.sha256",
        "state/brain.json",
        "state/engine.json",
        "tensors/core.safetensors",
        "packed/current/manifest.json",
        "packed/origin/manifest.json",
        "substrate/current/manifest.json",
        `substrate/current/${String(substratePointer.generationManifest)}`,
        "substrate/origin/manifest.json",
        "origin/state/brain.json"
      ])
    );
    expect(Object.keys(entries).some((name) => name.startsWith("blobs/"))).toBe(false);
    expect(portableState.trainingSources[0]?.rawText).toBeUndefined();
    expect(portableState.trainingSources[0]?.path).toBeUndefined();
    expect(strFromU8(entries["state/brain.json"]!)).not.toContain(copiedCredential);
    expect(strFromU8(entries["state/brain.json"]!)).toContain("[REDACTED_SECRET]");
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as {
      architecture: string;
      architectureSchemaVersion: number;
      secretRedaction: { replacements: number };
      licenseLedger: { application: string; sources: Array<{ name: string; license: string }> };
    };
    expect(manifest).toMatchObject({
      architecture: "OmniCortex",
      architectureSchemaVersion: 1,
      packedTernary: {
        format: "omni-packed-ternary",
        formatVersion: 1,
        currentTensorCount: 1,
        originTensorCount: 1
      }
    });
    expect(manifest.secretRedaction.replacements).toBeGreaterThan(0);
    expect(manifest.licenseLedger.application).toContain("PolyForm");
    expect(manifest.licenseLedger.sources).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "private.txt",
          license: "Undeclared; verify before redistribution"
        })
      ])
    );

    const imported = await repository.importBundle(portablePath);
    expect(imported.id).not.toBe(brain.id);
    expect(imported.name).toBe(brain.name);
    const importedEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8")
    ) as { brain_id: string };
    expect(importedEngine.brain_id).toBe(imported.id);
    await expect(
      readFile(
        join(
          repository.brainDirectory(imported.id),
          "engine",
          "packed-ternary",
          "manifest.json"
        )
      )
    ).resolves.toBeInstanceOf(Buffer);
    await expect(
      readFile(
        join(
          repository.brainDirectory(imported.id),
          "engine",
          "substrate",
          "manifest.json"
        ),
        "utf8"
      )
    ).resolves.toBe(canonicalJson(substratePointer));

    const visibleBeforeFailure = (await repository.list()).map((item) => item.id).sort();
    const directoriesBeforeFailure = (await readdir(repository.root, { withFileTypes: true }))
      .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
      .map((entry) => entry.name)
      .sort();
    const storeFailure = vi
      .spyOn(repository, "storeFileAsBlob")
      .mockRejectedValueOnce(new Error("simulated streamed install failure"));
    try {
      await expect(repository.importBundle(portablePath)).rejects.toThrow(
        /simulated streamed install failure/
      );
    } finally {
      storeFailure.mockRestore();
    }
    expect((await repository.list()).map((item) => item.id).sort()).toEqual(
      visibleBeforeFailure
    );
    expect(
      (await readdir(repository.root, { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
        .map((entry) => entry.name)
        .sort()
    ).toEqual(directoriesBeforeFailure);

    const privatePath = join(temporaryRoot, "private.omni");
    await repository.exportBundle(brain.id, privatePath, "private-archive");
    const privateEntries = unzipSync(new Uint8Array(await readFile(privatePath)));
    expect(privateEntries[`blobs/${blobHash}`]).toBeDefined();
    await repository.importBundle(privatePath);
    await expect(repository.getBlob(blobHash)).resolves.toEqual(sourceBytes);
  }, 20_000);

  it("rejects ZIP traversal entries before extraction", async () => {
    const malicious = Buffer.from(
      zipSync({
        "../outside.exe": new Uint8Array([1, 2, 3])
      })
    );
    await expect(repository.importBundleBuffer(malicious, "malicious.omni")).rejects.toThrow(
      /Unsafe path|Executable content/
    );
  });

  it("rejects an otherwise checksummed bundle for another architecture schema", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Schema mind" });
    const portablePath = join(temporaryRoot, "schema.omni");
    await repository.exportBundle(brain.id, portablePath, "current");
    const entries = unzipSync(new Uint8Array(await readFile(portablePath)));
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as Record<string, unknown>;
    manifest.architectureSchemaVersion = 999;
    entries["manifest.json"] = Buffer.from(JSON.stringify(manifest, null, 2));
    entries["checksums.sha256"] = Buffer.from(
      Object.entries(entries)
        .filter(([path]) => path !== "checksums.sha256")
        .map(([path, contents]) => `${digest(contents)}  ${path}`)
        .sort()
        .join("\n") + "\n"
    );
    const incompatible = Buffer.from(zipSync(entries));
    await expect(repository.importBundleBuffer(incompatible, "future.omni")).rejects.toThrow(
      /incompatible architecture schema/
    );
  });

  it("rejects a checksum-valid beta bundle without the stable release discriminator", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Stable bundle" });
    const portablePath = join(temporaryRoot, "stable.omni");
    await repository.exportBundle(brain.id, portablePath, "current");
    const entries = unzipSync(new Uint8Array(await readFile(portablePath)));
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as Record<
      string,
      unknown
    >;
    delete manifest.releaseFormat;
    entries["manifest.json"] = Buffer.from(JSON.stringify(manifest, null, 2));
    entries["checksums.sha256"] = Buffer.from(
      Object.entries(entries)
        .filter(([path]) => path !== "checksums.sha256")
        .map(([path, contents]) => `${digest(contents)}  ${path}`)
        .sort()
        .join("\n") + "\n"
    );
    await expect(
      repository.importBundleBuffer(
        Buffer.from(zipSync(entries)),
        "old-beta.omni"
      )
    ).rejects.toThrow(/beta .omni bundle/i);
  });

  it("refuses a private source archive when retained text appears to contain a credential", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Private mind" });
    const secretBytes = Buffer.from(`api_key=sk-${"b".repeat(36)}`);
    const blobHash = await repository.storeBlob(secretBytes);
    brain.trainingSources.push({
      id: randomUUID(),
      name: "credentials.txt",
      path: "C:\\private\\credentials.txt",
      kind: "text",
      bytes: secretBytes.byteLength,
      learnedIdeas: 0,
      learnedConcepts: 0,
      learnedSynapses: 0,
      importedAt: new Date().toISOString(),
      rawTextRetained: true,
      rawText: secretBytes.toString("utf8"),
      contentHash: blobHash,
      blobHash,
      policy: "archive"
    });
    await repository.save(brain);
    await expect(
      repository.exportBundle(brain.id, join(temporaryRoot, "credentials.omni"), "private-archive")
    ).rejects.toThrow(/appears to contain credentials/);
  });

  it("round-trips a lightweight local reference and fails clearly without its blob store", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Referenced mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: {},
      expert_count: 0,
      training_sources: []
    };
    const core = byteTensorSafetensors(256 * 1024);
    const plastic = byteTensorSafetensors(64 * 1024);
    await Promise.all([
      writeFile(join(engine, "brain.json"), JSON.stringify(engineState)),
      writeFile(join(engine, "core.safetensors"), core),
      writeFile(join(engine, "plasticity.safetensors"), plastic),
      writeFile(join(engine, "origin", "brain.json"), JSON.stringify(engineState)),
      writeFile(join(engine, "origin", "core.safetensors"), core),
      writeFile(join(engine, "origin", "plasticity.safetensors"), plastic)
    ]);
    await Promise.all([
      writePackedTernaryFixture(join(engine, "packed-ternary")),
      writePackedTernaryFixture(join(engine, "origin", "packed-ternary"))
    ]);

    const portablePath = join(temporaryRoot, "full.omni");
    const referencePath = join(temporaryRoot, "reference.omni");
    await repository.exportBundle(brain.id, portablePath, "current");
    await repository.exportBundle(brain.id, referencePath, "referenced");
    const [portableBytes, referenceBytes] = await Promise.all([
      readFile(portablePath),
      readFile(referencePath)
    ]);
    expect(referenceBytes.byteLength).toBeLessThan(portableBytes.byteLength / 2);
    const referenceEntries = unzipSync(new Uint8Array(referenceBytes));
    const referenceManifest = JSON.parse(
      strFromU8(referenceEntries["manifest.json"]!)
    ) as {
      mode: string;
      references: Record<string, string>;
    };
    expect(referenceManifest.mode).toBe("referenced-local");
    expect(Object.values(referenceManifest.references)).toHaveLength(4);
    expect(
      Object.values(referenceManifest.references).every((hash) =>
        /^[a-f0-9]{64}$/.test(hash)
      )
    ).toBe(true);

    const imported = await repository.importBundle(referencePath);
    const importedCorePath = join(
      repository.brainDirectory(imported.id),
      "engine",
      "core.safetensors"
    );
    await expect(readFile(importedCorePath)).resolves.toEqual(core);
    const [blobInfo, importedInfo] = await Promise.all([
      stat(join(repository.root, ".blobs", referenceManifest.references.currentCore!)),
      stat(importedCorePath)
    ]);
    expect(importedInfo.ino).toBe(blobInfo.ino);

    const separate = new BrainRepository(join(temporaryRoot, "separate-brains"));
    await separate.initialize();
    await expect(
      separate.importBundleBuffer(referenceBytes, "reference.omni")
    ).rejects.toThrow(/unavailable on this installation/);

    referenceManifest.references.currentCore = "not-a-content-hash";
    referenceEntries["manifest.json"] = Buffer.from(
      JSON.stringify(referenceManifest, null, 2)
    );
    referenceEntries["checksums.sha256"] = Buffer.from(
      Object.entries(referenceEntries)
        .filter(([path]) => path !== "checksums.sha256")
        .map(([path, contents]) => `${digest(contents)}  ${path}`)
        .sort()
        .join("\n") + "\n"
    );
    await expect(
      repository.importBundleBuffer(
        Buffer.from(zipSync(referenceEntries)),
        "tampered-reference.omni"
      )
    ).rejects.toThrow(/invalid tensor reference/);
  }, 30_000);
});
