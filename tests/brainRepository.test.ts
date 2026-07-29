import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { strFromU8, unzipSync, zipSync } from "fflate";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { DEFAULT_CONFIG } from "../src/shared/types";

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
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
    .join(",")}}`;
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
  const manifest = {
    ...manifestBody,
    contentSha256: digest(canonicalJson(manifestBody))
  };
  const manifestBytes = Buffer.from(canonicalJson(manifest));
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

  it("creates an immutable origin and copy-on-write neural fork", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Ada" });
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
        config: {},
        expert_count: 0
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
    expect(fork.lineage.rootId).toBe(brain.id);
    expect(forkEngine.brain_id).toBe(fork.id);
    expect(forkEngine.name).toBe("Ada branch");
    await expect(
      readFile(join(repository.brainDirectory(fork.id), "engine", "origin", "core.safetensors"))
    ).resolves.toBeInstanceOf(Buffer);
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
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        marker: "before"
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
  });

  it("round-trips a checksum-verified ZIP and omits private sources by default", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Portable mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: { name: brain.name },
      expert_count: 0,
      training_sources: []
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

    const privatePath = join(temporaryRoot, "private.omni");
    await repository.exportBundle(brain.id, privatePath, "private-archive");
    const privateEntries = unzipSync(new Uint8Array(await readFile(privatePath)));
    expect(privateEntries[`blobs/${blobHash}`]).toBeDefined();
  });

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
  });
});
