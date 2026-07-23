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

  it("creates an immutable origin and copy-on-write neural fork", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Ada" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
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

  it("snapshots and restores both inspectable and neural state", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Snapshot mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        brain_id: brain.id,
        marker: "before"
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
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
        brain_id: brain.id,
        marker: "after"
      })
    );

    const restored = await repository.restoreSnapshot(brain.id, snapshot.id);
    const engineState = JSON.parse(await readFile(join(engine, "brain.json"), "utf8")) as {
      marker: string;
    };
    expect(restored.name).toBe("Snapshot mind");
    expect(engineState.marker).toBe("before");
    expect(snapshot.engineChecksum).toMatch(/^[a-f0-9]{64}$/);
  });

  it("round-trips a checksum-verified ZIP and omits private sources by default", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Portable mind" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
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
      architectureSchemaVersion: 1
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
