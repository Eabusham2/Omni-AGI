import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { strFromU8, unzipSync, zipSync } from "fflate";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrainRepository,
  MUTABLE_STATE_STORE_FORMAT,
  SUBSTRATE_STORE_FORMAT
} from "../src/main/brainRepository";
import {
  ArtifactIndexStore,
  serializeArtifactIndex
} from "../src/main/mediaArtifactRegistry";
import { verifyPortableReplaySqlite } from "../src/main/portableReplayIntegrity";
import type { BrainStorageOperationHooks } from "../src/main/brainStorageOperations";
import { DEFAULT_CONFIG, type DiskSpaceTelemetry } from "../src/shared/types";

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

function safeDiskTelemetry(): DiskSpaceTelemetry {
  return {
    schemaVersion: 1,
    measuredAt: new Date().toISOString(),
    diskTotalBytes: 100 * 1024 ** 3,
    diskFreeBytes: 80 * 1024 ** 3,
    mandatoryReserveBytes: 20 * 1024 ** 3,
    selectedDatasetBytes: 0,
    modelBytes: 0,
    checkpointBytes: 0,
    maximumWorkingMemorySpillBytes: 0,
    futureGrowthBytes: 0,
    operationWriteBytes: 0,
    projectedRemainingBytes: 80 * 1024 ** 3,
    projectedAboveReserveBytes: 60 * 1024 ** 3,
    paused: false
  };
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
  const { readiness: _readiness, ...originState } = value;
  return digest(JSON.stringify({ ...originState, originChecksum: undefined }));
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
  const canonicalBody = canonicalJson(manifestBody).replace('"scale":1,', '"scale":1.0,');
  const manifest = {
    ...manifestBody,
    contentSha256: digest(canonicalBody)
  };
  const manifestBytes = Buffer.from(canonicalJson(manifest).replace('"scale":1,', '"scale":1.0,'));
  await mkdir(directory, { recursive: true });
  await Promise.all([
    writeFile(join(directory, "manifest.json"), manifestBytes),
    writeFile(join(directory, "manifest.sha256"), `${digest(manifestBytes)}\n`),
    writeFile(join(directory, `ternary-00000-${shardHash.slice(0, 16)}.bin`), shard)
  ]);
}

async function materializeNativeEngineFixture(
  repository: BrainRepository,
  brain: { id: string; name: string }
): Promise<void> {
  const engine = join(repository.brainDirectory(brain.id), "engine");
  const origin = join(engine, "origin");
  await Promise.all([
    mkdir(engine, { recursive: true }),
    mkdir(origin, { recursive: true })
  ]);
  const state = {
    schema_version: 1,
    format: "omni-cortex-engine",
    release_format: "stable-1.0",
    brain_id: brain.id,
    name: brain.name,
    config: {
      name: brain.name,
      origin_kind: "ground-up"
    },
    runtime_card: {
      origin_kind: "ground-up",
      pretrained: false,
      baseFrozen: false
    }
  };
  await Promise.all([
    writeFile(join(engine, "brain.json"), JSON.stringify(state)),
    writeFile(join(engine, "core.safetensors"), emptySafetensors()),
    writeFile(join(engine, "plasticity.safetensors"), emptySafetensors()),
    writeFile(join(origin, "brain.json"), JSON.stringify(state)),
    writeFile(join(origin, "core.safetensors"), emptySafetensors()),
    writeFile(join(origin, "plasticity.safetensors"), emptySafetensors()),
    writePackedTernaryFixture(join(engine, "packed-ternary")),
    writePackedTernaryFixture(join(origin, "packed-ternary"))
  ]);
}

async function writeSubstrateFixture(
  engineDirectory: string,
  sourceLabel?: string,
  generationNote?: string,
  formatVersion: 1 | 2 = 1
): Promise<Record<string, unknown>> {
  const store = join(engineDirectory, "substrate");
  const record = {
    kind: "neurons",
    ids: ["neuron-fixture"],
    records: [{
      id: "neuron-fixture",
      region: "cortical",
      ...(sourceLabel === undefined ? {} : { source_label: sourceLabel })
    }],
    vectorIds: []
  };
  const recordBytes = Buffer.from(canonicalJson(record));
  const recordHash = digest(recordBytes);
  const generationBody = {
    format: "omni-substrate-shards",
    formatVersion,
    schema: 1,
    dimensions: 16,
    seed: 7,
    growthEvents: 1,
    growthPauses: 0,
    recordsPerShard: 512,
    counts: { neurons: 1, assemblies: 0, synapses: 0 },
    ...(generationNote === undefined ? {} : { source_note: generationNote }),
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
    formatVersion,
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

async function materializeSubstrateExportFixture(
  repository: BrainRepository,
  brain: { id: string; name: string },
  labels: {
    current?: string;
    origin?: string;
    currentManifest?: string;
    originManifest?: string;
  } = {},
  formatVersion: 1 | 2 = 1
): Promise<void> {
  await materializeNativeEngineFixture(repository, brain);
  const engine = join(repository.brainDirectory(brain.id), "engine");
  const currentPointer = await writeSubstrateFixture(
    engine,
    labels.current,
    labels.currentManifest,
    formatVersion
  );
  const originPointer = await writeSubstrateFixture(
    join(engine, "origin"),
    labels.origin,
    labels.originManifest,
    formatVersion
  );
  for (const [directory, pointer] of [
    [engine, currentPointer],
    [join(engine, "origin"), originPointer]
  ] as const) {
    const statePath = join(directory, "brain.json");
    const state = JSON.parse(await readFile(statePath, "utf8")) as Record<string, unknown>;
    state.substrate = { schema: 1, dimensions: 16, seed: 7, persistence: pointer };
    await writeFile(statePath, JSON.stringify(state));
  }
}

async function writeMutableStateFixture(
  engineDirectory: string,
  keepWalOpen = false
): Promise<{
  pointer: Record<string, unknown>;
  replay: Buffer;
  writer?: DatabaseSync;
}> {
  const store = join(engineDirectory, "state");
  const tensors = emptySafetensors();
  const tensorHash = digest(tensors);
  const tensorDescriptor = {
    path: `blobs/${tensorHash}.safetensors`,
    sha256: tensorHash,
    bytes: tensors.byteLength,
    tensorCount: 0
  };
  const replayRows = [1, 2, 3].map((number) => {
    const payload = Buffer.alloc(4);
    payload.writeFloatLE(number);
    const sha256 = digest(Buffer.concat([
      Buffer.from("torch.float32\0[1]\0", "ascii"), payload
    ]));
    return { payload, sha256 };
  });
  const replayContentSha256 = digest(replayRows.map((row, index) =>
    `${index + 1}\0${row.sha256}\n`
  ).join(""));
  const generationBody = {
    format: "omni-mutable-state",
    formatVersion: 1,
    brainId: "fixture-brain",
    createdAt: 1,
    roles: {
      core: tensorDescriptor,
      plasticity: tensorDescriptor,
      optimizer: tensorDescriptor
    },
    optimizerStructure: {
      type: "dict",
      items: [
        [
          { type: "scalar", value: "param_groups" },
          {
            type: "list",
            items: [
              {
                type: "dict",
                items: [
                  [
                    { type: "scalar", value: "eps" },
                    { type: "scalar", value: 1e-8 }
                  ]
                ]
              }
            ]
          }
        ]
      ]
    },
    replay: {
      format: "omni-replay-sqlite",
      formatVersion: 1,
      path: "replay.sqlite3",
      count: 3,
      highWaterId: 3,
      contentSha256: replayContentSha256,
      transactional: true,
      silentEviction: false
    },
    metadata: { trainingSteps: 7 },
    activationState: ["state.liquid", "state.working_memory"],
    safeTensorOnly: true,
    transactional: true
  };
  // Python's json.dumps preserves the leading zero in a negative exponent;
  // JavaScript JSON.stringify does not. The production worker therefore
  // writes `1e-08`, and the repository must verify those exact canonical
  // bytes rather than reserializing the parsed object as `1e-8`.
  const generationBodyText = canonicalJson(generationBody).replace('"value":1e-8', '"value":1e-08');
  expect(generationBodyText).toContain('"value":1e-08');
  const contentHash = digest(generationBodyText);
  const generationText = generationBodyText.replace(
    `"brainId":"fixture-brain","createdAt":`,
    `"brainId":"fixture-brain","contentSha256":"${contentHash}","createdAt":`
  );
  expect(generationText).toContain(`"contentSha256":"${contentHash}"`);
  const generationBytes = Buffer.from(generationText);
  const generationRelative = `generations/${contentHash}/manifest.json`;
  const pointer = {
    format: "omni-mutable-state",
    formatVersion: 1,
    activeGeneration: contentHash,
    contentSha256: contentHash,
    generationManifest: generationRelative,
    generationManifestSha256: digest(generationBytes),
    replayCount: 3
  };
  await Promise.all([
    mkdir(join(store, "blobs"), { recursive: true }),
    mkdir(join(store, "generations", contentHash), { recursive: true })
  ]);
  await Promise.all([
    writeFile(join(store, "blobs", `${tensorHash}.safetensors`), tensors),
    writeFile(join(store, ...generationRelative.split("/")), generationBytes),
    writeFile(join(store, "manifest.json"), canonicalJson(pointer))
  ]);
  const replayPath = join(store, "replay.sqlite3");
  const database = new DatabaseSync(replayPath);
  let handedOff = false;
  try {
    if (keepWalOpen) {
      database.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;");
    }
    database.exec(`
      CREATE TABLE replay (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at REAL NOT NULL,
        sha256 TEXT NOT NULL,
        dtype TEXT NOT NULL,
        shape_json TEXT NOT NULL,
        payload BLOB NOT NULL
      )
    `);
    const insert = database.prepare(
      "INSERT INTO replay (created_at,sha256,dtype,shape_json,payload) VALUES (?,?,?,?,?)"
    );
    for (const [index, row] of replayRows.entries()) {
      insert.run(index + 1, row.sha256, "torch.float32", "[1]", row.payload);
    }
    const replay = await readFile(replayPath);
    handedOff = keepWalOpen;
    return { pointer, replay, ...(keepWalOpen ? { writer: database } : {}) };
  } finally {
    if (!handedOff) database.close();
  }
}

async function materializeMutableReplayExportFixture(
  repository: BrainRepository,
  brain: { id: string; name: string },
  walScope?: "current" | "origin"
): Promise<{
  currentReplayPath: string;
  originReplayPath: string;
  walWriter?: DatabaseSync;
}> {
  await materializeSubstrateExportFixture(repository, brain);
  const engine = join(repository.brainDirectory(brain.id), "engine");
  const origin = join(engine, "origin");
  const [currentState, originState] = await Promise.all([
    writeMutableStateFixture(engine, walScope === "current"),
    writeMutableStateFixture(origin, walScope === "origin")
  ]);
  for (const [directory, state] of [
    [engine, currentState],
    [origin, originState]
  ] as const) {
    const metadataPath = join(directory, "brain.json");
    const metadata = JSON.parse(await readFile(metadataPath, "utf8")) as Record<string, unknown>;
    metadata.mutable_state = state.pointer;
    await writeFile(metadataPath, JSON.stringify(metadata));
  }
  return {
    currentReplayPath: join(engine, "state", "replay.sqlite3"),
    originReplayPath: join(origin, "state", "replay.sqlite3"),
    ...(walScope ? { walWriter: walScope === "current" ? currentState.writer : originState.writer } : {})
  };
}

function replayRowCount(path: string): number {
  const database = new DatabaseSync(path, { readOnly: true });
  try {
    const row = database.prepare("SELECT COUNT(*) AS count FROM replay").get() as
      | { count: number }
      | undefined;
    return Number(row?.count ?? 0);
  } finally {
    database.close();
  }
}

async function assertDeclaredNeuralGenerationsLoadable(engineDirectory: string): Promise<void> {
  const metadata = JSON.parse(
    await readFile(join(engineDirectory, "brain.json"), "utf8")
  ) as Record<string, unknown>;
  const substrate = metadata.substrate as { persistence?: Record<string, unknown> } | undefined;
  const substratePointer = substrate?.persistence;
  if (substratePointer) {
    const relative = String(substratePointer.generationManifest);
    const bytes = await readFile(join(engineDirectory, "substrate", ...relative.split("/")));
    expect(digest(bytes)).toBe(substratePointer.generationManifestSha256);
    const generation = JSON.parse(bytes.toString("utf8")) as {
      shards: Array<{
        records: { path: string; sha256: string; bytes: number };
        tensors: { path: string; sha256: string; bytes: number } | null;
      }>;
    };
    for (const descriptor of generation.shards.flatMap((shard) =>
      [shard.records, shard.tensors].filter(
        (value): value is { path: string; sha256: string; bytes: number } => value !== null
      )
    )) {
      const blob = await readFile(
        join(engineDirectory, "substrate", ...descriptor.path.split("/"))
      );
      expect(blob.byteLength).toBe(descriptor.bytes);
      expect(digest(blob)).toBe(descriptor.sha256);
    }
  }
  const mutablePointer = metadata.mutable_state as Record<string, unknown> | undefined;
  if (mutablePointer) {
    const relative = String(mutablePointer.generationManifest);
    const bytes = await readFile(join(engineDirectory, "state", ...relative.split("/")));
    expect(digest(bytes)).toBe(mutablePointer.generationManifestSha256);
    const generation = JSON.parse(bytes.toString("utf8")) as {
      roles: Record<string, { path: string; sha256: string; bytes: number }>;
    };
    for (const descriptor of Object.values(generation.roles)) {
      const blob = await readFile(join(engineDirectory, "state", ...descriptor.path.split("/")));
      expect(blob.byteLength).toBe(descriptor.bytes);
      expect(digest(blob)).toBe(descriptor.sha256);
    }
    await expect(
      readFile(join(engineDirectory, "state", "replay.sqlite3"))
    ).resolves.toBeInstanceOf(Buffer);
  }
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
      new RegExp(`_SUBSTRATE_STORE_FORMAT\\s*=\\s*["']${SUBSTRATE_STORE_FORMAT}["']`)
    );
    expect(pythonSource).toMatch(/hexdigest\(\)\[:1\]/);
  });

  it("shares the authoritative mutable-state format with the Python worker", async () => {
    const pythonSource = await readFile(
      join(process.cwd(), "engine", "omni_core", "offload.py"),
      "utf8"
    );
    expect(pythonSource).toContain(`"format": "${MUTABLE_STATE_STORE_FORMAT}"`);
  });

  it("accepts checksum-validated packed-v2 native substrate totals", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Packed-v2 substrate fixture"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const pointer = await writeSubstrateFixture(engine, undefined, undefined, 2);
    await writeFile(
      join(engine, "brain.json"),
      canonicalJson({ brain_id: brain.id, substrate: { persistence: pointer } })
    );
    await expect(repository.persistedSubstrateOverview(brain.id)).resolves.toMatchObject({
      revision: pointer.activeGeneration,
      totals: { neurons: 1, assemblies: 0, synapses: 0 }
    });
  });

  it("returns only checksum-validated persisted substrate totals", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Persisted overview fixture"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const pointer = await writeSubstrateFixture(engine);
    await writeFile(
      join(engine, "brain.json"),
      canonicalJson({
        brain_id: brain.id,
        substrate: { persistence: pointer },
        runtime_card: {
          parameterAccounting: {
            mutableDenseParameters: 5,
            substrateDynamicSparseSynapses: 0,
            dynamicSparseSynapses: 0,
            totalNeuralParameters: 5,
            countingRule: "fixture exact non-overlapping count"
          }
        }
      })
    );

    await expect(repository.persistedSubstrateOverview(brain.id)).resolves.toEqual({
      brainId: brain.id,
      revision: pointer.activeGeneration,
      source: "validated-persisted-substrate",
      totals: { neurons: 1, assemblies: 0, synapses: 0 },
      parameterAccounting: {
        mutableDenseParameters: 5,
        substrateDynamicSparseSynapses: 0,
        dynamicSparseSynapses: 0,
        totalNeuralParameters: 5,
        countingRule: "fixture exact non-overlapping count"
      }
    });
    expect((await repository.list()).find((entry) => entry.id === brain.id)).toMatchObject({
      substrateTotals: { neurons: 1, assemblies: 0, synapses: 0 }
    });

    await writeFile(
      join(engine, "brain.json"),
      canonicalJson({
        brain_id: brain.id,
        substrate: { persistence: pointer },
        runtime_card: {
          parameterAccounting: {
            mutableDenseParameters: 5,
            substrateDynamicSparseSynapses: 0,
            dynamicSparseSynapses: 1,
            totalNeuralParameters: 6,
            countingRule: "mismatched dynamic count"
          }
        }
      })
    );
    await expect(
      repository.persistedSubstrateOverview(brain.id)
    ).rejects.toThrow(/parameter accounting is invalid/i);
    await writeFile(
      join(engine, "brain.json"),
      canonicalJson({
        brain_id: brain.id,
        substrate: { persistence: pointer }
      })
    );

    await writeFile(
      join(engine, "substrate", "manifest.json"),
      canonicalJson({
        ...pointer,
        counts: { neurons: 2, assemblies: 0, synapses: 0 }
      })
    );
    await expect(
      repository.persistedSubstrateOverview(brain.id)
    ).rejects.toThrow(/pointer does not match engine state/i);

    await writeFile(
      join(engine, "substrate", "manifest.json"),
      canonicalJson(pointer)
    );
    const generationPath = join(
      engine,
      "substrate",
      ...String(pointer.generationManifest).split("/")
    );
    await writeFile(generationPath, `${await readFile(generationPath, "utf8")} `);
    await expect(
      repository.persistedSubstrateOverview(brain.id)
    ).rejects.toThrow(/generation checksum failed/i);
  });

  it("validates native parameter accounting and rejects old sequence counters", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Native accounting fixture" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const pointer = await writeSubstrateFixture(engine);
    const accounting = {
      mutableDenseParameters: 120,
      substrateDynamicSparseSynapses: 0,
      dynamicSparseSynapses: 0,
      totalNeuralParameters: 120,
      countingRule: "unique learned modules plus substrate synapses"
    };
    const writeMetadata = async (
      parameterAccounting: Record<string, unknown>,
      additional: Record<string, unknown> = {}
    ): Promise<void> => {
      await writeFile(join(engine, "brain.json"), canonicalJson({
        brain_id: brain.id,
        substrate: { persistence: pointer },
        runtime_card: { parameterAccounting },
        ...additional
      }));
    };
    await writeMetadata(accounting);
    await expect(repository.persistedSubstrateOverview(brain.id)).resolves.toMatchObject({
      parameterAccounting: accounting
    });
    for (const stale of [
      { ...accounting, foundationEffectiveParameters: 0 },
      { ...accounting, sequenceDynamicSparseSynapses: 0 },
      { ...accounting, fixedSequenceStatisticalCapacity: 0 },
      { ...accounting, totalNeuralParameters: 121 },
      { ...accounting, substrateDynamicSparseSynapses: 1 }
    ]) {
      await writeMetadata(stale);
      await expect(repository.persistedSubstrateOverview(brain.id))
        .rejects.toThrow(/parameter accounting is invalid/i);
    }
    await writeMetadata(accounting, { neural_sequence_memory: { synapses: 0 } });
    await expect(repository.persistedSubstrateOverview(brain.id))
      .rejects.toThrow(/legacy sequence memory/i);
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

  it("persists Active Mode independently without treating it as a Ponder capability", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Dormant instance",
      idleCognition: false
    });
    expect(brain.config.idleCognition).toBe(false);
    const reloaded = await repository.get(brain.id);
    expect(reloaded.config.idleCognition).toBe(false);
  });

  it("switches the single persisted Active Mode lease visibly and atomically", async () => {
    const first = await repository.create({
      ...DEFAULT_CONFIG,
      name: "First active mind"
    });
    const second = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Second dormant mind",
      idleCognition: false
    });

    const switched = await repository.setActiveMode(second.id, true);
    expect(switched).toMatchObject({
      schemaVersion: 1,
      enabled: true,
      activeBrainId: second.id,
      brain: { id: second.id, config: { idleCognition: true } },
      deactivated: [{ id: first.id, name: first.name }]
    });
    await expect(repository.get(first.id)).resolves.toMatchObject({
      config: { idleCognition: false }
    });
    expect((await repository.list()).map(({ id, activeMode }) => ({ id, activeMode })))
      .toEqual(expect.arrayContaining([
        { id: first.id, activeMode: false },
        { id: second.id, activeMode: true }
      ]));
  });

  it("collapses legacy multi-active files to the newest single owner", async () => {
    const older = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Legacy older",
      idleCognition: false
    });
    const newer = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Legacy newer",
      idleCognition: false
    });
    older.config.idleCognition = true;
    older.updatedAt = "2026-09-11T00:00:00.000Z";
    await repository.save(older, false);
    newer.config.idleCognition = true;
    newer.updatedAt = "2026-09-12T00:00:00.000Z";
    await repository.save(newer, false);

    const repaired = await repository.reconcileActiveModeLease();
    expect(repaired).toMatchObject({
      activeBrainId: newer.id,
      deactivated: [{ id: older.id, name: older.name }]
    });
    await expect(repository.get(older.id)).resolves.toMatchObject({
      config: { idleCognition: false }
    });
    await expect(repository.get(newer.id)).resolves.toMatchObject({
      config: { idleCognition: true }
    });
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

  it("shares identical immutable origins by content hash without linking live state", async () => {
    const first = await repository.create({ ...DEFAULT_CONFIG, name: "First instance" });
    const second = await repository.create({ ...DEFAULT_CONFIG, name: "Second instance" });
    const payload = byteTensorSafetensors(32 * 1024);
    for (const brain of [first, second]) {
      const origin = join(repository.brainDirectory(brain.id), "engine", "origin");
      const live = join(repository.brainDirectory(brain.id), "engine", "core.safetensors");
      await mkdir(origin, { recursive: true });
      await writeFile(join(origin, "core.safetensors"), payload);
      await writeFile(live, payload);
    }

    const [firstReport, secondReport] = await Promise.all([
      repository.deduplicateImmutableOrigin(first.id),
      repository.deduplicateImmutableOrigin(second.id)
    ]);
    const firstOrigin = await stat(
      join(repository.brainDirectory(first.id), "engine", "origin", "core.safetensors")
    );
    const secondOrigin = await stat(
      join(repository.brainDirectory(second.id), "engine", "origin", "core.safetensors")
    );
    const firstLive = await stat(
      join(repository.brainDirectory(first.id), "engine", "core.safetensors")
    );

    expect(firstReport).toMatchObject({ files: 1, contentHashes: 1 });
    expect(secondReport).toMatchObject({ files: 1, contentHashes: 1 });
    expect(firstOrigin.nlink).toBeGreaterThan(1);
    expect(secondOrigin.nlink).toBeGreaterThan(1);
    expect(firstOrigin.ino).toBe(secondOrigin.ino);
    expect(firstLive.ino).not.toBe(firstOrigin.ino);
    expect((await readdir(join(repository.root, ".blobs"))).filter((name) => /^[a-f0-9]{64}$/.test(name))).toEqual([
      digest(payload)
    ]);
  });

  it("requires three exact confirmations, deletes only one instance, and safely collects blobs", async () => {
    const first = await repository.create({ ...DEFAULT_CONFIG, name: "Delete Me Exactly" });
    const second = await repository.create({ ...DEFAULT_CONFIG, name: "Keep Me" });
    const payload = byteTensorSafetensors(16 * 1024);
    for (const brain of [first, second]) {
      const origin = join(repository.brainDirectory(brain.id), "engine", "origin");
      await mkdir(origin, { recursive: true });
      await writeFile(join(origin, "core.safetensors"), payload);
      await repository.deduplicateImmutableOrigin(brain.id);
    }

    await expect(
      repository.permanentlyDeleteInstance({
        brainId: first.id,
        acknowledgedIrreversible: true,
        typedName: "delete me exactly",
        finalConfirmation: "PERMANENTLY DELETE Delete Me Exactly"
      })
    ).rejects.toThrow(/name does not match exactly/i);
    await expect(
      repository.permanentlyDeleteInstance({
        brainId: first.id,
        acknowledgedIrreversible: true,
        typedName: first.name,
        finalConfirmation: "DELETE Delete Me Exactly"
      })
    ).rejects.toThrow(/final confirmation/i);
    await expect(repository.get(first.id)).resolves.toMatchObject({ id: first.id });

    const firstDeletion = await repository.permanentlyDeleteInstance({
      brainId: first.id,
      acknowledgedIrreversible: true,
      typedName: first.name,
      finalConfirmation: `PERMANENTLY DELETE ${first.name}`
    });
    expect(firstDeletion).toMatchObject({
      deleted: true,
      brainId: first.id,
      recoverable: false,
      removedSharedBlobs: 0,
      reclaimedBytes: 0
    });
    await expect(repository.get(first.id)).rejects.toThrow(/not found/i);
    await expect(repository.get(second.id)).resolves.toMatchObject({ id: second.id });
    expect((await readdir(join(repository.root, ".trash"))).some((name) => name.startsWith(first.id))).toBe(false);
    await expect(repository.getBlob(digest(payload))).resolves.toEqual(payload);

    const secondDeletion = await repository.permanentlyDeleteInstance({
      brainId: second.id,
      acknowledgedIrreversible: true,
      typedName: second.name,
      finalConfirmation: `PERMANENTLY DELETE ${second.name}`
    });
    expect(secondDeletion).toMatchObject({
      deleted: true,
      brainId: second.id,
      recoverable: false,
      removedSharedBlobs: 1,
      reclaimedBytes: payload.byteLength
    });
    await expect(repository.getBlob(digest(payload))).rejects.toThrow();
  });

  it("exports a native brain without phantom conversation rows or transient chat text", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Portable native mind" });
    await materializeNativeEngineFixture(repository, brain);
    const enginePath = join(repository.brainDirectory(brain.id), "engine", "brain.json");
    const native = JSON.parse(await readFile(enginePath, "utf8")) as Record<string, unknown>;
    const privateText = "private turn: violet lantern at midnight";
    native.conversation = {
      format: "omni-neural-conversation-ledger",
      formatVersion: 1,
      totalEntries: 3,
      messageCount: 2,
      actionCount: 0,
      traceCount: 1,
      attentionEpoch: 2,
      headSequence: 3,
      headSha256: "a".repeat(64)
    };
    native.completed_chat_turns = [{ content: privateText }];
    native.completed_chat_slow_learning = ["b".repeat(64)];
    native.pending_chat_slow_learning = [{ content: privateText }];
    native.recent_token_context = [65, 66, 67];
    native.current_context = {
      tokenCount: 3,
      tokenHash: "c".repeat(64),
      sensorySlots: 1,
      raw: privateText
    };
    native.fresh_attention_boundary = { note: privateText };
    native.attention_overlay = { note: privateText };
    native.workspace_items = [{ note: privateText }];
    native.messages = [];
    native.traces = [];
    native.runtime_card = {
      ...(native.runtime_card as Record<string, unknown>),
      workspace: { content: privateText },
      fresh_attention: { content: privateText }
    };
    await writeFile(enginePath, JSON.stringify(native));
    const original = await readFile(enginePath);
    const bundle = join(temporaryRoot, "portable-native-conversation.omni");
    await repository.exportBundle(brain.id, bundle, "current");
    expect(await readFile(enginePath)).toEqual(original);
    const bundleBytes = new Uint8Array(await readFile(bundle));
    const archive = unzipSync(bundleBytes);
    expect(archive["conversation/neural-ledger.sqlite3"]).toBeUndefined();
    const manifest = JSON.parse(strFromU8(archive["manifest.json"]!)) as Record<string, unknown>;
    expect(manifest.conversationProjection).toEqual({
      historyIncluded: false,
      omittedLedgerRows: 3,
      omittedPendingReplayJobs: 1
    });
    const modelCard = strFromU8(archive["model-card.md"]!);
    expect(modelCard).toContain("chat history is omitted (3 ledger rows)");
    expect(modelCard).toContain("1 pending background replay job(s) are omitted and will not resume");
    for (const path of ["state/engine.json", "origin/state/engine.json"]) {
      const exported = JSON.parse(strFromU8(archive[path]!)) as Record<string, unknown>;
      expect(exported.conversation).toEqual({
        format: "omni-neural-conversation-ledger",
        formatVersion: 1,
        totalEntries: 0,
        messageCount: 0,
        actionCount: 0,
        traceCount: 0,
        attentionEpoch: 0,
        headSequence: 0,
        headSha256: "0".repeat(64)
      });
      expect(exported.completed_chat_turns).toEqual([]);
      expect(exported.completed_chat_slow_learning).toEqual([]);
      expect(exported.pending_chat_slow_learning).toEqual([]);
      expect(exported.recent_token_context).toEqual([]);
      expect(exported.current_context).toMatchObject({
        tokenCount: 0,
        tokenHash: "",
        sensorySlots: 0
      });
      expect(exported.fresh_attention_boundary).toBeNull();
      expect(exported.attention_overlay).toBeNull();
      expect(exported.workspace_items).toEqual([]);
      expect(exported.messages).toBeUndefined();
      expect(exported.traces).toBeUndefined();
    }
    expect(Object.values(archive).some((entry) =>
      Buffer.from(entry).includes(Buffer.from(privateText))
    )).toBe(false);
    const portableBrain = JSON.parse(strFromU8(archive["state/brain.json"]!)) as Record<string, unknown>;
    expect(portableBrain.conversation).toBeUndefined();
    const imported = await repository.importBundle(bundle);
    const importedEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8")
    ) as Record<string, unknown>;
    expect(importedEngine.brain_id).toBe(imported.id);
    expect((importedEngine.conversation as Record<string, unknown>).headSequence).toBe(0);
    expect(imported.originChecksum).toBe(portableBrain.originChecksum);

    for (const path of ["state/engine.json", "origin/state/engine.json"]) {
      const tampered = unzipSync(bundleBytes);
      const changed = JSON.parse(strFromU8(tampered[path]!)) as Record<string, unknown>;
      changed.conversation = {
        ...(changed.conversation as Record<string, unknown>),
        totalEntries: 1,
        messageCount: 1,
        headSequence: 1,
        headSha256: "d".repeat(64)
      };
      tampered[path] = Buffer.from(JSON.stringify(changed));
      refreshArchiveIntegrity(tampered);
      await expect(repository.importBundleBuffer(Buffer.from(zipSync(tampered))))
        .rejects.toThrow(/omits their neural ledger/i);
    }
  });

  it("preserves an imported checkpoint workspace above the former product cap", async () => {
    const recordedSlots = 8_192;
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large imported workspace",
      workingMemorySlots: recordedSlots,
      systemRamMode: "manual",
      systemRamSharePercent: 47,
      storagePoolMode: "manual",
      storagePoolBytes: 18 * 1024 ** 3
    });
    const bundle = join(temporaryRoot, "large-workspace.omni");

    await materializeNativeEngineFixture(repository, source);
    await repository.exportBundle(source.id, bundle, "current");
    const imported = await repository.importBundle(bundle);

    expect(source.config.workingMemorySlots).toBe(recordedSlots);
    expect(imported.config.workingMemorySlots).toBe(recordedSlots);
    expect(imported.config).toMatchObject({
      systemRamMode: "manual",
      systemRamSharePercent: 47,
      storagePoolMode: "manual",
      storagePoolBytes: 18 * 1024 ** 3
    });
    expect(imported.config.idleCognition).toBe(false);
    expect((await repository.get(imported.id)).config.workingMemorySlots).toBe(
      recordedSlots
    );
  });

  it("cancels export without replacing an existing destination or leaving private staging", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Cancelled export source"
    });
    const destination = join(temporaryRoot, "cancelled-export.omni");
    await writeFile(destination, "existing destination remains");
    const controller = new AbortController();
    const hooks: BrainStorageOperationHooks = {
      signal: controller.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        if (
          update.phase === "writing" &&
          Number(update.logicalBytesCompleted ?? 0) > 0
        ) {
          controller.abort();
        }
      })
    };

    await expect(
      repository.exportBundle(source.id, destination, "current", hooks)
    ).rejects.toMatchObject({ name: "AbortError" });
    await expect(readFile(destination, "utf8")).resolves.toBe(
      "existing destination remains"
    );
    expect(
      (await readdir(temporaryRoot)).some((name) => name.startsWith(".omni-export-"))
    ).toBe(false);
  });

  it("cancels import during extraction and removes only its private staging", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Cancelled import source"
    });
    const bundle = join(temporaryRoot, "cancelled-import.omni");
    await repository.exportBundle(source.id, bundle, "current");
    const before = (await repository.list()).map((entry) => entry.id).sort();
    const controller = new AbortController();
    const hooks: BrainStorageOperationHooks = {
      signal: controller.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        if (
          update.phase === "extracting" &&
          Number(update.logicalBytesCompleted ?? 0) > 0
        ) {
          controller.abort();
        }
      })
    };

    await expect(
      repository.importBundle(bundle, {}, hooks)
    ).rejects.toMatchObject({ name: "AbortError" });
    expect((await repository.list()).map((entry) => entry.id).sort()).toEqual(before);
    expect(
      (await readdir(join(repository.root, ".imports"))).some((name) =>
        name.startsWith(".omni-import-")
      )
    ).toBe(false);
  });

  it("atomically reserves unique identities for concurrent imports of one bundle", async () => {
    const source = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Concurrent import source"
    });
    const bundle = join(temporaryRoot, "concurrent-import.omni");
    await materializeNativeEngineFixture(repository, source);
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
          `concurrent-${index}.omni`)
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
          await readFile(join(destination.brainDirectory(brain.id), "origin.json"), "utf8")
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
    const mutableState = await writeMutableStateFixture(engine);
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
        substrate: { persistence: substratePointer },
        mutable_state: mutableState.pointer
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
    const forkArtifactBytes = Buffer.from("copy-on-write artifact");
    const forkArtifactSha256 = digest(forkArtifactBytes);
    const forkArtifactName = `${forkArtifactSha256}.bin`;
    const forkArtifactDirectory = join(engine, "artifacts");
    await mkdir(forkArtifactDirectory, { recursive: true });
    await writeFile(join(forkArtifactDirectory, forkArtifactName), forkArtifactBytes);
    await ArtifactIndexStore.replace(forkArtifactDirectory, brain.id, [{
      id: digest("fork-artifact"),
      modality: "image",
      mimeType: "image/png",
      sha256: forkArtifactSha256,
      bytes: forkArtifactBytes.length,
      relativePath: `artifacts/${forkArtifactName}`,
      createdAt: "2026-09-07T00:00:00.000Z",
      initialization: "local-only clone metadata"
    }]);

    const fork = await repository.fork(brain.id, "Ada branch");
    const forkEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(fork.id), "engine", "brain.json"), "utf8")
    ) as { brain_id: string; name: string };

    expect(fork.lineage.parentId).toBe(brain.id);
    expect(fork.config.idleCognition).toBe(false);
    await expect(
      readFile(
        join(repository.brainDirectory(fork.id), "engine", "substrate", "manifest.json"),
        "utf8"
      )
    ).resolves.toBe(canonicalJson(substratePointer));
    expect(fork.lineage.rootId).toBe(brain.id);
    expect(forkEngine.brain_id).toBe(fork.id);
    expect(forkEngine.name).toBe("Ada branch");
    const forkArtifactStore = await ArtifactIndexStore.open(
      join(repository.brainDirectory(fork.id), "engine", "artifacts"),
      fork.id
    );
    expect(forkArtifactStore.snapshot().artifacts).toMatchObject([{
      sha256: forkArtifactSha256,
      initialization: "local-only clone metadata"
    }]);
    forkArtifactStore.close();
    await expect(
      readFile(
        join(repository.brainDirectory(fork.id), "engine", "artifacts", forkArtifactName)
      )
    ).resolves.toEqual(forkArtifactBytes);
    await Promise.all([
      assertDeclaredNeuralGenerationsLoadable(join(repository.brainDirectory(fork.id), "engine")),
      assertDeclaredNeuralGenerationsLoadable(
        join(repository.brainDirectory(fork.id), "engine", "origin")
      )
    ]);
    await expect(
      readFile(join(repository.brainDirectory(fork.id), "engine", "origin", "core.safetensors"))
    ).resolves.toBeInstanceOf(Buffer);

    const visibleBeforeFailure = (await repository.list()).map((item) => item.id).sort();
    const cloneFailure = vi
      .spyOn(
        repository as unknown as {
          adoptFileAsBlob(path: string): Promise<unknown>;
        },
        "adoptFileAsBlob"
      )
      .mockRejectedValueOnce(new Error("simulated clone staging failure"));
    try {
      await expect(repository.fork(brain.id, "Broken fork")).rejects.toThrow(
        /simulated clone staging failure/
      );
    } finally {
      cloneFailure.mockRestore();
    }
    expect((await repository.list()).map((item) => item.id).sort()).toEqual(visibleBeforeFailure);
  });

  it("publishes a fork engine only after its substrate generation is complete", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Atomic fork source" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    const mutableState = await writeMutableStateFixture(engine);
    await Promise.all([
      writeFile(
        join(engine, "brain.json"),
        JSON.stringify({
          schema_version: 1,
          format: "omni-cortex-engine",
          release_format: "stable-1.0",
          brain_id: brain.id,
          name: brain.name,
          config: {},
          expert_count: 0,
          substrate: { persistence: substratePointer },
          mutable_state: mutableState.pointer
        })
      ),
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);

    let releaseSubstrateCopy = (): void => undefined;
    let substrateCopyStarted = (): void => undefined;
    const substrateCopyGate = new Promise<void>((resolve) => {
      releaseSubstrateCopy = resolve;
    });
    const substrateCopyReached = new Promise<void>((resolve) => {
      substrateCopyStarted = resolve;
    });
    const adoptFileAsBlob = (
      repository as unknown as {
        adoptFileAsBlob(path: string): Promise<unknown>;
      }
    ).adoptFileAsBlob.bind(repository);
    let heldGenerationManifest = false;
    const stagedCopy = vi.spyOn(
      repository as unknown as {
        adoptFileAsBlob(path: string): Promise<unknown>;
      },
      "adoptFileAsBlob"
    ).mockImplementation(async (path) => {
      const normalized = path.replaceAll("\\", "/");
      if (
        !heldGenerationManifest &&
        normalized.includes("/substrate/generations/") &&
        normalized.endsWith("/manifest.json")
      ) {
        heldGenerationManifest = true;
        substrateCopyStarted();
        await substrateCopyGate;
      }
      return adoptFileAsBlob(path);
    });

    let pendingFork: Promise<Awaited<ReturnType<BrainRepository["fork"]>>> | undefined;
    try {
      pendingFork = repository.fork(brain.id, "Atomic fork target");
      await substrateCopyReached;
      const targetIds = (await readdir(join(temporaryRoot, "brains"), { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() && entry.name !== brain.id && !entry.name.startsWith("."))
        .map((entry) => entry.name);
      expect(targetIds).toHaveLength(1);
      const targetDirectory = repository.brainDirectory(targetIds[0]!);
      await expect(stat(join(targetDirectory, "engine"))).rejects.toMatchObject({ code: "ENOENT" });
      const stagingDirectories = (await readdir(targetDirectory, { withFileTypes: true }))
        .filter((entry) => entry.isDirectory() && /^\.engine-.*\.clone-next$/.test(entry.name))
        .map((entry) => entry.name);
      expect(stagingDirectories).toHaveLength(1);
      await expect(
        stat(join(targetDirectory, stagingDirectories[0]!, "brain.json"))
      ).rejects.toMatchObject({ code: "ENOENT" });

      releaseSubstrateCopy();
      const fork = await pendingFork;
      expect(fork.id).toBe(targetIds[0]);
      await assertDeclaredNeuralGenerationsLoadable(
        join(repository.brainDirectory(fork.id), "engine")
      );
      expect(
        (await readdir(repository.brainDirectory(fork.id))).some((entry) =>
          /^\.engine-.*\.clone-next$/.test(entry)
        )
      ).toBe(false);
    } finally {
      releaseSubstrateCopy();
      stagedCopy.mockRestore();
      await pendingFork?.catch(() => undefined);
    }
  });

  it("cancels immediately before promotion, removes only private staging, and leaves the original intact", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Cancel-safe duplicate source"
    });
    const sourceDirectory = repository.brainDirectory(brain.id);
    const engine = join(sourceDirectory, "engine");
    await mkdir(engine, { recursive: true });
    await Promise.all([
      writeFile(
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
      ),
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
    const before = {
      document: digest(await readFile(join(sourceDirectory, "brain.json"))),
      engine: digest(await readFile(join(engine, "brain.json"))),
      core: digest(await readFile(join(engine, "core.safetensors"))),
      plasticity: digest(await readFile(join(engine, "plasticity.safetensors")))
    };
    const controller = new AbortController();
    const phases: string[] = [];
    const hooks: BrainStorageOperationHooks = {
      signal: controller.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        phases.push(update.phase ?? "");
        if (update.phase === "promoting") controller.abort();
      })
    };

    await expect(
      repository.duplicate(brain.id, "Cancelled target", hooks)
    ).rejects.toMatchObject({ name: "AbortError" });

    expect(phases).toContain("promoting");
    expect((await repository.list()).map((entry) => entry.id)).toEqual([brain.id]);
    expect({
      document: digest(await readFile(join(sourceDirectory, "brain.json"))),
      engine: digest(await readFile(join(engine, "brain.json"))),
      core: digest(await readFile(join(engine, "core.safetensors"))),
      plasticity: digest(await readFile(join(engine, "plasticity.safetensors")))
    }).toEqual(before);
    const rootEntries = await readdir(join(temporaryRoot, "brains"), {
      withFileTypes: true
    });
    expect(
      rootEntries.some((entry) => /\.clone-next$/.test(entry.name))
    ).toBe(false);
    for (const entry of rootEntries.filter((value) => value.isDirectory())) {
      const children = await readdir(
        join(temporaryRoot, "brains", entry.name)
      ).catch(() => []);
      expect(children.some((name) => /\.clone-next$/.test(name))).toBe(false);
    }
  });

  it("duplicates every brain as an independent copy-on-write identity", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Original"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const mutableState = await writeMutableStateFixture(engine);
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        name: brain.name,
        config: {},
        mutable_state: mutableState.pointer
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
    ]);
    await writePackedTernaryFixture(join(engine, "packed-ternary"));

    const cloneProgress: Array<Record<string, unknown>> = [];
    const cloneController = new AbortController();
    const duplicate = await repository.duplicate(brain.id, undefined, {
      signal: cloneController.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        cloneProgress.push({ ...update });
      })
    });
    const metadata = JSON.parse(
      await readFile(join(repository.brainDirectory(duplicate.id), "engine", "brain.json"), "utf8")
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
    expect(duplicate.journal?.at(-1)?.summary).toMatch(/Duplicated.*copy-on-write/i);
    expect(cloneProgress.at(-1)).toMatchObject({
      phase: "promoting",
      filesCompleted: expect.any(Number),
      filesTotal: expect.any(Number),
      physicalBytesAdded: expect.any(Number),
      sharedBytes: expect.any(Number)
    });
    expect(Number(cloneProgress.at(-1)?.filesCompleted)).toBe(
      Number(cloneProgress.at(-1)?.filesTotal)
    );
    expect(Number(cloneProgress.at(-1)?.sharedBytes)).toBeGreaterThan(0);
    const sourceCorePath = join(engine, "core.safetensors");
    const duplicateCorePath = join(
      repository.brainDirectory(duplicate.id),
      "engine",
      "core.safetensors"
    );
    const coreHash = digest(await readFile(sourceCorePath));
    const sourceStateBlob = join(
      engine,
      "state",
      "blobs",
      `${coreHash}.safetensors`
    );
    const duplicateStateBlob = join(
      repository.brainDirectory(duplicate.id),
      "engine",
      "state",
      "blobs",
      `${coreHash}.safetensors`
    );
    const sharedBlobPath = join(temporaryRoot, "brains", ".blobs", coreHash);
    const [
      sourceCore,
      duplicateCore,
      sourceSharedState,
      duplicateSharedState,
      sharedBlob
    ] = await Promise.all([
      stat(sourceCorePath),
      stat(duplicateCorePath),
      stat(sourceStateBlob),
      stat(duplicateStateBlob),
      stat(sharedBlobPath)
    ]);
    if (process.platform !== "win32") {
      // Live compatibility tensors are independently mutable paths. The
      // checksummed generation blobs remain hard-link deduplicated.
      expect(sourceCore.ino).not.toBe(duplicateCore.ino);
      expect(sourceCore.ino).not.toBe(sharedBlob.ino);
      expect(duplicateCore.ino).not.toBe(sharedBlob.ino);
      expect(sourceSharedState.ino).toBe(sharedBlob.ino);
      expect(duplicateSharedState.ino).toBe(sharedBlob.ino);
      expect(sharedBlob.nlink).toBeGreaterThanOrEqual(3);
    } else {
      expect(sourceCore.size).toBe(sharedBlob.size);
      expect(duplicateCore.size).toBe(sharedBlob.size);
    }

    const sourceMutablePaths = {
      core: sourceCorePath,
      pointer: join(engine, "state", "manifest.json"),
      replay: join(engine, "state", "replay.sqlite3")
    };
    const duplicateMutablePaths = {
      core: duplicateCorePath,
      pointer: join(
        repository.brainDirectory(duplicate.id),
        "engine",
        "state",
        "manifest.json"
      ),
      replay: join(
        repository.brainDirectory(duplicate.id),
        "engine",
        "state",
        "replay.sqlite3"
      )
    };
    const sourceMutableHashes = Object.fromEntries(
      await Promise.all(
        Object.entries(sourceMutablePaths).map(async ([name, path]) => [
          name,
          digest(await readFile(path))
        ])
      )
    );
    if (process.platform !== "win32") {
      for (const name of Object.keys(sourceMutablePaths) as Array<
        keyof typeof sourceMutablePaths
      >) {
        expect((await stat(duplicateMutablePaths[name])).ino).not.toBe(
          (await stat(sourceMutablePaths[name])).ino
        );
      }
    }
    await Promise.all([
      writeFile(duplicateMutablePaths.core, Buffer.from("fork-only core mutation")),
      writeFile(duplicateMutablePaths.pointer, Buffer.from("fork-only pointer mutation")),
      writeFile(duplicateMutablePaths.replay, Buffer.from("fork-only replay mutation"))
    ]);
    for (const [name, path] of Object.entries(sourceMutablePaths)) {
      expect(digest(await readFile(path))).toBe(sourceMutableHashes[name]);
    }
    expect(digest(await readFile(sharedBlobPath))).toBe(coreHash);
    const summaries = await repository.list();
    expect(summaries.find((summary) => summary.id === brain.id)).toMatchObject({
      instanceKind: "original",
      rootId: brain.id,
      originInstanceCount: 2
    });
    expect(summaries.find((summary) => summary.id === duplicate.id)).toMatchObject({
      instanceKind: "duplicate",
      rootId: brain.id,
      parentId: brain.id,
      originInstanceCount: 2
    });
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      id: brain.id,
      name: "Original"
    });
    await Promise.all([
      expect(
        readFile(
          join(repository.brainDirectory(duplicate.id), "engine", "packed-ternary", "manifest.json")
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
      ).resolves.toBeInstanceOf(Buffer),
      expect(
        readFile(
          join(
            repository.brainDirectory(duplicate.id),
            "engine",
            "origin",
            "state",
            "replay.sqlite3"
          )
        )
      ).resolves.toEqual(mutableState.replay)
    ]);
  });

  it("rejects a legacy starter from get, duplicate, and Library", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Legacy starter" });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    await writeFile(join(engine, "brain.json"), JSON.stringify({
      brain_id: brain.id,
      config: { origin_kind: "starter", foundation_model_id: "external-base" },
      runtime_card: {
        origin_kind: "starter",
        pretrained: true,
        baseFrozen: true,
        foundationModelId: "external-base",
        pretrained_text_cortex: { id: "external-base" }
      }
    }));
    await expect(repository.get(brain.id)).rejects.toThrow(/non-native origin/i);
    await expect(repository.duplicate(brain.id)).rejects.toThrow(/non-native origin/i);
    await expect(repository.list()).resolves.toEqual([]);
  });

  it("enumerates only app-managed beta directories and deletes them only after confirmation", async () => {
    const stable = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Stable mind"
    });
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
    await expect(repository.deleteManagedBetaBrains([betaId], false)).rejects.toThrow(
      /explicit confirmation/i
    );
    await expect(readFile(join(betaDirectory, "brain.json"), "utf8")).resolves.toContain(
      "Old beta mind"
    );

    await expect(repository.deleteManagedBetaBrains([betaId], true)).resolves.toEqual([betaId]);
    await expect(stat(betaDirectory)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(repository.get(stable.id)).resolves.toMatchObject({
      id: stable.id
    });
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

    const stable = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Stable shell"
    });
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
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Snapshot mind"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    const mutableState = await writeMutableStateFixture(engine);
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        marker: "before",
        substrate: { persistence: substratePointer },
        mutable_state: mutableState.pointer
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors()),
      writeFile(join(engine, "conversation.sqlite3"), "neural conversation before")
    ]);
    await writePackedTernaryFixture(join(engine, "packed-ternary"));
    const beforeArtifact = Buffer.from("recovery artifact before");
    const beforeArtifactHash = digest(beforeArtifact);
    const beforeArtifactName = `${beforeArtifactHash}.bin`;
    const artifactDirectory = join(engine, "artifacts");
    await mkdir(artifactDirectory, { recursive: true });
    await writeFile(join(artifactDirectory, beforeArtifactName), beforeArtifact);
    await ArtifactIndexStore.replace(artifactDirectory, brain.id, [{
      id: digest("recovery-artifact-before"),
      modality: "image",
      mimeType: "image/png",
      sha256: beforeArtifactHash,
      bytes: beforeArtifact.length,
      relativePath: `artifacts/${beforeArtifactName}`,
      createdAt: "2026-09-12T00:00:00.000Z"
    }]);
    const beforeDocument = await repository.get(brain.id);
    beforeDocument.messages.push({
      id: "recovery-message-before",
      role: "human",
      content: "history captured before recovery point",
      createdAt: "2026-09-12T00:00:00.000Z"
    });
    beforeDocument.journal?.push({
      id: "recovery-journal-before",
      kind: "system",
      summary: "Activity before recovery point.",
      createdAt: "2026-09-12T00:00:00.000Z"
    });
    await repository.save(beforeDocument);
    const packedBefore = await readFile(join(engine, "packed-ternary", "manifest.json"));
    const snapshotProgress = vi.fn(async () => undefined);
    const snapshot = await repository.snapshot(
      brain.id,
      "before mutation",
      {
        signal: new AbortController().signal,
        checkpoint: snapshotProgress,
        checkDisk: vi.fn(async () => safeDiskTelemetry())
      }
    );
    expect(snapshot.metrics.concepts).toBe(1);
    expect(snapshot.metrics.estimatedBytes).toBe(snapshot.storage?.logicalBytes);
    expect(snapshot.storage?.logicalBytes).toBeGreaterThan(0);
    expect(snapshot.storage?.sharedBytes).toBeGreaterThan(0);
    expect(snapshotProgress).toHaveBeenCalledWith(
      expect.objectContaining({
        phase: "verifying",
        filesCompleted: snapshot.storage?.files,
        logicalBytesCompleted: snapshot.storage?.logicalBytes,
        sharedBytes: snapshot.storage?.sharedBytes
      })
    );
    const snapshotConversation = join(
      repository.brainDirectory(brain.id),
      "snapshots",
      snapshot.id,
      "conversation",
      "ledger.sqlite3"
    );
    expect(digest(await readFile(snapshotConversation))).toBe(
      snapshot.checkpointComponentSha256?.at(-2)
    );

    const mutated = await repository.get(brain.id);
    mutated.name = "Mutated";
    mutated.config.name = "Mutated";
    mutated.messages.push({
      id: "recovery-message-after",
      role: "human",
      content: "history added after recovery point",
      createdAt: "2026-09-12T01:00:00.000Z"
    });
    mutated.journal?.push({
      id: "recovery-journal-after",
      kind: "system",
      summary: "Activity after recovery point.",
      createdAt: "2026-09-12T01:00:00.000Z"
    });
    await repository.save(mutated);
    expect(digest(await readFile(snapshotConversation))).toBe(
      snapshot.checkpointComponentSha256?.at(-2)
    );
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
    await writeFile(join(engine, "conversation.sqlite3"), "neural conversation after");
    const afterArtifact = Buffer.from("recovery artifact after");
    const afterArtifactHash = digest(afterArtifact);
    const afterArtifactName = `${afterArtifactHash}.bin`;
    await writeFile(join(artifactDirectory, afterArtifactName), afterArtifact);
    await ArtifactIndexStore.replace(artifactDirectory, brain.id, [{
      id: digest("recovery-artifact-after"),
      modality: "image",
      mimeType: "image/png",
      sha256: afterArtifactHash,
      bytes: afterArtifact.length,
      relativePath: `artifacts/${afterArtifactName}`,
      createdAt: "2026-09-12T01:00:00.000Z"
    }]);
    await rm(join(engine, "substrate"), { recursive: true, force: true });
    await rm(join(engine, "state"), { recursive: true, force: true });

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
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      name: "Mutated"
    });
    await expect(readFile(join(engine, "brain.json"), "utf8")).resolves.toContain(
      '"marker":"after"'
    );

    await expect(
      repository.restoreSnapshot(
        brain.id,
        snapshot.id,
        undefined,
        async () => {
          throw new Error("required worker reload failed");
        }
      )
    ).rejects.toThrow(/required worker reload failed/i);
    await expect(repository.get(brain.id)).resolves.toMatchObject({
      name: "Mutated"
    });
    await expect(readFile(join(engine, "brain.json"), "utf8")).resolves.toContain(
      '"marker":"after"'
    );
    const rolledBackConversation = await repository.conversationPage(brain.id);
    expect(
      rolledBackConversation.entries.some(
        ({ message }) => message?.id === "recovery-message-after"
      )
    ).toBe(true);

    const restored = await repository.restoreSnapshot(brain.id, snapshot.id);
    const engineState = JSON.parse(await readFile(join(engine, "brain.json"), "utf8")) as {
      marker: string;
    };
    expect(restored.name).toBe("Snapshot mind");
    const restoredConversation = await repository.conversationPage(brain.id);
    expect(
      restoredConversation.entries.some(
        ({ message }) => message?.id === "recovery-message-before"
      )
    ).toBe(true);
    expect(
      restoredConversation.entries.some(
        ({ message }) => message?.id === "recovery-message-after"
      )
    ).toBe(false);
    expect(restored.journal?.some(({ id }) => id === "recovery-journal-before")).toBe(true);
    expect(restored.journal?.some(({ id }) => id === "recovery-journal-after")).toBe(false);
    expect(engineState.marker).toBe("before");
    expect(snapshot.engineChecksum).toMatch(/^[a-f0-9]{64}$/);
    await expect(readFile(join(engine, "packed-ternary", "manifest.json"))).resolves.toEqual(
      packedBefore
    );
    await expect(readFile(join(engine, "substrate", "manifest.json"), "utf8")).resolves.toBe(
      canonicalJson(substratePointer)
    );
    await expect(readFile(join(engine, "state", "manifest.json"), "utf8")).resolves.toBe(
      canonicalJson(mutableState.pointer)
    );
    await expect(readFile(join(engine, "state", "replay.sqlite3"))).resolves.toEqual(
      mutableState.replay
    );
    await expect(readFile(join(engine, "conversation.sqlite3"), "utf8")).resolves.toBe(
      "neural conversation before"
    );
    await expect(readFile(join(engine, "artifacts", beforeArtifactName))).resolves.toEqual(
      beforeArtifact
    );
    await expect(stat(join(engine, "artifacts", afterArtifactName))).rejects.toMatchObject({
      code: "ENOENT"
    });
  });

  it("cancels recovery-point materialization without publishing an orphan", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Cancelled recovery point"
    });
    const snapshots = join(repository.brainDirectory(brain.id), "snapshots");
    const before = (await readdir(snapshots)).sort();
    const controller = new AbortController();
    const hooks: BrainStorageOperationHooks = {
      signal: controller.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        if (update.label === "Copying exact conversation and activity ledgers") {
          controller.abort();
        }
      })
    };

    await expect(
      repository.snapshot(brain.id, "cancelled", hooks)
    ).rejects.toMatchObject({ name: "AbortError" });
    expect((await readdir(snapshots)).sort()).toEqual(before);
    await expect(repository.listSnapshots(brain.id)).resolves.toEqual([]);
  });

  it.each(["current", "origin"] as const)(
    "includes %s generation-committed replay rows held only in WAL",
    async (mode) => {
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: `${mode} WAL export`
      });
      const fixture = await materializeMutableReplayExportFixture(repository, brain, mode);
      const writer = fixture.walWriter;
      if (!writer) throw new Error("WAL replay fixture did not keep its writer open.");
      try {
        const sourcePath = mode === "current"
          ? fixture.currentReplayPath
          : fixture.originReplayPath;
        expect((await stat(`${sourcePath}-wal`)).size).toBeGreaterThan(0);
        expect(replayRowCount(sourcePath)).toBe(3);

        const archivePath = join(temporaryRoot, `${mode}-wal.omni`);
        await repository.exportBundle(brain.id, archivePath, mode);
        const entries = unzipSync(new Uint8Array(await readFile(archivePath)));
        for (const scope of mode === "origin" ? ["current", "origin"] : ["current"]) {
          const replayPath = join(temporaryRoot, `${mode}-${scope}-archived.sqlite3`);
          await writeFile(replayPath, entries[`mutable/${scope}/replay.sqlite3`]!);
          expect(replayRowCount(replayPath)).toBe(3);
        }
        const imported = await repository.importBundle(archivePath);
        expect(replayRowCount(join(
          repository.brainDirectory(imported.id), "engine", "state", "replay.sqlite3"
        ))).toBe(3);
      } finally {
        writer.close();
      }
    },
    20_000
  );

  it("keeps a durable pending replay row that exists only in WAL", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Pending WAL export"
    });
    const fixture = await materializeMutableReplayExportFixture(repository, brain);
    const writer = new DatabaseSync(fixture.currentReplayPath);
    try {
      writer.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;");
      const payload = Buffer.alloc(4);
      payload.writeFloatLE(4.25);
      const checksum = digest(Buffer.concat([
        Buffer.from("torch.float32\0[1]\0", "ascii"), payload
      ]));
      writer.prepare(
        "INSERT INTO replay (created_at,sha256,dtype,shape_json,payload) VALUES (?,?,?,?,?)"
      ).run(4, checksum, "torch.float32", "[1]", payload);
      expect((await stat(`${fixture.currentReplayPath}-wal`)).size).toBeGreaterThan(0);
      expect(replayRowCount(fixture.currentReplayPath)).toBe(4);

      const archivePath = join(temporaryRoot, "pending-wal.omni");
      await repository.exportBundle(brain.id, archivePath, "current");
      const entries = unzipSync(new Uint8Array(await readFile(archivePath)));
      const archivedReplayPath = join(temporaryRoot, "pending-archived.sqlite3");
      await writeFile(archivedReplayPath, entries["mutable/current/replay.sqlite3"]!);
      expect(replayRowCount(archivedReplayPath)).toBe(4);
      const engine = join(repository.brainDirectory(brain.id), "engine");
      const metadata = JSON.parse(await readFile(join(engine, "brain.json"), "utf8")) as {
        mutable_state: { generationManifest: string };
      };
      const generation = JSON.parse(await readFile(
        join(engine, "state", ...metadata.mutable_state.generationManifest.split("/")),
        "utf8"
      )) as { replay: unknown };
      expect(verifyPortableReplaySqlite(archivedReplayPath, generation.replay)).toEqual({
        committedExamples: 3,
        durableExamples: 4,
        pendingExamples: 1
      });
      const imported = await repository.importBundle(archivePath);
      expect(replayRowCount(join(
        repository.brainDirectory(imported.id), "engine", "state", "replay.sqlite3"
      ))).toBe(4);
    } finally {
      writer.close();
    }
  }, 20_000);

  it("round-trips a checksum-verified ZIP and omits private sources by default", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Portable mind"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const substratePointer = await writeSubstrateFixture(engine);
    await writeSubstrateFixture(join(engine, "origin"));
    const mutableState = await writeMutableStateFixture(engine);
    await writeMutableStateFixture(join(engine, "origin"));
    const distributedReceiptBody = {
      format: "omni-distributed-ground-up-promotion",
      formatVersion: 2,
      brainId: brain.id,
      runIdentitySha256: "a".repeat(64),
      parameterChecksum: "b".repeat(64),
      foundationModelId: "none",
      pretrainedFoundation: false,
      runtimeReady: true
    };
    const distributedReceipt = {
      ...distributedReceiptBody,
      contentSha256: digest(canonicalJson(distributedReceiptBody))
    };
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: {
        name: brain.name,
        origin_kind: "ground-up"
      },
      runtime_card: {
        origin_kind: "ground-up",
        pretrained: false,
        baseFrozen: false
      },
      expert_count: 0,
      training_sources: [{
        id: "distributed-receipt-fixture",
        raw_text_retained: false,
        distributed_receipt_sha256: distributedReceipt.contentSha256,
        distributed_training_receipt: distributedReceipt
      }],
      ingestion_checkpoints: {
        "fixture-source": { nextRecordIndex: 2 }
      },
      paged_working_memory: {
        format: "omni-working-memory-pages",
        formatVersion: 1,
        count: 2,
        highWaterId: 2,
        contentSha256: "c".repeat(64),
        temporary: true,
        runtimeReadable: true,
        learningReadable: true,
        pageInSupported: true
      },
      substrate: {
        schema: 1,
        dimensions: 16,
        seed: 7,
        persistence: substratePointer
      },
      mutable_state: mutableState.pointer
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
    const artifactBytes = Buffer.concat([
      Buffer.from("89504e470d0a1a0a", "hex"),
      Buffer.from("portable generated artifact")
    ]);
    const copiedCredential = `sk-${"a".repeat(32)}`;
    const anthropicCredential = `sk-ant-api03-${"b".repeat(40)}`;
    const geminiCredential = `AIza${"C".repeat(35)}`;
    const bearerCredential = `Bearer ${"d".repeat(40)}`;
    const mcpEnvironmentSecret = `MCP_ENV_SECRET=${"e".repeat(32)}`;
    const localPosixPath = "/Users/example/private/tool-output.txt";
    const localWindowsPath = "C:\\Users\\example\\private\\tool-output.txt";
    const privateNeedles = [
      copiedCredential,
      anthropicCredential,
      geminiCredential,
      bearerCredential,
      mcpEnvironmentSecret,
      localPosixPath,
      localWindowsPath,
      "raw private tool output"
    ];
    const artifactSha256 = digest(artifactBytes);
    const artifactName = `${artifactSha256}.png`;
    await mkdir(join(engine, "artifacts"), { recursive: true });
    await Promise.all([
      writeFile(join(engine, "artifacts", artifactName), artifactBytes),
      writeFile(
        join(engine, "artifacts", "index.json"),
        serializeArtifactIndex(brain.id, [{
          id: digest("portable-artifact-job"),
          modality: "image",
          mimeType: "image/png",
          sha256: artifactSha256,
          bytes: artifactBytes.length,
          relativePath: `artifacts/${artifactName}`,
          createdAt: "2026-09-07T00:00:00.000Z",
          seed: 17,
          initialization: `loaded from ${localPosixPath}`,
          qualityNote: `tool said ${mcpEnvironmentSecret}`
        }])
      )
    ]);
    const sourceBytes = Buffer.from("private source material");
    brain.messages.push({
      id: randomUUID(),
      role: "human",
      content: `Do not share ${privateNeedles.join(" | ")}`,
      createdAt: new Date().toISOString()
    });
    brain.journal?.push({
      id: randomUUID(),
      createdAt: new Date().toISOString(),
      kind: "tool",
      summary: `credential=${copiedCredential}`,
      detail: JSON.stringify({
        apiKey: anthropicCredential,
        authorization: bearerCredential,
        environment: mcpEnvironmentSecret,
        path: localPosixPath,
        windowsPath: localWindowsPath,
        output: "raw private tool output"
      })
    });
    await writeFile(
      join(engine, "conversation.sqlite3"),
      Buffer.from(`unexportable neural history ${privateNeedles.join(" ")}`)
    );
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
    const portableEngineState = JSON.parse(
      strFromU8(entries["state/engine.json"]!)
    ) as {
      training_sources: Array<{
        distributed_receipt_sha256: string;
        distributed_training_receipt: Record<string, unknown>;
      }>;
      ingestion_checkpoints: Record<string, unknown>;
      paged_working_memory: { count: number };
    };
    expect(portableEngineState.ingestion_checkpoints).toEqual({});
    expect(portableEngineState.paged_working_memory.count).toBe(0);
    const portableOriginEngineState = JSON.parse(
      strFromU8(entries["origin/state/engine.json"]!)
    ) as {
      ingestion_checkpoints: Record<string, unknown>;
      paged_working_memory: { count: number };
    };
    expect(portableOriginEngineState.ingestion_checkpoints).toEqual({});
    expect(portableOriginEngineState.paged_working_memory.count).toBe(0);
    expect(strFromU8(entries["model-card.md"]!)).toContain(
      "1 active ingestion cursor(s) are omitted and will not resume after import"
    );
    expect(Object.keys(entries)).toEqual(
      expect.arrayContaining([
        "manifest.json",
        "checksums.sha256",
        "state/brain.json",
        "state/engine.json",
        "activity/ledger.sqlite3",
        "tensors/core.safetensors",
        "packed/current/manifest.json",
        "packed/origin/manifest.json",
        "substrate/current/manifest.json",
        `substrate/current/${String(substratePointer.generationManifest)}`,
        "substrate/origin/manifest.json",
        "mutable/current/manifest.json",
        `mutable/current/${String(mutableState.pointer.generationManifest)}`,
        "mutable/current/replay.sqlite3",
        "mutable/origin/manifest.json",
        "mutable/origin/replay.sqlite3",
        "origin/state/brain.json",
        "artifacts/index.json",
        `artifacts/files/${artifactName}`
      ])
    );
    expect(entries["conversation/ledger.sqlite3"]).toBeUndefined();
    expect(entries["conversation/neural-ledger.sqlite3"]).toBeUndefined();
    const portableArchiveText = Object.values(entries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n");
    for (const needle of privateNeedles) {
      expect(portableArchiveText).not.toContain(needle);
    }
    expect(Object.keys(entries).some((name) => name.startsWith("blobs/"))).toBe(false);
    expect(portableState.trainingSources[0]?.rawText).toBeUndefined();
    expect(portableState.trainingSources[0]?.path).toBeUndefined();
    expect(
      portableEngineState.training_sources[0]?.distributed_training_receipt
    ).toEqual(distributedReceipt);
    expect(
      portableEngineState.training_sources[0]?.distributed_receipt_sha256
    ).toBe(distributedReceipt.contentSha256);
    const {
      contentSha256: portableReceiptSha256,
      ...portableReceiptBody
    } = portableEngineState.training_sources[0]!.distributed_training_receipt;
    expect(portableReceiptSha256).toBe(digest(canonicalJson(portableReceiptBody)));
    expect(strFromU8(entries["state/brain.json"]!)).not.toContain(copiedCredential);
    expect(portableArchiveText).toContain("[REDACTED_SECRET]");
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as {
      architecture: string;
      architectureSchemaVersion: number;
      secretRedaction: { replacements: number };
      licenseLedger: {
        application: string;
        sourceCount: number;
        sourceLedger: string;
        sources: Array<{ name: string; license: string }>;
      };
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
    expect(manifest.licenseLedger).toMatchObject({
      sourceCount: 1,
      sourceLedger: "activity/ledger.sqlite3",
      sources: []
    });

    const originPath = join(temporaryRoot, "origin-portable.omni");
    await repository.exportBundle(brain.id, originPath, "origin");
    const originEntries = unzipSync(new Uint8Array(await readFile(originPath)));
    for (const engineEntry of ["state/engine.json", "origin/state/engine.json"]) {
      const exported = JSON.parse(strFromU8(originEntries[engineEntry]!)) as {
        ingestion_checkpoints: Record<string, unknown>;
        paged_working_memory: { count: number };
      };
      expect(exported.ingestion_checkpoints).toEqual({});
      expect(exported.paged_working_memory.count).toBe(0);
    }
    for (const source of [engine, join(engine, "origin")]) {
      const unchanged = JSON.parse(await readFile(join(source, "brain.json"), "utf8")) as {
        ingestion_checkpoints: Record<string, unknown>;
        paged_working_memory: { count: number };
      };
      expect(unchanged.ingestion_checkpoints).toEqual(engineState.ingestion_checkpoints);
      expect(unchanged.paged_working_memory).toEqual(engineState.paged_working_memory);
    }
    expect(originEntries["conversation/ledger.sqlite3"]).toBeUndefined();
    expect(originEntries["conversation/neural-ledger.sqlite3"]).toBeUndefined();
    const originArchiveText = Object.values(originEntries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n");
    for (const needle of privateNeedles) {
      expect(originArchiveText).not.toContain(needle);
    }

    const imported = await repository.importBundle(portablePath);
    expect(imported.id).not.toBe(brain.id);
    expect(imported.name).toBe(brain.name);
    expect(imported.activity).toMatchObject({
      journalCount: 3,
      trainingSourceCount: 1
    });
    expect(imported.trainingSources).toEqual([
      expect.objectContaining({ name: "private.txt" })
    ]);
    expect(imported.trainingSources[0]?.path).toBeUndefined();
    expect(imported.trainingSources[0]?.rawText).toBeUndefined();
    const importedEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8")
    ) as {
      brain_id: string;
      training_sources: Array<{
        distributed_receipt_sha256: string;
        distributed_training_receipt: Record<string, unknown>;
      }>;
    };
    expect(importedEngine.brain_id).toBe(imported.id);
    expect(
      importedEngine.training_sources[0]?.distributed_training_receipt
    ).toEqual(distributedReceipt);
    expect(
      importedEngine.training_sources[0]?.distributed_receipt_sha256
    ).toBe(distributedReceipt.contentSha256);
    const importedConversation = JSON.stringify(
      (await repository.conversationPage(imported.id)).entries
    );
    for (const needle of privateNeedles) {
      expect(importedConversation).not.toContain(needle);
    }
    await expect(
      readFile(
        join(repository.brainDirectory(imported.id), "engine", "packed-ternary", "manifest.json")
      )
    ).resolves.toBeInstanceOf(Buffer);
    await expect(
      readFile(
        join(repository.brainDirectory(imported.id), "engine", "substrate", "manifest.json"),
        "utf8"
      )
    ).resolves.toBe(canonicalJson(substratePointer));
    await expect(
      readFile(
        join(repository.brainDirectory(imported.id), "engine", "state", "manifest.json"),
        "utf8"
      )
    ).resolves.toBe(canonicalJson(mutableState.pointer));
    expect(replayRowCount(join(
      repository.brainDirectory(imported.id), "engine", "state", "replay.sqlite3"
    ))).toBe(3);
    await expect(readFile(join(engine, "state", "replay.sqlite3")))
      .resolves.toEqual(mutableState.replay);
    const visibleBeforeBadReplay = (await repository.list()).map((item) => item.id).sort();
    const tamperedReplayEntries = unzipSync(new Uint8Array(await readFile(portablePath)));
    tamperedReplayEntries["mutable/current/replay.sqlite3"] = Buffer.from(
      "SQLite format 3\0not a valid committed replay database"
    );
    refreshArchiveIntegrity(tamperedReplayEntries);
    const tamperedReplayPath = join(temporaryRoot, "tampered-replay.omni");
    await writeFile(tamperedReplayPath, zipSync(tamperedReplayEntries));
    await expect(repository.importBundle(tamperedReplayPath)).rejects.toThrow(
      /bundled mutable replay failed SQLite and checkpoint verification/i
    );
    expect((await repository.list()).map((item) => item.id).sort())
      .toEqual(visibleBeforeBadReplay);
    const importedArtifactStore = await ArtifactIndexStore.open(
      join(repository.brainDirectory(imported.id), "engine", "artifacts"),
      imported.id
    );
    const importedArtifactIndex = importedArtifactStore.snapshot();
    importedArtifactStore.close();
    expect(importedArtifactIndex.artifacts[0]).toMatchObject({
      sha256: artifactSha256,
      bytes: artifactBytes.length,
      seed: 17
    });
    expect(importedArtifactIndex.artifacts[0]?.initialization).toBeUndefined();
    expect(importedArtifactIndex.artifacts[0]?.qualityNote).toBeUndefined();
    await expect(
      readFile(join(repository.brainDirectory(imported.id), "engine", "artifacts", artifactName))
    ).resolves.toEqual(artifactBytes);

    const tamperedArtifactEntries = unzipSync(new Uint8Array(await readFile(portablePath)));
    tamperedArtifactEntries[`artifacts/files/${artifactName}`] = Buffer.concat([
      artifactBytes,
      Buffer.from("tampered")
    ]);
    refreshArchiveIntegrity(tamperedArtifactEntries);
    const tamperedArtifactPath = join(temporaryRoot, "tampered-artifact.omni");
    await writeFile(tamperedArtifactPath, zipSync(tamperedArtifactEntries));
    await expect(repository.importBundle(tamperedArtifactPath)).rejects.toThrow(
      /generated artifact failed integrity/i
    );

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
    expect((await repository.list()).map((item) => item.id).sort()).toEqual(visibleBeforeFailure);
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
    expect(privateEntries["conversation/ledger.sqlite3"]).toBeUndefined();
    expect(privateEntries["conversation/neural-ledger.sqlite3"]).toBeUndefined();
    const privateArchiveText = Object.values(privateEntries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n");
    for (const needle of privateNeedles) {
      expect(privateArchiveText).not.toContain(needle);
    }
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
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Schema mind"
    });
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
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Stable bundle"
    });
    const portablePath = join(temporaryRoot, "stable.omni");
    await repository.exportBundle(brain.id, portablePath, "current");
    const entries = unzipSync(new Uint8Array(await readFile(portablePath)));
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!)) as Record<string, unknown>;
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
      repository.importBundleBuffer(Buffer.from(zipSync(entries)), "old-beta.omni")
    ).rejects.toThrow(/beta .omni bundle/i);
  });

  it("refuses a private source archive when retained text appears to contain a credential", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Private mind"
    });
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

  it.each([
    { mode: "current", scope: "current", leak: "credential" },
    { mode: "origin", scope: "origin", leak: "private path" },
    { mode: "private-archive", scope: "current", leak: "private path" },
    { mode: "referenced", scope: "origin", leak: "credential" }
  ] as const)(
    "refuses $mode export when $scope substrate JSON contains a $leak",
    async ({ mode, scope, leak }) => {
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: "Substrate privacy fixture"
      });
      const leakValue = leak === "credential"
        ? `sk-${"q".repeat(36)}`
        : "C:\\Users\\fixture\\private\\neuron-label.txt";
      await materializeSubstrateExportFixture(repository, brain, { [scope]: leakValue });
      const destination = join(temporaryRoot, `blocked-${mode}-${scope}.omni`);
      const priorArchive = Buffer.from("existing archive must not be replaced");
      await writeFile(destination, priorArchive);

      let failure: unknown;
      try {
        await repository.exportBundle(brain.id, destination, mode);
      } catch (error) {
        failure = error;
      }
      expect(failure).toBeInstanceOf(Error);
      const message = (failure as Error).message;
      // Origin-portable projects immutable origin into both archive slots.
      const archiveScope = mode === "origin" ? "current" : scope;
      expect(message).toMatch(
        new RegExp(`substrate/${archiveScope}/blobs/[a-f0-9]{64}\\.json.*(?:credentials|private path)`, "i")
      );
      expect(message.includes(leakValue)).toBe(false);
      await expect(readFile(destination)).resolves.toEqual(priorArchive);
      expect((await readdir(temporaryRoot)).filter((name) => name.startsWith(".omni-export-")))
        .toEqual([]);
    }
  );

  it("refuses a credential in a selected substrate generation manifest", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Substrate manifest privacy fixture"
    });
    const fakeKey = `sk-${"m".repeat(36)}`;
    await materializeSubstrateExportFixture(repository, brain, {
      originManifest: fakeKey
    });
    const destination = join(temporaryRoot, "blocked-manifest.omni");

    let failure: unknown;
    try {
      await repository.exportBundle(brain.id, destination, "current");
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(Error);
    const message = (failure as Error).message;
    expect(message).toMatch(
      /substrate\/origin\/generations\/[a-f0-9]{64}\/manifest\.json.*credentials/i
    );
    expect(message.includes(fakeKey)).toBe(false);
    await expect(stat(destination)).rejects.toMatchObject({ code: "ENOENT" });
    expect((await readdir(temporaryRoot)).filter((name) => name.startsWith(".omni-export-")))
      .toEqual([]);
  });

  it("scans a substrate record beyond the initial file sample", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large substrate privacy fixture"
    });
    const fakeKey = `sk-${"l".repeat(36)}`;
    await materializeSubstrateExportFixture(repository, brain, {
      current: `${"ordinary label ".repeat(20_000)}${fakeKey}`
    });
    const destination = join(temporaryRoot, "blocked-late-key.omni");

    let failure: unknown;
    try {
      await repository.exportBundle(brain.id, destination, "current");
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(Error);
    const message = (failure as Error).message;
    expect(message).toMatch(/substrate\/current\/blobs\/[a-f0-9]{64}\.json.*credentials/i);
    expect(message.includes(fakeKey)).toBe(false);
    await expect(stat(destination)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("exports checksum-bound packed-v2 native substrate generations", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Packed-v2 portable fixture"
    });
    await materializeSubstrateExportFixture(repository, brain, {}, 2);
    const destination = join(temporaryRoot, "packed-v2.omni");
    await repository.exportBundle(brain.id, destination, "current");
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    for (const scope of ["current", "origin"] as const) {
      const prefix = `substrate/${scope}`;
      const pointer = JSON.parse(strFromU8(entries[`${prefix}/manifest.json`]!)) as {
        formatVersion: number;
        generationManifest: string;
      };
      expect(pointer.formatVersion).toBe(2);
      const generation = JSON.parse(
        strFromU8(entries[`${prefix}/${pointer.generationManifest}`]!)
      ) as { formatVersion: number };
      expect(generation.formatVersion).toBe(2);
    }
  });

  it("exports clean current and origin substrate JSON unchanged in every bundle mode", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Clean substrate privacy fixture"
    });
    await materializeSubstrateExportFixture(repository, brain, {
      current: "learned cortical source",
      origin: "clean immutable source"
    });

    for (const mode of ["current", "origin", "private-archive", "referenced"] as const) {
      const destination = join(temporaryRoot, `clean-${mode}.omni`);
      await repository.exportBundle(brain.id, destination, mode);
      const entries = unzipSync(new Uint8Array(await readFile(destination)));
      for (const scope of ["current", "origin"] as const) {
        const prefix = `substrate/${scope}`;
        const pointer = JSON.parse(strFromU8(entries[`${prefix}/manifest.json`]!)) as {
          generationManifest: string;
          generationManifestSha256: string;
        };
        const generationBytes = entries[`${prefix}/${pointer.generationManifest}`]!;
        expect(digest(generationBytes)).toBe(pointer.generationManifestSha256);
        const generation = JSON.parse(strFromU8(generationBytes)) as {
          shards: Array<{ records: { path: string; sha256: string } }>;
        };
        const record = generation.shards[0]!.records;
        expect(digest(entries[`${prefix}/${record.path}`]!)).toBe(record.sha256);
      }
    }
  });

  it("round-trips a lightweight local reference and fails clearly without its blob store", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Referenced mind"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(join(engine, "origin"), { recursive: true });
    const engineState = {
      schema_version: 1,
      format: "omni-cortex-engine",
      release_format: "stable-1.0",
      brain_id: brain.id,
      name: brain.name,
      config: { origin_kind: "ground-up" },
      runtime_card: {
        origin_kind: "ground-up",
        pretrained: false,
        baseFrozen: false
      },
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

    const historySecret = `sk-${"z".repeat(32)}`;
    const historyPath = "/Users/example/private/tool-output.txt";
    brain.messages.push({
      id: randomUUID(),
      role: "human",
      content: `Use ${historySecret} only with ${historyPath}`,
      createdAt: new Date().toISOString()
    });
    brain.journal?.push({
      id: randomUUID(),
      kind: "tool",
      summary: "system.files.read: complete.",
      detail: JSON.stringify({ path: historyPath, apiKey: historySecret }),
      createdAt: new Date().toISOString()
    });
    await repository.save(brain);

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
    const portableEntries = unzipSync(new Uint8Array(portableBytes));
    const referenceManifest = JSON.parse(strFromU8(referenceEntries["manifest.json"]!)) as {
      mode: string;
      references: Record<string, string>;
    };
    expect(referenceManifest.mode).toBe("referenced-local");
    for (const archiveEntries of [portableEntries, referenceEntries]) {
      expect(archiveEntries["conversation/ledger.sqlite3"]).toBeUndefined();
      expect(archiveEntries["conversation/neural-ledger.sqlite3"]).toBeUndefined();
      const exposedText = Object.values(archiveEntries)
        .map((value) => Buffer.from(value).toString("utf8"))
        .join("\n");
      expect(exposedText).not.toContain(historySecret);
      expect(exposedText).not.toContain(historyPath);
    }
    expect(Object.values(referenceManifest.references)).toHaveLength(4);
    expect(
      Object.values(referenceManifest.references).every((hash) => /^[a-f0-9]{64}$/.test(hash))
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
    // APFS may materialize a clone with a different inode while preserving
    // the verified content-addressed bytes.
    expect(importedInfo.size).toBe(blobInfo.size);
    expect(digest(await readFile(importedCorePath))).toBe(
      referenceManifest.references.currentCore
    );

    const separate = new BrainRepository(join(temporaryRoot, "separate-brains"));
    await separate.initialize();
    await expect(separate.importBundleBuffer(referenceBytes, "reference.omni")).rejects.toThrow(
      /unavailable on this installation/
    );

    referenceManifest.references.currentCore = "not-a-content-hash";
    referenceEntries["manifest.json"] = Buffer.from(JSON.stringify(referenceManifest, null, 2));
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
