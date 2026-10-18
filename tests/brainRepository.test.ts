import { createHash, randomUUID } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, readdir, rename, rm, stat, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { strFromU8, unzipSync, zipSync } from "fflate";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  BrainRepository,
  MUTABLE_STATE_STORE_FORMAT,
  SUBSTRATE_STORE_FORMAT,
  type BrainPluginPortabilityHooks
} from "../src/main/brainRepository";
import {
  ArtifactIndexStore,
  serializeArtifactIndex
} from "../src/main/mediaArtifactRegistry";
import { verifyPortableReplaySqlite } from "../src/main/portableReplayIntegrity";
import { persistAuthorizedToolIntent } from "../src/main/toolIntentJournal";
import type { BrainStorageOperationHooks } from "../src/main/brainStorageOperations";
import { DEFAULT_CONFIG, type DiskSpaceTelemetry } from "../src/shared/types";
import { rewriteNativeIdentity } from "../src/main/connectedBrainArchive";
import { APP_AUDIT_PNG, syntheticAppAudit } from "./fixtures/mcpAppAudit";
import type { NativeAppObservationHead } from "../src/shared/nativeAppObservation";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { tinyCatalog } from "./fixtures/wholeBrainCatalogArchive";
import { collectWholeBrainCatalogLineage } from "../src/main/wholeBrainCatalogPortability";
import { ApiTeacherStore } from "../src/main/apiTeacherStore";
import { completedMcpTaskFixture } from "./fixtures/mcpTaskCustody";
import { MCP_TASK_DIRECTORY, McpTaskStore } from "../src/main/mcpTaskStore";
import { mcpTaskCustodyFiles } from "../src/main/mcpTaskCustody";
import { pausedTeacherMediaFixture } from "./fixtures/apiTeacherMedia";
import { API_TEACHER_MEDIA_REQUIRED_FEATURE } from "../src/main/apiTeacherMedia";

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
  for (const path of Object.keys(manifest.files)) {
    if (!Object.hasOwn(entries, path)) delete manifest.files[path];
  }
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

async function writePackedTernaryFixture(directory: string, weighted = false): Promise<void> {
  const shard = Buffer.from([0b01010101]);
  const shardHash = digest(shard);
  const tensorHash = digest(
    Buffer.concat([
      Buffer.from(canonicalJson({ dtype: "int8", shape: [1] })),
      Buffer.from([0]),
      Buffer.from([0])
    ])
  );
  const connectionBytes = Buffer.from([0x13]);
  const connectionHash = digest(connectionBytes);
  const connectionFile = `connections-00000-${connectionHash.slice(0, 16)}.bin`;
  const parallelBytes = Buffer.alloc(26);
  parallelBytes[24] = 0x56;
  parallelBytes[25] = 0x12;
  const parallelHash = digest(parallelBytes);
  const parallelFile = `parallel-00000-${parallelHash.slice(0, 16)}.bin`;
  const manifestBody = {
    format: "omni-packed-ternary",
    formatVersion: weighted ? 2 : 1,
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
        tensorSha256: tensorHash,
        ...(weighted ? {
          connectionState: { formatVersion: 1, encoding: "uint4-strength-minus-one-high4-resistance",
            shape: [1, 1], shard: connectionFile, byteLength: 1, sha256: connectionHash },
          parallelConnections: { encoding: "u64-row-column-ordinal-u8-sign-state-le", count: 1,
            byteLength: 26, shard: parallelFile, sha256: parallelHash }
        } : {})
      }
    ],
    shards: [
      {
        file: `ternary-00000-${shardHash.slice(0, 16)}.bin`,
        byteLength: shard.byteLength,
        sha256: shardHash
      },
      ...(weighted ? [
        { file: connectionFile, byteLength: 1, sha256: connectionHash },
        { file: parallelFile, byteLength: 26, sha256: parallelHash }
      ] : [])
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
  if (weighted) await Promise.all([
    writeFile(join(directory, connectionFile), connectionBytes),
    writeFile(join(directory, parallelFile), parallelBytes)
  ]);
}

async function materializeNativeEngineFixture(
  repository: Pick<BrainRepository, "brainDirectory">,
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
  await saveCatalogFixture(engine);
  await saveCatalogFixture(origin);
}

/** Explicit test setup simulating a completed native save, never runtime repair.
 * Every catalog binding names real inert committed fixture generations. */
async function saveCatalogFixture(engine: string): Promise<Record<string, unknown>> {
  const path = join(engine, "brain.json"), raw = await readFile(path, "utf8");
  const metadata = JSON.parse(raw) as Record<string, unknown>;
  if (!metadata.mutable_state) metadata.mutable_state = await writeEmptyMutableStateFixture(engine, String(metadata.brain_id));
  if (!(metadata.substrate as { persistence?: unknown } | undefined)?.persistence) {
    metadata.substrate = { schema: 1, dimensions: 16, seed: 7, persistence: await writeSubstrateFixture(engine) };
  }
  const fixture = await tinyCatalog(engine, String(metadata.brain_id), metadata);
  // Hash-bound connected descriptors in these fixtures intentionally retain
  // Python float lexemes. Catalog setup does not change their numeric bytes.
  await writeFile(path, JSON.stringify(fixture.metadata).replaceAll('"unitStrength":1', '"unitStrength":1.0'));
  return fixture.metadata;
}

async function writeEmptyMutableStateFixture(engine: string, brainId: string): Promise<Record<string, unknown>> {
  const store = join(engine, "state"), roles: Record<string, unknown> = {};
  await mkdir(join(store, "blobs"), { recursive: true });
  for (const [role, name] of [["core", "core.safetensors"], ["plasticity", "plasticity.safetensors"], ["optimizer", undefined]] as const) {
    const bytes = name ? await readFile(join(engine, name)) : emptySafetensors(), sha256 = digest(bytes);
    const header = JSON.parse(bytes.subarray(8, 8 + Number(bytes.readBigUInt64LE(0))).toString("utf8"));
    await writeFile(join(store, "blobs", `${sha256}.safetensors`), bytes);
    roles[role] = { path: `blobs/${sha256}.safetensors`, sha256, bytes: bytes.length,
      tensorCount: Object.keys(header).filter(key => key !== "__metadata__").length };
  }
  const body = { format: MUTABLE_STATE_STORE_FORMAT, formatVersion: 1, brainId, createdAt: 1, roles,
    optimizerStructure: { type: "dict", items: [] },
    replay: { format: "omni-replay-sqlite", formatVersion: 1, path: "replay.sqlite3", count: 0,
      highWaterId: 0, contentSha256: digest(""), transactional: true, silentEviction: false },
    metadata: { trainingSteps: 0 }, activationState: [], safeTensorOnly: true, transactional: true };
  const contentSha256 = digest(canonicalJson(body)), bytes = Buffer.from(canonicalJson({ ...body, contentSha256 }));
  const pointer = { format: MUTABLE_STATE_STORE_FORMAT, formatVersion: 1, activeGeneration: contentSha256, contentSha256,
    generationManifest: `generations/${contentSha256}/manifest.json`, generationManifestSha256: digest(bytes), replayCount: 0 };
  await mkdir(join(store, "generations", contentSha256), { recursive: true });
  await writeFile(join(store, "generations", contentSha256, "manifest.json"), bytes);
  await writeFile(join(store, "manifest.json"), canonicalJson(pointer));
  const replay = new DatabaseSync(join(store, "replay.sqlite3"));
  try {
    replay.exec("CREATE TABLE replay(sequence INTEGER PRIMARY KEY AUTOINCREMENT,created_at REAL NOT NULL,sha256 TEXT NOT NULL,dtype TEXT NOT NULL,shape_json TEXT NOT NULL,payload BLOB NOT NULL)");
  } finally { replay.close(); }
  return pointer;
}

async function expectCopiedCatalogState(engine: string, source: Record<string, unknown>, brainId: string): Promise<void> {
  const actual = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
  const { whole_brain_catalog: sourcePointer, whole_brain_catalog_lineage: _priorLineage, ...saved } = source;
  const { whole_brain_catalog: current, whole_brain_catalog_lineage: lineage, ...copied } = actual;
  expect(copied).toEqual({ ...saved, brain_id: brainId });
  expect(current).toBeUndefined();
  expect(lineage).toMatchObject({ format: "omni-whole-brain-catalog-lineage" });
  const closure = await collectWholeBrainCatalogLineage(engine, actual);
  if (sourcePointer) expect([...closure.entries.values()].some(entry =>
    entry.sourceBrainId === source.brain_id && canonicalJson(entry.catalogPointer) === canonicalJson(sourcePointer))).toBe(true);
}

const fixtureWalWriters: DatabaseSync[] = [];

async function writeNeuralConversationFixture(
  engineDirectory: string, brainId: string, text: string, keepWalOpen = false
): Promise<Record<string, unknown>> {
  const database = new DatabaseSync(join(engineDirectory, "conversation.sqlite3"));
  database.exec(`
    PRAGMA journal_mode=${keepWalOpen ? "WAL" : "DELETE"};
    PRAGMA wal_autocheckpoint=0;
    CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
    CREATE TABLE entries(sequence INTEGER PRIMARY KEY,entry_key TEXT UNIQUE,kind TEXT,
      created_at TEXT,attention_epoch INTEGER,payload_json TEXT,payload_sha256 TEXT,
      previous_sha256 TEXT,row_sha256 TEXT UNIQUE);
  `);
  database.prepare("INSERT INTO meta VALUES('brain_id',?)").run(brainId);
  database.prepare("INSERT INTO meta VALUES('format','omni-neural-conversation-ledger')").run();
  let previous = "0".repeat(64);
  const records = [
    { kind: "message", value: { id: "saved-human", role: "human", content: text } },
    { kind: "message", value: { id: "saved-assistant", role: "assistant", content: "saved reply" } },
    { kind: "trace", value: { id: "saved-trace", input: text, seed: 7 } }
  ];
  const createdAt = "2026-09-29T00:00:00.000Z";
  for (const [index, record] of records.entries()) {
    const payload = canonicalJson(record.value);
    const body = {
      sequence: index + 1, entryKey: `${record.kind}:${record.value.id}`,
      kind: record.kind, createdAt, attentionEpoch: 2,
      payloadSha256: digest(payload), previousSha256: previous
    };
    const rowSha256 = digest(canonicalJson(body));
    database.prepare("INSERT INTO entries VALUES(?,?,?,?,?,?,?,?,?)").run(
      body.sequence, body.entryKey, body.kind, createdAt, 2, payload,
      body.payloadSha256, previous, rowSha256
    );
    previous = rowSha256;
  }
  if (keepWalOpen) fixtureWalWriters.push(database); else database.close();
  return {
    format: "omni-neural-conversation-ledger", formatVersion: 1,
    totalEntries: 3, messageCount: 2, actionCount: 0, traceCount: 1,
    attentionEpoch: 2, headSequence: 3, headSha256: previous
  };
}

async function writeWorkingPagesFixture(
  engineDirectory: string, text: string, keepWalOpen = false
): Promise<Record<string, unknown>> {
  await mkdir(join(engineDirectory, "state"), { recursive: true });
  const database = new DatabaseSync(join(engineDirectory, "state", "working-memory.sqlite3"));
  database.exec(`
    PRAGMA journal_mode=${keepWalOpen ? "WAL" : "DELETE"};
    PRAGMA wal_autocheckpoint=0;
    CREATE TABLE working_pages(sequence INTEGER PRIMARY KEY AUTOINCREMENT,page_id TEXT UNIQUE,
      assembly_id TEXT,metadata_json TEXT,sha256 TEXT,dtype TEXT,shape_json TEXT,
      payload BLOB,updated_at REAL);
  `);
  const rows = [1, 2].map((number) => {
    const payload = Buffer.from([number]);
    const sha256 = digest(Buffer.concat([Buffer.from("torch.uint8\0[1]\0"), payload]));
    const pageId = `page-${number}`;
    database.prepare("INSERT INTO working_pages VALUES(?,?,?,?,?,?,?,?,?)").run(
      number, pageId, `assembly-${number}`,
      JSON.stringify({ assemblyId: `assembly-${number}`, note: text }),
      sha256, "torch.uint8", "[1]", payload, 1.0
    );
    return `${number}\0${pageId}\0${sha256}\n`;
  });
  if (keepWalOpen) fixtureWalWriters.push(database); else database.close();
  return {
    format: "omni-working-memory-pages", formatVersion: 1, count: 2,
    highWaterId: 2, contentSha256: digest(rows.join("")), temporary: true,
    runtimeReadable: true, learningReadable: true, pageInSupported: true
  };
}

async function writeSubstrateFixture(
  engineDirectory: string,
  sourceLabel?: string,
  generationNote?: string,
  formatVersion: 1 | 2 | 3 = 1
): Promise<Record<string, unknown>> {
  const store = join(engineDirectory, "substrate");
  const packedRows = Buffer.alloc(4, 0x55);
  const updateCounters = Buffer.alloc(8);
  const packedHeaderValue = {
    packed_rows: { dtype: "U8", shape: [1, 4], data_offsets: [0, 4] },
    update_counters_le: { dtype: "U8", shape: [1, 8], data_offsets: [4, 12] }
  };
  const packedHeaderJson = JSON.stringify(packedHeaderValue);
  const packedHeader = Buffer.from(
    packedHeaderJson.padEnd(Math.ceil(packedHeaderJson.length / 8) * 8, " ")
  );
  const packedPrefix = Buffer.alloc(8);
  packedPrefix.writeBigUInt64LE(BigInt(packedHeader.byteLength));
  const packedTensorBytes = Buffer.concat([
    packedPrefix, packedHeader, packedRows, updateCounters
  ]);
  const packedTensorHash = digest(packedTensorBytes);
  const packedMetadata = {
    format: "omni-packed-vsa-vectors",
    formatVersion: 1,
    dimensions: 16,
    seed: 7,
    zeroDeadband: 0.25,
    rowCount: 1,
    ids: ["neuron-fixture"],
    idsSha256: digest(JSON.stringify(["neuron-fixture"])),
    packedSha256: digest(packedRows),
    countersSha256: digest(updateCounters)
  };
  const record = {
    kind: "neurons",
    ids: ["neuron-fixture"],
    records: [{
      id: "neuron-fixture",
      region: "cortical",
      ...(sourceLabel === undefined ? {} : { source_label: sourceLabel })
    }],
    vectorIds: formatVersion === 3 ? ["neuron-fixture"] : [],
    ...(formatVersion === 3 ? { packedVectorState: packedMetadata } : {})
  };
  const recordBytes = Buffer.from(canonicalJson(record));
  const recordHash = digest(recordBytes);
  const generationBody = {
    format: "omni-substrate-shards",
    formatVersion,
    schema: formatVersion === 3 ? "neural-substrate-2" : 1,
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
        bucket: digest("neurons:neuron-fixture").slice(0, 1),
        part: 0,
        count: 1,
        records: {
          path: `blobs/${recordHash}.json`,
          sha256: recordHash,
          bytes: recordBytes.byteLength
        },
        tensors: formatVersion === 3 ? {
          path: `blobs/${packedTensorHash}.safetensors`,
          sha256: packedTensorHash,
          bytes: packedTensorBytes.byteLength
        } : null
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
    ...(formatVersion === 3 ? [
      writeFile(join(store, "blobs", `${packedTensorHash}.safetensors`), packedTensorBytes)
    ] : []),
    writeFile(join(store, ...generationRelative.split("/")), generationBytes),
    writeFile(join(store, "manifest.json"), canonicalJson(pointer))
  ]);
  return pointer;
}

async function materializeSubstrateExportFixture(
  repository: Pick<BrainRepository, "brainDirectory">,
  brain: { id: string; name: string },
  labels: {
    current?: string;
    origin?: string;
    currentManifest?: string;
    originManifest?: string;
  } = {},
  formatVersion: 1 | 2 | 3 = 1
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
    state.substrate = {
      schema: formatVersion === 3 ? "neural-substrate-2" : 1,
      dimensions: 16,
      seed: 7,
      persistence: pointer
    };
    await writeFile(statePath, JSON.stringify(state));
    await saveCatalogFixture(directory);
  }
}

/** Tiny host filesystem fixture, not a learned-capability/native-runtime claim. */
async function materializeConnectedExportFixture(repository: Pick<BrainRepository, "brainDirectory" | "storeBlob">, brain: { id: string; name: string }, nested = false): Promise<void> {
  await materializeSubstrateExportFixture(repository, brain, {}, 3);
  const root = repository.brainDirectory(brain.id), engine = join(root, "engine");
  const shared = Buffer.from("saved source shared outside the original parent tree");
  const sharedHash = await repository.storeBlob(shared);
  const regions: Record<string, unknown>[] = [], index: Record<string, unknown>[] = [];
  for (const [parentId, regionId] of [["parent-a", "region-a"], ["parent-b", "region-b"]]) {
    const parent = join(root, "merge", "parents", parentId!), native = join(parent, "engine");
    await mkdir(join(parent, "workspace"), { recursive: true });
    await mkdir(native, { recursive: true });
    const pointer = await writeSubstrateFixture(native, parentId, undefined, 3);
    const metadata = { brain_id: parentId, config: { origin_kind: "ground-up", unitStrength: 1 },
      substrate: { schema: "neural-substrate-2", dimensions: 16, seed: 7, persistence: pointer } };
    let metadataText = canonicalJson(metadata).replace('"unitStrength":1', '"unitStrength":1.0');
    await writeFile(join(native, "brain.json"), metadataText);
    await writeFile(join(native, "core.safetensors"), emptySafetensors());
    await writeFile(join(native, "plasticity.safetensors"), emptySafetensors());
    await writeFile(join(parent, "brain.json"), JSON.stringify({ id: parentId, trainingSources: [{ blobHash: sharedHash }] }));
    await writeFile(join(parent, "workspace", "original-script.js"), "throw new Error('archived only, never execute');\n");
    await writeFile(join(parent, "workspace", "notes-journal"), "user document, not a SQLite sidecar");
    if (nested && parentId === "parent-a") {
      await materializeConnectedExportFixture({ brainDirectory: () => parent,
        storeBlob: (value: Buffer) => repository.storeBlob(value) }, { id: parentId, name: "Nested parent" });
      for (const database of fixtureWalWriters.splice(0)) database.close();
      metadataText = await readFile(join(native, "brain.json"), "utf8");
    }
    regions.push({ parentId, regionId, enginePath: `merge/parents/${parentId}/engine`,
      generationSha256: digest(parentId!), neuralSha256: digest(regionId!), config: JSON.parse(metadataText).config,
      metadataSha256: digest(metadataText), coreSha256: digest(emptySafetensors()), plasticitySha256: digest(emptySafetensors()) });
    const visit = async (directory: string, prefix = ""): Promise<void> => {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isDirectory()) await visit(join(directory, entry.name), path);
        else {
          const bytes = await readFile(join(directory, entry.name));
          index.push({ parentId, path, sha256: digest(bytes), bytes: bytes.length, mode: 0o600 });
        }
      }
    };
    await visit(parent);
  }
  const descriptorBody = { format: "omni-connected-native-regions", formatVersion: 1, regions,
    parents: regions.map(value => ({ parentId: value.parentId, regionId: value.regionId, generationSha256: value.generationSha256 })),
    deduplication: "identical-complete-neural-state-only" };
  const descriptor = { ...descriptorBody, sha256: digest(canonicalJson(descriptorBody).replaceAll('"unitStrength":1', '"unitStrength":1.0')) };
  await writeFile(join(root, "merge", "parents.jsonl"), [
    { format: "omni-full-merge-parent-index", formatVersion: 1, parents: descriptor.parents }, ...index
  ].map(value => JSON.stringify(value)).join("\n") + "\n");
  await writeFile(join(root, "full-merge.json"), JSON.stringify({ format: "omni-full-brain-merge", formatVersion: 1, targetBrainId: brain.id, parents: descriptor.parents }));
  const history = new DatabaseSync(join(root, "merge", "host-history.sqlite3"));
  history.exec("CREATE TABLE records(payload TEXT); INSERT INTO records VALUES('complete original history');"); history.close();
  for (const scope of [engine, join(engine, "origin")]) {
    if (!nested) for (const region of regions) {
      await cp(join(root, "merge", "parents", String(region.parentId), "engine"),
        join(scope, "connected-inputs", "merge", "parents", String(region.parentId), "engine"), { recursive: true });
    }
    const packedPath = join(scope, "packed-ternary", "manifest.json");
    const packed = JSON.parse(await readFile(packedPath, "utf8"));
    packed.metadata = { ...packed.metadata, brainId: brain.id, parameterChecksum: digest("fixture-parameters") };
    delete packed.contentSha256;
    packed.contentSha256 = digest(canonicalJson(packed).replace('"scale":1,', '"scale":1.0,'));
    const packedText = canonicalJson(packed).replace('"scale":1,', '"scale":1.0,');
    await writeFile(packedPath, packedText);
    await writeFile(join(scope, "packed-ternary", "manifest.sha256"), `${digest(packedText)}\n`);
    const graphs: Record<string, unknown> = {};
    for (const region of regions) {
      const graphEngine = join(scope, "connected-state", String(region.regionId));
      const pointer = await writeSubstrateFixture(graphEngine, `regional-${region.regionId}`, undefined, 3);
      graphs[String(region.regionId)] = { enginePath: `engine/connected-state/${region.regionId}`,
        substrate: { schema: "neural-substrate-2", dimensions: 16, seed: 7, persistence: pointer } };
      if (nested && region.parentId === "parent-a") {
        const parentEngine = join(root, "merge", "parents", "parent-a", "engine");
        const parent = JSON.parse(await readFile(join(parentEngine, "brain.json"), "utf8"));
        Object.assign(graphs[String(region.regionId)] as object, {
          nestedStateRoot: `engine/connected-state/${region.regionId}/nested`, nestedGraphStates: parent.connected_graph_states
        });
        await cp(join(parentEngine, "connected-state"), join(graphEngine, "nested", "engine", "connected-state"), { recursive: true });
      }
      // Interrupted-save mirrors do not select the committed generation.
      await writeFile(join(graphEngine, "substrate", "manifest.json"), '{"orphaned":true}');
    }
    const graphBody = { format: "omni-connected-graph-states", formatVersion: 1,
      descriptorSha256: descriptor.sha256, attentionEpoch: 0, regions: graphs };
    const metadata = JSON.parse(await readFile(join(scope, "brain.json"), "utf8"));
    metadata.connected_regions = descriptor;
    metadata.connected_graph_states = { ...graphBody, sha256: digest(canonicalJson(graphBody)) };
    metadata.connected_runtime_adoption = { format: "fixture-runtime-adoption", sha256: digest("adoption") };
    metadata.connected_pending_ingestions = [{ id: digest("pending"), status: "queued", sourceName: "saved original source" }];
    await writeFile(join(scope, "brain.json"), JSON.stringify(metadata).replaceAll('"unitStrength":1', '"unitStrength":1.0'));
    await saveCatalogFixture(scope);
    const runtime = new DatabaseSync(join(scope, "connected-runtime.sqlite3"));
    runtime.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0; CREATE TABLE provenance(source TEXT); INSERT INTO provenance VALUES('committed WAL provenance');");
    fixtureWalWriters.push(runtime);
    await writeFile(join(scope, "connected-assembly.json"), JSON.stringify({ historicalOwner: brain.id, source: "complete-parent-state" }));
  }
  const origin = join(engine, "origin");
  const marker = { format: "omni-connected-native-origin", formatVersion: 1, brainId: brain.id,
    descriptorSha256: descriptor.sha256, runtimeAdoptionSha256: digest("adoption"), initialCurriculumRerun: false,
    parameterChecksum: digest("fixture-parameters"),
    metadataSha256: digest(await readFile(join(origin, "brain.json"))),
    coreSha256: digest(await readFile(join(origin, "core.safetensors"))),
    plasticitySha256: digest(await readFile(join(origin, "plasticity.safetensors"))),
    packedManifestSha256: digest(await readFile(join(origin, "packed-ternary", "manifest.json"))) };
  await writeFile(join(origin, "connected-origin.json"), JSON.stringify({ ...marker, sha256: digest(canonicalJson(marker)) }));
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
      DROP TABLE IF EXISTS replay;
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
    await saveCatalogFixture(directory);
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
    for (const database of fixtureWalWriters.splice(0)) database.close();
    await rm(temporaryRoot, { recursive: true, force: true });
  });

  it("rekeys native identities without changing hash-bound numeric lexemes or nested strings", () => {
    const source = '{"brain_id":"old","name":"old","config":{"name":"old","gain":1.0},"connected_regions":{"brain_id":"ancestor","gain":1.0,"small":1e-08,"text":"name\\\":old"}}';
    const result = rewriteNativeIdentity(source, "new", "New name");
    expect(result).toBe(source.replace('"brain_id":"old"', '"brain_id":"new"').replaceAll('"name":"old"', '"name":"New name"'));
    expect(() => rewriteNativeIdentity('{"brain_id":"a","brain_id":"b"}', "new")).toThrow(/duplicate/);
  });

  it.each(["current", "private-archive", "referenced", "origin"] as const)(
    "round trips connected dependency closure as opaque payloads (%s)", async mode => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Connected portable filesystem fixture" });
    await materializeConnectedExportFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), engine = join(root, "engine");
    const point = await repository.snapshot(brain.id, "connected recovery");
    const before = await readFile(join(engine, "brain.json"));
    const bundle = join(temporaryRoot, `connected-${mode}.omni`);
    await repository.exportBundle(brain.id, bundle, mode);
    const entries = unzipSync(new Uint8Array(await readFile(bundle)));
    expect(entries["connected/manifest.json"]).toBeDefined();
    expect(entries[`snapshots/${point.id}/connected/manifest.json`]).toBeDefined();
    expect(Object.keys(entries).some(name => /\.js$/.test(name))).toBe(false);
    const manifest = JSON.parse(strFromU8(entries["connected/manifest.json"]!));
    expect(manifest.files["merge/parents/parent-a/workspace/original-script.js"]).toEqual(manifest.files["merge/parents/parent-b/workspace/original-script.js"]);
    const receiving = mode === "referenced" ? repository : new BrainRepository(join(temporaryRoot, `destination-${mode}`));
    await receiving.initialize();
    const imported = await receiving.importBundle(bundle), destination = receiving.brainDirectory(imported.id);
    expect(await readFile(join(destination, "merge", "parents", "parent-a", "workspace", "original-script.js"), "utf8"))
      .toContain("archived only, never execute");
    expect(await readFile(join(destination, "merge", "parents", "parent-a", "workspace", "notes-journal"), "utf8"))
      .toBe("user document, not a SQLite sidecar");
    const loadedText = await readFile(join(destination, "engine", "brain.json"), "utf8");
    expect(loadedText).toContain('"unitStrength":1.0');
    const native = JSON.parse(loadedText);
    expect(native.brain_id).toBe(imported.id);
    expect(native.connected_pending_ingestions[0].status).toBe("queued");
    for (const scope of ["engine", "engine/origin"]) {
      expect(JSON.parse(await readFile(join(destination, scope, "connected-state", "region-a", "substrate", "manifest.json"), "utf8")))
        .toEqual(native.connected_graph_states.regions["region-a"].substrate.persistence);
      const runtime = new DatabaseSync(join(destination, scope, "connected-runtime.sqlite3"), { readOnly: true });
      try { expect(runtime.prepare("SELECT source FROM provenance").get()?.source).toBe("committed WAL provenance"); }
      finally { runtime.close(); }
    }
    const shared = Buffer.from("saved source shared outside the original parent tree");
    expect(await receiving.getBlob(digest(shared))).toEqual(shared);
    if (mode === "current") {
      const disposable = await receiving.create({ ...DEFAULT_CONFIG, name: "Disposable unrelated fixture" });
      await receiving.permanentlyDeleteInstance({ brainId: disposable.id, acknowledgedIrreversible: true,
        typedName: disposable.name, finalConfirmation: `PERMANENTLY DELETE ${disposable.name}` });
      expect(await receiving.getBlob(digest(shared))).toEqual(shared);
    }
    expect(await readFile(join(destination, "engine", "origin", "connected-origin.json")))
      .toEqual(await readFile(join(engine, "origin", "connected-origin.json")));
    await receiving.restoreSnapshot(imported.id, point.id);
    expect(await readFile(join(destination, "merge", "parents", "parent-b", "workspace", "original-script.js"), "utf8")).toContain("never execute");
    expect(await readFile(join(engine, "brain.json"))).toEqual(before);
  });

  it("retains nested connected graph owners and every nested immutable parent tree", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Nested connected portability fixture" });
    await materializeConnectedExportFixture(repository, brain, true);
    const root = repository.brainDirectory(brain.id), bundle = join(temporaryRoot, "nested-connected.omni");
    await repository.exportBundle(brain.id, bundle);
    const imported = await repository.importBundle(bundle), target = repository.brainDirectory(imported.id);
    for (const scope of ["engine", "engine/origin"]) {
      const path = `${scope}/connected-state/region-a/nested/engine/connected-state/region-b/substrate/manifest.json`;
      const pointer = JSON.parse(await readFile(join(target, path), "utf8"));
      expect(pointer.format).toBe("omni-substrate-shards");
      expect(await readFile(join(root, path), "utf8")).toBe('{"orphaned":true}');
    }
    const nestedSource = "merge/parents/parent-a/merge/parents/parent-b/workspace/original-script.js";
    expect(await readFile(join(target, nestedSource))).toEqual(await readFile(join(root, nestedSource)));
    const fork = await repository.fork(imported.id, "Nested branch");
    expect(await readFile(join(repository.brainDirectory(fork.id), nestedSource))).toEqual(await readFile(join(root, nestedSource)));
  });

  it("isolates connected mutable fork state and rolls back failed recovery host promotion", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Connected COW fixture" });
    await materializeConnectedExportFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), engine = join(root, "engine");
    const point = await repository.snapshot(brain.id, "connected COW boundary");
    const fork = await repository.fork(brain.id, "Connected branch"), target = repository.brainDirectory(fork.id);
    for (const path of ["engine/connected-runtime.sqlite3", "engine/connected-state/region-a/substrate/manifest.json", "merge/host-history.sqlite3", "merge/parents/parent-a/engine/core.safetensors"]) {
      expect((await stat(join(target, path))).ino).not.toBe((await stat(join(root, path))).ino);
    }
    await writeFile(join(target, "engine", "connected-state", "region-a", "substrate", "manifest.json"), "fork-only edit");
    expect(await readFile(join(engine, "connected-state", "region-a", "substrate", "manifest.json"), "utf8")).toBe('{"orphaned":true}');
    const originalParent = await readFile(join(root, "merge", "parents", "parent-a", "engine", "core.safetensors"));
    await writeFile(join(target, "merge", "parents", "parent-a", "engine", "core.safetensors"), "fork-only corruption");
    expect(await readFile(join(root, "merge", "parents", "parent-a", "engine", "core.safetensors"))).toEqual(originalParent);
    for (const database of fixtureWalWriters.splice(0)) database.close();
    const history = new DatabaseSync(join(root, "merge", "host-history.sqlite3"));
    history.exec("INSERT INTO records VALUES('history after snapshot');"); history.close();
    const historyBefore = await readFile(join(root, "merge", "host-history.sqlite3"));
    await expect(repository.restoreSnapshot(brain.id, point.id, undefined, async () => { throw new Error("fixture reload failure"); })).rejects.toThrow("fixture reload failure");
    expect(await readFile(join(root, "merge", "host-history.sqlite3"))).toEqual(historyBefore);
    expect((await readdir(root)).some(name => /^\.connected-.*\.previous$/.test(name))).toBe(false);
  });

  it("rejects missing, redirected, and corrupted connected payloads before publishing an import", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Connected validation fixture" });
    await materializeConnectedExportFixture(repository, brain);
    const bundle = join(temporaryRoot, "connected-malformed.omni");
    await repository.exportBundle(brain.id, bundle);
    const original = await readFile(bundle), count = (await repository.list()).length;
    for (const change of ["missing", "escape", "unsafe", "corrupt", "alias"] as const) {
      const entries = unzipSync(new Uint8Array(original));
      const manifest = JSON.parse(strFromU8(entries["connected/manifest.json"]!));
      const path = "merge/parents/parent-a/engine/core.safetensors", descriptor = manifest.files[path];
      if (change === "missing") delete manifest.files[path];
      if (change === "escape") { manifest.files["merge/../../escape"] = descriptor; delete manifest.files[path]; }
      if (change === "unsafe") manifest.files["merge/parents/parent-a/engine/core.pt"] = descriptor;
      if (change === "corrupt") entries[`connected/payloads/${descriptor.sha256}`] = Buffer.from("corrupted");
      if (change === "alias") {
        const payload = Buffer.from("changed engine-local native alias"), sha256 = digest(payload);
        manifest.files["engine/connected-inputs/merge/parents/parent-a/engine/core.safetensors"] = { sha256, bytes: payload.length };
        entries[`connected/payloads/${sha256}`] = payload;
      }
      entries["connected/manifest.json"] = Buffer.from(JSON.stringify(manifest));
      refreshArchiveIntegrity(entries);
      await expect(repository.importBundleBuffer(Buffer.from(zipSync(entries)), `${change}.omni`)).rejects.toThrow(/connected|Connected|Unsafe|unsafe|checksum/);
      expect((await repository.list()).length).toBe(count);
    }
  });

  it("rejects a connected generation reached through an intermediate filesystem link", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Connected path boundary" });
    await materializeConnectedExportFixture(repository, brain);
    const substrate = join(repository.brainDirectory(brain.id), "engine", "connected-state", "region-a", "substrate");
    const outside = join(temporaryRoot, "outside-regional-substrate");
    await rename(substrate, outside);
    await symlink(outside, substrate, "dir");
    await expect(repository.exportBundle(brain.id, join(temporaryRoot, "rejected-link.omni"))).rejects.toThrow(/symbolic link/);
  });

  it("cancels connected materialization without publishing or changing its source", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Connected cancelled fork" });
    await materializeConnectedExportFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), before = await readFile(join(root, "engine", "brain.json"));
    const controller = new AbortController();
    const hooks: BrainStorageOperationHooks = { signal: controller.signal, checkDisk: async () => safeDiskTelemetry(),
      checkpoint: async value => { if (value?.label === "Restoring verified connected graph and parent history") controller.abort(); } };
    await expect(repository.fork(brain.id, "Cancelled connected branch", hooks)).rejects.toThrow();
    expect((await repository.list()).map(value => value.id)).toEqual([brain.id]);
    expect(await readFile(join(root, "engine", "brain.json"))).toEqual(before);
    expect((await readdir(repository.root)).some(name => name.startsWith(".connected-"))).toBe(false);
  });

  it.each(["current", "referenced"] as const)("round trips weighted native companions in %s exports", async mode => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Weighted native archive fixture" });
    await repository.updateToolPermission(brain.id, "system.files", "System Files", "full", {
      requestId: `export-policy-${mode}`
    });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await writePackedTernaryFixture(join(engine, "packed-ternary"), true);
    const before = await readFile(join(engine, "packed-ternary", "manifest.json"));
    const manifest = JSON.parse(before.toString("utf8"));
    const destination = join(temporaryRoot, `weighted-${mode}.omni`);
    await repository.exportBundle(brain.id, destination, mode);
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    expect(JSON.parse(strFromU8(entries["manifest.json"]!)).packedTernary.formatVersion).toBe(2);
    const imported = await repository.importBundle(destination);
    expect((await repository.get(brain.id)).toolPermissions?.find(
      (permission) => permission.toolId === "system.files"
    )?.level).toBe("full");
    expect(imported.toolPermissions?.find(
      (permission) => permission.toolId === "system.files"
    )?.level).toBe("ask");
    const loaded = join(repository.brainDirectory(imported.id), "engine", "packed-ternary");
    expect(await readFile(join(loaded, "manifest.json"))).toEqual(before);
    for (const descriptor of manifest.shards) {
      expect(await readFile(join(loaded, descriptor.file))).toEqual(await readFile(join(engine, "packed-ternary", descriptor.file)));
    }
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

  it("accepts checksum-validated packed-v3 native substrate totals", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Packed-v3 substrate fixture"
    });
    const engine = join(repository.brainDirectory(brain.id), "engine");
    await mkdir(engine, { recursive: true });
    const pointer = await writeSubstrateFixture(engine, undefined, undefined, 3);
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
    const vectorAccounting = {
      ...accounting, substrateVectorParameters: 16, totalNeuralParameters: 136
    };
    await writeMetadata(vectorAccounting);
    await expect(repository.persistedSubstrateOverview(brain.id)).resolves.toMatchObject({
      parameterAccounting: vectorAccounting
    });
    for (const stale of [
      { ...accounting, foundationEffectiveParameters: 0 },
      { ...accounting, sequenceDynamicSparseSynapses: 0 },
      { ...accounting, fixedSequenceStatisticalCapacity: 0 },
      { ...accounting, totalNeuralParameters: 121 },
      { ...accounting, substrateDynamicSparseSynapses: 1 },
      { ...accounting, substrateVectorParameters: -1 },
      { ...accounting, substrateVectorParameters: "16", totalNeuralParameters: 136 },
      { ...vectorAccounting, totalNeuralParameters: 120 }
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

  it("preserves optional Best Work settings across saves and rejects invalid search budgets", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, bestWorkEnabled: true, bestWorkExplorationBranches: 8 });
    expect((await repository.get(brain.id)).config).toMatchObject({ bestWorkEnabled: true, bestWorkExplorationBranches: 8 });
    const disabled = await repository.save({ ...brain, config: { ...brain.config, bestWorkEnabled: false } });
    expect((await repository.get(disabled.id)).config).toMatchObject({ bestWorkEnabled: false, bestWorkExplorationBranches: 8 });
    for (const config of [
      { ...DEFAULT_CONFIG, bestWorkEnabled: "true" },
      { ...DEFAULT_CONFIG, bestWorkExplorationBranches: true },
      { ...DEFAULT_CONFIG, bestWorkExplorationBranches: 0 },
      { ...DEFAULT_CONFIG, bestWorkExplorationBranches: 17 }
    ]) {
      expect(() => repository.prepareConfig(config as unknown as typeof DEFAULT_CONFIG)).toThrow(/Best Work/);
    }
    expect(repository.prepareConfig({ name: "Older native brain" } as typeof DEFAULT_CONFIG))
      .toMatchObject({ bestWorkEnabled: false, bestWorkExplorationBranches: 16 });
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

  it.each(["current", "private-archive", "referenced", "origin"] as const)(
    "round-trips saved chat, pending replay, workspace pages and cursors without sanitizing (%s)", async (mode) => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Portable native mind" });
    await materializeNativeEngineFixture(repository, brain);
    const engineDirectory = join(repository.brainDirectory(brain.id), "engine", ...(mode === "origin" ? ["origin"] : []));
    const enginePath = join(engineDirectory, "brain.json");
    const native = JSON.parse(await readFile(enginePath, "utf8")) as Record<string, unknown>;
    const privateText = "private turn: violet lantern at midnight";
    native.conversation = await writeNeuralConversationFixture(engineDirectory, brain.id, privateText, true);
    native.completed_chat_turns = [{ content: privateText }];
    native.completed_chat_slow_learning = ["b".repeat(64)];
    native.pending_chat_slow_learning = [{
      jobId: "e".repeat(64), inputSha256: digest(privateText),
      humanMessageId: "saved-human", content: privateText
    }];
    native.paged_working_memory = await writeWorkingPagesFixture(engineDirectory, privateText, true);
    native.ingestion_checkpoints = { ["f".repeat(64)]: { formatVersion: 2, nextRecordIndex: 2 } };
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
    Object.assign(native, await saveCatalogFixture(engineDirectory));
    const historyNative = JSON.parse(await readFile(join(repository.brainDirectory(brain.id), "engine", "brain.json"), "utf8"));
    const recoveryPoint = await repository.snapshot(brain.id, "saved private recovery note");
    const original = await readFile(enginePath);
    const bundle = join(temporaryRoot, `saved-native-conversation-${mode}.omni`);
    await repository.exportBundle(brain.id, bundle, mode);
    expect(await readFile(enginePath)).toEqual(original);
    const bundleBytes = new Uint8Array(await readFile(bundle));
    const archive = unzipSync(bundleBytes);
    expect(archive["conversation/neural-ledger.sqlite3"]).toBeDefined();
    expect(archive["working/current/working-memory.sqlite3"]).toBeDefined();
    const manifest = JSON.parse(strFromU8(archive["manifest.json"]!)) as Record<string, unknown>;
    expect(manifest.conversationProjection).toEqual({
      historyIncluded: true, omittedLedgerRows: 0, omittedPendingReplayJobs: 0
    });
    expect(manifest.savedInstance).toEqual({ version: 1, content: "unsanitized" });
    const modelCard = strFromU8(archive["model-card.md"]!);
    expect(modelCard).toContain("Unsanitized saved-instance export");
    expect(Buffer.from(archive["state/engine.json"]!)).toEqual(original);
    expect(Object.values(archive).some((entry) =>
      Buffer.from(entry).includes(Buffer.from(privateText))
    )).toBe(true);
    const portableBrain = JSON.parse(strFromU8(archive["state/brain.json"]!)) as Record<string, unknown>;
    expect(portableBrain.conversation).toBeDefined();
    const imported = await repository.importBundle(bundle);
    const importedEngine = JSON.parse(
      await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8")
    ) as Record<string, unknown>;
    expect(importedEngine.brain_id).toBe(imported.id);
    await expectCopiedCatalogState(join(repository.brainDirectory(imported.id), "engine"), native, imported.id);
    expect(imported.originChecksum).toBe(portableBrain.originChecksum);
    expect(imported.config.idleCognition).toBe(false);
    expect(await repository.listSnapshots(imported.id)).toEqual([
      expect.objectContaining({ id: recoveryPoint.id, brainId: imported.id, label: "saved private recovery note" })
    ]);
    const importedWorking = new DatabaseSync(join(repository.brainDirectory(imported.id), "engine", "state", "working-memory.sqlite3"), { readOnly: true });
    try { expect(importedWorking.prepare("SELECT COUNT(*) AS count FROM working_pages").get()?.count).toBe(2); }
    finally { importedWorking.close(); }
    await repository.restoreSnapshot(imported.id, recoveryPoint.id);
    const restoredHistory = JSON.parse(await readFile(
      join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8"
    ));
    await expectCopiedCatalogState(join(repository.brainDirectory(imported.id), "engine"), historyNative, imported.id);

    for (const path of ["state/engine.json", "origin/state/engine.json"]) {
      const tampered = unzipSync(bundleBytes);
      const changed = JSON.parse(strFromU8(tampered[path]!)) as Record<string, unknown>;
      changed.conversation = {
        format: "omni-neural-conversation-ledger", formatVersion: 1,
        actionCount: 0, traceCount: 0, attentionEpoch: 0,
        totalEntries: 1,
        messageCount: 1,
        headSequence: 1,
        headSha256: "d".repeat(64)
      };
      tampered[path] = Buffer.from(JSON.stringify(changed));
      refreshArchiveIntegrity(tampered);
      await expect(repository.importBundleBuffer(Buffer.from(zipSync(tampered))))
        .rejects.toThrow(/omits their neural ledger|does not match its saved ledger/i);
    }
    const tampered = unzipSync(bundleBytes);
    const broken = JSON.parse(strFromU8(tampered["state/engine.json"]!));
    broken.paged_working_memory.contentSha256 = "0".repeat(64);
    tampered["state/engine.json"] = Buffer.from(JSON.stringify(broken));
    refreshArchiveIntegrity(tampered);
    const visible = (await repository.list()).map((entry) => entry.id).sort();
    await expect(repository.importBundleBuffer(Buffer.from(zipSync(tampered))))
      .rejects.toThrow(/working-memory checkpoint checksum/i);
    expect((await repository.list()).map((entry) => entry.id).sort()).toEqual(visible);
  });

  it("preserves a WAL-held neural ledger suffix beyond the saved head but rejects dangling replay refs", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Neural committed-prefix fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine");
    const statePath = join(engine, "brain.json");
    const native = JSON.parse(await readFile(statePath, "utf8"));
    native.conversation = await writeNeuralConversationFixture(engine, brain.id, "committed human input");
    native.pending_chat_slow_learning = [{ jobId: "a".repeat(64), inputSha256: digest("committed human input"), humanMessageId: "saved-human" }];
    const database = new DatabaseSync(join(engine, "conversation.sqlite3"));
    database.exec("PRAGMA journal_mode=WAL; PRAGMA wal_autocheckpoint=0;");
    fixtureWalWriters.push(database);
    const payload = canonicalJson({ id: "uncommitted-human", role: "human", content: "saved ahead-of-checkpoint suffix" });
    const body = {
      sequence: 4, entryKey: "message:uncommitted-human", kind: "message",
      createdAt: "2026-09-29T01:00:00.000Z", attentionEpoch: 3,
      payloadSha256: digest(payload), previousSha256: native.conversation.headSha256
    };
    database.prepare("INSERT INTO entries VALUES(?,?,?,?,?,?,?,?,?)").run(
      4, body.entryKey, body.kind, body.createdAt, 3, payload,
      body.payloadSha256, body.previousSha256, digest(canonicalJson(body))
    );
    await writeFile(statePath, JSON.stringify(native));
    const destination = join(temporaryRoot, "neural-prefix.omni");
    await repository.exportBundle(brain.id, destination);
    const imported = await repository.importBundle(destination);
    const importedLedger = new DatabaseSync(join(repository.brainDirectory(imported.id), "engine", "conversation.sqlite3"), { readOnly: true });
    try { expect(importedLedger.prepare("SELECT COUNT(*) AS count FROM entries").get()?.count).toBe(4); }
    finally { importedLedger.close(); }
    const importedState = JSON.parse(await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"), "utf8"));
    expect(importedState.conversation.headSequence).toBe(3);
    expect(importedState.pending_chat_slow_learning).toEqual(native.pending_chat_slow_learning);
    native.pending_chat_slow_learning[0].humanMessageId = "uncommitted-human";
    await writeFile(statePath, JSON.stringify(native));
    await expect(repository.exportBundle(brain.id, join(temporaryRoot, "dangling-job.omni")))
      .rejects.toThrow(/pending chat replay has no saved human message/i);
  });

  it("preserves a cursor-bound v3 joint generation and rejects rebinding before import publication", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Saved joint cursor fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine");
    const statePath = join(engine, "brain.json");
    const native = JSON.parse(await readFile(statePath, "utf8"));
    const substrate = await writeSubstrateFixture(engine, undefined, undefined, 3);
    const mutable = await writeMutableStateFixture(engine);
    const mutableGeneration = JSON.parse(await readFile(
      join(engine, "state", String(mutable.pointer.generationManifest)), "utf8"
    ));
    // Transport-only fixture: no worker/model is instantiated or trained.
    const cursor = {
      formatVersion: 3, commitSequence: 2, committedRecords: 2,
      recordPrefixSha256: "a".repeat(64), sourceManifestSha256: "b".repeat(64),
      parserManifestSha256: "c".repeat(64), contentHash: "d".repeat(64),
      sourceLabel: "/Users/fixture/private/data.csv"
    };
    const generationId = "e".repeat(32);
    const relativeManifest = `generations/${generationId}/manifest.json`;
    const jointDirectory = join(engine, "state", "ingestion-joint", "generations", generationId);
    await mkdir(jointDirectory, { recursive: true });
    const snapshotPath = join(jointDirectory, "packed-vector-index.sqlite3");
    const snapshot = new DatabaseSync(snapshotPath);
    snapshot.exec("CREATE TABLE fixture(key TEXT PRIMARY KEY,value TEXT); INSERT INTO fixture VALUES('cursor','saved');");
    snapshot.close();
    const snapshotBytes = await readFile(snapshotPath);
    const blobs = Object.fromEntries(Object.entries(mutableGeneration.roles).map(([role, value]) => {
      const descriptor = value as { path: string; sha256: string; bytes: number };
      return [role, { path: descriptor.path, sha256: descriptor.sha256, bytes: descriptor.bytes }];
    }));
    const joint = {
      format: "omni-joint-checkpoint-generation", formatVersion: 2, generationId,
      neuralState: {
        generationId: mutable.pointer.activeGeneration,
        generationManifest: mutable.pointer.generationManifest,
        generationManifestSha256: mutable.pointer.generationManifestSha256, blobs
      },
      neuralGenerationSha256: mutable.pointer.activeGeneration,
      substrateState: {
        generationId: substrate.activeGeneration, generationManifest: substrate.generationManifest,
        generationManifestSha256: substrate.generationManifestSha256, counts: substrate.counts
      },
      substrateGenerationSha256: substrate.activeGeneration,
      sqliteSnapshot: { file: "packed-vector-index.sqlite3", sha256: digest(snapshotBytes), bytes: snapshotBytes.byteLength },
      checkpointSequence: cursor.commitSequence, previousManifestSha256: null,
      sourceManifestSha256: cursor.sourceManifestSha256, parserManifestSha256: cursor.parserManifestSha256,
      sourceContentSha256: cursor.contentHash,
      sourceParserManifestSha256: digest(canonicalJson({
        sourceManifestSha256: cursor.sourceManifestSha256, parserManifestSha256: cursor.parserManifestSha256
      })),
      cursor: { committedRecords: cursor.committedRecords, recordPrefixSha256: cursor.recordPrefixSha256 },
      coverage: {
        visitedRecords: 2, processedRecords: 2, rejectedRecords: 0, processedBytes: 10,
        expectedRecords: 5, sourceStreamExhausted: false, sourceContentReverifiedSha256: null
      }
    };
    const jointBytes = Buffer.from(canonicalJson(joint));
    await writeFile(join(jointDirectory, "manifest.json"), jointBytes);
    Object.assign(native, {
      substrate: { persistence: substrate }, mutable_state: mutable.pointer,
      ingestion_checkpoints: { ["f".repeat(64)]: cursor },
      ingestion_joint_generation: {
        format: "omni-joint-checkpoint-reference", formatVersion: 2,
        generationId, relativeManifest, sha256: digest(jointBytes)
      }
    });
    await writeFile(statePath, JSON.stringify(native));
    Object.assign(native, await saveCatalogFixture(engine));
    const recoveryPoint = await repository.snapshot(brain.id, "saved v3 cursor boundary");
    const destination = join(temporaryRoot, "saved-joint.omni");
    await repository.exportBundle(brain.id, destination);
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    expect(digest(entries[`joint/current/${relativeManifest}`]!)).toBe(digest(jointBytes));
    expect(digest(entries[`snapshots/${recoveryPoint.id}/engine/state/ingestion-joint/${relativeManifest}`]!)).toBe(digest(jointBytes));
    const imported = await repository.importBundle(destination);
    const importedEngine = join(repository.brainDirectory(imported.id), "engine");
    const importedState = JSON.parse(await readFile(join(importedEngine, "brain.json"), "utf8"));
    expect(importedState.ingestion_checkpoints).toEqual(native.ingestion_checkpoints);
    expect(importedState.ingestion_joint_generation).toEqual(native.ingestion_joint_generation);
    expect(digest(await readFile(join(importedEngine, "state", "ingestion-joint", relativeManifest)))).toBe(digest(jointBytes));
    expect(digest(await readFile(join(importedEngine, "state", "ingestion-joint", "generations", generationId, "packed-vector-index.sqlite3")))).toBe(digest(snapshotBytes));
    await repository.restoreSnapshot(imported.id, recoveryPoint.id);
    const restoredState = JSON.parse(await readFile(join(importedEngine, "brain.json"), "utf8"));
    expect(restoredState.ingestion_checkpoints).toEqual(native.ingestion_checkpoints);
    const visible = (await repository.list()).map((entry) => entry.id).sort();
    for (const mutation of ["cursor", "path", "sqlite"] as const) {
      const tampered = { ...entries };
      const state = JSON.parse(strFromU8(tampered["state/engine.json"]!));
      if (mutation === "cursor") state.ingestion_checkpoints["f".repeat(64)].commitSequence += 1;
      if (mutation === "path") state.ingestion_joint_generation.relativeManifest = "../external-vault.sqlite3";
      if (mutation === "sqlite") {
        tampered[`joint/current/generations/${generationId}/packed-vector-index.sqlite3`] = Buffer.from("not SQLite");
      }
      tampered["state/engine.json"] = Buffer.from(JSON.stringify(state));
      refreshArchiveIntegrity(tampered);
      await expect(repository.importBundleBuffer(Buffer.from(zipSync(tampered))))
        .rejects.toThrow(/joint.*(?:cursor|reference|SQLite)/i);
      expect((await repository.list()).map((entry) => entry.id).sort()).toEqual(visible);
    }
  });

  it("retains already-incomplete historical recovery records and reports missing payloads precisely", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Incomplete historical workspace fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine");
    const native = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
    native.paged_working_memory = await writeWorkingPagesFixture(engine, "saved historical scratch");
    await writeFile(join(engine, "brain.json"), JSON.stringify(native));
    const recoveryPoint = await repository.snapshot(brain.id, "old capture missing cold pages");
    const base = join(repository.brainDirectory(brain.id), "snapshots", recoveryPoint.id);
    const summary = JSON.parse(await readFile(`${base}.meta.json`, "utf8"));
    const missing = "engine/state/working-memory.sqlite3";
    // Reproduce an older capture protocol in a private storage fixture only.
    for (const relative of summary.savedContinuation.componentPaths) {
      const hash = digest(await readFile(join(base, ...relative.split("/"))));
      const index = summary.checkpointComponentSha256.indexOf(hash);
      expect(index).toBeGreaterThanOrEqual(0); summary.checkpointComponentSha256.splice(index, 1);
    }
    await rm(join(base, ...missing.split("/")));
    summary.engineChecksum = digest(summary.checkpointComponentSha256.join(":"));
    delete summary.savedContinuation;
    await writeFile(`${base}.meta.json`, JSON.stringify(summary, null, 2));
    const destination = join(temporaryRoot, "historical-missing-pages.omni");
    await repository.exportBundle(brain.id, destination);
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!));
    expect(manifest.recoveryPoints).toEqual([{ id: recoveryPoint.id, brainId: brain.id, missingPayloads: [missing] }]);
    expect(strFromU8(entries["model-card.md"]!).includes(recoveryPoint.id)).toBe(true);
    expect(JSON.parse(strFromU8(entries[`snapshots/${recoveryPoint.id}/engine/brain.json`]!)).paged_working_memory.count).toBe(2);
    const imported = await repository.importBundle(destination);
    expect(await repository.listSnapshots(imported.id)).toEqual([
      expect.objectContaining({ id: recoveryPoint.id, brainId: imported.id, exportMissingPayloads: [missing] })
    ]);
    const before = await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"));
    await expect(repository.restoreSnapshot(imported.id, recoveryPoint.id)).rejects.toThrow(/working-memory pages are missing/i);
    expect(await readFile(join(repository.brainDirectory(imported.id), "engine", "brain.json"))).toEqual(before);
  });

  it("keeps older sanitized native bundles readable without adding a legacy model fallback", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Earlier native bundle fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const path = join(temporaryRoot, "legacy-native-format.omni");
    await repository.exportBundle(brain.id, path);
    const entries = unzipSync(new Uint8Array(await readFile(path)));
    const manifest = JSON.parse(strFromU8(entries["manifest.json"]!));
    manifest.formatVersion = 1;
    delete manifest.requiredFeatures;
    delete manifest.savedInstance;
    delete manifest.recoveryPoints;
    manifest.conversationProjection = { historyIncluded: false, omittedLedgerRows: 0, omittedPendingReplayJobs: 0 };
    entries["manifest.json"] = Buffer.from(JSON.stringify(manifest));
    delete entries["conversation/ledger.sqlite3"];
    const host = JSON.parse(strFromU8(entries["state/brain.json"]!));
    delete host.conversation;
    entries["state/brain.json"] = Buffer.from(JSON.stringify(host));
    for (const name of ["state/engine.json", "origin/state/engine.json"]) {
      const engine = JSON.parse(strFromU8(entries[name]!));
      delete engine.whole_brain_catalog;
      delete engine.whole_brain_catalog_lineage;
      engine.paged_working_memory = {
        format: "omni-working-memory-pages", formatVersion: 1, count: 0,
        highWaterId: 0, contentSha256: digest(""), temporary: true,
        runtimeReadable: true, learningReadable: true, pageInSupported: true
      };
      engine.ingestion_checkpoints = {};
      entries[name] = Buffer.from(JSON.stringify(engine));
    }
    for (const name of Object.keys(entries)) if (name.startsWith("catalog/")) delete entries[name];
    refreshArchiveIntegrity(entries);
    const imported = await repository.importBundleBuffer(Buffer.from(zipSync(entries)));
    expect(imported.name).toBe(brain.name);
    expect(imported.config.idleCognition).toBe(false);
    expect((await repository.conversationPage(imported.id)).entries).toEqual([]);
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
    await materializeNativeEngineFixture(repository, source);
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
    await materializeNativeEngineFixture(repository, source);
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

  it("does not publish a mixed export when its saved engine commit changes during staging", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Export commit race fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const enginePath = join(repository.brainDirectory(brain.id), "engine", "brain.json");
    const destination = join(temporaryRoot, "prior-export.omni");
    const prior = Buffer.from("prior destination stays intact");
    await writeFile(destination, prior);
    let diskChecks = 0;
    const hooks: BrainStorageOperationHooks = {
      signal: new AbortController().signal,
      checkpoint: vi.fn(async () => undefined),
      checkDisk: vi.fn(async () => {
        diskChecks += 1;
        if (diskChecks === 3) {
          const state = JSON.parse(await readFile(enginePath, "utf8"));
          state.updated_at = "newly committed fixture state";
          await writeFile(enginePath, JSON.stringify(state));
        }
        return safeDiskTelemetry();
      })
    };
    await expect(repository.exportBundle(brain.id, destination, "current", hooks))
      .rejects.toThrow(/changed during export/i);
    expect(await readFile(destination)).toEqual(prior);
  });

  it("keeps an installing identity invisible and rolls it back if installation is cancelled", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Private install fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const bundle = join(temporaryRoot, "private-install.omni");
    await repository.exportBundle(brain.id, bundle);
    const controller = new AbortController();
    let checked = false;
    const hooks: BrainStorageOperationHooks = {
      signal: controller.signal,
      checkDisk: vi.fn(async () => safeDiskTelemetry()),
      checkpoint: vi.fn(async (update = {}) => {
        if (update.phase === "installing" && update.targetBrainId) {
          expect((await repository.list()).some((entry) => entry.id === update.targetBrainId)).toBe(false);
          checked = true;
          controller.abort();
        }
      })
    };
    await expect(repository.importBundle(bundle, {}, hooks)).rejects.toMatchObject({ name: "AbortError" });
    expect(checked).toBe(true);
    expect((await repository.list()).map((entry) => entry.id)).toEqual([brain.id]);
  });

  it("preserves historical training-source versions and their retained blobs, not only the latest row", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Historical source version fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const initial = Buffer.from("first deliberately retained private source");
    const latest = Buffer.from("new deliberately retained private source");
    const initialHash = await repository.storeBlob(initial);
    const latestHash = await repository.storeBlob(latest);
    brain.trainingSources.push({
      id: "saved-source", name: "private.txt", path: "/Users/fixture/private/original.txt", kind: "text",
      bytes: initial.byteLength, learnedIdeas: 0, learnedConcepts: 0, learnedSynapses: 0,
      importedAt: "2026-09-29T00:00:00.000Z", rawTextRetained: true, rawText: initial.toString(),
      contentHash: initialHash, blobHash: initialHash, policy: "archive"
    });
    await repository.save(brain);
    brain.trainingSources[0] = { ...brain.trainingSources[0]!, bytes: latest.byteLength,
      importedAt: "2026-09-29T01:00:00.000Z", rawText: latest.toString(), contentHash: latestHash, blobHash: latestHash };
    await repository.save(brain);
    const bundle = join(temporaryRoot, "historical-sources.omni");
    await repository.exportBundle(brain.id, bundle);
    const entries = unzipSync(new Uint8Array(await readFile(bundle)));
    expect(Buffer.from(entries[`blobs/${initialHash}`]!)).toEqual(initial);
    expect(Buffer.from(entries[`blobs/${latestHash}`]!)).toEqual(latest);
    const imported = await repository.importBundle(bundle);
    expect(imported.activity?.trainingSourceVersionCount).toBe(2);
    expect(imported.trainingSources[0]?.rawText).toBe(latest.toString());
    const database = new DatabaseSync(join(repository.brainDirectory(imported.id), "activity", "ledger.sqlite3"), { readOnly: true });
    try {
      const first = database.prepare("SELECT payload_json FROM training_source_versions ORDER BY sequence LIMIT 1").get();
      expect(JSON.parse(String(first?.payload_json)).rawText).toBe(initial.toString());
    } finally { database.close(); }
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

  it("preserves complete format2 custody through portable import, collision, and re-export without a model Load", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Format2 host-only saved fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const sourceEngine = join(repository.brainDirectory(brain.id), "engine"), originalMetadata = await readFile(join(sourceEngine, "brain.json"));
    const originalOrigin = await readFile(join(sourceEngine, "origin", "brain.json"));
    const sourceState = JSON.parse(originalMetadata.toString("utf8"));
    const firstBundle = join(temporaryRoot, "format2-first.omni"); await repository.exportBundle(brain.id, firstBundle);
    const firstEntries = unzipSync(new Uint8Array(await readFile(firstBundle)));
    expect(JSON.parse(strFromU8(firstEntries["manifest.json"]!))).toMatchObject({ formatVersion: 2,
      requiredFeatures: ["whole-brain-catalog-closure-v1"] });
    expect(firstEntries["catalog/current/projection.json"]).toBeUndefined();
    const receiving = new BrainRepository(join(temporaryRoot, "format2-receiving")); await receiving.initialize();
    const same = await receiving.importBundle(firstBundle);
    expect(same.id).toBe(brain.id);
    expect(JSON.parse(await readFile(join(receiving.brainDirectory(same.id), "engine", "brain.json"), "utf8")).whole_brain_catalog)
      .toEqual(sourceState.whole_brain_catalog);
    const collision = await receiving.importBundle(firstBundle);
    expect(collision.id).not.toBe(brain.id);
    await expectCopiedCatalogState(join(receiving.brainDirectory(collision.id), "engine"), sourceState, collision.id);
    const secondBundle = join(temporaryRoot, "format2-copied.omni"); await receiving.exportBundle(collision.id, secondBundle);
    const secondEntries = unzipSync(new Uint8Array(await readFile(secondBundle)));
    expect(JSON.parse(strFromU8(secondEntries["catalog/current/projection.json"]!))).toMatchObject({
      relationship: "preserved-current-state", destinationRegistry: false, sourceBrainId: brain.id,
      stateBinding: { brainId: collision.id } });
    expect(Buffer.from(secondEntries["origin/state/engine.json"]!)).toEqual(originalOrigin);
    const third = new BrainRepository(join(temporaryRoot, "format2-third")); await third.initialize();
    const importedCopy = await third.importBundle(secondBundle);
    expect(importedCopy.id).toBe(collision.id);
    await expectCopiedCatalogState(join(third.brainDirectory(importedCopy.id), "engine"), sourceState, importedCopy.id);
    expect(await readFile(join(sourceEngine, "brain.json"))).toEqual(originalMetadata);
    expect(await readFile(join(sourceEngine, "origin", "brain.json"))).toEqual(originalOrigin);
  });

  it("refuses an actually stale saved catalog instead of repairing its binding during export", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Stale catalog host fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const path = join(repository.brainDirectory(brain.id), "engine", "brain.json"), state = JSON.parse(await readFile(path, "utf8"));
    state.attention_overlay = { epoch: 2, changedWithoutCatalogSave: true };
    await writeFile(path, JSON.stringify(state)); const before = await readFile(path), destination = join(temporaryRoot, "must-not-publish.omni");
    await expect(repository.exportBundle(brain.id, destination)).rejects.toThrow(/catalog pointer identity/);
    expect(await readFile(path)).toEqual(before);
    await expect(stat(destination)).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("preserves MCP task current, origin, selected recovery and later dormant custody through copy and format2 import", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "MCP task custody host fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), engine = join(root, "engine"), f = await completedMcpTaskFixture(root, brain.id);
    const local = `${MCP_TASK_DIRECTORY}/${f.row.taskRef}.json`, original = await readFile(join(engine, local));
    const fragment = `${local}.11111111-1111-4111-8111-111111111111.next`, fragmentBytes = Buffer.from('inert interrupted JSON {');
    await writeFile(join(engine, fragment), fragmentBytes);
    await cp(join(engine, MCP_TASK_DIRECTORY), join(engine, "origin", MCP_TASK_DIRECTORY), { recursive: true });
    const point = await repository.snapshot(brain.id, "Exact task boundary");
    await f.store.update(brain.id, f.row.taskRef, { reason: "later source-original operational observation" });
    const fork = await repository.duplicate(brain.id), forkEngine = join(repository.brainDirectory(fork.id), "engine");
    const copied = new McpTaskStore({ brainDirectory: async () => repository.brainDirectory(fork.id) });
    expect((await copied.get(fork.id, f.row.taskRef)).source).toEqual(f.row.source);
    await expect(copied.update(fork.id, f.row.taskRef, { reason: "must not mutate ancestral receipt" })).rejects.toThrow(/dormant/);
    expect(await readFile(join(forkEngine, "origin", local))).toEqual(original);
    await repository.restoreSnapshot(brain.id, point.id);
    expect(await readFile(join(engine, local))).toEqual(original);
    const recovered = await f.store.localPage(brain.id);
    expect(recovered.records.some(item => item.record.reason === "later source-original operational observation" && item.custodyPath.includes("/lineage/"))).toBe(true);
    const bundle = join(temporaryRoot, "mcp-tasks.omni"); await repository.exportBundle(brain.id, bundle);
    const entries = unzipSync(new Uint8Array(await readFile(bundle))), manifest = JSON.parse(strFromU8(entries["manifest.json"]!));
    expect(manifest.requiredFeatures).toContain("mcp-task-custody-v1");
    expect(Buffer.from(entries[`mcp-tasks/current/${local}`]!)).toEqual(original);
    expect(Buffer.from(entries[`mcp-tasks/origin/${local}`]!)).toEqual(original);
    expect(entries[`snapshots/${point.id}/engine/${local}`]).toBeDefined();
    const imported = await repository.importBundle(bundle), importedRoot = repository.brainDirectory(imported.id);
    expect(await readFile(join(importedRoot, "engine", local))).toEqual(original);
    expect(await readFile(join(importedRoot, "engine", "origin", local))).toEqual(original);
    expect(await readFile(join(importedRoot, "engine", fragment))).toEqual(fragmentBytes);
    expect(await readFile(join(importedRoot, "engine", "origin", fragment))).toEqual(fragmentBytes);
    expect(await readFile(join(importedRoot, "snapshots", point.id, "engine", fragment))).toEqual(fragmentBytes);
    expect(await readFile(join(importedRoot, "snapshots", point.id, "engine", local))).toEqual(original);
    const importedStore = new McpTaskStore({ brainDirectory: async () => importedRoot });
    expect(await importedStore.retainedOutcome(imported.id, f.row.taskRef)).toMatchObject({ result: f.payload, source: f.row.source, retained: true });
    await expect(importedStore.update(imported.id, f.row.taskRef, { reason: "no imported execution authority" })).rejects.toThrow(/dormant/);
    await repository.exportBundle(imported.id, join(temporaryRoot, "mcp-tasks-reexport.omni"));
  });

  it("rejects undeclared, tampered or live-authority MCP task imports before identity reservation", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "MCP task import validation fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const f = await completedMcpTaskFixture(repository.brainDirectory(brain.id), brain.id), path = join(temporaryRoot, "task-validation.omni");
    await repository.exportBundle(brain.id, path);
    const source = unzipSync(new Uint8Array(await readFile(path))), before = (await repository.list()).map(row => row.id).sort();
    const rootRecord = `mcp-tasks/current/${MCP_TASK_DIRECTORY}/${f.row.taskRef}.json`;
    for (const mutation of ["feature", "payload", "grant", "partial", "format1"] as const) {
      const changed = { ...source };
      if (mutation === "feature" || mutation === "format1") {
        const manifest = JSON.parse(strFromU8(changed["manifest.json"]!));
        manifest.requiredFeatures = mutation === "format1" ? [] : manifest.requiredFeatures.filter((feature: string) => feature !== "mcp-task-custody-v1");
        if (mutation === "format1") manifest.formatVersion = 1;
        changed["manifest.json"] = Buffer.from(JSON.stringify(manifest));
      }
      if (mutation === "payload") changed[`mcp-tasks/current/${MCP_TASK_DIRECTORY}/payloads/${f.outcome.payloadSha256}.json`] = Buffer.from('{"changed":true}');
      if (mutation === "grant") changed[rootRecord] = Buffer.from(JSON.stringify({ ...JSON.parse(strFromU8(changed[rootRecord]!)), managementGrant: "not portable" }));
      if (mutation === "partial") changed[`mcp-tasks/current/${MCP_TASK_DIRECTORY}/unfinished.next`] = Buffer.from("partial");
      refreshArchiveIntegrity(changed);
      await expect(repository.importBundleBuffer(Buffer.from(zipSync(changed)), `${mutation}.omni`)).rejects.toThrow(/MCP task|MCP.*custody|content address|authority|format2|partial|non-regular/i);
      expect((await repository.list()).map(row => row.id).sort()).toEqual(before);
    }
  });

  it("checks selected metadata RAM before capturing task bytes and leaves source and destination unchanged on cancellation", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Task capture admission fixture" });
    await materializeNativeEngineFixture(repository, brain); await completedMcpTaskFixture(repository.brainDirectory(brain.id), brain.id);
    const before = await mcpTaskCustodyFiles(join(repository.brainDirectory(brain.id), "engine"));
    const denied = new BrainRepository(repository.root, undefined, { catalogAdmission: (_bytes, purpose) => {
      if (purpose.includes("MCP task")) throw new Error("selected task metadata RAM is full");
    } });
    await expect(denied.exportBundle(brain.id, join(temporaryRoot, "denied-task.omni"))).rejects.toThrow("selected task metadata RAM is full");
    const controller = new AbortController(), target = join(temporaryRoot, "cancelled-task.omni"); await writeFile(target, "prior archive");
    const operation: BrainStorageOperationHooks = { signal: controller.signal, checkpoint: async () => undefined,
      checkDisk: async () => { controller.abort(); return safeDiskTelemetry(); } };
    await expect(repository.exportBundle(brain.id, target, "current", operation)).rejects.toThrow();
    expect(await readFile(target, "utf8")).toBe("prior archive");
    expect([...(await mcpTaskCustodyFiles(join(repository.brainDirectory(brain.id), "engine"))).keys()].sort()).toEqual([...before.keys()].sort());
    expect((await readdir(repository.root)).some(name => name.startsWith(".mcp-task-capture-") || name.startsWith(".activity-export-"))).toBe(false);
  });

  it("requires and validates historical-only MCP task custody before adopting an archive identity", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Historical-only task fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine"), task = await completedMcpTaskFixture(repository.brainDirectory(brain.id), brain.id);
    const point = await repository.snapshot(brain.id, "Historical task only"), local = `${MCP_TASK_DIRECTORY}/${task.row.taskRef}.json`;
    const original = await readFile(join(engine, local));
    // Inert fixture setup: current state no longer has this operational tree;
    // its exact older recovery point remains the sole retained task owner.
    await rm(join(engine, MCP_TASK_DIRECTORY), { recursive: true, force: true });
    const bundle = join(temporaryRoot, "historical-only-tasks.omni"); await repository.exportBundle(brain.id, bundle);
    const entries = unzipSync(new Uint8Array(await readFile(bundle)));
    expect(entries[`mcp-tasks/current/${local}`]).toBeUndefined();
    expect(JSON.parse(strFromU8(entries["manifest.json"]!)).requiredFeatures).toContain("mcp-task-custody-v1");
    const historical = `snapshots/${point.id}/engine/${local}`, changed = { ...entries };
    changed[historical] = Buffer.from(JSON.stringify({ ...task.row, managementGrant: "must never import" })); refreshArchiveIntegrity(changed);
    const before = (await repository.list()).map(item => item.id).sort();
    await expect(repository.importBundleBuffer(Buffer.from(zipSync(changed)))).rejects.toThrow(/MCP task custody contains unknown or live authority/);
    expect((await repository.list()).map(item => item.id).sort()).toEqual(before);
    const imported = await repository.importBundle(bundle);
    expect(await readFile(join(repository.brainDirectory(imported.id), "snapshots", point.id, "engine", local))).toEqual(original);
  });

  it("declares typed API teacher archive features only when needed and preserves dormant current/origin exports", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Typed teacher conditional host fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const plain = join(temporaryRoot, "teacher-absent.omni"); await repository.exportBundle(brain.id, plain);
    expect(JSON.parse(strFromU8(unzipSync(new Uint8Array(await readFile(plain)))["manifest.json"]!)).requiredFeatures).not.toContain(API_TEACHER_MEDIA_REQUIRED_FEATURE);
    const f = await pausedTeacherMediaFixture(repository.brainDirectory(brain.id), brain.id, "current");
    for (const mode of ["current", "origin"] as const) {
      const bundle = join(temporaryRoot, `teacher-${mode}.omni`); await repository.exportBundle(brain.id, bundle, mode);
      const entries = unzipSync(new Uint8Array(await readFile(bundle))), manifest = JSON.parse(strFromU8(entries["manifest.json"]!));
      expect(manifest.requiredFeatures).toContain(API_TEACHER_MEDIA_REQUIRED_FEATURE);
      const receiving = new BrainRepository(join(temporaryRoot, `teacher-${mode}-receiving`)); await receiving.initialize();
      const imported = await receiving.importBundle(bundle), store = await ApiTeacherStore.open(receiving.brainDirectory(imported.id), imported.id, async () => undefined, { create: false });
      try { expect(store!.jobs()[0]).toMatchObject({ state: "dormant", originalBrainId: brain.id, originalJobId: f.job.id });
        expect(store!.items(store!.jobs()[0]!.id)[0]!.mediaLesson).toEqual(f.mediaLesson); } finally { store?.close(); }
      expect(await readFile(join(receiving.brainDirectory(imported.id), f.imagePath))).toEqual(f.image);
      const changed = { ...entries }; manifest.requiredFeatures = manifest.requiredFeatures.filter((feature: string) => feature !== API_TEACHER_MEDIA_REQUIRED_FEATURE);
      changed["manifest.json"] = Buffer.from(JSON.stringify(manifest)); refreshArchiveIntegrity(changed);
      const before = (await receiving.list()).map(item => item.id).sort();
      await expect(receiving.importBundleBuffer(Buffer.from(zipSync(changed)))).rejects.toThrow(/teacher.*format2 feature.*before identity/i);
      expect((await receiving.list()).map(item => item.id).sort()).toEqual(before);
    }
  });

  it("requires typed API teacher archive custody when only a historical snapshot retains its media", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Historical-only typed teacher" }); await materializeNativeEngineFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), f = await pausedTeacherMediaFixture(root, brain.id, "historical");
    const point = await repository.snapshot(brain.id, "Received media custody only");
    await rm(join(root, "api-teachers"), { recursive: true, force: true });
    const bundle = join(temporaryRoot, "historical-teacher.omni"); await repository.exportBundle(brain.id, bundle);
    const entries = unzipSync(new Uint8Array(await readFile(bundle))), manifest = JSON.parse(strFromU8(entries["manifest.json"]!));
    expect(manifest.requiredFeatures).toContain(API_TEACHER_MEDIA_REQUIRED_FEATURE);
    const imagePath = `snapshots/${point.id}/learning-workspace/blobs/${f.imageSha256}.data`;
    expect(Buffer.from(entries[imagePath]!)).toEqual(f.image);
    const changed = { ...entries }; changed[imagePath] = Buffer.from("not the same media"); refreshArchiveIntegrity(changed);
    const before = (await repository.list()).map(item => item.id).sort();
    await expect(repository.importBundleBuffer(Buffer.from(zipSync(changed)))).rejects.toThrow(/hash|checksum|source length|bytes/i);
    expect((await repository.list()).map(item => item.id).sort()).toEqual(before);
    const imported = await repository.importBundle(bundle);
    expect(await readFile(join(repository.brainDirectory(imported.id), "snapshots", point.id, "learning-workspace", "blobs", `${f.imageSha256}.data`))).toEqual(f.image);
  });

  it("honors selected metadata RAM and late cancellation on typed API teacher archive reads before adoption", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Typed teacher RAM fixture" }); await materializeNativeEngineFixture(repository, brain);
    const root = repository.brainDirectory(brain.id), f = await pausedTeacherMediaFixture(root, brain.id, "ram");
    const bundle = join(temporaryRoot, "teacher-ram.omni"); await repository.exportBundle(brain.id, bundle);
    const admission = vi.fn(async (_bytes: number, purpose: string) => { if (purpose.includes("teacher")) throw new Error("selected teacher metadata RAM denied"); });
    const guarded = new BrainRepository(join(temporaryRoot, "teacher-ram-receiving"), undefined, { catalogAdmission: admission }); await guarded.initialize();
    await expect(guarded.importBundle(bundle)).rejects.toThrow(/selected teacher metadata RAM denied/);
    expect(await guarded.list()).toEqual([]); expect(admission.mock.calls.some(([bytes, purpose]) => bytes > 0 && purpose.includes("teacher"))).toBe(true);
    const controller = new AbortController(), late = new BrainRepository(repository.root, undefined, { catalogAdmission: async (_bytes, purpose) => {
      if (purpose.includes("teacher")) controller.abort(new Error("teacher transfer was stopped after admission"));
    } });
    const operation: BrainStorageOperationHooks = { signal: controller.signal, checkpoint: async () => undefined, checkDisk: async () => safeDiskTelemetry() };
    const target = join(temporaryRoot, "teacher-cancelled.omni");
    await expect(late.exportBundle(brain.id, target, "current", operation)).rejects.toThrow(/teacher transfer was stopped/);
    await expect(stat(target)).rejects.toMatchObject({ code: "ENOENT" });
    expect(await readFile(join(root, f.imagePath))).toEqual(f.image);
  });

  it("creates an immutable origin and copy-on-write neural fork", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Ada" });
    await repository.updateToolPermission(brain.id, "system.files", "System Files", "full", {
      requestId: "fork-policy"
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
    expect(fork.toolPermissions?.find(
      (permission) => permission.toolId === "system.files"
    )?.level).toBe("full");
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
    await repository.updateToolPermission(brain.id, "system.files", "System Files", "full", {
      requestId: "duplicate-policy"
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
    expect(duplicate.toolPermissions?.find(
      (permission) => permission.toolId === "system.files"
    )?.level).toBe("full");
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
    await materializeNativeEngineFixture(repository, stable);
    const engine = join(repository.brainDirectory(stable.id), "engine");
    await mkdir(engine, { recursive: true });
    const betaEngine = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
    delete betaEngine.release_format;
    await writeFile(join(engine, "brain.json"), JSON.stringify(betaEngine));
    await expect(
      repository.exportBundle(stable.id, join(temporaryRoot, "beta-engine.omni"))
    ).rejects.toThrow(/beta engine/i);
  });

  it("rejects draft or missing-origin exports before writing an unimportable identity backup", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Unmaterialized storage fixture" });
    const destination = join(temporaryRoot, "unfinished.omni");
    const prior = Buffer.from("existing destination is preserved");
    await writeFile(destination, prior);
    for (const mode of ["current", "origin", "private-archive", "referenced"] as const) {
      await expect(repository.exportBundle(brain.id, destination, mode))
        .rejects.toThrow(/finish initial learning before exporting this mind/i);
      expect(await readFile(destination)).toEqual(prior);
    }
    await materializeNativeEngineFixture(repository, brain);
    brain.readiness = { state: "initializing", startedAt: brain.createdAt, attempt: 1 };
    await repository.save(brain);
    await expect(repository.exportBundle(brain.id, destination, "current"))
      .rejects.toThrow(/finish initial learning/i);
    brain.readiness = { state: "ready", startedAt: brain.createdAt, completedAt: brain.createdAt, attempt: 1 };
    await repository.save(brain);
    await rm(join(repository.brainDirectory(brain.id), "engine", "origin", "brain.json"));
    await expect(repository.exportBundle(brain.id, destination, "origin"))
      .rejects.toThrow(/materialized origin/i);
    await expect(repository.exportBundle(brain.id, destination, "current"))
      .rejects.toThrow(/materialized origin/i);
    expect(await readFile(destination)).toEqual(prior);
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
    const neuralSummary = await writeNeuralConversationFixture(engine, brain.id, "neural conversation before");
    await writeFile(
      join(engine, "brain.json"),
      JSON.stringify({
        schema_version: 1,
        format: "omni-cortex-engine",
        release_format: "stable-1.0",
        brain_id: brain.id,
        marker: "before",
        conversation: neuralSummary,
        substrate: { persistence: substratePointer },
        mutable_state: mutableState.pointer
      })
    );
    await Promise.all([
      writeFile(join(engine, "core.safetensors"), emptySafetensors()),
      writeFile(join(engine, "plasticity.safetensors"), emptySafetensors())
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
    await repository.updateToolPermission(brain.id, "system.files", "System Files", "full", {
      requestId: "post-snapshot-policy"
    });
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
    expect(restored.toolPermissions?.find(
      (permission) => permission.toolId === "system.files"
    )?.level).toBe("full");
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
    expect(replayRowCount(join(engine, "state", "replay.sqlite3"))).toBe(3);
    await expect(readFile(join(engine, "conversation.sqlite3"))).resolves.toEqual(
      await readFile(join(repository.brainDirectory(brain.id), "snapshots", snapshot.id, "engine", "conversation.sqlite3"))
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

  it("round-trips checksum-verified saved sources, activity and annotated artifacts in every mode", async () => {
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
      conversation: await writeNeuralConversationFixture(engine, brain.id, "raw private tool output"),
      paged_working_memory: await writeWorkingPagesFixture(engine, "saved scratch notes"),
      substrate: {
        schema: 1,
        dimensions: 16,
        seed: 7,
        persistence: substratePointer
      },
      mutable_state: mutableState.pointer
    };
    await writeNeuralConversationFixture(join(engine, "origin"), brain.id, "raw private tool output");
    await writeWorkingPagesFixture(join(engine, "origin"), "saved scratch notes");
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
    await saveCatalogFixture(engine);
    await saveCatalogFixture(join(engine, "origin"));
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
    expect(portableEngineState.ingestion_checkpoints).toEqual(engineState.ingestion_checkpoints);
    expect(portableEngineState.paged_working_memory.count).toBe(2);
    const portableOriginEngineState = JSON.parse(
      strFromU8(entries["origin/state/engine.json"]!)
    ) as {
      ingestion_checkpoints: Record<string, unknown>;
      paged_working_memory: { count: number };
    };
    expect(portableOriginEngineState.ingestion_checkpoints).toEqual(engineState.ingestion_checkpoints);
    expect(portableOriginEngineState.paged_working_memory.count).toBe(2);
    expect(strFromU8(entries["model-card.md"]!)).toContain(
      "ingestion cursors are preserved"
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
    expect(entries["conversation/ledger.sqlite3"]).toBeDefined();
    expect(entries["conversation/neural-ledger.sqlite3"]).toBeDefined();
    expect(entries["artifacts/index.sqlite3"]).toBeDefined();
    const portableArchiveText = Object.values(entries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n").replace(/\\\\/g, "\\");
    for (const needle of privateNeedles) {
      expect(portableArchiveText.includes(needle), "Saved user content is included").toBe(true);
    }
    expect(Object.keys(entries).some((name) => name.startsWith("blobs/"))).toBe(true);
    expect(portableState.trainingSources[0]?.rawText).toBe(sourceBytes.toString("utf8"));
    expect(portableState.trainingSources[0]?.path).toBe("C:\\private\\private.txt");
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
    expect(strFromU8(entries["state/brain.json"]!)).toContain(copiedCredential);
    expect(portableArchiveText).not.toContain("[REDACTED_SECRET]");
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
    expect(manifest.secretRedaction.replacements).toBe(0);
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
      expect(exported.ingestion_checkpoints).toEqual(engineState.ingestion_checkpoints);
      expect(exported.paged_working_memory.count).toBe(2);
    }
    for (const source of [engine, join(engine, "origin")]) {
      const unchanged = JSON.parse(await readFile(join(source, "brain.json"), "utf8")) as {
        ingestion_checkpoints: Record<string, unknown>;
        paged_working_memory: { count: number };
      };
      expect(unchanged.ingestion_checkpoints).toEqual(engineState.ingestion_checkpoints);
      expect(unchanged.paged_working_memory).toEqual(engineState.paged_working_memory);
    }
    expect(originEntries["conversation/ledger.sqlite3"]).toBeDefined();
    expect(originEntries["conversation/neural-ledger.sqlite3"]).toBeDefined();
    const originArchiveText = Object.values(originEntries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n");
    for (const needle of privateNeedles) {
      if (needle === "raw private tool output") expect(originArchiveText).toContain(needle);
      else expect(originArchiveText).not.toContain(needle);
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
    expect(imported.trainingSources[0]?.path).toBe("C:\\private\\private.txt");
    expect(imported.trainingSources[0]?.rawText).toBe(sourceBytes.toString("utf8"));
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
    ).replace(/\\\\/g, "\\");
    for (const needle of privateNeedles) {
      expect(importedConversation).toContain(needle);
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
    expect(importedArtifactIndex.artifacts[0]?.initialization).toBe(`loaded from ${localPosixPath}`);
    expect(importedArtifactIndex.artifacts[0]?.qualityNote).toBe(`tool said ${mcpEnvironmentSecret}`);
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
    expect(privateEntries["conversation/ledger.sqlite3"]).toBeDefined();
    expect(privateEntries["conversation/neural-ledger.sqlite3"]).toBeDefined();
    const privateArchiveText = Object.values(privateEntries)
      .map((value) => Buffer.from(value).toString("utf8"))
      .join("\n").replace(/\\\\/g, "\\");
    for (const needle of privateNeedles) {
      expect(privateArchiveText).toContain(needle);
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
    await materializeNativeEngineFixture(repository, brain);
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
    await materializeNativeEngineFixture(repository, brain);
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
    ).rejects.toThrow(/beta \.omni (?:bundle|neural release)/i);
  });

  it("preserves credential-like text deliberately saved as user source content", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Private mind"
    });
    await materializeNativeEngineFixture(repository, brain);
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
    const destination = join(temporaryRoot, "credentials.omni");
    await repository.exportBundle(brain.id, destination, "private-archive");
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    expect(Buffer.from(entries[`blobs/${blobHash}`]!)).toEqual(secretBytes);
    const imported = await repository.importBundle(destination);
    expect(imported.trainingSources[0]?.rawText).toBe(secretBytes.toString("utf8"));
  });

  it("preserves inert ancestral tool intents through recovery and every saved-instance export", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Intent history fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const intentPath = await persistAuthorizedToolIntent(repository.brainDirectory(brain.id), {
      brainId: brain.id, toolId: "files.read", action: "read", arguments: { path: "/private/unchanged-user-path" }
    }, { id: randomUUID(), requestId: "fixture-turn", startedAt: new Date().toISOString(), permission: "ask", permissionRevision: "fixture-approved" });
    const original = await readFile(intentPath);
    const intentName = intentPath.split(/[\\/]/).at(-1)!;
    const snapshot = await repository.snapshot(brain.id, "With historical intent");
    expect(await readFile(join(repository.brainDirectory(brain.id), "snapshots", snapshot.id,
      "engine", "operational-tool-intents", intentName))).toEqual(original);
    for (const mode of ["current", "private-archive", "referenced"] as const) {
      const destination = join(temporaryRoot, `intent-${mode}.omni`);
      await repository.exportBundle(brain.id, destination, mode);
      const imported = await repository.importBundle(destination);
      expect(imported.id).not.toBe(brain.id);
      expect(await readFile(join(repository.brainDirectory(imported.id), "engine", "operational-tool-intents", intentName))).toEqual(original);
      expect(imported.config.idleCognition).toBe(false);
      const recovered = (await repository.listSnapshots(imported.id)).find(item => item.id === snapshot.id)!;
      expect(await readFile(join(repository.brainDirectory(imported.id), "snapshots", recovered.id,
        "engine", "operational-tool-intents", intentName))).toEqual(original);
    }
  });

  it("preserves the current native App ACK audit through fork, recovery and portable import without a live session", async () => {
    // Host filesystem fixture only: no brain training, media decode or chat.
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "App audit portability fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const engine = join(repository.brainDirectory(brain.id), "engine"), state = JSON.parse(await readFile(join(engine, "brain.json"), "utf8"));
    const dataJson = JSON.stringify({ method: "ui/update-model-context", params: { content: [{ type: "image", data: APP_AUDIT_PNG, mimeType: "image/png" }] } });
    const f = syntheticAppAudit(dataJson, "a".repeat(64), brain.id);
    const head: NativeAppObservationHead = { format: "omni-native-app-observation-head", version: 1,
      streamId: randomUUID(), ordinal: 1, recordSha256: f.content.audit.recordSha256, dataSha256: f.content.audit.dataSha256,
      previousSha256: "0".repeat(64), baseFrontierSha256: "0".repeat(64), disposition: "observe", state: "observed",
      committedAt: new Date().toISOString(), learning: { kind: "experience", source: "mcp-app", sourceLabel: "Synthetic host fixture",
        steps: 0, experienceCountBefore: 0, experienceCountAfter: 1, assemblyId: "synthetic-only", afterimageId: "",
        textSha256: digest(""), textBytes: 0 }, content: f.content };
    state.app_observation_frontier = head;
    await mkdir(join(engine, "app-observation-audits")); await writeFile(join(engine, f.content.audit.path), f.raw);
    await writeFile(join(engine, "brain.json"), JSON.stringify(state));
    const point = await repository.snapshot(brain.id, "Applied ACK before host receipt");
    expect(await readFile(join(repository.brainDirectory(brain.id), "snapshots", point.id, "engine", f.content.audit.path), "utf8")).toBe(f.raw);
    const fork = await repository.duplicate(brain.id);
    expect(await readFile(join(repository.brainDirectory(fork.id), "engine", f.content.audit.path), "utf8")).toBe(f.raw);
    const bundle = join(temporaryRoot, "current-app-audit.omni"); await repository.exportBundle(brain.id, bundle, "current");
    const entries = unzipSync(new Uint8Array(await readFile(bundle)));
    expect(strFromU8(entries[`app-audits/current/${f.content.audit.path}`]!)).toBe(f.raw);
    const imported = await repository.importBundle(bundle), importedEngine = join(repository.brainDirectory(imported.id), "engine");
    expect(await readFile(join(importedEngine, f.content.audit.path), "utf8")).toBe(f.raw);
    const importedState = JSON.parse(await readFile(join(importedEngine, "brain.json"), "utf8"));
    expect(importedState.app_observation_frontier.content.audit.sourceBrainId).toBe(brain.id);
    expect(importedState.app_observation_frontier).not.toHaveProperty("sessionId");
    await repository.restoreSnapshot(imported.id, point.id);
    expect(await readFile(join(importedEngine, f.content.audit.path), "utf8")).toBe(f.raw);
    const missing = { ...entries }; delete missing[`app-audits/current/${f.content.audit.path}`]; refreshArchiveIntegrity(missing);
    const incomplete = join(temporaryRoot, "missing-app-audit.omni"); await writeFile(incomplete, zipSync(missing));
    await expect(repository.importBundle(incomplete)).rejects.toThrow(/ENOENT|missing|App audit/);
  });

  it.each([
    { mode: "current", scope: "current", leak: "credential" },
    { mode: "origin", scope: "origin", leak: "private path" },
    { mode: "private-archive", scope: "current", leak: "private path" },
    { mode: "referenced", scope: "origin", leak: "credential" }
  ] as const)(
    "preserves $scope substrate $leak content unchanged in $mode export",
    async ({ mode, scope, leak }) => {
      const brain = await repository.create({
        ...DEFAULT_CONFIG,
        name: "Substrate privacy fixture"
      });
      const leakValue = leak === "credential"
        ? `sk-${"q".repeat(36)}`
        : "C:\\Users\\fixture\\private\\neuron-label.txt";
      await materializeSubstrateExportFixture(repository, brain, { [scope]: leakValue });
      const destination = join(temporaryRoot, `unsanitized-${mode}-${scope}.omni`);
      await repository.exportBundle(brain.id, destination, mode);
      const entries = unzipSync(new Uint8Array(await readFile(destination)));
      const archiveScope = mode === "origin" ? "current" : scope;
      const records = Object.entries(entries).filter(([name]) =>
        name.startsWith(`substrate/${archiveScope}/blobs/`) && name.endsWith(".json")
      );
      expect(records.some(([, bytes]) => JSON.parse(strFromU8(bytes)).records[0]?.source_label === leakValue)).toBe(true);
      const imported = await repository.importBundle(destination);
      expect(imported.name).toBe(brain.name);
    }
  );

  it("carries plugin portability through current export, external import, and trusted fork only", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Plugin portability host fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const archive = { manifest: { fixture: true }, packages: new Map() } as never;
    const cloneTrusted = vi.fn(async () => undefined);
    const adoptExternal = vi.fn(async () => undefined);
    const removeBrain = vi.fn(async () => undefined);
    const validateImport = vi.fn(async (
      files: Map<string, string>,
      _declaration: Parameters<BrainPluginPortabilityHooks["validateImport"]>[1],
      sourceBrainId: string
    ) => {
      expect(sourceBrainId).toBe(brain.id);
      expect(await readFile(files.get("manifest.json")!, "utf8")).toBe('{"fixture":true}');
      return archive;
    });
    const exportBrain = vi.fn(async (
      _brainId: string,
      mode: Parameters<BrainPluginPortabilityHooks["exportBrain"]>[1],
      destination: string
    ) => {
      if (mode === "origin") return undefined;
      await mkdir(join(destination, "packages"), { recursive: true });
      const path = join(destination, "manifest.json");
      const bytes = Buffer.from('{"fixture":true}');
      const packageBytes = Buffer.from('{"format":"omni-plugin-package","fixture":true}');
      const packageHash = digest(packageBytes);
      const packagePath = join(destination, "packages", `${packageHash}.json`);
      await Promise.all([writeFile(path, bytes), writeFile(packagePath, packageBytes)]);
      return {
        declaration: {
          formatVersion: 1 as const,
          selectionCount: 1,
          packageCount: 1,
          manifestSha256: digest(bytes)
        },
        files: new Map([
          ["manifest.json", path],
          [`packages/${packageHash}.json`, packagePath]
        ])
      };
    });
    repository.configurePluginPortability({
      cloneTrusted,
      exportBrain,
      validateImport,
      adoptExternal,
      removeBrain
    });

    const fork = await repository.duplicate(brain.id);
    expect(cloneTrusted).toHaveBeenCalledWith(brain.id, fork.id);

    const currentPath = join(temporaryRoot, "plugin-current.omni");
    await repository.exportBundle(brain.id, currentPath, "current");
    const currentEntries = unzipSync(new Uint8Array(await readFile(currentPath)));
    expect(strFromU8(currentEntries["plugins/manifest.json"]!)).toBe('{"fixture":true}');
    expect(Object.keys(currentEntries).some((name) => name.startsWith("plugins/packages/"))).toBe(true);
    expect(JSON.parse(strFromU8(currentEntries["manifest.json"]!))).toMatchObject({
      pluginPortability: { formatVersion: 1, selectionCount: 1, packageCount: 1 }
    });
    const imported = await repository.importBundle(currentPath);
    expect(validateImport).toHaveBeenCalledOnce();
    expect(adoptExternal).toHaveBeenCalledWith(imported.id, archive);

    const originPath = join(temporaryRoot, "plugin-origin.omni");
    await repository.exportBundle(brain.id, originPath, "origin");
    const originEntries = unzipSync(new Uint8Array(await readFile(originPath)));
    expect(Object.keys(originEntries).some((name) => name.startsWith("plugins/"))).toBe(false);
    expect(JSON.parse(strFromU8(originEntries["manifest.json"]!))).not.toHaveProperty("pluginPortability");
  });

  it("preserves exact bound task attachments through duplicate, recovery, export and import", async () => {
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Task attachment custody fixture" });
    await materializeNativeEngineFixture(repository, brain);
    const attachments = new TaskAttachmentService(repository);
    const source = join(temporaryRoot, "private-unsanitized-task-note.txt");
    await writeFile(source, "exact attached task bytes");
    const [staged] = await attachments.stagePaths(brain.id, "repository-attachment-stage", [{ path: source }], "files");
    await attachments.bindToTurn(brain.id, "repository-attachment-turn", [staged!.id]);
    await repository.appendReceivedChatInput(brain.id, {
      id: "repository-attachment-human",
      role: "human",
      content: "read the attached task note",
      createdAt: "2026-10-11T00:00:00.000Z",
      turnId: "repository-attachment-turn",
      taskAttachmentIds: [staged!.id],
      inputReceipt: {
        committed: true,
        turnId: "repository-attachment-turn",
        inputSha256: digest("read the attached task note"),
        humanMessageId: "repository-attachment-human",
        afterimageId: "repository-attachment-afterimage",
        createdAt: "2026-10-11T00:00:00.000Z",
        attentionEpoch: 0,
        taskAttachmentIds: [staged!.id]
      }
    });
    const teacherRoot = repository.brainDirectory(brain.id), teacherId = "saved-teacher-custody", teacherPrompt = "Preserve this pending teacher task as inert history.";
    const teacher = await ApiTeacherStore.open(teacherRoot, brain.id, async () => undefined);
    if (!teacher) throw new Error("Inert teacher fixture unavailable.");
    try {
      const now = "2026-10-11T00:00:00.000Z";
      await teacher.createJob({ format: "omni-api-teacher-job", version: 1, id: teacherId, brainId: brain.id,
        originalBrainId: brain.id, originalJobId: teacherId, requestId: "saved-teacher-request", requestSha256: digest("saved-teacher-request"),
        provider: "openai", model: "inert-fixture-model", maxOutputTokens: 32, promptCount: 1, learned: 0, failed: 0,
        skipped: 0, nextIndex: 0, state: "paused", revision: 1, createdAt: now, updatedAt: now },
      [{ format: "omni-api-teacher-item", version: 1, jobId: teacherId, index: 0, prompt: teacherPrompt,
        promptSha256: digest(teacherPrompt), deliveryId: "saved-teacher-delivery", state: "pending" }]);
    } finally { teacher.close(); }
    const point = await repository.snapshot(brain.id, "Bound task attachment");

    const fork = await repository.duplicate(brain.id);
    const forkStore = new TaskAttachmentStore(repository.brainDirectory(fork.id), fork.id);
    await expect(forkStore.readForTurn("repository-attachment-turn", staged!.id)).resolves.toMatchObject({
      content: "exact attached task bytes",
      attachment: { brainId: fork.id, state: "bound" }
    });

    const later = join(temporaryRoot, "later-unsent.txt");
    await writeFile(later, "not in selected recovery point");
    await attachments.stagePaths(brain.id, "later-attachment-stage", [{ path: later }], "files");
    const changedTeacher = await ApiTeacherStore.open(teacherRoot, brain.id, async () => undefined);
    try { await changedTeacher!.updateJob(teacherId, { state: "stopped", error: "later state after selected recovery" }, "fixture-later-update"); }
    finally { changedTeacher?.close(); }
    await repository.restoreSnapshot(brain.id, point.id);
    const restoredStore = new TaskAttachmentStore(repository.brainDirectory(brain.id), brain.id);
    expect((await restoredStore.list()).map((item) => item.id)).toEqual([staged!.id]);
    for (const absent of ["clipboard", "environments", "local-teachers"]) {
      await expect(stat(join(teacherRoot, absent))).rejects.toMatchObject({ code: "ENOENT" });
    }
    const recoveredTeacher = await ApiTeacherStore.open(teacherRoot, brain.id, async () => undefined, { create: false });
    try {
      expect(recoveredTeacher).toBeDefined();
      const retained = recoveredTeacher!.jobs().filter(job => job.originalJobId === teacherId);
      expect(recoveredTeacher!.job(teacherId)).toMatchObject({ state: "paused", revision: 1 });
      expect(recoveredTeacher!.items(teacherId)[0]).toMatchObject({ prompt: teacherPrompt, state: "pending" });
      expect(retained.some(job => job.state === "dormant")).toBe(true);
      expect(retained.some(job => recoveredTeacher!.items(job.id).some(item => item.prompt === teacherPrompt))).toBe(true);
      let exactLaterCapture = false;
      for (const lineage of recoveredTeacher!.lineages()) {
        const archive = new DatabaseSync(join(teacherRoot, ...lineage.relativePath.split("/")), { readOnly: true });
        try {
          const job = archive.prepare("SELECT payload FROM jobs WHERE id=?").get(teacherId);
          if (job && JSON.parse(String(job.payload)).state === "stopped" && JSON.parse(String(job.payload)).revision === 2) exactLaterCapture = true;
        } finally { archive.close(); }
      }
      expect(exactLaterCapture).toBe(true);
    } finally { recoveredTeacher?.close(); }

    const bundle = join(temporaryRoot, "task-attachment.omni");
    await repository.exportBundle(brain.id, bundle, "current");
    const imported = await repository.importBundle(bundle);
    const importedStore = new TaskAttachmentStore(repository.brainDirectory(imported.id), imported.id);
    await expect(importedStore.readForTurn("repository-attachment-turn", staged!.id)).resolves.toMatchObject({
      content: "exact attached task bytes",
      attachment: { brainId: imported.id, state: "bound", parametersChanged: false }
    });
    expect((await repository.conversationPage(imported.id)).entries).toEqual(expect.arrayContaining([
      expect.objectContaining({ message: expect.objectContaining({
        id: "repository-attachment-human",
        taskAttachmentIds: [staged!.id],
        inputReceipt: expect.objectContaining({ taskAttachmentIds: [staged!.id] })
      }) })
    ]));
    const registry = new DatabaseSync(TaskAttachmentStore.databasePath(repository.brainDirectory(imported.id)), { readOnly: true });
    try {
      expect(String(registry.prepare("SELECT source_json FROM attachments WHERE id=?").get(staged!.id)?.source_json))
        .toContain(source);
    } finally { registry.close(); }
  }, 20_000);

  it("retains credential-like annotations in a hash-bound substrate manifest", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Substrate manifest privacy fixture"
    });
    const fakeKey = `sk-${"m".repeat(36)}`;
    await materializeSubstrateExportFixture(repository, brain, {
      originManifest: fakeKey
    });
    const destination = join(temporaryRoot, "unsanitized-manifest.omni");
    await repository.exportBundle(brain.id, destination, "current");
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    const pointer = JSON.parse(strFromU8(entries["substrate/origin/manifest.json"]!));
    const bytes = entries[`substrate/origin/${pointer.generationManifest}`]!;
    expect(digest(bytes)).toBe(pointer.generationManifestSha256);
    expect(JSON.parse(strFromU8(bytes)).source_note).toBe(fakeKey);
  });

  it("does not redact a large saved substrate record beyond the initial file sample", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Large substrate privacy fixture"
    });
    const fakeKey = `sk-${"l".repeat(36)}`;
    await materializeSubstrateExportFixture(repository, brain, {
      current: `${"ordinary label ".repeat(20_000)}${fakeKey}`
    });
    const destination = join(temporaryRoot, "unsanitized-late-key.omni");
    await repository.exportBundle(brain.id, destination, "current");
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    expect(Object.entries(entries).some(([name, bytes]) =>
      name.startsWith("substrate/current/blobs/") && Buffer.from(bytes).includes(Buffer.from(fakeKey))
    )).toBe(true);
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

  it("exports checksum-bound packed-v3 native substrate generations", async () => {
    const brain = await repository.create({
      ...DEFAULT_CONFIG,
      name: "Packed-v3 portable fixture"
    });
    await materializeSubstrateExportFixture(repository, brain, {}, 3);
    const destination = join(temporaryRoot, "packed-v3.omni");
    await repository.exportBundle(brain.id, destination, "current");
    const entries = unzipSync(new Uint8Array(await readFile(destination)));
    for (const scope of ["current", "origin"] as const) {
      const prefix = `substrate/${scope}`;
      const pointer = JSON.parse(strFromU8(entries[`${prefix}/manifest.json`]!)) as {
        formatVersion: number;
        generationManifest: string;
      };
      expect(pointer.formatVersion).toBe(3);
      const generation = JSON.parse(
        strFromU8(entries[`${prefix}/${pointer.generationManifest}`]!)
      ) as {
        formatVersion: number;
        schema: string;
        shards: Array<{ kind: string; tensors: { path: string } | null }>;
      };
      expect(generation.formatVersion).toBe(3);
      expect(generation.schema).toBe("neural-substrate-2");
      const neuronShard = generation.shards.find((shard) => shard.kind === "neurons")!;
      expect(neuronShard.tensors).not.toBeNull();
      expect(entries[`${prefix}/${neuronShard.tensors!.path}`]).toBeDefined();
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

    await saveCatalogFixture(engine);
    await saveCatalogFixture(join(engine, "origin"));
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
    // Saved mutable generations/catalogs remain complete in either mode.
    // Referenced mirrors are smaller, but unrelated retained state does not
    // imply an arbitrary fixed archive-size reduction ratio.
    expect(referenceBytes.byteLength).toBeLessThan(portableBytes.byteLength);
    const referenceEntries = unzipSync(new Uint8Array(referenceBytes));
    const portableEntries = unzipSync(new Uint8Array(portableBytes));
    const referenceManifest = JSON.parse(strFromU8(referenceEntries["manifest.json"]!)) as {
      mode: string;
      references: Record<string, string>;
    };
    expect(referenceManifest.mode).toBe("referenced-local");
    for (const archiveEntries of [portableEntries, referenceEntries]) {
      expect(archiveEntries["conversation/ledger.sqlite3"]).toBeDefined();
      expect(archiveEntries["conversation/neural-ledger.sqlite3"]).toBeUndefined();
      const exposedText = Object.values(archiveEntries)
        .map((value) => Buffer.from(value).toString("utf8"))
        .join("\n");
      expect(exposedText).toContain(historySecret);
      expect(exposedText).toContain(historyPath);
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
