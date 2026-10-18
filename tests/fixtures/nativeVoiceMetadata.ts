/** Synthetic committed metadata/manifests only. Never constructs/trains a brain. */
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

export const canonicalVoiceFixture = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(canonicalVoiceFixture).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
    .map(([key, item]) => `${JSON.stringify(key)}:${canonicalVoiceFixture(item)}`).join(",")}}`;
  return JSON.stringify(value);
};
const hash = (value: unknown): string => createHash("sha256").update(canonicalVoiceFixture(value)).digest("hex");

export async function nativeVoiceMetadata(root: string, total: number, options: { brainId?: string; connected?: boolean } = {}) {
  const brainId = options.brainId ?? "voice-brain", graphSynapses = 5, sparse = options.connected ? 25 : graphSynapses;
  const counts = { neurons: 0, assemblies: 0, synapses: graphSynapses };
  const substrateBody = { format: "omni-substrate-shards", formatVersion: 3, schema: "neural-substrate-2", dimensions: 16, seed: 7,
    counts, recordsPerShard: 512, shards: [{ kind: "synapses", bucket: "a", part: 0, count: graphSynapses }] };
  const substrate = { ...substrateBody, contentSha256: hash(substrateBody) };
  const substratePointer = { format: substrate.format, formatVersion: 3, activeGeneration: substrate.contentSha256, contentSha256: substrate.contentSha256,
    generationManifest: `generations/${substrate.contentSha256}/manifest.json`, generationManifestSha256: hash(substrate), counts, shardCount: 1 };
  const blob = { path: `blobs/${"a".repeat(64)}.safetensors`, sha256: "a".repeat(64), bytes: 0 };
  const mutableBody = { format: "omni-mutable-state", formatVersion: 1, brainId, roles: { core: blob, plasticity: blob, optimizer: blob },
    replay: { format: "omni-replay-sqlite", formatVersion: 1, path: "replay.sqlite3", count: 0, highWaterId: 0, contentSha256: "0".repeat(64) },
    metadata: { parameterChecksum: "b".repeat(64), substrateContentSha256: substrate.contentSha256 } };
  const mutable = { ...mutableBody, contentSha256: hash(mutableBody) };
  const mutablePointer = { format: mutable.format, formatVersion: 1, activeGeneration: mutable.contentSha256, contentSha256: mutable.contentSha256,
    generationManifest: `generations/${mutable.contentSha256}/manifest.json`, generationManifestSha256: hash(mutable) };
  const metadata = { brain_id: brainId, release_format: "stable-1.0", config: { origin_kind: "ground-up", advertised_size: "999T" },
    ...(options.connected ? { connected_regions: { fixture: true } } : {}),
    substrate: { persistence: substratePointer }, mutable_state: mutablePointer,
    runtime_card: { workspace: { brainId, queriedAt: "2026-10-11T10:00:00Z" }, parameterAccounting: {
      mutableDenseParameters: total - sparse, substrateVectorParameters: 0, dynamicSparseSynapses: sparse,
      substrateDynamicSparseSynapses: sparse, totalNeuralParameters: total, countingRule: "distinct-native-logical-parameters" } } };
  const engine = join(root, "engine");
  for (const [store, pointer, generation] of [["substrate", substratePointer, substrate], ["state", mutablePointer, mutable]] as const) {
    const directory = join(engine, store, "generations", pointer.activeGeneration); await mkdir(directory, { recursive: true });
    await writeFile(join(directory, "manifest.json"), canonicalVoiceFixture(generation));
    await writeFile(join(engine, store, "manifest.json"), canonicalVoiceFixture(pointer));
  }
  await writeFile(join(engine, "brain.json"), canonicalVoiceFixture(metadata));
  return { engine, metadata, substrate, mutable, substratePointer, mutablePointer };
}
