import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it } from "vitest";
import { connectedResourceProfile } from "../src/main/connectedResourceProfile";
import type { SavedTensorShape } from "../src/main/nativeCheckpointMemory";
import { GIB, planWorkingMemory, ResourcePlanner } from "../src/main/resourcePlanner";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
const canonical = (value: unknown): string => Array.isArray(value) ? `[${value.map(canonical).join(",")}]`
  : value && typeof value === "object" ? `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, v]) => `${JSON.stringify(key)}:${canonical(v)}`).join(",")}}` : JSON.stringify(value);
const digest = (value: string) => createHash("sha256").update(value).digest("hex");
const HASH = "1".repeat(64);
const config = (width: number) => ({ d_model: width, n_layers: 1, router_neurons: width, working_memory_slots: 16, hardware_tier: "micro" });
type Fixture = { config: ReturnType<typeof config>; connected_regions?: unknown; connected_layout?: unknown };
async function connected(root: string, name: string, children: Array<{ metadata: Fixture; layers: number }>): Promise<Fixture> {
  const regions: Array<Record<string, unknown>> = [], rows: Array<Record<string, unknown>> = [];
  for (const [index, child] of children.entries()) {
    const enginePath = `merge/${name}-${index}/engine`, directory = join(root, enginePath);
    await mkdir(directory, { recursive: true });
    const text = canonical(child.metadata);
    await writeFile(join(directory, "brain.json"), text);
    await writeFile(join(directory, "core.safetensors"), "native fixture");
    await writeFile(join(directory, "plasticity.safetensors"), "native fixture");
    regions.push({ regionId: `region-${index}`, parentId: `parent-${index}`, enginePath, config: child.metadata.config,
      metadataSha256: digest(text), coreSha256: HASH, plasticitySha256: HASH, generationSha256: HASH, neuralSha256: HASH });
    rows.push({ regionId: `region-${index}`, layers: child.layers, experts: 0, ...(child.metadata.connected_layout ? { nested: child.metadata.connected_layout } : {}) });
  }
  const descriptor = { format: "omni-connected-native-regions", formatVersion: 1, regions,
    parents: Array.from({ length: Math.max(2, regions.length) }, (_, i) => ({ parentId: `alias-${i}`, regionId: regions[i % regions.length]!.regionId, generationSha256: HASH })),
    deduplication: "identical-complete-neural-state-only" };
  return { config: config(children.reduce((sum, child) => sum + child.metadata.config.d_model, 0)),
    connected_regions: { ...descriptor, sha256: digest(canonical(descriptor)) },
    connected_layout: { format: "omni-connected-layout", formatVersion: 1, regions: rows } };
}
const tensors = (leaves: Array<{ prefix: string; width: number; layers: number }>) => {
  const result = new Map<string, SavedTensorShape>();
  const add = (name: string, rows: number, width: number) => {
    result.set(name + "._packed_forward_weight", { dtype: "U8", shape: [rows, Math.ceil(width / 4)], bytes: rows * Math.ceil(width / 4) });
    result.set(name + "._connection_state", { dtype: "U8", shape: [rows, width], bytes: rows * width });
  };
  for (const leaf of leaves) {
    for (let i = 0; i < leaf.layers; i += 1) add(`${leaf.prefix}blocks.${i}.attention.qkv`, leaf.width * 3, leaf.width);
    add(`${leaf.prefix}global_workspace.latent_table`, 4, leaf.width);
  }
  return result;
};
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-connected-budget-")); roots.push(root);
  await mkdir(join(root, "engine"));
  return root;
}

async function saveTensorFixture(path: string, values: ReadonlyMap<string, SavedTensorShape>): Promise<void> {
  let offset = 0; const header: Record<string, unknown> = {};
  for (const [name, tensor] of values) { header[name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.bytes] }; offset += tensor.bytes; }
  const body = Buffer.from(JSON.stringify(header)), prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(body.length));
  await writeFile(path, Buffer.concat([prefix, body, Buffer.alloc(offset)]));
}

it("uses grown regional depths and all separate workspace tables, counting parent aliases only once", async () => {
  const root = await fixture();
  const saved = await connected(root, "pair", [{ metadata: { config: config(8) }, layers: 3 }]);
  await writeFile(join(root, "engine/brain.json"), canonical(saved));
  const result = await connectedResourceProfile(join(root, "engine"), tensors([{ prefix: "decoder.regions.0.decoder.", width: 8, layers: 3 }]));
  expect(result.leaves).toHaveLength(1);
  expect(result.kvBytesPerToken).toBe(8 * 3 * 8);
  expect(result.workspaceStateBytes).toBe(4 * 4 * 17);
});

it("recurses nested shapes without multiplying a composite width by its summed layers", async () => {
  const root = await fixture();
  const parent = join(root, "merge/top-0"); await mkdir(parent, { recursive: true });
  const nested = await connected(parent, "nested", [{ metadata: { config: config(8) }, layers: 2 }, { metadata: { config: config(12) }, layers: 1 }]);
  const saved = await connected(root, "top", [{ metadata: nested, layers: 3 }, { metadata: { config: config(4) }, layers: 3 }]);
  await writeFile(join(root, "engine/brain.json"), canonical(saved));
  const result = await connectedResourceProfile(join(root, "engine"), tensors([
    { prefix: "decoder.regions.0.decoder.regions.0.decoder.", width: 8, layers: 2 },
    { prefix: "decoder.regions.0.decoder.regions.1.decoder.", width: 12, layers: 1 },
    { prefix: "decoder.regions.1.decoder.", width: 4, layers: 3 }
  ]));
  expect(result.attentionLayers).toBe(6);
  expect(result.kvBytesPerToken).toBe((8 * 2 + 12 + 4 * 3) * 8);
  const plan = planWorkingMemory({ request: { mode: "auto" }, resources: { totalMemoryBytes: 16 * GIB, availableMemoryBytes: 12 * GIB,
    diskFreeBytes: 100 * GIB, diskTotalBytes: 500 * GIB }, benchmark: { measuredAt: "2026-10-07T00:00:00Z", sampleBytes: 1,
    memoryBytesPerSecond: GIB, storageBytesPerSecond: GIB, cacheHit: true }, hardwareTier: "micro", modelBytes: 4096,
    modelHiddenSize: 24, modelLayers: 3, modelContextGeometry: result });
  expect(plan.context.evidence.estimatedKvActivationBytesPerToken).toBe(320);
  expect(plan.context.evidence.modelLayers).toBe(6);
});

it("rejects missing grown blocks, undeclared attention owners and changed provenance", async () => {
  const root = await fixture();
  const saved = await connected(root, "pair", [{ metadata: { config: config(8) }, layers: 2 }]);
  await writeFile(join(root, "engine/brain.json"), canonical(saved));
  const short = tensors([{ prefix: "decoder.regions.0.decoder.", width: 8, layers: 1 }]);
  await expect(connectedResourceProfile(join(root, "engine"), short)).rejects.toThrow(/geometry/);
  const extra = tensors([{ prefix: "decoder.regions.0.decoder.", width: 8, layers: 3 }]);
  await expect(connectedResourceProfile(join(root, "engine"), extra)).rejects.toThrow(/absent/);
  await writeFile(join(root, "merge/pair-0/engine/brain.json"), canonical({ config: config(16) }));
  await expect(connectedResourceProfile(join(root, "engine"), extra)).rejects.toThrow(/checksum/);
});

it("rejects descriptor tampering instead of falling back to the root stack", async () => {
  const root = await fixture(), saved = await connected(root, "pair", [{ metadata: { config: config(8) }, layers: 1 }]);
  await writeFile(join(root, "engine/brain.json"), canonical(saved));
  const text = (await readFile(join(root, "engine/brain.json"), "utf8")).replace('"deduplication":"identical-complete-neural-state-only"', '"deduplication":"changed"');
  await writeFile(join(root, "engine/brain.json"), text);
  await expect(connectedResourceProfile(join(root, "engine"), tensors([]))).rejects.toThrow(/descriptor checksum/);
});

it("checks native float lexemes and connects real header-only geometry to saved-instance preflight", async () => {
  const root = await fixture(), saved = await connected(root, "pair", [{ metadata: { config: config(8) }, layers: 2 }]);
  const descriptor = saved.connected_regions as Record<string, unknown>;
  descriptor.floatFixture = 1;
  const { sha256: _oldHash, ...body } = descriptor;
  descriptor.sha256 = digest(canonical(body).replace('"floatFixture":1', '"floatFixture":1.0'));
  await writeFile(join(root, "engine/brain.json"), canonical(saved).replace('"floatFixture":1', '"floatFixture":1.0'));
  const inventory = tensors([{ prefix: "decoder.regions.0.decoder.", width: 8, layers: 2 }]);
  await saveTensorFixture(join(root, "engine/core.safetensors"), inventory);
  await saveTensorFixture(join(root, "engine/plasticity.safetensors"), new Map());
  const planner = new ResourcePlanner(root, {
    readResources: async () => ({ totalMemoryBytes: 16 * GIB, availableMemoryBytes: 12 * GIB, diskTotalBytes: 500 * GIB, diskFreeBytes: 100 * GIB }),
    benchmark: async () => ({ measuredAt: "2026-10-07T00:00:00Z", sampleBytes: 1, memoryBytesPerSecond: GIB, storageBytesPerSecond: GIB, cacheHit: true })
  });
  const plan = await planner.plan({ mode: "auto" }, { brainDirectory: root, admissionScope: "existing-runtime" });
  expect(plan.context.evidence.estimatedKvActivationBytesPerToken).toBe(128);
  expect(plan.context.evidence.modelLayers).toBe(2);
  await rm(join(root, "merge/pair-0/engine/brain.json"));
  await expect(planner.plan({ mode: "auto" }, { brainDirectory: root })).rejects.toThrow();
});
