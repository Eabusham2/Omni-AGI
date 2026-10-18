import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { nativeCheckpointMemory } from "../src/main/nativeCheckpointMemory";
import { ResourcePlanner, GIB } from "../src/main/resourcePlanner";

function safetensors(entries: Record<string, { dtype: string; shape: number[] }>): Buffer {
  const header: Record<string, unknown> = {};
  let offset = 0;
  for (const [name, value] of Object.entries(entries)) {
    const bytes = value.shape.reduce((a, b) => a * b, 1) * (value.dtype === "U8" ? 1 : 4);
    header[name] = { ...value, data_offsets: [offset, offset + bytes] };
    offset += bytes;
  }
  const json = JSON.stringify(header);
  const encoded = Buffer.from(json.padEnd(Math.ceil(json.length / 8) * 8, " "));
  const prefix = Buffer.alloc(8); prefix.writeBigUInt64LE(BigInt(encoded.length));
  return Buffer.concat([prefix, encoded, Buffer.alloc(offset)]);
}

describe("native checkpoint strength migration memory", () => {
  let root: string;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-native-memory-")); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });

  it("counts existing strength bytes once and reserves missing native state without reading weight payloads", async () => {
    const path = join(root, "core.safetensors");
    const old = safetensors({ "decoder.a._packed_forward_weight": { dtype: "U8", shape: [3, 2] },
      "decoder.a._packed_forward_bias": { dtype: "U8", shape: [1, 1] } });
    await writeFile(path, old);
    const before = await nativeCheckpointMemory([path]);
    expect(before).toEqual({ checkpointBytes: old.length, connectionStateBytes: 0,
      unitStrengthMigrationReserveBytes: 28, nativeControlMigrationReserveBytes: 8,
      runtimeMetadataReserveBytes: 16 * 1024, projectedCheckpointBytes: old.length + 36 });
    expect(await readFile(path)).toEqual(old);
    const current = safetensors({ "decoder.a._packed_forward_weight": { dtype: "U8", shape: [3, 2] },
      "decoder.a._packed_forward_bias": { dtype: "U8", shape: [1, 1] },
      "decoder.a._connection_state": { dtype: "U8", shape: [3, 5] },
      "decoder.a._bias_connection_state": { dtype: "U8", shape: [1, 3] },
      "decoder.a.parallel_connections._next_page": { dtype: "U8", shape: [8] } });
    await writeFile(path, current);
    expect(await nativeCheckpointMemory([path])).toEqual({ checkpointBytes: current.length,
      connectionStateBytes: 18, unitStrengthMigrationReserveBytes: 0, nativeControlMigrationReserveBytes: 0,
      runtimeMetadataReserveBytes: 16 * 1024, projectedCheckpointBytes: current.length });
  });

  it("counts sparse router and repeated contact state, and refuses linked or invalid headers", async () => {
    const path = join(root, "plasticity.safetensors");
    await writeFile(path, safetensors({ "router.blocks.r0_c0_p1._packed_weights": { dtype: "U8", shape: [2, 1] },
      "router.blocks.r0_c0_p1._connection_state": { dtype: "U8", shape: [2, 2] },
      "decoder.parallel_connections.w0_c0_p1._packed_sign": { dtype: "U8", shape: [1, 1] },
      "decoder.parallel_connections.w0_c0_p1._connection_state": { dtype: "U8", shape: [1, 1] } }));
    expect(await nativeCheckpointMemory([path])).toMatchObject({ connectionStateBytes: 5, unitStrengthMigrationReserveBytes: 0 });
    const linked = join(root, "linked.safetensors");
    await symlink(path, linked);
    await expect(nativeCheckpointMemory([linked])).rejects.toThrow("regular");
    const malformed = Buffer.alloc(16); malformed.writeBigUInt64LE(100000n);
    await writeFile(path, malformed);
    await expect(nativeCheckpointMemory([path])).rejects.toThrow("header size");
  });

  it("reserves migration for a loaded instance without rewriting its archived descriptor or file", async () => {
    const engine = join(root, "engine"); await mkdir(engine);
    const core = safetensors({ "decoder.a._packed_forward_weight": { dtype: "U8", shape: [3, 2] } });
    const plasticity = safetensors({});
    await writeFile(join(engine, "core.safetensors"), core);
    await writeFile(join(engine, "plasticity.safetensors"), plasticity);
    await writeFile(join(engine, "brain.json"), JSON.stringify({ config: { hardware_tier: "micro", d_model: 32, n_layers: 1, working_memory_slots: 64 } }));
    const planner = new ResourcePlanner(root, {
      readResources: async () => ({ totalMemoryBytes: 16 * GIB, availableMemoryBytes: 14 * GIB, diskTotalBytes: 1024 * GIB, diskFreeBytes: 800 * GIB }),
      benchmark: async () => ({ measuredAt: new Date().toISOString(), sampleBytes: 1024, memoryBytesPerSecond: GIB, storageBytesPerSecond: GIB, cacheHit: true })
    });
    const plan = await planner.plan({ mode: "auto" }, { brainDirectory: root, admissionScope: "existing-runtime" });
    expect(plan.resources.modelBytes).toBe(core.length + plasticity.length + 32);
    expect(plan.resources.checkpointHeadroomBytes).toBe(Math.ceil((core.length + plasticity.length + 32) * 0.2));
    expect(plan.resources.modelMetadataReserveBytes).toBe(16 * 1024);
    expect(await readFile(join(engine, "core.safetensors"))).toEqual(core);
  });

  it("keeps pooled registry and derived sort-cache RAM separate from persisted tensor bytes", async () => {
    const path = join(root, "pooled.safetensors");
    const body = safetensors({
      "decoder.layer._packed_forward_weight": { dtype: "U8", shape: [1, 1] },
      "decoder.layer._connection_state": { dtype: "U8", shape: [1, 1] },
      "decoder.layer.parallel_connections._next_page": { dtype: "U8", shape: [8] },
      "decoder.layer.parallel_connections.pages.w_0_0._packed_sign": { dtype: "U8", shape: [1, 256] },
      "decoder.layer.parallel_connections.pages.w_0_0._connection_state": { dtype: "U8", shape: [1, 1024] },
      "decoder.layer.parallel_connections.pages.w_0_0._record_meta": { dtype: "U8", shape: [1024, 10] },
      "router.synapses.blocks.r0_c0_p1._packed_weights": { dtype: "U8", shape: [64, 16] },
      "router.synapses.blocks.r0_c0_p1._connection_state": { dtype: "U8", shape: [64, 64] }
    });
    await writeFile(path, body);
    expect(await nativeCheckpointMemory([path])).toMatchObject({
      connectionStateBytes: 5121, checkpointBytes: body.length, projectedCheckpointBytes: body.length,
      nativeControlMigrationReserveBytes: 0, unitStrengthMigrationReserveBytes: 0,
      runtimeMetadataReserveBytes: 16384 + 8192 + 2048 + 8192
    });
  });
});
