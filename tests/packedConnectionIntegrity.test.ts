import { createHash } from "node:crypto";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { verifyPackedConnectionCompanions, type VerifiedPackedShard } from "../src/main/packedConnectionIntegrity";

const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });

async function fixture(count = 1) {
  const root = await mkdtemp(join(tmpdir(), "omni-connection-integrity-")); roots.push(root);
  const state = Buffer.alloc(6);
  const contacts = Buffer.alloc(count * 26);
  for (let i = 0; i < count; ++i) {
    contacts.writeBigUInt64LE(BigInt(i), i * 26 + 16);
    contacts[i * 26 + 24] = 0x56;
  }
  const shards = new Map<string, VerifiedPackedShard>();
  for (const [name, bytes] of [["connections-00000-fixture.bin", state], ["parallel-00000-fixture.bin", contacts]] as const) {
    const path = join(root, name); await writeFile(path, bytes);
    shards.set(name, { path, byteLength: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") });
  }
  const connection = shards.get("connections-00000-fixture.bin")!;
  const parallel = shards.get("parallel-00000-fixture.bin")!;
  const tensor = { kind: "projection", shape: [2, 3], numel: 6,
    connectionState: { formatVersion: 1, encoding: "uint4-strength-minus-one-high4-resistance", shape: [2, 3],
      shard: "connections-00000-fixture.bin", byteLength: 6, sha256: connection.sha256 },
    parallelConnections: { encoding: "u64-row-column-ordinal-u8-sign-state-le", count,
      shard: "parallel-00000-fixture.bin", byteLength: contacts.length, sha256: parallel.sha256 }
  };
  return { tensor, shards, contacts };
}

describe("native weighted companion validation", () => {
  it("streams records across filesystem chunk boundaries", async () => {
    const { tensor, shards } = await fixture(3000);
    const used = new Set<string>();
    await verifyPackedConnectionCompanions(tensor, 2, shards, used, "fixture");
    expect(used.size).toBe(2);
  });
  it("rejects equal-size but incorrectly oriented strength matrices", async () => {
    const { tensor, shards } = await fixture();
    tensor.connectionState.shape = [3, 2];
    await expect(verifyPackedConnectionCompanions(tensor, 2, shards, new Set(), "fixture")).rejects.toThrow("strength rows");
  });
  it("validates explicitly declared transposed convolution order", async () => {
    const { tensor, shards } = await fixture();
    const transposed = { ...tensor, shape: [3, 1, 2], packedRowLayout: { kind: "transpose-convolution", groups: 1 } };
    await expect(verifyPackedConnectionCompanions(transposed, 2, shards, new Set(), "fixture")).resolves.toBeUndefined();
    await expect(verifyPackedConnectionCompanions({ ...transposed, packedRowLayout: { kind: "transpose-convolution", groups: 2 } },
      2, shards, new Set(), "fixture")).rejects.toThrow("row layout");
  });
  it("rejects duplicates and endpoints outside the saved matrix", async () => {
    const { tensor, shards, contacts } = await fixture(2);
    contacts.writeBigUInt64LE(0n, 26 + 16);
    await writeFile(shards.get(tensor.parallelConnections.shard)!.path, contacts);
    await expect(verifyPackedConnectionCompanions(tensor, 2, shards, new Set(), "fixture")).rejects.toThrow("endpoint, order or sign");
    contacts.writeBigUInt64LE(99n, 0);
    await writeFile(shards.get(tensor.parallelConnections.shard)!.path, contacts);
    await expect(verifyPackedConnectionCompanions(tensor, 2, shards, new Set(), "fixture")).rejects.toThrow("endpoint, order or sign");
  });
  it("requires v2 state and forbids undeclared contacts in native v1", async () => {
    await expect(verifyPackedConnectionCompanions({ shape: [1], numel: 1 }, 2, new Map(), new Set(), "fixture")).rejects.toThrow("encoding");
    await expect(verifyPackedConnectionCompanions({ parallelConnections: {} }, 1, new Map(), new Set(), "fixture")).rejects.toThrow("explicit connection");
    await expect(verifyPackedConnectionCompanions({}, 1, new Map(), new Set(), "fixture")).resolves.toBeUndefined();
  });
});
