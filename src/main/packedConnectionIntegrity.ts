import { createReadStream } from "node:fs";

export interface VerifiedPackedShard {
  path: string;
  sha256: string;
  byteLength: number;
}

const record = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);
const count = (value: unknown): value is number =>
  typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

/** Validate v2 strength/contact companions after the caller hashes each shard. */
export async function verifyPackedConnectionCompanions(
  tensor: Record<string, unknown>,
  formatVersion: 1 | 2,
  shards: ReadonlyMap<string, VerifiedPackedShard>,
  used: Set<string>,
  label: string
): Promise<void> {
  function fail(reason: string): never { throw new Error(`${label} packed connections: ${reason}.`); }
  const connection = tensor.connectionState;
  const parallel = tensor.parallelConnections;
  if (formatVersion === 1 && connection === undefined) {
    if (parallel !== undefined) fail("contacts require explicit connection state");
    return;
  }
  if (record(connection) && connection.encoding === "unit-strength" &&
      connection.strength === 1 && Object.keys(connection).length === 2) {
    if (parallel !== undefined) fail("unit strength has undeclared parallel contacts");
    return;
  }
  if (!record(connection) || connection.formatVersion !== 1 ||
      connection.encoding !== "uint4-strength-minus-one-high4-resistance" ||
      !Array.isArray(connection.shape) || connection.shape.length !== 2 ||
      !connection.shape.every(count) || !count(tensor.numel) ||
      (connection.shape[0] as number) * (connection.shape[1] as number) !== tensor.numel) {
    fail("invalid strength shape or encoding");
  }
  const shape = connection.shape as number[];
  const tensorShape = tensor.shape as number[];
  let rows = tensorShape.length < 2 ? 1 : tensorShape[0]!;
  let columns = tensorShape.slice(tensorShape.length < 2 ? 0 : 1).reduce((a, b) => a * b, 1);
  if (tensor.packedRowLayout !== undefined) {
    const layout = tensor.packedRowLayout;
    if (!record(layout) || Object.keys(layout).length !== 2 ||
        layout.kind !== "transpose-convolution" || !count(layout.groups) || layout.groups < 1 ||
        tensor.kind !== "projection" || tensorShape.length < 3 || tensorShape.length > 5 ||
        tensorShape[0]! % layout.groups !== 0) fail("invalid packed row layout");
    rows = tensorShape[1]! * layout.groups * tensorShape.slice(2).reduce((a, b) => a * b, 1);
    columns = tensorShape[0]! / layout.groups;
  }
  if (shape[0] !== rows || shape[1] !== columns) fail("strength rows do not match the projection");
  const claim = (descriptor: Record<string, unknown>, prefix: string, bytes: number): VerifiedPackedShard => {
    const name = descriptor.shard;
    if (typeof name !== "string" || !name.startsWith(prefix) || used.has(name)) fail("missing or reused companion shard");
    const shard = shards.get(name);
    if (!shard || descriptor.byteLength !== bytes || shard.byteLength !== bytes || descriptor.sha256 !== shard.sha256) {
      fail("companion length or checksum mismatch");
    }
    used.add(name);
    return shard;
  };
  claim(connection, "connections-", tensor.numel as number);
  if (parallel === undefined) return;
  if (!record(parallel) || parallel.encoding !== "u64-row-column-ordinal-u8-sign-state-le" ||
      !count(parallel.count) || parallel.count < 1 || !Number.isSafeInteger(parallel.count * 26)) {
    fail("invalid parallel contact encoding or count");
  }
  const shard = claim(parallel, "parallel-", (parallel.count as number) * 26);
  let carry = Buffer.alloc(0);
  let previous: readonly bigint[] | undefined;
  let seen = 0;
  for await (const data of createReadStream(shard.path)) {
    const chunk = carry.length ? Buffer.concat([carry, data as Buffer]) : data as Buffer;
    let offset = 0;
    for (; offset + 26 <= chunk.length; offset += 26) {
      const key = [chunk.readBigUInt64LE(offset), chunk.readBigUInt64LE(offset + 8), chunk.readBigUInt64LE(offset + 16)] as const;
      const sign = chunk[offset + 24]!;
      if (key[0] >= BigInt(shape[0]!) || key[1] >= BigInt(shape[1]!) ||
          sign >> 2 !== 0x15 || (sign & 3) === 3 ||
          (previous && (key[0] < previous[0]! || (key[0] === previous[0] &&
            (key[1] < previous[1]! || (key[1] === previous[1] && key[2] <= previous[2]!)))))) {
        fail("invalid contact endpoint, order or sign");
      }
      previous = key;
      seen++;
    }
    carry = Buffer.from(chunk.subarray(offset));
  }
  if (carry.length || seen !== parallel.count) fail("truncated parallel contacts");
}
