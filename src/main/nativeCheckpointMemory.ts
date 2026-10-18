import { constants } from "node:fs";
import { lstat, open } from "node:fs/promises";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { packedOwnerMetadataReserve } from "./nativeCoreInventory";

export interface SavedTensorShape { dtype: string; shape: number[]; bytes: number }

/** Read headers only; no weight payload or learned model is materialized. */
export async function nativeCheckpointMemory(paths: readonly string[], inspect?: (tensors: ReadonlyMap<string, SavedTensorShape>) => void): Promise<{
  checkpointBytes: number;
  connectionStateBytes: number;
  unitStrengthMigrationReserveBytes: number;
  nativeControlMigrationReserveBytes: number;
  runtimeMetadataReserveBytes: number;
  projectedCheckpointBytes: number;
}> {
  const tensors = new Map<string, SavedTensorShape>();
  let checkpointBytes = 0;
  for (const path of paths) {
    const info = await lstat(path);
    if (!info.isFile() || info.isSymbolicLink() || info.size < 10) throw new Error("Native checkpoint memory inventory requires regular safetensor files.");
    const handle = await open(path, constants.O_RDONLY | (process.platform === "win32" ? 0 : constants.O_NOFOLLOW));
    try {
      const actual = await handle.stat();
      if (actual.ino !== info.ino || actual.dev !== info.dev) throw new Error("Native checkpoint changed while opening its inventory.");
      const prefix = Buffer.alloc(8);
      if ((await handle.read(prefix, 0, 8, 0)).bytesRead !== 8) throw new Error("Native checkpoint header is truncated.");
      const size = Number(prefix.readBigUInt64LE());
      if (!Number.isSafeInteger(size) || size < 2 || size > info.size - 8) throw new Error("Native checkpoint header size is invalid.");
      // JSON expansion is admitted from currently usable process/host memory.
      // No arbitrary tensor/file-size limit and no whole payload read.
      const parseReserve = size * 12;
      if (parseReserve > freemem() / 2 || parseReserve > (getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) / 2) {
        throw new Error("Native checkpoint header awaits enough RAM for resource planning.");
      }
      const buffer = Buffer.allocUnsafe(size);
      let position = 0;
      while (position < size) {
        const read = await handle.read(buffer, position, Math.min(1024 * 1024, size - position), 8 + position);
        if (!read.bytesRead) throw new Error("Native checkpoint header is truncated.");
        position += read.bytesRead;
      }
      const header: unknown = JSON.parse(buffer.toString("utf8"));
      if (!header || typeof header !== "object" || Array.isArray(header)) throw new Error("Native checkpoint header is invalid.");
      for (const [name, value] of Object.entries(header)) {
        if (name === "__metadata__") continue;
        if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Native checkpoint tensor descriptor is invalid.");
        const descriptor = value as Record<string, unknown>;
        const shape = descriptor.shape;
        const offsets = descriptor.data_offsets;
        if (typeof descriptor.dtype !== "string" || !Array.isArray(shape) || shape.some((n) => !Number.isSafeInteger(n) || n < 0)
          || !Array.isArray(offsets) || offsets.length !== 2 || offsets.some((n) => !Number.isSafeInteger(n) || n < 0)
          || offsets[1] < offsets[0] || offsets[1] > info.size - 8 - size) throw new Error("Native checkpoint tensor geometry is invalid.");
        if (tensors.has(name)) throw new Error("Native checkpoint contains duplicate tensor ownership.");
        const bytes = offsets[1] - offsets[0];
        const numel = shape.reduce((total, n) => total * n, 1);
        if (!Number.isSafeInteger(numel) || (descriptor.dtype === "U8" && numel !== bytes)) throw new Error("Native checkpoint tensor bytes do not match its shape.");
        tensors.set(name, { dtype: descriptor.dtype, shape, bytes });
      }
      const after = await handle.stat();
      if (after.size !== info.size || after.mtimeMs !== info.mtimeMs) throw new Error("Native checkpoint changed during resource planning.");
      checkpointBytes += info.size;
    } finally { await handle.close(); }
  }
  let connectionStateBytes = 0, unitStrengthMigrationReserveBytes = 0, nativeControlMigrationReserveBytes = 0;
  let packedOwners = 0, registryAndIndexReserveBytes = 0;
  for (const [name, tensor] of tensors) {
    if (/\.parallel_connections\.pages\.[^.]+\._record_meta$/.test(name)) registryAndIndexReserveBytes += 8192 + (tensor.shape[0] ?? 0) * 2;
    if (/(?:^|\.)synapses\.blocks\.[^.]+\._packed_weights$/.test(name)) registryAndIndexReserveBytes += 8192;
    if (/(?:^|\.)_(?:bias_)?connection_state$/.test(name)) {
      if (tensor.dtype !== "U8") throw new Error("Native connection state must use byte storage.");
      connectionStateBytes += tensor.bytes;
    }
    const match = /(^|\.)(_packed_forward_weight|_packed_forward_bias|_packed_weights)$/.exec(name);
    if (!match) continue;
    if (tensor.dtype !== "U8" || tensor.shape.length !== 2) throw new Error("Native packed signs must use row-packed byte storage.");
    const prefix = name.slice(0, name.length - match[2]!.length);
    const stateName = `${prefix}${match[2] === "_packed_forward_bias" ? "_bias_connection_state" : "_connection_state"}`;
    if (match[2] === "_packed_forward_weight") packedOwners += 1;
    if (match[2] === "_packed_forward_weight" && !tensors.has(`${prefix}parallel_connections._next_page`)) nativeControlMigrationReserveBytes += 8;
    if (!tensors.has(stateName)) {
      // Old native files do not carry the unpadded logical width. Four states
      // per packed byte is a conservative migration reserve (at most 3 extra
      // padding bytes per row), never a reported actual parameter count.
      unitStrengthMigrationReserveBytes += tensor.bytes * 4;
    }
  }
  const projectedCheckpointBytes = checkpointBytes + unitStrengthMigrationReserveBytes + nativeControlMigrationReserveBytes;
  const runtimeMetadataReserveBytes = packedOwnerMetadataReserve(packedOwners) + registryAndIndexReserveBytes;
  if (![checkpointBytes, connectionStateBytes, unitStrengthMigrationReserveBytes, nativeControlMigrationReserveBytes, projectedCheckpointBytes, runtimeMetadataReserveBytes].every(Number.isSafeInteger)) throw new Error("Native checkpoint inventory exceeds exact integer accounting.");
  inspect?.(tensors);
  return { checkpointBytes, connectionStateBytes, unitStrengthMigrationReserveBytes, nativeControlMigrationReserveBytes, runtimeMetadataReserveBytes, projectedCheckpointBytes };
}
