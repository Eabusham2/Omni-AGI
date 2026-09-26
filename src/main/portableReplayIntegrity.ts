import { createHash } from "node:crypto";
import { DatabaseSync } from "node:sqlite";

export interface PortableReplayCheckpoint {
  format: string;
  formatVersion: number;
  path: string;
  count: number;
  highWaterId: number;
  contentSha256: string;
}

const DTYPE_BYTES = new Map<string, number>([
  ["torch.bool", 1],
  ["torch.uint8", 1],
  ["torch.int8", 1],
  ["torch.int16", 2],
  ["torch.int32", 4],
  ["torch.int64", 8],
  ["torch.float16", 2],
  ["torch.bfloat16", 2],
  ["torch.float32", 4],
  ["torch.float64", 8]
]);

function checkpointShape(value: unknown): asserts value is PortableReplayCheckpoint {
  if (
    typeof value !== "object" || value === null ||
    (value as PortableReplayCheckpoint).format !== "omni-replay-sqlite" ||
    (value as PortableReplayCheckpoint).formatVersion !== 1 ||
    (value as PortableReplayCheckpoint).path !== "replay.sqlite3" ||
    !Number.isSafeInteger((value as PortableReplayCheckpoint).count) ||
    (value as PortableReplayCheckpoint).count < 0 ||
    !Number.isSafeInteger((value as PortableReplayCheckpoint).highWaterId) ||
    (value as PortableReplayCheckpoint).highWaterId < 0 ||
    !/^[a-f0-9]{64}$/.test((value as PortableReplayCheckpoint).contentSha256)
  ) {
    throw new Error("Portable replay checkpoint metadata is invalid.");
  }
}

function tensorShape(value: unknown): number[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(String(value));
  } catch {
    throw new Error("Portable replay tensor shape is invalid.");
  }
  if (
    !Array.isArray(parsed) || parsed.length < 1 ||
    parsed.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 0)
  ) {
    throw new Error("Portable replay tensor shape is invalid.");
  }
  return parsed as number[];
}

/** Match the Python worker's dtype + NUL + json.dumps(shape) + NUL + payload digest. */
function rowTensorSha256(dtype: string, shape: number[], payload: Uint8Array): string {
  const pythonShape = `[${shape.join(", ")}]`;
  return createHash("sha256")
    .update(dtype, "ascii")
    .update("\0")
    .update(pythonShape, "ascii")
    .update("\0")
    .update(payload)
    .digest("hex");
}

/**
 * Validate a replay database before a portable export/import can publish it.
 * SQLite rows are visited one at a time; all pending rows are validated even
 * though only the committed prefix participates in the generation digest.
 */
export function verifyPortableReplaySqlite(
  path: string,
  checkpoint: unknown
): { committedExamples: number; durableExamples: number; pendingExamples: number } {
  checkpointShape(checkpoint);
  const database = new DatabaseSync(path, { readOnly: true });
  let transactionOpen = false;
  try {
    database.exec("BEGIN");
    transactionOpen = true;
    const quick = database.prepare("PRAGMA quick_check").get() as
      { quick_check?: unknown } | undefined;
    if (quick?.quick_check !== "ok") {
      throw new Error("Portable replay SQLite integrity failed.");
    }
    const replayTable = database.prepare(
      "SELECT type FROM sqlite_master WHERE name='replay'"
    ).get() as { type?: unknown } | undefined;
    if (replayTable?.type !== "table") {
      throw new Error("Portable replay table is missing.");
    }
    const digest = createHash("sha256");
    let committedExamples = 0;
    let durableExamples = 0;
    let committedHighWater = 0;
    let priorSequence = 0;
    const rows = database.prepare(
      "SELECT sequence,sha256,dtype,shape_json,payload FROM replay ORDER BY sequence"
    ).iterate();
    for (const row of rows) {
      const sequence = Number(row.sequence);
      const dtype = String(row.dtype);
      const checksum = String(row.sha256);
      const payload = row.payload;
      const bytesPerElement = DTYPE_BYTES.get(dtype);
      if (
        !Number.isSafeInteger(sequence) || sequence <= priorSequence ||
        !/^[a-f0-9]{64}$/.test(checksum) || bytesPerElement === undefined ||
        !(payload instanceof Uint8Array)
      ) {
        throw new Error("Portable replay row is invalid.");
      }
      const shape = tensorShape(row.shape_json);
      let elementCount = 1;
      for (const dimension of shape) {
        elementCount *= dimension;
        if (!Number.isSafeInteger(elementCount)) {
          throw new Error("Portable replay tensor shape is invalid.");
        }
      }
      const expectedBytes = elementCount * bytesPerElement;
      if (
        !Number.isSafeInteger(expectedBytes) ||
        expectedBytes !== payload.byteLength ||
        rowTensorSha256(dtype, shape, payload) !== checksum
      ) {
        throw new Error("Portable replay tensor checksum failed.");
      }
      durableExamples += 1;
      if (sequence <= checkpoint.highWaterId) {
        digest.update(String(sequence), "ascii");
        digest.update("\0");
        digest.update(checksum, "ascii");
        digest.update("\n");
        committedExamples += 1;
        committedHighWater = sequence;
      }
      priorSequence = sequence;
    }
    if (
      committedExamples !== checkpoint.count ||
      committedHighWater !== checkpoint.highWaterId ||
      digest.digest("hex") !== checkpoint.contentSha256
    ) {
      throw new Error("Portable replay checkpoint checksum failed.");
    }
    database.exec("COMMIT");
    transactionOpen = false;
    return {
      committedExamples,
      durableExamples,
      pendingExamples: durableExamples - committedExamples
    };
  } finally {
    if (transactionOpen) {
      try { database.exec("ROLLBACK"); } catch { /* Preserve the validation error. */ }
    }
    database.close();
  }
}
