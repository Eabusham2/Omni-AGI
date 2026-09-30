import { createHash } from "node:crypto";
import { isUtf8 } from "node:buffer";
import { createReadStream } from "node:fs";
import { lstat, readdir } from "node:fs/promises";
import { join } from "node:path";

const owner = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

/** Check complete owned argument files in bounded byte windows, never arrays. */
export async function validateConceptIdViewFile(path: string, filename: string): Promise<void> {
  if (!/^[a-f0-9]{64}\.jsonl$/.test(filename)) throw new Error("Invalid saved concept ID view name.");
  const before = await lstat(path, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink()) throw new Error("Unsafe saved concept ID view.");
  const digest = createHash("sha256");
  let carry = Buffer.alloc(0);
  let count = -1;
  let seen = 0;
  const line = (bytes: Buffer): void => {
    if (bytes.byteLength > 4096) throw new Error("Concept ID view record exceeds its byte window.");
    if (!isUtf8(bytes)) throw new Error("Concept ID view is not valid UTF-8.");
    const value = JSON.parse(bytes.toString("utf8")) as unknown;
    if (count < 0) {
      if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Invalid concept ID view header.");
      const header = value as Record<string, unknown>;
      if (Object.keys(header).sort().join(",") !== "brainId,count,format,turnId,version" ||
        header.format !== "omni-structural-concept-id-view" || header.version !== 1 ||
        typeof header.brainId !== "string" || !owner.test(header.brainId) ||
        typeof header.turnId !== "string" || !owner.test(header.turnId) ||
        !Number.isSafeInteger(header.count) || (header.count as number) < 0) throw new Error("Invalid concept ID view ownership/count.");
      count = header.count as number;
      if (!bytes.equals(Buffer.from(JSON.stringify(header, Object.keys(header).sort())))) throw new Error("Concept ID header is not canonical.");
    } else {
      if (typeof value !== "string" || !/^[^\s\x00-\x1f\x7f]{1,512}$/u.test(value)) throw new Error("Concept ID view contains non-structural data.");
      seen += 1;
      if (!bytes.equals(Buffer.from(JSON.stringify(value)))) throw new Error("Concept ID is not canonical.");
      if (seen > count) throw new Error("Concept ID view exceeds declared count.");
    }
  };
  for await (const raw of createReadStream(path, { highWaterMark: 65536 })) {
    const block = Buffer.from(raw as Buffer);
    digest.update(block);
    let start = 0;
    while (start < block.length) {
      const end = block.indexOf(10, start);
      const part = block.subarray(start, end < 0 ? block.length : end);
      if (carry.length + part.length > 4096) throw new Error("Concept ID view has an oversized record.");
      carry = Buffer.concat([carry, part]);
      if (end < 0) break;
      line(carry);
      carry = Buffer.alloc(0);
      start = end + 1;
    }
  }
  const after = await lstat(path, { bigint: true });
  if (carry.length || count < 0 || seen !== count || digest.digest("hex") !== filename.slice(0, 64) ||
    before.dev !== after.dev || before.ino !== after.ino || before.size !== after.size ||
    before.mtimeNs !== after.mtimeNs || before.ctimeNs !== after.ctimeNs) throw new Error("Concept ID view checksum/coverage/identity changed.");
}

/** Saved history may retain original owners; execution separately binds current brain+turn. */
export async function savedConceptIdViewFiles(engine: string): Promise<Map<string, string>> {
  const state = join(engine, "state");
  const directory = join(state, "concept-id-views");
  const files = new Map<string, string>();
  for (const path of [state, directory]) {
    try {
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Unsafe concept ID view directory.");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return files;
      throw error;
    }
  }
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (entry.name.startsWith(".")) continue; // Expendable uncommitted scratch, not saved state.
    if (!entry.isFile() || entry.isSymbolicLink() || !/^[a-f0-9]{64}\.jsonl$/.test(entry.name)) throw new Error("Unknown file in saved concept ID views.");
    const path = join(directory, entry.name);
    await validateConceptIdViewFile(path, entry.name);
    files.set(`state/concept-id-views/${entry.name}`, path);
  }
  return files;
}
