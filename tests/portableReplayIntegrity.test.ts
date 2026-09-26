import { createHash } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, expect, test } from "vitest";
import { verifyPortableReplaySqlite } from "../src/main/portableReplayIntegrity";

const temporary: string[] = [];
afterEach(() => {
  for (const path of temporary.splice(0)) rmSync(path, { recursive: true, force: true });
});

function fixture() {
  const folder = mkdtempSync(join(tmpdir(), "omni-replay-integrity-"));
  temporary.push(folder);
  const path = join(folder, "replay.sqlite3");
  const database = new DatabaseSync(path);
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
  const rows: Array<{ sequence: number; sha256: string }> = [];
  for (const [index, payload] of [
    Buffer.from(new Float32Array([1.25, -3.5]).buffer),
    Buffer.from(new Float32Array([7.5]).buffer)
  ].entries()) {
    const shape = index === 0 ? "[2]" : "[1]";
    const checksum = createHash("sha256")
      .update("torch.float32\0" + shape + "\0", "ascii")
      .update(payload)
      .digest("hex");
    const result = database.prepare(
      "INSERT INTO replay (created_at,sha256,dtype,shape_json,payload) VALUES (?,?,?,?,?)"
    ).run(index + 1, checksum, "torch.float32", shape, payload);
    rows.push({ sequence: Number(result.lastInsertRowid), sha256: checksum });
  }
  database.close();
  const digest = createHash("sha256")
    .update(`${rows[0]!.sequence}\0${rows[0]!.sha256}\n`, "ascii")
    .digest("hex");
  const checkpoint = {
    format: "omni-replay-sqlite",
    formatVersion: 1,
    path: "replay.sqlite3",
    count: 1,
    highWaterId: rows[0]!.sequence,
    contentSha256: digest
  };
  return { path, checkpoint };
}

test("validates a committed prefix and all pending replay rows", () => {
  const { path, checkpoint } = fixture();
  expect(verifyPortableReplaySqlite(path, checkpoint)).toEqual({
    committedExamples: 1,
    durableExamples: 2,
    pendingExamples: 1
  });
});

test("rejects corrupt pending data rather than trusting the committed digest", () => {
  const { path, checkpoint } = fixture();
  const database = new DatabaseSync(path);
  database.prepare("UPDATE replay SET payload=? WHERE sequence=2").run(Buffer.alloc(4));
  database.close();
  expect(() => verifyPortableReplaySqlite(path, checkpoint)).toThrow(/checksum failed/);
});

test("rejects a mismatched replay checkpoint", () => {
  const { path, checkpoint } = fixture();
  expect(() => verifyPortableReplaySqlite(path, { ...checkpoint, count: 2 }))
    .toThrow(/checkpoint checksum failed/);
});

test("rejects non-SQLite bytes", () => {
  const folder = mkdtempSync(join(tmpdir(), "omni-replay-fake-"));
  temporary.push(folder);
  const path = join(folder, "replay.sqlite3");
  writeFileSync(path, "not a sqlite database");
  expect(() => verifyPortableReplaySqlite(path, {
    format: "omni-replay-sqlite",
    formatVersion: 1,
    path: "replay.sqlite3",
    count: 0,
    highWaterId: 0,
    contentSha256: createHash("sha256").digest("hex")
  })).toThrow();
});

test("rejects a replay view even when it returns plausible rows", () => {
  const folder = mkdtempSync(join(tmpdir(), "omni-replay-view-"));
  temporary.push(folder);
  const path = join(folder, "replay.sqlite3");
  const database = new DatabaseSync(path);
  database.exec("CREATE VIEW replay AS SELECT 1 AS sequence, '' AS sha256, '' AS dtype, '[]' AS shape_json, x'' AS payload WHERE 0");
  database.close();
  expect(() => verifyPortableReplaySqlite(path, {
    format: "omni-replay-sqlite",
    formatVersion: 1,
    path: "replay.sqlite3",
    count: 0,
    highWaterId: 0,
    contentSha256: createHash("sha256").digest("hex")
  })).toThrow(/table is missing/);
});
