import { createHash } from "node:crypto";
import { mkdtemp, readFile, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import {
  copyMutableFileIsolated,
  snapshotMutableSqliteIsolated,
  writeMutableFileIsolated
} from "../src/main/mutableFileIsolation";

function digest(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

describe("mutable file isolation", () => {
  const roots: string[] = [];

  afterEach(async () => {
    const { rm } = await import("node:fs/promises");
    await Promise.all(roots.splice(0).map((root) =>
      rm(root, { recursive: true, force: true })
    ));
  });

  it("uses distinct inodes for mutable reflink copies and byte materializations", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-file-"));
    roots.push(root);
    const source = join(root, "source.json");
    const copied = join(root, "sibling", "pointer.json");
    const written = join(root, "other", "pointer.json");
    await writeFile(source, JSON.stringify({ generation: 1 }));
    const sourceBefore = digest(await readFile(source));

    await copyMutableFileIsolated(source, copied, {
      expectedSha256: sourceBefore
    });
    await writeMutableFileIsolated(
      written,
      Buffer.from(JSON.stringify({ generation: 1 }))
    );

    const [sourceInfo, copiedInfo, writtenInfo] = await Promise.all([
      stat(source),
      stat(copied),
      stat(written)
    ]);
    expect(copiedInfo.ino).not.toBe(sourceInfo.ino);
    expect(writtenInfo.ino).not.toBe(sourceInfo.ino);

    await writeFile(copied, JSON.stringify({ generation: 2 }));
    expect(digest(await readFile(source))).toBe(sourceBefore);
  });

  it("captures committed WAL rows without copying sidecars or sharing an inode", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-sqlite-"));
    roots.push(root);
    const source = join(root, "events.sqlite3");
    const sibling = join(root, "fork", "events.sqlite3");
    const sourceDatabase = new DatabaseSync(source);
    sourceDatabase.exec(`
      PRAGMA journal_mode=WAL;
      PRAGMA wal_autocheckpoint=0;
      CREATE TABLE events(sequence INTEGER PRIMARY KEY, value TEXT NOT NULL);
      INSERT INTO events(value) VALUES ('one'), ('two');
    `);

    await snapshotMutableSqliteIsolated(source, sibling);
    const [sourceInfo, siblingInfo] = await Promise.all([
      stat(source),
      stat(sibling)
    ]);
    expect(siblingInfo.ino).not.toBe(sourceInfo.ino);
    await expect(stat(`${sibling}-wal`)).rejects.toMatchObject({ code: "ENOENT" });
    await expect(stat(`${sibling}-shm`)).rejects.toMatchObject({ code: "ENOENT" });

    const siblingDatabase = new DatabaseSync(sibling);
    try {
      expect(
        siblingDatabase.prepare("SELECT value FROM events ORDER BY sequence").all()
      ).toEqual([{ value: "one" }, { value: "two" }]);
      siblingDatabase.prepare("INSERT INTO events(value) VALUES (?)").run("fork-only");
    } finally {
      siblingDatabase.close();
    }
    expect(
      sourceDatabase.prepare("SELECT value FROM events ORDER BY sequence").all()
    ).toEqual([{ value: "one" }, { value: "two" }]);
    sourceDatabase.close();
  });
});
