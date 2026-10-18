import { createHash } from "node:crypto";
import { lstat, mkdtemp, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  copyMutableFileIsolated,
  snapshotMutableSqliteIsolated,
  writeMutableFileIsolated
} from "../src/main/mutableFileIsolation";

vi.mock("node:fs/promises", async original => {
  const actual = await original<typeof import("node:fs/promises")>();
  return { ...actual, lstat: vi.fn(actual.lstat), stat: vi.fn(actual.stat), rename: vi.fn(actual.rename) };
});

function digest(value: Buffer): string {
  return createHash("sha256").update(value).digest("hex");
}

describe("mutable file isolation", () => {
  const roots: string[] = [];

  afterEach(async () => {
    vi.resetAllMocks();
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
      stat(source, { bigint: true }),
      stat(copied, { bigint: true }),
      stat(written, { bigint: true })
    ]);
    expect(copiedInfo.ino).not.toBe(sourceInfo.ino);
    expect(writtenInfo.ino).not.toBe(sourceInfo.ino);

    await writeFile(copied, JSON.stringify({ generation: 2 }));
    expect(digest(await readFile(source))).toBe(sourceBefore);
  });

  it.each([false, true])("compares exact 64-bit file identities without accepting a shared inode (shared=%s)", async shared => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-identity-"));
    roots.push(root);
    const source = join(root, "source.json"), copied = join(root, "copy.json");
    await writeFile(source, "original private bytes");
    const sourceBefore = digest(await readFile(source));
    const sourceInode = 9007199254740992n, copiedInode = shared ? sourceInode : sourceInode + 1n;
    expect(Number(copiedInode)).toBe(Number(sourceInode));
    const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
    vi.mocked(stat).mockImplementation(async (path, options) => {
      const info = await actual.stat(path, options);
      const inode = path === source ? sourceInode : copiedInode;
      return Object.assign(info, { dev: typeof info.dev === "bigint" ? 1n : 1,
        ino: typeof info.ino === "bigint" ? inode : Number(inode) });
    });

    const copy = copyMutableFileIsolated(source, copied, { expectedSha256: sourceBefore });
    if (shared) {
      await expect(copy).rejects.toThrow("reused the source inode");
    } else {
      await expect(copy).resolves.toBe(Buffer.byteLength("original private bytes"));
      const [sourceInfo, copiedInfo] = await Promise.all([
        actual.stat(source, { bigint: true }), actual.stat(copied, { bigint: true })
      ]);
      expect([copiedInfo.dev, copiedInfo.ino]).not.toEqual([sourceInfo.dev, sourceInfo.ino]);
      await writeFile(copied, "changed copy");
      expect(digest(await readFile(source))).toBe(sourceBefore);
    }
  });

  it("rejects a source identity change even when its 64-bit IDs round to the same number", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-source-change-"));
    roots.push(root);
    const source = join(root, "source.json"), copied = join(root, "copy.json");
    await writeFile(source, "original private bytes");
    const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
    let sourceReads = 0;
    vi.mocked(lstat).mockImplementation(async (path, options) => {
      const info = await actual.lstat(path, options);
      if (path !== source) return info;
      const inode = 9007199254740992n + BigInt(sourceReads++ > 0);
      return Object.assign(info, { ino: typeof info.ino === "bigint" ? inode : Number(inode) });
    });
    await expect(copyMutableFileIsolated(source, copied)).rejects.toThrow("source changed during materialization");
    await expect(actual.stat(copied)).rejects.toMatchObject({ code: "ENOENT" });
    expect(await readdir(root)).toEqual(["source.json"]);
  });

  it("retains the only previous destination if promotion and restoration both fail", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-rollback-"));
    roots.push(root);
    const destination = join(root, "pointer.json"), previous = "only previous private bytes";
    await writeFile(destination, previous);
    const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
    const blocked = Object.assign(new Error("injected promotion refusal"), { code: "EPERM" });
    const restoration = Object.assign(new Error("injected restoration refusal"), { code: "EACCES" });
    let backup = "";
    vi.mocked(rename).mockImplementation(async (source, target) => {
      if (String(source).endsWith(".private-next")) throw blocked;
      if (String(source).endsWith(".private-previous")) throw restoration;
      await actual.rename(source, target);
      backup = String(target);
    });

    const failure = await writeMutableFileIsolated(destination, Buffer.from("new private bytes")).catch(error => error);
    expect(backup).toMatch(/\.private-previous$/);
    expect(await readdir(root)).toEqual([backup.slice(root.length + 1)]);
    expect(await readFile(backup, "utf8")).toBe(previous);
    expect(failure).toBeInstanceOf(AggregateError);
    expect(failure.message).toContain(backup);
    expect(failure.errors).toEqual([blocked, restoration]);
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
      stat(source, { bigint: true }),
      stat(sibling, { bigint: true })
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
