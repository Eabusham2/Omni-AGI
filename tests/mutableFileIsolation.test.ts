import { createHash } from "node:crypto";
import { lstat, mkdir, mkdtemp, readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";
import * as ts from "typescript";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  copyMutableFileIsolated,
  privateSqliteFilename,
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

  it.each([
    [String.raw`C:\owned\teacher's #?% & data\workspace.sqlite3`, String.raw`\\?\C:\owned\teacher's #?% & data\workspace.sqlite3`],
    [String.raw`\\server\share\teacher #%.sqlite3`, String.raw`\\?\UNC\server\share\teacher #%.sqlite3`],
    [String.raw`\\?\C:\owned\workspace.sqlite3`, String.raw`\\?\C:\owned\workspace.sqlite3`]
  ])("uses the locking Windows longpath VFS without interpreting filename options: %s", (path, namespaced) => {
    const filename = privateSqliteFilename(path, "win32");
    const [encoded, query] = filename.slice("file:".length).split("?");
    expect(filename.startsWith("file:")).toBe(true);
    expect(decodeURIComponent(encoded!)).toBe(namespaced);
    expect(query).toBe("vfs=win32-longpath");
    expect(filename).not.toContain("#");
    expect(privateSqliteFilename(path, "darwin")).toBe(path);
  });

  it.each([":memory:", ""])("preserves SQLite's special %j database name", path => {
    expect(privateSqliteFilename(path, "win32")).toBe(path);
    expect(privateSqliteFilename(path, "darwin")).toBe(path);
    const database = new DatabaseSync(privateSqliteFilename(path));
    try {
      database.exec("CREATE TABLE private_state(value TEXT); INSERT INTO private_state VALUES ('not a filesystem path');");
      expect(database.prepare("SELECT value FROM private_state").get()?.value).toBe("not a filesystem path");
    } finally { database.close(); }
  });

  it.each(["file:owned.sqlite3?immutable=1", "file::memory:?cache=shared", "file:///owned.sqlite3?vfs=win32-none"])(
    "does not accept caller-provided SQLite URI options: %s", path => {
      expect(() => privateSqliteFilename(path, "win32")).toThrow(/filesystem path, not a URI/);
      expect(() => privateSqliteFilename(path, "darwin")).toThrow(/filesystem path, not a URI/);
    }
  );

  it("routes every main-process SQLite open, backup, and VACUUM target through the standalone filename boundary", async () => {
    const root = fileURLToPath(new URL("../src/main/", import.meta.url));
    const unchecked: string[] = []; let constructors = 0, backups = 0, vacuums = 0;
    async function visitDirectory(directory: string): Promise<void> {
      for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) { await visitDirectory(path); continue; }
        if (!entry.name.endsWith(".ts")) continue;
        const source = ts.createSourceFile(path, await readFile(path, "utf8"), ts.ScriptTarget.Latest, true);
        const adapters = new Set<string>(), nativeConstructors = new Set(["DatabaseSync"]), nativeBackups = new Set<string>(), nativeNamespaces = new Set<string>();
        const aliases = new Map<string, ts.Expression>();
        for (const statement of source.statements) if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier)) {
          const bindings = statement.importClause?.namedBindings;
          if (bindings && ts.isNamespaceImport(bindings) && statement.moduleSpecifier.text === "node:sqlite") nativeNamespaces.add(bindings.name.text);
          if (!bindings || !ts.isNamedImports(bindings)) continue;
          for (const item of bindings.elements) {
            const imported = item.propertyName?.text ?? item.name.text;
            if (statement.moduleSpecifier.text === "./sqliteFilename" && imported === "sqliteFilename") adapters.add(item.name.text);
            if (statement.moduleSpecifier.text === "node:sqlite" && imported === "DatabaseSync") nativeConstructors.add(item.name.text);
            if (statement.moduleSpecifier.text === "node:sqlite" && imported === "backup") nativeBackups.add(item.name.text);
          }
        }
        function collectAliases(node: ts.Node): void {
          if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer) aliases.set(node.name.text, node.initializer);
          ts.forEachChild(node, collectAliases);
        }
        collectAliases(source);
        function usesAdapter(node: ts.Node, seen = new Set<string>()): boolean {
          if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && adapters.has(node.expression.text)) return true;
          if (ts.isIdentifier(node) && aliases.has(node.text) && !seen.has(node.text)) {
            const next = new Set(seen); next.add(node.text); return usesAdapter(aliases.get(node.text)!, next);
          }
          return ts.forEachChild(node, child => usesAdapter(child, seen) || undefined) === true;
        }
        function inspect(node: ts.Node): void {
          let argument: ts.Node | undefined, kind = "";
          const isNative = (expression: ts.Expression, names: Set<string>, name: string): boolean =>
            ts.isIdentifier(expression) && names.has(expression.text) || ts.isPropertyAccessExpression(expression) &&
            ts.isIdentifier(expression.expression) && nativeNamespaces.has(expression.expression.text) && expression.name.text === name;
          if (ts.isNewExpression(node) && isNative(node.expression, nativeConstructors, "DatabaseSync")) {
            constructors++; argument = node.arguments?.[0]; kind = "constructor";
          } else if (ts.isCallExpression(node) && isNative(node.expression, nativeBackups, "backup")) {
            backups++; argument = node.arguments[1]; kind = "backup destination";
          } else if (ts.isCallExpression(node) && ts.isPropertyAccessExpression(node.expression) && node.expression.name.text === "exec") {
            const sql = node.arguments[0];
            if (sql && ts.isTemplateExpression(sql) && /^VACUUM\s+INTO\b/i.test(sql.head.text)) {
              vacuums++; argument = sql; kind = "VACUUM destination";
            }
          }
          const routed = argument && (kind === "VACUUM destination" ? usesAdapter(argument) : ts.isCallExpression(argument) &&
            ts.isIdentifier(argument.expression) && adapters.has(argument.expression.text));
          if (kind && !routed) unchecked.push(`${entry.name}:${source.getLineAndCharacterOfPosition(node.getStart(source)).line + 1} ${kind}`);
          ts.forEachChild(node, inspect);
        }
        inspect(source);
      }
    }
    await visitDirectory(root);
    expect(constructors).toBeGreaterThan(0); expect(backups).toBeGreaterThan(0); expect(vacuums).toBeGreaterThan(0);
    expect(unchecked).toEqual([]);
  });

  it("snapshots deep managed paths beyond default Windows limits without changing the source or sharing its inode", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-private-long-path-"));
    roots.push(root);
    const source = join(root, "source's #&%.sqlite3"), restored = join(root, "restored.sqlite3");
    let directory = root;
    // Exceed SQLite win32's default 1040-byte buffer as well as MAX_PATH on
    // Windows; stay within the smaller native pathname limit on other hosts.
    const minimumLength = process.platform === "win32" ? 1200 : 320;
    for (let i = 0; directory.length < minimumLength; i++) directory = join(directory, `owned-${i}-${"x".repeat(60)}`);
    await mkdir(directory, { recursive: true });
    const destination = join(directory, "teacher's #&%.sqlite3");
    expect(destination.length).toBeGreaterThan(260);
    const database = new DatabaseSync(source);
    try { database.exec("CREATE TABLE events(value TEXT); INSERT INTO events VALUES ('original private row');"); }
    finally { database.close(); }
    const sourceBefore = digest(await readFile(source));

    await snapshotMutableSqliteIsolated(source, destination);
    const [sourceInfo, destinationInfo] = await Promise.all([
      stat(source, { bigint: true }), stat(destination, { bigint: true })
    ]);
    expect([destinationInfo.dev, destinationInfo.ino]).not.toEqual([sourceInfo.dev, sourceInfo.ino]);
    expect(await readdir(directory)).toEqual(["teacher's #&%.sqlite3"]);
    const snapshot = new DatabaseSync(privateSqliteFilename(destination));
    try {
      expect(snapshot.prepare("SELECT value FROM events").all()).toEqual([{ value: "original private row" }]);
      snapshot.exec("INSERT INTO events VALUES ('copy-only row');");
    } finally { snapshot.close(); }
    expect(digest(await readFile(source))).toBe(sourceBefore);

    // A deep source must use the same safe filename handling, not just INTO.
    await snapshotMutableSqliteIsolated(destination, restored);
    const roundTrip = new DatabaseSync(restored);
    try { expect(roundTrip.prepare("SELECT value FROM events ORDER BY rowid").all()).toEqual([
      { value: "original private row" }, { value: "copy-only row" }
    ]); } finally { roundTrip.close(); }
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
