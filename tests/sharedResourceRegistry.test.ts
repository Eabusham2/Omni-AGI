/** Pure SQLite/environment/source fixtures: never launch a worker or brain. */
import { readFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SharedResourceRegistry } from "../src/main/sharedResourceRegistry";
import { managedWorkerMemoryEnvironment } from "../src/main/engineSupervisor";
import { BrainRepository } from "../src/main/brainRepository";
import type { BrainDocument } from "../src/shared/types";

const roots: string[] = [];
const connections: Array<{ close(): void }> = [];
const GIB = 1024 ** 3;

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), "omni-shared-resource-registry-"));
  roots.push(directory);
  const path = join(directory, "shared-resources.sqlite3");
  const diagnostics: string[] = [];
  const registry = new SharedResourceRegistry(path, (message) => diagnostics.push(message));
  connections.push(registry);
  return { registry, path, diagnostics };
}

afterEach(async () => {
  for (const connection of connections.splice(0)) {
    try { connection.close(); } catch { /* Already closed by a lifetime fixture. */ }
  }
  await Promise.all(roots.splice(0).map((path) => rm(path, { recursive: true, force: true })));
});

describe("main-owned shared resource declarations", () => {
  it("registers inactive brains and takes the largest pool instead of summing them", async () => {
    const { registry } = await fixture();
    registry.observeSaved("brain-a", 24 * GIB);
    registry.observeSaved("brain-b", 64 * GIB);
    expect(registry.status()).toMatchObject({
      largestConfiguredPoolBytes: 64 * GIB,
      globalRamCeilingBytes: null,
      pendingDeclarations: 0
    });
    registry.observeSaved("brain-b", 12 * GIB);
    expect(registry.status().largestConfiguredPoolBytes).toBe(24 * GIB);
    registry.observeRemoved("brain-a");
    expect(registry.status().largestConfiguredPoolBytes).toBe(12 * GIB);
  });

  it("persists owner declarations and the exact selected RAM ceiling across restart", async () => {
    const { registry, path } = await fixture();
    registry.selectActiveRuntime("active-brain", 32 * GIB, 7 * GIB);
    registry.observeSaved("inactive-larger-pool", 100 * GIB);
    expect(registry.status().globalRamCeilingBytes).toBe(7 * GIB);
    registry.close();
    const restored = new SharedResourceRegistry(path);
    connections.push(restored);
    expect(restored.status()).toMatchObject({
      largestConfiguredPoolBytes: 100 * GIB,
      globalRamCeilingBytes: 7 * GIB
    });
    restored.selectActiveRuntime("active-brain", 32 * GIB, 4 * GIB);
    expect(restored.status().globalRamCeilingBytes).toBe(4 * GIB);
  });

  it("reconciles crashed app owners without touching worker owners or file usage", async () => {
    const { registry, path } = await fixture();
    const saved = "d105bac6-542e-48f2-b91b-9fbaab7d0c18";
    const removed = "6cc0693d-98a8-4bb1-9183-1a21fba25237";
    registry.observeSaved(saved, 24 * GIB);
    registry.observeSaved(removed, 96 * GIB);
    const db = new DatabaseSync(path);
    connections.push(db);
    db.prepare("INSERT INTO owners VALUES(?,?,?,?,?)").run("stdio:999", 2 * GIB, null, 999, "worker");
    db.exec("CREATE TABLE files(identity TEXT PRIMARY KEY,bytes INTEGER NOT NULL,path TEXT NOT NULL)");
    db.prepare("INSERT INTO files VALUES(?,?,?)").run("held-file", 123, "held-backing");
    registry.reconcileSavedOwners([saved]);
    expect(registry.status().largestConfiguredPoolBytes).toBe(24 * GIB);
    expect(db.prepare("SELECT owner FROM owners ORDER BY owner").all()).toEqual([
      { owner: saved },
      { owner: "stdio:999" }
    ]);
    expect(db.prepare("SELECT * FROM files").all()).toEqual([
      { identity: "held-file", bytes: 123, path: "held-backing" }
    ]);
  });

  it("never credits Python file usage or leases on declaration, removal or RAM selection", async () => {
    const { registry, path } = await fixture();
    const pythonTables = new DatabaseSync(path);
    connections.push(pythonTables);
    pythonTables.exec(`
      CREATE TABLE files(identity TEXT PRIMARY KEY,bytes INTEGER NOT NULL,path TEXT NOT NULL);
      CREATE TABLE leases(token TEXT PRIMARY KEY,owner TEXT NOT NULL,kind TEXT NOT NULL,bytes INTEGER NOT NULL,
        state TEXT NOT NULL,pid INTEGER NOT NULL,session TEXT NOT NULL,committed INTEGER,identity TEXT,path TEXT,
        held INTEGER NOT NULL DEFAULT 0);
      INSERT INTO policy VALUES('spill_used',123456);
      INSERT INTO files VALUES('fixture-inode',123456,'still-live-backing');
      INSERT INTO leases VALUES('fixture-lease','brain-a','spill',0,'allocated',1,'fixture',NULL,'fixture-inode','still-live-backing',1);
    `);
    const beforeFiles = pythonTables.prepare("SELECT * FROM files").all();
    const beforeLeases = pythonTables.prepare("SELECT * FROM leases").all();
    registry.observeSaved("brain-a", 32 * GIB);
    registry.selectActiveRuntime("brain-a", 32 * GIB, 6 * GIB);
    registry.observeRemoved("brain-a");
    expect(pythonTables.prepare("SELECT * FROM files").all()).toEqual(beforeFiles);
    expect(pythonTables.prepare("SELECT * FROM leases").all()).toEqual(beforeLeases);
    expect(pythonTables.prepare("SELECT bytes FROM policy WHERE key='spill_used'").get()?.bytes).toBe(123456);
    expect(registry.status().largestConfiguredPoolBytes).toBe(0);
  });

  it("queues a contended post-save declaration without failing the committed save", async () => {
    const { registry, path, diagnostics } = await fixture();
    const blocker = new DatabaseSync(path);
    connections.push(blocker);
    blocker.exec("BEGIN IMMEDIATE");
    expect(() => registry.observeSaved("committed-brain", 32 * GIB)).not.toThrow();
    expect(registry.status().pendingDeclarations).toBe(1);
    expect(diagnostics.length).toBeGreaterThan(0);
    expect(() => registry.requireSynchronized()).toThrow(/admission paused/i);
    blocker.exec("ROLLBACK");
    registry.requireSynchronized();
    expect(registry.status()).toMatchObject({ largestConfiguredPoolBytes: 32 * GIB, pendingDeclarations: 0 });
    expect(registry.status().lastError).toBeUndefined();
  });

  it("does not silently clear an invalid committed declaration on a healthy DB read", async () => {
    const { registry } = await fixture();
    expect(() => registry.observeSaved("valid-owner", -1)).not.toThrow();
    expect(() => registry.requireSynchronized()).toThrow(/invalid committed owner declaration/i);
    expect(() => registry.requireSynchronized()).toThrow(/invalid committed owner declaration/i);
    expect(registry.status().largestConfiguredPoolBytes).toBe(0);
  });

  it("keeps the latest confirmed removal when an earlier saved declaration is pending", async () => {
    const { registry, path } = await fixture();
    registry.observeSaved("removed-brain", 32 * GIB);
    const blocker = new DatabaseSync(path);
    connections.push(blocker);
    blocker.exec("BEGIN IMMEDIATE");
    registry.observeSaved("removed-brain", 64 * GIB);
    registry.observeRemoved("removed-brain");
    blocker.exec("ROLLBACK");
    registry.requireSynchronized();
    expect(registry.status().largestConfiguredPoolBytes).toBe(0);
  });

  it("rolls back only its own failed selection, never a newer active brain", async () => {
    const { registry } = await fixture();
    const first = registry.selectActiveRuntime("first", 32 * GIB, 4 * GIB);
    const second = registry.selectActiveRuntime("second", 32 * GIB, 6 * GIB);
    registry.rollbackSelection(first);
    expect(registry.status().globalRamCeilingBytes).toBe(6 * GIB);
    registry.rollbackSelection(second);
    expect(registry.status().globalRamCeilingBytes).toBe(4 * GIB);
  });

  it("restores an absent original RAM policy without touching spill accounting", async () => {
    const { registry } = await fixture();
    const selection = registry.selectActiveRuntime("first", 32 * GIB, 4 * GIB);
    registry.rollbackSelection(selection);
    expect(registry.status().globalRamCeilingBytes).toBeNull();
  });

  it("validates owners and byte declarations before any policy mutation", async () => {
    const { registry } = await fixture();
    registry.selectActiveRuntime("valid", 1, 2);
    for (const bytes of [-1, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
      expect(() => registry.selectActiveRuntime("valid", 1, bytes)).toThrow(/safe byte count/i);
    }
    expect(() => registry.selectActiveRuntime("invalid/id", 1, 2)).toThrow(/saved brain id/i);
    expect(registry.status().globalRamCeilingBytes).toBe(2);
  });

  it("binds both worker roles to the trusted path, not an inherited environment override", async () => {
    const { path } = await fixture();
    const inherited = { OMNI_SHARED_RESOURCE_LEDGER: "/untrusted.sqlite3", OMNI_MEMORY_OWNER_PID: "1", FIXTURE: "retained" };
    const actual = managedWorkerMemoryEnvironment(inherited, path);
    expect(actual).toMatchObject({ OMNI_SHARED_RESOURCE_LEDGER: path, OMNI_MEMORY_OWNER_PID: String(process.pid), FIXTURE: "retained" });
    expect(inherited.OMNI_SHARED_RESOURCE_LEDGER).toBe("/untrusted.sqlite3");
    expect(managedWorkerMemoryEnvironment(inherited).OMNI_SHARED_RESOURCE_LEDGER).toBeUndefined();
    expect(() => managedWorkerMemoryEnvironment(inherited, "relative.sqlite3")).toThrow(/absolute trusted path/i);
    const supervisor = readFileSync(resolve("src/main/engineSupervisor.ts"), "utf8");
    expect(supervisor).toContain("managedWorkerMemoryEnvironment(process.env, this.options.sharedResourceLedgerPath)");
    expect(supervisor).toMatch(/new EngineSupervisor\(\s*this\.options,\s*"inspection"/u);
  });

  it("does not turn failed post-commit observers into duplicate neural mutations", async () => {
    const report = vi.fn();
    const warning = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    try {
      // Invoke the real repository observer methods unbound. No repository
      // constructor/create/save, brain, neural state or filesystem fixture.
      const scope = Object.create(BrainRepository.prototype) as {
        commitObserver: { saved(): Promise<void>; removed(): Promise<void>; error: typeof report };
        observeCommittedBrain(brain: BrainDocument): Promise<void>;
        observeRemovedBrain(id: string): Promise<void>;
      };
      scope.commitObserver = {
        saved: async () => { throw Error("save observer failure"); },
        removed: async () => { throw Error("remove observer failure"); },
        error: report
      };
      await expect(scope.observeCommittedBrain({ id: "fixture", config: { storagePoolBytes: 3 } } as BrainDocument)).resolves.toBeUndefined();
      await expect(scope.observeRemovedBrain("fixture")).resolves.toBeUndefined();
      expect(report).toHaveBeenCalledTimes(2);
      expect(report.mock.calls[0]?.slice(0, 2)).toEqual(["saved", "fixture"]);
      expect(report.mock.calls[1]?.slice(0, 2)).toEqual(["removed", "fixture"]);
    } finally {
      warning.mockRestore();
    }
  });

  it("wires all current saved declarations and passive startup checks before worker launch", () => {
    const startup = readFileSync(resolve("src/main/index.ts"), "utf8");
    expect(startup).toContain('join(app.getPath("userData"), "shared-resources.sqlite3")');
    expect(startup).toContain("sharedResourceLedgerPath: registry.ledgerPath");
    expect(startup).toContain("registry.observeSaved(saved.id, saved.config.storagePoolBytes)");
    expect(startup).toContain("registry.reconcileSavedOwners(savedBrains.map((summary) => summary.id))");
    expect(startup.indexOf("registry.requireSynchronized()")).toBeLessThan(startup.indexOf("engine = new EngineSupervisor"));
    expect(startup).toContain("preflightStart(summary.id, { selectActiveRuntime: false })");
    const repository = readFileSync(resolve("src/main/brainRepository.ts"), "utf8");
    expect(repository).toMatch(/await atomicWrite\(this\.documentPath\(normalized\.id\)[\s\S]*?await this\.observeCommittedBrain\(normalized\)/u);
    expect(repository).toMatch(/await rm\(exactPath,[^\n]+\);\s*await this\.observeRemovedBrain\(id\)/u);
    expect(repository).toMatch(/await rename\(source,[^\n]+\);\s*await this\.observeRemovedBrain\(id\)/u);
    expect(repository).toContain("await this.observeCommittedBrain(completed)");
    const registry = readFileSync(resolve("src/main/sharedResourceRegistry.ts"), "utf8");
    expect(registry).not.toMatch(/(?:INSERT|UPDATE|DELETE)\s+(?:OR\s+\w+\s+)?(?:INTO|FROM)?\s*(?:leases|files)\b/iu);
    expect(registry).not.toMatch(/VALUES\s*\(\s*['"]spill_used/iu);
  });

  it("selects the approved RAM ceiling before the first new-brain worker request", () => {
    const source = readFileSync(resolve("src/main/brainService.ts"), "utf8");
    const creation = source.slice(source.indexOf("  async create(\n"), source.indexOf("  async resumeFoundation("));
    expect(creation).toContain("this.sharedResources?.selectActiveRuntime(");
    expect(creation.indexOf("this.sharedResources?.requireSynchronized()")).toBeLessThan(
      creation.indexOf("this.repository.create(")
    );
    expect(creation).toContain("memoryPlan.resources.systemRamBudgetBytes");
    expect(creation.indexOf("this.sharedResources?.selectActiveRuntime(")).toBeLessThan(
      creation.indexOf('this.engine.requestStream("create"')
    );
    expect(creation).toContain("rollbackSelection(resourceSelection)");
  });
});
