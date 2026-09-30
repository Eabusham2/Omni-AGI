import { randomUUID } from "node:crypto";
import { chmodSync, existsSync, lstatSync, mkdirSync } from "node:fs";
import { dirname, isAbsolute, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

interface OwnerMutation {
  kind: "saved" | "removed";
  poolBytes?: number;
}

export interface SharedRamSelection {
  readonly epoch: number;
  readonly previousBytes: number | null;
  readonly selectedBytes: number;
}

export interface SharedResourceRegistryStatus {
  ledgerPath: string;
  largestConfiguredPoolBytes: number;
  globalRamCeilingBytes: number | null;
  pendingDeclarations: number;
  lastError?: string;
}

function ownerId(value: string): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(value)) {
    throw new Error("Shared resource owner must be a valid saved brain id.");
  }
  return value;
}

function byteCount(value: number): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error("Shared resource declaration must be a nonnegative safe byte count.");
  }
  return value;
}

/**
 * Trusted desktop declarations only. Python owns actual allocation leases,
 * file identities, reconciliation and spill_used; Node never credits them.
 */
export class SharedResourceRegistry {
  readonly ledgerPath: string;
  private readonly database: DatabaseSync;
  private readonly session = randomUUID();
  private readonly pending = new Map<string, OwnerMutation>();
  private pendingRamRestore?: SharedRamSelection;
  private selectionEpoch = 0;
  private lastError?: string;
  private invalidDeclaration?: string;
  private closed = false;

  constructor(
    path: string,
    private readonly diagnostic: (message: string) => void = (message) => console.warn(message)
  ) {
    if (!isAbsolute(path)) throw new Error("Shared resource ledger must be an absolute app-managed path.");
    this.ledgerPath = resolve(path);
    mkdirSync(dirname(this.ledgerPath), { recursive: true, mode: 0o700 });
    if (existsSync(this.ledgerPath) && lstatSync(this.ledgerPath).isSymbolicLink()) {
      throw new Error("Shared resource ledger cannot be a symbolic link.");
    }
    this.database = new DatabaseSync(this.ledgerPath);
    chmodSync(this.ledgerPath, 0o600);
    // Declaration observers must not add a multi-second stall to a committed
    // reply. Contention is queued/reported, then admission retries fail closed.
    this.database.exec("PRAGMA busy_timeout=100; PRAGMA synchronous=FULL;");
    this.transaction(() => {
      this.database.exec(`
        CREATE TABLE IF NOT EXISTS owners(
          owner TEXT PRIMARY KEY, pool INTEGER NOT NULL, ram INTEGER,
          pid INTEGER NOT NULL, session TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS policy(key TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
        CREATE INDEX IF NOT EXISTS owner_pool ON owners(pool);
      `);
    });
  }

  private transaction<T>(operation: () => T): T {
    if (this.closed) throw new Error("Shared resource registry is closed.");
    this.database.exec("BEGIN IMMEDIATE");
    try {
      const result = operation();
      this.database.exec("COMMIT");
      return result;
    } catch (error) {
      this.database.exec("ROLLBACK");
      throw error;
    }
  }

  private recordError(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    const changed = message !== this.lastError;
    this.lastError = message;
    if (changed) {
      try {
        this.diagnostic(`Shared resource declaration pending: ${message}`);
      } catch {
        // Diagnostics cannot convert an already durable brain save to failure.
      }
    }
  }

  private upsertOwner(id: string, poolBytes: number, ramBytes?: number): void {
    this.database.prepare(`
      INSERT INTO owners(owner,pool,ram,pid,session) VALUES(?,?,?,?,?)
      ON CONFLICT(owner) DO UPDATE SET
        pool=excluded.pool, ram=COALESCE(excluded.ram,owners.ram),
        pid=excluded.pid, session=excluded.session
    `).run(id, poolBytes, ramBytes ?? null, process.pid, this.session);
  }

  private ramCeiling(): number | null {
    const row = this.database.prepare("SELECT bytes FROM policy WHERE key='ram'").get();
    return row ? byteCount(Number(row.bytes)) : null;
  }

  private restoreRam(selection: SharedRamSelection): void {
    // A failed older operation cannot overwrite a subsequently admitted brain.
    if (selection.epoch !== this.selectionEpoch || this.ramCeiling() !== selection.selectedBytes) return;
    if (selection.previousBytes === null) {
      this.database.prepare("DELETE FROM policy WHERE key='ram'").run();
    } else {
      this.database.prepare("INSERT OR REPLACE INTO policy(key,bytes) VALUES('ram',?)").run(selection.previousBytes);
    }
  }

  private flush(): void {
    if (this.pending.size === 0 && !this.pendingRamRestore) return;
    this.transaction(() => {
      for (const [id, mutation] of this.pending) {
        if (mutation.kind === "removed") {
          this.database.prepare("DELETE FROM owners WHERE owner=?").run(id);
        } else {
          this.upsertOwner(id, mutation.poolBytes!);
        }
      }
      if (this.pendingRamRestore) this.restoreRam(this.pendingRamRestore);
    });
    this.pending.clear();
    this.pendingRamRestore = undefined;
    this.lastError = undefined;
  }

  /** Called only after the repository has durably committed its document. */
  observeSaved(brainId: string, poolBytes: number): void {
    let id: string;
    let pool: number;
    try {
      id = ownerId(brainId);
      pool = byteCount(poolBytes);
    } catch (error) {
      this.invalidDeclaration = error instanceof Error ? error.message : String(error);
      this.recordError(error);
      return;
    }
    try {
      this.pending.set(id, { kind: "saved", poolBytes: pool });
      this.flush();
    } catch (error) {
      this.recordError(error);
    }
  }

  /** Called only after confirmed app-managed removal, never absence guesses. */
  observeRemoved(brainId: string): void {
    let id: string;
    try {
      id = ownerId(brainId);
    } catch (error) {
      this.invalidDeclaration = error instanceof Error ? error.message : String(error);
      this.recordError(error);
      return;
    }
    try {
      this.pending.set(id, { kind: "removed" });
      this.flush();
    } catch (error) {
      this.recordError(error);
    }
  }

  requireSynchronized(): void {
    try {
      if (this.closed) throw new Error("Shared resource registry is closed.");
      if (this.invalidDeclaration) {
        throw new Error(`invalid committed owner declaration: ${this.invalidDeclaration}`);
      }
      this.flush();
      if (this.lastError) {
        // A previously rejected admission is retryable once SQLite is usable;
        // it must not become a permanent lockout with no pending mutation.
        this.transaction(() => this.ramCeiling());
        this.lastError = undefined;
      }
    } catch (error) {
      this.recordError(error);
      throw new Error(`Worker admission paused until shared resource declarations synchronize: ${this.lastError}`);
    }
  }

  /** Reconcile only app-managed UUID identities after the saved-brain scan.
   * Worker-owned stdio declarations and allocation/file records are untouched.
   * A crash between durable removal and its observer must not leave a stale
   * high pool in MAX(pool) on the next launch.
   */
  reconcileSavedOwners(savedBrainIds: readonly string[]): void {
    const saved = new Set(savedBrainIds.map(ownerId));
    this.requireSynchronized();
    this.transaction(() => {
      const rows = this.database.prepare("SELECT owner FROM owners").all() as Array<{ owner: string }>;
      const appUuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/iu;
      for (const { owner } of rows) {
        if (appUuid.test(owner) && !saved.has(owner)) {
          this.database.prepare("DELETE FROM owners WHERE owner=?").run(owner);
        }
      }
    });
  }

  /** Commit the exact already-approved main-process plan; never maximize RAM. */
  selectActiveRuntime(brainId: string, poolBytes: number, systemRamBudgetBytes: number): SharedRamSelection {
    const id = ownerId(brainId);
    const pool = byteCount(poolBytes);
    const budget = byteCount(systemRamBudgetBytes);
    this.requireSynchronized();
    try {
      const selection = this.transaction(() => {
        const previousBytes = this.ramCeiling();
        this.upsertOwner(id, pool, budget);
        this.database.prepare("INSERT OR REPLACE INTO policy(key,bytes) VALUES('ram',?)").run(budget);
        return { epoch: this.selectionEpoch + 1, previousBytes, selectedBytes: budget };
      });
      this.selectionEpoch = selection.epoch;
      return selection;
    } catch (error) {
      this.recordError(error);
      throw error;
    }
  }

  /** Roll back only this selection; a failed restore remains an admission gate. */
  rollbackSelection(selection: SharedRamSelection): void {
    if (selection.epoch !== this.selectionEpoch) return;
    this.pendingRamRestore = selection;
    try {
      this.flush();
    } catch (error) {
      this.recordError(error);
    }
  }

  status(): SharedResourceRegistryStatus {
    const row = this.database.prepare("SELECT COALESCE(MAX(pool),0) AS bytes FROM owners").get();
    return {
      ledgerPath: this.ledgerPath,
      largestConfiguredPoolBytes: byteCount(Number(row!.bytes)),
      globalRamCeilingBytes: this.ramCeiling(),
      pendingDeclarations: this.pending.size + Number(Boolean(this.pendingRamRestore)),
      ...(this.lastError ? { lastError: this.lastError } : {})
    };
  }

  close(): void {
    if (this.closed) return;
    this.database.close();
    this.closed = true;
  }
}
