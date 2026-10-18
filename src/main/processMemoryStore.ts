import { DatabaseSync } from "node:sqlite";
import { lstat, mkdir } from "node:fs/promises";
import { join } from "node:path";
import type { ProcessMemoryReceipt } from "../shared/processMemory";
import { memoryId, processMemoryCanonical } from "./processMemoryFormat";

/** Installation-local host authority. Never placed in a brain/portable archive. */
export class ProcessMemoryStore {
  private constructor(readonly db: DatabaseSync) {}
  static open(root: string): Promise<ProcessMemoryStore>;
  static open(root: string, create: false): Promise<ProcessMemoryStore | undefined>;
  static async open(root: string, create = true): Promise<ProcessMemoryStore | undefined> {
    if (create) await mkdir(root, { recursive: true, mode: 0o700 });
    const path = join(root, "process-memory.sqlite3");
    try { const item = await lstat(path); if (!item.isFile() || item.isSymbolicLink()) throw new Error("Host-memory settings must be a real local file."); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; if (!create) return undefined; }
    const parent = await lstat(root);
    if (!parent.isDirectory() || parent.isSymbolicLink()) throw new Error("Host-memory settings require a real installation directory.");
    const db = new DatabaseSync(path, { readOnly: !create });
    try {
      db.exec("PRAGMA cache_size=-64; PRAGMA mmap_size=0; PRAGMA busy_timeout=5000;");
      if (create) db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; " +
          "CREATE TABLE IF NOT EXISTS settings(brain_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,enabled INTEGER NOT NULL,updated_at TEXT NOT NULL); " +
          "CREATE TABLE IF NOT EXISTS writes(brain_id TEXT NOT NULL,request_id TEXT NOT NULL,argument_sha TEXT NOT NULL,phase TEXT NOT NULL,receipt TEXT,PRIMARY KEY(brain_id,request_id));");
      return new ProcessMemoryStore(db);
    } catch (error) { db.close(); throw error; }
  }
  close(): void { this.db.close(); }
  preferences(brainId: string): { enabled: boolean; revision: number; updatedAt: string | null } {
    const row = this.db.prepare("SELECT revision,enabled,updated_at FROM settings WHERE brain_id=?").get(memoryId(brainId));
    if (!row) return { enabled: false, revision: 0, updatedAt: null };
    if (!Number.isSafeInteger(row.revision) || Number(row.revision) < 1 || ![0, 1].includes(Number(row.enabled)) ||
        typeof row.updated_at !== "string" || !Number.isFinite(Date.parse(row.updated_at))) throw new Error("Host-memory settings are corrupt; access remains unavailable.");
    return { enabled: row.enabled === 1, revision: Number(row.revision), updatedAt: row.updated_at };
  }
  update(brainId: string, expectedRevision: number, enabled: boolean): void {
    if (!Number.isSafeInteger(expectedRevision) || expectedRevision < 0 || expectedRevision >= Number.MAX_SAFE_INTEGER || typeof enabled !== "boolean") throw new Error("Invalid host-memory preference update.");
    this.db.exec("BEGIN IMMEDIATE");
    try {
      if (this.preferences(brainId).revision !== expectedRevision) throw new Error("Host-memory preferences changed; reload them before saving.");
      this.db.prepare("INSERT INTO settings VALUES(?,?,?,?) ON CONFLICT(brain_id) DO UPDATE SET revision=excluded.revision,enabled=excluded.enabled,updated_at=excluded.updated_at")
        .run(brainId, expectedRevision + 1, enabled ? 1 : 0, new Date().toISOString());
      this.db.exec("COMMIT");
    } catch (error) { this.db.exec("ROLLBACK"); throw error; }
  }
  write(brainId: string, requestId: string): { argumentSha256: string; phase: string; receipt?: ProcessMemoryReceipt } | undefined {
    const row = this.db.prepare("SELECT argument_sha,phase,receipt FROM writes WHERE brain_id=? AND request_id=?").get(memoryId(brainId), memoryId(requestId));
    if (!row) return undefined;
    if (typeof row.argument_sha !== "string" || !/^[a-f0-9]{64}$/.test(row.argument_sha) || !["prepared", "dispatched", "terminal"].includes(String(row.phase)) ||
        (row.receipt !== null && (typeof row.receipt !== "string" || row.receipt.length > 65536))) throw new Error("The host-memory effect journal is invalid; no write may be retried.");
    return { argumentSha256: row.argument_sha, phase: String(row.phase), ...(row.receipt ? { receipt: JSON.parse(String(row.receipt)) as ProcessMemoryReceipt } : {}) };
  }
  prepareWrite(brainId: string, requestId: string, argumentSha: string): boolean {
    return this.db.prepare("INSERT OR IGNORE INTO writes VALUES(?,?,?,'prepared',NULL)").run(memoryId(brainId), memoryId(requestId), argumentSha).changes === 1;
  }
  dispatched(brainId: string, requestId: string): void {
    if (this.db.prepare("UPDATE writes SET phase='dispatched' WHERE brain_id=? AND request_id=? AND phase='prepared'").run(brainId, requestId).changes !== 1) throw new Error("The exact memory write is no longer prepared.");
  }
  terminal(brainId: string, requestId: string, receipt: ProcessMemoryReceipt): void {
    const text = processMemoryCanonical(receipt);
    if (text.length > 65536 || receipt.dataBase64 !== undefined) throw new Error("Raw process memory must not be persisted in the write journal.");
    if (this.db.prepare("UPDATE writes SET phase='terminal',receipt=? WHERE brain_id=? AND request_id=?").run(text, brainId, requestId).changes !== 1) throw new Error("The memory write receipt lost its exact owner.");
  }
}
