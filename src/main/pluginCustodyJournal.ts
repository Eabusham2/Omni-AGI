import type { DatabaseSync } from "node:sqlite";

/** Keep every acknowledgement durable without recreating a rollback-journal
 * file for every small plugin RPC. FULL still syncs every committed WAL
 * transaction; SQLite's default automatic checkpoints remain enabled.
 *
 * Older supported Node builds may carry SQLite's pre-2026 WAL-reset race.
 * Use persistent rollback journals there, never lower synchronous or assume
 * an unavailable VFS has accepted WAL. Portable copies already use a
 * consistent SQLite snapshot, not a raw main-file copy.
 * https://sqlite.org/wal.html
 */
export function configurePluginCustodyJournal(db: DatabaseSync): void {
  const version = db.prepare("SELECT sqlite_version() AS version").get()?.version;
  const parsed = typeof version === "string" ? /^(\d+)\.(\d+)\.(\d+)$/.exec(version) : null;
  const major = Number(parsed?.[1]), minor = Number(parsed?.[2]), patch = Number(parsed?.[3]);
  const fixed = major > 3 || major === 3 && (minor > 51 || minor === 51 && patch >= 3 || minor === 50 && patch >= 7 || minor === 44 && patch >= 6);
  // A VFS without WAL support returns its existing safe journal mode. Do not
  // turn that limitation into weaker durability or a new activation gate.
  db.exec(`PRAGMA journal_mode=${fixed ? "WAL" : "PERSIST"}; PRAGMA synchronous=FULL;`);
}
