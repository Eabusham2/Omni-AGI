import { createHash, randomUUID } from "node:crypto";
import { chmod, lstat, mkdtemp, realpath, rmdir, statfs } from "node:fs/promises";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { AuthorizedSpillStorageSnapshot, DeviceResourceSettings, DeviceResourceTopology } from "../shared/deviceResources";
import { spillSelectionSignature } from "./spillSelectionSignature";

export interface SpillPoolSelection { selectionId: string; directories: string[]; }
interface PoolRow { pool_id: string; path: string; backing_path: string; device: string; inode: string;
  backing_device: string; backing_inode: string; physical_group: string; rate: number; }
const identifier = (value: string) => typeof value === "string" && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value);

/** Installation-local authority. Never serialized into a brain or .omni bundle. */
export class SpillPoolAuthority {
  private readonly database: DatabaseSync;
  constructor(ledgerPath: string) {
    this.database = new DatabaseSync(ledgerPath);
    this.database.exec(`PRAGMA busy_timeout=5000;
      CREATE TABLE IF NOT EXISTS drive_pools(pool_id TEXT PRIMARY KEY,path TEXT UNIQUE NOT NULL,backing_path TEXT NOT NULL,device TEXT NOT NULL,inode TEXT NOT NULL,backing_device TEXT NOT NULL,backing_inode TEXT NOT NULL,physical_group TEXT NOT NULL,rate INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS drive_access(owner TEXT NOT NULL,pool_id TEXT NOT NULL,PRIMARY KEY(owner,pool_id));
      CREATE TABLE IF NOT EXISTS drive_selections(token TEXT NOT NULL,pool_id TEXT NOT NULL,PRIMARY KEY(token,pool_id));
      CREATE TABLE IF NOT EXISTS drive_selection_claims(token TEXT PRIMARY KEY,owner TEXT NOT NULL,selection TEXT NOT NULL);`);
    this.database.exec("CREATE TABLE IF NOT EXISTS spill_volumes(device TEXT PRIMARY KEY,physical_group TEXT NOT NULL)");
  }
  close(): void { this.database.close(); }

  observeVolumes(topology: DeviceResourceTopology): void {
    this.transaction(() => {
      for (const device of topology.devices) {
        if (device.kind !== "storage" || !device.filesystemDeviceId || device.memoryGroupId === "darwin:storage-unverified") continue;
        this.database.prepare("INSERT OR REPLACE INTO spill_volumes VALUES(?,?)").run(device.filesystemDeviceId, device.memoryGroupId);
      }
    });
  }

  async planningSnapshot(settings: DeviceResourceSettings, topology: DeviceResourceTopology,
    context: { brainId?: string; selectionId?: string; primaryDirectory: string }): Promise<AuthorizedSpillStorageSnapshot> {
    const rows = (context.brainId
      ? this.database.prepare("SELECT p.* FROM drive_pools p JOIN drive_access a USING(pool_id) WHERE a.owner=?").all(context.brainId)
      : this.database.prepare("SELECT p.* FROM drive_pools p JOIN drive_selections s USING(pool_id) WHERE s.token=?").all(context.selectionId ?? "")) as unknown as PoolRow[];
    const has = (name: string) => Boolean(this.database.prepare("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?").get(name));
    if (has("files") !== has("leases")) throw new Error("Shared storage accounting is incomplete.");
    const count = (value: unknown): number => { const result = Number(value ?? 0); if (!Number.isSafeInteger(result) || result < 0) throw new Error("Storage accounting exceeds exact integer range."); return result; };
    const scalar = (sql: string, ...args: Array<string | number>): number => count(Object.values(this.database.prepare(sql).get(...args) ?? {})[0]);
    const primary = topology.devices.find((device) => device.poolDirectory === context.primaryDirectory)?.memoryGroupId;
    if (!primary) throw new Error("Primary checkpoint volume identity is unavailable.");
    const groups = new Map<string, AuthorizedSpillStorageSnapshot["groups"][number]>();
    for (const path of settings.storageDirectories) {
      const grant = rows.find((row) => row.path === path);
      if (!grant) throw new Error("Choose the requested spill folders on this device before using their capacity.");
      const directory = await lstat(path, { bigint: true });
      const backing = await lstat(grant.backing_path, { bigint: true });
      if (!directory.isDirectory() || directory.isSymbolicLink() || !backing.isDirectory() || backing.isSymbolicLink()
        || String(directory.dev) !== grant.device || String(directory.ino) !== grant.inode
        || String(backing.dev) !== grant.backing_device || String(backing.ino) !== grant.backing_inode) throw new Error("A selected spill folder changed; select it again.");
      const device = topology.devices.find((entry) => entry.poolDirectory === path);
      if (!device || device.filesystemDeviceId !== grant.device || device.memoryGroupId !== grant.physical_group) throw new Error("Selected spill volume identity changed after discovery.");
      const id = "storage:" + createHash("sha256").update(path).digest("hex").slice(0, 24);
      const choice = settings.allocations.find((entry) => entry.deviceId === id) ?? { deviceId: id, enabled: true, mode: "auto" as const, percent: 90, autoIncrease: false };
      if (!choice.enabled || !settings.allowStorageSpill) continue;
      const stats = await statfs(path, { bigint: true });
      const available = Math.max(0, count(stats.bavail * stats.bsize) - 20 * 1024 ** 3);
      let ceiling = Math.floor(available * choice.percent / 100);
      if (context.brainId && has("drive_limits")) {
        const previous = this.database.prepare("SELECT ceiling,signature FROM drive_limits WHERE owner=? AND pool_id=?").get(context.brainId, grant.pool_id) as { ceiling: number; signature: string } | undefined;
        if (previous?.signature === spillSelectionSignature(choice)) ceiling = choice.autoIncrease && settings.autoIncreaseStorage ? Math.max(previous.ceiling, ceiling) : previous.ceiling;
      }
      if (has("drive_limits") && has("owners")) ceiling = Math.max(ceiling, scalar("SELECT COALESCE(MAX(l.ceiling),0) FROM drive_limits l JOIN drive_pools p USING(pool_id) JOIN owners o ON o.owner=l.owner WHERE p.physical_group=? AND l.owner<>?", grant.physical_group, context.brainId ?? ""));
      const existing = groups.get(grant.physical_group);
      groups.set(grant.physical_group, { physicalGroup: grant.physical_group, ceilingBytes: Math.max(existing?.ceilingBytes ?? 0, ceiling),
        availableBytes: Math.max(existing?.availableBytes ?? 0, available), allocatedBytes: 0, pendingBytes: 0,
        unmaterializedPromiseBytes: 0, ownerBytes: 0, primaryVolume: grant.physical_group === primary,
        allocationUnitBytes: Math.max(existing?.allocationUnitBytes ?? 0, count(stats.bsize), process.platform === "win32" ? 65536 : 4096) });
    }
    const fileWhere = "(pool_group=? OR (pool_group IS NULL AND EXISTS(SELECT 1 FROM spill_volumes v WHERE v.physical_group=? AND v.device=substr(f.identity,1,instr(f.identity,':')-1))))";
    if (has("files")) {
      const columns = new Set((this.database.prepare("PRAGMA table_info(files)").all() as Array<{ name: string }>).map((row) => row.name));
      if (!columns.has("pool_group") || !columns.has("physical")) throw new Error("Start the current worker to update storage accounting before extending capacity.");
      for (const group of groups.values()) {
        group.allocatedBytes = scalar(`SELECT COALESCE(SUM(bytes),0) FROM files f WHERE ${fileWhere}`, group.physicalGroup, group.physicalGroup);
        group.unmaterializedPromiseBytes = scalar(`SELECT COALESCE(SUM(MAX(0,bytes-physical)),0) FROM files f WHERE ${fileWhere}`, group.physicalGroup, group.physicalGroup);
        group.pendingBytes = scalar("SELECT COALESCE(SUM(bytes),0) FROM leases WHERE kind='spill' AND state='pending' AND (pool_group=? OR pool_group IS NULL)", group.physicalGroup);
        group.ownerBytes = context.brainId ? scalar(`SELECT COALESCE(SUM(bytes),0) FROM files f WHERE ${fileWhere} AND identity IN (SELECT identity FROM leases WHERE owner=?)`, group.physicalGroup, group.physicalGroup, context.brainId) : 0;
      }
    }
    return { authorized: true, groups: [...groups.values()],
      largestSharedPoolBytes: has("owners") ? scalar("SELECT COALESCE(MAX(pool),0) FROM owners") : 0,
      sharedUsedBytes: has("files") ? scalar("SELECT COALESCE(SUM(bytes),0) FROM files") : 0,
      sharedPendingBytes: has("leases") ? scalar("SELECT COALESCE(SUM(bytes),0) FROM leases WHERE kind='spill' AND state='pending'") : 0,
      ownerBytes: has("files") && context.brainId ? scalar("SELECT COALESCE(SUM(bytes),0) FROM files WHERE identity IN (SELECT identity FROM leases WHERE owner=?)", context.brainId) : 0 };
  }

  private transaction<T>(operation: () => T): T {
    this.database.exec("BEGIN IMMEDIATE");
    try { const value = operation(); this.database.exec("COMMIT"); return value; }
    catch (error) { this.database.exec("ROLLBACK"); throw error; }
  }

  /** Call only with actual native-directory-dialog results and observed topology. */
  async selected(directories: string[], topology: DeviceResourceTopology, brainId?: string): Promise<SpillPoolSelection> {
    if (brainId !== undefined && !identifier(brainId)) throw new Error("Invalid spill owner.");
    const token = randomUUID();
    const pools: PoolRow[] = [];
    for (const supplied of directories) {
      const path = await realpath(supplied);
      const root = await lstat(path, { bigint: true });
      if (!root.isDirectory() || root.isSymbolicLink()) throw new Error("Select a real drive directory.");
      const observed = topology.devices.find((value) => value.kind === "storage" && value.poolDirectory === path);
      if (!observed || observed.readOnly || !observed.memoryGroupId || observed.memoryGroupId === "darwin:storage-unverified") throw new Error("Selected drive identity must be verified before granting spill access.");
      if (observed.filesystemDeviceId !== String(root.dev)) throw new Error("Selected filesystem changed after discovery; select the drive again.");
      let row = this.database.prepare("SELECT * FROM drive_pools WHERE path=?").get(path) as unknown as PoolRow | undefined;
      if (row) {
        const backing = await lstat(row.backing_path, { bigint: true });
        if (!backing.isDirectory() || backing.isSymbolicLink() || dirname(row.backing_path) !== path
          || row.device !== String(root.dev) || row.inode !== String(root.ino)
          || row.backing_device !== String(backing.dev) || row.backing_inode !== String(backing.ino)
          || row.physical_group !== observed.memoryGroupId) throw new Error("The selected spill root changed. Choose a new empty folder on this drive; existing backing data remains untouched.");
        // Keep an existing verified rate when a fresh discovery did not probe.
        row = { ...row, physical_group: observed.memoryGroupId, rate: observed.measurement?.bytesPerSecond ?? row.rate };
      } else {
        const backingPath = await mkdtemp(join(path, ".omni-spill-"));
        try {
          await chmod(backingPath, 0o700);
          const backing = await lstat(backingPath, { bigint: true });
          row = { pool_id: randomUUID(), path, backing_path: backingPath, device: String(root.dev), inode: String(root.ino),
            backing_device: String(backing.dev), backing_inode: String(backing.ino), physical_group: observed.memoryGroupId,
            rate: observed.measurement?.bytesPerSecond ?? 0 };
          this.database.prepare("INSERT INTO drive_pools VALUES(?,?,?,?,?,?,?,?,?)").run(...Object.values(row));
        } catch (error) { await rmdir(backingPath).catch(() => undefined); throw error; }
      }
      pools.push(row);
    }
    this.transaction(() => {
      for (const row of pools) {
        this.database.prepare("UPDATE drive_pools SET physical_group=?,rate=? WHERE pool_id=?").run(row.physical_group, row.rate, row.pool_id);
        if (brainId) this.database.prepare("INSERT OR IGNORE INTO drive_access VALUES(?,?)").run(brainId, row.pool_id);
        else this.database.prepare("INSERT INTO drive_selections VALUES(?,?)").run(token, row.pool_id);
      }
    });
    return { selectionId: token, directories: pools.map((row) => row.path) };
  }

  claim(selectionId: string, brainId: string, requestedDirectories: readonly string[]): void {
    if (!identifier(selectionId) || !identifier(brainId)) throw new Error("Invalid local spill selection.");
    this.transaction(() => {
      const previous = this.database.prepare("SELECT owner,selection FROM drive_selection_claims WHERE token=?").get(selectionId) as { owner: string; selection: string } | undefined;
      if (previous) {
        const claimed = JSON.parse(previous.selection) as Array<{ pool_id: string; path: string }>;
        if (previous.owner !== brainId || claimed.length !== requestedDirectories.length || new Set(requestedDirectories).size !== claimed.length
          || claimed.some((row) => !requestedDirectories.includes(row.path)
            || !this.database.prepare("SELECT 1 FROM drive_access WHERE owner=? AND pool_id=?").get(brainId, row.pool_id))) {
          throw new Error("The spill selection was claimed elsewhere, changed, or revoked; choose it again.");
        }
        return; // Exact same local operation/owner; no new authority is granted.
      }
      const rows = this.database.prepare("SELECT p.pool_id,p.path FROM drive_pools p JOIN drive_selections s USING(pool_id) WHERE s.token=?").all(selectionId) as Array<{ pool_id: string; path: string }>;
      if (!rows.length || rows.length !== requestedDirectories.length
        || new Set(requestedDirectories).size !== rows.length || rows.some((row) => !requestedDirectories.includes(row.path))) throw new Error("The spill selection does not match the requested directories; choose them again.");
      for (const row of rows) this.database.prepare("INSERT OR IGNORE INTO drive_access VALUES(?,?)").run(brainId, row.pool_id);
      this.database.prepare("INSERT INTO drive_selection_claims VALUES(?,?,?)").run(selectionId, brainId, JSON.stringify(rows));
      this.database.prepare("DELETE FROM drive_selections WHERE token=?").run(selectionId);
    });
  }
}
