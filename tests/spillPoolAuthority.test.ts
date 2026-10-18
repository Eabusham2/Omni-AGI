import { mkdtemp, mkdir, readdir, realpath, rename, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, expect, it } from "vitest";
import { SpillPoolAuthority } from "../src/main/spillPoolAuthority";
import type { DeviceResourceSettings, DeviceResourceTopology } from "../src/shared/deviceResources";
import { createHash } from "node:crypto";
import { spillSelectionSignature } from "../src/main/spillSelectionSignature";

const cleanups: Array<() => Promise<void>> = [];
afterEach(async () => { for (const cleanup of cleanups.splice(0).reverse()) await cleanup(); });
async function fixture() {
  const root = await realpath(await mkdtemp(join(tmpdir(), "omni-spill-authority-")));
  cleanups.push(() => rm(root, { recursive: true, force: true }));
  const paths = [join(root, "drive-a"), join(root, "drive-b")];
  await Promise.all(paths.map((path) => mkdir(path)));
  const identities = await Promise.all(paths.map(async (path) => String((await stat(path, { bigint: true })).dev)));
  const ledger = join(root, "ledger.sqlite3");
  const authority = new SpillPoolAuthority(ledger); cleanups.push(async () => authority.close());
  const reader = new DatabaseSync(ledger); cleanups.push(async () => reader.close());
  const topology: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-07T00:00:00Z",
    platform: "fixture", architecture: "fixture", links: [], warnings: [], measurementRequested: true, neuralModelConstructed: false,
    devices: paths.map((path, index) => ({ id: `storage:fixture-${index}`, label: path, kind: "storage", backend: "filesystem",
      memoryGroupId: `volume:fixture-${index}`, totalBytes: 500 * 1024 ** 3, availableBytes: 100 * 1024 ** 3,
      allocatedBytes: null, reservedBytes: null, unifiedMemory: false, nativeComputeSupported: false,
      poolDirectory: path, filesystemDeviceId: identities[index]! })) };
  return { paths, root, authority, reader, topology };
}

it("stages native selections then grants one new owner exactly once", async () => {
  const { authority, paths, topology, reader } = await fixture();
  const picked = await authority.selected(paths, topology);
  expect(reader.prepare("SELECT * FROM drive_access").all()).toHaveLength(0);
  expect(picked.directories).toEqual(paths);
  authority.claim(picked.selectionId, "new-brain", paths);
  expect(() => authority.claim(picked.selectionId, "new-brain", paths)).not.toThrow();
  expect(reader.prepare("SELECT pool_id FROM drive_access WHERE owner='new-brain'").all()).toHaveLength(2);
  expect(() => authority.claim(picked.selectionId, "imported-brain", paths)).toThrow(/choose it again/);
  expect(reader.prepare("SELECT * FROM drive_access WHERE owner='imported-brain'").all()).toHaveLength(0);
  for (const path of paths) expect((await readdir(path)).every((name) => name.startsWith(".omni-spill-"))).toBe(true);
});

it("does not accept an imported path or mismatched selection as authority", async () => {
  const { authority, paths, topology, reader } = await fixture();
  expect(() => authority.claim("made-up-selection", "imported-brain", paths)).toThrow();
  const picked = await authority.selected(paths, topology);
  expect(() => authority.claim(picked.selectionId, "new-brain", [paths[0]!])).toThrow();
  expect(reader.prepare("SELECT * FROM drive_access").all()).toHaveLength(0);
  authority.claim(picked.selectionId, "new-brain", paths);
  expect(reader.prepare("SELECT * FROM drive_access").all()).toHaveLength(2);
});

it("existing-instance picks do not leave a reusable cross-instance token", async () => {
  const { authority, paths, topology, reader } = await fixture();
  const picked = await authority.selected(paths, topology, "existing-brain");
  expect(reader.prepare("SELECT * FROM drive_access WHERE owner='existing-brain'").all()).toHaveLength(2);
  expect(() => authority.claim(picked.selectionId, "other-brain", paths)).toThrow();
});

it("requires verified group identities and unchanged owned directories", async () => {
  const { authority, paths, topology, reader } = await fixture();
  topology.devices[0]!.memoryGroupId = "darwin:storage-unverified";
  await expect(authority.selected(paths, topology)).rejects.toThrow(/verified/);
  expect(await readdir(paths[0]!)).toHaveLength(0);
  topology.devices[0]!.memoryGroupId = "volume:fixture-0";
  await authority.selected(paths, topology);
  const row = reader.prepare("SELECT backing_path FROM drive_pools WHERE path=?").get(paths[0]!) as { backing_path: string };
  await rename(row.backing_path, `${row.backing_path}-old`);
  await mkdir(row.backing_path);
  await expect(authority.selected(paths, topology)).rejects.toThrow(/root changed/);
});

it("rejects a filesystem replacement between topology discovery and native selection", async () => {
  const { authority, paths, topology } = await fixture();
  topology.devices[0]!.filesystemDeviceId = "0";
  await expect(authority.selected(paths, topology)).rejects.toThrow(/changed after discovery/);
  expect(await readdir(paths[0]!)).toHaveLength(0);
});

it("an idempotent claim cannot restore a revoked owner grant or change directories", async () => {
  const { authority, paths, topology, reader } = await fixture();
  const picked = await authority.selected(paths, topology);
  authority.claim(picked.selectionId, "new-brain", paths);
  expect(() => authority.claim(picked.selectionId, "new-brain", [paths[0]!])).toThrow(/changed/);
  reader.prepare("DELETE FROM drive_access WHERE owner=?").run("new-brain");
  expect(() => authority.claim(picked.selectionId, "new-brain", paths)).toThrow(/revoked/);
  expect(reader.prepare("SELECT * FROM drive_access").all()).toHaveLength(0);
});

it("keeps stored volume ceilings, deduplicates physical groups and reports live ledger pressure separately", async () => {
  const { authority, paths, topology, reader } = await fixture();
  topology.devices.forEach((device) => { device.memoryGroupId = "volume:shared"; });
  const selected = await authority.selected(paths, topology); authority.claim(selected.selectionId, "brain", paths);
  reader.exec("CREATE TABLE owners(owner TEXT PRIMARY KEY,pool INTEGER); CREATE TABLE drive_limits(owner TEXT,pool_id TEXT,ceiling INTEGER,signature TEXT); CREATE TABLE files(identity TEXT PRIMARY KEY,bytes INTEGER,pool_group TEXT,physical INTEGER); CREATE TABLE leases(owner TEXT,identity TEXT,bytes INTEGER,kind TEXT,state TEXT,pool_group TEXT);");
  reader.prepare("INSERT INTO owners VALUES(?,?)").run("brain", 100_000);
  reader.prepare("INSERT INTO owners VALUES(?,?)").run("other", 200_000);
  const settings: DeviceResourceSettings = { version: 1, allowStorageSpill: true, autoIncreaseStorage: false, storageDirectories: paths,
    allocations: paths.map((path) => ({ deviceId: "storage:" + createHash("sha256").update(path).digest("hex").slice(0, 24), enabled: true, mode: "manual", percent: 50, autoIncrease: false })) };
  for (const [index, path] of paths.entries()) {
    const pool = reader.prepare("SELECT pool_id FROM drive_pools WHERE path=?").get(path) as { pool_id: string };
    reader.prepare("INSERT INTO drive_limits VALUES(?,?,?,?)").run("brain", pool.pool_id, 60_000 + index * 10_000, spillSelectionSignature(settings.allocations[index]!));
  }
  reader.prepare("INSERT INTO files VALUES(?,?,?,?)").run("file", 16_000, "volume:shared", 4_000);
  reader.prepare("INSERT INTO leases VALUES(?,?,?,?,?,?)").run("brain", "file", 0, "spill", "allocated", "volume:shared");
  reader.prepare("INSERT INTO leases VALUES(?,?,?,?,?,?)").run("other", null, 8_000, "spill", "pending", "volume:shared");
  const result = await authority.planningSnapshot(settings, topology, { brainId: "brain", primaryDirectory: paths[0]! });
  expect(result.groups).toHaveLength(1);
  expect(result.groups[0]).toMatchObject({ ceilingBytes: 70_000, allocatedBytes: 16_000, pendingBytes: 8_000,
    ownerBytes: 16_000, unmaterializedPromiseBytes: 12_000, primaryVolume: true });
  expect(result.largestSharedPoolBytes).toBe(200_000);
  expect(result.sharedUsedBytes).toBe(16_000);
  expect(result.sharedPendingBytes).toBe(8_000);
  await expect(authority.planningSnapshot(settings, topology, { brainId: "ungranted", primaryDirectory: paths[0]! })).rejects.toThrow(/Choose/);
});

it("uses the same exact float signature as the native quota consumer", () => {
  expect(spillSelectionSignature({ deviceId: "storage:a", enabled: true, mode: "manual", percent: 37.5, autoIncrease: false }))
    .toBe("native-spill-selection-1:1bee182426d8261e2e7dab20a3fd03738032066a13b15018711acddfe3b14e23");
});
