import { mkdtemp, mkdir, writeFile, readFile, readdir, rm, cp, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { pathToFileURL } from "node:url";
import { DatabaseSync } from "node:sqlite";
import { createPackage } from "@electron/asar";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainStorageOperationSession } from "../src/main/brainStorageOperations";
import { calculateDiskSpaceReport } from "../src/main/diskSpace";
import { readPortableBootstrap } from "../src/main/portableBootstrap";
import { PortableWorkspaceService, type PortableWorkspaceDependencies } from "../src/main/portableWorkspaceService";
import { retirePortableWorkspaceAuthority } from "../src/main/portableWorkspaceAuthority";
import { portableFile, portableSha256, portableTreeSha256, publishPortableDirectory, type PortableDirectoryIdentity, type PortableIoHooks } from "../src/main/portableWorkspaceFiles";
import { portableSavedDataReadOwned, withPortableSavedDataRead } from "../src/main/portableWorkspaceReadScope";
import { checkedPortablePresentation, type PortablePresentationSettings } from "../src/shared/portableWorkspace";
import { DEFAULT_CONFIG } from "../src/shared/types";
import type { BrainDocument, ToolRuntimePreferences, WorkingMemoryResourcePlan } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const presentation: PortablePresentationSettings = { appearance: { schemaVersion: 1, mode: "dark", palette: "aqua", layout: "glass" },
  workspace: { schemaVersion: 1, arrangement: "focus", notificationsEnabled: true, profiles: [{ id: "work", name: "Work", appearance: { schemaVersion: 1, mode: "light", palette: "graphite", layout: "classic" }, arrangement: "stacked", createdAt: "2026-10-15T12:00:00Z", updatedAt: "2026-10-15T12:00:00Z" }] }, theme: null };
const origin = "a".repeat(64), brainId = "00000000-1111-2222-3333-444444444444";
const disk = () => calculateDiskSpaceReport({ diskTotalBytes: 200 * 1024 ** 3, diskFreeBytes: 150 * 1024 ** 3 });
const io: PortableIoHooks = { admit: bytes => { if (bytes > 16 * 1024 ** 2) throw new Error("Synthetic metadata exceeds fixture RAM budget"); }, checkDisk: async () => disk() };
function pe(arch: "x64" | "arm64"): Buffer { const data = Buffer.alloc(70); data.writeUInt16LE(0x5a4d, 0); data.writeUInt32LE(64, 0x3c); data.writeUInt32LE(0x4550, 64); data.writeUInt16LE(arch === "x64" ? 0x8664 : 0xaa64, 68); return data; }
async function runtime(root: string): Promise<void> {
  await mkdir(join(root, "resources", "engine-runtime"), { recursive: true });
  await writeFile(join(root, "Omni AGI Studio.exe"), pe("arm64")); await writeFile(join(root, "resources", "engine-runtime", "omni-engine.exe"), pe("x64"));
  const asar = Buffer.from("Synthetic non-executable ASAR fixture; no Electron application/model is launched."); await writeFile(join(root, "resources", "app.asar"), asar);
  await writeFile(join(root, "resources", "portable-runtime.json"), JSON.stringify({ format: "omni-portable-runtime", version: 1, portableProtocol: 1, appVersion: "1.1.1", sourceCommit: null,
    platform: "win32", architecture: "arm64", workerArchitecture: "x64", executable: "Omni AGI Studio.exe", appAsar: { path: "resources/app.asar", bytes: asar.length, sha256: portableSha256(asar) } }));
}
interface FakePayload { brain: BrainDocument; history: string; context: string; rawSource: string }
function fakeBrain(): BrainDocument {
  // Host DTO only; deliberately not a valid neural checkpoint/constructed brain.
  return { id: brainId, name: "Synthetic saved continuity", originChecksum: origin, readiness: { state: "ready" },
    config: { storagePoolBytes: 123456, storagePoolMode: "manual", idleCognition: true,
      deviceResources: { version: 1, allocations: [], allowStorageSpill: true, autoIncreaseStorage: false, storageDirectories: ["C:\\source-only-drive"] } },
    messages: [{ role: "human", content: "Synthetic private history retained verbatim" }], toolPermissions: [] } as unknown as BrainDocument;
}
function fakeRepository(root: string, importCalls: { count: number }, onExport?: () => Promise<void>): PortableWorkspaceDependencies["repository"] {
  const directory = (id: string) => join(root, id);
  return {
    initialize: async () => { await mkdir(join(root, ".blobs"), { recursive: true }); }, configurePluginPortability: () => undefined, brainDirectory: directory,
    get: async id => JSON.parse(await readFile(join(directory(id), "brain.json"), "utf8")) as BrainDocument,
    save: async brain => { await writeFile(join(directory(brain.id), "brain.json"), JSON.stringify(brain)); return brain; },
    exportBundle: async (id, destination, mode, hooks, guard) => {
      await guard?.(); expect(mode).toBe("private-archive"); await hooks?.checkpoint({ phase: "synthetic-export", label: "No native archive writer in this fake adapter" }); await onExport?.();
      const payload: FakePayload = { brain: JSON.parse(await readFile(join(directory(id), "brain.json"), "utf8")), history: await readFile(join(directory(id), "history.fixture"), "utf8"),
        context: await readFile(join(directory(id), "context.fixture"), "utf8"), rawSource: "Synthetic retained source bytes" };
      await writeFile(destination, JSON.stringify(payload));
    },
    importBundle: async (path, _options, hooks) => {
      importCalls.count++; await hooks?.checkpoint({ phase: "synthetic-import", label: "No neural initialization, inference or native importer in fake adapter" });
      const payload = JSON.parse(await readFile(path, "utf8")) as FakePayload, target = directory(payload.brain.id); await mkdir(target);
      await writeFile(join(target, "history.fixture"), payload.history); await writeFile(join(target, "context.fixture"), payload.context);
      await writeFile(join(root, ".blobs", portableSha256(payload.rawSource)), payload.rawSource);
      await writeFile(join(target, "brain.json"), JSON.stringify(payload.brain)); return payload.brain;
    }
  };
}
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-portable-host-")); roots.push(root);
  const source = join(root, "source"), windows = join(root, "windows"), output = join(root, "portable-output"), host = join(root, "device-host");
  await mkdir(join(source, brainId), { recursive: true }); await mkdir(host); await runtime(windows);
  await writeFile(join(source, brainId, "brain.json"), JSON.stringify(fakeBrain())); await writeFile(join(source, brainId, "history.fixture"), "Full synthetic history, emitted Ponder fixture and completed effect receipt");
  await writeFile(join(source, brainId, "context.fixture"), "Exact synthetic saved cursor and pending progress=3, not neural state");
  const imports = { count: 0 }; let preferences: ToolRuntimePreferences = { approvalTimeoutSeconds: 137 }, fits = true;
  const declared: Array<[string, number]> = [];
  const dependencies: PortableWorkspaceDependencies = { repository: fakeRepository(source, imports), userDataRoot: host, deviceDataRoot: host, io,
    toolsPreferences: () => preferences, setToolsPreferences: async value => { preferences = value; }, assertUnloaded: vi.fn(),
    stagingRepository: path => fakeRepository(path, imports), planDestination: async () => ({ allowed: fits, resources: { sharedStoragePoolBytes: 123456, mandatoryFreeDiskBytes: 20 * 1024 ** 3 } }) as WorkingMemoryResourcePlan,
    declareReservation: (id, bytes) => { declared.push([id, bytes]); },
    retireCopiedAuthority: vi.fn(async (_root, _id, completed, checkpoint) => { if (!completed.includes("synthetic-dormancy")) await checkpoint("synthetic-dormancy"); }) };
  const session = (id: string) => new BrainStorageOperationSession(id, "export", brainId, async () => disk());
  const service = new PortableWorkspaceService(dependencies);
  return { root, source, windows, output, host, imports, dependencies, service, session, declared, preferences: () => preferences, setFits: (value: boolean) => { fits = value; },
    create: (id = "portable-create") => service.create({ brainId, requestId: id, runtimeDirectory: windows, destination: output, presentation }, session(id).hooks),
    destination: async (path = output, localHost = host) => {
      const boot = readPortableBootstrap({ platform: "win32", packaged: true, executablePath: join(path, "Omni AGI Studio.exe"), installedUserDataRoot: localHost })!;
      const repo = fakeRepository(boot.brainRoot, imports); await repo.initialize();
      const deps = { ...dependencies, bootstrap: boot, userDataRoot: boot.userDataRoot, deviceDataRoot: boot.deviceDataRoot, repository: repo };
      return { boot, repo, service: new PortableWorkspaceService(deps), deps };
    } };
}

describe("Windows portable workspace source contract (tiny fake files; no model/package execution)", () => {
  it("copies the chosen runtime, complete synthetic selected archive and typed settings without private runtime data, padding or old device grants", async () => {
    const f = await fixture(); await mkdir(join(f.windows, "data")); await writeFile(join(f.windows, "data", "integration-secrets.json"), "dummy excluded fixture, not a real key");
    const original = await readFile(join(f.source, brainId, "context.fixture")); const result = await f.create();
    expect(result).toMatchObject({ brainId, destinationReviewRequired: true }); expect(await readdir(f.output)).not.toContain("data");
    const settings = JSON.parse(await readFile(join(f.output, "portable", "settings.json"), "utf8")); expect(settings).toEqual({ version: 1, tools: { approvalTimeoutSeconds: 137 }, presentation });
    const manifest = JSON.parse(await readFile(join(f.output, "portable-workspace.json"), "utf8"));
    expect(manifest).toMatchObject({ runtime: { architecture: "arm64", workerArchitecture: "x64" }, reservation: { shippedPaddingBytes: 0, requestedPoolBytes: 123456 }, savedContent: "unsanitized-private-archive" });
    expect(result.bytes).toBeLessThan(20000); expect(await readFile(join(f.source, brainId, "context.fixture"))).toEqual(original);
    expect(await f.create()).toEqual(result); // Lost final ACK returns the same folder, never another export.
  });
  it("metadata-only bootstrap holds admission; explicit cancellable preparation preserves history/context/CAS and original identity, with reviewed destination pool", async () => {
    const f = await fixture(); await f.create(); const dest = await f.destination(); await dest.service.initialize();
    expect(await dest.service.status()).toMatchObject({ state: "preparation-required", brainId }); expect(f.imports.count).toBe(0);
    expect(() => dest.service.assertRuntimeAllowed()).toThrow("review");
    const session = f.session("portable-prepare"); await dest.service.prepare({ requestId: "portable-prepare" }, session.hooks);
    expect(await dest.service.status()).toMatchObject({ state: "review-required", reservation: { allocatedPaddingBytes: 0 } }); expect(f.imports.count).toBe(1);
    expect(await readFile(join(dest.boot.brainRoot, brainId, "history.fixture"), "utf8")).toContain("completed effect receipt");
    expect(await readFile(join(dest.boot.brainRoot, brainId, "context.fixture"), "utf8")).toContain("pending progress=3");
    expect(await readFile(join(dest.boot.brainRoot, ".blobs", portableSha256("Synthetic retained source bytes")), "utf8")).toBe("Synthetic retained source bytes");
    const current = await dest.repo.get(brainId); expect(current.config.deviceResources?.storageDirectories).toEqual([]); expect(current.config.idleCognition).toBe(false);
    const review = await dest.service.reviewDestination(); expect(review).toMatchObject({ copiedExternalAuthorityDormant: true, credentialsTransferred: false, sourceDriveSelectionsDiscarded: true });
    await dest.service.approveDestination(review); expect(() => dest.service.assertRuntimeAllowed()).not.toThrow(); expect(f.declared).toContainEqual([brainId, 123456]);
    const restarted = new PortableWorkspaceService(dest.deps); await restarted.initialize(); await restarted.prepare({ requestId: "after-ack" }, f.session("after-ack").hooks);
    expect(f.imports.count).toBe(1); expect(await restarted.status()).toMatchObject({ state: "ready" });
  });
  it("moving an already used folder preserves newer saved content, requires fresh destination review and never reimports stale cargo", async () => {
    const f = await fixture(); await f.create(); const first = await f.destination(); await first.service.initialize(); await first.service.prepare({ requestId: "first" }, f.session("first").hooks);
    await first.service.approveDestination(await first.service.reviewDestination());
    await writeFile(join(first.boot.brainRoot, brainId, "context.fixture"), "Newer exact synthetic context, progress=9"); await writeFile(join(first.boot.brainRoot, brainId, "history.fixture"), "Newer private history after portable use");
    const moved = join(f.root, "relocated"); await cp(f.output, moved, { recursive: true }); const second = await f.destination(moved, join(f.root, "different-host")); await second.service.initialize();
    expect(second.boot.deviceDataRoot.startsWith(moved)).toBe(false); expect(await second.service.status()).toMatchObject({ state: "preparation-required" });
    await second.service.prepare({ requestId: "moved" }, f.session("moved").hooks);
    expect(f.imports.count).toBe(1); expect(await readFile(join(second.boot.brainRoot, brainId, "context.fixture"), "utf8")).toContain("progress=9");
    expect(await readFile(join(second.boot.brainRoot, brainId, "history.fixture"), "utf8")).toBe("Newer private history after portable use"); expect(() => second.service.assertRuntimeAllowed()).toThrow();
  });
  it("Stop removes only the exact incomplete created folder and retains a non-replayable cancellation receipt", async () => {
    const f = await fixture(), control = new AbortController(); let checks = 0;
    const session = f.session("cancelled"); const hooks = { ...session.hooks, signal: control.signal, checkpoint: async () => { if (++checks === 2) control.abort(new Error("Explicit fixture Stop")); control.signal.throwIfAborted(); } };
    await expect(f.service.create({ brainId, requestId: "cancelled", runtimeDirectory: f.windows, destination: f.output, presentation }, hooks)).rejects.toThrow("Explicit fixture Stop");
    expect(await readdir(f.root)).not.toContain("portable-output"); expect((await readdir(f.root)).filter(name => name.startsWith(".omni-portable-"))).toEqual([]);
    await expect(f.service.create({ brainId, requestId: "cancelled", runtimeDirectory: f.windows, destination: f.output, presentation }, f.session("other").hooks)).rejects.toThrow();
    expect(await readFile(join(f.source, brainId, "history.fixture"), "utf8")).toContain("Full synthetic history");
  });
  it("requires the actual portable-aware runtime receipt and rejects changed ASAR, raw settings/keys and stale resource reviews", async () => {
    const f = await fixture(); expect(() => checkedPortablePresentation({ ...presentation, apiKey: "dummy" })).toThrow("unsupported fields");
    await writeFile(join(f.windows, "resources", "app.asar"), "changed package bytes"); await expect(f.create()).rejects.toThrow("checksum");
    await runtime(f.windows); await f.create("corrected-runtime"); const dest = await f.destination(); await dest.service.initialize(); await dest.service.prepare({ requestId: "prepare" }, f.session("prepare").hooks);
    const review = await dest.service.reviewDestination(); f.setFits(false); await expect(dest.service.approveDestination(review)).rejects.toThrow("no longer fits"); expect(() => dest.service.assertRuntimeAllowed()).toThrow();
  });
  it("streamed payload reads reject links and after-open replacements rather than publish another source", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-fd-")); roots.push(root); await mkdir(join(root, "source")); await mkdir(join(root, "output"));
    await writeFile(join(root, "outside.fixture"), "private synthetic fixture"); await symlink(join(root, "outside.fixture"), join(root, "source", "linked"));
    await expect(portableFile(join(root, "source"), "linked", io)).rejects.toThrow();
    await writeFile(join(root, "source", "payload"), Buffer.alloc(128000, 1)); let replaced = false;
    await expect(portableFile(join(root, "source"), "payload", { ...io, checkDisk: async () => {
      if (!replaced) { replaced = true; await rm(join(root, "source", "payload")); await writeFile(join(root, "source", "payload"), Buffer.alloc(128000, 2)); } return disk();
    } }, join(root, "output", "payload"))).rejects.toThrow("changed");
  });
  it("new device policy storage never bootstraps copied Full permissions and ordinary installer constructors remain unchanged", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-policy-")); roots.push(root); const data = join(root, "data"), device = join(root, "device");
    const repository = new BrainRepository(data, undefined, { toolPermissionRoot: device, bootstrapToolPermissions: false }); await repository.initialize(); await mkdir(repository.brainDirectory(brainId));
    await writeFile(join(repository.brainDirectory(brainId), "brain.json"), "{}");
    const snapshot = await repository.toolPermissionSnapshot(brainId, [{ toolId: "system.shell", label: "Synthetic copied Full", level: "full", updatedAt: "2026-10-15T12:00:00Z" }]);
    expect(snapshot.records.find(row => row.toolId === "system.shell")?.level).not.toBe("full"); expect(await readdir(data)).not.toContain(".host-policy"); expect(await readdir(device)).toContain(".host-policy");
  });
  it("existing host dormancy adapters pause pending learning without altering saved synthetic context or replaying any action", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-dormant-")); roots.push(root); await mkdir(join(root, "learning")); const context = Buffer.from("Synthetic retained cursor, not an actual native checkpoint"); await writeFile(join(root, "context.fixture"), context);
    const database = new DatabaseSync(join(root, "learning", "tasks.sqlite3")); database.exec("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE tasks(id TEXT PRIMARY KEY,body TEXT); CREATE TABLE sources(task_id TEXT,id TEXT,body TEXT);");
    database.prepare("INSERT INTO meta VALUES('brainId',?)").run(brainId); database.prepare("INSERT INTO tasks VALUES(?,?)").run("pending", JSON.stringify({ id: "pending", brainId, state: "running", cursor: 3 })); database.prepare("INSERT INTO sources VALUES(?,?,?)").run("pending", "source", JSON.stringify({ state: "running", traversed: 3 })); database.close();
    const completed: string[] = []; await retirePortableWorkspaceAuthority(root, brainId, completed, async step => { completed.push(step); });
    const check = new DatabaseSync(join(root, "learning", "tasks.sqlite3")); try { expect(JSON.parse(String(check.prepare("SELECT body FROM tasks").get()?.body))).toMatchObject({ state: "paused", cursor: 3 }); expect(JSON.parse(String(check.prepare("SELECT body FROM sources").get()?.body))).toMatchObject({ state: "paused", traversed: 3 }); } finally { check.close(); }
    expect(await readFile(join(root, "context.fixture"))).toEqual(context); expect(completed).toContain("clock");
  });
  it("the afterPack receipt hashes actual tiny synthetic packaged files and records emulated worker architecture, never an execution claim", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-hook-")); roots.push(root); await runtime(root); await rm(join(root, "resources", "portable-runtime.json"));
    const contents = join(root, "synthetic-asar-source"); await mkdir(join(contents, "out", "main"), { recursive: true });
    await writeFile(join(contents, "package.json"), JSON.stringify({ main: "out/main/index.js", version: "1.1.1" }));
    await writeFile(join(contents, "out", "main", "index.js"), '/* Synthetic inert file, never executed: omni-portable-launcher omni-portable-workspace portable-destination.json portable-adoption */');
    await createPackage(contents, join(root, "resources", "app.asar"));
    const path = pathToFileURL(join(process.cwd(), "scripts", "portable-runtime-receipt.mjs")).href;
    const writer = await import(path) as { writeWindowsPortableRuntimeReceipt(input: { directory: string; architecture: string; version: string }): Promise<Record<string, unknown>> };
    const receipt = await writer.writeWindowsPortableRuntimeReceipt({ directory: root, architecture: "arm64", version: "1.1.1" });
    expect(receipt).toMatchObject({ portableProtocol: 1, architecture: "arm64", workerArchitecture: "x64", sourceCommit: null }); expect(Object.keys(receipt)).not.toContain("brainTested");
    expect(readPortableBootstrap({ platform: "darwin", packaged: true, executablePath: join(root, "Omni AGI Studio.exe"), installedUserDataRoot: join(root, "host") })).toBeUndefined();
  });
  it("recovers publication-before-ACK from exact prepared hashes without repeating even the synthetic import", async () => {
    const f = await fixture(); await f.create(); const dest = await f.destination(); let injected = false;
    const dependencies = { ...dest.deps, io: { ...io, checkDisk: async (path: string) => {
      if (!injected && path.includes("portable-adoption") && await readFile(join(dest.boot.brainRoot, brainId, "brain.json")).then(() => true, () => false)) { injected = true; throw new Error("Synthetic lost adoption ACK"); }
      return disk();
    } } };
    const first = new PortableWorkspaceService(dependencies); await first.initialize();
    await expect(first.prepare({ requestId: "lost-ack" }, f.session("lost-ack").hooks)).rejects.toThrow("lost adoption ACK"); expect(f.imports.count).toBe(1);
    const resumed = new PortableWorkspaceService(dependencies); await resumed.initialize(); await resumed.prepare({ requestId: "recover-ack" }, f.session("recover-ack").hooks);
    expect(f.imports.count).toBe(1); expect(await resumed.status()).toMatchObject({ state: "review-required", brainId });
  });
  it("real host-only get/save stay behind the private adoption read scope and cannot bootstrap copied Full grants", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-read-policy-")); roots.push(root);
    const data = join(root, "data"), device = join(root, "device"), policyId = "portable-host-policy-only";
    const repository = new BrainRepository(data, undefined, { toolPermissionRoot: device, bootstrapToolPermissions: false,
      beforeSavedDataAccess: () => { if (!portableSavedDataReadOwned()) throw new Error("Exact portable adoption is not ready"); } });
    await repository.initialize(); await mkdir(repository.brainDirectory(policyId));
    // Host projection only: no engine/origin/checkpoint, create(), native Load or model exists in this fixture.
    const copied = { releaseFormat: "stable-1.0", id: policyId, name: "Inert policy projection", config: { ...DEFAULT_CONFIG },
      toolPermissions: [{ toolId: "system.shell", label: "Copied Full", level: "full", updatedAt: "2026-10-15T12:00:00Z" }] };
    const path = join(repository.brainDirectory(policyId), "brain.json"), raw = JSON.stringify(copied); await writeFile(path, raw);
    await expect(repository.get(policyId)).rejects.toThrow("adoption is not ready");
    expect(await readFile(path, "utf8")).toBe(raw); expect(await readdir(repository.brainDirectory(policyId))).toEqual(["brain.json"]);
    const admitted = await withPortableSavedDataRead(async () => {
      const brain = await repository.get(policyId); expect(Array.isArray(brain.toolPermissions)).toBe(true);
      expect(brain.toolPermissions!.find(row => row.toolId === "system.shell")?.level).not.toBe("full");
      brain.toolPermissions = copied.toolPermissions as BrainDocument["toolPermissions"];
      return repository.save(brain);
    });
    expect(Array.isArray(admitted.toolPermissions)).toBe(true);
    expect(admitted.toolPermissions!.find(row => row.toolId === "system.shell")?.level).not.toBe("full");
    await expect(repository.get(policyId)).rejects.toThrow("adoption is not ready");
    await expect(repository.save(admitted)).rejects.toThrow("adoption is not ready");
    expect(await readdir(device)).toContain(".host-policy"); expect(await readdir(data)).not.toContain(".host-policy");
  });
  it("publication refuses an existing directory and resumes only its exact split prepared tree with the commit file last", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-portable-publish-")); roots.push(root);
    const stage = join(root, "stage"), destination = join(root, "destination"), occupied = join(root, "occupied");
    await mkdir(stage); await mkdir(occupied); await writeFile(join(stage, "a.fixture"), "Exact first payload"); await writeFile(join(stage, "brain.json"), "Exact commit marker");
    const prepared = await portableTreeSha256(stage, io);
    await expect(publishPortableDirectory(stage, occupied, "brain.json", io, async () => undefined)).rejects.toMatchObject({ code: "EEXIST" });
    expect(await readdir(occupied)).toEqual([]);
    let owner: PortableDirectoryIdentity | undefined, checkpoints = 0;
    await expect(publishPortableDirectory(stage, destination, "brain.json", { ...io, checkpoint: async () => {
      if (++checkpoints === 2) throw new Error("Interrupted after one file, before visible commit");
    } }, async value => { owner = value; })).rejects.toThrow("before visible commit");
    expect(await readdir(destination)).toEqual(["a.fixture"]); expect(await readdir(stage)).toEqual(["brain.json"]);
    if (!owner) throw new Error("Expected exact persisted publication directory identity");
    await publishPortableDirectory(stage, destination, "brain.json", io, async () => { throw new Error("Owner must not be rebound"); }, owner, prepared);
    expect(await portableTreeSha256(destination, io)).toBe(prepared); expect(await readdir(stage)).toEqual([]);
  });
});
