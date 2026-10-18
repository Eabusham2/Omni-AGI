import { randomUUID } from "node:crypto";
import { link, lstat, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";
import type { BrainDocument, ToolRuntimePreferences, WorkingMemoryResourcePlan } from "../shared/types";
import { PORTABLE_MANIFEST, PORTABLE_MARKER, checkedPortablePresentation, checkedPortableSettings,
  type PortableDestinationApproval, type PortableDestinationReview, type PortableFile, type PortablePresentationSettings,
  type PortableWorkspaceCreateResult, type PortableWorkspaceManifest, type PortableWorkspaceSettings, type PortableWorkspaceStatus } from "../shared/portableWorkspace";
import type { BrainRepository, BrainPluginPortabilityHooks } from "./brainRepository";
import type { BrainStorageOperationHooks } from "./brainStorageOperations";
import type { OmniPluginPackage } from "../shared/plugins";
import type { PortableBootstrap } from "./portableBootstrap";
import { withPortableSavedDataRead } from "./portableWorkspaceReadScope";
import { portableExists, portableFile, portableFiles, portableOwnsDirectory, portableReadJson, portableRelative, portableSha256, portableTreeSha256, portableWriteJson, publishPortableDirectory, windowsPortableArchitecture, type PortableDirectoryIdentity, type PortableIoHooks } from "./portableWorkspaceFiles";

interface RuntimeReceipt { format: "omni-portable-runtime"; version: 1; portableProtocol: 1; appVersion: string; platform: "win32";
  architecture: "x64" | "arm64"; workerArchitecture: "x64" | "arm64"; executable: string; appAsar: PortableFile }
type PortableRepository = Pick<BrainRepository, "get" | "save" | "exportBundle" | "importBundle" | "brainDirectory" | "initialize" | "configurePluginPortability">;
export interface PortableWorkspaceDependencies {
  repository: PortableRepository;
  userDataRoot: string;
  /** Device-private and not part of the copied folder. */
  deviceDataRoot: string;
  bootstrap?: PortableBootstrap;
  io: PortableIoHooks;
  toolsPreferences(): ToolRuntimePreferences;
  setToolsPreferences(value: ToolRuntimePreferences): Promise<void>;
  assertUnloaded(brainId: string): void;
  stagingRepository(root: string): PortableRepository;
  pluginPortability?: BrainPluginPortabilityHooks;
  themePackage?(selection: NonNullable<PortablePresentationSettings["theme"]>): Promise<OmniPluginPackage>;
  installTheme?(value: unknown): Promise<void>;
  planDestination(brain: BrainDocument): Promise<WorkingMemoryResourcePlan>;
  declareReservation(brainId: string, poolBytes: number): void;
  /** Existing per-workspace dormant-copy paths, called before any scheduler is admitted. */
  retireCopiedAuthority(root: string, brainId: string, completed: readonly string[], checkpoint: (step: string) => Promise<void>, operation: BrainStorageOperationHooks): Promise<void>;
}
export interface PortableWorkspaceChosenPaths { brainId: string; requestId: string; runtimeDirectory: string; destination: string; presentation: PortablePresentationSettings }
interface CreateReceipt { format: "omni-portable-create-receipt"; version: 1; requestSha256: string; state: "preparing" | "complete" | "failed" | "cancelled";
  destination: string; workspaceId: string; stage: string; manifestSha256?: string; result?: PortableWorkspaceCreateResult; error?: string;
  outputOwner?: PortableDirectoryIdentity; preparedTreeSha256?: string }
interface AdoptionState { manifestSha256: string; brainId: string; preparedSha256?: string; published: boolean; outputOwner?: PortableDirectoryIdentity }
interface DestinationReceipt { format: "omni-portable-destination"; version: 1; workspaceId: string; manifestSha256: string; locationSha256: string;
  brainId: string; originChecksum: string; completedSteps: string[]; approved: boolean; approvedPoolBytes?: number; sourceDriveSelectionsDiscarded: boolean }
const topFiles = new Set(["Omni AGI Studio.exe", "chrome_100_percent.pak", "chrome_200_percent.pak", "d3dcompiler_47.dll", "dxcompiler.dll", "dxil.dll", "ffmpeg.dll", "icudtl.dat", "libEGL.dll", "libGLESv2.dll", "resources.pak", "snapshot_blob.bin", "v8_context_snapshot.bin", "vk_swiftshader.dll", "vk_swiftshader_icd.json", "vulkan-1.dll", "LICENSE", "LICENSES.chromium.html"]);
const resourceNames = new Set(["app.asar", "app.asar.unpacked", "app-update.yml", "portable-runtime.json", "engine-runtime", "engine", "catalog", "tools", "licenses", "LICENSE.md", "COMMERCIAL_LICENSE.md", "THIRD_PARTY_NOTICES.md"]);
const idPattern = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
function id(value: string): string { if (typeof value !== "string" || idPattern.exec(value)?.[0] !== value) throw new Error("Invalid portable request/brain identity."); return value; }
function file(value: PortableFile): PortableFile { portableRelative(value?.path); if (!Number.isSafeInteger(value.bytes) || value.bytes < 0 || typeof value.sha256 !== "string" || /^[a-f0-9]{64}$/.exec(value.sha256)?.[0] !== value.sha256) throw new Error("Invalid portable file descriptor."); return value; }
function equal(actual: PortableFile, expected: PortableFile): void { if (actual.path !== expected.path || actual.sha256 !== expected.sha256 || actual.bytes !== expected.bytes) throw new Error(`Portable payload checksum failed for ${expected.path}.`); }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error); }

/** Creation uses exact saved archives and only main-chosen paths; no native Load. */
export class PortableWorkspaceService {
  private current: PortableWorkspaceStatus = { portable: false, state: "installed" };
  private initialized?: Promise<void>;
  private manifest?: PortableWorkspaceManifest;
  private settings?: PortableWorkspaceSettings;
  private destination?: DestinationReceipt;
  private review?: { value: PortableDestinationReview; configSha256: string; poolBytes: number };
  private readonly creating = new Map<string, { sha: string; promise: Promise<PortableWorkspaceCreateResult> }>();
  private preparing?: { requestId: string; promise: Promise<PortableWorkspaceStatus> };
  constructor(private readonly dependencies: PortableWorkspaceDependencies) {
    if (dependencies.bootstrap) this.current = { portable: true, state: "review-required", workspaceId: dependencies.bootstrap.workspaceId };
  }
  assertRuntimeAllowed(): void { if (this.current.portable && this.current.state !== "ready") throw new Error(this.current.error ?? "This portable folder waits for current-device destination and resource review. Its saved brain remains intact and Unloaded."); }
  savedDataReady(): boolean { return !this.current.portable || ["review-required", "ready"].includes(this.current.state) && this.destination?.completedSteps.includes("adoption-complete") === true; }
  async initialize(): Promise<void> {
    if (!this.initialized) this.initialized = this.initializeDestination().catch(error => { this.current = { ...this.current, state: "blocked", error: errorMessage(error) }; throw error; });
    return this.initialized;
  }
  async status(): Promise<PortableWorkspaceStatus> { return structuredClone(this.current); }

  async create(input: PortableWorkspaceChosenPaths, operation: BrainStorageOperationHooks): Promise<PortableWorkspaceCreateResult> {
    id(input.brainId); id(input.requestId); const presentation = checkedPortablePresentation(input.presentation);
    const requested = { ...input, presentation, runtimeDirectory: resolve(input.runtimeDirectory), destination: resolve(input.destination) };
    if (!isAbsolute(input.runtimeDirectory) || !isAbsolute(input.destination)) throw new Error("Portable source/destination must come from the main-process picker.");
    const sha = portableSha256(JSON.stringify(requested)), prior = this.creating.get(input.requestId);
    if (prior) { if (prior.sha !== sha) throw new Error("Portable request ID changed operands."); return prior.promise; }
    const promise = this.createOwned(requested, sha, operation).finally(() => { if (this.creating.get(input.requestId)?.promise === promise) this.creating.delete(input.requestId); });
    this.creating.set(input.requestId, { sha, promise }); return promise;
  }
  private async createOwned(input: PortableWorkspaceChosenPaths, requestSha256: string, operation: BrainStorageOperationHooks): Promise<PortableWorkspaceCreateResult> {
    const privateRoot = join(this.dependencies.deviceDataRoot, "portable-creation"); await mkdir(privateRoot, { recursive: true, mode: 0o700 });
    const receiptName = `${portableSha256(input.requestId)}.json`, receiptPath = join(privateRoot, receiptName);
    if (await portableExists(receiptPath)) {
      const retained = await portableReadJson<CreateReceipt>(privateRoot, receiptName, this.dependencies.io);
      if (retained.format !== "omni-portable-create-receipt" || retained.version !== 1 || retained.requestSha256 !== requestSha256 || retained.destination !== input.destination) throw new Error("Portable request receipt belongs to different operands.");
      if (!["preparing", "complete"].includes(retained.state)) throw new Error(retained.error ?? "The earlier portable request is terminal and was not replayed.");
      if (retained.state === "preparing" && retained.outputOwner && retained.preparedTreeSha256 && retained.result &&
          !await portableExists(join(input.destination, "Omni AGI Studio.exe"))) {
        if (retained.stage !== join(dirname(input.destination), `.omni-portable-${retained.workspaceId}`)) throw new Error("Portable publication staging path changed.");
        const recoveringIo = { ...this.dependencies.io, signal: operation.signal, checkDisk: (path: string, bytes: number) => operation.checkDisk(path, bytes), checkpoint: () => operation.checkpoint() };
        await publishPortableDirectory(retained.stage, input.destination, "Omni AGI Studio.exe", recoveringIo, async () => undefined, retained.outputOwner, retained.preparedTreeSha256);
      }
      if (retained.manifestSha256 && await portableExists(join(input.destination, PORTABLE_MARKER))) {
        const marker = await portableReadJson<{ id: string; manifestSha256: string }>(input.destination, PORTABLE_MARKER, this.dependencies.io);
        if (marker.id !== retained.workspaceId || marker.manifestSha256 !== retained.manifestSha256 || !retained.result) throw new Error("Portable publication differs from its durable creation intent.");
        if ((await portableFile(input.destination, PORTABLE_MANIFEST, this.dependencies.io)).sha256 !== retained.manifestSha256) throw new Error("Published portable manifest changed after its creation receipt.");
        await this.readRuntime(input.destination);
        retained.state = "complete"; await portableWriteJson(receiptPath, retained, this.dependencies.io); return retained.result;
      }
      throw new Error(retained.error ?? "This earlier portable creation has no complete publication receipt. Review its preserved intent and use a new request; it was not replayed.");
    }
    if (await portableExists(input.destination)) throw new Error("Portable export creates a new folder and never overwrites an existing destination.");
    const rel = relative(input.runtimeDirectory, input.destination); if (!rel || !rel.startsWith("..") && !isAbsolute(rel)) throw new Error("Portable output cannot be inside its source runtime.");
    const io: PortableIoHooks = { ...this.dependencies.io, signal: operation.signal, checkDisk: (path, bytes) => operation.checkDisk(path, bytes), checkpoint: () => operation.checkpoint() };
    const workspaceId = randomUUID(), stage = join(dirname(input.destination), `.omni-portable-${workspaceId}`);
    const receipt: CreateReceipt = { format: "omni-portable-create-receipt", version: 1, requestSha256, state: "preparing", destination: input.destination, workspaceId, stage };
    await portableWriteJson(receiptPath, receipt, this.dependencies.io);
    let published = false, stageCreated = false, outputOwner: PortableDirectoryIdentity | undefined;
    try {
      this.dependencies.assertUnloaded(input.brainId); operation.signal.throwIfAborted();
      const brain = await this.dependencies.repository.get(input.brainId, false);
      if (brain.readiness.state !== "ready" || !brain.originChecksum) throw new Error("Portable export requires a completed native brain with its immutable origin.");
      const runtime = await this.readRuntime(input.runtimeDirectory, io);
      await mkdir(stage, { recursive: false, mode: 0o700 }); stageCreated = true; await mkdir(join(stage, "portable"), { mode: 0o700 });
      const names: string[] = [];
      for (const entry of await readdir(input.runtimeDirectory, { withFileTypes: true })) {
        if (["data", "portable", PORTABLE_MARKER, PORTABLE_MANIFEST].includes(entry.name)) continue;
        if (entry.isFile() && topFiles.has(entry.name)) names.push(entry.name);
        else if (entry.isDirectory() && ["locales", "resources"].includes(entry.name)) {
          if (entry.name === "resources") for (const resource of await readdir(join(input.runtimeDirectory, "resources"))) if (!resourceNames.has(resource)) throw new Error(`Unsupported Windows resources entry ${resource}.`);
          for await (const name of portableFiles(input.runtimeDirectory, entry.name)) { this.dependencies.io.admit((names.length + 1) * 256); names.push(name); }
        } else throw new Error(`Select a clean Windows application folder; unsupported runtime payload ${entry.name}.`);
      }
      let bytes = 0; const files: PortableFile[] = [];
      for (const name of names.sort()) {
        const copied = await portableFile(input.runtimeDirectory, name, io, join(stage, ...name.split("/"))); files.push(copied); bytes += copied.bytes;
        await operation.checkpoint({ phase: "portable-runtime", label: "Copying the selected Windows application", filesCompleted: files.length, filesTotal: names.length,
          logicalBytesCompleted: bytes, physicalBytesAdded: bytes });
      }
      await this.readRuntime(stage, io);
      const bundlePath = join(stage, "portable", "selected.omni");
      this.dependencies.assertUnloaded(input.brainId);
      await this.dependencies.repository.exportBundle(input.brainId, bundlePath, "private-archive", operation, () => this.dependencies.assertUnloaded(input.brainId));
      const bundle = await portableFile(stage, "portable/selected.omni", io); bytes += bundle.bytes;
      const settings = checkedPortableSettings({ version: 1, tools: this.dependencies.toolsPreferences(), presentation: input.presentation });
      const settingsFile = await portableWriteJson(join(stage, "portable", "settings.json"), settings, io, true); settingsFile.path = "portable/settings.json"; bytes += settingsFile.bytes;
      let themePackage: PortableFile | undefined;
      if (settings.presentation.theme) {
        if (!this.dependencies.themePackage) throw new Error("The selected immutable theme package cannot be copied by this installation.");
        const value = await this.dependencies.themePackage(settings.presentation.theme);
        themePackage = await portableWriteJson(join(stage, "portable", "theme.json"), value, io, true); themePackage.path = "portable/theme.json"; bytes += themePackage.bytes;
      }
      const manifest: PortableWorkspaceManifest = { format: "omni-portable-workspace", version: 1, id: workspaceId, createdAt: new Date().toISOString(),
        runtime: { platform: "win32", architecture: runtime.architecture, workerArchitecture: runtime.workerArchitecture, executable: "Omni AGI Studio.exe", files },
        selected: { brainId: brain.id, name: brain.name, originChecksum: brain.originChecksum, bundle }, settings: settingsFile, ...(themePackage ? { themePackage } : {}),
        authority: "destination-review-required", reservation: { policy: "destination-local-planner", requestedPoolBytes: brain.config.storagePoolBytes, mode: brain.config.storagePoolMode, shippedPaddingBytes: 0 }, savedContent: "unsanitized-private-archive" };
      const manifestFile = await portableWriteJson(join(stage, PORTABLE_MANIFEST), manifest, io, true);
      const markerFile = await portableWriteJson(join(stage, PORTABLE_MARKER), { format: "omni-portable-launcher", version: 1, id: workspaceId, manifest: PORTABLE_MANIFEST, manifestSha256: manifestFile.sha256 }, io, true);
      const result: PortableWorkspaceCreateResult = { destination: input.destination, workspaceId, brainId: brain.id, files: files.length + 4 + (themePackage ? 1 : 0), bytes, destinationReviewRequired: true };
      receipt.manifestSha256 = manifestFile.sha256; receipt.result = result;
      receipt.preparedTreeSha256 = portableSha256(JSON.stringify([...files, bundle, settingsFile, ...(themePackage ? [themePackage] : []), manifestFile, markerFile].sort((a, b) => a.path.localeCompare(b.path))));
      await portableWriteJson(receiptPath, receipt, this.dependencies.io);
      await operation.checkpoint({ phase: "portable-publication", label: "Publishing the verified portable folder", filesCompleted: result.files, filesTotal: result.files, logicalBytesCompleted: bytes, logicalBytesTotal: bytes, physicalBytesAdded: bytes });
      operation.signal.throwIfAborted(); if (await portableExists(input.destination)) throw new Error("Portable destination appeared before publication; nothing was overwritten.");
      await publishPortableDirectory(stage, input.destination, "Omni AGI Studio.exe", io, async owner => {
        outputOwner = owner; receipt.outputOwner = owner; await portableWriteJson(receiptPath, receipt, this.dependencies.io);
      }); published = true;
      // The folder is now real. A late Stop never converts it into a non-effect.
      receipt.state = "complete"; await portableWriteJson(receiptPath, receipt, this.dependencies.io); return result;
    } catch (error) {
      if (published && receipt.result) return receipt.result;
      receipt.state = operation.signal.aborted ? "cancelled" : "failed"; receipt.error = errorMessage(error);
      await portableWriteJson(receiptPath, receipt, this.dependencies.io).catch(() => undefined); throw error;
    } finally {
      if (!published && outputOwner && await portableOwnsDirectory(input.destination, outputOwner)) await rm(input.destination, { recursive: true, force: true });
      if (stageCreated) await rm(stage, { recursive: true, force: true });
    }
  }
  private async readRuntime(root: string, io = this.dependencies.io): Promise<RuntimeReceipt> {
    const runtime = await portableReadJson<RuntimeReceipt>(root, "resources/portable-runtime.json", io);
    if (runtime.format !== "omni-portable-runtime" || runtime.version !== 1 || runtime.portableProtocol !== 1 || runtime.platform !== "win32" || runtime.executable !== "Omni AGI Studio.exe" || !["x64", "arm64"].includes(runtime.architecture) || !["x64", "arm64"].includes(runtime.workerArchitecture) || runtime.appAsar?.path !== "resources/app.asar") throw new Error("The selected Windows application does not declare the supported portable protocol.");
    equal(await portableFile(root, runtime.appAsar.path, io), file(runtime.appAsar));
    if (await windowsPortableArchitecture(root, runtime.executable) !== runtime.architecture || await windowsPortableArchitecture(root, "resources/engine-runtime/omni-engine.exe") !== runtime.workerArchitecture) throw new Error("Portable runtime architecture receipt differs from its executable bytes.");
    return runtime;
  }
  private async initializeDestination(): Promise<void> {
    const boot = this.dependencies.bootstrap; if (!boot) return;
    await mkdir(this.dependencies.deviceDataRoot, { recursive: true, mode: 0o700 });
    const descriptor = await portableFile(boot.root, PORTABLE_MANIFEST, this.dependencies.io);
    if (descriptor.sha256 !== boot.manifestSha256) throw new Error("Portable manifest differs from its launcher marker.");
    const manifest = await portableReadJson<PortableWorkspaceManifest>(boot.root, PORTABLE_MANIFEST, this.dependencies.io, descriptor);
    if (manifest.format !== "omni-portable-workspace" || manifest.version !== 1 || manifest.id !== boot.workspaceId || manifest.authority !== "destination-review-required" || manifest.savedContent !== "unsanitized-private-archive" || manifest.selected?.bundle?.path !== "portable/selected.omni" || manifest.settings?.path !== "portable/settings.json" || manifest.reservation?.policy !== "destination-local-planner" || manifest.reservation.shippedPaddingBytes !== 0 || !["auto", "manual"].includes(manifest.reservation.mode) || !Number.isSafeInteger(manifest.reservation.requestedPoolBytes) || manifest.reservation.requestedPoolBytes < 0 || typeof manifest.selected.originChecksum !== "string" || /^[a-f0-9]{64}$/.exec(manifest.selected.originChecksum)?.[0] !== manifest.selected.originChecksum) throw new Error("Portable workspace has an unsupported manifest.");
    id(manifest.selected.brainId); file(manifest.selected.bundle); file(manifest.settings);
    await this.readRuntime(boot.root); this.manifest = manifest;
    this.settings = checkedPortableSettings(await portableReadJson(boot.root, manifest.settings.path, this.dependencies.io, manifest.settings));
    this.current = { ...this.current, brainId: manifest.selected.brainId, name: manifest.selected.name, presentation: this.settings.presentation, settingsSha256: manifest.settings.sha256,
      reservation: { requestedPoolBytes: manifest.reservation.requestedPoolBytes, admittedPoolBytes: null, mandatoryFreeDiskBytes: 20 * 1024 ** 3, kind: "logical-shared-pool", allocatedPaddingBytes: 0 } };
    const receiptPath = join(this.dependencies.deviceDataRoot, "portable-destination.json");
    if (await portableExists(receiptPath)) {
      const retained = await portableReadJson<DestinationReceipt>(this.dependencies.deviceDataRoot, "portable-destination.json", this.dependencies.io);
      if (retained.format !== "omni-portable-destination" || retained.version !== 1 || retained.workspaceId !== boot.workspaceId || retained.manifestSha256 !== boot.manifestSha256 || retained.locationSha256 !== boot.locationSha256 || retained.brainId !== manifest.selected.brainId || retained.originChecksum !== manifest.selected.originChecksum || !Array.isArray(retained.completedSteps) || typeof retained.approved !== "boolean") throw new Error("Portable destination receipt is not bound to this current device/path/workspace.");
      this.destination = retained;
    } else {
      this.destination = { format: "omni-portable-destination", version: 1, workspaceId: boot.workspaceId, manifestSha256: boot.manifestSha256, locationSha256: boot.locationSha256,
        brainId: manifest.selected.brainId, originChecksum: manifest.selected.originChecksum, completedSteps: [], approved: false, sourceDriveSelectionsDiscarded: false };
      await portableWriteJson(receiptPath, this.destination, this.dependencies.io);
    }
    if (this.destination.approved && !this.destination.completedSteps.includes("adoption-complete")) throw new Error("Portable destination approval has no completed saved-data preparation receipt.");
    if (this.destination.completedSteps.includes("adoption-complete")) {
      const selectedRoot = this.dependencies.repository.brainDirectory(manifest.selected.brainId);
      if (!await portableExists(join(selectedRoot, "brain.json"))) throw new Error("The prepared portable instance is missing. Its preserved cargo is a backup, not automatic replacement for newer saved data.");
      const header = await portableReadJson<{ id: string; originChecksum: string }>(selectedRoot, "brain.json", this.dependencies.io);
      if (header.id !== manifest.selected.brainId || header.originChecksum !== manifest.selected.originChecksum) throw new Error("Prepared portable data has another identity/origin.");
    }
    this.current.state = this.destination.approved ? "ready" : this.destination.completedSteps.includes("adoption-complete") ? "review-required" : "preparation-required";
    if (this.current.reservation) this.current.reservation.admittedPoolBytes = this.destination.approvedPoolBytes ?? null;
    if (this.destination.approvedPoolBytes !== undefined) this.dependencies.declareReservation(manifest.selected.brainId, this.destination.approvedPoolBytes);
  }
  async prepare(input: { requestId: string }, operation: BrainStorageOperationHooks): Promise<PortableWorkspaceStatus> {
    id(input.requestId); await this.initialize();
    if (!this.manifest || !this.destination) throw new Error("This is not a portable workspace.");
    if (this.current.state === "ready" || this.destination.completedSteps.includes("adoption-complete")) {
      this.current.state = this.destination.approved ? "ready" : "review-required"; delete this.current.error; return this.status();
    }
    if (this.preparing) { if (this.preparing.requestId !== input.requestId) throw new Error("Portable files are already being prepared by another exact operation."); return this.preparing.promise; }
    const promise = withPortableSavedDataRead(() => this.finishDestination(operation)).then(() => this.status()).catch(error => {
      if (this.destination?.completedSteps.includes("adoption-complete")) { this.current.state = this.destination.approved ? "ready" : "review-required"; delete this.current.error; return this.status(); }
      this.current = { ...this.current, state: "preparation-required", error: errorMessage(error) }; throw error;
    }).finally(() => { if (this.preparing?.promise === promise) this.preparing = undefined; });
    this.preparing = { requestId: input.requestId, promise }; return promise;
  }
  private async finishDestination(operation: BrainStorageOperationHooks): Promise<void> {
    const manifest = this.manifest!, destination = this.destination!, receiptPath = join(this.dependencies.deviceDataRoot, "portable-destination.json");
    const io: PortableIoHooks = { ...this.dependencies.io, signal: operation.signal, checkDisk: (path, bytes) => operation.checkDisk(path, bytes), checkpoint: () => operation.checkpoint() };
    this.current = { ...this.current, state: "preparing", phase: "portable-adoption", label: "Preparing the saved selected instance" }; delete this.current.error;
    const checkpoint = async (step: string): Promise<void> => { operation.signal.throwIfAborted(); const completedSteps = destination.completedSteps.includes(step) ? destination.completedSteps : [...destination.completedSteps, step];
      await portableWriteJson(receiptPath, { ...destination, completedSteps }, io); destination.completedSteps = completedSteps;
      this.current.phase = step; await operation.checkpoint({ phase: step, label: "Preserving saved work and retiring copied external authority" }); };
    const brainPath = this.dependencies.repository.brainDirectory(manifest.selected.brainId);
    await this.recoverAdoptionPublication(manifest, io);
    if (!await portableExists(join(brainPath, "brain.json"))) await this.adoptInitialCargo(manifest, operation, io);
    const brain = await this.dependencies.repository.get(manifest.selected.brainId, false);
    if (brain.id !== manifest.selected.brainId || brain.originChecksum !== manifest.selected.originChecksum) throw new Error("Portable saved data is not the selected original identity/origin.");
    if (!destination.approved) {
      await this.dependencies.retireCopiedAuthority(brainPath, brain.id, destination.completedSteps, checkpoint, operation);
      if (!destination.completedSteps.includes("destination-resource-settings")) {
        const storageDirectories = brain.config.deviceResources?.storageDirectories ?? [];
        destination.sourceDriveSelectionsDiscarded = storageDirectories.length > 0;
        await portableWriteJson(join(this.dependencies.deviceDataRoot, "requested-instance-settings.json"), { config: brain.config, toolPermissions: brain.toolPermissions }, io);
        brain.config.idleCognition = false;
        if (brain.config.deviceResources) brain.config.deviceResources = { ...brain.config.deviceResources, storageDirectories: [] };
        await this.dependencies.repository.save(brain); await checkpoint("destination-resource-settings");
      }
    }
    const preferencePath = join(this.dependencies.userDataRoot, "tool-preferences.json");
    if (!destination.completedSteps.includes("ordinary-tool-settings")) {
      const selected = await portableExists(preferencePath) ? await portableReadJson<ToolRuntimePreferences>(this.dependencies.userDataRoot, "tool-preferences.json", io) : this.settings!.tools;
      const checked = checkedPortableSettings({ version: 1, tools: selected, presentation: this.settings!.presentation });
      await this.dependencies.setToolsPreferences(checked.tools); await checkpoint("ordinary-tool-settings");
    }
    if (manifest.themePackage && !destination.completedSteps.includes("theme-package")) {
      if (manifest.themePackage.path !== "portable/theme.json" || !this.dependencies.installTheme) throw new Error("Portable theme cannot be installed by this host.");
      await this.dependencies.installTheme(await portableReadJson(this.dependencies.bootstrap!.root, manifest.themePackage.path, io, file(manifest.themePackage))); await checkpoint("theme-package");
    }
    this.dependencies.declareReservation(brain.id, brain.config.storagePoolBytes);
    await checkpoint("adoption-complete"); this.current.state = destination.approved ? "ready" : "review-required"; delete this.current.phase; delete this.current.label;
    if (this.current.reservation) this.current.reservation.admittedPoolBytes = destination.approvedPoolBytes ?? null;
  }
  private async recoverAdoptionPublication(manifest: PortableWorkspaceManifest, io: PortableIoHooks): Promise<void> {
    const root = join(this.dependencies.userDataRoot, "portable-adoption"), name = `${manifest.id}.json`;
    if (!await portableExists(join(root, name))) return;
    const state = await portableReadJson<AdoptionState>(root, name, io);
    if (state.manifestSha256 !== this.dependencies.bootstrap!.manifestSha256 || state.brainId !== manifest.selected.brainId) throw new Error("Portable publication intent changed.");
    const target = this.dependencies.repository.brainDirectory(state.brainId);
    if (!state.published && await portableExists(join(target, "brain.json"))) {
      if (!state.outputOwner || !await portableOwnsDirectory(target, state.outputOwner) || !state.preparedSha256 || await portableTreeSha256(target, io) !== state.preparedSha256) throw new Error("Lost-ACK portable publication differs from its prepared files; no import was repeated.");
      state.published = true; await portableWriteJson(join(root, name), state, io);
    }
  }
  private async adoptInitialCargo(manifest: PortableWorkspaceManifest, operation: BrainStorageOperationHooks, io: PortableIoHooks): Promise<void> {
    const boot = this.dependencies.bootstrap!, stateRoot = join(this.dependencies.userDataRoot, "portable-adoption");
    await mkdir(stateRoot, { recursive: true, mode: 0o700 });
    const name = `${manifest.id}.json`, path = join(stateRoot, name), stageRoot = join(stateRoot, `${manifest.id}.staging`);
    let state: AdoptionState;
    if (await portableExists(path)) {
      state = await portableReadJson(stateRoot, name, io);
      if (state.manifestSha256 !== boot.manifestSha256 || state.brainId !== manifest.selected.brainId || typeof state.published !== "boolean") throw new Error("Portable adoption intent changed.");
    } else { state = { manifestSha256: boot.manifestSha256, brainId: manifest.selected.brainId, published: false }; await portableWriteJson(path, state, io); }
    const staged = this.dependencies.stagingRepository(join(stageRoot, "brains")); await staged.initialize();
    if (this.dependencies.pluginPortability) staged.configurePluginPortability(this.dependencies.pluginPortability);
    const stagedBrain = staged.brainDirectory(state.brainId);
    const target = this.dependencies.repository.brainDirectory(state.brainId);
    if (state.outputOwner) {
      await publishPortableDirectory(stagedBrain, target, "brain.json", io, async () => undefined, state.outputOwner, state.preparedSha256);
      state.published = true; await portableWriteJson(path, state, this.dependencies.io); return;
    }
    if (!await portableExists(join(stagedBrain, "brain.json"))) {
      if (await portableExists(stagedBrain)) throw new Error("Interrupted portable import has no final identity commit; preserved staging requires review, not duplicate import.");
      equal(await portableFile(boot.root, manifest.selected.bundle.path, io), manifest.selected.bundle);
      const imported = await staged.importBundle(join(boot.root, ...manifest.selected.bundle.path.split("/")), {}, operation);
      if (imported.id !== state.brainId || imported.originChecksum !== manifest.selected.originChecksum) throw new Error("Fresh portable import unexpectedly changed identity or immutable origin.");
    }
    const imported = await staged.get(state.brainId, false);
    if (imported.originChecksum !== manifest.selected.originChecksum || imported.readiness.state !== "ready") throw new Error("Portable import has no complete original identity commit.");
    const fingerprint = await portableTreeSha256(stagedBrain, io);
    if (state.preparedSha256 && state.preparedSha256 !== fingerprint) throw new Error("Prepared portable import changed before publication.");
    state.preparedSha256 = fingerprint; await portableWriteJson(path, state, io);
    if (await portableExists(target)) throw new Error("Portable target appeared before adoption; no identity was overwritten.");
    // Raw-source/connected-state CAS is repository-wide, not always linked into
    // the selected directory. Preserve it before making the identity visible.
    const blobRoot = join(stageRoot, "brains", ".blobs"), destinationBlobs = join(dirname(target), ".blobs");
    await mkdir(destinationBlobs, { recursive: true, mode: 0o700 });
    const destinationInfo = await lstat(destinationBlobs); if (!destinationInfo.isDirectory() || destinationInfo.isSymbolicLink()) throw new Error("Portable shared-blob destination is unsafe.");
    if (await portableExists(blobRoot)) for await (const name of portableFiles(blobRoot)) {
      if (/^[a-f0-9]{64}$/.exec(name)?.[0] !== name) throw new Error("Portable source CAS contains an unexpected file.");
      const descriptor = await portableFile(blobRoot, name, io); if (descriptor.sha256 !== name) throw new Error("Portable source CAS checksum failed.");
      const existing = join(destinationBlobs, name);
      if (await portableExists(existing)) equal(await portableFile(destinationBlobs, name, io), descriptor);
      else {
        try { await link(join(blobRoot, name), existing); }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error; equal(await portableFile(destinationBlobs, name, io), descriptor); }
      }
    }
    operation.signal.throwIfAborted();
    await publishPortableDirectory(stagedBrain, target, "brain.json", io, async owner => { state.outputOwner = owner; await portableWriteJson(path, state, io); });
    state.published = true; await portableWriteJson(path, state, this.dependencies.io);
    // Keep immutable cargo and staging evidence; neither can become replay authority.
  }
  async reviewDestination(): Promise<PortableDestinationReview> {
    await this.initialize(); const manifest = this.manifest, destination = this.destination;
    if (!manifest || !destination) throw new Error("This is not a portable workspace.");
    if (!destination.completedSteps.includes("adoption-complete")) throw new Error("Prepare the saved portable files before reviewing destination resources.");
    const brain = await this.dependencies.repository.get(manifest.selected.brainId, false), plan = await this.dependencies.planDestination(brain);
    const configSha256 = portableSha256(JSON.stringify(brain.config)), poolBytes = plan.resources.sharedStoragePoolBytes;
    const reviewId = randomUUID(), reviewSha256 = portableSha256(JSON.stringify({ workspaceId: manifest.id, location: destination.locationSha256, brainId: brain.id, configSha256, poolBytes, allowed: plan.allowed }));
    const value: PortableDestinationReview = { reviewId, reviewSha256, workspaceId: manifest.id, brainId: brain.id, plan,
      sourceDriveSelectionsDiscarded: destination.sourceDriveSelectionsDiscarded, copiedExternalAuthorityDormant: true, credentialsTransferred: false };
    this.review = { value, configSha256, poolBytes }; return structuredClone(value);
  }
  async approveDestination(input: PortableDestinationApproval): Promise<PortableWorkspaceStatus> {
    await this.initialize(); const review = this.review, destination = this.destination;
    if (!review || !destination || input.reviewId !== review.value.reviewId || input.reviewSha256 !== review.value.reviewSha256) throw new Error("Review the exact current portable destination first.");
    const brain = await this.dependencies.repository.get(destination.brainId, false);
    if (portableSha256(JSON.stringify(brain.config)) !== review.configSha256) throw new Error("Portable instance settings changed; review the current destination again.");
    const plan = await this.dependencies.planDestination(brain);
    if (!plan.allowed || !review.value.plan.allowed || plan.resources.sharedStoragePoolBytes !== review.poolBytes) throw new Error("The reviewed destination no longer fits; adjust settings or free resources and review again.");
    brain.config.storagePoolBytes = review.poolBytes; await this.dependencies.repository.save(brain);
    this.dependencies.declareReservation(brain.id, review.poolBytes);
    destination.approved = true; destination.approvedPoolBytes = review.poolBytes;
    await portableWriteJson(join(this.dependencies.deviceDataRoot, "portable-destination.json"), destination, this.dependencies.io);
    this.current.state = "ready"; delete this.current.error; if (this.current.reservation) this.current.reservation.admittedPoolBytes = review.poolBytes;
    this.review = undefined; return this.status();
  }
}
