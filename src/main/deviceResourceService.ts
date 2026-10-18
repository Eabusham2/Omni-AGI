import { execFile } from "node:child_process";
import { lstat, realpath } from "node:fs/promises";
import { freemem } from "node:os";
import { isAbsolute } from "node:path";
import { promisify } from "node:util";
import type { EngineRequestContext, EngineSupervisor } from "./engineSupervisor";
import type { DeviceResourceSettings, DeviceResourceTopology } from "../shared/deviceResources";
import { validateDeviceResourceTopology } from "./deviceResourcePlan";
import type { SpillPoolAuthority, SpillPoolSelection } from "./spillPoolAuthority";

const execute = promisify(execFile);

export interface DeviceTopologyRequest {
  measure?: boolean;
  storageDirectories?: string[];
}

interface DeviceResourceServiceDependencies {
  platform?: NodeJS.Platform;
  containerIdentity?: (directory: string) => Promise<string | undefined>;
  spillAuthority?: SpillPoolAuthority;
}

async function darwinContainerIdentity(directory: string): Promise<string | undefined> {
  const { stdout } = await execute("/usr/sbin/diskutil", ["info", "-plist", directory], { timeout: 5000, maxBuffer: 1024 * 1024 });
  const match = /<key>APFSContainerReference<\/key>\s*<string>(disk[0-9]+)<\/string>/.exec(stdout);
  if (match) return `apfs:${match[1]}`;
  const type = /<key>(?:FilesystemType|FilesystemName)<\/key>\s*<string>([^<]*)<\/string>/.exec(stdout)?.[1];
  if (!type || /apfs/i.test(type)) throw new Error("Storage container identity is unavailable.");
  return undefined; // A confirmed non-APFS filesystem retains its volume identity.
}

function samePhysicalInterface(previous: DeviceResourceTopology, current: DeviceResourceTopology, id: string): boolean {
  const old = previous.devices.find(device => device.id === id), fresh = current.devices.find(device => device.id === id);
  // Mutable free/allocated bytes are deliberately excluded. Stable physical
  // identity may keep a prior rate while a verified interface/group change may not.
  return Boolean(old && fresh && old.kind === fresh.kind && old.backend === fresh.backend
    && old.memoryGroupId === fresh.memoryGroupId && old.unifiedMemory === fresh.unifiedMemory);
}

/** Main-owned OS/worker observations. Renderer requests never supply capacity figures. */
export class DeviceResourceService {
  private readonly snapshots = new Map<string, DeviceResourceTopology>();
  private readonly pending = new Map<string, Promise<DeviceResourceTopology>>();
  constructor(private readonly engine: Pick<EngineSupervisor, "request"> & Partial<Pick<EngineSupervisor, "requestLoadedRegionalExecution">>, private readonly defaultPoolDirectory: string,
    private readonly dependencies: DeviceResourceServiceDependencies = {}) {}

  async selectedSpillDirectories(paths: string[], brainId?: string): Promise<SpillPoolSelection> {
    if (!this.dependencies.spillAuthority) throw new Error("Local spill-root authorization is unavailable.");
    const topology = await this.topology({ storageDirectories: paths, measure: true });
    return this.dependencies.spillAuthority.selected(paths, topology, brainId);
  }

  claimSpillSelection(selectionId: string, brainId: string, paths: readonly string[]): void {
    if (!this.dependencies.spillAuthority) throw new Error("Local spill-root authorization is unavailable.");
    this.dependencies.spillAuthority.claim(selectionId, brainId, paths);
  }

  async spillCapacity(settings: DeviceResourceSettings, topology: DeviceResourceTopology, context: { brainId?: string; selectionId?: string }) {
    if (!this.dependencies.spillAuthority) throw new Error("Local spill-root authorization is unavailable.");
    return this.dependencies.spillAuthority.planningSnapshot(settings, topology, {
      ...context, primaryDirectory: await realpath(this.defaultPoolDirectory)
    });
  }

  async topology(request: DeviceTopologyRequest = {}): Promise<DeviceResourceTopology> {
    return this.collectTopology(request);
  }

  /** Same loaded pipe and serial guard, not the general worker-starting discovery path. */
  async topologyForLoadedOwner(brainId: string, storagePath: string, signal: AbortSignal, context: EngineRequestContext,
    request: DeviceTopologyRequest = {}): Promise<DeviceResourceTopology> {
    if (!this.engine.requestLoadedRegionalExecution || context.brainId !== brainId) throw new Error("Loaded regional topology observation is unavailable.");
    return this.collectTopology(request, (params, timeout) => this.engine.requestLoadedRegionalExecution!("hardware_topology",
      { ...params, brainId, storagePath }, timeout, signal, context), `${brainId}:${context.requestId}`);
  }

  private async collectTopology(request: DeviceTopologyRequest,
    loadedDispatch?: (params: Record<string, unknown>, timeout: number) => Promise<unknown>, ownerKey = "ordinary"): Promise<DeviceResourceTopology> {
    if (!request || typeof request !== "object" || Array.isArray(request)
      || Object.keys(request).some((key) => !["measure", "storageDirectories"].includes(key))) throw new Error("Invalid device topology request.");
    if (request.measure !== undefined && typeof request.measure !== "boolean") throw new Error("Device measurement must be an explicit switch.");
    if (request.storageDirectories !== undefined && !Array.isArray(request.storageDirectories)) throw new Error("Selected storage directories must be a list.");
    const directories: string[] = [];
    for (const path of [this.defaultPoolDirectory, ...(request.storageDirectories ?? [])]) {
      if (typeof path !== "string" || !isAbsolute(path) || path.includes("\0")) throw new Error("Choose an absolute storage-pool directory.");
      const info = await lstat(path);
      if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("A selected storage pool must be an existing real directory.");
      const resolved = await realpath(path);
      if (!directories.includes(resolved)) directories.push(resolved);
    }
    const key = JSON.stringify(directories);
    // An explicit measurement must never attach to an in-flight discovery
    // that promised no probes. Equal measurement requests still coalesce.
    const pendingKey = `${ownerKey}:${request.measure === true ? "measure" : "discover"}:${key}`;
    const existing = this.pending.get(pendingKey);
    if (existing) return structuredClone(await existing);
    const operation = (async () => {
      const params = {
        measure: request.measure === true, storageDirectories: directories,
        probeBudgetBytes: Math.max(0, Math.min(32 * 1024 * 1024, Math.floor(freemem() / 16)))
      };
      const raw = loadedDispatch ? await loadedDispatch(params, request.measure ? 120_000 : 30_000)
        : await this.engine.request<unknown>("hardware_topology", params, request.measure ? 120_000 : 30_000, undefined, "background");
      const profile = validateDeviceResourceTopology(raw);
      // APFS volumes may share one physical container and the same free bytes.
      // st_dev identifies volumes, so ask the OS for their container identity.
      if ((this.dependencies.platform ?? process.platform) === "darwin") {
        let unverifiedStorage = false;
        for (const device of profile.devices) {
          if (device.kind !== "storage" || !device.poolDirectory) continue;
          try {
            const group = await (this.dependencies.containerIdentity ?? darwinContainerIdentity)(device.poolDirectory);
            if (group) device.memoryGroupId = group;
          } catch {
            unverifiedStorage = true;
            profile.warnings.push(`Shared-container identity could not be verified for ${device.label}; storage capacity is conservatively counted once.`);
          }
        }
        // A volume with unknown identity could share space with any known
        // container. Group all selected volumes in that case, not only the
        // failed paths, so an API failure cannot inflate available storage.
        if (unverifiedStorage) for (const device of profile.devices) {
          if (device.kind === "storage") device.memoryGroupId = "darwin:storage-unverified";
        }
      }
      const previous = this.snapshots.get(key);
      if (!request.measure && previous) {
        for (const device of profile.devices) {
          const old = previous.devices.find((entry) => entry.id === device.id && entry.memoryGroupId === device.memoryGroupId);
          if (!device.measurement && old?.measurement && samePhysicalInterface(previous, profile, device.id)) device.measurement = old.measurement;
        }
        for (const link of profile.links) {
          const old = previous.links.find((entry) => entry.from === link.from && entry.to === link.to);
          if (!link.measurement && old?.measurement && old.peerAccess === link.peerAccess
            && old.unavailableReason === link.unavailableReason
            && samePhysicalInterface(previous, profile, link.from)
            && samePhysicalInterface(previous, profile, link.to)) link.measurement = old.measurement;
        }
      }
      const validated = validateDeviceResourceTopology(profile);
      this.dependencies.spillAuthority?.observeVolumes(validated);
      this.snapshots.set(key, validated);
      return validated;
    })();
    this.pending.set(pendingKey, operation);
    try { return structuredClone(await operation); }
    finally { if (this.pending.get(pendingKey) === operation) this.pending.delete(pendingKey); }
  }
}
