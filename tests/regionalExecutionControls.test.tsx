import { describe, expect, it, vi } from "vitest";

const hookBridge = vi.hoisted(() => ({ current: undefined as undefined | {
  useState(initial: unknown): [unknown, (value: unknown) => void];
  useRef(initial: unknown): { current: unknown };
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void;
} }));
vi.mock("react", async importOriginal => {
  const actual = await importOriginal<typeof import("react")>();
  return { ...actual, useState: (initial: unknown) => hookBridge.current!.useState(initial),
    useRef: (initial: unknown) => hookBridge.current!.useRef(initial),
    useEffect: (effect: () => void | (() => void), dependencies?: readonly unknown[]) => hookBridge.current!.useEffect(effect, dependencies) };
});

import { isValidElement, type ReactElement, type ReactNode } from "react";
import { RegionalExecutionControls, regionalExecutionDeviceChoices, regionalExecutionStatusLabel,
  setRegionalExecutionRoute } from "../src/renderer/src/RegionalExecutionControls";
import type { DeviceResourceSettings, DeviceResourceTopology, RegionalExecutionRegion, RegionalExecutionStatus, ResourceDevice } from "../src/shared/deviceResources";
import type { RegionalExecutionApi, RegionalExecutionPrepared } from "../src/shared/regionalExecution";

type Props = Parameters<typeof RegionalExecutionControls>[0];
class HookHarness {
  private cursor = 0; private readonly states = new Map<number, unknown>(); private readonly refs = new Map<number, { current: unknown }>();
  private readonly effects = new Map<number, { dependencies?: readonly unknown[]; cleanup?: void | (() => void) }>();
  useState(initial: unknown): [unknown, (value: unknown) => void] { const index = this.cursor++;
    if (!this.states.has(index)) this.states.set(index, typeof initial === "function" ? (initial as () => unknown)() : initial);
    return [this.states.get(index), value => this.states.set(index, typeof value === "function" ? (value as (current: unknown) => unknown)(this.states.get(index)) : value)]; }
  useRef(initial: unknown): { current: unknown } { const index = this.cursor++; if (!this.refs.has(index)) this.refs.set(index, { current: initial }); return this.refs.get(index)!; }
  useEffect(effect: () => void | (() => void), dependencies?: readonly unknown[]): void { const index = this.cursor++, prior = this.effects.get(index);
    const changed = !prior || dependencies === undefined || prior.dependencies === undefined || dependencies.length !== prior.dependencies.length ||
      dependencies.some((value, offset) => !Object.is(value, prior.dependencies![offset])); if (!changed) return;
    if (typeof prior?.cleanup === "function") prior.cleanup(); this.effects.set(index, { dependencies: dependencies ? [...dependencies] : undefined, cleanup: effect() }); }
  render(props: Props): ReactNode { this.cursor = 0; hookBridge.current = this; try { return RegionalExecutionControls(props); } finally { hookBridge.current = undefined; } }
  unmount(): void { for (const value of this.effects.values()) if (typeof value.cleanup === "function") value.cleanup(); this.effects.clear(); }
}

const GIB = 1024 ** 3;
const device = (id: string, backend: ResourceDevice["backend"], label: string, nativeComputeSupported: boolean): ResourceDevice => ({
  id, label, kind: backend === "cpu" ? "ram" : backend === "npu" ? "npu" : "gpu", backend,
  executionDevice: backend === "cpu" ? "cpu" : backend === "mps" ? "mps" : backend === "cuda" ? "cuda:0" : backend === "xpu" ? "xpu:0" : backend === "npu" ? "npu:0" : "privateuseone:0",
  memoryGroupId: backend === "cpu" || backend === "mps" ? "ram:system" : `memory:${id}`, totalBytes: 16 * GIB,
  availableBytes: 12 * GIB, allocatedBytes: 0, reservedBytes: 2 * GIB, unifiedMemory: backend === "mps", nativeComputeSupported,
  measurement: { bytesPerSecond: backend === "cpu" ? 24 * GIB : 12 * GIB, sampleBytes: 1024, runs: 3,
    medianNanoseconds: 1000, measuredAt: "2026-10-15T12:00:00Z", scope: backend === "cpu" ? "memory-copy" : "device-transfer" }
});
const DEVICES = [device("ram:system", "cpu", "System CPU", true), device("gpu:metal", "mps", "Apple Metal", true),
  device("gpu:cuda", "cuda", "NVIDIA CUDA", false), device("gpu:xpu", "xpu", "Intel XPU", true),
  device("npu:0", "npu", "System NPU", true), device("gpu:directml", "directml", "DirectML", true)];
const TOPOLOGY: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1, measuredAt: "2026-10-15T12:00:00Z",
  platform: "fixture", architecture: "arm64", devices: DEVICES, links: [], measurementRequested: true, neuralModelConstructed: false, warnings: [] };
const region = (id: string): RegionalExecutionRegion => ({ regionId: id, dimensions: 512, heads: 8, feedForward: 2048,
  packedOwners: 4, packedOwnerBytes: 4096, maximumPackedOwnerBytes: 8192, maximumDevicePackedOwnerBytes: 4096 });
const status = (ownerId: string, active = false): RegionalExecutionStatus => ({ format: "omni-manual-regional-execution-status", version: 1,
  ownerId, active, leaseValid: active, epoch: active ? "epoch-current" : null, reason: active ? "Exact manual lease" : "Not activated",
  registeredRegions: 513, activeCalls: 1, routes: [], completedRegionDispatches: 12, actualActivationTransfers: 4,
  actualActivationTransferBytes: 8192, trainingInapplicableCalls: 2, estimatedKnownTransferSeconds: 0.25,
  unknownTransferCosts: 1, trainingSupported: false, automaticDefault: false, contextGpuCapacityCredited: false,
  wholePathPerformanceMeasured: false, trace: [] });
const SETTINGS: DeviceResourceSettings = { version: 1, allocations: [
  { deviceId: "ram:system", enabled: true, mode: "manual", percent: 62, autoIncrease: false },
  { deviceId: "gpu:cuda", enabled: false, mode: "manual", percent: 37, autoIncrease: false }
], allowStorageSpill: true, autoIncreaseStorage: false, storageDirectories: [],
regionalExecution: { version: 1, enabled: true, mode: "manual", routes: [{ regionId: "decoder.layers.0", deviceId: "ram:system", executionMode: "native" }] } };
const prepared = (ownerId: string, reviewId: string, regions = [region(`${ownerId}.decoder.layers.0`)]): RegionalExecutionPrepared => ({
  format: "omni-manual-regional-execution-review", version: 1, reviewId, ownerId, policySha256: "a".repeat(64),
  topologyIdentitySha256: "b".repeat(64), registrySha256: "c".repeat(64), topology: TOPOLOGY,
  settings: SETTINGS.regionalExecution!, start: 0, total: 513, regions, status: status(ownerId)
});
function api(overrides: Partial<RegionalExecutionApi> = {}): RegionalExecutionApi { return {
  prepare: vi.fn(async request => prepared(request.brainId, "review-default")), status: vi.fn(async brainId => status(brainId)),
  activate: vi.fn(async request => status(request.brainId, true)), deactivate: vi.fn(async request => status(request.brainId)), ...overrides };
}
function props(value: RegionalExecutionApi, patch: Partial<Props> = {}): Props { return { api: value, brainId: "brain-a", loaded: true,
  foregroundTurnActive: false, settings: SETTINGS, dirty: false, onChange: vi.fn(), ...patch }; }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(yes => { resolve = yes; }); return { promise, resolve }; }
const flush = async () => { await Promise.resolve(); await Promise.resolve(); await Promise.resolve(); };
function children(node: ReactNode): ReactNode[] { if (node == null || typeof node === "boolean") return []; if (Array.isArray(node)) return node;
  if (isValidElement(node)) { const value = (node.props as { children?: ReactNode }).children; return value === undefined ? [] : Array.isArray(value) ? value : [value]; } return []; }
function elements(node: ReactNode): ReactElement[] { const result: ReactElement[] = []; if (isValidElement(node)) result.push(node); for (const child of children(node)) result.push(...elements(child)); return result; }
function text(node: ReactNode): string { if (typeof node === "string" || typeof node === "number") return String(node); return children(node).map(text).join(""); }
function button(node: ReactNode, label: string): ReactElement { const found = elements(node).find(value => value.type === "button" && text(value).trim() === label); if (!found) throw new Error(`Missing button ${label}`); return found; }
function routeSelect(node: ReactNode): ReactElement { const found = elements(node).find(value => value.type === "select"); if (!found) throw new Error("Missing regional route selector"); return found; }
const click = (value: ReactElement) => (value.props as { onClick(): void }).onClick();

describe("manual regional execution controls", () => {
  it("offers only CPU/Metal native and explicit CUDA fallback while enabling the allocation without changing its percentage", () => {
    const choices = regionalExecutionDeviceChoices(DEVICES, SETTINGS);
    expect(choices.map(value => value.device.backend)).toEqual(["cpu", "mps", "cuda"]);
    expect(choices[0]?.label).toContain("allocation enabled"); expect(choices[2]?.label).toContain("will be enabled");
    expect(() => setRegionalExecutionRoute(SETTINGS, "exact.decoder.7", choices[2]!, false)).toThrow("explicit");
    const cuda = setRegionalExecutionRoute(SETTINGS, "exact.decoder.7", choices[2]!, true);
    expect(cuda.allocations.find(value => value.deviceId === "gpu:cuda")).toMatchObject({ enabled: true, percent: 37, mode: "manual" });
    expect(cuda.regionalExecution?.routes.find(value => value.regionId === "exact.decoder.7")).toEqual({ regionId: "exact.decoder.7",
      deviceId: "gpu:cuda", executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true });
    const metal = setRegionalExecutionRoute(undefined, "exact.decoder.8", choices[1]!, false);
    expect(metal.allocations).toContainEqual({ deviceId: "gpu:metal", enabled: true, mode: "manual", percent: 75, autoIncrease: false });
  });

  it("is inert on mount and keeps Status available while dirty or replying without enabling topology mutation", () => {
    const value = api(), harness = new HookHarness(); let tree = harness.render(props(value));
    expect(value.prepare).not.toHaveBeenCalled(); expect(value.status).not.toHaveBeenCalled(); expect(value.activate).not.toHaveBeenCalled();
    expect(text(tree)).toContain("Advanced · manual regional execution");
    tree = harness.render(props(value, { dirty: true, foregroundTurnActive: true }));
    expect(text(tree)).toContain("Apply settings, then Prepare routes again.");
    expect((button(tree, "Prepare routes").props as { disabled?: boolean }).disabled).toBe(true);
    expect((button(tree, "Refresh status").props as { disabled?: boolean }).disabled).toBe(false); harness.unmount();
  });

  it("deduplicates Prepare and discards a late review after brain ownership changes", async () => {
    const first = deferred<RegionalExecutionPrepared>(), prepare = vi.fn()
      .mockImplementationOnce(() => first.promise).mockResolvedValueOnce(prepared("brain-b", "review-b"));
    const value = api({ prepare }), harness = new HookHarness(); let tree = harness.render(props(value));
    click(button(tree, "Prepare routes")); click(button(tree, "Prepare routes")); expect(prepare).toHaveBeenCalledTimes(1);
    tree = harness.render(props(value, { brainId: "brain-b" })); tree = harness.render(props(value, { brainId: "brain-b" }));
    first.resolve(prepared("brain-a", "review-a", [region("brain-a.exact.region")])); await flush();
    tree = harness.render(props(value, { brainId: "brain-b" })); expect(text(tree)).not.toContain("brain-a.exact.region");
    click(button(tree, "Prepare routes")); await flush(); tree = harness.render(props(value, { brainId: "brain-b" }));
    expect(text(tree)).toContain("brain-b.decoder.layers.0"); expect(prepare).toHaveBeenCalledTimes(2);
    tree = harness.render(props(value, { brainId: "brain-b", dirty: true }));
    expect((button(tree, "Activate reviewed routes").props as { disabled?: boolean }).disabled).toBe(true); harness.unmount();
  });

  it("uses exact review and epoch identities for explicit activation, status and deactivation", async () => {
    const activate = vi.fn(async request => status(request.brainId, true)), deactivate = vi.fn(async request => status(request.brainId));
    const readStatus = vi.fn(async brainId => status(brainId, true)), value = api({ activate, deactivate, status: readStatus });
    const harness = new HookHarness(); let tree = harness.render(props(value)); click(button(tree, "Prepare routes")); await flush();
    tree = harness.render(props(value)); click(button(tree, "Activate reviewed routes")); await flush();
    expect(activate).toHaveBeenCalledWith({ brainId: "brain-a", reviewId: "review-default" });
    tree = harness.render(props(value, { foregroundTurnActive: true })); click(button(tree, "Refresh status")); await flush();
    expect(readStatus).toHaveBeenCalledWith("brain-a"); tree = harness.render(props(value));
    click(button(tree, "Deactivate")); await flush(); expect(deactivate).toHaveBeenCalledWith({ brainId: "brain-a", epoch: "epoch-current" }); harness.unmount();
  });

  it("retains descriptor inspection for batch route edits but retires review authority until saved routes are prepared again", async () => {
    const regions = [region("brain-a.decoder.batch.1"), region("brain-a.decoder.batch.2")];
    const prepare = vi.fn().mockResolvedValueOnce(prepared("brain-a", "review-before-edit", regions))
      .mockResolvedValueOnce(prepared("brain-a", "review-after-apply", regions));
    const firstApi = api({ prepare }), changed = vi.fn(), harness = new HookHarness();
    let currentSettings = SETTINGS, tree = harness.render(props(firstApi, { settings: currentSettings, onChange: changed }));
    click(button(tree, "Prepare routes")); await flush(); tree = harness.render(props(firstApi, { settings: currentSettings, onChange: changed }));
    expect(text(tree)).toContain("brain-a.decoder.batch.1"); expect(text(tree)).toContain("brain-a.decoder.batch.2");
    click(button(tree, "Add saved route · enable allocation")); currentSettings = changed.mock.calls.at(-1)![0] as DeviceResourceSettings;
    tree = harness.render(props(firstApi, { settings: currentSettings, dirty: true, onChange: changed }));
    tree = harness.render(props(firstApi, { settings: currentSettings, dirty: true, onChange: changed }));
    expect(text(tree)).toContain("Descriptor page retained for editing; its activation review is stale.");
    expect(text(tree)).toContain("brain-a.decoder.batch.2"); expect((button(tree, "Activate reviewed routes").props as { disabled?: boolean }).disabled).toBe(true);
    click(button(tree, "Add saved route · enable allocation")); currentSettings = changed.mock.calls.at(-1)![0] as DeviceResourceSettings;
    expect(currentSettings.regionalExecution?.routes.map(value => value.regionId)).toEqual(expect.arrayContaining(regions.map(value => value.regionId)));
    tree = harness.render(props(firstApi, { settings: currentSettings, dirty: true, onChange: changed }));
    tree = harness.render(props(firstApi, { settings: currentSettings, dirty: true, onChange: changed }));
    tree = harness.render(props(firstApi, { settings: currentSettings, dirty: false, onChange: changed }));
    expect((button(tree, "Activate reviewed routes").props as { disabled?: boolean }).disabled).toBe(true);
    click(button(tree, "Prepare routes")); await flush(); tree = harness.render(props(firstApi, { settings: currentSettings, dirty: false, onChange: changed }));
    expect((button(tree, "Activate reviewed routes").props as { disabled?: boolean }).disabled).toBe(false);
    const secondApi = api(); tree = harness.render(props(secondApi, { settings: currentSettings, onChange: changed }));
    tree = harness.render(props(secondApi, { settings: currentSettings, onChange: changed }));
    expect(text(tree)).not.toContain("Exact DecoderBlock registry"); expect(text(tree)).not.toContain("activation review is stale"); harness.unmount();
  });

  it("defaults a fresh prepared selector to the exact saved CUDA route instead of the first CPU choice", async () => {
    const regionId = "brain-a.saved.cuda", cudaSettings: DeviceResourceSettings = { ...SETTINGS,
      allocations: SETTINGS.allocations.map(value => value.deviceId === "gpu:cuda" ? { ...value, enabled: true } : value),
      regionalExecution: { version: 1, enabled: true, mode: "manual", routes: [{ regionId, deviceId: "gpu:cuda",
        executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true }] } };
    const changed = vi.fn(), value = api({ prepare: vi.fn(async () => prepared("brain-a", "saved-cuda", [region(regionId)])) });
    const harness = new HookHarness(); let tree = harness.render(props(value, { settings: cudaSettings, onChange: changed }));
    click(button(tree, "Prepare routes")); await flush(); tree = harness.render(props(value, { settings: cudaSettings, onChange: changed }));
    expect((routeSelect(tree).props as { value?: string }).value).toBe("gpu:cuda");
    expect((button(tree, "Update saved route").props as { disabled?: boolean }).disabled).toBe(false);
    click(button(tree, "Update saved route"));
    const next = changed.mock.calls.at(-1)![0] as DeviceResourceSettings;
    expect(next.regionalExecution?.routes).toContainEqual({ regionId, deviceId: "gpu:cuda",
      executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true }); harness.unmount();
  });

  it("labels an armed invalid lease without implying it is actively dispatching", async () => {
    const invalid = { ...status("brain-a", true), leaseValid: false, reason: "Synthetic topology changed" };
    const label = regionalExecutionStatusLabel(invalid);
    expect(label).toBe("Armed but invalid · Synthetic topology changed"); expect(label).not.toContain("Active manual lease");
    const value = api({ status: vi.fn(async () => invalid) }), harness = new HookHarness(); const tree = harness.render(props(value));
    click(button(tree, "Refresh status")); await flush();
    expect(value.status).toHaveBeenCalledWith("brain-a"); harness.unmount();
  });
});
