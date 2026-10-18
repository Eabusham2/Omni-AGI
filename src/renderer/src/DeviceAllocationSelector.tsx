import React, { useCallback, useEffect, useId, useRef, useState } from "react";
import type { OmniApi, SystemRamMode } from "../../shared/types";
import { DEFAULT_DEVICE_RESOURCE_SETTINGS, type DeviceResourceAllocation, type DeviceResourcePlan,
  type DeviceResourceSettings, type DeviceResourceTopology, type ResourceDevice } from "../../shared/deviceResources";
import "./deviceAllocation.css";

type Catalog = Pick<OmniApi["catalog"], "deviceTopology" | "selectSpillPools">;
export type DeviceAllocationChange = (settings: DeviceResourceSettings, selectionId?: string) => void;

/** Legacy and per-device RAM controls describe one ceiling, never two budgets. */
export function synchronizeDeviceRam(settings: DeviceResourceSettings | undefined, mode: SystemRamMode, percent: number): DeviceResourceSettings | undefined {
  if (!settings) return undefined;
  return { ...settings, allocations: settings.allocations.map((entry) => entry.deviceId === "ram:system"
    ? { ...entry, mode, percent: mode === "manual" ? Math.max(30, Math.min(100, percent)) : entry.percent } : entry) };
}

export function deviceAllocationChoice(device: ResourceDevice, settings: DeviceResourceSettings | undefined,
  ramMode: SystemRamMode, ramPercent: number): DeviceResourceAllocation {
  const stored = settings?.allocations.find((entry) => entry.deviceId === device.id);
  if (stored) return { ...stored };
  return { deviceId: device.id, enabled: device.kind === "ram" || device.kind === "storage" || device.nativeComputeSupported,
    mode: device.kind === "ram" ? ramMode : "auto", percent: device.kind === "ram" ? Math.max(30, ramPercent) : 90, autoIncrease: false };
}

export function deviceAvailability(device: ResourceDevice): string | undefined {
  if (device.kind !== "storage" && !device.nativeComputeSupported) return device.unavailableReason || "Native computation is unavailable on this device.";
  if (device.kind === "storage") {
    if (device.readOnly) return "This filesystem is read-only.";
    if (!device.poolDirectory || !device.filesystemDeviceId || device.memoryGroupId === "darwin:storage-unverified") return "Physical storage identity is not verified.";
  }
  if (device.totalBytes === null && device.kind !== "storage" && device.recommendedLimitBytes == null) return "Memory capacity has not been reported.";
  if (device.kind === "storage" && device.availableBytes === null) return "Available drive capacity has not been reported.";
  return undefined;
}

const capacity = (value: number | null | undefined): string => value == null ? "Not reported"
  : value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(1)} GiB`
    : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1)} MiB` : `${value.toLocaleString()} bytes`;
const rate = (value: number | undefined): string => value === undefined ? "Not measured" : `${capacity(value)}/s`;
const baseSettings = (value?: DeviceResourceSettings): DeviceResourceSettings => value ?? { ...DEFAULT_DEVICE_RESOURCE_SETTINGS, allocations: [], storageDirectories: [] };

export function DeviceAllocationCards({ topology, settings, ramMode, ramPercent, disabled, plan, onChange }: {
  topology: DeviceResourceTopology; settings?: DeviceResourceSettings; ramMode: SystemRamMode; ramPercent: number;
  disabled?: boolean; plan?: DeviceResourcePlan; onChange: DeviceAllocationChange;
}) {
  const update = (device: ResourceDevice, patch: Partial<DeviceResourceAllocation>) => {
    const next = { ...deviceAllocationChoice(device, settings, ramMode, ramPercent), ...patch };
    onChange({ ...baseSettings(settings), allocations: [...(settings?.allocations ?? []).filter((entry) => entry.deviceId !== device.id), next] });
  };
  const missing = (settings?.allocations ?? []).filter((entry) => !topology.devices.some((device) => device.id === entry.deviceId));
  return <>
    <div className="device-allocation-grid">{topology.devices.map((device) => {
      const choice = deviceAllocationChoice(device, settings, ramMode, ramPercent);
      const unavailable = deviceAvailability(device);
      const reviewedCpuFallback = device.backend === "cuda" && settings?.regionalExecution?.enabled === true
        && settings.regionalExecution.routes.some(route => route.deviceId === device.id
          && route.executionMode === "packed-cpu-tile-fallback" && route.acknowledgePackedCpuFallback === true);
      const shared = topology.devices.filter((entry) => entry.id !== device.id && entry.memoryGroupId === device.memoryGroupId);
      const budget = plan?.budgets.find((entry) => entry.deviceId === device.id);
      const configurable = device.kind !== "storage" || Boolean(device.poolDirectory && settings?.storageDirectories.includes(device.poolDirectory));
      const appliedStorage = device.kind === "storage" && configurable
        ? plan?.appliedStorage?.groups.find((group) => group.memoryGroupId === device.memoryGroupId) : undefined;
      const controlsDisabled = Boolean(disabled || !choice.enabled || unavailable && !reviewedCpuFallback || !configurable);
      return <article key={device.id} className="device-allocation-card" aria-label={`${device.label} allocation`}>
        <div className="device-allocation-card__heading"><h3>{device.label}</h3><span>{device.kind === "gpu" ? device.unifiedMemory ? "Unified GPU" : "GPU / VRAM" : device.kind === "ram" ? "System RAM" : device.kind === "storage" ? "Drive" : "NPU"}</span></div>
        {configurable ? <label className="device-allocation-check"><input type="checkbox" aria-label={`Enable ${device.label}`} checked={choice.enabled}
          disabled={disabled || Boolean(unavailable && !choice.enabled && !reviewedCpuFallback)} onChange={(event) => update(device, { enabled: event.target.checked })} /> Enabled for this instance</label>
          : <p className="device-allocation-note">Default checkpoint/spill volume; select this folder to customize its spill budget.</p>}
        {unavailable ? <p className="device-allocation-warning">{unavailable}</p> : null}
        {reviewedCpuFallback ? <p className="device-allocation-note">A selected manual route explicitly permits packed CPU-tile fallback. This does not provide a native CUDA kernel or a measured speedup.</p> : null}
        {device.kind === "storage" && settings?.allowStorageSpill === false ? <p className="device-allocation-note">Storage overflow is off; this drive preference is inactive.</p> : null}
        {shared.length ? <p className="device-allocation-shared">Same physical pool as {shared.map((entry) => entry.label).join(", ")}. These bytes are not added together.</p> : null}
        <dl className="device-allocation-readings"><div><dt>{device.kind === "storage" ? "Volume capacity" : "Physical capacity"}</dt><dd>{capacity(device.totalBytes)}</dd></div>
          <div><dt>Available (snapshot)</dt><dd>{capacity(device.availableBytes)}</dd></div>
          {device.kind !== "storage" ? <div><dt>Allocated / reserved</dt><dd>{capacity(device.allocatedBytes)} / {capacity(device.reservedBytes)}</dd></div> : null}
          {device.kind === "storage" ? configurable && plan
            ? <div><dt>Shared volume ceiling</dt><dd>{capacity(appliedStorage?.ceilingBytes)}</dd></div> : null
            : budget ? <div><dt>Selected ceiling after reserves</dt><dd>{capacity(budget.ceilingBytes)}</dd></div> : null}
        </dl>
        {configurable && (!unavailable || choice.enabled) ? <><div className="device-allocation-controls">
          <label>Allocation policy<select aria-label={`${device.label} allocation policy`} value={choice.mode} disabled={controlsDisabled}
            onChange={(event) => update(device, { mode: event.target.value as "auto" | "manual" })}><option value="auto">Auto</option><option value="manual">Manual percentage</option></select></label>
          {device.kind === "ram" && choice.mode === "auto" ? <p className="device-allocation-note">Ceiling follows the shared Auto RAM policy.</p> :
            <label>{choice.mode === "auto" ? "Auto selection" : "Safe-pool ceiling"} · {choice.percent}%<input type="range" aria-label={`${device.label} allocation percent`}
              min={device.kind === "ram" ? 30 : 1} max={100} step={1} value={choice.percent} disabled={controlsDisabled}
              onChange={(event) => update(device, { percent: Number(event.target.value) })} /></label>}
        </div>
        <label className="device-allocation-check"><input type="checkbox" aria-label={`Allow automatic increases for ${device.label}`} checked={choice.autoIncrease}
          disabled={controlsDisabled} onChange={(event) => update(device, { autoIncrease: event.target.checked })} /> Allow automatic budget increases</label>
        </> : null}
        {configurable && device.kind === "storage" && choice.autoIncrease && !settings?.autoIncreaseStorage ? <p className="device-allocation-note">Inactive until storage budget increases are allowed below.</p> : null}
        <details><summary>Identity and measured speed</summary><dl className="device-allocation-readings">
          <div><dt>Stable device ID</dt><dd>{device.id}</dd></div><div><dt>Physical memory group</dt><dd>{device.memoryGroupId}</dd></div>
          <div><dt>Backend</dt><dd>{device.backend}{device.executionDevice ? ` · ${device.executionDevice}` : ""}</dd></div>
          <div><dt>{device.measurement?.scope === "durable-file-write" ? "Durable write" : device.measurement?.scope === "device-transfer" ? "Device transfer" : "Memory copy"}</dt><dd>{rate(device.measurement?.bytesPerSecond)}</dd></div>
          {device.measurement ? <div><dt>Measurement</dt><dd>{device.measurement.runs} runs · {capacity(device.measurement.sampleBytes)} per run · {new Date(device.measurement.measuredAt).toLocaleString()}</dd></div> : null}
          {device.poolDirectory ? <div><dt>Pool directory</dt><dd>{device.poolDirectory}</dd></div> : null}
          {appliedStorage ? <>
            <div><dt>Charged storage</dt><dd>{capacity(appliedStorage.chargedBytes)}</dd></div>
            <div><dt>Pending storage</dt><dd>{capacity(appliedStorage.pendingBytes)}</dd></div>
            <div><dt>Unmaterialized sparse promises</dt><dd>{capacity(appliedStorage.unmaterializedPromiseBytes)}</dd></div>
          </> : null}
        </dl></details>
      </article>;
    })}</div>
    {missing.map((entry) => <div key={entry.deviceId} className="device-allocation-warning"><p>Saved selection is unavailable: <code>{entry.deviceId}</code>. No capacity is assumed.</p>
      <button className="button" type="button" disabled={disabled} onClick={() => onChange({ ...baseSettings(settings), allocations: settings!.allocations.filter((value) => value.deviceId !== entry.deviceId) })}>Remove unavailable selection</button>
    </div>)}
    {topology.links.length ? <details><summary>Measured transfer links</summary><ul className="device-allocation-links">{topology.links.map((link) => <li key={`${link.from}>${link.to}`}>
      <strong>{topology.devices.find((device) => device.id === link.from)?.label ?? link.from} → {topology.devices.find((device) => device.id === link.to)?.label ?? link.to}</strong>
      <span>{rate(link.measurement?.bytesPerSecond)} · {link.peerAccess === true ? "Peer access available" : link.peerAccess === false ? "No peer access" : "Peer access not reported"}</span>
      {link.unavailableReason ? <small>{link.unavailableReason}</small> : null}
    </li>)}</ul></details> : null}
  </>;
}

export function DeviceAllocationSelector({ settings, ramMode, ramPercent, onChange, catalog, brainId, disabled, plan, planPending }: {
  settings?: DeviceResourceSettings; ramMode: SystemRamMode; ramPercent: number; onChange: DeviceAllocationChange;
  catalog?: Catalog; brainId?: string; disabled?: boolean; plan?: DeviceResourcePlan; planPending?: boolean;
}) {
  const headingId = useId();
  const [topology, setTopology] = useState<DeviceResourceTopology>();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const generation = useRef(0), alive = useRef(true), value = useRef(settings), change = useRef(onChange);
  value.current = settings; change.current = onChange;
  const pathsKey = JSON.stringify(settings?.storageDirectories ?? []);
  const discover = useCallback(async (measure = false) => {
    if (!catalog?.deviceTopology) return;
    const request = ++generation.current; setBusy(true); setError("");
    try {
      const result = await catalog.deviceTopology({ measure, storageDirectories: JSON.parse(pathsKey) as string[] });
      if (alive.current && request === generation.current) setTopology(result);
    } catch (reason) { if (alive.current && request === generation.current) setError(reason instanceof Error ? reason.message : "Device discovery is unavailable."); }
    finally { if (alive.current && request === generation.current) setBusy(false); }
  }, [catalog, pathsKey]);
  useEffect(() => { alive.current = true; void discover(false); return () => { alive.current = false; generation.current += 1; }; }, [discover]);
  const choose = async () => {
    if (!catalog || busy || disabled) return;
    setBusy(true); setError("");
    try {
      const selection = await catalog.selectSpillPools(brainId);
      if (!selection || !alive.current) return;
      const previous = baseSettings(value.current);
      const removed = new Set((topology?.devices ?? []).filter((device) => device.kind === "storage" && device.poolDirectory &&
        previous.storageDirectories.includes(device.poolDirectory) && !selection.directories.includes(device.poolDirectory)).map((device) => device.id));
      change.current({ ...previous, storageDirectories: selection.directories, allocations: previous.allocations.filter((entry) => !removed.has(entry.deviceId)) }, brainId ? undefined : selection.selectionId);
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : "Drive selection failed."); }
    finally { if (alive.current) setBusy(false); }
  };
  return <section className="device-allocation-selector" aria-labelledby={headingId} aria-busy={busy}>
    <div className="device-allocation-heading"><div><h3 id={headingId}>Per-device allocation</h3><p>Choose eligible RAM, accelerator and drive pools. Actual saved-state fit is checked by the runtime; GPU capacities are not treated as one interchangeable pool.</p></div>
      <div className="device-allocation-actions">
        <button type="button" className="button" disabled={disabled || busy || !catalog?.deviceTopology} onClick={() => void discover(false)}>Refresh devices</button>
        <button type="button" className="button" disabled={disabled || busy || !catalog?.deviceTopology} onClick={() => void discover(true)}>{busy ? "Reading devices…" : "Measure transfer rates"}</button>
      </div></div>
    {!catalog?.deviceTopology ? <p>Device discovery requires the desktop runtime.</p> : null}
    {error ? <p className="device-allocation-warning" role="alert">{error}</p> : null}
    {topology ? <><DeviceAllocationCards topology={topology} settings={settings} ramMode={ramMode} ramPercent={ramPercent}
      disabled={disabled || busy} plan={planPending ? undefined : plan} onChange={onChange} />
      <p className="device-allocation-note">Observed {new Date(topology.measuredAt).toLocaleString()} · {topology.platform} / {topology.architecture}. Unknown measurements are not speed estimates. System/driver reserves remain excluded.</p>
      {topology.warnings.length ? <ul className="device-allocation-warning">{topology.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}</> : busy ? <p role="status">Reading the actual device topology; no neural model is constructed.</p> : null}
    <fieldset disabled={disabled || busy || !catalog?.deviceTopology || !catalog?.selectSpillPools}><legend>Temporary storage policy</legend>
      <label className="device-allocation-check"><input type="checkbox" checked={settings?.allowStorageSpill ?? true} onChange={(event) => onChange({ ...baseSettings(settings), allowStorageSpill: event.target.checked })} /> Allow storage-backed overflow</label>
      <label className="device-allocation-check"><input type="checkbox" checked={settings?.autoIncreaseStorage ?? false} onChange={(event) => onChange({ ...baseSettings(settings), autoIncreaseStorage: event.target.checked })} /> Allow automatic storage budget increases</label>
      <button type="button" className="button" onClick={() => void choose()}>Select spill drives or folders…</button>
      {settings?.storageDirectories.length ? <ul className="device-allocation-paths">{settings.storageDirectories.map((path) => <li key={path}>{path}</li>)}</ul> : <p className="device-allocation-note">No extra drive folders selected.</p>}
      <p className="device-allocation-note">Drive access is local to this installation and is checked at preflight. Imported path strings never grant access. The picker replaces the complete extra-drive selection.</p>
    </fieldset>
    {planPending ? <p role="status">Updating the resource preflight…</p> : plan?.admittedContextMaximumTokens !== undefined ? <p className="device-allocation-note">Current runtime-admitted context maximum: {plan.admittedContextMaximumTokens.toLocaleString()} tokens. This is distinct from a diagnostic sum of device capacities.</p> : null}
    {plan?.executionBlockers?.length ? <ul className="device-allocation-warning">{plan.executionBlockers.map((blocker) => <li key={blocker}>{blocker}</li>)}</ul> : null}
  </section>;
}
