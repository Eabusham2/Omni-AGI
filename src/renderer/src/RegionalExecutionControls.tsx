import { useEffect, useRef, useState } from "react";
import { DEFAULT_DEVICE_RESOURCE_SETTINGS, type DeviceResourceAllocation, type DeviceResourceSettings,
  type RegionalExecutionRegion, type RegionalExecutionRoute, type RegionalExecutionStatus, type ResourceDevice } from "../../shared/deviceResources";
import type { RegionalExecutionApi, RegionalExecutionPrepared } from "../../shared/regionalExecution";
import "./regionalExecution.css";

const failure = (error: unknown): string => error instanceof Error ? error.message : String(error);
const bytes = (value: number): string => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GiB`
  : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(2)} MiB` : `${value.toLocaleString()} bytes`;
const rate = (value: number | undefined): string => value === undefined ? "Not measured" : `${bytes(value)}/s`;
const baseSettings = (settings?: DeviceResourceSettings): DeviceResourceSettings => settings
  ? { ...settings, allocations: settings.allocations.map(value => ({ ...value })), storageDirectories: [...settings.storageDirectories],
      ...(settings.regionalExecution ? { regionalExecution: { ...settings.regionalExecution,
        routes: settings.regionalExecution.routes.map(value => ({ ...value })) } } : {}) }
  : { ...DEFAULT_DEVICE_RESOURCE_SETTINGS, allocations: [], storageDirectories: [] };

export interface RegionalExecutionDeviceChoice {
  device: ResourceDevice;
  mode: RegionalExecutionRoute["executionMode"];
  allocationEnabled: boolean;
  label: string;
}

/** Only implemented route kinds are offered. XPU/NPU/DirectML are deliberately absent. */
export function regionalExecutionDeviceChoices(devices: ResourceDevice[], settings?: DeviceResourceSettings): RegionalExecutionDeviceChoice[] {
  return devices.flatMap<RegionalExecutionDeviceChoice>(device => {
    const allocationEnabled = settings?.allocations.some(value => value.deviceId === device.id && value.enabled) ?? false;
    if ((device.backend === "cpu" || device.backend === "mps") && device.nativeComputeSupported) return [{ device, mode: "native" as const,
      allocationEnabled, label: `${device.label} · ${device.backend === "mps" ? "Metal native" : "CPU native"} · allocation ${allocationEnabled ? "enabled" : "will be enabled"}` }];
    if (device.backend === "cuda") return [{ device, mode: "packed-cpu-tile-fallback" as const, allocationEnabled,
      label: `${device.label} · packed CPU-tile fallback · allocation ${allocationEnabled ? "enabled" : "will be enabled"}` }];
    return [];
  });
}

function enabledAllocation(existing: DeviceResourceAllocation | undefined, deviceId: string): DeviceResourceAllocation {
  return existing ? { ...existing, enabled: true } : { deviceId, enabled: true, mode: "manual", percent: 75, autoIncrease: false };
}

export function setRegionalExecutionRoute(settings: DeviceResourceSettings | undefined, regionId: string,
  choice: RegionalExecutionDeviceChoice, acknowledgeCudaFallback: boolean): DeviceResourceSettings {
  if (!regionId || regionId.includes("\0")) throw new Error("Select an exact prepared DecoderBlock region.");
  if (choice.device.backend === "cuda" && !acknowledgeCudaFallback) throw new Error("A CUDA route requires explicit packed CPU-tile fallback acknowledgement.");
  if (choice.device.backend !== "cuda" && choice.mode !== "native") throw new Error("CPU and Metal routes use native execution only.");
  const current = baseSettings(settings), allocation = enabledAllocation(current.allocations.find(value => value.deviceId === choice.device.id), choice.device.id);
  const route: RegionalExecutionRoute = { regionId, deviceId: choice.device.id, executionMode: choice.mode,
    ...(choice.mode === "packed-cpu-tile-fallback" ? { acknowledgePackedCpuFallback: true } : {}) };
  const routes = [...(current.regionalExecution?.routes ?? []).filter(value => value.regionId !== regionId), route];
  return { ...current, allocations: [...current.allocations.filter(value => value.deviceId !== choice.device.id), allocation],
    regionalExecution: { version: 1, enabled: true, mode: "manual", routes } };
}

export function removeRegionalExecutionRoute(settings: DeviceResourceSettings | undefined, regionId: string): DeviceResourceSettings {
  const current = baseSettings(settings), regional = current.regionalExecution ?? { version: 1 as const, enabled: false, mode: "manual" as const, routes: [] };
  return { ...current, regionalExecution: { ...regional, routes: regional.routes.filter(value => value.regionId !== regionId) } };
}

export function regionalExecutionStatusLabel(value: RegionalExecutionStatus): string {
  return value.active ? value.leaseValid ? `Active manual lease · ${value.reason}` : `Armed but invalid · ${value.reason}` : `Inactive · ${value.reason}`;
}

function StatusView({ value }: { value: RegionalExecutionStatus }) {
  return <div className="regional-execution__status" role="status">
    <div className="regional-execution__metrics">
      <span><small>Registered</small><strong>{value.registeredRegions.toLocaleString()}</strong></span>
      <span><small>Active calls</small><strong>{value.activeCalls.toLocaleString()}</strong></span>
      <span><small>Completed dispatches</small><strong>{value.completedRegionDispatches.toLocaleString()}</strong></span>
      <span><small>Activation transfers</small><strong>{value.actualActivationTransfers.toLocaleString()} · {bytes(value.actualActivationTransferBytes)}</strong></span>
      <span><small>Known transfer time</small><strong>{value.estimatedKnownTransferSeconds.toLocaleString()} s</strong></span>
      <span><small>Unknown transfer costs</small><strong>{value.unknownTransferCosts.toLocaleString()}</strong></span>
      <span><small>Training-inapplicable calls</small><strong>{value.trainingInapplicableCalls.toLocaleString()}</strong></span>
    </div>
    <p>{regionalExecutionStatusLabel(value)}</p>
    <p className="regional-execution__note">Inference routes only. Training support: no. Automatic default: no. GPU context capacity credited: no. Whole-path performance measured: no.</p>
  </div>;
}

export function RegionalExecutionControls({ api, brainId, loaded, foregroundTurnActive, settings, dirty, disabled, onChange }: {
  api?: RegionalExecutionApi; brainId: string; loaded: boolean; foregroundTurnActive: boolean; settings?: DeviceResourceSettings;
  dirty: boolean; disabled?: boolean; onChange(settings: DeviceResourceSettings): void;
}) {
  const [prepared, setPrepared] = useState<RegionalExecutionPrepared>();
  const [activationReviewId, setActivationReviewId] = useState<string>();
  const [status, setStatus] = useState<RegionalExecutionStatus>();
  const [busy, setBusy] = useState(""); const [error, setError] = useState("");
  const [count, setCount] = useState(64);
  const [deviceByRegion, setDeviceByRegion] = useState<Record<string, string>>({});
  const [cudaAcknowledged, setCudaAcknowledged] = useState<Record<string, boolean>>({});
  const alive = useRef(false), generation = useRef(0), flight = useRef<object | undefined>(undefined);
  const settingsKey = JSON.stringify(settings ?? null);
  useEffect(() => {
    alive.current = true; const owner = ++generation.current; flight.current = undefined;
    setPrepared(undefined); setActivationReviewId(undefined); setStatus(undefined); setBusy(""); setError(""); setCount(64);
    setDeviceByRegion({}); setCudaAcknowledged({});
    return () => { if (generation.current === owner) generation.current++; alive.current = false; };
  }, [api, brainId, loaded]);
  useEffect(() => {
    // Preserve the inspected page so several exact region preferences can be
    // drafted together, but a settings change retires its activation authority.
    generation.current++; flight.current = undefined; setActivationReviewId(undefined);
    setStatus(undefined); setBusy(""); setError("");
  }, [settingsKey]);
  const perform = async (label: string, operation: (current: () => boolean) => Promise<void>) => {
    if (flight.current) return;
    const owner = generation.current, token = {}; flight.current = token; setBusy(label); setError("");
    const current = () => alive.current && generation.current === owner && flight.current === token;
    try { await operation(current); } catch (reason) { if (current()) setError(failure(reason)); }
    finally { if (current()) { flight.current = undefined; setBusy(""); } }
  };
  const prepare = (start: number) => perform("Preparing exact regional registry…", async current => {
    if (!api) throw new Error("Regional execution requires the desktop runtime.");
    const value = await api.prepare({ brainId, start, count });
    if (value.ownerId !== brainId) throw new Error("Regional review belongs to another instance.");
    if (current()) { setPrepared(value); setActivationReviewId(value.reviewId); setStatus(value.status); setDeviceByRegion({}); setCudaAcknowledged({}); }
  });
  const queryStatus = () => perform("Reading regional status…", async current => {
    if (!api) throw new Error("Regional execution requires the desktop runtime.");
    const value = await api.status(brainId); if (value.ownerId !== brainId) throw new Error("Regional status belongs to another instance.");
    if (current()) setStatus(value);
  });
  const choices = regionalExecutionDeviceChoices(prepared?.topology.devices ?? [], settings);
  const mutationBlocked = Boolean(disabled || busy || !loaded || foregroundTurnActive);
  const settingsBlocked = Boolean(disabled || busy);
  const selectedChoice = (region: RegionalExecutionRegion): RegionalExecutionDeviceChoice | undefined => {
    const savedDevice = settings?.regionalExecution?.routes.find(value => value.regionId === region.regionId)?.deviceId;
    const selected = deviceByRegion[region.regionId] ?? (choices.some(value => value.device.id === savedDevice) ? savedDevice : choices[0]?.device.id);
    return choices.find(value => value.device.id === selected);
  };
  const pageEnd = prepared ? prepared.start + prepared.regions.length : 0;
  return <details className="regional-execution">
    <summary>Advanced · manual regional execution</summary>
    <div className="regional-execution__body">
      <p>Explicit inference-only DecoderBlock routing. Saved preferences never arm a route; Prepare and Activate are separate current-owner operations.</p>
      {!api ? <p>Regional execution requires the desktop runtime.</p> : null}
      {!loaded ? <p>Load this instance before reading status, preparing routes, or changing the live lease.</p> : null}
      {foregroundTurnActive ? <p>Status remains available during a reply. Wait for the reply boundary before Prepare, Activate, or Deactivate.</p> : null}
      {dirty ? <p className="regional-execution__warning" role="status">Apply settings, then Prepare routes again.</p> : null}
      {prepared && !activationReviewId ? <p className="regional-execution__note" role="status">Descriptor page retained for editing; its activation review is stale. Prepare again after settings are applied.</p> : null}
      <div className="regional-execution__actions">
        <label>Descriptors per page <input type="number" min={1} max={256} value={count} disabled={Boolean(disabled || busy)}
          onChange={event => setCount(Math.max(1, Math.min(256, Number.isSafeInteger(event.target.valueAsNumber) ? event.target.valueAsNumber : 64)))} /></label>
        <button type="button" className="button" disabled={!api || mutationBlocked || dirty} onClick={() => void prepare(0)}>Prepare routes</button>
        <button type="button" className="button" disabled={!api || Boolean(disabled || busy) || !loaded} onClick={() => void queryStatus()}>Refresh status</button>
        <button type="button" className="button" disabled={!api || mutationBlocked || !status?.active} onClick={() => void perform("Deactivating regional routes…", async current => {
          const value = await api!.deactivate({ brainId, ...(status?.epoch ? { epoch: status.epoch } : {}) });
          if (value.ownerId !== brainId) throw new Error("Regional status belongs to another instance."); if (current()) { setStatus(value); setPrepared(undefined); setActivationReviewId(undefined); }
        })}>Deactivate</button>
      </div>
      {busy ? <p role="status">{busy}</p> : null}
      {error ? <p role="alert" className="regional-execution__warning">{error}</p> : null}
      {status ? <StatusView value={status} /> : null}

      <fieldset disabled={settingsBlocked}>
        <legend>Saved manual route preferences</legend>
        <label><input type="checkbox" checked={settings?.regionalExecution?.enabled ?? false} onChange={event => {
          const current = baseSettings(settings), regional = current.regionalExecution ?? { version: 1 as const, enabled: false, mode: "manual" as const, routes: [] };
          onChange({ ...current, regionalExecution: { ...regional, enabled: event.target.checked } });
        }} /> Enable saved manual routes (does not activate)</label>
        {settings?.regionalExecution?.routes.length ? <ul className="regional-execution__routes">{settings.regionalExecution.routes.map(route => {
          const allocationEnabled = settings.allocations.some(value => value.deviceId === route.deviceId && value.enabled);
          return <li key={route.regionId}>
            <code>{route.regionId}</code><span>{route.deviceId} · {route.executionMode === "native" ? "native" : "packed CPU-tile fallback · unknown speed · not native CUDA"} · allocation {allocationEnabled ? "enabled" : "disabled"}</span>
            <button type="button" className="button button--ghost" onClick={() => onChange(removeRegionalExecutionRoute(settings, route.regionId))}>Remove</button>
          </li>;
        })}</ul> : <p className="regional-execution__note">No saved regional routes.</p>}
        <p className="regional-execution__note">Changing preferences only marks the parent settings dirty. Use its existing Apply control, then Prepare and Activate explicitly.</p>
      </fieldset>

      {prepared ? <section className="regional-execution__registry" aria-label="Prepared DecoderBlock registry">
        <div className="regional-execution__registry-head"><h3>Exact DecoderBlock registry</h3><span>{prepared.regions.length ? `${prepared.start + 1}–${pageEnd}` : "0"} of {prepared.total.toLocaleString()}</span></div>
        <p>Registry hash <code>{prepared.registrySha256}</code>. This page is not an all-region claim; every descriptor remains pageable.</p>
        {prepared.topology.devices.filter(device => choices.some(value => value.device.id === device.id)).map(device => <p className="regional-execution__device" key={device.id}>
          <strong>{device.label}</strong> · {device.backend} · observed rate {rate(device.measurement?.bytesPerSecond)}
        </p>)}
        {!choices.length ? <p role="alert" className="regional-execution__warning">No eligible CPU, Metal, or explicitly acknowledged CUDA-fallback device is available.</p> : null}
        <ul className="regional-execution__regions">{prepared.regions.map(region => {
          const savedRoute = settings?.regionalExecution?.routes.find(value => value.regionId === region.regionId);
          const choice = selectedChoice(region), cuda = choice?.device.backend === "cuda", already = Boolean(savedRoute);
          const cudaAck = cudaAcknowledged[region.regionId] ?? Boolean(cuda && savedRoute?.deviceId === choice?.device.id && savedRoute.acknowledgePackedCpuFallback === true);
          return <li key={region.regionId}><div><code>{region.regionId}</code><span>{region.dimensions.toLocaleString()} dims · {region.heads.toLocaleString()} heads · {region.feedForward.toLocaleString()} FF · {region.packedOwners.toLocaleString()} packed owners · {bytes(region.packedOwnerBytes)}</span></div>
            <label>Route device <select value={choice?.device.id ?? ""} disabled={!choices.length || settingsBlocked}
              onChange={event => { setDeviceByRegion(current => ({ ...current, [region.regionId]: event.target.value })); setCudaAcknowledged(current => ({ ...current, [region.regionId]: false })); }}>
              {choices.map(value => <option key={value.device.id} value={value.device.id}>{value.label}</option>)}</select></label>
            {cuda ? <label className="regional-execution__cuda"><input type="checkbox" checked={cudaAck} disabled={settingsBlocked}
              onChange={event => setCudaAcknowledged(current => ({ ...current, [region.regionId]: event.target.checked }))} /> I understand this is packed CPU-tile fallback, with unknown speed, not native CUDA.</label> : null}
            <button type="button" className="button" disabled={!choice || settingsBlocked || Boolean(cuda && !cudaAck)}
              onClick={() => choice && onChange(setRegionalExecutionRoute(settings, region.regionId, choice, cudaAck))}>{already ? "Update saved route" : "Add saved route · enable allocation"}</button>
          </li>;
        })}</ul>
        <div className="regional-execution__actions">
          <button type="button" className="button" disabled={mutationBlocked || dirty || prepared.start === 0}
            onClick={() => void prepare(Math.max(0, prepared.start - count))}>Previous regions</button>
          <button type="button" className="button" disabled={mutationBlocked || dirty || pageEnd >= prepared.total}
            onClick={() => void prepare(pageEnd)}>Next regions</button>
          <button type="button" className="button button--primary" disabled={!api || !activationReviewId || mutationBlocked || dirty || !settings?.regionalExecution?.enabled || !settings.regionalExecution.routes.length}
            onClick={() => void perform("Activating reviewed routes…", async current => {
              if (!activationReviewId) throw new Error("Prepare the current saved routes before activation.");
              const value = await api!.activate({ brainId, reviewId: activationReviewId });
              if (value.ownerId !== brainId) throw new Error("Regional status belongs to another instance."); if (current()) { setStatus(value); setPrepared(undefined); setActivationReviewId(undefined); }
            })}>Activate reviewed routes</button>
        </div>
      </section> : null}
    </div>
  </details>;
}
