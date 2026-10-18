import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { RegionalExecutionControls } from "../../src/renderer/src/RegionalExecutionControls";
import type { DeviceResourceSettings, DeviceResourceTopology, RegionalExecutionRegion, RegionalExecutionStatus,
  ResourceDevice } from "../../src/shared/deviceResources";
import type { RegionalExecutionApi, RegionalExecutionPrepared } from "../../src/shared/regionalExecution";
import "../../src/renderer/src/styles.css";

const BRAIN_ID = "regional-fixture-brain", GIB = 1024 ** 3;
const device = (id: string, backend: ResourceDevice["backend"], label: string, nativeComputeSupported: boolean,
  unavailableReason?: string): ResourceDevice => ({ id, label, backend, kind: backend === "cpu" ? "ram" : backend === "npu" ? "npu" : "gpu",
  executionDevice: backend === "cpu" ? "cpu" : backend === "mps" ? "mps" : backend === "cuda" ? "cuda:0" : "npu:0",
  memoryGroupId: backend === "cpu" || backend === "mps" ? "ram:system" : `memory:${id}`,
  totalBytes: 16 * GIB, availableBytes: 11 * GIB, allocatedBytes: 0, reservedBytes: 2 * GIB,
  unifiedMemory: backend === "mps", nativeComputeSupported, ...(unavailableReason ? { unavailableReason } : {}),
  measurement: backend === "npu" ? undefined : { bytesPerSecond: backend === "cpu" ? 28 * GIB : backend === "mps" ? 18 * GIB : 9 * GIB,
    sampleBytes: 1024 ** 2, runs: 3, medianNanoseconds: 1000, measuredAt: "2026-10-15T12:00:00.000Z",
    scope: backend === "cpu" ? "memory-copy" : "device-transfer" } });
const TOPOLOGY: DeviceResourceTopology = { format: "omni-device-resource-topology", version: 1,
  measuredAt: "2026-10-15T12:00:00.000Z", platform: "synthetic-no-engine", architecture: "arm64",
  devices: [device("ram:system", "cpu", "Synthetic CPU", true), device("gpu:metal", "mps", "Synthetic Metal", true),
    device("gpu:cuda", "cuda", "Synthetic CUDA", false, "Native CUDA packed kernels are unavailable."),
    device("npu:unavailable", "npu", "Unavailable synthetic NPU", false, "No supported NPU route exists.")],
  links: [], measurementRequested: true, neuralModelConstructed: false,
  warnings: ["Synthetic topology for UI inspection only; no device was measured."] };
const REGIONS: RegionalExecutionRegion[] = [
  { regionId: "root.decoder.blocks.000", dimensions: 512, heads: 8, feedForward: 2048, packedOwners: 4,
    packedOwnerBytes: 8192, maximumPackedOwnerBytes: 16384, maximumDevicePackedOwnerBytes: 8192 },
  { regionId: "root.decoder.blocks.001", dimensions: 512, heads: 8, feedForward: 2048, packedOwners: 4,
    packedOwnerBytes: 8192, maximumPackedOwnerBytes: 16384, maximumDevicePackedOwnerBytes: 8192 },
  { regionId: "connected.parent-a.decoder.blocks.000", dimensions: 768, heads: 12, feedForward: 3072, packedOwners: 6,
    packedOwnerBytes: 12288, maximumPackedOwnerBytes: 24576, maximumDevicePackedOwnerBytes: 12288 }
];
const INITIAL_SETTINGS: DeviceResourceSettings = { version: 1, allocations: [
  { deviceId: "ram:system", enabled: true, mode: "manual", percent: 62, autoIncrease: false },
  { deviceId: "gpu:metal", enabled: true, mode: "manual", percent: 80, autoIncrease: false },
  { deviceId: "gpu:cuda", enabled: false, mode: "manual", percent: 37, autoIncrease: false }
], allowStorageSpill: true, autoIncreaseStorage: false, storageDirectories: [],
regionalExecution: { version: 1, enabled: false, mode: "manual", routes: [] } };
// Isolated, explicitly synthetic presentation state for the two affected UI
// regressions. This never configures production authority or a real device.
const INSPECT_SAVED_INVALID = new URLSearchParams(location.search).get("inspection") === "saved-cuda-invalid";
if (INSPECT_SAVED_INVALID) {
  INITIAL_SETTINGS.allocations.find(value => value.deviceId === "gpu:cuda")!.enabled = true;
  INITIAL_SETTINGS.regionalExecution = { version: 1, enabled: true, mode: "manual", routes: [
    { regionId: "root.decoder.blocks.001", deviceId: "gpu:cuda", executionMode: "packed-cpu-tile-fallback", acknowledgePackedCpuFallback: true }
  ] };
}

function Fixture() {
  const [loaded, setLoaded] = useState(true), [foregroundBusy, setForegroundBusy] = useState(false);
  const [apiOwner, setApiOwner] = useState(1), [saved, setSaved] = useState(INITIAL_SETTINGS);
  const [draft, setDraft] = useState(INITIAL_SETTINGS), [logs, setLogs] = useState<string[]>([
    "NO-ENGINE fixture ready · synthetic API/counters only · neural effects 0"
  ]);
  const savedRef = useRef(saved), loadedRef = useRef(loaded), busyRef = useRef(foregroundBusy);
  savedRef.current = saved; loadedRef.current = loaded; busyRef.current = foregroundBusy;
  const log = (value: string): void => setLogs(current => [...current.slice(-9), value]);
  const settingsJson = JSON.stringify(draft), savedJson = JSON.stringify(saved), dirty = settingsJson !== savedJson;
  const api = useMemo<RegionalExecutionApi>(() => {
    const reviews = new Map<string, string>(); let reviewSequence = 0;
    let currentStatus: RegionalExecutionStatus = { format: "omni-manual-regional-execution-status", version: 1,
      ownerId: BRAIN_ID, active: false, leaseValid: false, epoch: null, reason: "Synthetic inactive status",
      registeredRegions: 3, activeCalls: 0, routes: [], completedRegionDispatches: 7, actualActivationTransfers: 3,
      actualActivationTransferBytes: 24576, trainingInapplicableCalls: 2, estimatedKnownTransferSeconds: 0.03,
      unknownTransferCosts: 1, trainingSupported: false, automaticDefault: false, contextGpuCapacityCredited: false,
      wholePathPerformanceMeasured: false, trace: [], ...(INSPECT_SAVED_INVALID ? {
        active: true, leaseValid: false, epoch: "synthetic-invalid-epoch", reason: "Synthetic stale device review; no dispatch"
      } : {}) };
    const requireBoundary = (): void => { if (!loadedRef.current) throw new Error("Synthetic explicit Load required.");
      if (busyRef.current) throw new Error("Synthetic foreground reply must finish first."); };
    return {
      prepare: async request => {
        requireBoundary(); const start = request.start ?? 0, count = request.count ?? 2;
        if (!Number.isSafeInteger(start) || start < 0 || !Number.isSafeInteger(count) || count < 1 || count > 256) throw new Error("Invalid synthetic registry page.");
        const page = start === 0 ? REGIONS.slice(0, Math.min(2, count)) : REGIONS.slice(start, start + count);
        const reviewId = `synthetic-review-${apiOwner}-${++reviewSequence}`, snapshot = JSON.stringify(savedRef.current); reviews.set(reviewId, snapshot);
        log(`Synthetic Prepare · owner ${apiOwner} · descriptors ${start + 1}–${start + page.length} of 3 · neural effects 0`);
        return { format: "omni-manual-regional-execution-review", version: 1, reviewId, ownerId: BRAIN_ID,
          policySha256: "a".repeat(64), topologyIdentitySha256: "b".repeat(64), registrySha256: "c".repeat(64),
          topology: structuredClone(TOPOLOGY), settings: structuredClone(savedRef.current.regionalExecution ?? { version: 1, enabled: false, mode: "manual", routes: [] }),
          start, total: REGIONS.length, regions: structuredClone(page), status: structuredClone(currentStatus) } satisfies RegionalExecutionPrepared;
      },
      status: async brainId => { if (!loadedRef.current || brainId !== BRAIN_ID) throw new Error("Synthetic exact loaded owner required.");
        log(`Synthetic Status · owner ${apiOwner} · counters only · neural effects 0`); return structuredClone(currentStatus); },
      activate: async request => {
        requireBoundary(); const snapshot = reviews.get(request.reviewId); reviews.delete(request.reviewId);
        if (request.brainId !== BRAIN_ID || !snapshot || snapshot !== JSON.stringify(savedRef.current)) throw new Error("Synthetic review is stale; Prepare current saved settings again.");
        currentStatus = { ...currentStatus, active: true, leaseValid: true, epoch: `synthetic-epoch-${apiOwner}`,
          reason: "Synthetic manual lease for presentation only", routes: (savedRef.current.regionalExecution?.routes ?? []).map(route => ({ ...route,
            executionDevice: route.executionMode === "native" ? route.deviceId === "gpu:metal" ? "mps" : "cpu" : "cpu",
            memoryGroupId: route.deviceId === "gpu:metal" || route.deviceId === "ram:system" ? "ram:system" : `memory:${route.deviceId}`,
            nativeComputeSupported: route.executionMode === "native" })) };
        log(`Synthetic Activate · ${request.reviewId} consumed · displayed routes ${currentStatus.routes.length} · neural effects 0`);
        return structuredClone(currentStatus);
      },
      deactivate: async request => { requireBoundary();
        if (request.brainId !== BRAIN_ID || request.epoch && request.epoch !== currentStatus.epoch) throw new Error("Synthetic activation epoch changed.");
        currentStatus = { ...currentStatus, active: false, leaseValid: false, epoch: null, reason: "Synthetic lease deactivated" };
        log(`Synthetic Deactivate · owner ${apiOwner} · neural effects 0`); return structuredClone(currentStatus); }
    };
  }, [apiOwner]);

  return <main style={{ height: "100dvh", overflow: "auto", boxSizing: "border-box", maxWidth: 1080, margin: "0 auto", padding: 24 }}>
    <header><p className="eyebrow">NO-ENGINE UI fixture</p><h1>Production manual regional execution controls</h1>
      <p><strong>Synthetic API, topology, counters, reviews, and settings only.</strong> No preload, device probe, brain, Load, model, training, transfer, dispatch, or neural effect.</p></header>
    <section aria-label="Synthetic regional host controls" style={{ display: "flex", flexWrap: "wrap", gap: 8, marginBlock: 16 }}>
      <button className="button" type="button" aria-pressed={loaded} onClick={() => setLoaded(value => !value)}>{loaded ? "Loaded · synthetic" : "Unloaded · synthetic"}</button>
      <button className="button" type="button" aria-pressed={foregroundBusy} onClick={() => setForegroundBusy(value => !value)}>{foregroundBusy ? "Foreground busy" : "Foreground quiescent"}</button>
      <button className="button" type="button" onClick={() => { setApiOwner(value => value + 1); log("Synthetic API owner changed · old review tokens retired"); }}>Change API owner</button>
      <button className="button button--primary" type="button" disabled={!dirty} onClick={() => { setSaved(structuredClone(draft)); log("Apply synthetic settings · draft became saved · no route activated"); }}>Apply synthetic settings</button>
      <button className="button" type="button" disabled={!dirty} onClick={() => { setDraft(structuredClone(saved)); log("Discard synthetic draft"); }}>Discard draft</button>
      <button className="button" type="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
    </section>
    <p><strong>Settings:</strong> {dirty ? "draft differs from saved" : "draft matches saved"} · saved routes {saved.regionalExecution?.routes.length ?? 0} · draft routes {draft.regionalExecution?.routes.length ?? 0}</p>
    <p><strong>Synthetic topology:</strong> CPU and Metal native choices, CUDA packed CPU-tile fallback only, and one unavailable NPU that is intentionally not offered as a route.</p>
    <RegionalExecutionControls api={api} brainId={BRAIN_ID} loaded={loaded} foregroundTurnActive={foregroundBusy}
      settings={draft} dirty={dirty} onChange={value => { setDraft(value); log(`Synthetic draft changed · routes ${value.regionalExecution?.routes.length ?? 0} · no activation`); }} />
    <section aria-labelledby="regional-fixture-log"><h2 id="regional-fixture-log">Synthetic callback log</h2>
      <ol aria-live="polite">{logs.map((entry, index) => <li key={`${index}-${entry}`}>{entry}</li>)}</ol>
      <p><strong>Device probes:</strong> 0 · <strong>Neural dispatches:</strong> 0 · <strong>Training calls:</strong> 0 · <strong>Context credits:</strong> 0</p>
    </section>
  </main>;
}

document.documentElement.dataset.colorScheme = "dark";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
