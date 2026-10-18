import { useEffect, useMemo, useRef, useState } from "react";
import type {
  BrainDocument,
  HardwareTier,
  LiveObservationCaptureMode
} from "@shared/types";
import {
  LivePerceptionController,
  createBrowserLivePerceptionCapture,
  planLivePerception,
  type LivePerceptionPlanInput,
  type LivePerceptionPreview,
  type LivePerceptionSnapshotResolution,
  type LivePerceptionSource,
  type LivePerceptionState
} from "./livePerceptionController";
import { Icon } from "./icons";
import { livePerceptionErrorMessage } from "./uiPresentation";
import { LiveCapturePreview } from "./LiveCapturePreview";
import "./cameraScreenPanel.css";

const CAPTURE_MODES: LiveObservationCaptureMode[] = [
  "auto",
  "motion",
  "balanced",
  "detail"
];

const SOURCES: Array<{ id: LivePerceptionSource; label: string }> = [
  { id: "camera", label: "Camera" },
  { id: "screen", label: "Screen" }
];

const EMPTY_STATE: LivePerceptionState = {
  enabled: false,
  phase: "idle",
  localPacketsDropped: 0,
  lastActions: []
};

function measuredTokensPerSecond(brain: BrainDocument): number | undefined {
  for (const trace of [...brain.traces].reverse()) {
    for (const step of [...trace.steps].reverse()) {
      const match = step.value?.match(/([\d.]+)\s+tokens?\/s/i);
      const value = match ? Number(match[1]) : Number.NaN;
      if (Number.isFinite(value) && value > 0) return value;
    }
  }
  return undefined;
}

function sourceAvailable(
  source: LivePerceptionSource,
  capabilities: { camera: boolean; microphone: boolean; screen: boolean }
): boolean {
  if (source === "camera") return capabilities.camera;
  if (source === "microphone") return capabilities.microphone;
  if (source === "screen") return capabilities.screen;
  return capabilities.camera && capabilities.microphone;
}

function titleCase(value: string): string {
  return value.slice(0, 1).toLocaleUpperCase() + value.slice(1);
}

function captureLabel(state: LivePerceptionState): string {
  const capture = state.negotiatedCapture ?? state.session?.capture;
  if (!capture) return "Waiting for an explicit source";
  if (capture.width && capture.height) {
    return `${capture.width}×${capture.height}${capture.fps ? ` @ ${capture.fps} FPS` : ""}`;
  }
  if (capture.audioSampleRate) return `${capture.audioSampleRate.toLocaleString()} Hz audio`;
  return "Audio source negotiated";
}

function controlLabel(state: LivePerceptionState): string | null {
  const control = state.activeControl ?? state.lastControl;
  if (!control) return null;
  const actor = control.source === "brain" ? "Brain" : "You";
  const actual = control.actual;
  const capture = actual?.width && actual.height
    ? ` · ${actual.width}×${actual.height}${actual.fps ? ` @ ${actual.fps} FPS` : ""}`
    : "";
  const duration = actual?.durationMs ? ` · ${Math.round(actual.durationMs / 1_000)}s` : "";
  const revision = actual?.revision !== undefined ? ` · r${actual.revision}` : "";
  return `${actor} ${control.kind} · ${control.state}${capture}${duration}${revision}`;
}

export function LivePerceptionPanel({
  brain,
  runtimeLoaded = true
}: {
  brain: BrainDocument;
  runtimeLoaded?: boolean;
}) {
  // A different instance must never inherit a view of the previous device handle.
  return <CameraScreenPanel key={brain.id} brain={brain} runtimeLoaded={runtimeLoaded} />;
}

function CameraScreenPanel({
  brain,
  runtimeLoaded = true
}: {
  brain: BrainDocument;
  runtimeLoaded?: boolean;
}) {
  const captureAdapter = useMemo(() => createBrowserLivePerceptionCapture(), []);
  const controllerRef = useRef<LivePerceptionController | null>(null);
  const operation = useRef(0);
  const [preview, setPreview] = useState<Readonly<LivePerceptionPreview>>({});
  const defaultTier = brain.readiness.recovery?.foundation.hardwareTier ?? "personal";
  const measuredTps = useMemo(() => measuredTokensPerSecond(brain), [brain.traces]);
  const { workingMemoryMode, workingMemorySlots, systemRamMode, systemRamSharePercent } = brain.config;
  const [planInput, setPlanInput] = useState<LivePerceptionPlanInput>({
    hardwareTier: defaultTier,
    mode: "auto",
    generationTokensPerSecond: measuredTps
  });
  const [state, setState] = useState<LivePerceptionState>(EMPTY_STATE);
  const [source, setSource] = useState<LivePerceptionSource>(
    captureAdapter.capabilities.camera
      ? "camera"
      : "screen"
  );
  const [delivery, setDelivery] = useState<"continuous" | "snapshots">("continuous");
  const [resolutionMode, setResolutionMode] =
    useState<LivePerceptionSnapshotResolution>("native");
  const [customWidth, setCustomWidth] = useState("1280");
  const [customHeight, setCustomHeight] = useState("720");
  const [burstCount, setBurstCount] = useState(3);
  const [intervalMs, setIntervalMs] = useState(250);
  const [pending, setPending] = useState(false);
  const [localError, setLocalError] = useState("");
  const requestedPlan = useMemo(() => planLivePerception(planInput), [planInput]);
  const plan = state.enabled && state.plan ? state.plan : requestedPlan;

  useEffect(() => {
    setPlanInput((current) => ({
      ...current,
      hardwareTier: defaultTier,
      generationTokensPerSecond: measuredTps
    }));
  }, [measuredTps, defaultTier]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void (async () => {
      const profile = await window.omni!.catalog.hardwareProfile().catch(() => null);
      const hardwareTier: HardwareTier = profile?.recommendedTier ?? defaultTier;
      const resource = await window.omni!.catalog.resourcePlan({
        mode: workingMemoryMode,
        hardwareTier,
        ...(workingMemoryMode === "manual"
          ? { requestedItems: String(workingMemorySlots) }
          : {}),
        systemRamMode,
        ...(systemRamMode === "manual"
          ? { systemRamSharePercent }
          : {})
      }).catch(() => null);
      if (!active) return;
      setPlanInput((current) => ({
        ...current,
        hardwareTier,
        ...(resource
          ? {
            memoryBytesPerSecond: resource.offload.benchmark.memoryBytesPerSecond,
            ioBytesPerSecond: resource.offload.benchmark.storageBytesPerSecond
          }
          : {})
      }));
    })();
    return () => {
      active = false;
    };
  }, [workingMemoryMode, workingMemorySlots, systemRamMode, systemRamSharePercent, defaultTier]);

  useEffect(() => {
    if (!window.omni) {
      setState(EMPTY_STATE);
      return;
    }
    const controller = new LivePerceptionController({
      brainId: brain.id,
      modality: window.omni.modality,
      capture: captureAdapter
    });
    controllerRef.current = controller;
    const unsubscribe = controller.subscribe((next) => setState({ ...next }));
    const unsubscribePreview = controller.subscribePreview(setPreview);
    return () => {
      operation.current += 1;
      unsubscribe();
      unsubscribePreview();
      if (controllerRef.current === controller) controllerRef.current = null;
      void controller.dispose();
    };
  }, [brain.id, captureAdapter]);

  useEffect(() => {
    if (runtimeLoaded) return;
    operation.current += 1;
    setPending(false);
    const controller = controllerRef.current;
    // Invalidates a pending OS prompt too; its eventual stream is closed by the owner.
    void controller?.cancel();
  }, [runtimeLoaded]);

  useEffect(() => {
    setBurstCount((current) => Math.min(Math.max(1, current), plan.maxSnapshotBurst));
    setIntervalMs((current) => Math.max(current, plan.minSnapshotIntervalMs));
  }, [plan.maxSnapshotBurst, plan.minSnapshotIntervalMs]);

  const run = async (task: (controller: LivePerceptionController) => Promise<unknown>, requiresLoaded = true) => {
    const controller = controllerRef.current;
    if (!controller) {
      setLocalError("Camera and screen sharing require the packaged desktop runtime.");
      return;
    }
    if (requiresLoaded && !runtimeLoaded) {
      setLocalError("Load this instance before starting capture.");
      return;
    }
    const revision = ++operation.current;
    setPending(true);
    setLocalError("");
    try {
      await task(controller);
    } catch (error) {
      if (controllerRef.current === controller && revision === operation.current) {
        setLocalError(livePerceptionErrorMessage(error));
      }
    } finally {
      if (controllerRef.current === controller && revision === operation.current) setPending(false);
    }
  };

  const selectMode = (mode: LiveObservationCaptureMode): void => {
    setPlanInput((current) => ({ ...current, mode }));
    if (state.enabled) {
      void run((controller) => controller.requestConfigure({ mode, durationMs: 15_000 }));
    }
  };

  const requestSnapshot = (): void => {
    void run((controller) => controller.requestSnapshot({
      resolutionMode,
      ...(resolutionMode === "custom"
        ? { width: Number(customWidth), height: Number(customHeight) }
        : {}),
      burstCount,
      intervalMs
    }));
  };

  const cancelSnapshot = (): void => {
    controllerRef.current?.cancelSnapshot();
  };

  const capture = state.negotiatedCapture ?? state.session?.capture;
  const selectedMode = state.enabled ? capture?.mode ?? plan.mode : plan.mode;
  const snapshotBusy = state.snapshot?.state === "capturing";
  const sessionNeedsClose = !state.enabled && state.session?.state === "active";
  const status = state.phase === "observing"
    ? "Sharing"
    : state.phase === "requesting"
      ? "Requesting permission"
      : state.phase === "stopping"
        ? "Stopping"
        : state.phase === "error"
          ? "Needs attention"
          : "Off";
  const latestControl = controlLabel(state);

  return (
    <section className="camera-screen-panel" aria-label="Camera and screen sharing">
      <header>
        <span className="live-perception-panel__icon"><Icon name="activity" size={14} /></span>
        <span>
          <strong>Camera & screen</strong>
          <small>{state.enabled ? `${titleCase(state.source ?? source)} · ${captureLabel(state)}` : "Share a source when you choose"}</small>
        </span>
        <em className={state.enabled ? "is-active" : ""}><i /> {status}</em>
      </header>
      <div className="camera-screen-panel__body">
        <LiveCapturePreview preview={preview} state={state} loaded={runtimeLoaded} />
        <div className="live-perception-toolbar">
          <fieldset>
            <legend>Source</legend>
            {SOURCES.map((item) => {
              const available = sourceAvailable(item.id, captureAdapter.capabilities);
              return (
                <button
                  key={item.id}
                  type="button"
                  aria-pressed={source === item.id}
                  disabled={!available || state.enabled || state.phase === "requesting" || state.phase === "stopping"}
                  title={available ? `Share ${item.label.toLocaleLowerCase()} after Start` : `${item.label} is unavailable in this runtime`}
                  onClick={() => setSource(item.id)}
                >
                  {item.label}
                </button>
              );
            })}
          </fieldset>
          <fieldset>
            <legend>Sharing mode</legend>
            {(["continuous", "snapshots"] as const).map(mode => <button type="button" key={mode} aria-pressed={delivery === mode}
              disabled={state.enabled || state.phase === "requesting" || state.phase === "stopping"}
              title={mode === "continuous" ? "Share frames continuously within the device resource plan" : "Share only audited snapshots requested by you or the brain"}
              onClick={() => setDelivery(mode)}>{mode === "continuous" ? "Stream" : "Snapshots"}</button>)}
          </fieldset>
          <div className="live-perception-actions">
            {!state.enabled && !sessionNeedsClose && state.phase !== "requesting" && state.phase !== "stopping" ? (
              <button
                type="button"
                className="is-primary"
                disabled={!runtimeLoaded || pending || !sourceAvailable(source, captureAdapter.capabilities)}
                onClick={() => void run((controller) => controller.start(source, plan, "neural", delivery))}
              >
                <Icon name="play" size={12} /> Start
              </button>
            ) : null}
            {state.enabled || sessionNeedsClose ? (
              <button type="button" onClick={() => void run((controller) => controller.stop(), false)}>
                <Icon name="close" size={12} /> Stop
              </button>
            ) : null}
            {state.phase === "requesting" ? (
              <button type="button" onClick={() => void run((controller) => controller.cancel(), false)}>
                Cancel request
              </button>
            ) : null}
            {state.enabled ? <button type="button" disabled={pending || snapshotBusy} onClick={requestSnapshot} title="Capture an audited snapshot burst with the selected resolution">
              <Icon name="image" size={12} /> Snapshot
            </button> : null}
          </div>
        </div>

        {state.error || localError ? (
          <p className="live-perception-error" role="alert"><Icon name="warning" size={13} /> {localError || state.error}</p>
        ) : null}

        <details className="live-perception-advanced">
          <summary>Quality, snapshots & activity</summary>
          <div>
            <fieldset className="camera-screen-quality">
              <legend>Capture priority</legend>
              {CAPTURE_MODES.map((mode) => <button type="button" key={mode} aria-pressed={selectedMode === mode}
                disabled={pending || state.phase === "requesting" || state.phase === "stopping"}
                title={mode === "motion" ? "Prioritize movement and frame rate" : mode === "detail" ? "Prioritize still-image detail" : mode === "auto" ? "Adapt to actual source and device resources" : "Balance detail and movement"}
                onClick={() => selectMode(mode)}>{mode === "detail" ? "Detail" : titleCase(mode)}</button>)}
            </fieldset>
            <section className="live-perception-snapshot" aria-labelledby="snapshot-burst-title">
              <div>
                <strong id="snapshot-burst-title">Snapshot burst</strong>
                <small>Native, current, or custom resolution. Each frame is submitted separately.</small>
              </div>
              <fieldset>
                <legend>Resolution</legend>
                {(["native", "current", "custom"] as const).map((resolution) => (
                  <button
                    key={resolution}
                    type="button"
                    aria-pressed={resolutionMode === resolution}
                    disabled={snapshotBusy}
                    onClick={() => setResolutionMode(resolution)}
                  >
                    {titleCase(resolution)}
                  </button>
                ))}
              </fieldset>
              {resolutionMode === "custom" ? (
                <div className="live-perception-custom-size">
                  <label><span>Width</span><input inputMode="numeric" value={customWidth} onChange={(event) => setCustomWidth(event.target.value)} /></label>
                  <span aria-hidden="true">×</span>
                  <label><span>Height</span><input inputMode="numeric" value={customHeight} onChange={(event) => setCustomHeight(event.target.value)} /></label>
                </div>
              ) : null}
              <div className="live-perception-burst-values">
                <label>
                  <span>Frames</span>
                  <input type="number" min="1" max={plan.maxSnapshotBurst} value={burstCount} onChange={(event) => setBurstCount(Number(event.target.value))} />
                </label>
                <label>
                  <span>Interval (ms)</span>
                  <input type="number" min={plan.minSnapshotIntervalMs} value={intervalMs} onChange={(event) => setIntervalMs(Number(event.target.value))} />
                </label>
              </div>
              <div className="live-perception-snapshot-actions">
                <button
                  className="is-primary"
                  type="button"
                  disabled={!state.enabled || pending || snapshotBusy}
                  onClick={requestSnapshot}
                >
                  <Icon name="image" size={12} /> Capture burst
                </button>
                {snapshotBusy ? <button type="button" onClick={cancelSnapshot}>Stop burst</button> : null}
                {state.snapshot ? (
                  <span role="status">{titleCase(state.snapshot.state)} · {state.snapshot.acceptedFrames}/{state.snapshot.requestedFrames} accepted</span>
                ) : null}
              </div>
            </section>

            <div className="live-perception-facts">
              <span><small>Capture</small><strong>{captureLabel(state)}</strong></span>
              <span><small>Resources</small><strong>{titleCase(capture?.benchmarkClass ?? plan.benchmarkClass)}</strong></span>
              <span><small>Accepted</small><strong>{state.session?.packetsAccepted.toLocaleString() ?? "0"}</strong></span>
              <span><small>Backpressure</small><strong>{((state.session?.packetsDroppedBackpressure ?? 0) + state.localPacketsDropped).toLocaleString()}</strong></span>
            </div>

            <dl className="live-perception-audit">
              <div><dt>Source native</dt><dd>{capture?.sourceNativeWidth && capture.sourceNativeHeight ? `${capture.sourceNativeWidth}×${capture.sourceNativeHeight}${capture.sourceNativeFps ? ` @ ${capture.sourceNativeFps} FPS` : ""}` : "Available after source negotiation"}</dd></div>
              <div><dt>Capture revision</dt><dd>{capture ? `r${capture.revision}` : "—"}</dd></div>
              <div><dt>Transport window</dt><dd>{state.session ? `${state.session.maxInFlight} in flight · ${state.session.maxPacketBytes.toLocaleString()} bytes/packet` : `${plan.maxInFlight} planned`}</dd></div>
              <div><dt>Current plan</dt><dd>{state.enabled && capture ? `${captureLabel(state)} · ${titleCase(capture.mode)} priority` : plan.reason}</dd></div>
              <div><dt>Last control</dt><dd>{latestControl ?? "No human or brain capture control yet"}</dd></div>
              <div><dt>Brain actions</dt><dd>{state.lastActions.length ? state.lastActions.map((action) => action.kind).join(", ") : "None from the latest observation"}</dd></div>
            </dl>
            <p className="live-perception-privacy">
              The preview stays on this device. Raw packets are not stored. Accepted perception may update working or neural state, but it does not commit dataset coverage and it never injects a hidden behavioral prompt.
            </p>
          </div>
        </details>
      </div>
    </section>
  );
}
