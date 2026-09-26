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
  type LivePerceptionSnapshotResolution,
  type LivePerceptionSource,
  type LivePerceptionState
} from "./livePerceptionController";
import { Icon } from "./icons";
import { livePerceptionErrorMessage } from "./uiPresentation";

const CAPTURE_MODES: LiveObservationCaptureMode[] = [
  "auto",
  "motion",
  "balanced",
  "detail"
];

const SOURCES: Array<{ id: LivePerceptionSource; label: string }> = [
  { id: "camera", label: "Camera" },
  { id: "microphone", label: "Microphone" },
  { id: "screen", label: "Screen" },
  { id: "mixed", label: "Mixed" }
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
  brain
}: {
  brain: BrainDocument;
}) {
  const captureAdapter = useMemo(() => createBrowserLivePerceptionCapture(), []);
  const controllerRef = useRef<LivePerceptionController | null>(null);
  const defaultTier = brain.readiness.recovery?.foundation.hardwareTier ?? "personal";
  const [planInput, setPlanInput] = useState<LivePerceptionPlanInput>({
    hardwareTier: defaultTier,
    mode: "auto",
    generationTokensPerSecond: measuredTokensPerSecond(brain)
  });
  const [state, setState] = useState<LivePerceptionState>(EMPTY_STATE);
  const [source, setSource] = useState<LivePerceptionSource>(
    captureAdapter.capabilities.camera
      ? "camera"
      : captureAdapter.capabilities.screen
        ? "screen"
        : "microphone"
  );
  const [resolutionMode, setResolutionMode] =
    useState<LivePerceptionSnapshotResolution>("native");
  const [customWidth, setCustomWidth] = useState("1280");
  const [customHeight, setCustomHeight] = useState("720");
  const [burstCount, setBurstCount] = useState(3);
  const [intervalMs, setIntervalMs] = useState(250);
  const [pending, setPending] = useState(false);
  const [localError, setLocalError] = useState("");
  const plan = useMemo(() => planLivePerception(planInput), [planInput]);

  useEffect(() => {
    setPlanInput((current) => ({
      ...current,
      hardwareTier: defaultTier,
      generationTokensPerSecond: measuredTokensPerSecond(brain)
    }));
  }, [brain, defaultTier]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void (async () => {
      const profile = await window.omni!.catalog.hardwareProfile().catch(() => null);
      const hardwareTier: HardwareTier = profile?.recommendedTier ?? defaultTier;
      const resource = await window.omni!.catalog.resourcePlan({
        mode: brain.config.workingMemoryMode,
        hardwareTier,
        ...(brain.config.workingMemoryMode === "manual"
          ? { requestedItems: String(brain.config.workingMemorySlots) }
          : {}),
        systemRamMode: brain.config.systemRamMode,
        ...(brain.config.systemRamMode === "manual"
          ? { systemRamSharePercent: brain.config.systemRamSharePercent }
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
  }, [brain.config, defaultTier]);

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
    return () => {
      unsubscribe();
      if (controllerRef.current === controller) controllerRef.current = null;
      void controller.dispose();
    };
  }, [brain.id, captureAdapter]);

  useEffect(() => {
    setBurstCount((current) => Math.min(Math.max(1, current), plan.maxSnapshotBurst));
    setIntervalMs((current) => Math.max(current, plan.minSnapshotIntervalMs));
  }, [plan.maxSnapshotBurst, plan.minSnapshotIntervalMs]);

  const run = async (task: (controller: LivePerceptionController) => Promise<unknown>) => {
    const controller = controllerRef.current;
    if (!controller) {
      setLocalError("Live Perception requires the packaged desktop runtime.");
      return;
    }
    setPending(true);
    setLocalError("");
    try {
      await task(controller);
    } catch (error) {
      setLocalError(livePerceptionErrorMessage(error));
    } finally {
      setPending(false);
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
  const visualSource = source !== "microphone";
  const snapshotBusy = state.snapshot?.state === "capturing";
  const status = state.phase === "observing"
    ? "Observing"
    : state.phase === "requesting"
      ? "Requesting permission"
      : state.phase === "stopping"
        ? "Stopping"
        : state.phase === "error"
          ? "Needs attention"
          : "Off";
  const latestControl = controlLabel(state);

  return (
    <details className="live-perception-panel">
      <summary>
        <span className="live-perception-panel__icon"><Icon name="activity" size={14} /></span>
        <span>
          <strong>Live Perception</strong>
          <small>{state.enabled ? `${titleCase(state.source ?? source)} · ${captureLabel(state)}` : "Camera, screen, or direct neural audio"}</small>
        </span>
        <em className={state.enabled ? "is-active" : ""}><i /> {status}</em>
      </summary>
      <div className="live-perception-panel__body">
        <div className="live-perception-toolbar">
          <fieldset>
            <legend>Source</legend>
            {SOURCES.map((item) => {
              const available = sourceAvailable(item.id, captureAdapter.capabilities);
              return (
                <button
                  key={item.id}
                  aria-pressed={source === item.id}
                  disabled={!available || state.enabled || state.phase === "requesting"}
                  title={available ? `Use ${item.label.toLocaleLowerCase()} perception` : `${item.label} is unavailable in this runtime`}
                  onClick={() => setSource(item.id)}
                >
                  {item.label}
                </button>
              );
            })}
          </fieldset>
          <fieldset>
            <legend>Quality</legend>
            {CAPTURE_MODES.map((mode) => (
              <button
                key={mode}
                aria-pressed={selectedMode === mode}
                disabled={pending || state.phase === "requesting" || state.phase === "stopping"}
                onClick={() => selectMode(mode)}
              >
                {titleCase(mode)}
              </button>
            ))}
          </fieldset>
          <div className="live-perception-actions">
            {!state.enabled && state.phase !== "requesting" ? (
              <button
                className="is-primary"
                disabled={pending || !sourceAvailable(source, captureAdapter.capabilities)}
                onClick={() => void run((controller) => controller.start(source, plan, "neural"))}
              >
                <Icon name="play" size={12} /> Start
              </button>
            ) : null}
            {state.enabled ? (
              <button disabled={pending} onClick={() => void run((controller) => controller.stop())}>
                <Icon name="close" size={12} /> Stop
              </button>
            ) : null}
            {state.phase === "requesting" || state.enabled || state.phase === "stopping" ? (
              <button className="is-danger" onClick={() => void run((controller) => controller.cancel())}>
                Cancel
              </button>
            ) : null}
          </div>
        </div>

        <div className="live-perception-facts" aria-live="polite">
          <span><small>Applied capture</small><strong>{captureLabel(state)}</strong></span>
          <span><small>Resource class</small><strong>{titleCase(capture?.benchmarkClass ?? plan.benchmarkClass)}</strong></span>
          <span><small>Accepted</small><strong>{state.session?.packetsAccepted.toLocaleString() ?? "0"}</strong></span>
          <span><small>Backpressure</small><strong>{((state.session?.packetsDroppedBackpressure ?? 0) + state.localPacketsDropped).toLocaleString()}</strong></span>
        </div>

        {latestControl ? (
          <p className="live-perception-control-status" role="status">
            <Icon name={state.lastControl?.state === "rejected" ? "warning" : "pulse"} size={13} />
            <span>{latestControl}</span>
          </p>
        ) : null}
        {state.error || localError ? (
          <p className="live-perception-error" role="alert"><Icon name="warning" size={13} /> {localError || state.error}</p>
        ) : null}

        <details className="live-perception-advanced">
          <summary>Snapshot bursts, source bounds, and control audit</summary>
          <div>
            <section className="live-perception-snapshot" aria-labelledby="snapshot-burst-title">
              <div>
                <strong id="snapshot-burst-title">Snapshot burst</strong>
                <small>Temporary visual samples enter this same brain only after every frame is accepted.</small>
              </div>
              <fieldset>
                <legend>Resolution</legend>
                {(["native", "current", "custom"] as const).map((resolution) => (
                  <button
                    key={resolution}
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
                  disabled={!state.enabled || !visualSource || pending || snapshotBusy}
                  onClick={requestSnapshot}
                >
                  <Icon name="image" size={12} /> Capture burst
                </button>
                {snapshotBusy ? <button onClick={cancelSnapshot}>Cancel burst</button> : null}
                {state.snapshot ? (
                  <span role="status">{titleCase(state.snapshot.state)} · {state.snapshot.acceptedFrames}/{state.snapshot.requestedFrames} accepted</span>
                ) : null}
              </div>
            </section>

            <dl className="live-perception-audit">
              <div><dt>Source native</dt><dd>{capture?.sourceNativeWidth && capture.sourceNativeHeight ? `${capture.sourceNativeWidth}×${capture.sourceNativeHeight}${capture.sourceNativeFps ? ` @ ${capture.sourceNativeFps} FPS` : ""}` : "Available after source negotiation"}</dd></div>
              <div><dt>Capture revision</dt><dd>{capture ? `r${capture.revision}` : "—"}</dd></div>
              <div><dt>Transport window</dt><dd>{state.session ? `${state.session.maxInFlight} in flight · ${state.session.maxPacketBytes.toLocaleString()} bytes/packet` : `${plan.maxInFlight} planned`}</dd></div>
              <div><dt>Auto plan</dt><dd>{plan.reason}</dd></div>
              <div><dt>Last control</dt><dd>{latestControl ?? "No human or brain capture control yet"}</dd></div>
              <div><dt>Brain actions</dt><dd>{state.lastActions.length ? state.lastActions.map((action) => action.kind).join(", ") : "None from the latest observation"}</dd></div>
            </dl>
            <p className="live-perception-privacy">
              Raw packets are not stored. Accepted perception may update working or neural state, but it does not commit dataset coverage and it never injects a hidden behavioral prompt.
            </p>
          </div>
        </details>
      </div>
    </details>
  );
}
