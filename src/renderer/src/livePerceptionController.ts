import type {
  HardwareTier,
  LiveObservationBenchmarkClass,
  LiveObservationCaptureMode,
  LiveObservationControl,
  LiveObservationEvent,
  LiveObservationModality,
  LiveObservationPacket,
  LiveObservationPacketResult,
  LiveObservationSession,
  OmniApi,
  StructuredAction
} from "../../shared/types";
import { livePerceptionErrorMessage } from "./uiPresentation";

export type LivePerceptionSource = "camera" | "microphone" | "screen" | "mixed";
export type LivePerceptionSnapshotResolution = "native" | "current" | "custom";
export type LivePerceptionNegotiatedCapture = LiveObservationSession["capture"];

export interface LivePerceptionPlan {
  mode: LiveObservationCaptureMode;
  width: number;
  height: number;
  framesPerSecond: number;
  audioChunkMs: number;
  maxInFlight: number;
  benchmarkClass: LiveObservationBenchmarkClass;
  sourceNativeWidth?: number;
  sourceNativeHeight?: number;
  sourceNativeFps?: number;
  /** Measured transport/encode budget, not a product resolution ceiling. */
  resourcePixelsPerSecond: number;
  maxSnapshotBurst: number;
  minSnapshotIntervalMs: number;
  maxTemporaryConfigureMs: number;
  realtime: true;
  reason: string;
}

export interface LivePerceptionPlanInput {
  hardwareTier: HardwareTier;
  mode?: LiveObservationCaptureMode;
  /** Measured from a prior neural turn; omitted before the first turn. */
  generationTokensPerSecond?: number;
  /** Measured memory-copy throughput. */
  memoryBytesPerSecond?: number;
  /** Measured capture/IPC throughput when available. */
  ioBytesPerSecond?: number;
  sourceNativeWidth?: number;
  sourceNativeHeight?: number;
  sourceNativeFps?: number;
}

export interface LivePerceptionConfigureRequest {
  mode?: LiveObservationCaptureMode;
  width?: number;
  height?: number;
  fps?: number;
}

export interface LivePerceptionSnapshotRequest {
  resolutionMode?: LivePerceptionSnapshotResolution;
  width?: number;
  height?: number;
  burstCount?: number;
  intervalMs?: number;
}

export interface LivePerceptionSnapshotActual {
  mode: LiveObservationCaptureMode;
  resolutionMode: LivePerceptionSnapshotResolution;
  width: number;
  height: number;
  fps?: number;
  fullResolution?: true;
  burstCount: number;
  intervalMs: number;
  framesAccepted: number;
  revision: number;
}

interface NativeCaptureBounds {
  width?: number;
  height?: number;
  fps?: number;
}

const INITIAL_CAPTURE_TARGETS: Record<
  HardwareTier,
  { width: number; height: number; fps: number; benchmark: LiveObservationBenchmarkClass }
> = {
  micro: { width: 320, height: 180, fps: 1, benchmark: "fallback" },
  personal: { width: 960, height: 540, fps: 8, benchmark: "balanced" },
  gpu: { width: 1920, height: 1080, fps: 24, benchmark: "high" },
  workstation: { width: 2560, height: 1440, fps: 30, benchmark: "native" }
};

function positiveFinite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value > 0
    ? value
    : undefined;
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isSafeInteger(value) || Number(value) < 1) {
    throw new Error(`${label} must be a positive safe integer.`);
  }
  return Number(value);
}

function evenDimension(value: number): number {
  const rounded = Math.max(2, Math.round(value));
  return rounded % 2 === 0 ? rounded : rounded - 1;
}

function benchmarkFor(input: LivePerceptionPlanInput): LiveObservationBenchmarkClass {
  let benchmark = INITIAL_CAPTURE_TARGETS[input.hardwareTier].benchmark;
  const tokens = positiveFinite(input.generationTokensPerSecond);
  const memory = positiveFinite(input.memoryBytesPerSecond);
  const io = positiveFinite(input.ioBytesPerSecond);
  if (
    input.hardwareTier === "micro" ||
    (tokens !== undefined && tokens < 1.5) ||
    (memory !== undefined && memory < 768 * 1024 ** 2) ||
    (io !== undefined && io < 1 * 1024 ** 2)
  ) {
    return "fallback";
  }
  if (
    (tokens !== undefined && tokens < 6) ||
    (memory !== undefined && memory < 3 * 1024 ** 3) ||
    (io !== undefined && io < 8 * 1024 ** 2)
  ) {
    benchmark = "balanced";
  } else if (
    benchmark === "native" &&
    ((tokens !== undefined && tokens < 20) ||
      (memory !== undefined && memory < 12 * 1024 ** 3) ||
      (io !== undefined && io < 64 * 1024 ** 2))
  ) {
    benchmark = "high";
  }
  return benchmark;
}

function modeScales(
  mode: LiveObservationCaptureMode,
  benchmark: LiveObservationBenchmarkClass
): { resolution: number; fps: number } {
  if (benchmark === "fallback") return { resolution: 1, fps: 1 };
  const resourceScale = { balanced: 0.5, high: 0.78, native: 1 }[benchmark];
  if (mode === "motion") {
    return { resolution: Math.max(0.35, resourceScale * 0.7), fps: 1 };
  }
  if (mode === "detail") {
    return { resolution: 1, fps: Math.max(0.2, resourceScale * 0.5) };
  }
  if (mode === "balanced") {
    return { resolution: Math.max(0.5, resourceScale), fps: Math.max(0.45, resourceScale) };
  }
  return { resolution: resourceScale, fps: resourceScale };
}

function constrainPixelRate(
  width: number,
  height: number,
  fps: number,
  budget: number,
  mode: LiveObservationCaptureMode
): { width: number; height: number; fps: number } {
  const requested = width * height * fps;
  if (requested <= budget) return { width, height, fps };
  if (mode === "detail") {
    const boundedFps = Math.max(1, Math.floor(budget / (width * height)));
    if (width * height * boundedFps <= budget) {
      return { width, height, fps: boundedFps };
    }
    const scale = Math.sqrt(budget / (width * height * boundedFps));
    return {
      width: evenDimension(width * scale),
      height: evenDimension(height * scale),
      fps: boundedFps
    };
  }
  if (mode === "motion") {
    const scale = Math.sqrt(budget / requested);
    return {
      width: evenDimension(width * scale),
      height: evenDimension(height * scale),
      fps
    };
  }
  const scale = Math.cbrt(budget / requested);
  return {
    width: evenDimension(width * scale),
    height: evenDimension(height * scale),
    fps: Math.max(1, Math.floor(fps * scale))
  };
}

function fitExactPixelBudget(
  capture: { width: number; height: number; fps: number },
  budget: number
): { width: number; height: number; fps: number } {
  let { width, height, fps } = capture;
  while (width * height * fps > budget && (width > 2 || height > 2)) {
    if (width >= height && width > 2) width -= 2;
    else if (height > 2) height -= 2;
  }
  return { width, height, fps };
}

export function planLivePerception(input: LivePerceptionPlanInput): LivePerceptionPlan {
  const mode = input.mode ?? "auto";
  const benchmarkClass = benchmarkFor(input);
  const initial = INITIAL_CAPTURE_TARGETS[input.hardwareTier];
  if (benchmarkClass === "fallback") {
    return {
      mode,
      width: 320,
      height: 180,
      framesPerSecond: 1,
      audioChunkMs: 1_500,
      maxInFlight: 1,
      benchmarkClass,
      ...(input.sourceNativeWidth ? { sourceNativeWidth: input.sourceNativeWidth } : {}),
      ...(input.sourceNativeHeight ? { sourceNativeHeight: input.sourceNativeHeight } : {}),
      ...(input.sourceNativeFps ? { sourceNativeFps: input.sourceNativeFps } : {}),
      resourcePixelsPerSecond: 320 * 180,
      // Even the continuity fallback keeps a short anti-blur burst; its
      // interval, not its evidence count, absorbs the resource pressure.
      maxSnapshotBurst: 3,
      minSnapshotIntervalMs: 1_000,
      maxTemporaryConfigureMs: 15_000,
      realtime: true,
      reason: "Measured resources selected the 320x180 at 1 FPS continuity fallback."
    };
  }

  const nativeWidth = positiveFinite(input.sourceNativeWidth) ?? initial.width;
  const nativeHeight = positiveFinite(input.sourceNativeHeight) ?? initial.height;
  const nativeFps = positiveFinite(input.sourceNativeFps) ?? initial.fps;
  const scales = modeScales(mode, benchmarkClass);
  let width = evenDimension(nativeWidth * scales.resolution);
  let height = evenDimension(nativeHeight * scales.resolution);
  let fps = Math.max(1, Math.round(nativeFps * scales.fps));

  const nativePixelRate = nativeWidth * nativeHeight * nativeFps;
  const classFraction = { balanced: 0.25, high: 0.65, native: 1 }[benchmarkClass];
  const measuredBudgets = [nativePixelRate * classFraction];
  const memory = positiveFinite(input.memoryBytesPerSecond);
  const io = positiveFinite(input.ioBytesPerSecond);
  const tokens = positiveFinite(input.generationTokensPerSecond);
  if (memory !== undefined) measuredBudgets.push(memory / 32);
  if (io !== undefined) measuredBudgets.push(io / 0.22);
  if (tokens !== undefined) {
    measuredBudgets.push(nativePixelRate * Math.max(0.08, Math.min(1, tokens / 30)));
  }
  const resourcePixelsPerSecond = Math.max(
    320 * 180,
    Math.floor(Math.min(...measuredBudgets))
  );
  ({ width, height, fps } = fitExactPixelBudget(
    constrainPixelRate(width, height, fps, resourcePixelsPerSecond, mode),
    resourcePixelsPerSecond
  ));
  width = Math.min(width, Math.floor(nativeWidth));
  height = Math.min(height, Math.floor(nativeHeight));
  fps = Math.min(fps, Math.floor(nativeFps));

  const nativeFramePixels = Math.max(1, nativeWidth * nativeHeight);
  const boundedNativeFps = Math.max(
    1,
    Math.min(nativeFps, resourcePixelsPerSecond / nativeFramePixels)
  );
  const maxSnapshotBurst = Math.max(1, Math.floor(boundedNativeFps * 5));
  const minSnapshotIntervalMs = Math.max(1, Math.ceil(1_000 / boundedNativeFps));
  const classTransport = { balanced: 2, high: 4, native: 8 }[benchmarkClass];
  return {
    mode,
    width,
    height,
    framesPerSecond: fps,
    audioChunkMs: benchmarkClass === "balanced" ? 1_000 : benchmarkClass === "high" ? 750 : 500,
    maxInFlight: classTransport,
    benchmarkClass,
    ...(input.sourceNativeWidth ? { sourceNativeWidth: input.sourceNativeWidth } : {}),
    ...(input.sourceNativeHeight ? { sourceNativeHeight: input.sourceNativeHeight } : {}),
    ...(input.sourceNativeFps ? { sourceNativeFps: input.sourceNativeFps } : {}),
    resourcePixelsPerSecond,
    maxSnapshotBurst,
    minSnapshotIntervalMs,
    maxTemporaryConfigureMs:
      benchmarkClass === "balanced" ? 30_000 : benchmarkClass === "high" ? 60_000 : 120_000,
    realtime: true,
    reason:
      input.sourceNativeWidth && input.sourceNativeHeight
        ? `Scaled ${mode} capture from measured resources and the source-native signal.`
        : `Opened with a ${input.hardwareTier} ideal; source-native capabilities will replace it after negotiation.`
  };
}

export interface LivePerceptionCapturePacket {
  modality: LiveObservationModality;
  timestampMs: number;
  mimeType: string;
  data: Uint8Array;
  settings?: LiveObservationPacket["settings"];
}

export interface LivePerceptionSnapshotCaptureRequest {
  mode: LiveObservationCaptureMode;
  resolutionMode: LivePerceptionSnapshotResolution;
  width: number;
  height: number;
  fps?: number;
  burstCount: number;
  intervalMs: number;
  maxPacketBytes: number;
}

export interface LivePerceptionCaptureHandle {
  getCapture(): Readonly<LivePerceptionNegotiatedCapture>;
  setPacketByteLimit(bytes: number): void;
  start(listener: (packet: LivePerceptionCapturePacket) => void): void;
  reconfigure(
    request: Required<Pick<LivePerceptionConfigureRequest, "mode" | "width" | "height" | "fps">>
  ): Promise<LivePerceptionNegotiatedCapture>;
  snapshotBurst(
    request: LivePerceptionSnapshotCaptureRequest,
    listener: (packet: LivePerceptionCapturePacket) => Promise<void>,
    signal: AbortSignal
  ): Promise<LivePerceptionSnapshotActual>;
  stop(): Promise<void> | void;
}

export interface LivePerceptionCaptureAdapter {
  capabilities: {
    camera: boolean;
    microphone: boolean;
    screen: boolean;
  };
  open(
    source: LivePerceptionSource,
    plan: LivePerceptionPlan
  ): Promise<LivePerceptionCaptureHandle>;
}

export interface LivePerceptionSnapshotState {
  state: "capturing" | "complete" | "cancelled" | "failed";
  resolutionMode: LivePerceptionSnapshotResolution;
  requestedFrames: number;
  acceptedFrames: number;
  controlId?: string;
  error?: string;
}

export interface LivePerceptionState {
  enabled: boolean;
  phase: "idle" | "requesting" | "observing" | "stopping" | "error";
  source?: LivePerceptionSource;
  plan?: LivePerceptionPlan;
  negotiatedCapture?: LivePerceptionNegotiatedCapture;
  session?: LiveObservationSession;
  localPacketsDropped: number;
  lastPacket?: LiveObservationPacketResult;
  lastActions: StructuredAction[];
  activeControl?: LiveObservationControl;
  lastControl?: LiveObservationControl;
  snapshot?: LivePerceptionSnapshotState;
  lastSnapshot?: LivePerceptionSnapshotActual;
  error?: string;
}

export interface LivePerceptionControllerOptions {
  brainId: string;
  modality: Pick<
    OmniApi["modality"],
    | "startObservation"
    | "pushObservation"
    | "stopObservation"
    | "cancelObservation"
    | "requestObservationControl"
    | "resolveObservationControl"
    | "onObservation"
  >;
  capture: LivePerceptionCaptureAdapter;
  now?: () => Date;
  setTimeout?: typeof globalThis.setTimeout;
  clearTimeout?: typeof globalThis.clearTimeout;
  sleep?: (milliseconds: number, signal: AbortSignal) => Promise<void>;
  onActions?: (actions: StructuredAction[]) => void;
}

type StateListener = (state: Readonly<LivePerceptionState>) => void;

function modalitiesFor(source: LivePerceptionSource): LiveObservationModality[] {
  if (source === "microphone") return ["audio"];
  if (source === "mixed") return ["video", "audio"];
  return ["video"];
}

function captureStartShape(
  capture: LivePerceptionNegotiatedCapture
): Omit<LivePerceptionNegotiatedCapture, "revision"> {
  const { revision: _revision, ...start } = capture;
  return start;
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && /aborted|cancelled/i.test(error.message);
}

function abortError(): Error {
  const error = new Error("Live Perception capture was cancelled.");
  error.name = "AbortError";
  return error;
}

function snapshotBounds(
  plan: LivePerceptionPlan,
  capture: LivePerceptionNegotiatedCapture,
  request: LivePerceptionSnapshotRequest
): LivePerceptionSnapshotCaptureRequest {
  const resolutionMode = request.resolutionMode ?? "native";
  const nativeWidth = capture.sourceNativeWidth ?? capture.width;
  const nativeHeight = capture.sourceNativeHeight ?? capture.height;
  const currentWidth = capture.width ?? nativeWidth;
  const currentHeight = capture.height ?? nativeHeight;
  if (!nativeWidth || !nativeHeight || !currentWidth || !currentHeight) {
    throw new Error("The active source has no negotiated visual resolution.");
  }
  if (
    resolutionMode === "custom" &&
    (!positiveFinite(request.width) || !positiveFinite(request.height))
  ) {
    throw new Error("Custom snapshot resolution requires width and height.");
  }
  let width = resolutionMode === "native"
    ? nativeWidth
    : resolutionMode === "current"
      ? currentWidth
      : Math.min(positiveInteger(request.width, "Snapshot width"), nativeWidth);
  let height = resolutionMode === "native"
    ? nativeHeight
    : resolutionMode === "current"
      ? currentHeight
      : Math.min(positiveInteger(request.height, "Snapshot height"), nativeHeight);
  width = evenDimension(width);
  height = evenDimension(height);
  const requestedInterval = request.intervalMs === undefined
    ? plan.minSnapshotIntervalMs
    : positiveInteger(request.intervalMs, "Snapshot interval");
  let intervalMs = Math.max(plan.minSnapshotIntervalMs, requestedInterval);
  if (resolutionMode === "native") {
    intervalMs = Math.max(
      intervalMs,
      Math.ceil((width * height * 1_000) / plan.resourcePixelsPerSecond)
    );
  } else {
    const resourceScale = Math.min(
      1,
      Math.sqrt((plan.resourcePixelsPerSecond * intervalMs) / (width * height * 1_000))
    );
    if (resourceScale < 1 && resolutionMode === "custom") {
      width = evenDimension(width * resourceScale);
      height = evenDimension(height * resourceScale);
    } else if (resourceScale < 1) {
      intervalMs = Math.ceil((width * height * 1_000) / plan.resourcePixelsPerSecond);
    }
  }
  const burstCount = Math.min(
    request.burstCount === undefined
      ? Math.min(3, plan.maxSnapshotBurst)
      : positiveInteger(request.burstCount, "Snapshot burst count"),
    plan.maxSnapshotBurst
  );
  return {
    mode: capture.mode,
    resolutionMode,
    width,
    height,
    fps: capture.fps,
    burstCount,
    intervalMs,
    maxPacketBytes: Number.MAX_SAFE_INTEGER
  };
}

function configurationBounds(
  plan: LivePerceptionPlan,
  capture: LivePerceptionNegotiatedCapture,
  request: LivePerceptionConfigureRequest
): Required<Pick<LivePerceptionConfigureRequest, "mode" | "width" | "height" | "fps">> {
  const nativeWidth = capture.sourceNativeWidth ?? capture.width ?? plan.width;
  const nativeHeight = capture.sourceNativeHeight ?? capture.height ?? plan.height;
  const nativeFps = capture.sourceNativeFps ?? capture.fps ?? plan.framesPerSecond;
  const mode = request.mode ?? capture.mode;
  let width = Math.min(
    request.width === undefined ? capture.width ?? plan.width : positiveInteger(request.width, "Capture width"),
    nativeWidth
  );
  let height = Math.min(
    request.height === undefined ? capture.height ?? plan.height : positiveInteger(request.height, "Capture height"),
    nativeHeight
  );
  let fps = Math.min(
    request.fps === undefined ? capture.fps ?? plan.framesPerSecond : positiveInteger(request.fps, "Capture FPS"),
    nativeFps
  );
  ({ width, height, fps } = fitExactPixelBudget(
    constrainPixelRate(
      evenDimension(width),
      evenDimension(height),
      Math.max(1, Math.floor(fps)),
      plan.resourcePixelsPerSecond,
      mode
    ),
    plan.resourcePixelsPerSecond
  ));
  return { mode, width, height, fps };
}

/**
 * Owns one explicit capture session. It never creates prompt text: accepted
 * packets enter the same brain's sensory assemblies through the typed channel.
 */
export class LivePerceptionController {
  private state: LivePerceptionState = {
    enabled: false,
    phase: "idle",
    localPacketsDropped: 0,
    lastActions: []
  };
  private readonly listeners = new Set<StateListener>();
  private captureHandle?: LivePerceptionCaptureHandle;
  private lifecycle = 0;
  private sequence = 0;
  private localInFlight = 0;
  private disposed = false;
  private snapshotAbort?: AbortController;
  private controlQueue: Promise<void> = Promise.resolve();
  private readonly processedControls = new Set<string>();
  private readonly revertTimers = new Map<string, ReturnType<typeof globalThis.setTimeout>>();
  private readonly removeEvents: () => void;

  constructor(private readonly options: LivePerceptionControllerOptions) {
    this.removeEvents = options.modality.onObservation((event) => this.handleEvent(event));
  }

  getState(): Readonly<LivePerceptionState> {
    return this.state;
  }

  subscribe(listener: StateListener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  async start(
    source: LivePerceptionSource,
    plan: LivePerceptionPlan,
    retention: "working" | "neural" = "working"
  ): Promise<void> {
    if (this.disposed) throw new Error("Live Perception has been disposed.");
    if (this.state.enabled) return;
    if (
      (source === "camera" || source === "mixed") &&
      !this.options.capture.capabilities.camera
    ) {
      this.fail("Camera capture is unavailable in this runtime.");
      return;
    }
    if (
      (source === "microphone" || source === "mixed") &&
      !this.options.capture.capabilities.microphone
    ) {
      this.fail("Microphone capture is unavailable in this runtime.");
      return;
    }
    if (source === "screen" && !this.options.capture.capabilities.screen) {
      this.fail("Screen capture is unavailable in this runtime.");
      return;
    }
    const lifecycle = ++this.lifecycle;
    this.patch({
      enabled: false,
      phase: "requesting",
      source,
      plan,
      negotiatedCapture: undefined,
      session: undefined,
      localPacketsDropped: 0,
      lastPacket: undefined,
      lastActions: [],
      activeControl: undefined,
      snapshot: undefined,
      error: undefined
    });
    let capture: LivePerceptionCaptureHandle | undefined;
    let openedSession: LiveObservationSession | undefined;
    try {
      capture = await this.options.capture.open(source, plan);
      if (lifecycle !== this.lifecycle) {
        await capture.stop();
        return;
      }
      const negotiated = capture.getCapture();
      const session = await this.options.modality.startObservation({
        brainId: this.options.brainId,
        modalities: modalitiesFor(source),
        permission: {
          source,
          granted: true,
          scope: "session",
          grantedAt: (this.options.now?.() ?? new Date()).toISOString()
        },
        retention,
        capture: captureStartShape(negotiated)
      });
      openedSession = session;
      if (lifecycle !== this.lifecycle) {
        await capture.stop();
        await this.options.modality.cancelObservation(session.id);
        return;
      }
      capture.setPacketByteLimit(session.maxPacketBytes);
      this.captureHandle = capture;
      this.sequence = 0;
      this.localInFlight = 0;
      this.patch({
        enabled: true,
        phase: "observing",
        plan: session.maxInFlight === plan.maxInFlight
          ? plan
          : {
            ...plan,
            maxInFlight: session.maxInFlight,
            reason: `${plan.reason} Transport concurrency adapted to the current device envelope.`
          },
        session,
        negotiatedCapture: session.capture
      });
      capture.start((packet) => this.pushPacket(lifecycle, packet));
    } catch (error) {
      await capture?.stop();
      if (openedSession) {
        await this.options.modality.cancelObservation(openedSession.id).catch(() => undefined);
      }
      if (lifecycle !== this.lifecycle) return;
      this.fail(livePerceptionErrorMessage(error));
    }
  }

  /** Audited human request; the emitted typed event is applied by the same path as brain controls. */
  async requestConfigure(
    request: LivePerceptionConfigureRequest & { durationMs?: number }
  ): Promise<LiveObservationControl> {
    const session = this.state.session;
    if (!this.state.enabled || !session) throw new Error("Live Perception is not observing.");
    return this.options.modality.requestObservationControl({
      sessionId: session.id,
      kind: "configure",
      requested: {
        ...request,
        durationMs: request.durationMs ?? 5_000
      }
    });
  }

  /** Audited human request; no local untraced burst is fabricated. */
  async requestSnapshot(
    request: LivePerceptionSnapshotRequest = {}
  ): Promise<LiveObservationControl> {
    const session = this.state.session;
    if (!this.state.enabled || !session) throw new Error("Live Perception is not observing.");
    return this.options.modality.requestObservationControl({
      sessionId: session.id,
      kind: "snapshot",
      requested: {
        resolutionMode: request.resolutionMode ?? "native",
        ...(request.resolutionMode === "native" || request.resolutionMode === undefined
          ? { fullResolution: true as const }
          : {}),
        ...(request.width !== undefined ? { width: request.width } : {}),
        ...(request.height !== undefined ? { height: request.height } : {}),
        ...(request.burstCount !== undefined ? { burstCount: request.burstCount } : {}),
        ...(request.intervalMs !== undefined ? { intervalMs: request.intervalMs } : {})
      }
    });
  }

  private async applyReconfigure(
    request: LivePerceptionConfigureRequest
  ): Promise<LivePerceptionNegotiatedCapture> {
    const handle = this.captureHandle;
    const plan = this.state.plan;
    const capture = this.state.negotiatedCapture;
    if (!this.state.enabled || !handle || !plan || !capture) {
      throw new Error("Live Perception is not observing.");
    }
    const bounded = configurationBounds(plan, capture, request);
    const actual = await handle.reconfigure(bounded);
    this.patch({ negotiatedCapture: actual, plan: { ...plan, mode: actual.mode } });
    return actual;
  }

  private async performSnapshotBurst(
    request: LivePerceptionSnapshotRequest = {},
    signal?: AbortSignal
  ): Promise<LivePerceptionSnapshotActual> {
    const handle = this.captureHandle;
    const session = this.state.session;
    const plan = this.state.plan;
    const capture = this.state.negotiatedCapture;
    if (!this.state.enabled || !handle || !session || !plan || !capture) {
      throw new Error("Live Perception is not observing.");
    }
    if (!session.modalities.includes("video")) {
      throw new Error("The active source does not provide visual frames.");
    }
    if (this.snapshotAbort) throw new Error("A snapshot burst is already active.");
    const bounded = snapshotBounds(plan, capture, request);
    bounded.maxPacketBytes = session.maxPacketBytes;
    const controlId = this.state.activeControl?.id;
    if (!controlId) {
      throw new Error("Snapshot capture requires an audited observation control.");
    }
    const controller = new AbortController();
    this.snapshotAbort = controller;
    const abort = (): void => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    if (signal?.aborted) controller.abort();
    this.patch({
      snapshot: {
        state: "capturing",
        resolutionMode: bounded.resolutionMode,
        requestedFrames: bounded.burstCount,
        acceptedFrames: 0,
        controlId: this.state.activeControl?.id
      }
    });
    try {
      let burstIndex = 0;
      const actual = await handle.snapshotBurst(
        bounded,
        async (packet) => {
          await this.pushSnapshotPacket(
            this.lifecycle,
            packet,
            controller.signal,
            {
              observationControlId: controlId,
              resolutionMode: bounded.resolutionMode,
              burstIndex,
              burstCount: bounded.burstCount
            }
          );
          burstIndex += 1;
          const current = this.state.snapshot;
          if (current?.state === "capturing") {
            this.patch({
              snapshot: { ...current, acceptedFrames: current.acceptedFrames + 1 }
            });
          }
        },
        controller.signal
      );
      this.patch({
        negotiatedCapture: handle.getCapture(),
        snapshot: {
          state: "complete",
          resolutionMode: actual.resolutionMode,
          requestedFrames: actual.burstCount,
          acceptedFrames: actual.framesAccepted,
          controlId: this.state.activeControl?.id
        },
        lastSnapshot: actual
      });
      return actual;
    } catch (error) {
      const cancelled = controller.signal.aborted || isAbort(error);
      this.patch({
        negotiatedCapture: handle.getCapture(),
        snapshot: {
          state: cancelled ? "cancelled" : "failed",
          resolutionMode: bounded.resolutionMode,
          requestedFrames: bounded.burstCount,
          acceptedFrames: this.state.snapshot?.acceptedFrames ?? 0,
          controlId: this.state.activeControl?.id,
          error: error instanceof Error ? error.message : "Snapshot burst failed."
        }
      });
      throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
      if (this.snapshotAbort === controller) this.snapshotAbort = undefined;
    }
  }

  /** Cancels only the active burst; continuous observation remains running. */
  cancelSnapshot(): boolean {
    if (!this.snapshotAbort) return false;
    this.snapshotAbort.abort();
    return true;
  }

  async stop(): Promise<void> {
    const session = this.state.session;
    if (!session && !this.captureHandle) {
      this.patch({ enabled: false, phase: "idle" });
      return;
    }
    ++this.lifecycle;
    this.cancelLocalActivity();
    this.patch({ enabled: false, phase: "stopping" });
    await this.controlQueue.catch(() => undefined);
    const capture = this.captureHandle;
    this.captureHandle = undefined;
    await capture?.stop();
    try {
      const stopped = session
        ? await this.options.modality.stopObservation(session.id)
        : undefined;
      this.patch({
        enabled: false,
        phase: "idle",
        session: stopped,
        negotiatedCapture: stopped?.capture,
        source: undefined,
        activeControl: undefined
      });
    } catch (error) {
      this.fail(error instanceof Error ? error.message : "Live Perception could not stop.");
    }
  }

  async cancel(): Promise<void> {
    const session = this.state.session;
    ++this.lifecycle;
    this.cancelLocalActivity();
    await this.controlQueue.catch(() => undefined);
    const capture = this.captureHandle;
    this.captureHandle = undefined;
    await capture?.stop();
    try {
      const cancelled = session
        ? await this.options.modality.cancelObservation(session.id)
        : undefined;
      this.patch({
        enabled: false,
        phase: "idle",
        session: cancelled,
        negotiatedCapture: cancelled?.capture,
        source: undefined,
        activeControl: undefined
      });
    } catch (error) {
      this.fail(error instanceof Error ? error.message : "Live Perception cancellation failed.");
    }
  }

  async dispose(): Promise<void> {
    if (this.disposed) return;
    await this.cancel();
    this.disposed = true;
    this.removeEvents();
    this.listeners.clear();
  }

  private async pushSnapshotPacket(
    lifecycle: number,
    captured: LivePerceptionCapturePacket,
    signal: AbortSignal,
    audit: {
      observationControlId: string;
      resolutionMode: LivePerceptionSnapshotResolution;
      burstIndex: number;
      burstCount: number;
    }
  ): Promise<void> {
    while (true) {
      if (signal.aborted || lifecycle !== this.lifecycle) throw abortError();
      while (this.localInFlight > 0) {
        await this.delay(5, signal);
      }
      const session = this.state.session;
      if (!session || session.state !== "active") throw abortError();
      const result = await this.options.modality.pushObservation({
        sessionId: session.id,
        modality: captured.modality,
        sequence: ++this.sequence,
        timestampMs: captured.timestampMs,
        mimeType: captured.mimeType,
        data: captured.data,
        settings: { ...captured.settings, ...audit }
      });
      if (lifecycle !== this.lifecycle || signal.aborted) throw abortError();
      this.applyPacketResult(result);
      if (result.accepted) return;
      if (result.reason === "session-stopped") throw abortError();
      await this.delay(10, signal);
    }
  }

  private pushPacket(lifecycle: number, captured: LivePerceptionCapturePacket): void {
    const session = this.state.session;
    if (
      lifecycle !== this.lifecycle ||
      !this.state.enabled ||
      !session ||
      !session.modalities.includes(captured.modality)
    ) {
      return;
    }
    if (this.localInFlight >= session.maxInFlight) {
      this.patch({ localPacketsDropped: this.state.localPacketsDropped + 1 });
      return;
    }
    const packet: LiveObservationPacket = {
      sessionId: session.id,
      modality: captured.modality,
      sequence: ++this.sequence,
      timestampMs: captured.timestampMs,
      mimeType: captured.mimeType,
      data: captured.data,
      settings: captured.settings
    };
    this.localInFlight += 1;
    void this.options.modality
      .pushObservation(packet)
      .then((result) => {
        if (lifecycle !== this.lifecycle) return;
        this.applyPacketResult(result);
      })
      .catch((error: unknown) => {
        if (lifecycle !== this.lifecycle) return;
        this.patch({ error: error instanceof Error ? error.message : "A live packet was rejected." });
      })
      .finally(() => {
        this.localInFlight = Math.max(0, this.localInFlight - 1);
      });
  }

  private applyPacketResult(result: LiveObservationPacketResult): void {
    const actions = result.actions ?? [];
    this.patch({
      session: result.session,
      negotiatedCapture: result.session.capture,
      lastPacket: result,
      lastActions: actions.length > 0 ? actions : this.state.lastActions
    });
    if (actions.length > 0) this.options.onActions?.(actions);
  }

  private handleEvent(event: LiveObservationEvent): void {
    if (event.session.brainId !== this.options.brainId) return;
    if (this.state.session && event.session.id !== this.state.session.id) return;
    const actions = event.actions ?? event.packet?.actions ?? [];
    this.patch({
      session: event.session,
      negotiatedCapture: event.session.capture,
      lastPacket: event.packet ?? this.state.lastPacket,
      lastActions: actions.length > 0 ? actions : this.state.lastActions,
      ...(event.control ? { lastControl: event.control } : {})
    });
    if (actions.length > 0) this.options.onActions?.(actions);
    if (
      event.type === "control" &&
      event.control?.state === "requested" &&
      !this.processedControls.has(event.control.id)
    ) {
      this.processedControls.add(event.control.id);
      this.controlQueue = this.controlQueue
        .then(() => this.processControl(event.control!))
        .catch((error: unknown) => {
          this.patch({ error: error instanceof Error ? error.message : "Capture control failed." });
        });
    }
  }

  private async processControl(control: LiveObservationControl): Promise<void> {
    if (!this.state.enabled || !this.captureHandle || control.sessionId !== this.state.session?.id) {
      await this.resolveControl(control, "rejected", undefined, "The capture session is not active.");
      return;
    }
    this.patch({ activeControl: control });
    if (control.kind === "snapshot") {
      try {
        const actual = await this.performSnapshotBurst({
          resolutionMode: control.requested.resolutionMode ?? "native",
          width: control.requested.width,
          height: control.requested.height,
          burstCount: control.requested.burstCount,
          intervalMs: control.requested.intervalMs
        });
        await this.resolveControl(control, "applied", {
          mode: actual.mode,
          width: actual.width,
          height: actual.height,
          fps: actual.fps,
          ...(actual.resolutionMode === "native" ? { fullResolution: true as const } : {}),
          resolutionMode: actual.resolutionMode,
          burstCount: actual.burstCount,
          intervalMs: actual.intervalMs,
          revision: (this.state.session?.capture.revision ?? actual.revision) + 1
        });
      } catch (error) {
        await this.resolveControl(
          control,
          isAbort(error) ? "cancelled" : "rejected",
          undefined,
          error instanceof Error ? error.message : "Snapshot capture failed."
        );
      } finally {
        this.patch({ activeControl: undefined });
      }
      return;
    }

    const previous = { ...this.state.negotiatedCapture! };
    const durationMs = Math.min(
      control.requested.durationMs ?? 5_000,
      this.state.plan!.maxTemporaryConfigureMs
    );
    try {
      const actual = await this.applyReconfigure({
        mode: control.requested.mode,
        width: control.requested.width,
        height: control.requested.height,
        fps: control.requested.fps
      });
      await this.resolveControl(control, "applied", {
        mode: actual.mode,
        width: actual.width,
        height: actual.height,
        fps: actual.fps,
        durationMs,
        revertToRevision: previous.revision,
        revision: (this.state.session?.capture.revision ?? actual.revision) + 1
      });
      const schedule = this.options.setTimeout ?? globalThis.setTimeout.bind(globalThis);
      const timer = schedule(() => {
        this.revertTimers.delete(control.id);
        void this.revertControl(control, previous);
      }, durationMs);
      this.revertTimers.set(control.id, timer);
    } catch (error) {
      await this.resolveControl(
        control,
        "rejected",
        undefined,
        error instanceof Error ? error.message : "Capture reconfiguration failed."
      );
    } finally {
      this.patch({ activeControl: undefined });
    }
  }

  private async revertControl(
    control: LiveObservationControl,
    previous: LivePerceptionNegotiatedCapture
  ): Promise<void> {
    if (!this.state.enabled || control.sessionId !== this.state.session?.id) return;
    try {
      const handle = this.captureHandle;
      if (!handle || !previous.width || !previous.height || !previous.fps) {
        throw new Error("The previous negotiated capture is unavailable.");
      }
      // This is an already-negotiated rollback point, not a new request, so
      // restore it exactly instead of applying today's planner a second time.
      const restored = await handle.reconfigure({
        mode: previous.mode,
        width: previous.width,
        height: previous.height,
        fps: previous.fps
      });
      this.patch({ negotiatedCapture: restored });
      await this.resolveControl(control, "reverted", {
        mode: restored.mode,
        width: restored.width,
        height: restored.height,
        fps: restored.fps,
        revision: (this.state.session?.capture.revision ?? restored.revision) + 1
      });
    } catch (error) {
      this.patch({ error: error instanceof Error ? error.message : "Capture settings could not revert." });
    }
  }

  private async resolveControl(
    control: LiveObservationControl,
    state: "applied" | "rejected" | "cancelled" | "reverted",
    actual?: LiveObservationControl["actual"],
    reason?: string
  ): Promise<void> {
    const resolved = await this.options.modality.resolveObservationControl({
      sessionId: control.sessionId,
      controlId: control.id,
      state,
      ...(actual ? { actual } : {}),
      ...(reason ? { reason: reason.slice(0, 500) } : {})
    });
    this.patch({ lastControl: resolved });
  }

  private delay(milliseconds: number, signal: AbortSignal): Promise<void> {
    if (signal.aborted) return Promise.reject(abortError());
    if (this.options.sleep) return this.options.sleep(milliseconds, signal);
    const schedule = this.options.setTimeout ?? globalThis.setTimeout.bind(globalThis);
    const clear = this.options.clearTimeout ?? globalThis.clearTimeout.bind(globalThis);
    return new Promise((resolve, reject) => {
      const timer = schedule(done, Math.max(1, milliseconds));
      const abort = (): void => {
        clear(timer);
        signal.removeEventListener("abort", abort);
        reject(abortError());
      };
      function done(): void {
        signal.removeEventListener("abort", abort);
        resolve();
      }
      signal.addEventListener("abort", abort, { once: true });
    });
  }

  private cancelLocalActivity(): void {
    this.snapshotAbort?.abort();
    this.snapshotAbort = undefined;
    const clear = this.options.clearTimeout ?? globalThis.clearTimeout.bind(globalThis);
    for (const timer of this.revertTimers.values()) clear(timer);
    this.revertTimers.clear();
  }

  private fail(message: string): void {
    this.patch({ enabled: false, phase: "error", error: message });
  }

  private patch(patch: Partial<LivePerceptionState>): void {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((listener) => listener(this.state));
  }
}

export interface BrowserLivePerceptionEnvironment {
  mediaDevices?: Pick<MediaDevices, "getUserMedia" | "getDisplayMedia">;
  mediaRecorderConstructor?: typeof MediaRecorder;
  createMediaStream?: (tracks: MediaStreamTrack[]) => MediaStream;
  createVideo?: () => HTMLVideoElement;
  createCanvas?: () => HTMLCanvasElement;
  setInterval?: typeof globalThis.setInterval;
  clearInterval?: typeof globalThis.clearInterval;
  setTimeout?: typeof globalThis.setTimeout;
  clearTimeout?: typeof globalThis.clearTimeout;
  now?: () => number;
}

function browserEnvironment(): BrowserLivePerceptionEnvironment {
  return {
    mediaDevices: globalThis.navigator?.mediaDevices,
    mediaRecorderConstructor: globalThis.MediaRecorder,
    createMediaStream: (tracks) => new MediaStream(tracks),
    createVideo: () => document.createElement("video"),
    createCanvas: () => document.createElement("canvas"),
    setInterval: globalThis.setInterval.bind(globalThis),
    clearInterval: globalThis.clearInterval.bind(globalThis),
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
    now: () => performance.now()
  };
}

function capabilityMaximum(value: number | MediaSettingsRange | undefined): number | undefined {
  if (typeof value === "number") return positiveFinite(value);
  return value && "max" in value ? positiveFinite(value.max) : undefined;
}

function nativeBounds(track?: MediaStreamTrack): NativeCaptureBounds {
  if (!track) return {};
  const settings = track.getSettings?.() ?? {};
  const capabilities = track.getCapabilities?.() ?? {};
  return {
    width: capabilityMaximum(capabilities.width) ?? positiveFinite(settings.width),
    height: capabilityMaximum(capabilities.height) ?? positiveFinite(settings.height),
    fps: capabilityMaximum(capabilities.frameRate) ?? positiveFinite(settings.frameRate)
  };
}

function planForNative(plan: LivePerceptionPlan, bounds: NativeCaptureBounds): LivePerceptionPlan {
  if (!bounds.width || !bounds.height) return plan;
  return planLivePerception({
    hardwareTier:
      plan.benchmarkClass === "fallback"
        ? "micro"
        : plan.benchmarkClass === "balanced"
          ? "personal"
          : plan.benchmarkClass === "high"
            ? "gpu"
            : "workstation",
    mode: plan.mode,
    sourceNativeWidth: bounds.width,
    sourceNativeHeight: bounds.height,
    sourceNativeFps: bounds.fps ?? plan.framesPerSecond,
    memoryBytesPerSecond: plan.resourcePixelsPerSecond * 32
  });
}

function canvasBlob(
  canvas: HTMLCanvasElement,
  maxBytes: number
): Promise<Blob> {
  const qualities = [0.82, 0.68, 0.52, 0.38, 0.26];
  return new Promise((resolve, reject) => {
    const encode = (index: number): void => {
      canvas.toBlob(
        (blob) => {
          if (!blob) {
            reject(new Error("Camera frame encoding failed."));
            return;
          }
          if (blob.size <= maxBytes || index === qualities.length - 1) {
            if (blob.size > maxBytes) {
              reject(new Error("The source-native frame exceeds the negotiated packet budget."));
            } else {
              resolve(blob);
            }
            return;
          }
          encode(index + 1);
        },
        "image/jpeg",
        qualities[index]
      );
    };
    encode(0);
  });
}

function abortableBrowserDelay(
  milliseconds: number,
  signal: AbortSignal,
  schedule: typeof globalThis.setTimeout,
  clear: typeof globalThis.clearTimeout
): Promise<void> {
  if (signal.aborted) return Promise.reject(abortError());
  return new Promise((resolve, reject) => {
    const timer = schedule(done, Math.max(1, milliseconds));
    const abort = (): void => {
      clear(timer);
      signal.removeEventListener("abort", abort);
      reject(abortError());
    };
    function done(): void {
      signal.removeEventListener("abort", abort);
      resolve();
    }
    signal.addEventListener("abort", abort, { once: true });
  });
}

/** Browser capture stays renderer-only; raw device identifiers never cross IPC. */
export function createBrowserLivePerceptionCapture(
  environment: BrowserLivePerceptionEnvironment = browserEnvironment()
): LivePerceptionCaptureAdapter {
  const devices = environment.mediaDevices;
  const Recorder = environment.mediaRecorderConstructor;
  return {
    capabilities: {
      camera: typeof devices?.getUserMedia === "function",
      microphone: typeof devices?.getUserMedia === "function" && Boolean(Recorder),
      screen: typeof devices?.getDisplayMedia === "function"
    },
    async open(source, requestedPlan) {
      if (!devices) throw new Error("Media capture is unavailable in this runtime.");
      const stream = source === "screen"
        ? await devices.getDisplayMedia({
          video: { frameRate: { ideal: requestedPlan.framesPerSecond } },
          audio: false
        })
        : await devices.getUserMedia({
          video:
            source === "camera" || source === "mixed"
              ? {
                width: { ideal: requestedPlan.width },
                height: { ideal: requestedPlan.height },
                frameRate: { ideal: requestedPlan.framesPerSecond }
              }
              : false,
          audio: source === "microphone" || source === "mixed"
        });
      const videoTracks = stream.getVideoTracks();
      const audioTracks = stream.getAudioTracks();
      const videoTrack = videoTracks[0];
      const bounds = nativeBounds(videoTrack);
      const plan = planForNative(requestedPlan, bounds);
      const createVideo = environment.createVideo ?? (() => document.createElement("video"));
      const createCanvas = environment.createCanvas ?? (() => document.createElement("canvas"));
      const video = videoTrack ? createVideo() : undefined;
      const canvas = video ? createCanvas() : undefined;
      if (video && canvas) {
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
      }
      const interval = environment.setInterval ?? globalThis.setInterval.bind(globalThis);
      const clearIntervalHandle =
        environment.clearInterval ?? globalThis.clearInterval.bind(globalThis);
      const timeout = environment.setTimeout ?? globalThis.setTimeout.bind(globalThis);
      const clearTimeoutHandle =
        environment.clearTimeout ?? globalThis.clearTimeout.bind(globalThis);
      const now = environment.now ?? (() => performance.now());
      let timer: ReturnType<typeof globalThis.setInterval> | undefined;
      let audioTimer: ReturnType<typeof globalThis.setTimeout> | undefined;
      let audioRecorder: MediaRecorder | undefined;
      let listener: ((packet: LivePerceptionCapturePacket) => void) | undefined;
      let stopped = false;
      let encoding = false;
      let maxPacketBytes = Number.MAX_SAFE_INTEGER;
      let revision = 0;
      let activeCapture: LivePerceptionNegotiatedCapture = {
        mode: plan.mode,
        ...(videoTrack ? { width: plan.width, height: plan.height, fps: plan.framesPerSecond } : {}),
        ...(bounds.width ? { sourceNativeWidth: bounds.width } : {}),
        ...(bounds.height ? { sourceNativeHeight: bounds.height } : {}),
        ...(bounds.fps ? { sourceNativeFps: bounds.fps } : {}),
        ...(audioTracks[0]?.getSettings().sampleRate
          ? { audioSampleRate: audioTracks[0]!.getSettings().sampleRate }
          : {}),
        benchmarkClass: plan.benchmarkClass,
        revision
      };

      const constraintsFor = (
        request: Required<Pick<LivePerceptionConfigureRequest, "mode" | "width" | "height" | "fps">>
      ): MediaTrackConstraints => ({
        width: { ideal: request.width },
        height: { ideal: request.height },
        frameRate: { ideal: request.fps }
      });

      const applyCapture = async (
        request: Required<Pick<LivePerceptionConfigureRequest, "mode" | "width" | "height" | "fps">>,
        nextRevision = revision + 1
      ): Promise<LivePerceptionNegotiatedCapture> => {
        if (!videoTrack || !canvas) throw new Error("The active source has no visual track.");
        await videoTrack.applyConstraints?.(constraintsFor(request));
        const settings = videoTrack.getSettings?.() ?? {};
        const width = evenDimension(Math.min(request.width, settings.width ?? request.width));
        const height = evenDimension(Math.min(request.height, settings.height ?? request.height));
        const fps = Math.max(1, Math.min(request.fps, settings.frameRate ?? request.fps));
        canvas.width = width;
        canvas.height = height;
        revision = nextRevision;
        activeCapture = {
          ...activeCapture,
          mode: request.mode,
          width,
          height,
          fps,
          revision
        };
        return { ...activeCapture };
      };

      if (videoTrack && canvas) {
        await applyCapture({
          mode: plan.mode,
          width: plan.width,
          height: plan.height,
          fps: plan.framesPerSecond
        }, 0);
      }

      const captureFrame = async (
        capture: LivePerceptionNegotiatedCapture,
        packetLimit: number
      ): Promise<LivePerceptionCapturePacket> => {
        if (stopped || !video || !canvas || video.readyState < 2) {
          throw new Error("The visual source is not ready.");
        }
        const context = canvas.getContext("2d", { alpha: false });
        if (!context || !capture.width || !capture.height) {
          throw new Error("The visual encoder is unavailable.");
        }
        context.drawImage(video, 0, 0, capture.width, capture.height);
        const blob = await canvasBlob(canvas, packetLimit);
        return {
          modality: "video",
          timestampMs: now(),
          mimeType: blob.type || "image/jpeg",
          data: new Uint8Array(await blob.arrayBuffer()),
          settings: { width: capture.width, height: capture.height }
        };
      };

      const clearVideoTimer = (): void => {
        if (timer !== undefined) clearIntervalHandle(timer);
        timer = undefined;
      };
      const beginVideoTimer = (): void => {
        clearVideoTimer();
        if (!listener || !videoTrack || !activeCapture.fps || stopped) return;
        const run = async (): Promise<void> => {
          if (encoding || stopped || !listener) return;
          encoding = true;
          try {
            listener(await captureFrame(activeCapture, maxPacketBytes));
          } catch {
            // A later frame can recover from a transient not-ready encoder.
          } finally {
            encoding = false;
          }
        };
        timer = interval(() => void run(), Math.max(1, 1_000 / activeCapture.fps));
        void run();
      };

      const audioMimeType = (() => {
        if (!Recorder) return "";
        for (const candidate of [
          "audio/webm;codecs=opus",
          "audio/ogg;codecs=opus",
          "audio/mp4",
          "audio/webm"
        ]) {
          if (typeof Recorder.isTypeSupported !== "function" || Recorder.isTypeSupported(candidate)) {
            return candidate;
          }
        }
        return "";
      })();

      return {
        getCapture: () => ({ ...activeCapture }),
        setPacketByteLimit(bytes) {
          maxPacketBytes = positiveInteger(bytes, "Live packet byte limit");
        },
        start(nextListener) {
          if (stopped) return;
          listener = nextListener;
          beginVideoTimer();
          if (audioTracks.length > 0) {
            if (!Recorder || !environment.createMediaStream) {
              throw new Error(
                "Direct neural audio capture is unavailable; platform STT remains available separately."
              );
            }
            const audioStream = environment.createMediaStream(audioTracks);
            const recordClip = (): void => {
              if (stopped) return;
              const chunks: Blob[] = [];
              const recorder = audioMimeType
                ? new Recorder(audioStream, { mimeType: audioMimeType })
                : new Recorder(audioStream);
              audioRecorder = recorder;
              recorder.ondataavailable = (event) => {
                if (event.data.size > 0) chunks.push(event.data);
              };
              recorder.onstop = () => {
                if (stopped) return;
                const blob = new Blob(chunks, {
                  type: recorder.mimeType || audioMimeType || "audio/webm"
                });
                if (blob.size > 0 && blob.size <= maxPacketBytes) {
                  void blob.arrayBuffer().then((buffer) => {
                    if (stopped || !listener) return;
                    const settings = audioTracks[0]?.getSettings();
                    listener({
                      modality: "audio",
                      timestampMs: now(),
                      mimeType: blob.type,
                      data: new Uint8Array(buffer),
                      settings: {
                        durationMs: plan.audioChunkMs,
                        ...(settings?.sampleRate ? { sampleRate: settings.sampleRate } : {}),
                        ...(settings?.channelCount ? { channels: settings.channelCount } : {})
                      }
                    });
                  });
                }
                if (!stopped) recordClip();
              };
              recorder.start();
              audioTimer = timeout(() => {
                if (recorder.state !== "inactive") recorder.stop();
              }, plan.audioChunkMs);
            };
            recordClip();
          }
        },
        async reconfigure(request) {
          const actual = await applyCapture(request);
          beginVideoTimer();
          return actual;
        },
        async snapshotBurst(request, emit, signal) {
          if (!videoTrack || !canvas) throw new Error("The active source has no visual track.");
          clearVideoTimer();
          while (encoding) {
            await abortableBrowserDelay(2, signal, timeout, clearTimeoutHandle);
          }
          const previous = { ...activeCapture };
          const snapshotRevision = previous.revision + 1;
          let snapshotCapture: LivePerceptionNegotiatedCapture | undefined;
          let accepted = 0;
          try {
            snapshotCapture = await applyCapture({
              mode: request.mode,
              width: request.width,
              height: request.height,
              fps: request.fps ?? previous.fps ?? 1
            }, snapshotRevision);
            for (let index = 0; index < request.burstCount; index += 1) {
              if (signal.aborted) throw abortError();
              await emit(await captureFrame(snapshotCapture, request.maxPacketBytes));
              accepted += 1;
              if (index + 1 < request.burstCount) {
                await abortableBrowserDelay(
                  request.intervalMs,
                  signal,
                  timeout,
                  clearTimeoutHandle
                );
              }
            }
            return {
              mode: snapshotCapture.mode,
              resolutionMode: request.resolutionMode,
              width: snapshotCapture.width!,
              height: snapshotCapture.height!,
              fps: snapshotCapture.fps,
              ...(request.resolutionMode === "native" ? { fullResolution: true as const } : {}),
              burstCount: request.burstCount,
              intervalMs: request.intervalMs,
              framesAccepted: accepted,
              revision: snapshotRevision
            };
          } finally {
            if (!stopped) {
              await applyCapture({
                mode: previous.mode,
                width: previous.width!,
                height: previous.height!,
                fps: previous.fps ?? 1
              }, snapshotRevision).catch(() => undefined);
              beginVideoTimer();
            }
          }
        },
        stop() {
          stopped = true;
          clearVideoTimer();
          if (audioTimer !== undefined) clearTimeoutHandle(audioTimer);
          if (audioRecorder && audioRecorder.state !== "inactive") audioRecorder.stop();
          video?.pause();
          video?.removeAttribute("src");
          stream.getTracks().forEach((track) => track.stop());
        }
      };
    }
  };
}
