import { describe, expect, it, vi } from "vitest";
import type {
  LiveObservationControl,
  LiveObservationEvent,
  LiveObservationPacketResult,
  LiveObservationSession,
  OmniApi
} from "../src/shared/types";
import {
  LivePerceptionController,
  planLivePerception,
  type LivePerceptionCaptureAdapter,
  type LivePerceptionCaptureHandle,
  type LivePerceptionCapturePacket,
  type LivePerceptionNegotiatedCapture
} from "../src/renderer/src/livePerceptionController";

const baseCapture: LivePerceptionNegotiatedCapture = {
  mode: "auto",
  width: 960,
  height: 540,
  fps: 8,
  sourceNativeWidth: 3840,
  sourceNativeHeight: 2160,
  sourceNativeFps: 60,
  benchmarkClass: "balanced",
  revision: 0
};

function session(patch: Partial<LiveObservationSession> = {}): LiveObservationSession {
  return {
    id: "observation-one",
    brainId: "brain-one",
    state: "active",
    modalities: ["video"],
    permission: {
      source: "camera",
      granted: true,
      scope: "session",
      grantedAt: "2026-08-22T00:00:00.000Z"
    },
    retention: "working",
    maxInFlight: 2,
    maxPacketBytes: 8 * 1024 * 1024,
    inFlight: 0,
    packetsReceived: 0,
    packetsAccepted: 0,
    packetsDroppedBackpressure: 0,
    bytesAccepted: 0,
    lastSequence: -1,
    createdAt: "2026-08-22T00:00:00.000Z",
    updatedAt: "2026-08-22T00:00:00.000Z",
    capabilities: { imageNeural: true, audioNeural: true, videoNeural: true },
    capture: { ...baseCapture },
    ...patch
  };
}

function packetResult(
  sequence: number,
  accepted = true,
  reason?: LiveObservationPacketResult["reason"]
): LiveObservationPacketResult {
  return {
    session: session({
      packetsReceived: sequence + 1,
      packetsAccepted: accepted ? sequence + 1 : sequence,
      lastSequence: accepted ? sequence : sequence - 1
    }),
    accepted,
    ...(reason ? { reason } : {}),
    ...(accepted
      ? {
        observation: {
          assemblyId: "assembly-live",
          spikeRate: 0.2,
          novelty: 0.4,
          packetSha256: "a".repeat(64),
          rawPacketStored: false as const,
          datasetCoverageCommitted: false as const,
          sameBrainSharedIdeaSpace: true as const,
          hiddenBehavioralPrompt: false as const
        }
      }
      : {})
  };
}

function control(
  kind: LiveObservationControl["kind"],
  requested: LiveObservationControl["requested"],
  source: LiveObservationControl["source"] = "brain"
): LiveObservationControl {
  return {
    id: `control-${kind}`,
    sessionId: "observation-one",
    kind,
    source,
    temporary: true,
    requested,
    state: "requested",
    createdAt: "2026-08-22T00:00:00.000Z",
    updatedAt: "2026-08-22T00:00:00.000Z"
  };
}

function harness() {
  let packetListener: ((packet: LivePerceptionCapturePacket) => void) | undefined;
  let eventListener: ((event: LiveObservationEvent) => void) | undefined;
  let currentCapture = { ...baseCapture };
  const captureStop = vi.fn();
  const reconfigure = vi.fn(async (request) => {
    currentCapture = {
      ...currentCapture,
      ...request,
      revision: currentCapture.revision + 1
    };
    return { ...currentCapture };
  });
  let snapshotImplementation: LivePerceptionCaptureHandle["snapshotBurst"] = async (
    request,
    listener,
    signal
  ) => {
    const previous = { ...currentCapture };
    for (let index = 0; index < request.burstCount; index += 1) {
      if (signal.aborted) throw Object.assign(new Error("cancelled"), { name: "AbortError" });
      await listener({
        modality: "video",
        timestampMs: 100 + index,
        mimeType: "image/jpeg",
        data: new Uint8Array([index + 1]),
        settings: { width: request.width, height: request.height }
      });
    }
    currentCapture = { ...previous, revision: previous.revision + 1 };
    return {
      mode: request.mode,
      resolutionMode: request.resolutionMode,
      width: request.width,
      height: request.height,
      fps: request.fps,
      ...(request.resolutionMode === "native" ? { fullResolution: true as const } : {}),
      burstCount: request.burstCount,
      intervalMs: request.intervalMs,
      framesAccepted: request.burstCount,
      revision: currentCapture.revision
    };
  };
  const captureHandle: LivePerceptionCaptureHandle = {
    getCapture: () => ({ ...currentCapture }),
    setPacketByteLimit: vi.fn(),
    start: (listener) => {
      packetListener = listener;
    },
    reconfigure,
    snapshotBurst: (...args) => snapshotImplementation(...args),
    stop: captureStop
  };
  const capture: LivePerceptionCaptureAdapter = {
    capabilities: { camera: true, microphone: true, screen: true },
    open: vi.fn(async () => captureHandle)
  };
  const active = session();
  const requestedControls: LiveObservationControl[] = [];
  const modality = {
    startObservation: vi.fn(async () => active),
    pushObservation: vi.fn(async (packet) => packetResult(packet.sequence)),
    stopObservation: vi.fn(async () => session({ state: "stopped" })),
    cancelObservation: vi.fn(async () => session({ state: "cancelled" })),
    requestObservationControl: vi.fn(async (request) => {
      const requested = control(request.kind, request.requested, "human");
      requestedControls.push(requested);
      return requested;
    }),
    resolveObservationControl: vi.fn(async (resolution) => ({
      ...control(resolution.controlId.endsWith("snapshot") ? "snapshot" : "configure", {}),
      id: resolution.controlId,
      state: resolution.state,
      actual: resolution.actual,
      reason: resolution.reason
    })),
    onObservation: vi.fn((listener) => {
      eventListener = listener;
      return () => {
        eventListener = undefined;
      };
    })
  } satisfies Pick<
    OmniApi["modality"],
    | "startObservation"
    | "pushObservation"
    | "stopObservation"
    | "cancelObservation"
    | "requestObservationControl"
    | "resolveObservationControl"
    | "onObservation"
  >;
  const actions = vi.fn();
  const scheduled: Array<() => void> = [];
  const controller = new LivePerceptionController({
    brainId: "brain-one",
    capture,
    modality,
    now: () => new Date("2026-08-22T00:00:00.000Z"),
    setTimeout: vi.fn((callback: TimerHandler) => {
      scheduled.push(callback as () => void);
      return scheduled.length;
    }) as unknown as typeof setTimeout,
    clearTimeout: vi.fn(),
    sleep: async (_milliseconds, signal) => {
      if (signal.aborted) throw abortFixture();
    },
    onActions: actions
  });
  return {
    controller,
    capture,
    captureHandle,
    captureStop,
    reconfigure,
    modality,
    actions,
    scheduled,
    requestedControls,
    setSnapshotImplementation: (implementation: LivePerceptionCaptureHandle["snapshotBurst"]) => {
      snapshotImplementation = implementation;
    },
    emitPacket: (packet: LivePerceptionCapturePacket) => packetListener?.(packet),
    emitEvent: (event: LiveObservationEvent) => eventListener?.(event)
  };
}

async function start(live: ReturnType<typeof harness>) {
  await live.controller.start("camera", planLivePerception({ hardwareTier: "personal" }));
}

describe("LivePerceptionController", () => {
  it("uses source-native and measured resources without a 720p or 10 FPS product cap", () => {
    const fast = planLivePerception({
      hardwareTier: "workstation",
      mode: "motion",
      generationTokensPerSecond: 80,
      memoryBytesPerSecond: 40 * 1024 ** 3,
      ioBytesPerSecond: 1 * 1024 ** 3,
      sourceNativeWidth: 3840,
      sourceNativeHeight: 2160,
      sourceNativeFps: 60
    });
    const slow = planLivePerception({
      hardwareTier: "workstation",
      generationTokensPerSecond: 0.8,
      memoryBytesPerSecond: 512 * 1024 ** 2,
      sourceNativeWidth: 3840,
      sourceNativeHeight: 2160,
      sourceNativeFps: 60
    });

    expect(fast.width).toBeGreaterThan(1280);
    expect(fast.height).toBeGreaterThan(720);
    expect(fast.framesPerSecond).toBe(60);
    expect(fast.benchmarkClass).toBe("native");
    expect(slow).toMatchObject({
      width: 320,
      height: 180,
      framesPerSecond: 1,
      benchmarkClass: "fallback",
      maxSnapshotBurst: 3,
      minSnapshotIntervalMs: 1_000
    });
  });

  it("opens the OS source before a typed session and records actual negotiated capture", async () => {
    const live = harness();
    const plan = planLivePerception({ hardwareTier: "personal" });
    await live.controller.start("camera", plan);

    expect(live.capture.open).toHaveBeenCalledWith("camera", plan);
    expect(live.modality.startObservation).toHaveBeenCalledWith(
      expect.objectContaining({
        brainId: "brain-one",
        modalities: ["video"],
        capture: expect.objectContaining({
          width: 960,
          height: 540,
          sourceNativeWidth: 3840,
          benchmarkClass: "balanced"
        })
      })
    );
    expect(live.modality.startObservation).not.toHaveBeenCalledWith(
      expect.objectContaining({ maxInFlight: expect.anything() })
    );
    expect(live.captureHandle.setPacketByteLimit).toHaveBeenCalledWith(8 * 1024 * 1024);
    expect(live.controller.getState()).toMatchObject({
      enabled: true,
      phase: "observing",
      negotiatedCapture: activeCapture()
    });
  });

  it("lets the authoritative start envelope adapt UI transport concurrency", async () => {
    const live = harness();
    const ideal = planLivePerception({ hardwareTier: "workstation" });
    expect(ideal.maxInFlight).toBeGreaterThan(2);

    await live.controller.start("camera", ideal);

    expect(live.modality.startObservation).not.toHaveBeenCalledWith(
      expect.objectContaining({ maxInFlight: expect.anything() })
    );
    expect(live.controller.getState()).toMatchObject({
      phase: "observing",
      plan: {
        maxInFlight: 2,
        reason: expect.stringContaining("adapted to the current device envelope")
      },
      session: { maxInFlight: 2 }
    });
  });

  it("sanitizes a raced envelope rejection instead of exposing Electron IPC", async () => {
    const live = harness();
    live.modality.startObservation.mockRejectedValueOnce(new Error(
      "Error invoking remote method 'omni:modality:start-observation': Error: Live observation maxInFlight exceeds the current resource envelope (2)."
    ));

    await live.controller.start(
      "camera",
      planLivePerception({ hardwareTier: "workstation" })
    );

    expect(live.controller.getState()).toMatchObject({
      enabled: false,
      phase: "error",
      error: "Device resources changed before Live Perception started. Try Start again; Auto will adapt safely."
    });
    expect(live.controller.getState().error).not.toMatch(/IPC|remote method|maxInFlight/);
  });

  it("pushes ordinary packets with bounded local pressure and surfaces organic actions", async () => {
    const live = harness();
    await start(live);
    let release!: (result: LiveObservationPacketResult) => void;
    live.modality.pushObservation.mockImplementationOnce(
      () => new Promise((resolve) => {
        release = resolve;
      })
    );
    const packet = {
      modality: "video" as const,
      timestampMs: 42,
      mimeType: "image/jpeg",
      data: new Uint8Array([1])
    };
    live.emitPacket(packet);
    live.emitPacket({ ...packet, timestampMs: 43 });
    live.emitPacket({ ...packet, timestampMs: 44 });
    expect(live.modality.pushObservation).toHaveBeenCalledTimes(2);
    expect(live.controller.getState().localPacketsDropped).toBe(1);
    release(packetResult(1));

    const action = {
      kind: "imagine" as const,
      source: "organic" as const,
      arguments: { modality: "image" }
    };
    live.emitEvent({ type: "action", session: session(), actions: [action] });
    expect(live.actions).toHaveBeenCalledWith([action]);
  });

  it("routes user configure and Native/Current/Custom snapshots through audited controls", async () => {
    const live = harness();
    await start(live);

    await live.controller.requestConfigure({ mode: "detail", width: 2000, durationMs: 8_000 });
    await live.controller.requestSnapshot({ resolutionMode: "native", burstCount: 4 });
    await live.controller.requestSnapshot({ resolutionMode: "current", burstCount: 2 });
    await live.controller.requestSnapshot({
      resolutionMode: "custom",
      width: 1280,
      height: 720,
      burstCount: 3,
      intervalMs: 40
    });

    expect(live.modality.requestObservationControl).toHaveBeenNthCalledWith(1, {
      sessionId: "observation-one",
      kind: "configure",
      requested: expect.objectContaining({ mode: "detail", width: 2000, durationMs: 8_000 })
    });
    expect(live.modality.requestObservationControl).toHaveBeenNthCalledWith(2, {
      sessionId: "observation-one",
      kind: "snapshot",
      requested: expect.objectContaining({ resolutionMode: "native", fullResolution: true })
    });
    expect(live.modality.requestObservationControl).toHaveBeenNthCalledWith(4, {
      sessionId: "observation-one",
      kind: "snapshot",
      requested: expect.objectContaining({
        resolutionMode: "custom",
        width: 1280,
        height: 720,
        burstCount: 3,
        intervalMs: 40
      })
    });
    expect(live.reconfigure).not.toHaveBeenCalled();
  });

  it("applies a temporary brain configure, resolves it, then restores and traces the revision", async () => {
    const live = harness();
    await start(live);
    const requested = control("configure", {
      mode: "detail",
      width: 5000,
      height: 3000,
      fps: 60,
      durationMs: 2_000
    });

    live.emitEvent({ type: "control", session: session(), control: requested });
    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledOnce());

    expect(live.reconfigure).toHaveBeenCalledWith(
      expect.objectContaining({ mode: "detail" })
    );
    const appliedRequest = live.reconfigure.mock.calls[0]?.[0];
    expect(appliedRequest.width).toBeLessThanOrEqual(baseCapture.sourceNativeWidth!);
    expect(appliedRequest.height).toBeLessThanOrEqual(baseCapture.sourceNativeHeight!);
    expect(appliedRequest.width * appliedRequest.height * appliedRequest.fps)
      .toBeLessThanOrEqual(planLivePerception({ hardwareTier: "personal" }).resourcePixelsPerSecond);
    expect(live.modality.resolveObservationControl).toHaveBeenNthCalledWith(1, {
      sessionId: "observation-one",
      controlId: "control-configure",
      state: "applied",
      actual: expect.objectContaining({ durationMs: 2_000, revertToRevision: 0 })
    });

    live.scheduled.shift()?.();
    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledTimes(2));
    expect(live.modality.resolveObservationControl).toHaveBeenNthCalledWith(2, {
      sessionId: "observation-one",
      controlId: "control-configure",
      state: "reverted",
      actual: expect.objectContaining({ width: 960, height: 540, mode: "auto" })
    });
  });

  it("ingests every native snapshot frame before resolving the traced control", async () => {
    const live = harness();
    await start(live);
    const requested = control("snapshot", {
      resolutionMode: "native",
      fullResolution: true,
      burstCount: 3,
      intervalMs: 1
    });

    live.emitEvent({ type: "control", session: session(), control: requested });
    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledOnce());

    expect(live.modality.pushObservation).toHaveBeenCalledTimes(3);
    expect(
      live.modality.pushObservation.mock.calls.map(([packet]) => packet.settings)
    ).toEqual([
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "native",
        burstIndex: 0,
        burstCount: 3
      }),
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "native",
        burstIndex: 1,
        burstCount: 3
      }),
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "native",
        burstIndex: 2,
        burstCount: 3
      })
    ]);
    expect(live.modality.resolveObservationControl).toHaveBeenCalledWith({
      sessionId: "observation-one",
      controlId: "control-snapshot",
      state: "applied",
      actual: expect.objectContaining({
        resolutionMode: "native",
        fullResolution: true,
        width: 3840,
        height: 2160,
        burstCount: 3
      })
    });
    expect(live.controller.getState().snapshot).toMatchObject({
      state: "complete",
      acceptedFrames: 3
    });
  });

  it("retries a backpressured snapshot frame with a new sequence instead of dropping it", async () => {
    const live = harness();
    await start(live);
    live.modality.pushObservation
      .mockImplementationOnce(async (packet) => packetResult(packet.sequence, false, "backpressure"))
      .mockImplementation(async (packet) => packetResult(packet.sequence));
    const requested = control("snapshot", {
      resolutionMode: "current",
      burstCount: 2,
      intervalMs: 1
    });

    live.emitEvent({ type: "control", session: session(), control: requested });
    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledOnce());

    expect(live.modality.pushObservation).toHaveBeenCalledTimes(3);
    expect(live.modality.pushObservation.mock.calls.map(([packet]) => packet.sequence)).toEqual([1, 2, 3]);
    expect(
      live.modality.pushObservation.mock.calls.map(([packet]) => packet.settings)
    ).toEqual([
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "current",
        burstIndex: 0,
        burstCount: 2
      }),
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "current",
        burstIndex: 0,
        burstCount: 2
      }),
      expect.objectContaining({
        observationControlId: "control-snapshot",
        resolutionMode: "current",
        burstIndex: 1,
        burstCount: 2
      })
    ]);
    expect(live.controller.getState().snapshot?.acceptedFrames).toBe(2);
  });

  it("cancels only an active snapshot and resolves cancellation without ending observation", async () => {
    const live = harness();
    let started!: () => void;
    live.setSnapshotImplementation(async (request, _listener, signal) => {
      started?.();
      await new Promise<void>((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(Object.assign(new Error("cancelled"), {
          name: "AbortError"
        })), { once: true });
      });
      throw new Error("unreachable");
    });
    const startSignal = new Promise<void>((resolve) => {
      started = resolve;
    });
    await start(live);
    live.emitEvent({
      type: "control",
      session: session(),
      control: control("snapshot", { resolutionMode: "native", burstCount: 2 })
    });
    await startSignal;

    expect(live.controller.cancelSnapshot()).toBe(true);
    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledOnce());
    expect(live.modality.resolveObservationControl).toHaveBeenCalledWith(
      expect.objectContaining({ state: "cancelled", controlId: "control-snapshot" })
    );
    expect(live.controller.getState()).toMatchObject({ enabled: true, phase: "observing" });
    expect(live.modality.stopObservation).not.toHaveBeenCalled();
  });

  it("stop aborts a burst, waits for its cancellation trace, restores capture, then closes", async () => {
    const live = harness();
    let snapshotStarted!: () => void;
    let restored = false;
    live.setSnapshotImplementation(async (_request, _listener, signal) => {
      snapshotStarted();
      try {
        await new Promise<void>((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(abortFixture()), { once: true });
        });
      } finally {
        restored = true;
      }
      throw abortFixture();
    });
    const began = new Promise<void>((resolve) => {
      snapshotStarted = resolve;
    });
    await start(live);
    live.emitEvent({
      type: "control",
      session: session(),
      control: control("snapshot", { resolutionMode: "native", burstCount: 2 })
    });
    await began;

    await live.controller.stop();

    expect(restored).toBe(true);
    expect(live.modality.resolveObservationControl).toHaveBeenCalledWith(
      expect.objectContaining({ state: "cancelled" })
    );
    expect(live.captureStop).toHaveBeenCalledBefore(live.modality.stopObservation);
    expect(live.controller.getState()).toMatchObject({ enabled: false, phase: "idle" });
  });

  it("ignores other-brain and duplicate requested controls", async () => {
    const live = harness();
    await start(live);
    const requested = control("configure", { mode: "motion", durationMs: 500 });
    live.emitEvent({
      type: "control",
      session: session({ brainId: "other-brain" }),
      control: requested
    });
    live.emitEvent({ type: "control", session: session(), control: requested });
    live.emitEvent({ type: "control", session: session(), control: requested });

    await vi.waitFor(() => expect(live.modality.resolveObservationControl).toHaveBeenCalledOnce());
    expect(live.reconfigure).toHaveBeenCalledOnce();
  });
});

function activeCapture(): LivePerceptionNegotiatedCapture {
  return { ...baseCapture };
}

function abortFixture(): Error {
  return Object.assign(new Error("cancelled"), { name: "AbortError" });
}
