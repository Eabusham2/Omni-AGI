import { describe, expect, it, vi } from "vitest";
import {
  createBrowserLivePerceptionCapture,
  planLivePerception,
  type BrowserLivePerceptionEnvironment
} from "../src/renderer/src/livePerceptionController";

describe("browser Live Perception capture", () => {
  it("reports device capabilities honestly", () => {
    expect(
      createBrowserLivePerceptionCapture({}).capabilities
    ).toEqual({ camera: false, microphone: false, screen: false });
  });

  it("opens camera only on request and emits independently encoded frames", async () => {
    const trackStop = vi.fn();
    const stream = {
      getVideoTracks: () => [{ stop: trackStop }],
      getAudioTracks: () => [],
      getTracks: () => [{ stop: trackStop }]
    } as unknown as MediaStream;
    const getUserMedia = vi.fn(async () => stream);
    const drawImage = vi.fn();
    const video = {
      readyState: 4,
      srcObject: null,
      muted: false,
      playsInline: false,
      play: vi.fn(async () => undefined),
      pause: vi.fn(),
      removeAttribute: vi.fn()
    } as unknown as HTMLVideoElement;
    const canvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => ({ drawImage })),
      toBlob: vi.fn((callback: BlobCallback) =>
        callback(new Blob([new Uint8Array([1, 2, 3])], { type: "image/jpeg" }))
      )
    } as unknown as HTMLCanvasElement;
    const environment: BrowserLivePerceptionEnvironment = {
      mediaDevices: {
        getUserMedia,
        getDisplayMedia: vi.fn()
      },
      createVideo: () => video,
      createCanvas: () => canvas,
      setInterval: vi.fn(() => 1) as unknown as typeof setInterval,
      clearInterval: vi.fn(),
      now: () => 123
    };
    const adapter = createBrowserLivePerceptionCapture(environment);
    expect(getUserMedia).not.toHaveBeenCalled();
    const handle = await adapter.open(
      "camera",
      planLivePerception({ hardwareTier: "personal" })
    );
    const packets: Array<{ mimeType: string; timestampMs: number; bytes: number }> = [];
    handle.start((packet) =>
      packets.push({
        mimeType: packet.mimeType,
        timestampMs: packet.timestampMs,
        bytes: packet.data.byteLength
      })
    );
    await vi.waitFor(() => expect(packets).toHaveLength(1));
    expect(getUserMedia).toHaveBeenCalledWith(
      expect.objectContaining({ video: expect.any(Object), audio: false })
    );
    expect(drawImage).toHaveBeenCalled();
    expect(packets[0]).toEqual({ mimeType: "image/jpeg", timestampMs: 123, bytes: 3 });
    await handle.stop();
    expect(trackStop).toHaveBeenCalled();
  });

  it("restarts MediaRecorder per neural-audio packet so each blob has its own container", async () => {
    const trackStop = vi.fn();
    const audioTrack = {
      stop: trackStop,
      getSettings: () => ({ sampleRate: 48_000, channelCount: 1 })
    } as unknown as MediaStreamTrack;
    const stream = {
      getVideoTracks: () => [],
      getAudioTracks: () => [audioTrack],
      getTracks: () => [audioTrack]
    } as unknown as MediaStream;
    const timers: Array<() => void> = [];
    const recorders: FakeRecorder[] = [];
    class FakeRecorder {
      static isTypeSupported(): boolean {
        return true;
      }
      state: RecordingState = "inactive";
      mimeType = "audio/webm;codecs=opus";
      ondataavailable: ((event: BlobEvent) => void) | null = null;
      onstop: (() => void) | null = null;
      constructor() {
        recorders.push(this);
      }
      start(): void {
        this.state = "recording";
      }
      stop(): void {
        this.state = "inactive";
        this.ondataavailable?.({
          data: new Blob([new Uint8Array([4, 5])], { type: this.mimeType })
        } as BlobEvent);
        this.onstop?.();
      }
    }
    const environment: BrowserLivePerceptionEnvironment = {
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
        getDisplayMedia: vi.fn()
      },
      mediaRecorderConstructor: FakeRecorder as unknown as typeof MediaRecorder,
      createMediaStream: () => stream,
      setTimeout: vi.fn((callback: TimerHandler) => {
        timers.push(callback as () => void);
        return timers.length;
      }) as unknown as typeof setTimeout,
      clearTimeout: vi.fn(),
      now: () => 456
    };
    const adapter = createBrowserLivePerceptionCapture(environment);
    const handle = await adapter.open(
      "microphone",
      planLivePerception({ hardwareTier: "micro" })
    );
    const packets: Array<{ mimeType: string; settings?: object }> = [];
    handle.start((packet) =>
      packets.push({ mimeType: packet.mimeType, settings: packet.settings })
    );
    expect(recorders).toHaveLength(1);
    timers.shift()?.();
    await vi.waitFor(() => expect(packets).toHaveLength(1));
    expect(recorders.length).toBeGreaterThanOrEqual(2);
    expect(packets[0]).toMatchObject({
      mimeType: "audio/webm;codecs=opus",
      settings: { sampleRate: 48_000, channels: 1, durationMs: 1_500 }
    });
    await handle.stop();
    expect(trackStop).toHaveBeenCalled();
  });

  it("negotiates source-native capture, exceeds the old ceiling, and reapplies constraints live", async () => {
    let settings = { width: 3840, height: 2160, frameRate: 60 };
    const applyConstraints = vi.fn(async (constraints: MediaTrackConstraints) => {
      const ideal = (value: ConstrainULong | ConstrainDouble | undefined): number | undefined =>
        typeof value === "number"
          ? value
          : value && typeof value === "object" && "ideal" in value
            ? Number(value.ideal)
            : undefined;
      settings = {
        width: ideal(constraints.width) ?? settings.width,
        height: ideal(constraints.height) ?? settings.height,
        frameRate: ideal(constraints.frameRate) ?? settings.frameRate
      };
    });
    const track = {
      stop: vi.fn(),
      applyConstraints,
      getSettings: () => ({ ...settings }),
      getCapabilities: () => ({
        width: { min: 320, max: 3840 },
        height: { min: 180, max: 2160 },
        frameRate: { min: 1, max: 60 }
      })
    } as unknown as MediaStreamTrack;
    const stream = {
      getVideoTracks: () => [track],
      getAudioTracks: () => [],
      getTracks: () => [track]
    } as unknown as MediaStream;
    const intervalSpy = vi.fn((_callback: TimerHandler, _delay?: number) => 1);
    const canvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => ({ drawImage: vi.fn() })),
      toBlob: vi.fn((callback: BlobCallback) =>
        callback(new Blob([new Uint8Array([1])], { type: "image/jpeg" }))
      )
    } as unknown as HTMLCanvasElement;
    const environment: BrowserLivePerceptionEnvironment = {
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
        getDisplayMedia: vi.fn()
      },
      createVideo: () => ({
        readyState: 4,
        play: vi.fn(async () => undefined),
        pause: vi.fn(),
        removeAttribute: vi.fn()
      }) as unknown as HTMLVideoElement,
      createCanvas: () => canvas,
      setInterval: intervalSpy as unknown as typeof setInterval,
      clearInterval: vi.fn(),
      now: () => 1
    };
    const handle = await createBrowserLivePerceptionCapture(environment).open(
      "camera",
      planLivePerception({ hardwareTier: "workstation", mode: "motion" })
    );

    expect(handle.getCapture()).toMatchObject({
      sourceNativeWidth: 3840,
      sourceNativeHeight: 2160,
      sourceNativeFps: 60,
      fps: 60
    });
    expect(handle.getCapture().width).toBeGreaterThan(1280);
    handle.start(() => undefined);
    expect(intervalSpy).toHaveBeenCalledWith(expect.any(Function), expect.any(Number));
    expect(Number(intervalSpy.mock.calls[0]?.[1])).toBeLessThan(100);

    const changed = await handle.reconfigure({
      mode: "detail",
      width: 3000,
      height: 1600,
      fps: 45
    });
    expect(applyConstraints).toHaveBeenLastCalledWith({
      width: { ideal: 3000 },
      height: { ideal: 1600 },
      frameRate: { ideal: 45 }
    });
    expect(changed).toMatchObject({ mode: "detail", width: 3000, height: 1600, fps: 45 });
    expect(canvas).toMatchObject({ width: 3000, height: 1600 });
    await handle.stop();
  });

  it("captures every selected-resolution burst frame, cancels mid-burst, and restores capture", async () => {
    let settings = { width: 1280, height: 720, frameRate: 30 };
    const applyConstraints = vi.fn(async (constraints: MediaTrackConstraints) => {
      const number = (value: ConstrainULong | ConstrainDouble | undefined): number | undefined =>
        typeof value === "number"
          ? value
          : value && typeof value === "object" && "ideal" in value
            ? Number(value.ideal)
            : undefined;
      settings = {
        width: number(constraints.width) ?? settings.width,
        height: number(constraints.height) ?? settings.height,
        frameRate: number(constraints.frameRate) ?? settings.frameRate
      };
    });
    const track = {
      stop: vi.fn(),
      applyConstraints,
      getSettings: () => ({ ...settings }),
      getCapabilities: () => ({
        width: { min: 320, max: 3840 },
        height: { min: 180, max: 2160 },
        frameRate: { min: 1, max: 60 }
      })
    } as unknown as MediaStreamTrack;
    const stream = {
      getVideoTracks: () => [track],
      getAudioTracks: () => [],
      getTracks: () => [track]
    } as unknown as MediaStream;
    const video = {
      readyState: 4,
      play: vi.fn(async () => undefined),
      pause: vi.fn(),
      removeAttribute: vi.fn()
    } as unknown as HTMLVideoElement;
    const canvas = {
      width: 0,
      height: 0,
      getContext: vi.fn(() => ({ drawImage: vi.fn() })),
      toBlob: vi.fn((callback: BlobCallback) =>
        callback(new Blob([new Uint8Array([9])], { type: "image/jpeg" }))
      )
    } as unknown as HTMLCanvasElement;
    const handle = await createBrowserLivePerceptionCapture({
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
        getDisplayMedia: vi.fn()
      },
      createVideo: () => video,
      createCanvas: () => canvas,
      setInterval: vi.fn(() => 1) as unknown as typeof setInterval,
      clearInterval: vi.fn(),
      now: (() => {
        let value = 0;
        return () => value += 1;
      })()
    }).open("camera", planLivePerception({ hardwareTier: "personal" }));
    const before = handle.getCapture();
    const accepted: number[] = [];
    const complete = await handle.snapshotBurst(
      {
        mode: before.mode,
        resolutionMode: "native",
        width: 3840,
        height: 2160,
        fps: 10,
        burstCount: 3,
        intervalMs: 1,
        maxPacketBytes: 1024
      },
      async (packet) => {
        accepted.push(packet.data.byteLength);
      },
      new AbortController().signal
    );
    expect(accepted).toEqual([1, 1, 1]);
    expect(complete).toMatchObject({
      resolutionMode: "native",
      width: 3840,
      height: 2160,
      framesAccepted: 3
    });
    expect(handle.getCapture()).toMatchObject({
      width: before.width,
      height: before.height,
      fps: before.fps
    });

    const controller = new AbortController();
    let emitted = 0;
    await expect(handle.snapshotBurst(
      {
        mode: before.mode,
        resolutionMode: "custom",
        width: 1920,
        height: 1080,
        fps: 10,
        burstCount: 3,
        intervalMs: 1,
        maxPacketBytes: 1024
      },
      async () => {
        emitted += 1;
        controller.abort();
      },
      controller.signal
    )).rejects.toMatchObject({ name: "AbortError" });
    expect(emitted).toBe(1);
    expect(handle.getCapture()).toMatchObject({
      width: before.width,
      height: before.height,
      fps: before.fps
    });
    expect(applyConstraints.mock.calls.length).toBeGreaterThanOrEqual(5);
    await handle.stop();
  });
});
