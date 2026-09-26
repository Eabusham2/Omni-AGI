import { createHash } from "node:crypto";
import { EventEmitter } from "node:events";
import { describe, expect, it, vi } from "vitest";

vi.mock("node:os", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:os")>();
  return {
    ...actual,
    freemem: () => 8 * 1024 * 1024 * 1024
  };
});
import {
  RuntimeJobManager,
  type BrainService
} from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import {
  DEFAULT_CONFIG,
  type BrainDocument,
  type LiveObservationSessionStartRequest
} from "../src/shared/types";

function fixtureBrain(): BrainDocument {
  return {
    id: "brain-live",
    config: {
      ...DEFAULT_CONFIG,
      name: "Live brain",
      visionEnabled: true,
      imageEnabled: true,
      audioEnabled: true,
      videoEnabled: true
    },
    readiness: { state: "ready" },
    toolPermissions: []
  } as unknown as BrainDocument;
}

function managerWith(
  implementation: (method: string, params: Record<string, unknown>) => Promise<unknown>,
  schemas: Array<{ id: string; actions: string[]; grant: "auto" }> = [
    {
      id: "studio.ui",
      actions: ["notify"],
      grant: "auto"
    }
  ]
): { manager: RuntimeJobManager; request: ReturnType<typeof vi.fn> } {
  const events = new EventEmitter();
  const request = vi.fn(implementation);
  const engine = Object.assign(events, { request }) as unknown as EngineSupervisor;
  const brain = fixtureBrain();
  const service = {
    repository: {
      get: vi.fn(async () => brain),
      brainDirectory: vi.fn(() => "/tmp/omni-live-brain")
    },
    preflightStart: vi.fn(async () => undefined),
    neuralToolSchemas: vi.fn(() => schemas)
  } as unknown as BrainService;
  return { manager: new RuntimeJobManager(service, engine), request };
}

function startRequest(
  overrides: Partial<LiveObservationSessionStartRequest> = {}
): LiveObservationSessionStartRequest {
  return {
    brainId: "brain-live",
    modalities: ["image", "video"],
    permission: {
      source: "camera",
      granted: true,
      scope: "session",
      grantedAt: "2026-08-22T12:00:00.000Z"
    },
    retention: "neural",
    capture: {
      mode: "detail",
      width: 4096,
      height: 2160,
      fps: 60,
      sourceNativeWidth: 4096,
      sourceNativeHeight: 2160,
      sourceNativeFps: 60,
      benchmarkClass: "native"
    },
    ...overrides
  };
}

describe("live observation backend", () => {
  it("derives omitted concurrency from the live envelope but still rejects an unsafe explicit override", async () => {
    const implementation = async (method: string) => method === "start_observation"
      ? { capabilities: { imageNeural: true, audioNeural: true, videoNeural: true } }
      : {};
    const adaptive = managerWith(implementation).manager;
    const session = await adaptive.startObservation(startRequest());
    expect(session.maxInFlight).toBeGreaterThanOrEqual(1);

    const strict = managerWith(implementation).manager;
    await expect(strict.startObservation(startRequest({
      maxInFlight: Number.MAX_SAFE_INTEGER
    }))).rejects.toThrow(/maxInFlight exceeds the current resource envelope/i);
  });

  it("preserves every learned tool schema when adding live observation", async () => {
    const schemas = [
      { id: "studio.ui", actions: ["notify"], grant: "auto" as const },
      ...Array.from({ length: 137 }, (_, index) => ({
        id: `learned.tool.${index}`,
        actions: ["run"],
        grant: "auto" as const
      }))
    ];
    let delivered: Array<{ id: string }> = [];
    const { manager } = managerWith(async (method, params) => {
      if (method === "start_observation") {
        delivered = params.toolSchemas as Array<{ id: string }>;
        return { capabilities: {} };
      }
      return {};
    }, schemas);

    await manager.startObservation(startRequest());
    expect(delivered).toHaveLength(schemas.length + 1);
    expect(delivered.map((schema) => schema.id)).toEqual([
      ...schemas.filter((schema) => schema.id !== "studio.ui").map((schema) => schema.id),
      "studio.ui",
      "device.observe"
    ]);
  });

  it("accepts a capable source packet above the old 4 MiB cap and emits organic controls", async () => {
    let organic = true;
    const { manager } = managerWith(async (method, params) => {
      if (method === "start_observation") {
        return {
          createdAt: "2026-08-22T12:00:00.000Z",
          capabilities: {
            imageNeural: true,
            audioNeural: true,
            videoNeural: true
          }
        };
      }
      if (method === "observe_packet") {
        const bytes = Buffer.from(String(params.dataBase64), "base64");
        const actions = organic
          ? [
              {
                kind: "tool",
                toolId: "device.observe",
                action: "configure",
                arguments: {
                  mode: "motion",
                  width: 1920,
                  height: 1080,
                  fps: 60,
                  durationMs: 1500
                }
              }
            ]
          : [];
        organic = false;
        return {
          observation: {
            packetSha256: createHash("sha256").update(bytes).digest("hex"),
            spikeRate: 0.2,
            novelty: 0.7,
            assemblyId: "assembly-live",
            rawPacketStored: false,
            datasetCoverageCommitted: false,
            sameBrainSharedIdeaSpace: true,
            hiddenBehavioralPrompt: false
          },
          actions
        };
      }
      return {};
    });
    const events: unknown[] = [];
    manager.on("observation", (event) => events.push(event));
    const session = await manager.startObservation(startRequest());
    expect(session.maxPacketBytes).toBeGreaterThan(4 * 1024 * 1024);
    expect(session.capture).toMatchObject({
      width: 4096,
      height: 2160,
      fps: 60,
      benchmarkClass: "native"
    });

    const bytes = new Uint8Array(5 * 1024 * 1024);
    bytes[0] = 137;
    const packet = await manager.pushObservation({
      sessionId: session.id,
      modality: "image",
      sequence: 0,
      timestampMs: 1,
      mimeType: "image/png",
      data: bytes,
      settings: { width: 4096, height: 2160 }
    });
    expect(packet.accepted).toBe(true);
    expect(packet.observation).toMatchObject({
      assemblyId: "assembly-live",
      rawPacketStored: false,
      datasetCoverageCommitted: false,
      sameBrainSharedIdeaSpace: true
    });
    expect(packet.controls?.[0]).toMatchObject({
      kind: "configure",
      source: "brain",
      state: "requested",
      requested: { mode: "motion", fps: 60, durationMs: 1500 }
    });
    expect(events).toEqual(
      expect.arrayContaining([expect.objectContaining({ type: "control" })])
    );
    await expect(
      manager.pushObservation({
        sessionId: session.id,
        modality: "image",
        sequence: 1,
        timestampMs: 2,
        mimeType: "image/png",
        data: new Uint8Array(session.maxPacketBytes + 1)
      })
    ).rejects.toThrow(/source\/resource envelope/i);
  });

  it.each([
    "audio/webm;codecs=opus",
    "audio/ogg;codecs=opus",
    "audio/mp4",
    "audio/webm"
  ])("accepts browser-recorded neural audio as %s", async (mimeType) => {
    let deliveredMime = "";
    const { manager } = managerWith(async (method, params) => {
      if (method === "start_observation") {
        return {
          capabilities: {
            imageNeural: true,
            audioNeural: true,
            videoNeural: true
          }
        };
      }
      if (method === "observe_packet") {
        deliveredMime = String(params.mimeType);
        const bytes = Buffer.from(String(params.dataBase64), "base64");
        return {
          observation: {
            packetSha256: createHash("sha256").update(bytes).digest("hex"),
            spikeRate: 0.12,
            novelty: 0.3,
            assemblyId: "assembly-audio",
            rawPacketStored: false,
            datasetCoverageCommitted: false,
            sameBrainSharedIdeaSpace: true,
            hiddenBehavioralPrompt: false
          },
          actions: []
        };
      }
      return {};
    });
    const session = await manager.startObservation(startRequest({
      modalities: ["audio"],
      permission: {
        source: "microphone",
        granted: true,
        scope: "session",
        grantedAt: "2026-08-22T12:00:00.000Z"
      },
      capture: {
        mode: "auto",
        audioSampleRate: 48_000,
        benchmarkClass: "balanced"
      }
    }));
    const result = await manager.pushObservation({
      sessionId: session.id,
      modality: "audio",
      sequence: 0,
      timestampMs: 1,
      mimeType,
      data: new Uint8Array([0x1a, 0x45, 0xdf, 0xa3]),
      settings: { sampleRate: 48_000, channels: 1, durationMs: 750 }
    });
    expect(result).toMatchObject({
      accepted: true,
      observation: {
        assemblyId: "assembly-audio",
        sameBrainSharedIdeaSpace: true,
        rawPacketStored: false
      }
    });
    expect(deliveredMime).toBe(mimeType);
  });

  it("drops excess concurrent packets without claiming dataset coverage", async () => {
    let release: ((value: unknown) => void) | undefined;
    const { manager } = managerWith(async (method, params) => {
      if (method === "start_observation") {
        return {
          capabilities: {
            imageNeural: true,
            audioNeural: true,
            videoNeural: true
          }
        };
      }
      if (method === "observe_packet") {
        return new Promise((resolve) => {
          release = () => {
            const bytes = Buffer.from(String(params.dataBase64), "base64");
            resolve({
              observation: {
                packetSha256: createHash("sha256").update(bytes).digest("hex"),
                spikeRate: 0.1,
                novelty: 0.1,
                rawPacketStored: false,
                datasetCoverageCommitted: false,
                sameBrainSharedIdeaSpace: true,
                hiddenBehavioralPrompt: false
              },
              actions: []
            });
          };
        });
      }
      return {};
    });
    const session = await manager.startObservation(
      startRequest({
        maxInFlight: 1,
        capture: {
          mode: "auto",
          sourceNativeWidth: 320,
          sourceNativeHeight: 180,
          sourceNativeFps: 1,
          benchmarkClass: "fallback"
        }
      })
    );
    expect(session.maxInFlight).toBe(1);
    expect(session.capture).toMatchObject({ width: 320, height: 180, fps: 1 });
    const first = manager.pushObservation({
      sessionId: session.id,
      modality: "image",
      sequence: 0,
      timestampMs: 0,
      mimeType: "image/jpeg",
      data: new Uint8Array([1, 2, 3])
    });
    await vi.waitFor(() => expect(release).toBeTypeOf("function"));
    const dropped = await manager.pushObservation({
      sessionId: session.id,
      modality: "image",
      sequence: 1,
      timestampMs: 1,
      mimeType: "image/jpeg",
      data: new Uint8Array([4, 5, 6])
    });
    expect(dropped).toMatchObject({ accepted: false, reason: "backpressure" });
    expect(dropped.session.packetsDroppedBackpressure).toBe(1);
    release?.({});
    await expect(first).resolves.toMatchObject({ accepted: true });
  });

  it("audits a human custom snapshot and applies it only after every frame", async () => {
    const { manager } = managerWith(async (method, params) => {
      if (method === "start_observation") {
        return {
          capabilities: {
            imageNeural: true,
            audioNeural: true,
            videoNeural: true
          }
        };
      }
      if (method === "observe_packet") {
        const bytes = Buffer.from(String(params.dataBase64), "base64");
        const settings = params.settings as Record<string, unknown>;
        return {
          observation: {
            packetSha256: createHash("sha256").update(bytes).digest("hex"),
            spikeRate: 0.15,
            novelty: 0.4,
            rawPacketStored: false,
            datasetCoverageCommitted: false,
            sameBrainSharedIdeaSpace: true,
            hiddenBehavioralPrompt: false,
            observationControlId: settings.observationControlId,
            resolutionMode: settings.resolutionMode,
            burstIndex: settings.burstIndex,
            burstCount: settings.burstCount
          },
          actions: []
        };
      }
      return {};
    });
    const session = await manager.startObservation(startRequest());
    const control = await manager.requestObservationControl({
      sessionId: session.id,
      kind: "snapshot",
      requested: {
        resolutionMode: "custom",
        width: 1920,
        height: 1080,
        burstCount: 2,
        intervalMs: 30
      }
    });
    expect(control).toMatchObject({ source: "human", state: "requested" });

    for (let index = 0; index < 2; index += 1) {
      await manager.pushObservation({
        sessionId: session.id,
        modality: "image",
        sequence: index,
        timestampMs: index * 30,
        mimeType: "image/jpeg",
        data: new Uint8Array([index + 1, 7, 9]),
        settings: {
          width: 1920,
          height: 1080,
          observationControlId: control.id,
          resolutionMode: "custom",
          burstIndex: index,
          burstCount: 2
        }
      });
      if (index === 0) {
        await expect(
          manager.resolveObservationControl({
            sessionId: session.id,
            controlId: control.id,
            state: "applied",
            actual: {
              mode: "detail",
              resolutionMode: "custom",
              width: 1920,
              height: 1080,
              burstCount: 2,
              intervalMs: 30,
              revision: 1
            }
          })
        ).rejects.toThrow(/every audited burst frame enters the brain/i);
      }
    }
    await expect(
      manager.resolveObservationControl({
        sessionId: session.id,
        controlId: control.id,
        state: "applied",
        actual: {
          mode: "detail",
          resolutionMode: "custom",
          width: 1920,
          height: 1080,
          burstCount: 2,
          intervalMs: 30,
          revision: 1
        }
      })
    ).resolves.toMatchObject({
      state: "applied",
      actual: {
        resolutionMode: "custom",
        width: 1920,
        height: 1080,
        burstCount: 2
      }
    });
  });
});
