import { describe, expect, it } from "vitest";
import {
  connectionCountPresentation,
  conciseUiMessage,
  isPristineBrainSummary,
  livePerceptionErrorMessage,
  recoveryPointCreationErrorMessage,
  substrateOverviewCount,
  utf8DraftTokenCount
} from "../src/renderer/src/uiPresentation";

describe("bounded renderer presentation", () => {
  it("keeps the useful diagnostic and removes JavaScript and Python stack frames", () => {
    expect(conciseUiMessage(
      "Error: Storage write failed\n    at persist (/private/app.js:42:7)\n    at run (/private/app.js:9:2)"
    )).toBe("Storage write failed");
    expect(conciseUiMessage(
      "Traceback (most recent call last):\n  File \"/private/worker.py\", line 8, in run\nRuntimeError: Device is out of memory"
    )).toBe("Device is out of memory");
    expect(conciseUiMessage(
      "Error invoking remote method 'omni:chat:send': Error: Worker traceback:\nTraceback (most recent call last):\n  File \"/private/worker.py\", line 42, in run\nTernaryPackingError: decoder.action_policy.hidden.weight has an invalid scale"
    )).toBe("decoder.action_policy.hidden.weight has an invalid scale");
  });

  it("bounds a single unbroken diagnostic instead of overflowing a card", () => {
    const result = conciseUiMessage(`Error: ${"x".repeat(500)}`);
    expect(result).toHaveLength(320);
    expect(result.endsWith("…")).toBe(true);
    expect(result).not.toContain("\n");
  });

  it("presents expected recovery-point cancellation without Electron IPC text", () => {
    expect(recoveryPointCreationErrorMessage(new Error(
      "Error invoking remote method 'omni:brain:snapshot': Error: Worker request \"checkpoint\" was cancelled."
    ))).toBe(
      "Recovery point creation cancelled; current brain unchanged."
    );
    expect(recoveryPointCreationErrorMessage(
      new Error("Error invoking remote method 'omni:brain:snapshot': Error: checkpoint disk verification failed")
    )).toBe("checkpoint disk verification failed");
  });

  it("turns a raced Live Perception envelope rejection into actionable UI copy", () => {
    expect(livePerceptionErrorMessage(new Error(
      "Error invoking remote method 'omni:modality:start-observation': Error: Live observation maxInFlight exceeds the current resource envelope (2)."
    ))).toBe(
      "Device resources changed before Live Perception started. Try Start again; Auto will adapt safely."
    );
  });

  it("counts the renderer draft at Omni's exact UTF-8 byte-token boundary", () => {
    expect(utf8DraftTokenCount("hello")).toBe(5);
    expect(utf8DraftTokenCount("hi 🌱")).toBe(7);
  });

  it("uses a compact empty card only for a completely new origin", () => {
    expect(isPristineBrainSummary({ concepts: 0, synapses: 0, generation: 0 })).toBe(true);
    expect(isPristineBrainSummary({ concepts: 1, synapses: 0, generation: 0 })).toBe(false);
    expect(isPristineBrainSummary({ concepts: 0, synapses: 0, generation: 1 })).toBe(false);
    expect(isPristineBrainSummary({
      concepts: 0,
      synapses: 0,
      generation: 0,
      substrateTotals: { neurons: 14_973, assemblies: 121, synapses: 4_812_443 }
    })).toBe(false);
    expect(isPristineBrainSummary({
      concepts: 0,
      synapses: 0,
      generation: 0,
      neuralUpdates: 12_000,
      inferenceCount: 2,
      trainingSources: 1
    })).toBe(false);
  });

  it("keeps Brain Map renderable when activity precedes its first mirrored page", () => {
    expect(substrateOverviewCount({ plasticityEvents: 7 })).toBe(1);
    expect(substrateOverviewCount({ neurons: 640, assemblies: 12, plasticityEvents: 7 })).toBe(640);
    expect(substrateOverviewCount({ mirroredEndpointCount: 8, plasticityEvents: 0 })).toBe(8);
    expect(substrateOverviewCount({ plasticityEvents: 0 })).toBe(0);
  });

  it("keeps unique live connections separate from cumulative update activity", () => {
    expect(connectionCountPresentation({
      queriedConnections: 4_815_971,
      persistedConnections: 4_812_443,
      mirroredConnections: 0,
      cumulativeUpdates: 9_002_117
    })).toEqual({
      uniqueConnections: 4_815_971,
      uniqueSource: "queried-substrate",
      cumulativeUpdates: 9_002_117,
      topologyPending: false
    });
    expect(connectionCountPresentation({
      persistedConnections: 4_812_443,
      mirroredConnections: 0,
      cumulativeUpdates: 9_002_117
    })).toMatchObject({
      uniqueConnections: 4_812_443,
      uniqueSource: "persisted-substrate",
      topologyPending: false
    });
    expect(connectionCountPresentation({
      mirroredConnections: 0,
      cumulativeUpdates: 5_712
    })).toEqual({
      uniqueConnections: 0,
      uniqueSource: "compatibility-mirror",
      cumulativeUpdates: 5_712,
      topologyPending: true
    });
    expect(connectionCountPresentation({
      persistedConnections: 0,
      mirroredConnections: 99,
      cumulativeUpdates: 5_712
    })).toMatchObject({
      uniqueConnections: 0,
      uniqueSource: "persisted-substrate",
      topologyPending: false
    });
  });
});
