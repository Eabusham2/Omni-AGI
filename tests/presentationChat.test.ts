import { describe, expect, it } from "vitest";
import { recordNeuralChat } from "../src/main/presentationChat";
import {
  BRAIN_SCHEMA_VERSION,
  DEFAULT_CONFIG,
  type BrainDocument
} from "../src/shared/types";

function blankPresentation(): BrainDocument {
  const now = new Date().toISOString();
  return {
    schemaVersion: BRAIN_SCHEMA_VERSION,
    releaseFormat: "stable-1.0",
    id: "presentation-test",
    name: "Presentation test",
    createdAt: now,
    updatedAt: now,
    lineage: { rootId: "presentation-test", generation: 0 },
    config: { ...DEFAULT_CONFIG, name: "Presentation test" },
    concepts: {},
    synapses: {},
    ideas: [],
    workingMemory: [],
    liquidState: {
      values: [],
      timeConstants: [],
      lastUpdatedAt: now
    },
    messages: [],
    traces: [],
    trainingSources: [],
    counters: {
      plasticityEvents: 0,
      inferenceCount: 0,
      consolidationCycles: 0
    }
  };
}

describe("neural chat presentation", () => {
  it("records worker output without creating an Electron memory substrate", () => {
    const brain = blankPresentation();
    const result = recordNeuralChat(brain, "  hello  ", "  learned response  ");

    expect(result.humanMessage.content).toBe("hello");
    expect(result.brainMessage.content).toBe("learned response");
    expect(result.brain.messages).toHaveLength(2);
    expect(result.brain.traces).toHaveLength(1);
    expect(result.brain.concepts).toEqual({});
    expect(result.brain.synapses).toEqual({});
    expect(result.brain.ideas).toEqual([]);
    expect(result.brain.workingMemory).toEqual([]);
  });

  it("rejects empty message boundaries", () => {
    expect(() => recordNeuralChat(blankPresentation(), " \0 ", "response")).toThrow(
      "A chat message cannot be empty."
    );
    expect(() => recordNeuralChat(blankPresentation(), "message", " \0 ")).toThrow(
      "The neural worker returned an empty response."
    );
  });
});
