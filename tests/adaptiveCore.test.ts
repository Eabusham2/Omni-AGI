import { describe, expect, it } from "vitest";
import {
  consolidateBrain,
  learnText,
  runFallbackChat
} from "../src/main/adaptiveCore";
import {
  BRAIN_SCHEMA_VERSION,
  DEFAULT_CONFIG,
  type BrainDocument
} from "../src/shared/types";

function blankBrain(): BrainDocument {
  const now = new Date().toISOString();
  return {
    schemaVersion: BRAIN_SCHEMA_VERSION,
    id: "test-brain",
    name: "Test brain",
    createdAt: now,
    updatedAt: now,
    lineage: { rootId: "test-brain", generation: 0 },
    config: { ...DEFAULT_CONFIG, name: "Test brain", noise: 0 },
    concepts: {},
    synapses: {},
    ideas: [],
    workingMemory: [],
    liquidState: {
      values: Array.from({ length: 16 }, () => 0),
      timeConstants: Array.from({ length: 16 }, () => 0.5),
      lastUpdatedAt: now
    },
    messages: [],
    traces: [],
    trainingSources: [],
    counters: { plasticityEvents: 0, inferenceCount: 0, consolidationCycles: 0 }
  };
}

describe("degraded adaptive core", () => {
  it("turns experience into concepts, ternary synapses, and liquid state", () => {
    const brain = blankBrain();
    const delta = learnText(
      brain,
      "Liquid neurons connect adaptive memory. Liquid neurons reinforce adaptive memory.",
      "document",
      "fixture"
    );

    expect(delta.concepts).toBeGreaterThan(2);
    expect(delta.synapses).toBeGreaterThan(0);
    expect(Object.values(brain.synapses).every((synapse) =>
      [-1, 0, 1].includes(synapse.effectiveWeight)
    )).toBe(true);
    expect(brain.counters.plasticityEvents).toBeGreaterThan(0);
    expect(brain.liquidState.values.some((value) => value !== 0)).toBe(true);
  });

  it("persists a measured trace for every continuous-chat turn", () => {
    const brain = blankBrain();
    learnText(brain, "Synaptic timing changes the strength of a learned association.", "document");
    const result = runFallbackChat(
      brain,
      "How does synaptic timing change an association?",
      "A neural-worker response."
    );

    expect(result.brainMessage.content).toBe("A neural-worker response.");
    expect(result.brain.messages).toHaveLength(2);
    expect(result.brain.traces).toHaveLength(1);
    expect(result.trace.steps.map((step) => step.stage)).toEqual(
      expect.arrayContaining(["plasticity", "associative-recall", "liquid-routing"])
    );
    expect(result.brain.counters.inferenceCount).toBe(1);
  });

  it("consolidates frequently used synapses without adding a reward model", () => {
    const brain = blankBrain();
    for (let index = 0; index < 4; index += 1) {
      learnText(brain, "spikes reinforce pathways between recurring ideas", "conversation");
    }
    const before = Math.max(...Object.values(brain.synapses).map((synapse) => synapse.stability));
    consolidateBrain(brain);
    const after = Math.max(...Object.values(brain.synapses).map((synapse) => synapse.stability));

    expect(after).toBeGreaterThan(before);
    expect(brain.counters.consolidationCycles).toBe(1);
    expect(JSON.stringify(brain)).not.toContain("reward_model");
  });
});

