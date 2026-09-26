import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import {
  compactParameterCount,
  neuralParameterAccounting,
  neuralResourceRuntimeRows
} from "../src/renderer/src/neuralRuntimePresentation";

describe("native OmniCortex runtime presentation", () => {
  it("shows worker-owned neural parameter and memory measurements", () => {
    const app = readFileSync(resolve(import.meta.dirname, "../src/renderer/src/App.tsx"), "utf8");
    expect(app).toContain("neuralResourceRuntimeRows(workspace?.runtimeCard)");
    expect(app).toContain("neuralParameterAccounting(workspace?.runtimeCard)");
    expect(app).not.toContain("foundationRuntimeRows(workspace?.runtimeCard)");
    expect(app).toContain("<NeuralParameterCount accounting={parameterAccounting}");
    expect(app).toContain("title={accounting.exactLabel}");
    expect(app).toContain('workspace ? (workspace.hiddenBehavioralPrompt ? "Present" : "None") : "Pending measurement"');
    expect(app).toContain('workspace ? (workspace.rawLongTermTextInjected ? "Present" : "None") : "Pending measurement"');
    expect(app).toContain("One brain; each logical parameter counted once.");
  });

  it("counts native dense parameters and dynamic synapses without imported weights", () => {
    const accounting = {
      mutableDenseParameters: 4_226_915,
      substrateDynamicSparseSynapses: 4_811_811,
      dynamicSparseSynapses: 4_811_811,
      totalNeuralParameters: 9_038_726,
      countingRule: "mutable dense + dynamic sparse"
    };
    expect(compactParameterCount(accounting.totalNeuralParameters)).toBe("9.039M");
    expect(compactParameterCount(1_234)).toBe("1.234K");
    expect(compactParameterCount(1_234_567_890_123)).toBe("1.235T");
    expect(neuralParameterAccounting({ parameterAccounting: accounting })).toEqual({
      ...accounting,
      compactTotal: "9.039M",
      exactLabel:
        "9,038,726 logical neural weights and connections, counted once in one brain. " +
        "4,226,915 core weights and " +
        "4,811,811 grown connections. " +
        accounting.countingRule
    });
    expect(neuralParameterAccounting({
      parameterAccounting: { ...accounting, foundationEffectiveParameters: 1 }
    })).toBeUndefined();
    expect(neuralParameterAccounting({
      parameterAccounting: { ...accounting, sequenceDynamicSparseSynapses: 0 }
    })).toBeUndefined();
    expect(neuralParameterAccounting({
      parameterAccounting: { ...accounting, totalNeuralParameters: 9_038_727 }
    })).toBeUndefined();
  });

  it("shows measured Python physical footprint without an RSS estimate", () => {
    expect(neuralResourceRuntimeRows({
      state_offload: {
        resources: {
          processMemoryBytes: 2 * 1024 ** 3,
          processPeakMemoryBytes: 6 * 1024 ** 3
        }
      }
    })).toEqual([[
      "Python neural memory",
      "2.0 GiB current · 6.0 GiB peak · physical footprint"
    ]]);
  });
});
