import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { architectureMutationPolicy, normalizeCompatibleArchitectureMutation } from "../src/shared/architectureMutation";
import { cleanGeometryHoldouts } from "../src/main/evolutionController";

describe("compatible native architecture operation protocol", () => {
  it("admits typed depth/router/region growth without silently using expert growth", () => {
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-depth", addLayers: 2 })).toEqual({ mutation: "grow-depth", addLayers: 2 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-router", addNeurons: 64 })).toEqual({ mutation: "grow-router", addNeurons: 64 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-regions", addRegions: 3, neuronsPerRegion: 16 })).toEqual({ mutation: "grow-regions", addRegions: 3, neuronsPerRegion: 16 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "grow-experts" })).toEqual({ mutation: "grow-experts", addExperts: 1 });
  });
  it("rejects blind width changes, unexpected axes and invalid counts", () => {
    for (const mutation of [
      { mutation: "grow-width", width: 512 },
      { mutation: "grow-depth", addLayers: 0 },
      { mutation: "grow-router", addNeurons: true },
      { mutation: "grow-regions", addRegions: 2 },
      { mutation: "grow-depth", addLayers: 2, newHeads: 8 }
    ]) expect(() => normalizeCompatibleArchitectureMutation(mutation)).toThrow();
  });
  it("admits explicit isolated geometry candidates without a preservation/improvement claim", () => {
    expect(normalizeCompatibleArchitectureMutation({ mutation: "resize-width", dModel: 512, feedForward: 1536, nHeads: 8 }))
      .toEqual({ mutation: "resize-width", dModel: 512, feedForward: 1536, nHeads: 8 });
    expect(normalizeCompatibleArchitectureMutation({ mutation: "repartition-heads", nHeads: 16 }))
      .toEqual({ mutation: "repartition-heads", nHeads: 16 });
    expect(architectureMutationPolicy({ mutation: "repartition-heads", nHeads: 16 }))
      .toEqual({ candidateOnly: true, geometryChanges: true, functionPreservingAtInsertion: false, requiresTrainingEvaluation: true, improvementVerified: false });
    for (const mutation of [{ mutation: "resize-width" }, { mutation: "repartition-heads", nHeads: true },
      { mutation: "resize-width", dModel: 512, approved: true }])
      expect(() => normalizeCompatibleArchitectureMutation(mutation)).toThrow();
  });
  it("does not turn Auto permission into geometry approval", () => {
    const controller = readFileSync(resolve("src/main/evolutionController.ts"), "utf8");
    const approval = controller.slice(controller.indexOf("  private async approveWorker("));
    expect(approval).toContain('geometry && !["ask", "full"].includes(currentPermission)');
    expect(approval).toContain("evaluationSha256");
    expect(approval).toContain("candidateStateChecksum");
  });
  it("accepts only explicit complete real-data geometry holdout declarations", () => {
    const valid = { token: [{ path: resolve("fixtures", "words.jsonl"), records: 2 }],
      modality: [{ path: resolve("fixtures", "image.png"), kind: "image", conditionText: "a scene" }],
      tool: [{ path: resolve("fixtures", "actions.jsonl"), records: 1 }] };
    expect(cleanGeometryHoldouts(valid)).toEqual(valid);
    expect(() => cleanGeometryHoldouts({ ...valid, token: [] })).toThrow(/token holdouts/i);
    expect(() => cleanGeometryHoldouts({ ...valid, tool: [{ path: resolve("fixtures", "actions.jsonl"), records: 0 }] })).toThrow(/tool holdout/i);
    expect(() => cleanGeometryHoldouts({ ...valid, modality: [{ path: "relative.png", kind: "image", conditionText: "a scene" }] })).toThrow(/absolute selected/i);
  });
});
