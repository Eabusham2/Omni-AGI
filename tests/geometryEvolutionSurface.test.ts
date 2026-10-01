import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import { buildEvolutionStartRequest } from "../src/renderer/src/evolutionView";
import { cleanGeometryHoldouts, EvolutionController } from "../src/main/evolutionController";
import { portableGeometryReference } from "../src/main/brainRepository";
import type { RecursiveEvolutionReassessment } from "../src/shared/types";

const holdouts = {
  token: [{ path: "/selected/text.jsonl", records: 20 }],
  tool: [{ path: "/selected/tool.jsonl", records: 20 }],
  modality: [{ path: "/selected/audio.wav", kind: "audio" as const, conditionText: "literal observed utterance" }]
};

describe("typed native geometry and observed recursive reassessment", () => {
  it("builds usable width/head requests with real selected holdouts, no fictional scores", () => {
    const request = buildEvolutionStartRequest({ brainId: "fixture", objective: "measure language gain", recursive: true,
      candidateKind: "architecture", sourceIds: [], architectureChange: { mutation: "resize-width", dModel: 96, nHeads: 8, feedForward: 160 }, geometryHoldouts: holdouts });
    expect(request.architectureChange).toEqual({ mutation: "resize-width", dModel: 96, nHeads: 8, feedForward: 160 });
    expect(request.geometryHoldouts).toEqual(cleanGeometryHoldouts(holdouts));
    expect(request.objectives).toContain("language-prediction");
    expect(buildEvolutionStartRequest({ brainId: "fixture", objective: "wider", recursive: false,
      candidateKind: "architecture", sourceIds: [], architectureChange: { mutation: "repartition-heads", nHeads: 8 } })).not.toHaveProperty("geometryHoldouts");
    // Omission reuses protected saved registration; absence is worker-fail-closed.
    for (const candidateKind of ["neural", "data", "substrate", "architecture"] as const) {
      const native = buildEvolutionStartRequest({ brainId: "fixture", objective: "actual measured improvement", recursive: false,
        candidateKind, sourceIds: [], geometryHoldouts: holdouts });
      expect(native.geometryHoldouts).toEqual(holdouts);
      expect(native.objectives).toContain("language-prediction");
    }
  });

  it("advertises supported geometry in native schema and forwards its typed holdouts", () => {
    const service = readFileSync(new URL("../src/main/brainService.ts", import.meta.url), "utf8");
    const controller = readFileSync(new URL("../src/main/chatActionController.ts", import.meta.url), "utf8");
    const workspace = readFileSync(new URL("../src/renderer/src/EvolutionWorkspace.tsx", import.meta.url), "utf8");
    expect(service).toContain('"resize-width", "repartition-heads"');
    expect(service).toContain("properties.geometryHoldouts");
    expect(controller).toContain("geometryHoldouts: cleanGeometryHoldouts(action.arguments.geometryHoldouts)");
    expect(workspace).toContain('value="resize-width"');
    expect(workspace).toContain("Real held-out files (JSON declarations)");
  });

  it("resolves legacy physical aliases identically without weakening traversal checks", () => {
    const original = `evaluation/data/${"a".repeat(32)}.данные+!`;
    const portable = portableGeometryReference(original);
    expect(portable).toMatch(/^evaluation\/data\/[a-f0-9]{32}\.legacy-[a-f0-9]{64}$/);
    expect(portableGeometryReference(portable)).toBe(portable);
    expect(() => portableGeometryReference("evaluation/data/../outside")).toThrow();
  });

  it("hands observed promotion/configuration to the normal scheduler without invented candidate/prose", async () => {
    const controller = Object.create(EvolutionController.prototype) as EvolutionController;
    Object.assign(controller, { trustedReassessmentParents: new Map() });
    const observed: RecursiveEvolutionReassessment[] = [];
    controller.setRecursiveReassessmentHandler(async (event) => { observed.push(event); });
    const configuration = buildEvolutionStartRequest({ brainId: "fixture", objective: "actual explicit objective", recursive: true,
      candidateKind: "architecture", sourceIds: [], architectureChange: { mutation: "repartition-heads", nHeads: 8 }, geometryHoldouts: holdouts });
    const candidate = { id: "parent", brainId: "fixture", candidateKind: "architecture", workerCandidateId: "b".repeat(32),
      continuationRequest: configuration, evaluations: [{ passed: true, evaluatorSha256: "c".repeat(64) }],
      neuralPromotion: { stateChecksumAfter: "d".repeat(64) }, processMeasurement: { wallSeconds: 12, evaluatorSuccesses: 1 } };
    await (controller as unknown as { requestRecursiveReassessment(value: unknown): Promise<void> }).requestRecursiveReassessment(candidate);
    expect(observed).toHaveLength(1);
    expect(observed[0]!.parentCandidateId).toBe("parent");
    expect(observed[0]!.configuration.architectureChange).toEqual(configuration.architectureChange);
    expect(observed[0]!.configuration.geometryHoldouts).toEqual(holdouts);
    expect(observed[0]).not.toHaveProperty("prompt");
    expect(observed[0]).not.toHaveProperty("syntheticExperience");
  });
});
