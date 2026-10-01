import { describe, expect, it } from "vitest";
import { EVOLUTION_TEST_NAMES, sourceImprovementEvidence, isProtectedEvolutionPath } from "../src/main/evolutionPolicy";

const checks = (failed?: string) => EVOLUTION_TEST_NAMES.map(name => ({
  name, passed: name !== failed, exitCode: name === failed ? 1 : 0, durationMs: 10
}));

describe("observed source benefit is separate from all-green checks", () => {
  it("does not invent benefit from a nonempty diff or one faster test run", () => {
    const evidence = sourceImprovementEvidence(checks(), checks().map(value => ({ ...value, durationMs: 1 })));
    expect(evidence.passed).toBe(false);
    expect(evidence.improvedChecks).toEqual([]);
    expect(evidence.neuralQualityEstablished).toBe(false);
    expect(evidence.timingBenefitEstablished).toBe(false);
  });
  it("accepts an actually repaired fixed check while preserving the others", () => {
    expect(sourceImprovementEvidence(checks("unit"), checks())).toMatchObject({
      allRequiredPairs: true, passed: true, improvedChecks: ["unit"], regressions: []
    });
  });
  it("rejects missing pairs, duplicate checks, cancelled baselines and regressions", () => {
    const before = checks("unit");
    expect(sourceImprovementEvidence(before.slice(1), checks()).passed).toBe(false);
    expect(sourceImprovementEvidence([...before, before[0]], checks()).passed).toBe(false);
    expect(sourceImprovementEvidence(before.map(value => value.name === "unit" ? { ...value, exitCode: 130 } : value), checks()).passed).toBe(false);
    expect(sourceImprovementEvidence(before, checks("build"))).toMatchObject({ passed: false, regressions: ["build"] });
  });
  it("keeps native evaluator and paired evidence rules outside candidate-writable source", () => {
    for (const path of ["engine/omni_core/evolution.py", "engine/omni_core/paired_geometry_statistics.py",
      "engine/omni_core/geometry_holdout_evaluation.py", "engine/omni_core/registered_geometry_holdouts.py"]) {
      expect(isProtectedEvolutionPath(path, new Set())).toBe(true);
    }
    expect(isProtectedEvolutionPath("engine/omni_core/liquid.py", new Set())).toBe(false);
  });
});
