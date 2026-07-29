import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type {
  EvolutionCandidate,
  EvolutionRun
} from "../src/shared/types";
import {
  buildEvolutionStartRequest,
  canApproveEvolution,
  canRollbackEvolution,
  canStopEvolution,
  groupEvolutionRuns
} from "../src/renderer/src/evolutionView";

const timestamp = "2026-07-29T12:00:00.000Z";

function candidate(
  overrides: Partial<EvolutionCandidate> = {}
): EvolutionCandidate {
  return {
    id: "candidate-1",
    runId: "run-1",
    brainId: "brain-1",
    generation: 0,
    objective: "Reduce a measured retention regression",
    state: "experimenting",
    createdAt: timestamp,
    updatedAt: timestamp,
    evaluations: [],
    ...overrides
  };
}

describe("evolution renderer model", () => {
  it("builds route-specific requests without behavioral prompts or sliders", () => {
    expect(
      buildEvolutionStartRequest({
        brainId: "brain-1",
        objective: "  Improve retained coding concepts  ",
        recursive: true,
        candidateKind: "substrate",
        sourceIds: ["source-1"]
      })
    ).toEqual({
      brainId: "brain-1",
      objective: "Improve retained coding concepts",
      recursive: true,
      candidateKind: "substrate",
      latentReplay: true
    });

    expect(
      buildEvolutionStartRequest({
        brainId: "brain-1",
        objective: "Reassess the retained corpus",
        recursive: false,
        candidateKind: "data",
        sourceIds: ["source-1", "source-2"]
      })
    ).toMatchObject({
      candidateKind: "data",
      sourceIds: ["source-1", "source-2"]
    });

    expect(
      buildEvolutionStartRequest({
        brainId: "brain-1",
        objective: "Grow only a compatible expert",
        recursive: true,
        candidateKind: "architecture",
        sourceIds: []
      })
    ).toMatchObject({
      candidateKind: "architecture",
      architectureChange: { mutation: "grow-experts", addExperts: 1 }
    });
  });

  it("reconstructs persistent run groups from candidate lineage", () => {
    const knownRun: EvolutionRun = {
      id: "run-2",
      brainId: "brain-1",
      objective: "Known empty run",
      state: "experimenting",
      recursive: true,
      generation: 2,
      candidateIds: [],
      createdAt: "2026-07-29T11:00:00.000Z",
      updatedAt: "2026-07-29T11:00:00.000Z"
    };
    const groups = groupEvolutionRuns(
      [
        candidate(),
        candidate({
          id: "candidate-2",
          parentCandidateId: "candidate-1",
          generation: 1,
          state: "promoted",
          updatedAt: "2026-07-29T13:00:00.000Z"
        })
      ],
      [knownRun]
    );

    expect(groups.map((group) => group.id)).toEqual(["run-1", "run-2"]);
    const first = groups[0]!;
    const second = groups[1]!;
    expect(first).toMatchObject({
      state: "promoted",
      generation: 1
    });
    expect(first.candidates.map((entry) => entry.id)).toEqual([
      "candidate-1",
      "candidate-2"
    ]);
    expect(second.recursive).toBe(true);
  });

  it("offers only state-valid stop, approval, and rollback controls", () => {
    expect(canStopEvolution("experimenting")).toBe(true);
    expect(canStopEvolution("promoted")).toBe(false);
    expect(canApproveEvolution("awaiting-review")).toBe(true);
    expect(canApproveEvolution("rejected")).toBe(false);
    expect(canRollbackEvolution("promoted")).toBe(true);
    expect(canRollbackEvolution("rolled-back")).toBe(false);
  });

  it("wires every existing evolution preload action into the Run surface", () => {
    const root = resolve(import.meta.dirname, "..");
    const app = readFileSync(
      resolve(root, "src/renderer/src/App.tsx"),
      "utf8"
    );
    const workspace = readFileSync(
      resolve(root, "src/renderer/src/EvolutionWorkspace.tsx"),
      "utf8"
    );

    expect(app).toContain('"evolution"');
    expect(app).toContain("<EvolutionWorkspace");
    expect(workspace).toContain("window.omni.evolution.start");
    expect(workspace).toContain("window.omni.evolution.stop");
    expect(workspace).toContain("window.omni.evolution.listCandidates");
    expect(workspace).toContain("window.omni.evolution.approve");
    expect(workspace).toContain("window.omni.evolution.rollback");
    expect(workspace).not.toContain('type="range"');
    expect(workspace).not.toContain("system prompt");
  });
});
