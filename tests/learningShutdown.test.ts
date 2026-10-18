import { describe, expect, it, vi } from "vitest";
import { checkpointLearningBeforeQuit } from "../src/main/learningShutdown";
import type { InstanceLifecycleState } from "../src/shared/instanceLifecycle";

const state = (phase: InstanceLifecycleState["phase"], resident = false): InstanceLifecycleState => ({ brainId: "brain", phase,
  learnedRuntimeResident: resident, autoLoad: false, revision: 1, updatedAt: new Date().toISOString(), pausedJobIds: [], reconnect: [] });
describe("acknowledged learning quit boundary", () => {
  it("checkpoints before closing task ledgers or tearing down the worker", async () => {
    const order: string[] = [];
    await checkpointLearningBeforeQuit({ owners: async () => [{ id: "brain" }], status: async () => state("loaded", true), isLearning: () => true,
      unload: async () => { order.push("checkpoint"); return state("suspended"); }, closeLearning: async () => { order.push("ledger-close"); }, stopWorker: async () => { order.push("teardown"); } });
    expect(order).toEqual(["checkpoint", "ledger-close", "teardown"]);
  });
  it("does not claim saved state or tear down on an unsupported pending boundary", async () => {
    const closeLearning = vi.fn(), stopWorker = vi.fn();
    await expect(checkpointLearningBeforeQuit({ owners: async () => [{ id: "brain" }], status: async () => state("loaded", true), isLearning: () => true,
      unload: async () => ({ ...state("suspending", true), waitingFor: { kind: "external-effect", label: "Waiting for the host receipt" } }), closeLearning, stopWorker }))
      .rejects.toThrow(/not confirmed a saved boundary/);
    expect(closeLearning).not.toHaveBeenCalled(); expect(stopWorker).not.toHaveBeenCalled();
  });
});
