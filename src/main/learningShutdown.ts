import type { InstanceLifecycleState } from "../shared/instanceLifecycle";

/** A worker shutdown is teardown, never proof of an execution checkpoint. */
export async function checkpointLearningBeforeQuit(dependencies: {
  owners(): Promise<Array<{ id: string }>>;
  status(brainId: string): Promise<InstanceLifecycleState>;
  unload(brainId: string): Promise<InstanceLifecycleState>;
  isLearning(brainId: string): boolean;
  closeLearning(): Promise<void>;
  stopWorker(): Promise<void>;
}): Promise<void> {
  for (const { id } of await dependencies.owners()) {
    const state = await dependencies.status(id);
    if (!dependencies.isLearning(id) && !state.learnedRuntimeResident && !["loaded", "loading", "resuming", "suspending"].includes(state.phase)) continue;
    const suspended = await dependencies.unload(id);
    if (suspended.learnedRuntimeResident || !["unloaded", "suspended"].includes(suspended.phase) || suspended.waitingFor) {
      throw new Error(`Cannot safely quit while ${id} is waiting: ${suspended.waitingFor?.label ?? suspended.error?.message ?? suspended.phase}. The application has not confirmed a saved boundary.`);
    }
  }
  await dependencies.closeLearning();
  await dependencies.stopWorker();
}
