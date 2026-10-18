import { ClockService } from "./clockService";
import { clockHash } from "./clockStore";
import type { BrainService } from "./brainService";
import type { BrainRepository } from "./brainRepository";
import type { ChatActionController } from "./chatActionController";
import type { ToolExecutor } from "./toolExecutor";
import type { ScheduledOccurrence } from "../shared/clock";
import type { InstanceResumeAttachment } from "../shared/instanceLifecycleApi";
import type { ChatStreamEvent } from "../shared/types";

/** Main composition; this never grants Load or changes a mind's Active-mode policy. */
export function createClockRuntime(input: {
  repository: Pick<BrainRepository, "get" | "brainDirectory">;
  service: Pick<BrainService, "listToolPermissions" | "requireRuntimeAdmissionOpen" | "queryNativeTurnStatus" | "addInstanceRecoveryBoundary" | "onInstanceLifecycle" | "instanceLifecycle" | "loadInstance">;
  actions: Pick<ChatActionController, "isBusy" | "send" | "on">;
  tools: Pick<ToolExecutor, "setClockService">;
  isLearning(brainId: string): boolean;
  notify(occurrence: ScheduledOccurrence): Promise<void> | void;
  reserveWrite?(root: string, bytes: number): Promise<void>;
  onError(error: Error, brainId: string): void;
}): ClockService {
  const clock = new ClockService({
    brainDirectory: async id => { await input.repository.get(id, false); return input.repository.brainDirectory(id); },
    authority: async (id, toolId = "studio.schedule") => {
      const permission = (await input.service.listToolPermissions(id)).find(row => row.toolId === toolId);
      return { enabled: !!permission && permission.level !== "off", revision: clockHash(JSON.stringify(permission ?
        { toolId, level: permission.level, updatedAt: permission.updatedAt } : { toolId, level: "off" })) };
    },
    loaded: async id => input.service.instanceLifecycle.isLoaded(id), canRun: (id, turnId) => input.service.instanceLifecycle.canRun(id, turnId),
    load: async id => { const state = await input.service.loadInstance(id); if (state.phase !== "loaded") throw new Error(state.error?.message ?? state.phase); },
    busy: id => input.actions.isBusy(id) || input.isLearning(id),
    admissionOpen: () => { try { input.service.requireRuntimeAdmissionOpen(); return true; } catch { return false; } },
    turnStatus: async (id, turnId, task) => { const status = await input.service.queryNativeTurnStatus(id, turnId, task);
      return { state: status.status, ...(status.attachment ? { continuation: status.attachment } : {}),
        ...(status.receipt ? { detail: `Committed ${status.receipt.committedAt}; original native receipt recovered.` } : {}) }; },
    wake: async (occurrence, continuation) => {
      await input.actions.send(occurrence.brainId, occurrence.input, undefined, occurrence.turnId, undefined, continuation as InstanceResumeAttachment | undefined);
      const status = await input.service.queryNativeTurnStatus(occurrence.brainId, occurrence.turnId, occurrence.input);
      return { committed: status.status === "committed", detail: status.status === "committed" ? `Native turn committed ${status.receipt?.committedAt}.` : `Native completion remains ${status.status}.` };
    },
    notify: input.notify, reserveWrite: input.reserveWrite, onError: input.onError
  });
  input.tools.setClockService(clock);
  input.actions.on("stream", (event: ChatStreamEvent) => { try { clock.observeStream(event); } catch (error) { input.onError(error instanceof Error ? error : new Error(String(error)), event.brainId); } });
  input.service.addInstanceRecoveryBoundary(id => clock.forget(id));
  input.service.onInstanceLifecycle(state => { if (state.phase === "loaded") void clock.recover([state.brainId]).then(() => clock.tick()).catch(error => input.onError(error, state.brainId)); });
  return clock;
}
