import { AutomationService } from "./automationService";
import { AutomationAuthorityRegistry } from "./automationAuthority";
import { automationCanonical, automationHash } from "./automationFormat";
import type { BrainRepository } from "./brainRepository";
import type { BrainService } from "./brainService";
import type { ClockService } from "./clockService";
import type { ToolExecutor } from "./toolExecutor";

/** Native-independent host composition. All nested effects use ordinary ToolExecutor. */
export function createAutomationRuntime(input: {
  repository: Pick<BrainRepository, "root" | "get" | "brainDirectory">;
  service: Pick<BrainService, "listToolPermissions" | "requireRuntimeAdmissionOpen" | "addInstanceRecoveryBoundary" | "onToolPermissionChanged" | "queueCompletedToolExecution">;
  tools: Pick<ToolExecutor, "setAutomationService" | "execute" | "cancel" | "cancelAndWait" | "invalidateApproval">;
  clock: Pick<ClockService, "setAutomationDelegate">;
  checkResources(brainId: string, bytes: number, purpose: string): Promise<void>;
  reserveWrite(root: string, bytes: number): Promise<void>;
  onError(error: Error, brainId: string): void;
}): AutomationService {
  const authority = new AutomationAuthorityRegistry(input.repository.root);
  const automations = new AutomationService({
    brainDirectory: async id => { await input.repository.get(id, false); return input.repository.brainDirectory(id); },
    tools: input.tools,
    authority: async id => { const permission = (await input.service.listToolPermissions(id)).find(row => row.toolId === "studio.automation");
      return { enabled: !!permission && permission.level !== "off", revision: automationHash(automationCanonical(permission ?
        { toolId: permission.toolId, level: permission.level, updatedAt: permission.updatedAt } : { toolId: "studio.automation", level: "off" })) }; },
    ownerEpoch: id => authority.epoch(id), rotateOwnerEpoch: id => authority.rotate(id),
    admissionOpen: () => { try { input.service.requireRuntimeAdmissionOpen(); return true; } catch { return false; } },
    checkResources: input.checkResources, reserveWrite: input.reserveWrite,
    observeOutcome: (invocation, result) => input.service.queueCompletedToolExecution(invocation, result), onError: input.onError
  });
  input.tools.setAutomationService(automations);
  input.clock.setAutomationDelegate(occurrence => automations.acceptOccurrence(occurrence));
  input.service.addInstanceRecoveryBoundary(id => automations.forget(id));
  input.service.onToolPermissionChanged(({ brainId, toolId }) => { if (toolId === "studio.automation") void automations.revokeRuns(brainId, undefined, "Independent-action access changed; remaining steps require fresh review.").catch(error => input.onError(error, brainId)); });
  return automations;
}
