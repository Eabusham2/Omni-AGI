import { checkedPortablePresentation, type PortableWorkspaceCreateResult } from "../shared/portableWorkspace";
import type { BrainStorageOperationSession } from "./brainStorageOperations";
import type { PortableWorkspaceService } from "./portableWorkspaceService";

interface PortableCreateHost {
  portable: Pick<PortableWorkspaceService, "create">;
  brain(brainId: string): Promise<{ id: string; name: string }>;
  confirm(name: string): Promise<boolean>;
  chooseRuntime(): Promise<string | null>;
  chooseDestination(name: string): Promise<string | null>;
  session(requestId: string, brainId: string): Pick<BrainStorageOperationSession, "hooks" | "complete" | "fail">;
  finish(requestId: string): void;
}
function identity(value: unknown): string {
  if (typeof value !== "string" || /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.exec(value)?.[0] !== value) throw new Error("Invalid portable instance/request identity.");
  return value;
}

/** Renderer supplies only typed intent. Native dialogs own both filesystem paths. */
export async function createPortableWorkspaceFromIntent(host: PortableCreateHost, input: unknown): Promise<PortableWorkspaceCreateResult | null> {
  if (!input || typeof input !== "object" || Array.isArray(input) ||
      Object.keys(input).sort().join(",") !== "brainId,presentation,requestId") throw new Error("Portable creation accepts only a selected instance, request ID and ordinary presentation settings.");
  const row = input as Record<string, unknown>, brainId = identity(row.brainId), requestId = identity(row.requestId);
  const presentation = checkedPortablePresentation(row.presentation);
  const brain = await host.brain(brainId);
  if (brain.id !== brainId) throw new Error("Portable source belongs to another instance.");
  if (!await host.confirm(brain.name)) return null;
  const runtimeDirectory = await host.chooseRuntime(); if (!runtimeDirectory) return null;
  const destination = await host.chooseDestination(brain.name); if (!destination) return null;
  const session = host.session(requestId, brainId);
  try {
    const result = await host.portable.create({ brainId, requestId, presentation, runtimeDirectory, destination }, session.hooks);
    session.complete(); return result;
  } catch (error) { session.fail(error); throw error; }
  finally { host.finish(requestId); }
}
