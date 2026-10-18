import { describe, expect, it, vi } from "vitest";
import { createPortableWorkspaceFromIntent } from "../src/main/portableWorkspaceIpc";
import { BrainStorageOperationSession } from "../src/main/brainStorageOperations";

const request = { brainId: "fixture", requestId: "exact-portable-request", presentation: {
  appearance: { schemaVersion: 1 as const, mode: "system" as const, palette: "aqua" as const, layout: "glass" as const },
  workspace: { schemaVersion: 1 as const, arrangement: "split" as const, notificationsEnabled: false, profiles: [] }, theme: null
} };
function fixture() {
  const session = new BrainStorageOperationSession(request.requestId, "export", request.brainId);
  const result = { destination: "/host-chosen/output", workspaceId: "portable-fixture", brainId: request.brainId, files: 4, bytes: 100, destinationReviewRequired: true as const };
  const host = {
    portable: { create: vi.fn(async () => result) },
    brain: vi.fn(async () => ({ id: request.brainId, name: "Fixture" })),
    confirm: vi.fn(async () => true), chooseRuntime: vi.fn(async () => "/host-chosen/windows-runtime"),
    chooseDestination: vi.fn(async (): Promise<string | null> => "/host-chosen/output"),
    session: vi.fn(() => session), finish: vi.fn()
  };
  return { host, session, result };
}
describe("portable creation intent boundary", () => {
  it("takes exact paths only from host choices and completes the tracked storage receipt", async () => {
    const { host, session, result } = fixture();
    await expect(createPortableWorkspaceFromIntent(host, request)).resolves.toEqual(result);
    expect(host.portable.create).toHaveBeenCalledWith({ ...request,
      runtimeDirectory: "/host-chosen/windows-runtime", destination: "/host-chosen/output" }, session.hooks);
    expect(host.session).toHaveBeenCalledWith(request.requestId, request.brainId);
    expect(session.current().state).toBe("complete"); expect(host.finish).toHaveBeenCalledWith(request.requestId);
  });
  it("rejects renderer paths/credential storage and creates no job when a dialog is cancelled", async () => {
    const { host } = fixture();
    await expect(createPortableWorkspaceFromIntent(host, { ...request, destination: "/injected" })).rejects.toThrow("only a selected");
    expect(host.brain).not.toHaveBeenCalled();
    await expect(createPortableWorkspaceFromIntent(host, { ...request, presentation: { ...request.presentation, credentials: "not accepted" } })).rejects.toThrow("unsupported fields");
    host.chooseDestination.mockResolvedValueOnce(null);
    await expect(createPortableWorkspaceFromIntent(host, request)).resolves.toBeNull();
    expect(host.session).not.toHaveBeenCalled(); expect(host.portable.create).not.toHaveBeenCalled();
  });
  it("retains failed operation status and retires its exact session without claiming completion", async () => {
    const { host, session } = fixture();
    host.portable.create.mockRejectedValueOnce(new Error("Synthetic copy refusal; no files written"));
    await expect(createPortableWorkspaceFromIntent(host, request)).rejects.toThrow("Synthetic copy refusal");
    expect(session.current()).toMatchObject({ state: "failed", operationId: request.requestId });
    expect(host.finish).toHaveBeenCalledOnce();
  });
});
