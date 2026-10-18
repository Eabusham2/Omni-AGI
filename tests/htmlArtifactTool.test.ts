import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolPermissionRecord } from "../src/shared/types";

describe("HTML artifact ToolExecutor bridge", () => {
  const roots: string[] = [];
  afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))));

  it("uses the controller-owned action id and normal current tool permission", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-html-artifact-tool-")); roots.push(root);
    const directory = join(root, "brain"); await mkdir(directory);
    let permission: ToolPermissionRecord = { toolId: "studio.artifact", label: "Artifact", level: "auto",
      revision: 1, updatedAt: new Date(0).toISOString() };
    const service = { repository: { brainDirectory: () => directory }, listToolPermissions: async () => [permission],
      authorizeInlineImagination: async () => undefined } as unknown as BrainService;
    const tools = new ToolExecutor(service, {} as RuntimeJobManager);
    vi.spyOn(tools as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const adapter = { execute: vi.fn(async (_brain: string, _action: string, _args: Record<string, unknown>, sourceActionId: string) => ({
      format: "omni-chat-html-artifact-result", version: 1, sourceActionId
    })) };
    tools.setHtmlArtifactService(adapter);
    const invocation = { brainId: "brain", toolId: "studio.artifact", action: "create", arguments: {
      title: "Chart", html: "<canvas></canvas>", tools: []
    } };
    const completed = await tools.execute(invocation, undefined, "turn-one", undefined, { actionEventId: "event-one" });
    expect(completed).toMatchObject({ state: "complete", output: { sourceActionId: "event-one" } });
    expect(adapter.execute).toHaveBeenCalledWith("brain", "create", invocation.arguments, "event-one", expect.any(AbortSignal));
    permission = { ...permission, level: "off", revision: 2, updatedAt: new Date(1).toISOString() };
    const denied = await tools.execute(invocation, undefined, "turn-two", undefined, { actionEventId: "event-two" });
    expect(denied.state).toBe("failed");
    expect(adapter.execute).toHaveBeenCalledTimes(1);
  });
});
