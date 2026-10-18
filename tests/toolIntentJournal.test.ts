import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { persistAuthorizedToolIntent } from "../src/main/toolIntentJournal";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolExecutionResult, ToolInvocation } from "../src/shared/types";
import { vi } from "vitest";

describe("durable pre-effect tool intent", () => {
  it("writes a synced independent receipt without retaining raw argument values", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-intent-fixture-"));
    try {
      const owner = join(directory, "fixture");
      await mkdir(owner);
      const path = await persistAuthorizedToolIntent(owner, {
        brainId: "fixture", toolId: "mcp.example", action: "call",
        arguments: { content: "actual-user-content", authorization: "host-secret" }
      }, { id: randomUUID(), requestId: "turn-fixture", startedAt: new Date().toISOString(), permission: "auto", permissionRevision: "revision" });
      const text = await readFile(path, "utf8");
      expect(text).not.toContain("actual-user-content");
      expect(text).not.toContain("host-secret");
      expect(JSON.parse(text)).toMatchObject({ state: "authorized-before-side-effects", requestId: "turn-fixture", rawArgumentValuesRetained: false });
      expect(JSON.parse(text).argumentSha256).toMatch(/^[a-f0-9]{64}$/);
    } finally { await rm(directory, { recursive: true, force: true }); }
  });

  it("rejects Off and redirected journal storage before a dispatch can begin", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-intent-fixture-"));
    const outside = await mkdtemp(join(tmpdir(), "omni-intent-outside-"));
    const invocation = { brainId: "fixture", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } };
    try {
      const owner = join(directory, "fixture");
      await mkdir(owner);
      await expect(persistAuthorizedToolIntent(owner, invocation, { id: randomUUID(), requestId: "fixture", startedAt: "now", permission: "off", permissionRevision: "rev" })).rejects.toThrow("Disabled");
      await expect(persistAuthorizedToolIntent(owner, { ...invocation, brainId: "another-brain" }, { id: randomUUID(), requestId: "fixture", startedAt: "now", permission: "full", permissionRevision: "rev" })).rejects.toThrow("brain owner");
      await expect(persistAuthorizedToolIntent(owner, invocation, { id: randomUUID(), requestId: "wrong\ncontrol", startedAt: "now", permission: "full", permissionRevision: "rev" })).rejects.toThrow("request ownership");
      await symlink(outside, join(owner, "engine"), process.platform === "win32" ? "junction" : "dir");
      await expect(persistAuthorizedToolIntent(owner, invocation, { id: randomUUID(), requestId: "fixture", startedAt: "now", permission: "full", permissionRevision: "rev" })).rejects.toThrow("real directory");
    } finally {
      await rm(directory, { recursive: true, force: true });
      await rm(outside, { recursive: true, force: true });
    }
  });

  it("method-fixture dispatch requires actual permission and an existing durable intent", async () => {
    const directory = await mkdtemp(join(tmpdir(), "omni-intent-method-"));
    const owner = join(directory, "fixture");
    await mkdir(owner);
    let level: "off" | "ask" | "auto" | "full" = "ask";
    const boundaries: string[] = [];
    const service = {
      repository: { brainDirectory: () => owner },
      listToolPermissions: async () => [{ toolId: "web.fetch", level, updatedAt: "revision" }],
      // Production calls this guard for every admitted tool. For ordinary
      // web.fetch it is a no-op, after the synced intent and fresh permissions.
      authorizeInlineImagination: vi.fn(async (invocation: ToolInvocation, executionId: string, requestId: string, intentPath: string) => {
        expect(invocation.toolId).toBe("web.fetch");
        expect(JSON.parse(await readFile(intentPath, "utf8"))).toMatchObject({
          id: executionId, requestId, state: "authorized-before-side-effects"
        });
        boundaries.push("authorized");
      }),
      queueCompletedToolExecution: vi.fn(async (_invocation: ToolInvocation, result: ToolExecutionResult) => {
        expect(result).toMatchObject({ state: "complete", dispatchStarted: true,
          output: { body: "fixture only; no request sent" } });
        boundaries.push("retained");
      })
    } as unknown as BrainService;
    const executor = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() });
    const audit = vi.spyOn(executor as unknown as { audit(): Promise<void> }, "audit").mockImplementation(async () => { boundaries.push("audited"); });
    const dispatch = vi.spyOn(executor as unknown as { dispatch(): Promise<unknown> }, "dispatch").mockImplementation(async () => {
      const receipts = await readdir(join(owner, "engine", "operational-tool-intents"));
      expect(receipts.length).toBeGreaterThan(0);
      expect(JSON.parse(await readFile(join(owner, "engine", "operational-tool-intents", receipts.at(-1)!), "utf8"))).toMatchObject({ brainId: "fixture", state: "authorized-before-side-effects" });
      boundaries.push("dispatched");
      return { body: "fixture only; no request sent" };
    });
    const invocation = { brainId: "fixture", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org" } };
    try {
      const ask = await executor.execute(invocation, undefined, "turn-fixture");
      expect(ask.state).toBe("approval-required");
      expect(dispatch).not.toHaveBeenCalled();
      const approved = await executor.execute({ ...invocation, approvalToken: ask.approvalToken }, undefined, "turn-fixture");
      expect(approved.state).toBe("complete");
      expect(approved.error).toBeUndefined();
      expect(dispatch).toHaveBeenCalledOnce();
      expect(boundaries).toEqual(["authorized", "dispatched", "retained", "audited"]);
      level = "off";
      expect((await executor.execute(invocation)).state).toBe("failed");
      expect(dispatch).toHaveBeenCalledOnce();
      level = "full";
      expect((await executor.execute(invocation)).state).toBe("complete");
      expect(dispatch).toHaveBeenCalledTimes(2);
      level = "auto";
      expect((await executor.execute(invocation)).state).toBe("complete");
      expect(dispatch).toHaveBeenCalledTimes(3);
      expect(audit).toHaveBeenCalled();
    } finally { await rm(directory, { recursive: true, force: true }); }
  });
});
