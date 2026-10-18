import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";
import type { ToolInvocation } from "../src/shared/types";

describe("ToolExecutor main-owned live origin lease", () => {
  let root: string, executor: ToolExecutor, service: BrainService, origin: HostActionOriginAuthority;
  let owner: AbortController, current: boolean, revision: number;
  const invocation: ToolInvocation = { brainId: "origin-brain", toolId: "brain.substrate", action: "apply", arguments: { planId: "plan", reviewSha256: "a".repeat(64) } };
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-origin-lease-")); await mkdir(join(root, invocation.brainId));
    current = true; revision = 1; owner = new AbortController();
    origin = { revision: "exact-session", signal: owner.signal, assertCurrent: async () => { if (!current) throw new Error("Origin disabled"); } };
    service = { repository: { brainDirectory: () => join(root, invocation.brainId) },
      listToolPermissions: vi.fn(async () => [{ toolId: invocation.toolId, level: "full", revision, updatedAt: "2026-10-15T12:00:00Z" }]),
      authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined)
    } as unknown as BrainService;
    executor = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() });
    vi.spyOn(executor as unknown as { audit(): Promise<void> }, "audit").mockResolvedValue(undefined);
  });
  afterEach(async () => { executor.cancel(invocation.brainId); await rm(root, { recursive: true, force: true }); });
  const context = () => ({ actionEventId: "owned-event", originAuthority: origin });

  it("rejects origin revocation during awaited host policy before dispatch", async () => {
    const execute = vi.fn(async () => ({ state: "queued" })); executor.setSparseSubstrateService({ execute });
    vi.mocked(service.listToolPermissions).mockImplementation(async () => {
      current = false; return [{ toolId: invocation.toolId, label: "Fixture", level: "full", revision, updatedAt: "2026-10-15T12:00:00Z" }];
    });
    expect(await executor.execute(invocation, undefined, "new-request", undefined, context())).toMatchObject({ state: "failed", error: "Origin disabled" });
    expect(execute).not.toHaveBeenCalled();
  });

  it("retains a private exact permission/origin guard after the enqueue result returns", async () => {
    let retained: HostActionOriginAuthority | undefined;
    executor.setSparseSubstrateService({ execute: async (_brain, _action, _args, _request, _signal, authority) => { retained = authority; return { state: "queued" }; } });
    const result = await executor.execute(invocation, undefined, "detached-request", undefined, context());
    expect(result).toMatchObject({ state: "complete", output: { state: "queued" } });
    await expect(retained!.assertCurrent()).resolves.toBeUndefined();
    revision += 1; await expect(retained!.assertCurrent()).rejects.toThrow("Tool permission changed");
    revision -= 1; current = false; await expect(retained!.assertCurrent()).rejects.toThrow("Origin disabled");
    expect(JSON.stringify(result)).not.toContain("exact-session");
    current = true; origin.revision = "substituted-session";
    await expect(retained!.assertCurrent()).rejects.toThrow("origin authority changed");
  });

  it("propagates module Stop during a running host call without a serialized grant", async () => {
    let entered!: () => void; const started = new Promise<void>(resolve => { entered = resolve; });
    executor.setSparseSubstrateService({ execute: async (_brain, _action, _args, _request, signal) => {
      entered(); await new Promise<void>((_resolve, reject) => { signal.addEventListener("abort", () => reject(signal.reason), { once: true }); });
    } });
    const pending = executor.execute(invocation, undefined, "stopped-request", undefined, context());
    await started; owner.abort(new Error("Module stopped"));
    expect(await pending).toMatchObject({ state: "failed", dispatchStarted: true });
  });
});
