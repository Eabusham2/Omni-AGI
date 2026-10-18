/** Host policy/dispatch only. No model, key, map provider or network request. */
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionLevel } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-map-basemap-policy-")); roots.push(root);
  const directory = join(root, "brain-one"); await mkdir(directory);
  let level: ToolPermissionLevel = "ask", revision = 1;
  const service = { repository: { brainDirectory: () => directory },
    listToolPermissions: async () => [{ toolId: "maps.display", label: "Maps", level, revision, updatedAt: new Date(revision * 1000).toISOString() }],
    authorizeInlineImagination: async () => undefined
  } as unknown as BrainService;
  const executor = new ToolExecutor(service, {} as RuntimeJobManager);
  vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "maps.display", action: "basemaps", arguments: {} };
  return { executor, invocation, permission(next: ToolPermissionLevel) { level = next; revision++; } };
}

it("keeps optional imagery on the existing Off/Ask/exact-approval path and actual-outcome observer", async () => {
  const f = await fixture(), observed = vi.fn(async () => undefined);
  const execute = vi.fn(async (_brain: string, _action: string, _args: Record<string, unknown>, signal: AbortSignal, current: () => Promise<void>) => {
    await current(); signal.throwIfAborted(); return { profiles: [], defaultProvider: null, publicServiceSelected: false };
  });
  f.executor.setMapBasemapService({ execute });
  f.permission("off"); expect((await f.executor.execute(f.invocation)).state).toBe("failed"); expect(execute).not.toHaveBeenCalled();
  f.permission("ask"); const asked = await f.executor.execute(f.invocation, undefined, "map-turn", observed);
  expect(asked.state).toBe("approval-required"); expect(execute).not.toHaveBeenCalled();
  const actual = await f.executor.execute({ ...f.invocation, approvalToken: asked.approvalToken }, undefined, "map-turn", observed);
  expect(actual.state, actual.error).toBe("complete"); expect(execute).toHaveBeenCalledOnce();
  expect(actual.output).toEqual({ profiles: [], defaultProvider: null, publicServiceSelected: false });
  expect(observed).toHaveBeenCalledExactlyOnceWith(actual);
});

it("passes a live permission fence to imagery, so revocation during an await cannot publish a result", async () => {
  const f = await fixture(); f.permission("full"); let entered = false, delivered = false;
  f.executor.setMapBasemapService({ execute: async (_brain, _action, _args, _signal, current) => {
    entered = true; await current(); f.permission("off"); await current(); delivered = true; return {};
  } });
  const result = await f.executor.execute(f.invocation);
  expect(entered).toBe(true); expect(result.state).toBe("failed"); expect(delivered).toBe(false);
  expect(result.error).toContain("permission changed");
});
