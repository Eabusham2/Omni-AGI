/** Real ToolExecutor/intent+SQLite paths; host service edges explicitly mocked, no native model. */
import { mkdir, mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import { ClockService } from "../src/main/clockService";
import { clockHash } from "../src/main/clockStore";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-clock-tool-")); roots.push(root);
  const directory = join(root, "brain-one"); await mkdir(directory);
  let permission: ToolPermissionRecord = { toolId: "studio.schedule", label: "Schedules", level: "ask", updatedAt: "2026-10-11T12:00:00.000Z" };
  const service = { repository: { brainDirectory: () => directory }, listToolPermissions: async () => [permission], authorizeInlineImagination: async () => undefined } as unknown as BrainService;
  const executor = new ToolExecutor(service, {} as RuntimeJobManager);
  vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const clock = new ClockService({ brainDirectory: async () => directory,
    authority: async () => ({ enabled: permission.level !== "off", revision: clockHash(JSON.stringify({ toolId: permission.toolId, level: permission.level, updatedAt: permission.updatedAt })) }),
    loaded: async () => false, busy: () => false, admissionOpen: () => true, turnStatus: async () => ({ state: "absent" }),
    wake: async () => { throw new Error("No model in this bridge fixture"); }, notify: () => undefined });
  executor.setClockService(clock);
  const completed = vi.fn(async () => undefined);
  const invocation: ToolInvocation = { brainId: "brain-one", toolId: "studio.schedule", action: "create", arguments: { label: "Reviewed wake", kind: "timer", delayMs: 1000, delivery: "wake", task: "Review the selected report.", loadWhenDue: false, requestKey: "exact-create" } };
  return { directory, executor, clock, invocation, completed, permission(level: ToolPermissionRecord["level"]) { permission = { ...permission, level, updatedAt: new Date(Date.parse(permission.updatedAt) + 1000).toISOString() }; } };
}
describe("permissioned clock tools and durable authorized intent", () => {
  it("Ask approval binds the exact Load option and records an actual saved outcome, not a fabricated tool success", async () => {
    const f = await fixture();
    const pending = await f.executor.execute(f.invocation, undefined, undefined, f.completed); expect(pending.state).toBe("approval-required");
    expect((await f.clock.schedules("brain-one")).total).toBe(0); expect(f.completed).not.toHaveBeenCalled();
    const changed = { ...f.invocation, arguments: { ...f.invocation.arguments, loadWhenDue: true }, approvalToken: pending.approvalToken };
    const second = await f.executor.execute(changed, undefined, undefined, f.completed); expect(second.state).toBe("approval-required");
    const actual = await f.executor.execute({ ...changed, approvalToken: second.approvalToken }, undefined, "approved-clock-turn", f.completed);
    expect(actual.state).toBe("complete"); expect(actual.dispatchStarted).toBe(true); expect(f.completed).toHaveBeenCalledExactlyOnceWith(actual);
    expect((await f.clock.schedules("brain-one")).entries[0]?.loadWhenDue).toBe(true);
    const intentDirectory = join(f.directory, "engine", "operational-tool-intents");
    const paths = await readdir(intentDirectory); expect(paths).toHaveLength(1);
    const intent = JSON.parse(await readFile(join(intentDirectory, paths[0]!), "utf8"));
    expect(intent).toMatchObject({ state: "authorized-before-side-effects", toolId: "studio.schedule", action: "create", permission: "ask", requestId: "approved-clock-turn" });
    expect(intent.argumentSha256).toBe(clockHash(JSON.stringify(changed.arguments)));
  });
  it("Off/revision changes refuse stale tokens before clock dispatch or file intent publication", async () => {
    const f = await fixture(), dispatch = vi.spyOn(f.clock, "execute");
    const pending = await f.executor.execute(f.invocation, undefined, undefined, f.completed);
    f.permission("off"); const denied = await f.executor.execute({ ...f.invocation, approvalToken: pending.approvalToken }, undefined, undefined, f.completed);
    expect(denied.state).toBe("failed"); expect(dispatch).not.toHaveBeenCalled(); expect(f.completed).not.toHaveBeenCalled();
    f.permission("ask"); const renewed = await f.executor.execute({ ...f.invocation, approvalToken: pending.approvalToken }, undefined, undefined, f.completed);
    expect(renewed.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
  });
});
