import { createHash } from "node:crypto";
import { mkdtemp, mkdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ActionReviewer, type ActionReviewerHooks } from "../src/main/actionReviewer";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import { ToolExecutor, type CustomToolResolver } from "../src/main/toolExecutor";
import type { ToolInvocation, ToolPermissionLevel } from "../src/shared/types";
import type { PluginResolvedInvocation } from "../src/shared/plugins";
import { reviewTestClock, reviewTestConfiguration, observedAllow } from "./actionReviewerFixtures";

const flagged = { verdict: "ask", summary: "Review the exact observed proposal before proceeding.",
  evidence: [{ sourceId: "action.proposed", quote: "fixture.tool" }] };

describe("ToolExecutor unified optional-review authority", () => {
  let root: string, directory: string, executor: ToolExecutor, service: BrainService;
  let level: ToolPermissionLevel, revision: number, ownerConfiguration: ReturnType<typeof reviewTestConfiguration>;
  let dispatch: ReturnType<typeof vi.spyOn>, review: ReturnType<typeof vi.fn>;
  let permissionsChanged: ((event: { brainId: string; toolId: string }) => void) | undefined;
  const brainId = "review-brain", proposal: ToolInvocation = { brainId, toolId: "fixture.tool", action: "inspect", arguments: { value: "original" } };

  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-action-review-")); directory = join(root, brainId); await mkdir(directory);
    vi.useFakeTimers({ now: Date.parse("2026-10-13T12:00:00Z") });
    level = "full"; revision = 1; ownerConfiguration = reviewTestConfiguration(); permissionsChanged = undefined;
    service = { repository: { brainDirectory: () => directory },
      listToolPermissions: vi.fn(async () => [{ toolId: "fixture.tool", label: "Fixture", level, revision, updatedAt: "2026-10-13T12:00:00Z" },
        { toolId: "plugin.fixture.inspect", label: "Fixture plugin", level: "full", revision: 1, updatedAt: "2026-10-13T12:00:00Z" }]),
      authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined),
      onToolPermissionChanged: (listener: typeof permissionsChanged) => { permissionsChanged = listener; }
    } as unknown as BrainService;
    executor = makeExecutor(); review = vi.fn(async () => flagged);
  });
  afterEach(async () => { executor.cancel(brainId); vi.useRealTimers(); await rm(root, { recursive: true, force: true }); });

  function makeExecutor(custom?: CustomToolResolver): ToolExecutor {
    const value = new ToolExecutor(service, {} as RuntimeJobManager, undefined, { execute: vi.fn() }, undefined, undefined, custom);
    dispatch = vi.spyOn(value as unknown as { dispatch(...args: unknown[]): Promise<unknown> }, "dispatch").mockResolvedValue({ observed: "fixture-only effect" });
    vi.spyOn(value as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    return value;
  }
  function install(hooks: Partial<ActionReviewerHooks> = {}): ActionReviewer {
    const reviewer = new ActionReviewer({ configuration: () => ownerConfiguration, review, clock: reviewTestClock(), ...hooks });
    executor.setActionReviewer(reviewer); return reviewer;
  }
  async function run(value: ToolInvocation = proposal, requestId = "request-one") {
    const pending = executor.execute(value, undefined, requestId); await vi.advanceTimersByTimeAsync(1_000); return pending;
  }

  it("keeps default-disabled behavior and denies Off without consulting any reviewer", async () => {
    const disabled = new ActionReviewer({ review, clock: reviewTestClock() }); executor.setActionReviewer(disabled);
    expect((await executor.execute(proposal)).state).toBe("complete"); expect(review).not.toHaveBeenCalled();
    dispatch.mockClear(); install(); level = "off";
    expect((await executor.execute(proposal)).state).toBe("failed"); expect(review).not.toHaveBeenCalled(); expect(dispatch).not.toHaveBeenCalled();
  });

  it("uses one exact one-use approval for a flagged Full action, without an endless re-review", async () => {
    install(); const pending = await run();
    expect(pending).toMatchObject({ state: "approval-required", actionReview: { verdict: "ask", proposedInvocation: proposal } });
    expect(dispatch).not.toHaveBeenCalled(); expect(pending.output).toBeUndefined();
    const completed = await executor.execute({ ...proposal, approvalToken: pending.approvalToken });
    expect(completed).toMatchObject({ state: "complete", actionReview: { reviewId: pending.actionReview?.reviewId } });
    expect(review).toHaveBeenCalledOnce(); expect(dispatch).toHaveBeenCalledOnce();
    const reused = await run({ ...proposal, approvalToken: pending.approvalToken });
    expect(reused.state).toBe("approval-required"); expect(reused.approvalToken).not.toBe(pending.approvalToken); expect(dispatch).toHaveBeenCalledOnce();
  });

  it("never lets reviewer Allow override the existing Ask policy", async () => {
    level = "ask"; install({ review: async () => observedAllow }); const pending = await run();
    expect(pending).toMatchObject({ state: "approval-required", actionReview: { verdict: "allow" } }); expect(dispatch).not.toHaveBeenCalled();
    expect((await executor.execute({ ...proposal, approvalToken: pending.approvalToken })).state).toBe("complete"); expect(dispatch).toHaveBeenCalledOnce();
  });

  it("snapshots actual arguments before any asynchronous hold", async () => {
    install({ review: async () => observedAllow });
    const mutable = structuredClone(proposal), pending = executor.execute(mutable);
    mutable.arguments.value = "caller changed during hold";
    await vi.advanceTimersByTimeAsync(1_000); const result = await pending;
    expect(result).toMatchObject({ state: "complete", actionReview: { proposedInvocation: proposal } });
    expect(dispatch.mock.calls[0]?.[0]).toMatchObject(proposal);
  });

  it("rejects substituted arguments, owner, permission revision and reviewer grant revision", async () => {
    const reviewer = install(); const first = await run();
    const changed = await run({ ...proposal, arguments: { value: "different" }, approvalToken: first.approvalToken });
    expect(changed.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
    ownerConfiguration.ownerRevision = "new-device-owner"; reviewer.configurationChanged(brainId);
    const freshOwner = await run({ ...proposal, arguments: { value: "different" }, approvalToken: changed.approvalToken });
    expect(freshOwner.state).toBe("approval-required"); expect(freshOwner.actionReview?.ownerRevision).toBe("new-device-owner");
    revision += 1; permissionsChanged?.({ brainId, toolId: "fixture.tool" });
    const freshPolicy = await run({ ...proposal, arguments: { value: "different" }, approvalToken: freshOwner.approvalToken });
    expect(freshPolicy.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
    // Another owner cannot borrow the first brain's approval even under Full.
    const other = await run({ ...proposal, brainId: "different-brain", approvalToken: freshPolicy.approvalToken });
    expect(other.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
  });

  it("reviews resolved plugin operands while preserving the existing plugin approval output", async () => {
    const nested = structuredClone(proposal), resolved: PluginResolvedInvocation = { pluginId: "fixture", pluginName: "Fixture", contentSha256: "a".repeat(64),
      catalogRevision: 1, instanceRevision: 1, outerToolId: "plugin.fixture.inspect", outerAction: "run", nested,
      nestedArgumentsSha256: createHash("sha256").update(JSON.stringify(nested.arguments)).digest("hex"), capability: { toolId: nested.toolId, action: nested.action } };
    executor = makeExecutor({ resolve: async () => resolved }); install();
    const outer = { ...proposal, toolId: "plugin.fixture.inspect", action: "run", arguments: { selector: "fixture" } }, pending = await run(outer);
    expect(pending.output).toMatchObject({ kind: "plugin-nested-operation", resolved: { toolId: nested.toolId, arguments: nested.arguments } });
    expect(pending.actionReview).toMatchObject({ verdict: "ask", proposedInvocation: outer, resolvedInvocation: nested });
    expect((await executor.execute({ ...outer, approvalToken: pending.approvalToken })).state).toBe("complete");
    expect(dispatch.mock.calls[0]?.[0]).toEqual(nested);
  });

  it("bounds Stop during review and retires a pending review token without dispatch", async () => {
    install({ review: async () => new Promise(() => undefined) }); const active = executor.execute(proposal, undefined, "stop-request");
    await vi.advanceTimersByTimeAsync(50); expect(executor.cancel(brainId, "stop-request")).toBe(1);
    expect(await active).toMatchObject({ state: "failed" }); expect(dispatch).not.toHaveBeenCalled();
    install(); const pending = await run(proposal, "pending-request"); executor.cancel(brainId, "pending-request");
    expect((await run({ ...proposal, approvalToken: pending.approvalToken })).state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
  });

  it("rechecks policy and reviewer owner after the hold and before the effect", async () => {
    install({ review: async () => observedAllow });
    const pending = executor.execute(proposal); level = "off"; revision += 1;
    await vi.advanceTimersByTimeAsync(1_000); expect((await pending).state).toBe("failed"); expect(dispatch).not.toHaveBeenCalled();
    level = "full"; revision += 1;
    vi.mocked(service.authorizeInlineImagination).mockImplementation(async () => { ownerConfiguration.ownerRevision = "changed-after-intent"; });
    expect((await run()).state).toBe("failed"); expect(dispatch).not.toHaveBeenCalled();
  });

  it("does not review or replay an already committed authentic effect receipt", async () => {
    install(); const retained = { id: "stored-effect", toolId: proposal.toolId, action: proposal.action, state: "complete" as const,
      startedAt: "2026-10-13T10:00:00Z", output: { actual: "retained" } };
    service.recoverCompletedNativeEffect = vi.fn(async () => retained);
    const result = await executor.execute(proposal, undefined, "old-turn", undefined,
      { actionEventId: "old-event", chatTurnId: "old-turn", nativeActionId: "a".repeat(32) });
    expect(result).toBe(retained); expect(review).not.toHaveBeenCalled(); expect(service.listToolPermissions).not.toHaveBeenCalled(); expect(dispatch).not.toHaveBeenCalled();
  });

  it("binds approval to its exact trusted chat action rather than another same-args turn", async () => {
    install(); const firstContext = { actionEventId: "event-first", chatTurnId: "turn-first" };
    const first = executor.execute(proposal, undefined, "chat-first", undefined, firstContext);
    await vi.advanceTimersByTimeAsync(1_000); const original = await first;
    const secondContext = { actionEventId: "event-second", chatTurnId: "turn-second" };
    const second = executor.execute({ ...proposal, approvalToken: original.approvalToken }, undefined, "chat-second", undefined, secondContext);
    await vi.advanceTimersByTimeAsync(1_000); const current = await second;
    expect(current).toMatchObject({ state: "approval-required", actionReview: secondContext }); expect(dispatch).not.toHaveBeenCalled();
    expect((await executor.execute({ ...proposal, approvalToken: current.approvalToken }, undefined, "chat-second", undefined, secondContext)).state).toBe("complete");
    expect(review).toHaveBeenCalledTimes(2); expect(dispatch).toHaveBeenCalledOnce();
  });
});
