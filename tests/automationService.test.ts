/** Real ToolExecutor + FULL-sync SQLite; the only external effect is a mock counter. */
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AutomationService } from "../src/main/automationService";
import { AutomationStore } from "../src/main/automationStore";
import { AutomationAuthorityRegistry } from "../src/main/automationAuthority";
import { automationCanonical, automationHash } from "../src/main/automationFormat";
import { ToolExecutor } from "../src/main/toolExecutor";
import { ClockService } from "../src/main/clockService";
import type { ScheduleRecord } from "../src/shared/clock";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { AutomationDefinition, AutomationProgram, AutomationRun, AutomationStepReceipt } from "../src/shared/automation";
import type { ToolInvocation, ToolPermissionRecord } from "../src/shared/types";

const cleanup: Array<() => Promise<void>> = [];
afterEach(async () => { for (const finish of cleanup.splice(0).reverse()) await finish(); });
const program = (steps: AutomationProgram["steps"]): AutomationProgram => ({ format: "omni-host-action-file", version: 1, name: "Owned fixture sequence", steps });
const effect = (id: string, args: Record<string, any> = {}): AutomationProgram["steps"][number] => ({ id, kind: "tool", toolId: "fixture.effect", action: "count", arguments: args });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-host-actions-")), directory = join(root, "brain-one"); await mkdir(directory);
  const permissions = new Map<string, ToolPermissionRecord>(["studio.automation", "studio.schedule", "fixture.effect"].map(toolId => [toolId, { toolId, label: toolId, level: "full", updatedAt: "2026-10-11T12:00:00.000Z" }]));
  const listeners: Array<(event: { brainId: string; toolId: string }) => void> = [];
  const calls: Record<string, unknown>[] = [];
  let blocked = false, release: (() => void) | undefined, resourceRefusal = false;
  const nativeGet = vi.fn(() => { throw new Error("Automation must not load a neural owner."); });
  const service = { repository: { brainDirectory: () => directory }, get: nativeGet,
    listToolPermissions: async () => [...permissions.values()], authorizeInlineImagination: async () => undefined,
    onToolPermissionChanged: (listener: (event: { brainId: string; toolId: string }) => void) => { listeners.push(listener); return () => undefined; } } as unknown as BrainService;
  const tools = new ToolExecutor(service, {} as RuntimeJobManager, undefined, undefined, {
    hasTool: id => id === "fixture.effect",
    execute: async (_tool, _action, args, signal) => {
      signal.throwIfAborted(); calls.push(structuredClone(args));
      if (blocked) await new Promise<void>((resolve, reject) => { release = resolve; signal.addEventListener("abort", () => reject(signal.reason), { once: true }); });
      signal.throwIfAborted(); return { count: calls.length, opaque: args.opaque ?? null };
    }
  });
  vi.spyOn(tools as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
  const epochs = new AutomationAuthorityRegistry(root), observations = vi.fn(async () => undefined);
  const actions = new AutomationService({ brainDirectory: async () => directory, tools,
    authority: async () => { const permission = permissions.get("studio.automation")!; return { enabled: permission.level !== "off", revision: automationHash(automationCanonical(permission)) }; },
    ownerEpoch: id => epochs.epoch(id), rotateOwnerEpoch: id => epochs.rotate(id), admissionOpen: () => true,
    checkResources: async (_id, _bytes, purpose) => { if (resourceRefusal && purpose === "automation step") throw new Error("Fixture RAM budget exhausted before dispatch."); }, observeOutcome: observations });
  tools.setAutomationService(actions);
  cleanup.push(async () => { release?.(); await actions.checkpointBeforeQuit(); await rm(root, { recursive: true, force: true }); });
  const invoke = (action: string, args: Record<string, unknown>, approvalToken?: string, requestId: string = randomUUID()) => tools.execute({ brainId: "brain-one", toolId: "studio.automation", action, arguments: args, ...(approvalToken ? { approvalToken } : {}) }, undefined, requestId, async () => undefined);
  const save = async (value: AutomationProgram) => { const result = await invoke("save", { program: value }); expect(result.state).toBe("complete"); return result.output as AutomationDefinition; };
  const run = async (definition: AutomationDefinition, inputs: Record<string, unknown> = {}) => { const result = await invoke("run", { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs }); expect(result.state).toBe("complete"); await actions.settle(); return (result.output as AutomationRun).id; };
  const humanConfirm = async (action: string, args: Record<string, unknown>) => { const prompt = await invoke(action, args); expect(prompt.state).toBe("approval-required"); return invoke(action, args, prompt.approvalToken); };
  return { root, directory, actions, tools, invoke, humanConfirm, save, run, calls, observations, nativeGet,
    async row(id: string) { return (await actions.runs("brain-one")).entries.find(row => row.id === id)!; },
    permission(toolId: string, level: ToolPermissionRecord["level"]) { const old = permissions.get(toolId)!; permissions.set(toolId, { ...old, level, updatedAt: new Date(Date.parse(old.updatedAt) + 1000).toISOString() }); for (const listener of listeners) listener({ brainId: "brain-one", toolId }); },
    block() { blocked = true; }, release() { blocked = false; release?.(); }, refuseResources() { resourceRefusal = true; } };
}
describe("independent permissioned host action files", () => {
  it("runs exact authored references and conditions; incoming data cannot turn into a new instruction", async () => {
    const f = await fixture(), opaque = { $automationRef: { source: "input", path: ["secret"] } };
    const definition = await f.save(program([effect("first", { opaque: { $automationRef: { source: "input", path: ["payload"] } } }),
      { ...effect("second", { prior: { $automationRef: { source: "step", stepId: "first", path: ["count"] } } }), when: { reference: { source: "input", path: ["enabled"] }, operator: "equals", value: true } },
      { ...effect("skip"), when: { reference: { source: "input", path: ["enabled"] }, operator: "equals", value: false } }]));
    const id = await f.run(definition, { payload: opaque, enabled: true, secret: "must-not-resolve" });
    expect((await f.row(id)).state).toBe("complete"); expect(f.calls).toEqual([{ opaque }, { prior: 1 }]);
    expect((await f.actions.steps("brain-one", id)).entries.map(row => row.state)).toEqual(["complete", "complete", "skipped"]);
    expect(f.observations).toHaveBeenCalledTimes(2); expect(f.nativeGet).not.toHaveBeenCalled();
    await f.actions.recover(["brain-one"]); f.actions.kick("brain-one", id); await f.actions.settle(); expect(f.calls).toHaveLength(2);
  });
  it("keeps Ask pending and refuses a stale approval after Stop", async () => {
    const f = await fixture(); f.permission("fixture.effect", "ask"); const definition = await f.save(program([effect("ask")])), id = await f.run(definition);
    expect((await f.row(id)).state).toBe("approval-required"); expect(f.calls).toHaveLength(0);
    const token = (await f.actions.steps("brain-one", id)).entries[0]!.outcome!.approvalToken!;
    const resumed = await f.invoke("resume", { runId: id }); expect(resumed.state).toBe("failed");
    await f.invoke("stop", { runId: id }); expect((await f.row(id)).state).toBe("stopped");
    const denied = await f.humanConfirm("approve", { runId: id, approvalToken: token }); expect(denied.state).toBe("failed"); expect(f.calls).toHaveLength(0);
  });
  it("uses exact one-use Ask approval for the nested effect and then continues", async () => {
    const f = await fixture(); f.permission("fixture.effect", "ask"); const definition = await f.save(program([effect("ask", { value: 7 })])), id = await f.run(definition);
    const token = (await f.actions.steps("brain-one", id)).entries[0]!.outcome!.approvalToken!;
    const dispatch = vi.spyOn(f.actions, "execute"), args = { runId: id, approvalToken: token };
    const outer = await f.invoke("approve", args); expect(outer.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
    const changed = await f.invoke("approve", { ...args, approvalToken: "substituted-token" }, outer.approvalToken); expect(changed.state).toBe("approval-required"); expect(dispatch).not.toHaveBeenCalled();
    expect((await f.humanConfirm("approve", args)).state).toBe("complete"); await f.actions.settle();
    expect((await f.row(id)).state).toBe("complete"); expect(f.calls).toEqual([{ value: 7 }]);
    expect((await f.humanConfirm("approve", args)).state).toBe("failed"); expect(f.calls).toHaveLength(1);
  });
  it("retains unknown interrupted effects for explicit human review, never Resume/restart replay", async () => {
    const f = await fixture(), definition = await f.save(program([{ id: "delay", kind: "wait", delayMs: 60_000 }, effect("uncertain")]));
    const id = await f.run(definition), store = await AutomationStore.open(f.directory, "brain-one");
    try { const run = store.get<AutomationRun>("run", id)!; const intent: AutomationStepReceipt = { runId: id, stepId: "uncertain", index: 1, attempt: 1, requestId: "old-effect", state: "intent", startedAt: run.createdAt, updatedAt: run.updatedAt };
      store.transaction(() => { store.putStep(intent); store.put({ ...run, state: "running", nextStep: 1 }, "run"); }); } finally { store.close(); }
    await f.actions.recover(["brain-one"]); expect((await f.row(id)).state).toBe("review-required");
    expect((await f.invoke("resume", { runId: id })).state).toBe("failed"); expect(f.calls).toHaveLength(0);
    const args = { runId: id, resolution: "retry-confirmed", acknowledgePossibleEffects: true, reason: "Fixture operator inspected the mock receipt and approves one NEW attempt." };
    const requestId = "review-current-attempt", asked = await f.invoke("review", args, undefined, requestId); expect(asked.state).toBe("approval-required");
    const accepted = await f.invoke("review", args, asked.approvalToken, requestId); expect(accepted.state).toBe("complete");
    await f.invoke("resume", { runId: id }); await f.actions.settle(); expect(f.calls).toHaveLength(1); expect((await f.row(id)).state).toBe("complete");
    const receipts = (await f.actions.steps("brain-one", id)).entries; expect(receipts.find(row => row.requestId === "old-effect")?.state).toBe("unknown"); expect(receipts.find(row => row.attempt === 2)?.review?.acknowledgedPossibleEffects).toBe(true);
  });
  it("permission revocation preserves the truthful partial effect and prevents every later step", async () => {
    const f = await fixture(); f.block(); const definition = await f.save(program([effect("started"), effect("never")]));
    const actual = await f.invoke("run", { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: {} }), id = (actual.output as AutomationRun).id;
    await vi.waitFor(() => expect(f.calls).toHaveLength(1)); f.permission("fixture.effect", "off"); await f.actions.settle();
    const receipt = (await f.actions.steps("brain-one", id)).entries[0]!; expect(receipt.state).toBe("failed"); expect(receipt.outcome?.dispatchStarted).toBe(true);
    expect((await f.row(id)).state).toBe("failed"); expect(f.calls).toHaveLength(1); expect(f.observations).toHaveBeenCalledTimes(1);
  });
  it("refuses the next effect before dispatch when its resource admission fails", async () => {
    const f = await fixture(), definition = await f.save(program([effect("no-budget")])); f.refuseResources(); const id = await f.run(definition);
    expect((await f.row(id)).state).toBe("review-required"); expect(f.calls).toHaveLength(0); expect((await f.actions.steps("brain-one", id)).total).toBe(0);
  });
  it("keeps immutable command idempotency and invalidates an old file activation after edits", async () => {
    const f = await fixture(), value = program([effect("first")]); const request = "save-once";
    const saved = await f.invoke("save", { program: value }, undefined, request), definition = saved.output as AutomationDefinition;
    expect((await f.invoke("save", { program: value }, undefined, request)).output).toEqual(definition);
    const activated = (await f.invoke("activate", { definitionId: definition.id, revisionSha256: definition.revisionSha256, expectedRevision: definition.revision, allowTriggers: true })).output as AutomationDefinition;
    const changed = (await f.invoke("save", { definitionId: definition.id, expectedRevision: activated.revision, program: program([effect("changed")]) })).output as AutomationDefinition;
    expect(changed.activation).toBeUndefined(); const stale = await f.invoke("activate", { definitionId: definition.id, revisionSha256: activated.revisionSha256, expectedRevision: activated.revision, allowTriggers: true }); expect(stale.state).toBe("failed");
    expect(f.calls).toHaveLength(0);
  });
  it("durably hands a typed event to one exact action run without notify, native status, Load or token decoding", async () => {
    const f = await fixture(), definition = await f.save(program([{ ...effect("threshold", { value: { $automationRef: { source: "event", path: ["value"] } } }),
      when: { reference: { source: "event", path: ["value"] }, operator: "at-least", value: 100 } }]));
    const active = (await f.invoke("activate", { definitionId: definition.id, expectedRevision: definition.revision, revisionSha256: definition.revisionSha256, allowTriggers: true })).output as AutomationDefinition;
    const turnStatus = vi.fn(async () => ({ state: "absent" as const })), load = vi.fn(async () => undefined), wake = vi.fn(async () => ({ committed: false })), notify = vi.fn();
    const clock = new ClockService({ brainDirectory: async () => f.directory, authority: async () => ({ enabled: true, revision: "c".repeat(64) }), loaded: async () => false, load, busy: () => true, admissionOpen: () => true, turnStatus, wake, notify });
    f.tools.setClockService(clock); clock.setAutomationDelegate(occurrence => f.actions.acceptOccurrence(occurrence));
    const execute = (action: string, args: Record<string, unknown>) => f.tools.execute({ brainId: "brain-one", toolId: "studio.schedule", action, arguments: args }, undefined, undefined, async () => undefined);
    const created = await execute("create", { label: "Fixture threshold event", kind: "event", eventKey: "fixture.value", delivery: "automation", automation: { definitionId: active.id, revisionSha256: active.revisionSha256, inputs: {} }, loadWhenDue: false }); expect(created.state).toBe("complete");
    expect((created.output as ScheduleRecord).task).toBe("");
    await execute("signal", { eventKey: "fixture.value", eventId: "observed-value-one", data: { value: 100 } });
    await clock.tick(); await clock.settle("brain-one"); await f.actions.settle();
    const occurrence = (await clock.occurrences("brain-one")).entries[0]!; expect(occurrence.state).toBe("delegated"); expect(occurrence.delegatedRunId).toBeDefined(); expect(f.calls).toEqual([{ value: 100 }]);
    const duplicate = await execute("signal", { eventKey: "fixture.value", eventId: "observed-value-one", data: { value: 100 } }); expect(duplicate.output).toMatchObject({ duplicate: true });
    f.permission("studio.automation", "off"); expect(await f.actions.acceptOccurrence(occurrence)).toMatchObject({ runId: occurrence.delegatedRunId }); await clock.tick(); await f.actions.settle();
    expect(f.calls).toHaveLength(1); expect(turnStatus).not.toHaveBeenCalled(); expect(load).not.toHaveBeenCalled(); expect(wake).not.toHaveBeenCalled(); expect(notify).not.toHaveBeenCalled();
    clock.stop();
  });
  it("recovers a lost Clock handoff ACK as the same completed run, never a repeated effect", async () => {
    const f = await fixture(), definition = await f.save(program([effect("once")]));
    await f.invoke("activate", { definitionId: definition.id, expectedRevision: definition.revision, revisionSha256: definition.revisionSha256, allowTriggers: true });
    let wall = Date.parse("2026-10-11T13:00:00Z"), mono = 0, loseAck = true;
    const clock = new ClockService({ brainDirectory: async () => f.directory, source: { wallMs: () => wall, monotonicMs: () => mono, sessionId: "fixture-clock" }, authority: async () => ({ enabled: true, revision: "c".repeat(64) }), loaded: async () => false, busy: () => false, admissionOpen: () => true,
      turnStatus: async () => { throw new Error("Not a native wake"); }, wake: async () => { throw new Error("Not a native wake"); }, notify: () => { throw new Error("Not a notification"); } });
    clock.setAutomationDelegate(async occurrence => { const receipt = await f.actions.acceptOccurrence(occurrence); if (loseAck) { loseAck = false; throw new Error("Fixture lost ACK after durable run creation"); } return receipt; });
    await clock.execute("brain-one", "studio.schedule", "create", { label: "One timer", kind: "timer", delayMs: 1, delivery: "automation", automation: { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: {} } }, "create-timer");
    wall++; mono++; await clock.tick(); await clock.settle("brain-one"); await f.actions.settle();
    const prior = (await clock.occurrences("brain-one")).entries[0]!; expect(prior.state).toBe("review-required"); expect(f.calls).toHaveLength(1);
    await clock.execute("brain-one", "studio.schedule", "resume-occurrence", { occurrenceId: prior.id }, "review-same-handoff");
    await clock.tick(); await clock.settle("brain-one"); await f.actions.settle();
    expect((await clock.occurrences("brain-one")).entries[0]?.state).toBe("delegated"); expect(f.calls).toHaveLength(1); expect((await f.actions.runs("brain-one")).total).toBe(1); clock.stop();
  });
  it("keeps an event action independent while the existing native wake delivery is still pending", async () => {
    const f = await fixture(), definition = await f.save(program([effect("independent")]));
    await f.invoke("activate", { definitionId: definition.id, expectedRevision: definition.revision, revisionSha256: definition.revisionSha256, allowTriggers: true });
    let wall = Date.parse("2026-10-11T13:00:00Z"), mono = 0, releaseWake!: () => void;
    const wake = vi.fn(async () => { await new Promise<void>(resolve => { releaseWake = resolve; }); return { committed: true }; });
    const clock = new ClockService({ brainDirectory: async () => f.directory, source: { wallMs: () => wall, monotonicMs: () => mono, sessionId: "fixture-clock" }, authority: async () => ({ enabled: true, revision: "c".repeat(64) }), loaded: async () => true, busy: () => false, admissionOpen: () => true,
      turnStatus: async () => ({ state: "absent" }), wake, notify: () => undefined });
    clock.setAutomationDelegate(occurrence => f.actions.acceptOccurrence(occurrence));
    await clock.execute("brain-one", "studio.schedule", "create", { label: "Long native placeholder", kind: "timer", delayMs: 1, delivery: "wake", task: "Fixture only, no native model." }, "wake-create");
    wall++; mono++; await clock.tick(); await vi.waitFor(() => expect(wake).toHaveBeenCalledTimes(1));
    try {
      await clock.execute("brain-one", "studio.schedule", "create", { label: "Host event", kind: "event", eventKey: "fixture.host", delivery: "automation", automation: { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: {} } }, "event-create");
      await clock.execute("brain-one", "studio.schedule", "signal", { eventKey: "fixture.host", eventId: "host-event-one", data: {} }, "host-signal");
      await vi.waitFor(() => expect(f.calls).toHaveLength(1)); await f.actions.settle();
      expect((await f.actions.runs("brain-one")).entries[0]?.state).toBe("complete"); expect(f.nativeGet).not.toHaveBeenCalled();
    } finally { releaseWake(); await clock.settle("brain-one"); clock.stop(); }
  });
  it("reads the authoritative late pending step without traversing old history pages", async () => {
    const f = await fixture(); f.permission("fixture.effect", "ask");
    const definition = await f.save(program([...Array.from({ length: 60 }, (_, index) => ({ id: `wait-${index}`, kind: "wait" as const, delayMs: 0 })), effect("late-approval")]));
    const id = await f.run(definition);
    await vi.waitFor(async () => expect((await f.row(id)).state).toBe("approval-required"), { timeout: 5000 });
    expect((await f.actions.steps("brain-one", id, { limit: 50 })).entries.some(row => row.stepId === "late-approval")).toBe(false);
    expect(await f.actions.currentStep("brain-one", id)).toMatchObject({ index: 60, stepId: "late-approval", state: "approval-required", invocation: { toolId: "fixture.effect" } });
    expect(f.calls).toHaveLength(0);
  });
  it("pauses a timed frontier on normal Quit and never repeats a preceding completed effect", async () => {
    const f = await fixture(), definition = await f.save(program([effect("once"), { id: "remaining-delay", kind: "wait", delayMs: 60_000 }, effect("not-yet")])), id = await f.run(definition);
    expect(f.calls).toHaveLength(1); expect((await f.row(id)).state).toBe("waiting");
    await f.actions.checkpointBeforeQuit(); expect((await f.row(id)).state).toBe("paused"); const remaining = (await f.actions.currentStep("brain-one", id))!.wait!.remainingMs; expect(remaining).toBeGreaterThan(0); expect(remaining).toBeLessThanOrEqual(60_000);
    f.actions.resumeAdmissions(); await f.invoke("resume", { runId: id }); await f.actions.settle();
    expect((await f.actions.currentStep("brain-one", id))?.wait?.remainingMs).toBeLessThanOrEqual(remaining); expect(f.calls).toHaveLength(1);
    await f.invoke("stop", { runId: id }); expect((await f.row(id)).state).toBe("stopped"); expect((await f.actions.steps("brain-one", id)).entries[0]?.state).toBe("complete");
  });
  it("publishes Pause before a finishing effect can start the next step", async () => {
    const f = await fixture(); f.block(); const definition = await f.save(program([effect("current"), effect("future")]));
    const actual = await f.invoke("run", { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: {}, requestKey: "one-run-ui-key" }), id = (actual.output as AutomationRun).id;
    await vi.waitFor(() => expect(f.calls).toHaveLength(1)); await f.invoke("pause", { runId: id, requestKey: "pause-ui-key" }); f.release(); await f.actions.settle();
    expect((await f.row(id)).state).toBe("paused"); expect((await f.row(id)).nextStep).toBe(1); expect(f.calls).toHaveLength(1);
    const recovered = await f.invoke("run", { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: {}, requestKey: "one-run-ui-key" });
    expect((recovered.output as AutomationRun).id).toBe(id); expect(f.calls).toHaveLength(1);
    await f.invoke("stop", { runId: id }); expect((await f.row(id)).state).toBe("stopped");
  });
});
