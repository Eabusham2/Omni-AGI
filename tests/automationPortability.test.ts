/** Operational mock receipts only; no neural training or real external actions. */
import { mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { AutomationService } from "../src/main/automationService";
import { AutomationStore } from "../src/main/automationStore";
import { mergeAutomationWorkspace, validateAutomationWorkspace } from "../src/main/automationPortability";
import { adoptLearningWorkspace, cloneLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { mergeClockWorkspace } from "../src/main/clockPortability";
import { ClockService } from "../src/main/clockService";
import { ClockStore } from "../src/main/clockStore";
import type { AutomationDefinition, AutomationProgram, AutomationRun, AutomationStepReceipt } from "../src/shared/automation";
import type { ScheduleRecord } from "../src/shared/clock";
import type { ToolExecutionResult } from "../src/shared/types";
import { DatabaseSync } from "node:sqlite";
const roots: string[] = [], services: AutomationService[] = [];
afterEach(async () => { for (const service of services.splice(0)) await service.checkpointBeforeQuit(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory() { const root = await mkdtemp(join(tmpdir(), "omni-action-copy-")); roots.push(root); return root; }
async function seed(root: string, owner = "source-brain") {
  let count = 0;
  const service = new AutomationService({ brainDirectory: async () => root, ownerEpoch: async () => "11111111-1111-4111-8111-111111111111", rotateOwnerEpoch: async () => "22222222-2222-4222-8222-222222222222", authority: async () => ({ enabled: true, revision: "a".repeat(64) }), admissionOpen: () => true,
    tools: { cancel: () => 0, cancelAndWait: async () => 0, invalidateApproval: () => false,
      execute: async (invocation, _progress, requestId, complete) => { count++; const actual: ToolExecutionResult = { id: requestId!, toolId: invocation.toolId, action: invocation.action, state: "complete", startedAt: "2026-10-11T12:00:00.000Z", dispatchStarted: true,
        output: { brainId: "this-user-key-is-opaque", nested: { runId: "also-user-data" }, private: "Private remembered task output", fixtureOnly: true } }; await complete?.(actual); return actual; } } });
  services.push(service);
  const value: AutomationProgram = { format: "omni-host-action-file", version: 1, name: "Private saved action", steps: [{ id: "counter", kind: "tool", toolId: "fixture.effect", action: "mock", arguments: { userData: "Private task input" } }] };
  const definition = await service.execute(owner, "save", { program: value }, "save-action") as AutomationDefinition;
  const active = await service.execute(owner, "activate", { definitionId: definition.id, expectedRevision: definition.revision, revisionSha256: definition.revisionSha256, allowTriggers: true }, "activate-action") as AutomationDefinition;
  const run = await service.execute(owner, "run", { definitionId: definition.id, revisionSha256: definition.revisionSha256, inputs: { private: "Unsanitized user inputs" } }, "run-action") as AutomationRun;
  await service.settle(); expect(count).toBe(1); return { service, definition: active, run, value, count: () => count };
}
describe("action workspace ownership and complete dormant portability", () => {
  it("exports/imports exact program/input/outcome bytes without inherited execution authority or source mutation", async () => {
    const source = await directory(), archiveRoot = await directory(), target = await directory(), seeded = await seed(source);
    const before = await readFile(join(source, "automation/workspace.sqlite3"));
    const archive = await snapshotLearningWorkspace(source, archiveRoot, "source-brain"); expect(archive).toBeDefined(); await adoptLearningWorkspace(archive!, target, "copied-brain");
    await validateAutomationWorkspace(target, "copied-brain"); expect(await readFile(join(source, "automation/workspace.sqlite3"))).toEqual(before);
    const store = await AutomationStore.open(target, "copied-brain", true), original = await AutomationStore.open(source, "source-brain", true);
    try {
      expect(store.get<AutomationDefinition>("definition", seeded.definition.id)?.activation?.enabled).toBe(false);
      expect(store.get<AutomationRun>("run", seeded.run.id)?.inputs).toEqual({ private: "Unsanitized user inputs" });
      expect(store.latestStep(seeded.run.id, 0)).toEqual(original.latestStep(seeded.run.id, 0));
      expect(await store.program(seeded.definition.revisionSha256)).toEqual(seeded.value);
      expect(store.db.prepare("SELECT count(*) AS n FROM requests").get()?.n).toBe(0); expect(store.db.prepare("SELECT count(*) AS n FROM inherited_requests").get()?.n).toBe(3);
    } finally { store.close(); original.close(); }
  });
  it("forks every row beyond the page window, preserving pending progress only for human review", async () => {
    const source = await directory(), target = await directory(), seeded = await seed(source), store = await AutomationStore.open(source, "source-brain");
    try {
      for (let index = 0; index < 129; index++) store.put({ ...seeded.definition, id: `definition-${index}`, activation: undefined }, "definition");
      store.put({ ...store.get<AutomationRun>("run", seeded.run.id)!, id: "unknown-run", state: "running", nextStep: 0 }, "run");
      store.putStep({ runId: "unknown-run", index: 0, stepId: "counter", attempt: 1, requestId: "unknown-effect", state: "intent", startedAt: seeded.run.createdAt, updatedAt: seeded.run.updatedAt });
    } finally { store.close(); }
    await cloneLearningWorkspace(source, target, "source-brain", "fork-brain"); const copy = await AutomationStore.open(target, "fork-brain", true);
    try { expect(copy.page("definition", { limit: 50 }).total).toBe(130); expect([...copy.records("definition")]).toHaveLength(130);
      expect(copy.get<AutomationRun>("run", "unknown-run")?.state).toBe("review-required"); expect(copy.latestStep("unknown-run", 0)?.state).toBe("intent"); } finally { copy.close(); }
    expect(seeded.count()).toBe(1);
  });
  it("merges identical ancestry idempotently and maps dormant Clock references without rewriting user output keys", async () => {
    const source = await directory(), stage = await directory(), target = await directory(), seeded = await seed(source);
    const clock = new ClockService({ brainDirectory: async () => source, authority: async () => ({ enabled: true, revision: "a".repeat(64) }), loaded: async () => false, busy: () => false, admissionOpen: () => true,
      turnStatus: async () => ({ state: "absent" }), wake: async () => ({ committed: false }), notify: () => undefined });
    const schedule = await clock.execute("source-brain", "studio.schedule", "create", { label: "Future explicit action", kind: "event", eventKey: "source.changed", delivery: "automation", automation: { definitionId: seeded.definition.id, revisionSha256: seeded.definition.revisionSha256, inputs: { private: "Exact trigger input" } } }, "new-schedule") as ScheduleRecord;
    await cloneLearningWorkspace(source, stage, "source-brain", "merged-brain");
    const aliases = await mergeAutomationWorkspace(stage, target, "merged-brain", "parent-one");
    await mergeClockWorkspace(stage, target, "merged-brain", "parent-one", undefined, aliases);
    await mergeAutomationWorkspace(stage, target, "merged-brain", "parent-one"); await mergeAutomationWorkspace(stage, target, "merged-brain", "parent-two");
    const store = await AutomationStore.open(target, "merged-brain", true), old = await AutomationStore.open(source, "source-brain", true), clocks = await ClockStore.open(target, "merged-brain", true);
    try {
      expect(store.page("definition").total).toBe(1); expect(store.page("run").total).toBe(1);
      const run = store.get<AutomationRun>("run", aliases.runs.get(seeded.run.id)!)!; expect(run.activation.enabled).toBe(false);
      expect(store.latestStep(run.id, 0)?.outcome).toEqual(old.latestStep(seeded.run.id, 0)?.outcome);
      const saved = [...clocks.records<ScheduleRecord>("schedule")][0]!; expect(saved.automation?.definitionId).toBe(aliases.definitions.get(schedule.automation!.definitionId)); expect(saved.pendingPaused).toBe(true);
    } finally { store.close(); old.close(); clocks.close(); clock.stop(); }
  });
  it("same-identity recovery retains command keys while later actual receipts remain inert history", async () => {
    const source = await directory(), restored = await directory(), seeded = await seed(source);
    await cloneLearningWorkspace(source, restored, "source-brain", "source-brain");
    const later = await seeded.service.execute("source-brain", "run", { definitionId: seeded.definition.id, revisionSha256: seeded.definition.revisionSha256, inputs: { later: true } }, "later-run") as AutomationRun; await seeded.service.settle();
    const aliases = await mergeAutomationWorkspace(source, restored, "source-brain", "recovery-proof");
    const store = await AutomationStore.open(restored, "source-brain", true);
    try { expect(store.db.prepare("SELECT count(*) AS n FROM requests").get()?.n).toBe(3);
      expect(store.get<AutomationRun>("run", aliases.runs.get(later.id)!)?.state).toBe("complete"); expect(store.get<AutomationRun>("run", aliases.runs.get(later.id)!)?.activation.enabled).toBe(false);
      expect(store.get<AutomationDefinition>("definition", seeded.definition.id)?.activation?.enabled).toBe(false); } finally { store.close(); }
    expect(seeded.count()).toBe(2);
  });
  it("rejects extra executable files, symlinks and corrupt receipt hashes instead of exporting a partial success", async () => {
    const source = await directory(); const seeded = await seed(source);
    await writeFile(join(source, "automation", "unknown.sh"), "fixture inert"); await expect(validateAutomationWorkspace(source, "source-brain")).rejects.toThrow(/unsupported payload/);
    await rm(join(source, "automation", "unknown.sh"));
    await symlink(join(source, "automation/workspace.sqlite3"), join(source, "automation/programs", "a".repeat(64) + ".omni-action.json"));
    await expect(validateAutomationWorkspace(source, "source-brain")).rejects.toThrow(/unlisted|unsafe/); await rm(join(source, "automation/programs", "a".repeat(64) + ".omni-action.json"));
    const store = await AutomationStore.open(source, "source-brain");
    try { const receipt = store.latestStep(seeded.run.id, 0)!; store.db.prepare("UPDATE steps SET payload=? WHERE run_id=?").run(JSON.stringify({ ...receipt, outcome: { ...receipt.outcome, output: "substituted" } }), seeded.run.id); } finally { store.close(); }
    await expect(validateAutomationWorkspace(source, "source-brain")).rejects.toThrow(/checksum/);
  });
  it("refuses an executable SQLite trigger before reading or adopting imported action records", async () => {
    const source = await directory(); await seed(source);
    const raw = new DatabaseSync(join(source, "automation/workspace.sqlite3"));
    try {
      raw.exec("CREATE TRIGGER fixture_untrusted AFTER UPDATE ON records BEGIN UPDATE meta SET value='foreign-owner' WHERE key='brainId'; END;");
      await expect(validateAutomationWorkspace(source, "source-brain")).rejects.toThrow(/unsupported executable\/schema/);
    } finally { raw.exec("DROP TRIGGER fixture_untrusted;"); raw.close(); }
    await validateAutomationWorkspace(source, "source-brain");
  });
});
