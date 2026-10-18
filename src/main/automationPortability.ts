import { createHash } from "node:crypto";
import { lstat, opendir } from "node:fs/promises";
import { join } from "node:path";
import type { AutomationDefinition, AutomationRun, AutomationStepReceipt } from "../shared/automation";
import { automationCanonical, automationHash } from "./automationFormat";
import { AUTOMATION_RELATIVE_PATH, AutomationStore, validateAutomationDefinition, validateAutomationRun, validateAutomationStep } from "./automationStore";

const terminal = new Set(["complete", "stopped"]);
async function present(root: string): Promise<boolean> { return !!await lstat(join(root, AUTOMATION_RELATIVE_PATH)).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; }); }
export function safeAutomationWorkspacePath(name: string): boolean { return name === AUTOMATION_RELATIVE_PATH || /^automation\/programs\/[a-f0-9]{64}\.omni-action\.json$/.test(name); }
export async function validateAutomationWorkspace(root: string, brainId: string): Promise<void> {
  if (!await present(root)) return;
  const store = await AutomationStore.open(root, brainId, true);
  try {
    if (store.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Saved action database failed integrity verification.");
    for (const item of store.db.prepare("SELECT sha256 FROM programs ORDER BY sha256").iterate()) await store.program(String(item.sha256));
    const programs = await opendir(join(root, "automation", "programs"));
    for await (const item of programs) if (!item.isFile() || item.isSymbolicLink() || !/^[a-f0-9]{64}\.omni-action\.json$/.test(item.name) || !store.db.prepare("SELECT 1 FROM programs WHERE sha256=?").get(item.name.slice(0, 64))) throw new Error("Action workspace contains an unlisted or unsafe program.");
    const files = await opendir(join(root, "automation"));
    for await (const item of files) if (item.name !== "workspace.sqlite3" && item.name !== "programs") throw new Error("Action workspace contains an unsupported payload.");
    for (const row of store.records<AutomationDefinition>("definition")) { validateAutomationDefinition(row, brainId); await store.program(row.revisionSha256); }
    for (const row of store.records<AutomationRun>("run")) {
      validateAutomationRun(row, brainId); if (!store.get("definition", row.definitionId)) throw new Error("An action run lost its file owner.");
      const program = await store.program(row.revisionSha256);
      if (row.nextStep > program.steps.length) throw new Error("Action progress exceeds its exact saved program.");
      for (const item of store.db.prepare("SELECT payload,sha256 FROM steps WHERE run_id=? ORDER BY step_index,attempt").iterate(row.id)) {
        if (automationHash(String(item.payload)) !== item.sha256) throw new Error("Saved action-step checksum failed.");
        const step = validateAutomationStep(JSON.parse(String(item.payload)), row.id);
        if (program.steps[step.index]?.id !== step.stepId) throw new Error("Action receipt differs from its immutable program.");
      }
      let previous = "0".repeat(64), sequence = 0;
      for (const item of store.db.prepare("SELECT * FROM events WHERE run_id=? ORDER BY sequence").iterate(row.id)) {
        if (item.sequence !== ++sequence || item.previous_sha256 !== previous || automationHash(automationCanonical({ runId: row.id, sequence, kind: item.kind, body: item.payload, previousSha: previous })) !== item.sha256) throw new Error("Action receipt chain is inconsistent.");
        previous = String(item.sha256);
      }
    }
    if (store.db.prepare("SELECT 1 FROM steps s LEFT JOIN records r ON r.kind='run' AND r.id=s.run_id WHERE r.id IS NULL LIMIT 1").get() ||
        store.db.prepare("SELECT 1 FROM events e LEFT JOIN records r ON r.kind='run' AND r.id=e.run_id WHERE r.id IS NULL LIMIT 1").get()) throw new Error("Action evidence lost its run owner.");
  } finally { store.close(); }
}

/** Private copy only: historical receipts remain exact, but never carry activation. */
export async function rekeyAutomationWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  if (!await present(root)) return; await validateAutomationWorkspace(root, sourceBrainId);
  const store = await AutomationStore.open(root, sourceBrainId);
  try { store.transaction(() => {
    store.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(targetBrainId);
    const rewrite = (kind: string, row: AutomationDefinition | AutomationRun): void => {
      const body = automationCanonical(row); store.db.prepare("UPDATE records SET state=?,payload=?,sha256=? WHERE kind=? AND id=?")
        .run(kind === "definition" ? "dormant" : (row as AutomationRun).state, body, automationHash(body), kind, row.id);
    };
    for (const row of store.records<AutomationDefinition>("definition")) rewrite("definition", { ...row, brainId: targetBrainId,
      ...(row.activation ? { activation: { ...row.activation, enabled: false, allowTriggers: false, ownerBrainId: targetBrainId } } : {}),
      inheritedFrom: row.inheritedFrom ?? { brainId: sourceBrainId, definitionId: row.id } });
    for (const row of store.records<AutomationRun>("run")) rewrite("run", { ...row, brainId: targetBrainId, activation: { ...row.activation, enabled: false, allowTriggers: false, ownerBrainId: targetBrainId },
      inheritedFrom: row.inheritedFrom ?? { brainId: sourceBrainId, runId: row.id },
      ...(terminal.has(row.state) ? {} : { state: "review-required", pausedFor: sourceBrainId === targetBrainId ? "recovery" : "copied-instance", detail: "Historical work is retained without execution authority. Explicitly review any possible prior effects before a NEW attempt." }) });
    store.db.exec("CREATE TABLE IF NOT EXISTS inherited_requests(parent_id TEXT,id TEXT,arguments_sha256 TEXT,result TEXT,PRIMARY KEY(parent_id,id));");
    if (sourceBrainId !== targetBrainId) { store.db.prepare("INSERT OR IGNORE INTO inherited_requests SELECT ?,id,arguments_sha256,result FROM requests").run(sourceBrainId); store.db.exec("DELETE FROM requests;"); }
  }); } finally { store.close(); }
  await validateAutomationWorkspace(root, targetBrainId);
}

export interface AutomationMergeAliases { definitions: Map<string, string>; runs: Map<string, string> }
/** One merged owner, dormant operational projection; source files remain untouched. */
export async function mergeAutomationWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string): Promise<AutomationMergeAliases> {
  const aliases: AutomationMergeAliases = { definitions: new Map(), runs: new Map() };
  if (!await present(sourceRoot)) return aliases; await validateAutomationWorkspace(sourceRoot, owner);
  const source = await AutomationStore.open(sourceRoot, owner, true), target = await AutomationStore.open(targetRoot, owner);
  const normalized = (value: unknown): Record<string, unknown> => { const row = { ...(value as Record<string, unknown>) }; delete row.brainId; delete row.activation; delete row.inheritedFrom; return row; };
  const definitionBody = (value: AutomationDefinition): Record<string, unknown> => { const row = normalized(value); delete row.revision; delete row.updatedAt; return row; };
  try {
    for (const row of source.db.prepare("SELECT sha256 FROM programs ORDER BY sha256").iterate()) await target.saveProgram(await source.program(String(row.sha256)));
    target.transaction(() => {
      target.db.exec("CREATE TABLE IF NOT EXISTS automation_merge_aliases(parent_id TEXT,kind TEXT,source_id TEXT,target_id TEXT,PRIMARY KEY(parent_id,kind,source_id)); CREATE TABLE IF NOT EXISTS inherited_requests(parent_id TEXT,id TEXT,arguments_sha256 TEXT,result TEXT,PRIMARY KEY(parent_id,id)); CREATE TABLE IF NOT EXISTS inherited_event_chains(parent_id TEXT,run_id TEXT,sequence INTEGER,kind TEXT,payload TEXT,previous_sha256 TEXT,sha256 TEXT,PRIMARY KEY(parent_id,run_id,sequence));");
      const remember = (kind: string, id: string, mapped: string): void => { const previous = target.db.prepare("SELECT target_id FROM automation_merge_aliases WHERE parent_id=? AND kind=? AND source_id=?").get(parentId, kind, id);
        if (previous && previous.target_id !== mapped) throw new Error("The immutable parent action workspace changed during merge.");
        target.db.prepare("INSERT OR IGNORE INTO automation_merge_aliases VALUES(?,?,?,?)").run(parentId, kind, id, mapped); };
      for (const row of source.records<AutomationDefinition>("definition")) {
        const existing = target.get<AutomationDefinition>("definition", row.id);
        const id = existing && automationCanonical(definitionBody(existing)) === automationCanonical(definitionBody(row)) ? row.id
          : automationHash(automationCanonical({ kind: "definition", id: row.id, body: definitionBody(row) })); aliases.definitions.set(row.id, id); remember("definition", row.id, id);
        if (!target.get("definition", id)) target.put({ ...row, id, brainId: owner, revision: target.nextCatalogRevision(target.catalogRevision()), activation: undefined,
          inheritedFrom: { brainId: parentId, definitionId: row.id } }, "definition");
      }
      for (const row of source.records<AutomationRun>("run")) {
        const digest = createHash("sha256");
        for (const step of source.db.prepare("SELECT payload FROM steps WHERE run_id=? ORDER BY step_index,attempt").iterate(row.id)) digest.update(String(step.payload));
        const id = automationHash(automationCanonical({ kind: "run", id: row.id, body: normalized(row), steps: digest.digest("hex") })); aliases.runs.set(row.id, id); remember("run", row.id, id);
        if (target.get("run", id)) continue;
        const definitionId = aliases.definitions.get(row.definitionId); if (!definitionId) throw new Error("Merged action run lost its original program.");
        target.put({ ...row, id, brainId: owner, definitionId, activation: { ...row.activation, enabled: false, allowTriggers: false, ownerBrainId: owner },
          inheritedFrom: { brainId: parentId, runId: row.id }, ...(terminal.has(row.state) ? {} : { state: "review-required", pausedFor: "copied-instance", detail: "Merged historical work remains dormant; possible effects require explicit review." }) }, "run");
        for (const item of source.db.prepare("SELECT payload FROM steps WHERE run_id=? ORDER BY step_index,attempt").iterate(row.id)) {
          const step = JSON.parse(String(item.payload)) as AutomationStepReceipt; target.putStep({ ...step, runId: id }); // Original invocation/outcome bodies are not rewritten.
        }
        for (const item of source.db.prepare("SELECT * FROM events WHERE run_id=? ORDER BY sequence").iterate(row.id)) {
          target.db.prepare("INSERT INTO inherited_event_chains VALUES(?,?,?,?,?,?,?)").run(parentId, row.id, Number(item.sequence), String(item.kind), String(item.payload), String(item.previous_sha256), String(item.sha256));
          target.event(id, String(item.kind), JSON.parse(String(item.payload)));
        }
      }
      for (const row of source.db.prepare("SELECT * FROM requests").iterate()) target.db.prepare("INSERT OR IGNORE INTO inherited_requests VALUES(?,?,?,?)").run(parentId, String(row.id), String(row.arguments_sha256), String(row.result));
      if (source.db.prepare("SELECT name FROM sqlite_master WHERE type='table' AND name='inherited_requests'").get()) for (const row of source.db.prepare("SELECT * FROM inherited_requests").iterate())
        target.db.prepare("INSERT OR IGNORE INTO inherited_requests VALUES(?,?,?,?)").run(String(row.parent_id), String(row.id), String(row.arguments_sha256), String(row.result));
    });
  } finally { source.close(); target.close(); }
  await validateAutomationWorkspace(targetRoot, owner); return aliases;
}
