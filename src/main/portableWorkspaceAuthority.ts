import { lstat } from "node:fs/promises";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { rekeyClockWorkspace } from "./clockPortability";
import { rekeyAutomationWorkspace } from "./automationPortability";
import { rekeyEnvironmentWorkspace } from "./environmentPortability";
import { rekeyLocalTeacherWorkspace } from "./localTeacherPortability";
import { rekeyTaskAttachmentWorkspace } from "./taskAttachmentPortability";
import { rekeySteerDirectionWorkspace } from "./steerDirectionPortability";
import { rekeyImaginationWorkspace } from "./imaginationCaptureWorkspace";
import { rekeyReplayWorkspace } from "./imaginationAssemblyWorkspace";
import { rekeyExternalAdviceWorkspace } from "./externalAdvicePortability";
import { rekeyPluginRuntimeWorkspace } from "./pluginRuntimePortability";
import { rekeyPanelReceiptWorkspace } from "./pluginPanelReceiptPortability";
import { adoptSparseSubstrateTransferWorkspace } from "./sparseSubstrateTransferPortability";
import { rekeySavedUserQuestions } from "./userQuestionPortability";
import { rebindMcpAppArtifactOwner } from "./mcpAppArtifacts";
import { portableExists, portableFiles } from "./portableWorkspaceFiles";
import type { LearningTask } from "../shared/learning";
import { sqliteFilename } from "./sqliteFilename";

/** The destination is still unpublished to schedulers; no neural files are rewritten. */
export async function retirePortableWorkspaceAuthority(root: string, brainId: string,
  completed: readonly string[], checkpoint: (step: string) => Promise<void>, signal?: AbortSignal): Promise<void> {
  // A hard-linked mutable copy must not retire the still-live source's work.
  for await (const name of portableFiles(root)) if (/\.(?:json|sqlite3|sqlite3-wal|sqlite3-shm)$/.test(name)) {
    signal?.throwIfAborted();
    if ((await lstat(join(root, ...name.split("/")))).nlink > 1) throw new Error("Portable mutable state is hard-linked to another copy; make an independent copy before destination adoption.");
  }
  const steps: Array<[string, () => Promise<unknown>]> = [
    ["learning-tasks", async () => {
      const path = join(root, "learning", "tasks.sqlite3"); if (!await portableExists(path)) return;
      const database = new DatabaseSync(sqliteFilename(path));
      try {
        if (database.prepare("PRAGMA quick_check").get()?.quick_check !== "ok" || database.prepare("SELECT value FROM meta WHERE key='brainId'").get()?.value !== brainId) throw new Error("Portable learning tasks lost their source identity.");
        database.exec("BEGIN IMMEDIATE");
        for (const row of database.prepare("SELECT id,body FROM tasks").iterate()) {
          signal?.throwIfAborted();
          const task = JSON.parse(String(row.body)) as LearningTask; if (task.id !== row.id || task.brainId !== brainId) throw new Error("Portable learning task has a different owner.");
          if (["queued", "running", "waiting"].includes(task.state)) { task.state = "paused"; task.phase = "copied-instance"; task.label = "Saved work retained · review and resume when ready"; }
          delete task.pausedForUnload; database.prepare("UPDATE tasks SET body=? WHERE id=?").run(JSON.stringify(task), String(row.id));
        }
        for (const row of database.prepare("SELECT task_id,id,body FROM sources").iterate()) { const source = JSON.parse(String(row.body)); if (source.state === "running") source.state = "paused";
          database.prepare("UPDATE sources SET body=? WHERE task_id=? AND id=?").run(JSON.stringify(source), String(row.task_id), String(row.id)); }
        database.exec("COMMIT");
      } catch (error) { try { database.exec("ROLLBACK"); } catch { /* No transaction. */ } throw error; } finally { database.close(); }
    }],
    ["clock", () => rekeyClockWorkspace(root, brainId, brainId)],
    ["automation", () => rekeyAutomationWorkspace(root, brainId, brainId)],
    ["questions", () => rekeySavedUserQuestions(root, brainId, brainId)],
    ["mcp-apps", () => rebindMcpAppArtifactOwner(root, brainId, brainId)],
    ["task-attachments", () => rekeyTaskAttachmentWorkspace(root, brainId, brainId)],
    ["steer-directions", () => rekeySteerDirectionWorkspace(root, brainId, brainId)],
    ["imagination", () => rekeyImaginationWorkspace(root, brainId, brainId)],
    ["replay", () => rekeyReplayWorkspace(root, brainId, brainId)],
    ["environments", () => rekeyEnvironmentWorkspace(root, brainId, brainId)],
    ["local-teachers", () => rekeyLocalTeacherWorkspace(root, brainId, brainId)],
    ["external-advice", () => rekeyExternalAdviceWorkspace(root, brainId, brainId)],
    ["plugin-runtime", () => rekeyPluginRuntimeWorkspace(root, brainId, brainId)],
    ["panel-receipts", () => rekeyPanelReceiptWorkspace(root, brainId, brainId)],
    ["sparse-transfers", () => adoptSparseSubstrateTransferWorkspace(root, brainId, brainId)]
  ];
  for (const [name, action] of steps) if (!completed.includes(name)) { signal?.throwIfAborted(); await action(); await checkpoint(name); }
}
