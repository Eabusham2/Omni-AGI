import { lstat, opendir, mkdir } from "node:fs/promises";
import { join, relative, sep } from "node:path";
import { snapshotMutableSqliteIsolated, copyMutableFileIsolated } from "./mutableFileIsolation";
import { localTeacherCanonical, localTeacherHash, localTeacherId, localTeacherSha } from "./localTeacherFormat";
import { localTeacherHashFile, safeLocalTeacherPath } from "./localTeacherArtifacts";
import { LocalTeacherStore } from "./localTeacherStore";
import { withLocalTeacherSnapshot } from "./localTeacherWorkspaceBoundary";
import type { LocalTeacherItem, LocalTeacherJob } from "../shared/localTeacher";

async function exists(path: string): Promise<boolean> { return lstat(path).then(() => true, (error: NodeJS.ErrnoException) => error.code === "ENOENT" ? false : Promise.reject(error)); }
async function *workspaceFiles(root: string, current = join(root, "local-teachers")): AsyncGenerator<{ path: string; relativePath: string }> {
  if (!await exists(current)) return;
  const info = await lstat(current); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Teacher workspace contains an unsafe directory.");
  const entries = await opendir(current);
  for await (const entry of entries) {
    const path = join(current, entry.name), item = await lstat(path), fromRoot = relative(root, path).split(sep).join("/");
    if (item.isSymbolicLink()) throw new Error("Teacher custody contains a link; external dependencies were not copied.");
    if (item.isDirectory()) { yield * workspaceFiles(root, path); continue; }
    if (!item.isFile()) throw new Error("Teacher custody contains an unsupported filesystem object.");
    if (["local-teachers/workspace.sqlite3-wal", "local-teachers/workspace.sqlite3-shm"].includes(fromRoot)) continue;
    if (!safeLocalTeacherPath(fromRoot)) throw new Error("Teacher workspace contains an unrecognized source path.");
    yield { path, relativePath: fromRoot };
  }
}
export async function validateLocalTeacherWorkspace(root: string, owner: string): Promise<void> {
  localTeacherId(owner);
  const store = await LocalTeacherStore.open(root, owner, false);
  if (!store) { if (await exists(join(root, "local-teachers"))) { for await (const _file of workspaceFiles(root)) throw new Error("Teacher artifacts have no ownership database."); } return; }
  try {
    if (store.db.prepare("PRAGMA quick_check").get()?.quick_check !== "ok") throw new Error("Teacher database integrity failed.");
    for (const selection of store.selections()) {
      if (selection.selection.currentHostAuthorized || selection.selection.runtime.sha256 !== selection.runtime.sha256 || selection.selection.model.sha256 !== selection.model.sha256)
        throw new Error("Teacher dependencies lost their immutable source identity or contain a copied live grant.");
    }
    for (const job of store.jobs()) {
      if (["running", "queued"].includes(job.state)) throw new Error("Pause local teacher work before snapshot; an active/uncertain writer was not copied.");
      const teacher = store.selection(job.teacherId).selection;
      const count = Number(store.db.prepare("SELECT COUNT(*) AS n FROM items WHERE job_id=?").get(job.id)?.n ?? 0);
      if (count !== job.promptCount) throw new Error("Teacher job lost prompt positions.");
      for (let index = 0; index < job.promptCount; index++) {
        const item = store.item(job.id, index), receipt = item.receipt;
        if (["generated", "learning", "complete"].includes(item.state) && !store.hasSealedReceipt(job.id, item)) throw new Error("Teacher source has no authentic dependency-custody seal.");
        if (receipt) {
          if (receipt.inputSha256 !== item.inputSha256 || receipt.runtimeSha256 !== teacher.runtime.sha256 || receipt.modelSha256 !== teacher.model.sha256 || receipt.hiddenSystemPrompt !== false || receipt.foreignWeightsImported !== false)
            throw new Error("Teacher result lost its exact declared source binding.");
          for (const descriptor of [receipt.stdout, receipt.stderr, receipt.rawTransport].filter(value => value !== undefined)) {
            if (!safeLocalTeacherPath(descriptor.relativePath)) throw new Error("Teacher receipt escaped artifact custody.");
            const actual = await localTeacherHashFile(join(root, descriptor.relativePath));
            if (actual.sha256 !== descriptor.sha256 || actual.bytes !== descriptor.bytes) throw new Error("Teacher result source hash/count changed.");
          }
        }
        if (item.source) { if (!safeLocalTeacherPath(item.source.relativePath)) throw new Error("Teacher dialogue source escaped its owned path.");
          const actual = await localTeacherHashFile(join(root, item.source.relativePath)); if (actual.sha256 !== item.source.sha256 || actual.bytes !== item.source.bytes) throw new Error("Teacher dialogue source changed."); }
      }
    }
    for (const row of store.db.prepare("SELECT id,sha,payload FROM session_seals").iterate()) {
      if (localTeacherHash(String(row.payload)) !== row.sha) throw new Error("Teacher session custody projection changed.");
      const file = await localTeacherHashFile(join(root, `local-teachers/attempts/${row.id}/custody.json`));
      if (file.sha256 !== localTeacherHash(`${row.payload}\n`)) throw new Error("Teacher session lost its original durable custody seal.");
    }
    for await (const file of workspaceFiles(root)) if (/\/archives\/[a-f0-9]{64}\.sqlite3$/.test(file.relativePath) &&
      (await localTeacherHashFile(file.path)).sha256 !== file.relativePath.split("/").at(-1)!.split(".")[0]) throw new Error("Teacher original database archive changed.");
  } finally { store.close(); }
}
export async function rekeyLocalTeacherWorkspace(root: string, sourceBrainId: string, targetBrainId: string): Promise<void> {
  if (!await exists(join(root, "local-teachers/workspace.sqlite3"))) return;
  // Existing sealed fields/hashes/outputs remain exact. Only current operational
  // projections change owner; no capture/runtime/permission grant is restored.
  const store = await LocalTeacherStore.open(root, sourceBrainId).catch(async sourceError => {
    try { await validateLocalTeacherWorkspace(root, targetBrainId); return undefined; } catch { throw sourceError; }
  });
  if (!store) return;
  try { store.transaction(() => {
    for (const row of store.selections()) { const next = { ...row, selection: { ...row.selection, brainId: targetBrainId, currentHostAuthorized: false } };
      store.db.prepare("UPDATE selections SET payload=? WHERE id=?").run(localTeacherCanonical(next), row.selection.id); }
    for (const job of store.jobs()) { const next: LocalTeacherJob = { ...job, brainId: targetBrainId, revision: job.revision + 1,
      ...(job.state === "complete" ? {} : { state: "dormant", priorState: job.state, dormantReason: "copied-instance" }),
      phase: job.state === "complete" ? job.phase : "retained-without-host-authority", updatedAt: new Date().toISOString() };
      store.db.prepare("UPDATE jobs SET payload=? WHERE id=?").run(localTeacherCanonical(next), job.id); }
    store.db.prepare("UPDATE meta SET value=? WHERE key='brainId'").run(localTeacherId(targetBrainId));
  }); } finally { store.close(); }
  await validateLocalTeacherWorkspace(root, targetBrainId);
}
/** All original rows/artifacts retained; merged pending work never auto-runs. */
export async function mergeLocalTeacherWorkspace(sourceRoot: string, targetRoot: string, owner: string, parentId: string,
  reserve: (bytes: number) => Promise<void>, manifestAliases = new Map<string, string>(), taskAliases = new Map<string, string>(),
  history: "merged-history" | "recovery-history" = "merged-history"): Promise<void> {
  if (!await exists(join(sourceRoot, "local-teachers/workspace.sqlite3"))) return;
  await withLocalTeacherSnapshot(sourceRoot, async () => {
    await validateLocalTeacherWorkspace(sourceRoot, owner); const source = (await LocalTeacherStore.open(sourceRoot, owner, false))!, target = (await LocalTeacherStore.open(targetRoot, owner))!;
    try {
      for await (const file of workspaceFiles(sourceRoot)) { if (file.relativePath === "local-teachers/workspace.sqlite3") continue;
        const payload = await localTeacherHashFile(file.path), destination = join(targetRoot, file.relativePath); await reserve(payload.bytes);
        await mkdir(join(destination, ".."), { recursive: true });
        if (await exists(destination)) { const actual = await localTeacherHashFile(destination); if (actual.sha256 !== payload.sha256) throw new Error("Merged teacher artifact lineage collided; no original was overwritten."); }
        else await copyMutableFileIsolated(file.path, destination, { expectedSha256: payload.sha256 });
      }
      const temporary = join(targetRoot, "local-teachers", `source-${localTeacherHash(parentId)}.next`);
      await snapshotMutableSqliteIsolated(join(sourceRoot, "local-teachers/workspace.sqlite3"), temporary);
      const archive = await localTeacherHashFile(temporary), archivePath = join(targetRoot, "local-teachers", "archives", `${archive.sha256}.sqlite3`);
      await reserve(archive.bytes); await mkdir(join(targetRoot, "local-teachers", "archives"), { recursive: true });
      await copyMutableFileIsolated(temporary, archivePath, { expectedSha256: archive.sha256 });
      // Exact temporary is ours, not a user-selected dependency.
      const { unlink } = await import("node:fs/promises"); await unlink(temporary);
      const priorImport = target.db.prepare("SELECT source_sha FROM merge_receipts WHERE parent_id=?").get(parentId);
      if (priorImport) { if (priorImport.source_sha !== archive.sha256) throw new Error("This parent teacher custody changed after its exact merge receipt; no history was silently appended twice."); return; }
      const teacherAliases = new Map<string, string>(), jobAliases = new Map<string, string>();
      target.transaction(() => {
        for (const row of source.selections()) {
          const id = localTeacherHash(`teacher\0${parentId}\0${row.selection.id}`); teacherAliases.set(row.selection.id, id);
          const next = { ...row, selection: { ...row.selection, id, brainId: owner, currentHostAuthorized: false }, requestId: localTeacherHash(`teacher-request\0${parentId}\0${row.requestId}`) };
          const existing = target.db.prepare("SELECT payload FROM selections WHERE id=?").get(id);
          if (existing) { if (String(existing.payload) !== localTeacherCanonical(next)) throw new Error("Merged teacher source projection collision."); }
          else target.putSelection(next);
        }
        for (const job of source.jobs()) {
          const id = localTeacherHash(`teacher-job\0${parentId}\0${job.id}`); jobAliases.set(job.id, id);
          if (target.db.prepare("SELECT 1 FROM jobs WHERE id=?").get(id)) continue;
          const next: LocalTeacherJob = { ...job, id, brainId: owner, teacherId: teacherAliases.get(job.teacherId)!, requestId: localTeacherHash(`teacher-job-request\0${parentId}\0${job.requestId}`),
            state: job.state === "complete" && history !== "recovery-history" ? "complete" : "dormant", priorState: job.state, dormantReason: history, phase: "retained-teacher-history-not-replayed",
            ...(job.manifestId ? { manifestId: manifestAliases.get(job.manifestId) ?? job.manifestId } : {}),
            ...(job.learningTaskId ? { learningTaskId: taskAliases.get(job.learningTaskId) ?? job.learningTaskId } : {}) };
          target.db.prepare("INSERT INTO jobs VALUES(?,?,?,?)").run(id, next.requestId, next.requestSha256, localTeacherCanonical(next));
          for (let index = 0; index < job.promptCount; index++) { const item: LocalTeacherItem = { ...source.item(job.id, index), jobId: id };
            target.db.prepare("INSERT INTO items VALUES(?,?,?)").run(id, index, localTeacherCanonical(item)); }
        }
        for (const seal of source.db.prepare("SELECT id,job_id,sha,payload FROM session_seals").iterate()) {
          const { id: sealId, job_id: sourceJobId, sha: sealSha, payload: sealPayload } = seal;
          if (typeof sealId !== "string" || typeof sourceJobId !== "string" || typeof sealSha !== "string" || typeof sealPayload !== "string")
            throw new Error("Merged teacher custody row lost a required typed binding.");
          localTeacherId(sealId); localTeacherId(sourceJobId); localTeacherSha(sealSha);
          const localJob = jobAliases.get(sourceJobId);
          if (!localJob || localTeacherHash(sealPayload) !== sealSha) throw new Error("Merged teacher custody has no exact job/source binding.");
          const current = target.db.prepare("SELECT sha FROM session_seals WHERE id=? AND job_id=?").get(sealId, localJob);
          if (current && current.sha !== sealSha) throw new Error("Merged custody seal collided with different source history.");
          target.db.prepare("INSERT OR IGNORE INTO session_seals VALUES(?,?,?,?)").run(sealId, localJob, sealSha, sealPayload);
        }
        for (const row of source.db.prepare("SELECT job_id,event,body FROM history ORDER BY sequence").iterate())
          target.event(jobAliases.get(String(row.job_id)) ?? teacherAliases.get(String(row.job_id)) ?? parentId, "inherited-original-event", { parentId, event: row.event, exactOriginalBody: row.body });
        target.db.prepare("INSERT INTO merge_receipts VALUES(?,?)").run(parentId, archive.sha256);
      });
    } finally { source.close(); target.close(); }
    await validateLocalTeacherWorkspace(targetRoot, owner);
  });
}
