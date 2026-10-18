import { lstat, opendir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { localTeacherCanonical, localTeacherHash } from "./localTeacherFormat";
import { localTeacherHashFile, safeLocalTeacherPath } from "./localTeacherArtifacts";
import type { LocalTeacherSessionSeal, LocalTeacherStore } from "./localTeacherStore";

/** Reconcile a fsynced, post-shutdown custody seal whose SQL ACK was lost.
 * No runtime execution, permission grant, new teacher response or neural work. */
export async function recoverLocalTeacherSeals(store: LocalTeacherStore, admit: (bytes: number) => Promise<void>): Promise<void> {
  const root = join(store.root, "local-teachers", "attempts"), present = await lstat(root).catch((error: NodeJS.ErrnoException) => {
    if (error.code === "ENOENT") return undefined; throw error;
  });
  if (!present) return; if (!present.isDirectory() || present.isSymbolicLink()) throw new Error("Teacher attempt custody is not an owned directory.");
  const directories = await opendir(root);
  for await (const directory of directories) {
    if (!directory.isDirectory() || directory.isSymbolicLink()) throw new Error("Teacher attempt custody contains an unsupported filesystem object.");
    if (!/^[a-f0-9]{64}$/.test(directory.name)) continue;
    const path = join(root, directory.name, "custody.json"), info = await lstat(path).catch((error: NodeJS.ErrnoException) => {
      if (error.code === "ENOENT") return undefined; throw error;
    });
    if (!info) continue; if (!info.isFile() || info.isSymbolicLink()) throw new Error("Teacher custody seal is not an owned regular file.");
    await admit(info.size * 8 + 65536); const exact = await localTeacherHashFile(path), text = await readFile(path, "utf8"), seal = JSON.parse(text) as LocalTeacherSessionSeal;
    if (localTeacherHash(text) !== exact.sha256 || `${localTeacherCanonical(seal)}\n` !== text || seal.sessionId !== directory.name)
      throw new Error("Teacher durable custody seal changed before reconciliation.");
    // A merged original may retain the seal solely as archived audit. Only
    // its exact current local job can consume it; dormant lineage is not run.
    const current = store.db.prepare("SELECT id FROM jobs WHERE id=?").get(seal.jobId); if (!current) continue;
    const known = store.db.prepare("SELECT sha FROM session_seals WHERE id=? AND job_id=?").get(seal.sessionId, seal.jobId);
    if (known) { if (known.sha !== localTeacherHash(localTeacherCanonical(seal))) throw new Error("Teacher sealed SQL/source binding disagrees."); continue; }
    if (!Array.isArray(seal.receipts)) throw new Error("Teacher custody candidate positions are invalid.");
    for (const row of seal.receipts) {
      const item = store.item(seal.jobId, row.index), receipt = item.receipt;
      if (!receipt || item.attemptId !== row.attemptId || receipt.state !== "complete" || localTeacherHash(localTeacherCanonical(receipt)) !== row.sha256)
        throw new Error("Post-shutdown teacher seal has no exact retained candidate receipt.");
      for (const source of [receipt.stdout, receipt.stderr, receipt.rawTransport].filter(value => value !== undefined)) {
        if (!safeLocalTeacherPath(source.relativePath)) throw new Error("Teacher seal refers outside exact owned custody.");
        const actual = await localTeacherHashFile(join(store.root, source.relativePath));
        if (actual.sha256 !== source.sha256 || actual.bytes !== source.bytes) throw new Error("Teacher candidate source changed before lost-ACK reconciliation.");
      }
    }
    store.sealSession(seal);
  }
}
