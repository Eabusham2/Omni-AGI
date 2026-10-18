import { createHash, randomUUID } from "node:crypto";
import { link, lstat, mkdir, open, rename, unlink, type FileHandle } from "node:fs/promises";
import { dirname, join } from "node:path";
import type { LocalTeacherItem, LocalTeacherJob, LocalTeacherResultReceipt, LocalTeacherSelection } from "../shared/localTeacher";
import { localTeacherCanonical, localTeacherHash, localTeacherId } from "./localTeacherFormat";

export function safeLocalTeacherPath(value: string): boolean {
  return value === "local-teachers/workspace.sqlite3" || /^local-teachers\/(?:archives\/[a-f0-9]{64}\.sqlite3|(?:attempts|probes)\/[A-Za-z0-9._-]+\/[A-Za-z0-9._/-]+|jobs\/[a-f0-9]{64}\/sources\/[0-9]+\.jsonl)$/.test(value) &&
    !value.includes("\\") && !value.includes("\0") && !value.split("/").some(part => !part || part === "." || part === "..");
}
export async function localTeacherDirectory(root: string, relative: string): Promise<string> {
  if (!/^local-teachers\//.test(relative) || relative.split("/").some(part => !part || part === "." || part === ".." || part.includes("\\") || part.includes("\0"))) throw new Error("Unsafe teacher artifact directory.");
  let path = root;
  for (const part of relative.split("/")) { path = join(path, part); await mkdir(path, { mode: 0o700 }).catch((error: NodeJS.ErrnoException) => { if (error.code !== "EEXIST") throw error; });
    const info = await lstat(path); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Teacher artifact custody contains a filesystem link."); }
  return path;
}
export async function localTeacherHashFile(path: string, signal?: AbortSignal): Promise<{ sha256: string; bytes: number }> {
  const item = await lstat(path); if (!item.isFile() || item.isSymbolicLink()) throw new Error("Teacher artifact is not a real regular file.");
  const file = await open(path, "r"), digest = createHash("sha256"), scratch = Buffer.alloc(65536); let offset = 0;
  try { for (;;) { signal?.throwIfAborted(); const row = await file.read(scratch, 0, scratch.length, offset); if (!row.bytesRead) break; digest.update(scratch.subarray(0, row.bytesRead)); offset += row.bytesRead; }
    const end = await file.stat(), atPath = await lstat(path); if (atPath.isSymbolicLink() || offset !== item.size ||
      [end, atPath].some(info => info.dev !== item.dev || info.size !== item.size || info.ino !== item.ino || info.mtimeMs !== item.mtimeMs || info.ctimeMs !== item.ctimeMs)) throw new Error("Teacher artifact changed during its exact hash.");
    return { sha256: digest.digest("hex"), bytes: offset };
  } finally { await file.close(); }
}
export async function writeTeacherText(file: FileHandle, text: string, escaped = false, signal?: AbortSignal): Promise<void> {
  for (let start = 0; start < text.length;) {
    signal?.throwIfAborted();
    let end = Math.min(text.length, start + 32768);
    if (end < text.length && text.charCodeAt(end - 1) >= 0xd800 && text.charCodeAt(end - 1) <= 0xdbff) end++;
    const part = text.slice(start, end), bytes = Buffer.from(escaped ? JSON.stringify(part).slice(1, -1) : part);
    await file.writeFile(bytes); start = end;
  }
}
export async function writeTeacherReceipt(root: string, relativePath: string, body: unknown): Promise<void> {
  if (!safeLocalTeacherPath(relativePath)) throw new Error("Unsafe teacher receipt path.");
  const text = `${localTeacherCanonical(body)}\n`, path = join(root, ...relativePath.split("/"));
  const old = await lstat(path).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (old) { const current = await localTeacherHashFile(path); if (current.sha256 !== localTeacherHash(text)) throw new Error("An immutable teacher receipt cannot be overwritten."); return; }
  const temporary = `${path}.${randomUUID()}.next`, file = await open(temporary, "wx", 0o600);
  try { await file.writeFile(text); await file.sync(); } finally { await file.close(); }
  try { await link(temporary, path); }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
    const current = await localTeacherHashFile(path); if (current.sha256 !== localTeacherHash(text)) throw new Error("Concurrent teacher receipt differed; the original was not overwritten."); }
  finally { await unlink(temporary); }
  await syncTeacherDirectory(dirname(path));
}
async function syncTeacherDirectory(path: string): Promise<void> {
  // Windows Node has no portable directory-fsync contract. Atomic rename and
  // SQLite FULL remain in use, without pretending a power-loss guarantee.
  if (process.platform === "win32") return; const directory = await open(path, "r"); try { await directory.sync(); } finally { await directory.close(); }
}
/** Stream complete paired dialogue; no giant JSON string, prompt clipping or QA lookup. */
export async function writeTeacherDialogueSource(root: string, job: LocalTeacherJob, item: LocalTeacherItem,
  teacher: LocalTeacherSelection, receipt: LocalTeacherResultReceipt, signal: AbortSignal): Promise<NonNullable<LocalTeacherItem["source"]>> {
  if (receipt.state !== "complete" || receipt.inputSha256 !== item.inputSha256 || receipt.runtimeSha256 !== teacher.runtime.sha256 || receipt.modelSha256 !== teacher.model.sha256) throw new Error("Only an exact completed teacher response is eligible for source learning.");
  const contentPath = join(root, ...receipt.stdout.relativePath.split("/"));
  const inputHash = await localTeacherHashFile(contentPath, signal); if (inputHash.sha256 !== receipt.stdout.sha256 || inputHash.bytes !== receipt.stdout.bytes || !inputHash.bytes) throw new Error("Teacher response source failed its hash/count.");
  const directory = await localTeacherDirectory(root, `local-teachers/jobs/${job.originalJobId}/sources`);
  const staging = await localTeacherDirectory(root, `local-teachers/attempts/${receipt.attemptId}`), temporary = join(staging, `source-${randomUUID()}.next`);
  const destination = join(directory, `${item.index.toString().padStart(job.promptCount.toString().length, "0")}.jsonl`);
  const file = await open(temporary, "wx", 0o600), response = await open(contentPath, "r");
  try {
    await file.writeFile('{"messages":[{"role":"user","content":"'); await writeTeacherText(file, item.prompt, true, signal);
    await file.writeFile('"},{"role":"assistant","content":"');
    const decoder = new TextDecoder("utf-8", { fatal: true }), buffer = Buffer.alloc(65536), actual = createHash("sha256"); let offset = 0, hasText = false;
    while (offset < inputHash.bytes) { signal.throwIfAborted(); const row = await response.read(buffer, 0, Math.min(buffer.length, inputHash.bytes - offset), offset); if (!row.bytesRead) throw new Error("Teacher response ended during dialogue formatting.");
      actual.update(buffer.subarray(0, row.bytesRead)); const text = decoder.decode(buffer.subarray(0, row.bytesRead), { stream: true }); hasText ||= Boolean(text.trim());
      await writeTeacherText(file, text, true, signal); offset += row.bytesRead; }
    const tail = decoder.decode(); hasText ||= Boolean(tail.trim()); await writeTeacherText(file, tail, true, signal);
    if (!hasText || actual.digest("hex") !== receipt.stdout.sha256) throw new Error("Teacher content changed or contained no nonempty learning text; no source was published.");
    const provenance = { format: "omni-local-gguf-teacher-dialogue", version: 1, originalBrainId: job.originalBrainId, originalJobId: job.originalJobId,
      promptIndex: item.index, inputSha256: item.inputSha256, teacherId: teacher.originalSelectionId, runner: teacher.runner,
      modelSha256: teacher.model.sha256, runtimeSha256: teacher.runtime.sha256, responseSha256: receipt.stdout.sha256,
      outputChannel: receipt.outputChannel, transportDecorationsPreserved: receipt.transportDecorationsPreserved,
      license: teacher.license, hiddenBehavioralPrompt: false, foreignWeightsImported: false, rewardModel: false, verifiedTruth: false };
    await file.writeFile(`"}],"teacherProvenance":${localTeacherCanonical(provenance)}}\n`); await file.sync();
  } finally { await response.close(); await file.close(); }
  const payload = await localTeacherHashFile(temporary, signal), present = await lstat(destination).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (present) { const old = await localTeacherHashFile(destination, signal); if (old.sha256 !== payload.sha256 || old.bytes !== payload.bytes) throw new Error("A teacher source already exists with different complete data."); await unlink(temporary); }
  else { await rename(temporary, destination); await syncTeacherDirectory(directory); }
  return { relativePath: `local-teachers/jobs/${job.originalJobId}/sources/${item.index.toString().padStart(job.promptCount.toString().length, "0")}.jsonl`, ...payload };
}
export const localTeacherAttemptId = (job: LocalTeacherJob, index: number): string => localTeacherHash(`${job.originalBrainId}\0${job.originalJobId}\0${index}\0${localTeacherId(randomUUID())}`);
