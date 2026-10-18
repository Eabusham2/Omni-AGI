import { lstat, readdir } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import type { UserQuestionRecord } from "../shared/userQuestions";
import {
  readUserQuestionJson,
  userQuestionRequestSha256,
  validateUserQuestionPreferences,
  validateUserQuestionRecord,
  writeUserQuestionJson
} from "./userQuestions";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;
const RECORD_PATH = /^questions\/records\/[a-f0-9]{64}\.json$/;

export function safeUserQuestionPath(relative: string): boolean {
  return relative === "questions/preferences.json" || RECORD_PATH.test(relative);
}

/** Enumeration never starts timers or turns answers into live tool authority. */
export async function savedUserQuestionFiles(brainDirectory: string, brainId: string): Promise<Map<string, string>> {
  const files = new Map<string, string>();
  const root = join(brainDirectory, "questions");
  const info = await lstat(root).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
  if (!info) return files;
  if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Saved question directory is unsafe.");
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (entry.name === "preferences.json" && entry.isFile()) files.set("questions/preferences.json", join(root, entry.name));
    else if (entry.name === "records" && entry.isDirectory() && !entry.isSymbolicLink()) {
      for (const record of await readdir(join(root, "records"), { withFileTypes: true })) {
        if (record.isFile() && /^\.[a-f0-9]{64}\.json\.[a-f0-9-]{36}\.tmp$/.test(record.name)) continue;
        const relative = `questions/records/${record.name}`;
        if (!record.isFile() || !safeUserQuestionPath(relative)) throw new Error("Saved question entry is unsafe.");
        files.set(relative, join(root, "records", record.name));
      }
    } else if (entry.isFile() && /^\.preferences\.json\.[a-f0-9-]{36}\.tmp$/.test(entry.name)) continue;
    else throw new Error("Saved question storage contains an unsupported entry.");
  }
  await validateSavedUserQuestionFiles(files, brainId);
  return files;
}

export async function validateSavedUserQuestionFiles(files: ReadonlyMap<string, string>, brainId: string): Promise<void> {
  if (!SAFE_ID.test(brainId)) throw new Error("Invalid saved question instance.");
  for (const [relative, path] of files) {
    if (!safeUserQuestionPath(relative) || basename(path) !== relative.split("/").at(-1)) throw new Error("Unsafe saved question path.");
    const value = await readUserQuestionJson(path);
    if (relative === "questions/preferences.json") validateUserQuestionPreferences(value);
    else validateUserQuestionRecord(value, brainId, basename(path, ".json"));
  }
}

/**
 * Rebinds one copied record without changing the original resolution or its authority.
 * A copied instance remains dormant until the user explicitly loads/resumes it.
 */
export function rekeyUserQuestionRecord(
  value: unknown,
  sourceBrainId: string,
  destinationBrainId: string,
  copiedAt = new Date().toISOString()
): UserQuestionRecord {
  if (!SAFE_ID.test(sourceBrainId) || !SAFE_ID.test(destinationBrainId)) throw new Error("Invalid question copy identity.");
  if (!Number.isFinite(Date.parse(copiedAt)) || new Date(copiedAt).toISOString() !== copiedAt) throw new Error("Invalid question copy time.");
  const raw = value as Partial<UserQuestionRecord> | null;
  // A failed/cancelled import can retry its unpublished stage without appending parentage twice.
  if (raw?.brainId === destinationBrainId && (sourceBrainId !== destinationBrainId || raw.suspended?.fromBrainId === sourceBrainId)) {
    const rebound = validateUserQuestionRecord(value, destinationBrainId);
    const prior = rebound.lineage?.at(-1);
    if (!rebound.suspended || rebound.suspended.fromBrainId !== sourceBrainId || prior?.brainId !== sourceBrainId) {
      throw new Error("Question already belongs to a different copy operation.");
    }
    return rebound;
  }
  const source = validateUserQuestionRecord(value, sourceBrainId);
  const record: UserQuestionRecord = {
    ...source, brainId: destinationBrainId,
    lineage: [...(source.lineage ?? []), { brainId: sourceBrainId, questionId: source.id, copiedAt }],
    suspended: { reason: "copied-instance", fromBrainId: sourceBrainId, copiedAt }
  };
  record.requestSha256 = userQuestionRequestSha256(record);
  return validateUserQuestionRecord(record, destinationBrainId);
}

/**
 * Use only on an unpublished copy/import/snapshot staging directory. Each file is
 * atomically replaced; repository promotion makes the complete batch visible.
 * The source directory is never modified by this helper.
 */
export async function rekeySavedUserQuestions(
  copiedBrainDirectory: string,
  sourceBrainId: string,
  destinationBrainId: string,
  options: { copiedAt?: string; signal?: AbortSignal } = {}
): Promise<number> {
  if (!SAFE_ID.test(sourceBrainId) || !SAFE_ID.test(destinationBrainId)) throw new Error("Invalid question copy identity.");
  const root = join(copiedBrainDirectory, "questions");
  const records = join(root, "records");
  for (const directory of [copiedBrainDirectory, root, records]) {
    const info = await lstat(directory).catch((error: NodeJS.ErrnoException) => { if (error.code === "ENOENT") return undefined; throw error; });
    if (!info) {
      if (directory === records) await savedUserQuestionFiles(copiedBrainDirectory, sourceBrainId);
      return 0;
    }
    if (!info.isDirectory() || info.isSymbolicLink()) throw new Error("Question copy directory is unsafe.");
  }
  const copiedAt = options.copiedAt ?? new Date().toISOString();
  let count = 0;
  for (const entry of await readdir(records, { withFileTypes: true })) {
    options.signal?.throwIfAborted();
    if (entry.isFile() && /^\.[a-f0-9]{64}\.json\.[a-f0-9-]{36}\.tmp$/.test(entry.name)) continue;
    if (!entry.isFile() || !/^[a-f0-9]{64}\.json$/.test(entry.name)) throw new Error("Saved question entry is unsafe.");
    const path = join(records, entry.name);
    const raw = await readUserQuestionJson(path);
    const record = rekeyUserQuestionRecord(raw, sourceBrainId, destinationBrainId, copiedAt);
    if (record.id !== basename(path, ".json")) throw new Error("Question filename does not match its identity.");
    await writeUserQuestionJson(dirname(path), basename(path), record);
    count += 1;
  }
  await savedUserQuestionFiles(copiedBrainDirectory, destinationBrainId);
  return count;
}
