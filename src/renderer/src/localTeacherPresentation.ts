import type { LocalTeacherGeneration, LocalTeacherJob } from "../../shared/localTeacher";
export type TeacherGenerationDraft = Record<keyof LocalTeacherGeneration, string>;
export const teacherGenerationDraft = (value: LocalTeacherGeneration): TeacherGenerationDraft => Object.fromEntries(Object.entries(value).map(([key, item]) => [key, String(item)])) as TeacherGenerationDraft;
export function teacherGenerationFromDraft(draft: TeacherGenerationDraft): LocalTeacherGeneration {
  const result = {} as LocalTeacherGeneration;
  for (const key of ["contextTokens", "maxOutputTokens", "threads", "timeoutMs", "maxOutputBytes", "seed", "temperature"] as const) {
    const raw = draft[key].trim(), value = Number(raw);
    const minimum = key === "seed" || key === "temperature" || key === "timeoutMs" ? 0 : 1;
    const maximum = key === "temperature" ? 2 : key === "seed" ? 0xffffffff : 0x7fffffff;
    if (!raw || !Number.isFinite(value) || value < minimum || value > maximum || (key !== "temperature" && !Number.isSafeInteger(value))) throw new Error(`Invalid ${teacherGenerationLabel(key).toLowerCase()}.`);
    result[key] = value;
  }
  if (result.maxOutputTokens >= result.contextTokens) throw new Error("Teacher output must leave room within its context window.");
  return result;
}
export function teacherGenerationLabel(key: keyof LocalTeacherGeneration): string {
  return { contextTokens: "Teacher context tokens", maxOutputTokens: "Output tokens per lesson", threads: "CPU threads", temperature: "Sampling temperature", seed: "Seed", timeoutMs: "Deadline milliseconds · 0 = none", maxOutputBytes: "Output byte envelope" }[key];
}
export function splitTeacherLessons(value: string): string[] { return value.split(/^[ \t]*---[ \t]*\r?$/m).map(text => text.trim()).filter(Boolean); }
export function mergeTeacherJobs(current: LocalTeacherJob[], incoming: LocalTeacherJob[], brainId: string): LocalTeacherJob[] {
  const values = new Map(current.filter(value => value.brainId === brainId).map(value => [value.id, value]));
  for (const item of incoming) if (item.brainId === brainId && (!values.has(item.id) || values.get(item.id)!.revision <= item.revision)) values.set(item.id, item);
  return [...values.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || a.id.localeCompare(b.id));
}
export const teacherJobActive = (value: LocalTeacherJob): boolean => ["queued", "running", "waiting-load"].includes(value.state);
