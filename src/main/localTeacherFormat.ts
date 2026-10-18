import { createHash } from "node:crypto";
import type { LocalTeacherGeneration, LocalTeacherStartRequest } from "../shared/localTeacher";
export const localTeacherCanonical = (value: unknown): string => {
  if (Array.isArray(value)) return `[${value.map(localTeacherCanonical).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, child]) => `${JSON.stringify(key)}:${localTeacherCanonical(child)}`).join(",")}}`;
  const text = JSON.stringify(value); if (text === undefined) throw new Error("Teacher records require explicit JSON."); return text;
};
export const localTeacherHash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
export const localTeacherId = (value: unknown): string => {
  if (typeof value !== "string" || !/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(value)) throw new Error("Invalid local-teacher owner/request ID."); return value;
};
export const localTeacherSha = (value: unknown): string => {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Invalid local-teacher content SHA-256."); return value;
};
export function localTeacherText(value: unknown): string {
  if (typeof value !== "string" || !value.trim() || value.includes("\0")) throw new Error("Teacher prompts must be complete nonempty Unicode text without NUL.");
  for (const point of value) if (point.length === 1 && point.charCodeAt(0) >= 0xd800 && point.charCodeAt(0) <= 0xdfff) throw new Error("Teacher text contains an invalid Unicode surrogate.");
  return value;
}
export function localTeacherGeneration(value: unknown): LocalTeacherGeneration {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Choose a typed local-teacher generation envelope.");
  const row = value as Record<string, unknown>;
  if (Object.keys(row).some(key => !["contextTokens", "maxOutputTokens", "threads", "temperature", "seed", "timeoutMs", "maxOutputBytes"].includes(key))) throw new Error("Arbitrary llama.cpp flags/setup are not accepted.");
  for (const name of ["contextTokens", "maxOutputTokens", "threads", "timeoutMs", "maxOutputBytes"]) if (!Number.isSafeInteger(row[name]) || Number(row[name]) < (name === "timeoutMs" ? 0 : 1) || Number(row[name]) > 0x7fffffff) throw new Error("Teacher token/thread/output values must be explicit positive supported integers; timeout0 means no wall-clock deadline.");
  if (!Number.isSafeInteger(row.seed) || Number(row.seed) < 0 || Number(row.seed) > 0xffffffff || typeof row.temperature !== "number" || !Number.isFinite(row.temperature) || row.temperature < 0 || row.temperature > 2 ||
      Number(row.maxOutputTokens) >= Number(row.contextTokens)) throw new Error("Invalid teacher seed/temperature or output/context envelope.");
  return { contextTokens: Number(row.contextTokens), maxOutputTokens: Number(row.maxOutputTokens), threads: Number(row.threads),
    temperature: row.temperature, seed: Number(row.seed), timeoutMs: Number(row.timeoutMs), maxOutputBytes: Number(row.maxOutputBytes) };
}
export function localTeacherRequest(value: LocalTeacherStartRequest): { request: LocalTeacherStartRequest; sha256: string; promptBytes: number } {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !["brainId", "requestId", "teacherId", "prompts", "generation", "epochs"].includes(key))) throw new Error("Invalid local-teacher request.");
  const brainId = localTeacherId(value.brainId), requestId = localTeacherId(value.requestId), teacherId = localTeacherId(value.teacherId), generation = localTeacherGeneration(value.generation);
  if (!Array.isArray(value.prompts) || !value.prompts.length) throw new Error("Select explicit teacher prompts.");
  const prompts = value.prompts.map(localTeacherText), digest = createHash("sha256"); let promptBytes = 0;
  for (const prompt of prompts) { const bytes = Buffer.byteLength(prompt); promptBytes += bytes; digest.update(`${bytes}:`).update(prompt); }
  const epochs = value.epochs ?? 1;
  if (!Number.isSafeInteger(promptBytes) || !Number.isSafeInteger(epochs) || epochs < 1) throw new Error("Teacher prompt/source accounting exceeds supported exact integers.");
  const body = { brainId, requestId, teacherId, generation, epochs, promptCount: prompts.length, promptsSha256: digest.digest("hex") };
  return { request: { brainId, requestId, teacherId, generation, epochs, prompts }, promptBytes,
    sha256: localTeacherHash(localTeacherCanonical(body)) };
}
