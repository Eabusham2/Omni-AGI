import type { EvolutionRun, EvolutionStartRequest } from "./types";
import { normalizeCompatibleArchitectureMutation } from "./architectureMutation";

export type NativeCandidateKind = "neural" | "data" | "substrate" | "architecture";
export type NativeCandidateToolState = "queued" | "waiting-load" | "waiting-boundary" | "running" | "stopping" | "complete" | "failed" | "cancelled" | "uncertain" | "dormant";
export interface NativeCandidateToolJob {
  format: "omni-native-candidate-tool-job";
  version: 1;
  id: string;
  brainId: string;
  requestId: string;
  requestSha256: string;
  candidateKind: NativeCandidateKind;
  /** Exact declared operands, before truthful main-owned source attribution. */
  arguments: Record<string, unknown>;
  request: EvolutionStartRequest;
  authorizationRevision: string;
  state: NativeCandidateToolState;
  createdAt: string;
  updatedAt: string;
  revision: number;
  runId?: string;
  candidateId?: string;
  run?: EvolutionRun;
  error?: string;
  reassessmentPending?: boolean;
  reassessmentError?: string;
  /** Historical request/result evidence, not a copied live grant. */
  liveAuthorityPersisted: false;
}
export const NATIVE_CANDIDATE_KINDS: readonly NativeCandidateKind[] = ["neural", "data", "substrate", "architecture"];
export function isNativeCandidateKind(value: unknown): value is NativeCandidateKind {
  return typeof value === "string" && NATIVE_CANDIDATE_KINDS.includes(value as NativeCandidateKind);
}
const fields = new Set(["candidateKind", "objective", "recursive", "parentCandidateId", "texts", "sourceIds", "epochs", "learningRate",
  "latentReplay", "objectives", "provenance", "architectureChange", "addExperts", "geometryHoldouts"]);
function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be typed data.`);
  return value as Record<string, unknown>;
}
function strings(value: unknown, label: string): string[] | undefined {
  if (value === undefined) return undefined;
  if (!Array.isArray(value) || value.some(item => typeof item !== "string" || item.includes("\0"))) throw new Error(`${label} must contain exact string values.`);
  return [...value] as string[];
}
function positive(value: unknown, label: string, integer = false): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0 || integer && !Number.isSafeInteger(value)) throw new Error(`${label} must be positive${integer ? " and a safe integer" : ""}.`);
  return value;
}
function jsonData(value: unknown, seen = new Set<unknown>()): void {
  if (value === null || typeof value === "string" || typeof value === "boolean") return;
  if (typeof value === "number" && Number.isFinite(value)) return;
  if (!value || typeof value !== "object" || seen.has(value) ||
      !Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype && Object.getPrototypeOf(value) !== null) {
    throw new Error("Native candidate operands must be finite, acyclic JSON data.");
  }
  seen.add(value);
  for (const item of Array.isArray(value) ? value : Object.values(value)) jsonData(item, seen);
  seen.delete(value);
}
/** Stable semantic binding without dropping, coercing or truncating operands. */
export function nativeCandidateArgumentsJson(value: Record<string, unknown>): string {
  jsonData(value);
  const ordered = (item: unknown): unknown => Array.isArray(item) ? item.map(ordered) :
    item && typeof item === "object" ? Object.fromEntries(Object.keys(item).sort().map(key => [key, ordered((item as Record<string, unknown>)[key])])) : item;
  return JSON.stringify(ordered(value));
}
/** Strict route validation: unsupported/source operands never fall into Git. */
export function nativeCandidateToolRequest(brainId: string, input: Record<string, unknown>): EvolutionStartRequest & { candidateKind: NativeCandidateKind } {
  nativeCandidateArgumentsJson(input);
  if (!isNativeCandidateKind(input.candidateKind)) throw new Error("Select an explicit native candidate kind.");
  if (Object.keys(input).some(key => !fields.has(key))) throw new Error("Unknown/native-incompatible candidate operand; source edits/worktree/tests are not a native proposal.");
  if (typeof input.objective !== "string" || !input.objective.trim() || input.objective.includes("\0")) throw new Error("An exact native candidate objective is required.");
  for (const field of ["recursive", "latentReplay"] as const) if (input[field] !== undefined && typeof input[field] !== "boolean") throw new Error(`${field} must be boolean.`);
  if (input.parentCandidateId !== undefined && (typeof input.parentCandidateId !== "string" || !input.parentCandidateId || input.parentCandidateId.includes("\0"))) throw new Error("Invalid native parent candidate identity.");
  const request: EvolutionStartRequest & { candidateKind: NativeCandidateKind } = { brainId, objective: input.objective, candidateKind: input.candidateKind };
  for (const field of ["texts", "sourceIds", "objectives"] as const) { const value = strings(input[field], field); if (value !== undefined) request[field] = value; }
  const epochs = positive(input.epochs, "epochs", true), learningRate = positive(input.learningRate, "learningRate");
  if (epochs !== undefined) request.epochs = epochs; if (learningRate !== undefined) request.learningRate = learningRate;
  if (input.recursive !== undefined) request.recursive = input.recursive as boolean;
  if (input.latentReplay !== undefined) request.latentReplay = input.latentReplay as boolean;
  if (input.parentCandidateId !== undefined) request.parentCandidateId = input.parentCandidateId as string;
  if (input.provenance !== undefined) request.provenance = structuredClone(record(input.provenance, "Declared provenance"));
  if (input.architectureChange !== undefined || input.addExperts !== undefined) {
    if (input.candidateKind !== "architecture") throw new Error("Architecture operands require an architecture candidate.");
    if (input.architectureChange !== undefined && input.addExperts !== undefined) throw new Error("Use one exact architectureChange or addExperts, not conflicting shortcuts.");
    request.architectureChange = normalizeCompatibleArchitectureMutation(input.architectureChange ?? { mutation: "grow-experts", addExperts: positive(input.addExperts, "addExperts", true) });
  }
  if (input.candidateKind === "architecture" && !request.architectureChange) throw new Error("A native architecture proposal requires an explicit typed change.");
  if (input.geometryHoldouts !== undefined) {
    if (input.candidateKind !== "architecture") throw new Error("Geometry holdouts require an architecture candidate.");
    const value = record(input.geometryHoldouts, "Geometry holdouts");
    if (Object.keys(value).some(key => !["token", "modality", "tool"].includes(key))) throw new Error("Unknown geometry holdout category.");
    const normalized: NonNullable<EvolutionStartRequest["geometryHoldouts"]> = { token: [], modality: [], tool: [] };
    for (const kind of ["token", "modality", "tool"] as const) {
      if (!Array.isArray(value[kind])) throw new Error(`Geometry ${kind} holdouts must be an array.`);
      for (const item of value[kind]) {
        const row = record(item, "Geometry holdout");
        const allowed = kind === "modality" ? ["path", "kind", "conditionText"] : ["path", "records"];
        if (Object.keys(row).some(key => !allowed.includes(key)) || typeof row.path !== "string" || !row.path || row.path.includes("\0")) throw new Error("Invalid geometry holdout source.");
        if (kind === "modality") {
          if (!["image", "audio", "video"].includes(String(row.kind)) || typeof row.conditionText !== "string" || row.conditionText.includes("\0")) throw new Error("Invalid geometry modality holdout.");
          normalized.modality.push({ path: row.path, kind: row.kind as "image" | "audio" | "video", conditionText: row.conditionText });
        } else {
          const records = positive(row.records, "holdout records", true);
          if (records === undefined) throw new Error("Geometry holdout records are required.");
          normalized[kind].push({ path: row.path, records });
        }
      }
    }
    request.geometryHoldouts = normalized;
  }
  if (request.candidateKind === "data" && !request.texts?.length && !request.sourceIds?.length && request.latentReplay !== true) throw new Error("A data candidate requires exact source evidence or explicit latent replay.");
  return request;
}

export function validateNativeCandidateToolJob(value: unknown): NativeCandidateToolJob {
  const job = value as NativeCandidateToolJob;
  if (!job || job.format !== "omni-native-candidate-tool-job" || job.version !== 1 || job.liveAuthorityPersisted !== false || !isNativeCandidateKind(job.candidateKind) ||
      !["queued", "waiting-load", "waiting-boundary", "running", "stopping", "complete", "failed", "cancelled", "uncertain", "dormant"].includes(job.state) ||
      !Number.isSafeInteger(job.revision) || job.revision < 1 || !Number.isFinite(Date.parse(job.createdAt)) || !Number.isFinite(Date.parse(job.updatedAt)) ||
      !/^[a-f0-9]{64}$/.test(job.requestSha256) || typeof job.authorizationRevision !== "string" || !job.authorizationRevision) throw new Error("Invalid saved native candidate job.");
  for (const id of [job.id, job.brainId, job.requestId, ...(job.runId ? [job.runId] : []), ...(job.candidateId ? [job.candidateId] : [])]) {
    if (typeof id !== "string" || /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.exec(id)?.[0] !== id) throw new Error("Invalid native candidate job identity.");
  }
  const declared = nativeCandidateToolRequest(job.brainId, record(job.arguments, "Saved native operands"));
  if (!job.request || job.request.brainId !== job.brainId || job.request.candidateKind !== job.candidateKind || declared.candidateKind !== job.candidateKind) throw new Error("Native candidate job request belongs to another owner/kind.");
  return job;
}
