export type PonderSelfReportPhase = "pre-speech" | "mid-generation";

export interface PonderSelfReport {
  format: "omni-ponder-self-report";
  version: 1;
  id: string;
  actionId: string;
  phase: PonderSelfReportPhase;
  text: string;
  /** Publication custody only; never a claim that cognition or reasoning is complete. */
  state: "complete" | "partial";
  emittedBytes: number;
  textSha256?: string;
  modelEmitted: true;
  selfReportNotChainOfThought: true;
  hiddenPromptUsed: false;
  inferenceCompletedBeforePublication: true;
}

/** Whole neural argument inference completes first; only publication is framed. */
export interface PonderSelfReportDelta {
  format: "omni-ponder-self-report-delta";
  version: 1;
  reportId: string;
  actionId: string;
  phase: PonderSelfReportPhase;
  chunkIndex: number;
  byteOffset: number;
  delta: string;
  publicationComplete: boolean;
  totalBytes?: number;
  textSha256?: string;
  modelEmitted: true;
  selfReportNotChainOfThought: true;
  hiddenPromptUsed: false;
  inferenceCompletedBeforePublication: true;
}

const SHA = /^[a-f0-9]{64}$/;
const ACTION = /^[a-f0-9]{32}$/;
const record = (value: unknown): value is Record<string, unknown> => Boolean(value) && typeof value === "object" && !Array.isArray(value);
const DELTA_FIELDS = new Set(["format", "version", "reportId", "actionId", "phase", "chunkIndex", "byteOffset", "delta",
  "publicationComplete", "totalBytes", "textSha256", "modelEmitted", "selfReportNotChainOfThought", "hiddenPromptUsed", "inferenceCompletedBeforePublication"]);
const REPORT_FIELDS = new Set(["format", "version", "id", "actionId", "phase", "text", "state", "emittedBytes", "textSha256",
  "modelEmitted", "selfReportNotChainOfThought", "hiddenPromptUsed", "inferenceCompletedBeforePublication"]);
const utf8Bytes = (value: string): number => new TextEncoder().encode(value).byteLength;
export const ponderSelfReportIdentityInput = (actionId: string, phase: PonderSelfReportPhase): string =>
  `omni-ponder-self-report-v1\0${actionId}\0${phase}`;

export function normalizePonderSelfReportDelta(value: unknown): PonderSelfReportDelta | undefined {
  if (!record(value) || Object.keys(value).some(key => !DELTA_FIELDS.has(key)) || value.format !== "omni-ponder-self-report-delta" || value.version !== 1 ||
      typeof value.reportId !== "string" || !SHA.test(value.reportId) || typeof value.actionId !== "string" || !ACTION.test(value.actionId) ||
      !["pre-speech", "mid-generation"].includes(String(value.phase)) || !Number.isSafeInteger(value.chunkIndex) || Number(value.chunkIndex) < 0 ||
      !Number.isSafeInteger(value.byteOffset) || Number(value.byteOffset) < 0 || typeof value.delta !== "string" || !value.delta || value.delta.includes("\0") ||
      utf8Bytes(value.delta) > 64 * 1024 || typeof value.publicationComplete !== "boolean" || value.modelEmitted !== true ||
      value.selfReportNotChainOfThought !== true || value.hiddenPromptUsed !== false || value.inferenceCompletedBeforePublication !== true)
    return undefined;
  const complete = value.publicationComplete;
  if (complete !== (typeof value.totalBytes === "number" && typeof value.textSha256 === "string") ||
      complete && (!Number.isSafeInteger(value.totalBytes) || Number(value.totalBytes) < 1 || !SHA.test(String(value.textSha256))) ||
      !complete && (value.totalBytes !== undefined || value.textSha256 !== undefined)) return undefined;
  return structuredClone(value) as unknown as PonderSelfReportDelta;
}

export function validatePonderSelfReport(value: unknown, sha256?: (text: string) => string): PonderSelfReport {
  if (!record(value) || Object.keys(value).some(key => !REPORT_FIELDS.has(key)) || value.format !== "omni-ponder-self-report" || value.version !== 1 ||
      typeof value.id !== "string" || !SHA.test(value.id) || typeof value.actionId !== "string" || !ACTION.test(value.actionId) ||
      !["pre-speech", "mid-generation"].includes(String(value.phase)) || typeof value.text !== "string" || !value.text || value.text.includes("\0") ||
      !["complete", "partial"].includes(String(value.state)) || !Number.isSafeInteger(value.emittedBytes) || Number(value.emittedBytes) < 1 ||
      utf8Bytes(value.text) !== value.emittedBytes || value.modelEmitted !== true || value.selfReportNotChainOfThought !== true ||
      value.hiddenPromptUsed !== false || value.inferenceCompletedBeforePublication !== true)
    throw new Error("Ponder self-report is invalid.");
  if (value.state === "complete") {
    if (typeof value.textSha256 !== "string" || !SHA.test(value.textSha256))
      throw new Error("Completed Ponder self-report checksum is invalid.");
    if (sha256) {
      const digest = sha256(value.text);
      if (value.textSha256 !== digest) throw new Error("Completed Ponder self-report checksum is invalid.");
    }
  } else if (value.textSha256 !== undefined) throw new Error("Partial Ponder self-report cannot claim a complete text checksum.");
  return structuredClone(value) as unknown as PonderSelfReport;
}
