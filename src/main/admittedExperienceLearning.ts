import { createHash } from "node:crypto";

export interface AdmittedExperienceLearningReceipt {
  kind: "admitted-experience-learning";
  brainId: string;
  turnId: string;
  state: "queued" | "running" | "paused" | "complete" | "retained" | "unresolved" | "failed";
  admitted: boolean;
  jobId?: string;
  completedReplayPasses?: number;
  replayCursor?: { phase: "bos" | "text" | "eos" | "done"; character: number; byte: number };
  parameterChanged?: boolean;
  detail: string;
}
function rows(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter(row => row && typeof row === "object" && !Array.isArray(row)) : [];
}

/** Only bind a controller-owned turn to existing authenticated native metadata. */
export function admittedExperienceLearning(metadata: Record<string, unknown>, brainId: string, turnId: string): AdmittedExperienceLearningReceipt {
  const base = { kind: "admitted-experience-learning" as const, brainId, turnId };
  const receipt = rows(metadata.accepted_chat_inputs).find(row => row.turnId === turnId && row.committed === true)
    ?? rows(metadata.completed_chat_turns).find(row => row.turnId === turnId && typeof row.committedAt === "string");
  if (!receipt || typeof receipt.inputSha256 !== "string" || !/^[a-f0-9]{64}$/.test(receipt.inputSha256) || typeof receipt.humanMessageId !== "string") {
    return { ...base, state: "unresolved", admitted: false, detail: "No committed current experience is available; select an explicit learning source." };
  }
  const jobId = typeof receipt.slowJobId === "string" && /^[a-f0-9]{64}$/.test(receipt.slowJobId) ? receipt.slowJobId :
    createHash("sha256").update(`${brainId}\0${turnId}\0${receipt.inputSha256}\0chat-slow-learning-v1`).digest("hex");
  if (Array.isArray(metadata.completed_chat_slow_learning) && metadata.completed_chat_slow_learning.includes(jobId)) {
    return { ...base, jobId, state: "complete", admitted: true, detail: "The native replay receipt for this admitted experience is already committed." };
  }
  const pending = rows(metadata.pending_chat_slow_learning).find(row => row.jobId === jobId && row.turnId === turnId &&
    row.inputSha256 === receipt.inputSha256 && row.humanMessageId === receipt.humanMessageId);
  if (!pending) return { ...base, state: "retained", admitted: true,
    detail: "The experience was admitted to the brain's ordinary memory dynamics; those dynamics have not queued a slow replay job." };
  const cursor = pending.replayCursor as AdmittedExperienceLearningReceipt["replayCursor"] | undefined;
  if (!cursor || typeof cursor !== "object" || Object.keys(cursor).sort().join(",") !== "byte,character,phase" ||
      !["bos", "text", "eos", "done"].includes(cursor.phase) || !Number.isSafeInteger(cursor.character) || cursor.character < 0 ||
      !Number.isSafeInteger(cursor.byte) || cursor.byte < 0 || cursor.byte > 3 ||
      !Number.isSafeInteger(pending.completedReplayPasses) || Number(pending.completedReplayPasses) < 0) {
    return { ...base, jobId, state: "unresolved", admitted: true, detail: "The experience is admitted, but its native replay cursor failed validation." };
  }
  return { ...base, jobId, state: "queued", admitted: true,
    completedReplayPasses: Number(pending.completedReplayPasses), replayCursor: { ...cursor },
    detail: "Existing native replay prioritized; completion and parameter change remain unconfirmed." };
}
