import { learningHash } from "./learningTaskStore";
import type { FactCheckClaim, FactCheckWorkspace, LearningTask } from "../shared/learning";
import type { ChatResult } from "../shared/types";

/** Visible current user task data, never an injected persona/source retrieval. */
export function factCheckNativeTask(workspace: FactCheckWorkspace, claim: FactCheckClaim): { turnId: string; input: string } {
  return {
    turnId: learningHash({ task: workspace.task.id, claim: claim.id, policy: workspace.task.policy.sha256, answer: claim.answer?.id,
      evidence: claim.evidence.map(row => row.id) }),
    input: JSON.stringify({ kind: "user-requested-fact-check", taskId: workspace.task.id, claimId: claim.id,
      claim: claim.text, instructions: workspace.task.policy.instructions, constraints: workspace.task.policy.constraints,
      sourceReferences: claim.sources, capturedEvidence: claim.evidence, priorFindings: claim.findings, userAnswer: claim.answer })
  };
}

/** Validate the trusted controller's terminal commit, not answer correctness. */
export function committedNativeLearningTask(result: ChatResult, expected: { turnId: string; input: string }): { turnId: string; committedAt: string } {
  if (result.humanMessage.turnId !== expected.turnId || result.humanMessage.content !== expected.input ||
      result.humanMessage.status !== "complete" || result.brainMessage.turnId !== expected.turnId ||
      result.brainMessage.status !== "complete" || result.brainMessage.traceId !== result.trace.id ||
      result.generationEnd === "steered" || !Number.isFinite(Date.parse(result.brainMessage.createdAt))) {
    throw new Error("The native learning task has no matching terminal conversation receipt; its progress remains unconfirmed.");
  }
  return { turnId: expected.turnId, committedAt: result.brainMessage.createdAt };
}
export function specificLearnNativeTask(task: LearningTask): { turnId: string; input: string } {
  return {
    turnId: learningHash({ task: task.id, policy: task.policy.sha256, kind: "user-learning-instructions" }),
    input: JSON.stringify({ kind: "user-requested-specific-learning", taskId: task.id, instructions: task.policy.instructions,
      constraints: task.policy.constraints, sourceReceipts: task.sources })
  };
}
