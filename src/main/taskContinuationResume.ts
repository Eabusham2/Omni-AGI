import type { NativeInheritedPresentation, NativeInstanceStatus } from "../shared/instanceLifecycleApi";
import type { InstanceResumeFrontier } from "../shared/instanceLifecycle";
import type { TaskContinuationResumeProof } from "./taskContinuationScope";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SHA = /^[a-f0-9]{64}$/;

/** Main-only: the caller first authenticates this native result against the current registered runtime/queue owner. */
export function verifiedTaskContinuationResume(expected: {
  brainId: string; frontier: InstanceResumeFrontier; input: string; inputSha256: string; inheritedExecutionId?: string;
}, observed: NativeInstanceStatus | NativeInheritedPresentation): TaskContinuationResumeProof {
  const inherited = "executionId" in observed ? observed : undefined;
  const frontier = inherited?.frontier ?? (observed as NativeInstanceStatus).resume;
  const presentation = observed.presentation;
  const raw = presentation?.inputReceipt;
  const receipt = raw && typeof raw === "object" && !Array.isArray(raw) ? raw as Record<string, unknown> : undefined;
  if (!ID.test(expected.brainId) || observed.brainId !== expected.brainId || !frontier ||
      !ID.test(frontier.checkpointId) || !ID.test(frontier.turnId) || !SHA.test(frontier.inputSha256) ||
      frontier.inputSha256 !== expected.inputSha256 || frontier.phase !== "native-text-token-boundary" || frontier.exactSupportedResume !== true ||
      (Object.keys(frontier) as Array<keyof InstanceResumeFrontier>).some(key => frontier[key] !== expected.frontier[key]) ||
      [frontier.lastStreamSequence, frontier.sampledTokens, frontier.publishedTokens].some(value => !Number.isSafeInteger(value) || value < 0) ||
      presentation?.input !== expected.input || receipt?.committed !== true || receipt.inputSha256 !== expected.inputSha256 ||
      receipt.turnId !== (inherited?.originalTurnId ?? frontier.turnId) ||
      !Number.isSafeInteger(receipt.attentionEpoch) || Number(receipt.attentionEpoch) < 0 ||
      Boolean(inherited) !== Boolean(expected.inheritedExecutionId) || inherited && (inherited.executionId !== expected.inheritedExecutionId ||
        !SHA.test(inherited.executionId) || !ID.test(inherited.parentId) || !ID.test(inherited.originalTurnId))) {
    throw new Error("The saved execution cannot authenticate this task file scope.");
  }
  return { brainId: expected.brainId, turnId: frontier.turnId, inputSha256: expected.inputSha256,
    checkpointId: frontier.checkpointId, attentionEpoch: Number(receipt.attentionEpoch),
    ...(inherited ? { inherited: { executionId: inherited.executionId, sourceBrainId: inherited.parentId,
      originalTurnId: inherited.originalTurnId } } : {}) };
}
