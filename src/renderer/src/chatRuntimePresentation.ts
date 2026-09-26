import type {
  ChatCancellationState,
  ChatQueueState
} from "../../shared/types";

export function chatQueueStatus(queue: ChatQueueState): string {
  const position = Math.max(1, Math.floor(queue.position));
  return `Message queued #${position} behind ${queue.queuedBehind.label}. Stop cancels only this message.`;
}

export function chatCancellationStatus(
  cancellation: ChatCancellationState | undefined,
  priorQueue: ChatQueueState | null
): string {
  if (cancellation?.unrelatedActivityContinues) {
    const owner = priorQueue?.queuedBehind.label ?? "The activity ahead of it";
    return `Queued message cancelled. ${owner} continues independently.`;
  }
  if (
    cancellation?.phase === "running" &&
    cancellation.workerTerminationAcknowledged
  ) {
    return "Response stopped after the neural worker acknowledged termination.";
  }
  if (cancellation?.phase === "running") {
    return "Stopping response · waiting for neural worker acknowledgement.";
  }
  return "Reply cancelled. Your message remains visible and can be retried.";
}
