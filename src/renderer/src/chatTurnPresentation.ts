import type {
  ChatDeliveryReceiptState,
  ChatMessage
} from "../../shared/types";

/**
 * A renderer submission generation advances for every normal or Steer turn.
 * Results from an older cancelled request must never clear, restore, or commit
 * presentation state owned by a newer turn.
 */
export function isCurrentChatSubmission(
  submissionGeneration: number,
  currentGeneration: number
): boolean {
  return submissionGeneration === currentGeneration;
}

/**
 * Failed neural work is not committed as chat history. Keep any newer draft,
 * or return the failed submission to the composer when it is still the newest
 * turn so the user can retry without reconstructing their message.
 */
export function recoverFailedChatDraft(
  currentDraft: string,
  submittedText: string
): string {
  return currentDraft.trim() ? currentDraft : submittedText;
}

/**
 * Clear only the draft that was actually submitted. A cancellation, voice
 * shutdown, or slow IPC acknowledgement may settle after the person has
 * already started typing the next thought; that newer wording belongs to the
 * composer and must never be erased by the older turn.
 */
export function clearSubmittedChatDraft(
  currentDraft: string,
  submittedText: string
): string {
  return currentDraft.trim() === submittedText.trim() ? "" : currentDraft;
}

/** Keep the interrupted human direction visible when a Steer turn replaces
 * its in-flight response. It is presentation-only and no longer looks busy. */
export function preserveSteeredChatMessage(
  messages: ChatMessage[],
  replacedTurnId: string
): ChatMessage[] {
  const candidateIds = new Set([
    `pending-${replacedTurnId}`,
    `queued-${replacedTurnId}`
  ]);
  return messages.map((message) =>
    candidateIds.has(message.id)
      ? { ...message, id: `steered-${replacedTurnId}` }
      : message
  );
}

export type OptimisticChatTerminalState = "failed" | "cancelled";

/**
 * A terminal renderer/RPC failure must settle the local receipt, not erase the
 * person's visible turn. The id prefix is presentation-only: authoritative
 * history reconciliation still keys on the original timestamp and content.
 */
export function settleOptimisticChatMessage(
  messages: ChatMessage[],
  optimisticId: string,
  state: OptimisticChatTerminalState
): ChatMessage[] {
  return messages.map((message) =>
    message.id === optimisticId
      ? { ...message, id: `${state}-${message.id}` }
      : message
  );
}

/** Settle either an immediately sent or previously queued receipt by turn id. */
export function settleOptimisticChatTurn(
  messages: ChatMessage[],
  turnId: string,
  state: OptimisticChatTerminalState
): ChatMessage[] {
  const candidateIds = new Set([`pending-${turnId}`, `queued-${turnId}`]);
  return messages.map((message) =>
    candidateIds.has(message.id)
      ? { ...message, id: `${state}-${message.id}` }
      : message
  );
}

/** Remove presentation receipts as soon as an authoritative human message
 * with the same text and creation time arrives from the atomic chat commit. */
export function reconcileOptimisticChatMessages(
  optimistic: ChatMessage[],
  authoritative: ChatMessage[]
): ChatMessage[] {
  const committedTurnIds = new Set(
    authoritative
      .filter((message) =>
        message.role === "human" &&
        message.deliveryReceipt?.presentationOnly !== true &&
        message.turnId
      )
      .map((message) => message.turnId)
  );
  const committed = new Set(
    authoritative
      .filter(
        (message) =>
          message.role === "human" &&
          message.deliveryReceipt?.presentationOnly !== true
      )
      .map((message) => `${message.createdAt}\u0000${message.content}`)
  );
  return optimistic.filter((message) =>
    !committed.has(`${message.createdAt}\u0000${message.content}`) &&
    !committedTurnIds.has(optimisticTurnId(message))
  );
}

/** A successful result may timestamp its human row in the worker, not at the
 * renderer click. Once that exact row is retained locally, retire only the
 * submitted optimistic id; timestamp matching alone would duplicate it. */
export function reconcileCompletedChatSubmission(
  optimistic: ChatMessage[],
  submittedOptimisticId: string,
  committedHuman: ChatMessage
): ChatMessage[] {
  return reconcileOptimisticChatMessages(
    optimistic.filter((message) => message.id !== submittedOptimisticId),
    [committedHuman]
  );
}

export function chatMessageDeliveryState(
  message: ChatMessage
): ChatDeliveryReceiptState | "pending" | undefined {
  if (message.deliveryReceipt?.presentationOnly) {
    return message.deliveryReceipt.state;
  }
  if (message.id.startsWith("failed-")) return "failed";
  if (message.id.startsWith("cancelled-")) return "cancelled";
  if (message.id.startsWith("steered-")) return "steered";
  if (message.id.startsWith("queued-")) return "queued";
  if (message.id.startsWith("pending-")) return "pending";
  return undefined;
}

function optimisticTurnId(message: ChatMessage): string | undefined {
  if (message.deliveryReceipt?.presentationOnly) {
    return message.deliveryReceipt.turnId;
  }
  const normalized = message.id.replace(/^(?:failed-|cancelled-)/, "");
  const prefix = ["pending-", "queued-", "steered-"].find((value) =>
    normalized.startsWith(value)
  );
  return prefix ? normalized.slice(prefix.length) : undefined;
}

function messageIdentity(message: ChatMessage): string {
  return `${message.createdAt}\u0000${message.content}`;
}

/**
 * Collapse append-only receipt revisions and merge them with current-session
 * optimistic bubbles. A committed neural human message supersedes its display
 * receipt, while an uncommitted queued/steered/cancelled/failed receipt stays
 * visible after navigation or restart.
 */
export function mergeChatMessagesForPresentation(
  authoritative: ChatMessage[],
  optimistic: ChatMessage[]
): ChatMessage[] {
  const committedTurnIds = new Set(
    authoritative
      .filter((message) =>
        message.role === "human" &&
        message.deliveryReceipt?.presentationOnly !== true &&
        message.turnId
      )
      .map((message) => message.turnId)
  );
  const committed = new Set(
    authoritative
      .filter(
        (message) =>
          message.role === "human" &&
          message.deliveryReceipt?.presentationOnly !== true
      )
      .map(messageIdentity)
  );
  const latestReceiptByTurn = new Map<string, ChatMessage>();
  const receiptPriority: Record<ChatDeliveryReceiptState, number> = {
    queued: 0,
    pending: 1,
    steered: 2,
    cancelled: 3,
    failed: 3
  };
  for (const message of authoritative) {
    const receipt = message.deliveryReceipt;
    if (!receipt?.presentationOnly) continue;
    const previous = latestReceiptByTurn.get(receipt.turnId);
    const previousReceipt = previous?.deliveryReceipt;
    if (
      !previous ||
      !previousReceipt ||
      receiptPriority[receipt.state] > receiptPriority[previousReceipt.state] ||
      (
        receiptPriority[receipt.state] === receiptPriority[previousReceipt.state] &&
        Date.parse(receipt.updatedAt) >= Date.parse(previousReceipt.updatedAt)
      )
    ) {
      latestReceiptByTurn.set(receipt.turnId, message);
    }
  }
  const retainedAuthoritative = authoritative.filter((message) => {
    const receipt = message.deliveryReceipt;
    if (!receipt?.presentationOnly) return true;
    return latestReceiptByTurn.get(receipt.turnId)?.id === message.id &&
      !committedTurnIds.has(receipt.turnId) &&
      !committed.has(messageIdentity(message));
  });
  const durableTurns = new Set(latestReceiptByTurn.keys());
  const retainedOptimistic = optimistic.filter((message) => {
    if (committed.has(messageIdentity(message))) return false;
    const turnId = optimisticTurnId(message);
    return !turnId || (
      !durableTurns.has(turnId) && !committedTurnIds.has(turnId)
    );
  });
  return [...retainedAuthoritative, ...retainedOptimistic];
}

/** A chat result is authoritative even when a separately loaded ledger page
 * is stale or unavailable. Keep its exact messages without duplicating rows
 * already present in that page. */
export function mergeCommittedChatMessages(
  persisted: ChatMessage[],
  sessionCommitted: ChatMessage[]
): ChatMessage[] {
  const seen = new Set(persisted.map((message) => message.id));
  return [
    ...persisted,
    ...sessionCommitted.filter((message) => {
      if (seen.has(message.id)) return false;
      seen.add(message.id);
      return true;
    })
  ];
}

export type ComposerSubmitIntent = "none" | "newline" | "send" | "queue" | "steer";

export type ComposerResponsePhase =
  | "idle"
  | "loading"
  | "generating"
  | "reply-complete-learning"
  | "action-result-learning";

export interface ComposerTurnCapabilities {
  turnActive: boolean;
  queueAvailable: boolean;
  steerAvailable: boolean;
}

/**
 * Steering can only change a response that has not finished generating.
 * Post-reply learning still owns the serial runtime, but new input is
 * presented as an ordinary Send and serialized internally without exposing a
 * misleading Queue or Steer action.
 */
export function composerTurnCapabilities(
  phase: ComposerResponsePhase
): ComposerTurnCapabilities {
  if (phase === "idle") {
    return {
      turnActive: false,
      queueAvailable: false,
      steerAvailable: false
    };
  }
  return {
    turnActive: true,
    // Queue and Steer are choices only while visible output can still change.
    // Once that output is complete, the composer presents an ordinary Send;
    // the runtime still serializes it behind the atomic learning/save commit.
    queueAvailable: phase === "loading" || phase === "generating",
    steerAvailable: phase === "loading" || phase === "generating"
  };
}

/** Keyboard semantics stay explicit and testable across Windows and macOS. */
export function composerSubmitIntent(input: {
  key: string;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  turnActive: boolean;
  queueAvailable: boolean;
  steerAvailable: boolean;
}): ComposerSubmitIntent {
  if (input.key !== "Enter") return "none";
  if (input.shiftKey) return "newline";
  if (
    input.turnActive &&
    input.steerAvailable &&
    (input.ctrlKey || input.metaKey)
  ) return "steer";
  return input.turnActive && input.queueAvailable ? "queue" : "send";
}

/**
 * Streaming output follows only while the reader remains near the end. This
 * prevents each token or action update from dragging someone away from older
 * messages they deliberately scrolled back to inspect.
 */
export function shouldFollowChatOutput(
  scrollTop: number,
  clientHeight: number,
  scrollHeight: number,
  tolerance = 72
): boolean {
  if (![scrollTop, clientHeight, scrollHeight, tolerance].every(Number.isFinite)) {
    return false;
  }
  return scrollHeight - (Math.max(0, scrollTop) + Math.max(0, clientHeight)) <=
    Math.max(0, tolerance);
}
