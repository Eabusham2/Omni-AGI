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

export type OptimisticChatTerminalState = "failed" | "cancelled" | "stopped";

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
  const candidateIds = new Set([`pending-${turnId}`, `queued-${turnId}`, `steered-${turnId}`]);
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

/** The ordered main stream correlates a saved result even for legacy message
 * rows without turnId or a renderer-matching timestamp. Retire only that turn. */
export function reconcileCompletedChatTurn(
  optimistic: ChatMessage[],
  turnId: string,
  committedHuman: ChatMessage
): ChatMessage[] {
  return reconcileOptimisticChatMessages(
    optimistic.filter((message) => optimisticTurnId(message) !== turnId),
    [committedHuman]
  );
}

export function chatMessageDeliveryState(
  message: ChatMessage
): ChatDeliveryReceiptState | "no-reply" | undefined {
  if (message.deliveryReceipt?.presentationOnly) {
    return message.deliveryReceipt.state;
  }
  if (message.generationEnd === "steered") return "steered";
  if (message.generationEnd === "native-stop") return "stopped";
  if (message.generationEnd === "no-reply") return "no-reply";
  if (message.id.startsWith("failed-")) return "failed";
  if (message.id.startsWith("cancelled-")) return "cancelled";
  if (message.id.startsWith("stopped-")) return "stopped";
  if (message.id.startsWith("steered-")) return "steered";
  if (message.id.startsWith("queued-")) return "queued";
  if (message.id.startsWith("pending-")) return "pending";
  return undefined;
}

/** UI metadata, never assistant prose or a claim about the model's intent. */
export function chatNoReplyPresentation(message: ChatMessage): { label: string; ariaLabel: string } | undefined {
  if (message.role !== "brain" || message.generationEnd !== "no-reply" || message.content !== "") return undefined;
  return {
    label: "No reply · input saved",
    ariaLabel: "Generation completed without printable assistant text. The human input and measured trace are saved."
  };
}

function optimisticTurnId(message: ChatMessage): string | undefined {
  if (message.deliveryReceipt?.presentationOnly) {
    return message.deliveryReceipt.turnId;
  }
  const normalized = message.id.replace(/^(?:failed-|cancelled-|stopped-)/, "");
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
    stopped: 3,
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
 * Generation controls end with the final output. Post-reply learning may
 * still own a serial neural transaction, but the next ordinary Send enters
 * main's safe ordering without exposing response Stop, Queue or Steer.
 */
export function composerTurnCapabilities(
  phase: ComposerResponsePhase
): ComposerTurnCapabilities {
  if (phase === "idle" || phase === "reply-complete-learning" ||
      phase === "action-result-learning") {
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
