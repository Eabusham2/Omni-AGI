import type { ChatQueueState } from "../../shared/types";

export interface PendingChatOutputPresentation {
  phase: "queued" | "loading" | "streaming";
  label: string;
  ariaLabel: string;
  outputTokens: number;
  elapsedMs?: number;
}

function safeCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : undefined;
}

function safeTimestamp(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function elapsedCopy(elapsedMs: number): string {
  const seconds = Math.floor(elapsedMs / 1_000);
  if (seconds < 60) return `${seconds}s elapsed`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return `${minutes}m ${remainder.toString().padStart(2, "0")}s elapsed`;
}

/**
 * Present only measured chat-output state. This intentionally has no percent,
 * shard, or stage-progress field: the renderer knows elapsed wait and emitted
 * output tokens, but it cannot infer how far the neural worker is through a
 * model load or generation.
 */
export function pendingChatOutputPresentation(input: {
  outputTokens: number;
  startedAtMs?: number | null;
  nowMs?: number | null;
  queue?: ChatQueueState | null;
}): PendingChatOutputPresentation {
  const outputTokens = safeCount(input.outputTokens) ?? 0;
  if (input.queue) {
    const position = Math.max(1, safeCount(input.queue.position) ?? 1);
    const owner = input.queue.queuedBehind.label.replace(/\s+/g, " ").trim();
    const label = `Queued #${position} behind ${owner || "current neural work"}`;
    return {
      phase: "queued",
      label,
      ariaLabel:
        `${label}. Stopping this message does not cancel the activity ahead of it.`,
      outputTokens: 0
    };
  }
  if (outputTokens > 0) {
    const tokenNoun = outputTokens === 1 ? "token" : "tokens";
    const label = `${outputTokens.toLocaleString("en-US")} output ${tokenNoun} used`;
    return {
      phase: "streaming",
      label,
      ariaLabel: `Live response output: ${label}`,
      outputTokens
    };
  }

  const startedAtMs = safeTimestamp(input.startedAtMs);
  const nowMs = safeTimestamp(input.nowMs);
  const elapsedMs =
    startedAtMs !== undefined && nowMs !== undefined && nowMs >= startedAtMs
      ? Math.floor(nowMs - startedAtMs)
      : undefined;
  const label = elapsedMs !== undefined && elapsedMs >= 1_000
    ? `Loading brain · no output tokens yet · ${elapsedCopy(elapsedMs)}`
    : "Loading brain · no output tokens yet";
  return {
    phase: "loading",
    label,
    ariaLabel: label,
    outputTokens: 0,
    ...(elapsedMs !== undefined ? { elapsedMs } : {})
  };
}
