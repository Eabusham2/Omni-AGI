import type { ChatQueueState, RuntimeJob } from "../../shared/types";

export type ChatWorkspaceTurnPhase =
  | "idle"
  | "queued"
  | "responding"
  | "reply-complete-learning"
  | "action-result-learning";

export interface ChatWorkspaceActivitySnapshot {
  turnActive: boolean;
  phase: ChatWorkspaceTurnPhase;
  queuedMessages: number;
  queuedJobs: number;
  runningJobs: number;
  /** Completed text has no generation controls while optional work continues. */
  postReplyWork?: number;
  queue?: ChatQueueState;
}

export const EMPTY_CHAT_WORKSPACE_ACTIVITY: ChatWorkspaceActivitySnapshot = {
  turnActive: false,
  phase: "idle",
  queuedMessages: 0,
  queuedJobs: 0,
  runningJobs: 0
};

export interface WorkspaceChatActivityPresentation {
  headerLabel: string;
  headerAriaLabel: string;
  waitingLabel: string;
  showStreamingCursor: boolean;
  blocksImmediateBrainActions: boolean;
}

function safeCount(value: number): number {
  return Number.isSafeInteger(value) && value > 0 ? value : 0;
}

export function workspaceChatActivityPresentation(
  snapshot: ChatWorkspaceActivitySnapshot
): WorkspaceChatActivityPresentation | undefined {
  const queuedMessages = safeCount(snapshot.queuedMessages);
  const queuedJobs = safeCount(snapshot.queuedJobs);
  const runningJobs = safeCount(snapshot.runningJobs);
  const jobCount = queuedJobs + runningJobs;
  const postReplyWork = safeCount(snapshot.postReplyWork ?? 0);
  const queuedSuffix = queuedMessages > 0
    ? ` · ${queuedMessages.toLocaleString()} message${queuedMessages === 1 ? "" : "s"} queued`
    : "";

  const generating = snapshot.turnActive &&
    snapshot.phase !== "reply-complete-learning" && snapshot.phase !== "action-result-learning";
  if (generating) {
    if (snapshot.phase === "queued" && snapshot.queue) {
      const position = Math.max(1, Math.floor(snapshot.queue.position));
      const owner = snapshot.queue.queuedBehind.label;
      const headerLabel = `Queued #${position} behind ${owner}${queuedSuffix}`;
      return {
        headerLabel,
        headerAriaLabel:
          `${headerLabel}. Return to Conversation to inspect or cancel only this queued message.`,
        waitingLabel:
          `This message is waiting behind ${owner}. Queue and Steer controls remain in Conversation.`,
        showStreamingCursor: false,
        blocksImmediateBrainActions: true
      };
    }
    const headerLabel = `Reply in progress${queuedSuffix}`;
    return {
      headerLabel,
      headerAriaLabel:
        `${headerLabel}. Return to Conversation for explicit Queue or Steer controls.`,
      waitingLabel: "Waiting for the current reply. Return to Conversation to Queue or Steer.",
      showStreamingCursor: true,
      blocksImmediateBrainActions: true
    };
  }

  if (postReplyWork > 0 || snapshot.phase === "reply-complete-learning" ||
      snapshot.phase === "action-result-learning") {
    const headerLabel = snapshot.phase === "action-result-learning"
      ? `Action complete · integrating result${queuedSuffix}`
      : snapshot.phase === "reply-complete-learning"
        ? `Reply complete · learning/saving${queuedSuffix}`
        : `${postReplyWork.toLocaleString()} completed repl${postReplyWork === 1 ? "y" : "ies"} · save/action work`;
    return {
      headerLabel,
      headerAriaLabel: `${headerLabel}. Completed output is preserved; the next message can be sent normally.`,
      waitingLabel: "Completed output stays visible while save/action work runs independently.",
      showStreamingCursor: false,
      blocksImmediateBrainActions: false
    };
  }

  if (queuedMessages > 0 || jobCount > 0) {
    const messageLabel = queuedMessages > 0
      ? `${queuedMessages.toLocaleString()} message${queuedMessages === 1 ? "" : "s"} queued`
      : "";
    const jobLabel = jobCount > 0
      ? `${jobCount.toLocaleString()} learning task${jobCount === 1 ? "" : "s"} active`
      : "";
    const headerLabel = [messageLabel, jobLabel].filter(Boolean).join(" · ");
    return {
      headerLabel,
      headerAriaLabel: `${headerLabel}. Return to Conversation to inspect the shared queue.`,
      waitingLabel: jobCount > 0
        ? "Long learning work remains visible in Conversation and runs through the shared queue."
        : "Queued messages remain visible in Conversation and will run in order.",
      showStreamingCursor: false,
      blocksImmediateBrainActions: false
    };
  }

  return undefined;
}

export type BusyAwareWorkspaceView =
  | "chat"
  | "data"
  | "map"
  | "trace"
  | "imagine"
  | "tools"
  | "settings"
  | "agents"
  | "evolution";

const IMMEDIATE_BRAIN_ACTION_VIEWS = new Set<BusyAwareWorkspaceView>([
  "imagine",
  "tools",
  "settings",
  "agents",
  "evolution"
]);

/** Read-only map/trace navigation and queue-aware Data controls remain usable. */
export function workspaceViewWaitsForForegroundTurn(
  view: BusyAwareWorkspaceView,
  snapshot: ChatWorkspaceActivitySnapshot
): boolean {
  return snapshot.turnActive && snapshot.phase !== "reply-complete-learning" &&
    snapshot.phase !== "action-result-learning" && IMMEDIATE_BRAIN_ACTION_VIEWS.has(view);
}

export type DataActionScheduling = "queue-long-job" | "immediate-write";

export function dataActionAvailableDuringTurn(
  scheduling: DataActionScheduling,
  turnActive: boolean
): boolean {
  return !turnActive || scheduling === "queue-long-job";
}

const CHAT_VISIBLE_LEARNING_JOB_KINDS = new Set<RuntimeJob["kind"]>([
  "training",
  "ingestion",
  "crawl"
]);

export function isChatVisibleLearningJob(job: RuntimeJob): boolean {
  return CHAT_VISIBLE_LEARNING_JOB_KINDS.has(job.kind);
}

/** Keep every visible queue entry while replacing updates for the same job. */
export function mergeChatVisibleLearningJob(
  jobs: RuntimeJob[],
  job: RuntimeJob
): RuntimeJob[] {
  const next = jobs.some((candidate) => candidate.id === job.id)
    ? jobs.map((candidate) => candidate.id === job.id ? job : candidate)
    : [...jobs, job];
  return next.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}
