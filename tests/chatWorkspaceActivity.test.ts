import { describe, expect, it } from "vitest";
import {
  EMPTY_CHAT_WORKSPACE_ACTIVITY,
  dataActionAvailableDuringTurn,
  isChatVisibleLearningJob,
  mergeChatVisibleLearningJob,
  workspaceChatActivityPresentation,
  workspaceViewWaitsForForegroundTurn,
  type ChatWorkspaceActivitySnapshot
} from "../src/renderer/src/chatWorkspaceActivity";
import {
  reconcileOptimisticChatMessages,
  settleOptimisticChatTurn
} from "../src/renderer/src/chatTurnPresentation";
import type { RuntimeJob } from "../src/shared/types";

describe("cross-workspace chat activity", () => {
  it("stops the token cursor as soon as reply generation is complete", () => {
    const responding: ChatWorkspaceActivitySnapshot = {
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      turnActive: true,
      phase: "responding"
    };
    const saving: ChatWorkspaceActivitySnapshot = {
      ...responding,
      phase: "reply-complete-learning",
      queuedMessages: 2
    };

    expect(workspaceChatActivityPresentation(responding)).toMatchObject({
      headerLabel: "Reply in progress",
      showStreamingCursor: true,
      blocksImmediateBrainActions: true
    });
    expect(workspaceChatActivityPresentation(saving)).toMatchObject({
      headerLabel: "Reply complete · learning/saving · 2 messages queued",
      showStreamingCursor: false,
      blocksImmediateBrainActions: true
    });
    expect(workspaceChatActivityPresentation({
      ...saving,
      phase: "action-result-learning"
    })).toMatchObject({
      headerLabel: "Action complete · integrating result · 2 messages queued",
      waitingLabel:
        "The visible action is complete while its result enters neural learning. Return to Conversation to Queue or Steer.",
      showStreamingCursor: false,
      blocksImmediateBrainActions: true
    });
  });

  it("shows the real runtime owner while chat waits behind evolution", () => {
    const queued: ChatWorkspaceActivitySnapshot = {
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      turnActive: true,
      phase: "queued",
      queue: {
        position: 1,
        queuedBehind: {
          requestId: "evolution-one",
          owner: "evolution",
          label: "Evolution proposal",
          method: "evolution.propose",
          brainId: "brain-1"
        }
      }
    };
    expect(workspaceChatActivityPresentation(queued)).toEqual({
      headerLabel: "Queued #1 behind Evolution proposal",
      headerAriaLabel:
        "Queued #1 behind Evolution proposal. Return to Conversation to inspect or cancel only this queued message.",
      waitingLabel:
        "This message is waiting behind Evolution proposal. Queue and Steer controls remain in Conversation.",
      showStreamingCursor: false,
      blocksImmediateBrainActions: true
    });
  });

  it("keeps passive navigation and queue-aware long jobs available", () => {
    const active: ChatWorkspaceActivitySnapshot = {
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      turnActive: true,
      phase: "responding"
    };
    expect(workspaceViewWaitsForForegroundTurn("map", active)).toBe(false);
    expect(workspaceViewWaitsForForegroundTurn("trace", active)).toBe(false);
    expect(workspaceViewWaitsForForegroundTurn("data", active)).toBe(false);
    expect(workspaceViewWaitsForForegroundTurn("tools", active)).toBe(true);
    expect(workspaceViewWaitsForForegroundTurn("imagine", active)).toBe(true);
    expect(dataActionAvailableDuringTurn("queue-long-job", true)).toBe(true);
    expect(dataActionAvailableDuringTurn("immediate-write", true)).toBe(false);
  });

  it("surfaces queued and running long jobs even without an active reply", () => {
    expect(workspaceChatActivityPresentation({
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      queuedJobs: 2,
      runningJobs: 1
    })).toMatchObject({
      headerLabel: "3 learning tasks active",
      showStreamingCursor: false,
      blocksImmediateBrainActions: false
    });
    expect(workspaceChatActivityPresentation({
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      queuedMessages: 2,
      queuedJobs: 1
    })?.headerLabel).toBe("2 messages queued · 1 learning task active");
    expect(workspaceChatActivityPresentation(EMPTY_CHAT_WORKSPACE_ACTIVITY))
      .toBeUndefined();
  });

  it("keeps every Training, Data, and Crawl queue entry while updating it in place", () => {
    const job = (
      id: string,
      kind: RuntimeJob["kind"],
      state: RuntimeJob["state"],
      createdAt: string
    ): RuntimeJob => ({
      id,
      brainId: "brain-1",
      kind,
      state,
      progress: state === "running" ? 0.4 : 0,
      label: `${kind} ${id}`,
      createdAt,
      updatedAt: createdAt
    });
    const training = job("training", "training", "queued", "2026-09-01T00:00:00.000Z");
    const ingestion = job("data", "ingestion", "queued", "2026-09-01T00:00:01.000Z");
    const crawl = job("crawl", "crawl", "queued", "2026-09-01T00:00:02.000Z");
    const image = job("image", "image", "queued", "2026-09-01T00:00:03.000Z");

    expect([training, ingestion, crawl].every(isChatVisibleLearningJob)).toBe(true);
    expect(isChatVisibleLearningJob(image)).toBe(false);
    const queued = [training, ingestion, crawl].reduce(
      mergeChatVisibleLearningJob,
      [] as RuntimeJob[]
    );
    const running = mergeChatVisibleLearningJob(queued, {
      ...ingestion,
      state: "running",
      progress: 0.4,
      updatedAt: "2026-09-01T00:00:04.000Z"
    });
    expect(running.map((entry) => entry.id)).toEqual(["training", "data", "crawl"]);
    expect(running).toHaveLength(3);
    expect(running[1]).toMatchObject({ id: "data", state: "running", progress: 0.4 });

    const uncapped = Array.from({ length: 30 }, (_, index) =>
      job(
        `queued-${index}`,
        index % 2 ? "ingestion" : "crawl",
        "queued",
        `2026-09-02T00:00:${String(index).padStart(2, "0")}.000Z`
      )
    ).reduce(mergeChatVisibleLearningJob, [] as RuntimeJob[]);
    expect(uncapped).toHaveLength(30);
  });

  it("simulates token to phase to navigation and terminal reconciliation", () => {
    let view: "chat" | "map" = "chat";
    let partialText = "";
    let activity: ChatWorkspaceActivitySnapshot = {
      ...EMPTY_CHAT_WORKSPACE_ACTIVITY,
      turnActive: true,
      phase: "responding",
      queuedMessages: 1
    };
    const pending = {
      id: "pending-turn-1",
      role: "human" as const,
      content: "hello",
      createdAt: "2026-09-01T00:00:00.000Z"
    };
    const optimistic = [pending];

    partialText += "Visible reply";
    activity = { ...activity, phase: "reply-complete-learning" };
    expect(workspaceChatActivityPresentation(activity)?.showStreamingCursor).toBe(false);
    view = "map";
    expect(view).toBe("map");
    expect(partialText).toBe("Visible reply");
    expect(workspaceChatActivityPresentation(activity)?.headerLabel)
      .toContain("Reply complete · learning/saving");
    view = "chat";
    expect(view).toBe("chat");
    expect(partialText).toBe("Visible reply");

    const authoritative = [{ ...pending, id: "engine-human-1" }];
    activity = EMPTY_CHAT_WORKSPACE_ACTIVITY;
    partialText = "";
    expect(reconcileOptimisticChatMessages(optimistic, authoritative)).toEqual([]);
    expect(workspaceChatActivityPresentation(activity)).toBeUndefined();
    expect(partialText).toBe("");

    expect(settleOptimisticChatTurn(optimistic, "turn-1", "failed")[0]?.id)
      .toBe("failed-pending-turn-1");
    expect(settleOptimisticChatTurn(optimistic, "turn-1", "cancelled")[0]?.id)
      .toBe("cancelled-pending-turn-1");
  });
});
