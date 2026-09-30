import { describe, expect, it } from "vitest";
import {
  clearSubmittedChatDraft,
  chatMessageDeliveryState,
  composerSubmitIntent,
  composerTurnCapabilities,
  isCurrentChatSubmission,
  mergeCommittedChatMessages,
  mergeChatMessagesForPresentation,
  preserveSteeredChatMessage,
  reconcileCompletedChatSubmission,
  reconcileOptimisticChatMessages,
  recoverFailedChatDraft,
  settleOptimisticChatMessage,
  settleOptimisticChatTurn,
  shouldFollowChatOutput
} from "../src/renderer/src/chatTurnPresentation";

describe("chat turn presentation state", () => {
  it("prevents cancelled pre-Steer work from mutating the replacement or next turn", () => {
    const originalGeneration = 1;
    const steerGeneration = 2;
    const nextMessageGeneration = 3;

    expect(
      isCurrentChatSubmission(originalGeneration, nextMessageGeneration)
    ).toBe(false);
    expect(
      isCurrentChatSubmission(steerGeneration, nextMessageGeneration)
    ).toBe(false);
    expect(
      isCurrentChatSubmission(nextMessageGeneration, nextMessageGeneration)
    ).toBe(true);
  });

  it("returns a failed current message to an empty composer", () => {
    expect(
      recoverFailedChatDraft("", "hello, tell me what you notice")
    ).toBe("hello, tell me what you notice");
  });

  it("never overwrites wording already typed for a newer turn", () => {
    expect(
      recoverFailedChatDraft(
        "new wording typed while the request settled",
        "failed older wording"
      )
    ).toBe("new wording typed while the request settled");
  });

  it("keeps a failed submitted hi visible while restoring it for retry", () => {
    const pending = {
      id: "pending-turn-hi",
      role: "human" as const,
      content: "hi",
      createdAt: "2026-09-01T00:00:00.000Z"
    };
    const settled = settleOptimisticChatMessage(
      [pending],
      pending.id,
      "failed"
    );

    expect(settled).toEqual([{ ...pending, id: "failed-pending-turn-hi" }]);
    expect(reconcileOptimisticChatMessages(settled, [])).toEqual(settled);
    expect(recoverFailedChatDraft("", pending.content)).toBe("hi");
    expect(
      reconcileOptimisticChatMessages(settled, [
        { ...pending, id: "authoritative-turn-hi" }
      ])
    ).toEqual([]);
  });

  it("settles only the failed or cancelled active turn and preserves queued humans", () => {
    const messages = [
      {
        id: "pending-active",
        role: "human" as const,
        content: "active",
        createdAt: "2026-09-01T00:00:00.000Z"
      },
      {
        id: "queued-next",
        role: "human" as const,
        content: "next",
        createdAt: "2026-09-01T00:00:01.000Z"
      }
    ];

    expect(settleOptimisticChatTurn(messages, "active", "cancelled")).toEqual([
      { ...messages[0], id: "cancelled-pending-active" },
      messages[1]
    ]);
    expect(settleOptimisticChatTurn(messages, "next", "failed")).toEqual([
      messages[0],
      { ...messages[1], id: "failed-queued-next" }
    ]);
  });

  it("clears only the exact submitted draft and preserves newer wording", () => {
    expect(clearSubmittedChatDraft("first message", "first message")).toBe("");
    expect(
      clearSubmittedChatDraft("next message already being typed", "first message")
    ).toBe("next message already being typed");
  });

  it("queues ordinary Enter during a turn and reserves Ctrl/Cmd Enter for Steer", () => {
    expect(composerSubmitIntent({
      key: "Enter", shiftKey: false, ctrlKey: false, metaKey: false, turnActive: true,
      queueAvailable: true, steerAvailable: true
    })).toBe("queue");
    expect(composerSubmitIntent({
      key: "Enter", shiftKey: false, ctrlKey: true, metaKey: false, turnActive: true,
      queueAvailable: true, steerAvailable: true
    })).toBe("steer");
    expect(composerSubmitIntent({
      key: "Enter", shiftKey: false, ctrlKey: false, metaKey: true, turnActive: true,
      queueAvailable: true, steerAvailable: true
    })).toBe("steer");
    expect(composerSubmitIntent({
      key: "Enter", shiftKey: true, ctrlKey: true, metaKey: false, turnActive: true,
      queueAvailable: true, steerAvailable: true
    })).toBe("newline");
    expect(composerSubmitIntent({
      key: "Enter", shiftKey: false, ctrlKey: false, metaKey: false, turnActive: false,
      queueAvailable: false, steerAvailable: false
    })).toBe("send");
  });

  it("exposes Steer only while a response can still change", () => {
    expect(composerTurnCapabilities("idle")).toEqual({
      turnActive: false,
      queueAvailable: false,
      steerAvailable: false
    });
    for (const phase of ["loading", "generating"] as const) {
      expect(composerTurnCapabilities(phase)).toEqual({
        turnActive: true,
        queueAvailable: true,
        steerAvailable: true
      });
    }
    for (const phase of [
      "reply-complete-learning",
      "action-result-learning"
    ] as const) {
      expect(composerTurnCapabilities(phase)).toEqual({
        turnActive: false,
        queueAvailable: false,
        steerAvailable: false
      });
    }
  });

  it("presents ordinary Send after response generation completes", () => {
    for (const phase of [
      "reply-complete-learning",
      "action-result-learning"
    ] as const) {
      const capabilities = composerTurnCapabilities(phase);
      expect(composerSubmitIntent({
        key: "Enter",
        shiftKey: false,
        ctrlKey: true,
        metaKey: false,
        turnActive: capabilities.turnActive,
        queueAvailable: capabilities.queueAvailable,
        steerAvailable: capabilities.steerAvailable
      })).toBe("send");
    }
  });

  it("keeps the interrupted direction visible as a settled Steer receipt", () => {
    const messages = [{
      id: "pending-original-turn",
      role: "human" as const,
      content: "start broadly",
      createdAt: "2026-08-27T00:00:00.000Z"
    }];
    expect(preserveSteeredChatMessage(messages, "original-turn")).toEqual([{
      ...messages[0],
      id: "steered-original-turn"
    }]);
    expect(preserveSteeredChatMessage([{
      ...messages[0]!,
      id: "queued-original-turn"
    }], "original-turn")[0]?.id).toBe("steered-original-turn");
  });

  it("reconciles a local Steer receipt when the same human turn is committed", () => {
    const message = {
      id: "steered-original-turn",
      role: "human" as const,
      content: "start broadly",
      createdAt: "2026-08-27T00:00:00.000Z"
    };
    expect(reconcileOptimisticChatMessages([message], [{
      ...message,
      id: "committed-original-turn"
    }])).toEqual([]);
    expect(reconcileOptimisticChatMessages([message], [{
      ...message,
      id: "another-turn",
      content: "different wording"
    }])).toEqual([message]);
  });

  it("restores only the latest durable non-neural receipt and never replays it", () => {
    const createdAt = "2026-09-08T06:00:00.000Z";
    const queued = {
      id: "delivery-turn-one-queued",
      role: "human" as const,
      content: "keep this queued wording",
      createdAt,
      deliveryReceipt: {
        schemaVersion: 1 as const,
        presentationOnly: true as const,
        turnId: "turn-one",
        state: "queued" as const,
        updatedAt: "2026-09-08T06:00:00.100Z"
      }
    };
    const cancelled = {
      ...queued,
      id: "delivery-turn-one-cancelled",
      deliveryReceipt: {
        ...queued.deliveryReceipt,
        state: "cancelled" as const,
        updatedAt: "2026-09-08T06:00:01.000Z"
      }
    };
    const local = {
      id: "cancelled-queued-turn-one",
      role: "human" as const,
      content: queued.content,
      createdAt
    };

    const restored = mergeChatMessagesForPresentation(
      [queued, cancelled],
      [local]
    );
    expect(restored).toEqual([cancelled]);
    expect(chatMessageDeliveryState(restored[0]!)).toBe("cancelled");
    expect(restored[0]?.deliveryReceipt?.presentationOnly).toBe(true);
  });

  it("hides a queued display receipt after the exact neural human turn commits", () => {
    const createdAt = "2026-09-08T06:10:00.000Z";
    const receipt = {
      id: "delivery-turn-two-queued",
      role: "human" as const,
      content: "eventually committed",
      createdAt,
      deliveryReceipt: {
        schemaVersion: 1 as const,
        presentationOnly: true as const,
        turnId: "turn-two",
        state: "queued" as const,
        updatedAt: createdAt
      }
    };
    const committed = {
      id: "worker-human-two",
      role: "human" as const,
      content: receipt.content,
      createdAt
    };
    expect(mergeChatMessagesForPresentation(
      [receipt, committed],
      []
    )).toEqual([committed]);
  });

  it("keeps a sent human visible while a stale ledger page has only its pending receipt", () => {
    const createdAt = "2026-09-08T06:20:00.000Z";
    const optimistic = {
      id: "pending-turn-three",
      role: "human" as const,
      content: "show me immediately",
      createdAt
    };
    const pendingReceipt = {
      ...optimistic,
      id: "delivery-turn-three-pending",
      deliveryReceipt: {
        schemaVersion: 1 as const,
        presentationOnly: true as const,
        turnId: "turn-three",
        state: "pending" as const,
        updatedAt: createdAt
      }
    };
    const stalePage = [pendingReceipt];
    expect(mergeChatMessagesForPresentation(stalePage, [optimistic]))
      .toEqual([pendingReceipt]);
    expect(chatMessageDeliveryState(pendingReceipt)).toBe("pending");

    const committed = {
      ...optimistic,
      id: "worker-human-three",
      createdAt: "2026-09-08T06:20:01.000Z",
      turnId: "turn-three"
    };
    const reply = {
      id: "worker-brain-three",
      role: "brain" as const,
      content: "I can see it.",
      createdAt: committed.createdAt,
      turnId: "turn-three"
    };
    const fromResult = mergeCommittedChatMessages(stalePage, [committed, reply]);
    expect(mergeChatMessagesForPresentation(
      fromResult,
      reconcileCompletedChatSubmission([optimistic], optimistic.id, committed)
    )).toEqual([committed, reply]);
    expect(mergeCommittedChatMessages(fromResult, [committed, reply]))
      .toEqual(fromResult);
  });

  it("retire only the submitted bubble when worker and renderer timestamps differ", () => {
    const optimistic = {
      id: "pending-turn-five",
      role: "human" as const,
      content: "same turn, different clock",
      createdAt: "2026-09-08T06:30:00.000Z"
    };
    const queued = {
      ...optimistic,
      id: "queued-turn-six",
      content: "another turn"
    };
    const committed = {
      ...optimistic,
      id: "worker-human-five",
      createdAt: "2026-09-08T06:30:01.000Z"
    };
    expect(reconcileCompletedChatSubmission(
      [optimistic, queued],
      optimistic.id,
      committed
    )).toEqual([queued]);
  });

  it("does not let a late queue receipt replace a later pending or failed receipt", () => {
    const createdAt = "2026-09-08T06:25:00.000Z";
    const receipt = (state: "queued" | "pending" | "failed", updatedAt: string) => ({
      id: `delivery-turn-four-${state}`,
      role: "human" as const,
      content: "queued then sent",
      createdAt,
      deliveryReceipt: {
        schemaVersion: 1 as const,
        presentationOnly: true as const,
        turnId: "turn-four",
        state,
        updatedAt
      }
    });
    const queued = receipt("queued", "2026-09-08T06:25:03.000Z");
    const pending = receipt("pending", "2026-09-08T06:25:02.000Z");
    const failed = receipt("failed", "2026-09-08T06:25:01.000Z");
    expect(mergeChatMessagesForPresentation([pending, queued], []))
      .toEqual([pending]);
    expect(mergeChatMessagesForPresentation([failed, pending, queued], []))
      .toEqual([failed]);
  });

  it("follows streaming output near the end but not after a deliberate scroll up", () => {
    expect(shouldFollowChatOutput(900, 500, 1_450)).toBe(true);
    expect(shouldFollowChatOutput(300, 500, 1_450)).toBe(false);
    expect(shouldFollowChatOutput(Number.NaN, 500, 1_450)).toBe(false);
  });
});
