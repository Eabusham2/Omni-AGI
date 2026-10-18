import { expect, it, vi } from "vitest";
import { ChatActionController } from "../src/main/chatActionController";
import type { ActionEvent } from "../src/shared/types";

it("declines one exact pending chat proposal without dispatch, learning, deletion or cross-owner authority", async () => {
  const event: ActionEvent = { id: "proposal", brainId: "owner", action: { kind: "tool", source: "brain", toolId: "system.shell", action: "run", arguments: { command: "fixture" } },
    state: "approval-required", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), execution: {
      id: "execution", toolId: "system.shell", action: "run", state: "approval-required", startedAt: new Date().toISOString(), approvalToken: "exact-token" } };
  const requests = new Map([[event.id, { event, expiresAt: Date.now() + 30000 }]]);
  const invalidateApproval = vi.fn(), record = vi.fn(async () => undefined), publish = vi.fn(), execute = vi.fn();
  const controller = Object.assign(Object.create(ChatActionController.prototype), {
    pendingApprovedActions: requests, tools: { invalidateApproval, execute }, service: { recordConversationActions: record }, publish
  }) as ChatActionController;
  await expect(controller.declineAction({ brainId: "other", actionEventId: event.id, approvalToken: "exact-token" })).rejects.toThrow(/unavailable/);
  await expect(controller.declineAction({ brainId: "owner", actionEventId: event.id, approvalToken: "wrong-token" })).rejects.toThrow(/unavailable/);
  expect(requests.size).toBe(1); expect(invalidateApproval).not.toHaveBeenCalled();
  const result = await controller.declineAction({ brainId: "owner", actionEventId: event.id, approvalToken: "exact-token" });
  expect(result.state).toBe("stopped"); expect(result.statusLabel).toMatch(/no new action dispatched/);
  expect(result.action.arguments).toEqual({ command: "fixture" }); expect(requests.size).toBe(0);
  expect(invalidateApproval).toHaveBeenCalledWith("owner", "exact-token"); expect(record).toHaveBeenCalledWith("owner", [event]); expect(publish).toHaveBeenCalledTimes(1);
  expect(execute).not.toHaveBeenCalled();
  await expect(controller.approveAction({ brainId: "owner", actionEventId: event.id, approvalToken: "exact-token" })).rejects.toThrow(/unavailable/);
});
