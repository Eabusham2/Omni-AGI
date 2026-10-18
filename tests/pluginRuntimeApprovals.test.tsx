import React from "react";
import { readFileSync } from "node:fs";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import type { PluginRuntimeApi, PluginRuntimeEvent, PluginRuntimePendingApproval, PluginRuntimeRpcReply } from "../src/shared/pluginRuntime";
import { RuntimeApprovalController, ownedRuntimeApprovalSnapshot, type RuntimeApprovalSnapshot } from "../src/renderer/src/runtimeApprovalController";
import { PluginRuntimeApprovalCard, PluginRuntimeApprovalQueue } from "../src/renderer/src/PluginRuntimeApprovals";

const approval = (id = "a", brainId = "brain-a"): PluginRuntimePendingApproval => ({ brainId, sessionId: `session-${id}`, rpcId: `rpc-${id}`, reviewId: `review-${id}`,
  pluginId: "owned.fixture", contentSha256: "a".repeat(64), moduleId: "parser", executionId: `execution-${id}`,
  invocation: { brainId, toolId: "system.files", action: "read", arguments: { path: "/exact/reviewed/source.dat" } } });
function deferred<T>() { let resolve!: (value: T) => void; return { promise: new Promise<T>(done => { resolve = done; }), resolve }; }
const flush = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };

it("coalesces approval notifications and refuses late publication after owner teardown", async () => {
  const pending = deferred<PluginRuntimePendingApproval[]>(), changed = vi.fn(), unsubscribe = vi.fn();
  let listener!: (event: PluginRuntimeEvent) => void;
  const reads = vi.fn(() => pending.promise), api = { approvals: reads, onEvent: (value: typeof listener) => { listener = value; return unsubscribe; } } as unknown as PluginRuntimeApi;
  const controller = new RuntimeApprovalController(api, "brain-a", changed); controller.start();
  for (let i = 0; i < 30; i++) listener({ brainId: "brain-a", kind: "approval" } as PluginRuntimeEvent);
  expect(reads).toHaveBeenCalledTimes(1); controller.stop(); pending.resolve([approval()]); await flush();
  expect(changed).not.toHaveBeenCalled(); expect(reads).toHaveBeenCalledTimes(1); expect(unsubscribe).toHaveBeenCalledOnce();
  expect(ownedRuntimeApprovalSnapshot("brain-b", { brainId: "brain-a", approvals: [approval()], busyReviewIds: [], error: "old", message: "old" }).approvals).toEqual([]);
});

it("reviews independent exact pending actions concurrently without duplicating a review", async () => {
  const first = deferred<PluginRuntimeRpcReply>(), second = deferred<PluginRuntimeRpcReply>(), snapshots: RuntimeApprovalSnapshot[] = [], items = [approval("a"), approval("b")];
  const review = vi.fn((request: { reviewId: string }) => request.reviewId === "review-a" ? first.promise : second.promise);
  const api = { approvals: async () => [...items, approval("foreign", "brain-b")], review, onEvent: () => () => undefined } as unknown as PluginRuntimeApi;
  const controller = new RuntimeApprovalController(api, "brain-a", value => snapshots.push(value)); controller.start(); await flush();
  expect(snapshots.at(-1)!.approvals).toHaveLength(2);
  const a = controller.review(items[0]!, "approve"), b = controller.review(items[1]!, "deny");
  await controller.review(items[0]!, "approve");
  expect(review).toHaveBeenCalledTimes(2); expect(snapshots.at(-1)!.busyReviewIds).toEqual(["review-a", "review-b"]);
  expect(review.mock.calls[0]![0]).toEqual({ brainId: "brain-a", sessionId: "session-a", rpcId: "rpc-a", reviewId: "review-a", decision: "approve" });
  first.resolve({ format: "omni-plugin-runtime-reply", version: 1, id: "rpc-a", state: "complete" });
  second.resolve({ format: "omni-plugin-runtime-reply", version: 1, id: "rpc-b", state: "failed", error: "User denied" });
  await Promise.all([a, b]); controller.stop();
});

it("keeps a genuine reviewed-action failure visible after refreshing the pending queue", async () => {
  const item = approval(), snapshots: RuntimeApprovalSnapshot[] = [];
  const api = { approvals: async () => [item], onEvent: () => () => undefined,
    review: async () => ({ format: "omni-plugin-runtime-reply", version: 1, id: item.rpcId, state: "uncertain", error: "Lost effect acknowledgment; not replayed" }) } as unknown as PluginRuntimeApi;
  const controller = new RuntimeApprovalController(api, "brain-a", value => snapshots.push(value)); controller.start(); await flush();
  await controller.review(item, "approve"); await flush();
  expect(snapshots.at(-1)!.error).toBe("Lost effect acknowledgment; not replayed"); expect(snapshots.at(-1)!.message).toBe(""); controller.stop();
});

it("retains exact operands and review controls before chat without loading a model", () => {
  const item = approval(), html = renderToStaticMarkup(<PluginRuntimeApprovalCard item={item} busy={false} onReview={vi.fn()} />);
  expect(html).toContain("/exact/reviewed/source.dat"); expect(html).toContain("Approve exact action"); expect(html).toContain(">Deny</button>");
  expect(html.indexOf(item.rpcId)).toBeGreaterThan(html.indexOf("<summary>Request details</summary>")); expect(html).not.toContain("approvalToken");
  const api = { approvals: vi.fn(), onEvent: vi.fn() } as unknown as PluginRuntimeApi;
  expect(renderToStaticMarkup(<PluginRuntimeApprovalQueue brainId="brain-a" api={api} />)).toBe(""); expect(api.approvals).not.toHaveBeenCalled();
  const source = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
  expect(source.includes("<PluginRuntimeApprovalQueue key={initializationControlBrainId}")).toBe(true);
  expect(source.includes("brainId={(initializationControlBrainId ?? initialMaterialsBrainId)!}")).toBe(true);
});
