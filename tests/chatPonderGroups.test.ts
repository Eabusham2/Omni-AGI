import { describe, expect, it } from "vitest";
import { groupPonderActions, type PonderLinkedAction } from "../src/renderer/src/chatPonderGroups";

// Plain typed UI data only: no brain, neural runtime, host tool or effect.
const event = (id: string, kind: PonderLinkedAction["action"]["kind"],
  patch: Partial<PonderLinkedAction> = {}): PonderLinkedAction => ({
  id, brainId: "owner", chatTurnId: "turn", neuralActionId: (kind === "ponder" ? "a" : "b").repeat(32),
  createdAt: "2026-10-18T12:00:00.000Z", updatedAt: "2026-10-18T12:00:00.000Z",
  state: "running", action: { kind, source: "brain", arguments: {} }, ...patch
});

describe("explicit native Ponder episode presentation", () => {
  it("nests the exact issued child without changing either original event", () => {
    const parent = event("ponder", "ponder"), child = event("tool", "tool", { parentPonderActionEventId: parent.id });
    const before = JSON.stringify([parent, child]), result = groupPonderActions([parent, child]);
    expect(result.roots).toEqual([parent]);
    expect(result.childrenByParent.get(parent.id)).toEqual([child]);
    expect(result.childrenByParent.get(parent.id)?.[0]).toBe(child);
    expect(JSON.stringify([parent, child])).toBe(before);
  });

  it("never treats adjacency, time or the same prefix-selection fields as parentage", () => {
    const parent = event("ponder", "ponder"), child = event("tool", "tool");
    for (const row of [parent, child]) row.action = { ...row.action, selectionPhase: "ponder", selectionStep: 1 };
    const result = groupPonderActions([parent, child]);
    expect(result.roots).toEqual([parent, child]); expect(result.childrenByParent.size).toBe(0);
  });

  it("keeps missing or unloaded parent history visible instead of losing its child", () => {
    const child = event("tool", "tool", { parentPonderActionEventId: "outside-loaded-history" });
    expect(groupPonderActions([child])).toEqual({ roots: [child], childrenByParent: new Map() });
  });

  it("rejects different brains, turns, human actions and absent native identities", () => {
    const parent = event("ponder", "ponder");
    const invalid = [
      event("other-brain", "tool", { brainId: "other", parentPonderActionEventId: parent.id }),
      event("other-turn", "tool", { chatTurnId: "other", parentPonderActionEventId: parent.id }),
      event("no-turn", "tool", { chatTurnId: undefined, parentPonderActionEventId: parent.id }),
      event("no-native", "tool", { neuralActionId: undefined, parentPonderActionEventId: parent.id }),
      event("human", "tool", { parentPonderActionEventId: parent.id, action: { kind: "tool", source: "human", arguments: {} } })
    ];
    const result = groupPonderActions([parent, ...invalid]);
    expect(result.roots).toEqual([parent, ...invalid]); expect(result.childrenByParent.size).toBe(0);
    expect(groupPonderActions([{ ...parent, neuralActionId: undefined }, event("child", "tool", { parentPonderActionEventId: parent.id })]).roots).toHaveLength(2);
  });

  it("does not attach to a human or non-Ponder parent or create recursive loops", () => {
    for (const parent of [event("parent", "tool"), event("parent", "ponder", { action: { kind: "ponder", source: "human", arguments: {} } })]) {
      expect(groupPonderActions([parent, event("child", "tool", { parentPonderActionEventId: parent.id })]).roots).toHaveLength(2);
    }
    const parent = event("parent", "ponder", { parentPonderActionEventId: "child" });
    const child = event("child", "ponder", { parentPonderActionEventId: parent.id });
    expect(groupPonderActions([parent, child]).roots).toEqual([parent, child]);
  });

  it("preserves original order for separate episodes and their actual child revisions", () => {
    const first = event("p1", "ponder"), second = event("p2", "ponder"), ordinary = event("ordinary", "tool");
    const one = event("one", "tool", { parentPonderActionEventId: first.id });
    const two = event("two", "imagine", { parentPonderActionEventId: first.id, state: "complete" });
    const other = event("other", "learn", { parentPonderActionEventId: second.id, state: "failed" });
    const result = groupPonderActions([first, one, ordinary, second, two, other]);
    expect(result.roots).toEqual([first, ordinary, second]);
    expect(result.childrenByParent.get(first.id)).toEqual([one, two]);
    expect(result.childrenByParent.get(second.id)).toEqual([other]);
  });

  it("retains real child activity when its parent has stopped", () => {
    const parent = event("ponder", "ponder", { state: "stopped" });
    const child = event("tool", "tool", { parentPonderActionEventId: parent.id, state: "approval-required" });
    expect(groupPonderActions([parent, child]).childrenByParent.get(parent.id)?.[0]?.state).toBe("approval-required");
  });

  it("leaves ambiguous event identities flat rather than choosing an invented owner", () => {
    const parent = event("same", "ponder"), duplicate = { ...parent };
    const child = event("tool", "tool", { parentPonderActionEventId: parent.id });
    expect(groupPonderActions([parent, duplicate, child]).roots).toEqual([parent, duplicate, child]);
    expect(groupPonderActions([parent, child, { ...child }]).childrenByParent.size).toBe(0);
  });
});
