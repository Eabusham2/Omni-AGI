import { describe, expect, it } from "vitest";
import {
  conversationTouchBarState,
  matchTouchBarAction,
  touchBarActionRemainsReady
} from "../src/renderer/src/touchBarPresentation";
import { TOUCH_BAR_ACTION_FORMAT } from "../src/shared/touchBar";

const base = {
  revision: 7,
  brainId: "brain-one",
  instanceName: "Ada",
  visible: true,
  runtimeLoaded: true,
  draftHasContent: true,
  draftFits: true,
  draftBusy: false,
  inheritedReviewPending: false,
  turnActive: false,
  activeTurnId: null,
  cancellingTurn: false,
  callAvailable: true,
  callActive: false,
  callBusy: false
};

describe("conversation Touch Bar presentation", () => {
  it("mirrors Send, exact-turn Stop, Queue, and real call readiness", () => {
    const idle = conversationTouchBarState(base)!;
    expect(idle).toMatchObject({
      submit: { mode: "send", enabled: true },
      call: { mode: "start", enabled: true }
    });
    expect(idle.stop).toBeUndefined();

    const active = conversationTouchBarState({
      ...base,
      turnActive: true,
      activeTurnId: "turn-one",
      callActive: true
    })!;
    expect(active).toMatchObject({
      submit: { mode: "queue", enabled: true },
      stop: { turnId: "turn-one" },
      call: { mode: "end", enabled: true }
    });
    expect(conversationTouchBarState({ ...base, visible: false })).toBeNull();
    expect(conversationTouchBarState({ ...base, runtimeLoaded: false })).toMatchObject({
      submit: { enabled: false },
      call: { mode: "start", enabled: false }
    });
    expect(conversationTouchBarState({ ...base, turnActive: true, activeTurnId: "turn-one", cancellingTurn: true })?.stop).toBeUndefined();
  });

  it("accepts only the current brain, revision, action operands, and enabled control", () => {
    const state = conversationTouchBarState({ ...base, turnActive: true, activeTurnId: "turn-one" })!;
    const event = {
      format: TOUCH_BAR_ACTION_FORMAT,
      version: 1 as const,
      brainId: "brain-one",
      stateRevision: 7
    };
    expect(matchTouchBarAction({ ...event, action: "submit", submitMode: "queue" }, state)).toEqual({ action: "submit", mode: "queue" });
    expect(matchTouchBarAction({ ...event, action: "stop", turnId: "turn-one" }, state)).toEqual({ action: "stop", turnId: "turn-one" });
    expect(matchTouchBarAction({ ...event, action: "call", callMode: "start" }, state)).toEqual({ action: "call", mode: "start" });
    expect(matchTouchBarAction({ ...event, stateRevision: 6, action: "stop", turnId: "turn-one" }, state)).toBeNull();
    expect(matchTouchBarAction({ ...event, brainId: "brain-two", action: "stop", turnId: "turn-one" }, state)).toBeNull();
    expect(matchTouchBarAction({ ...event, action: "submit", submitMode: "send" }, state)).toBeNull();
    expect(matchTouchBarAction({ ...event, action: "stop", turnId: "turn-two" }, state)).toBeNull();

    const disabled = conversationTouchBarState({ ...base, draftHasContent: false })!;
    expect(matchTouchBarAction({ ...event, action: "submit", submitMode: "send" }, disabled)).toBeNull();
  });

  it("rechecks semantic readiness before a delayed native callback dispatches", () => {
    expect(touchBarActionRemainsReady(
      { action: "stop", turnId: "turn-one" },
      { ...base, turnActive: true, activeTurnId: "turn-two" }
    )).toBe(false);
    expect(touchBarActionRemainsReady(
      { action: "stop", turnId: "turn-one" },
      { ...base, runtimeLoaded: false, turnActive: true, activeTurnId: "turn-one" }
    )).toBe(false);
    expect(touchBarActionRemainsReady(
      { action: "submit", mode: "send" },
      { ...base, turnActive: true, activeTurnId: "turn-one" }
    )).toBe(false);
    expect(touchBarActionRemainsReady(
      { action: "call", mode: "start" },
      { ...base, callActive: true }
    )).toBe(false);
    expect(touchBarActionRemainsReady(
      { action: "stop", turnId: "turn-one" },
      { ...base, turnActive: true, activeTurnId: "turn-one" }
    )).toBe(true);
  });
});
