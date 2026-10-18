import {
  TOUCH_BAR_ACTION_FORMAT,
  TOUCH_BAR_STATE_FORMAT,
  type TouchBarActionEvent,
  type TouchBarState
} from "../../shared/touchBar";

export interface ConversationTouchBarInput {
  revision: number;
  brainId: string;
  instanceName: string;
  visible: boolean;
  runtimeLoaded: boolean;
  draftHasContent: boolean;
  draftFits: boolean;
  draftBusy: boolean;
  inheritedReviewPending: boolean;
  turnActive: boolean;
  activeTurnId?: string | null;
  cancellingTurn: boolean;
  callAvailable: boolean;
  callActive: boolean;
  callBusy: boolean;
}

export type MatchedTouchBarAction =
  | { action: "submit"; mode: "send" | "queue" }
  | { action: "stop"; turnId: string }
  | { action: "call"; mode: "start" | "end" };

/**
 * Mirrors the existing conversation controls without inventing another input
 * path. In particular, Stop is exposed only while a concrete text turn is
 * active; voice sessions retain their separate End Call control.
 */
export function conversationTouchBarState(
  input: ConversationTouchBarInput
): TouchBarState | null {
  if (!input.visible) return null;
  const canSubmit =
    input.runtimeLoaded &&
    input.draftHasContent &&
    input.draftFits &&
    !input.draftBusy &&
    !input.inheritedReviewPending;
  const callMode = input.callActive ? "end" : "start";
  return {
    format: TOUCH_BAR_STATE_FORMAT,
    version: 1,
    revision: input.revision,
    brainId: input.brainId,
    instanceName: input.instanceName,
    phase: "conversation",
    submit: { mode: input.turnActive ? "queue" : "send", enabled: canSubmit },
    ...(input.runtimeLoaded && input.turnActive && input.activeTurnId && !input.cancellingTurn
      ? { stop: { turnId: input.activeTurnId } }
      : {}),
    ...(input.callAvailable
      ? {
          call: {
            mode: callMode,
            enabled: !input.callBusy && (callMode === "end" || input.runtimeLoaded)
          }
        }
      : {})
  };
}

/** Re-evaluates the exact operation against the latest renderer semantics.
 * This closes the interval between a React state/ref change and the next
 * accepted native Touch Bar update. */
export function touchBarActionRemainsReady(
  action: MatchedTouchBarAction,
  latest: ConversationTouchBarInput
): boolean {
  const state = conversationTouchBarState({ ...latest, revision: 1 });
  if (!state) return false;
  if (action.action === "submit") {
    return state.submit?.enabled === true && state.submit.mode === action.mode;
  }
  if (action.action === "stop") {
    return state.stop?.turnId === action.turnId;
  }
  return state.call?.enabled === true && state.call.mode === action.mode;
}

/**
 * Renderer-side final fence for a native callback. The main process already
 * binds each button to a focused window and revision; this second comparison
 * prevents a delayed event from operating on another instance or a newer UI
 * state.
 */
export function matchTouchBarAction(
  event: TouchBarActionEvent,
  state: TouchBarState | null
): MatchedTouchBarAction | null {
  if (
    !state ||
    event.format !== TOUCH_BAR_ACTION_FORMAT ||
    event.version !== 1 ||
    event.brainId !== state.brainId ||
    event.stateRevision !== state.revision
  ) {
    return null;
  }
  if (
    event.action === "submit" &&
    state.submit?.enabled &&
    event.submitMode === state.submit.mode
  ) {
    return { action: "submit", mode: state.submit.mode };
  }
  if (
    event.action === "stop" &&
    state.stop &&
    event.turnId === state.stop.turnId
  ) {
    return { action: "stop", turnId: state.stop.turnId };
  }
  if (
    event.action === "call" &&
    state.call?.enabled &&
    event.callMode === state.call.mode
  ) {
    return { action: "call", mode: state.call.mode };
  }
  return null;
}
