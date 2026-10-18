import {
  TOUCH_BAR_ACTION_FORMAT,
  TOUCH_BAR_STATE_FORMAT,
  type TouchBarActionEvent,
  type TouchBarState,
  type TouchBarUpdateResult
} from "../shared/touchBar";

const SAFE_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$/;

export interface NativeTouchBarButton {
  label: string;
  enabled: boolean;
}

export interface NativeTouchBarFactory {
  button(options: { label: string; accessibilityLabel: string; enabled: boolean; click(): void }): NativeTouchBarButton;
  spacer(): unknown;
  bar(items: unknown[]): unknown;
}

export interface TouchBarWindowHost {
  isDestroyed(): boolean;
  isFocused(): boolean;
  setTouchBar(value: unknown | null): void;
}

function exactKeys(value: Record<string, unknown>, keys: readonly string[], label: string): void {
  if (Object.keys(value).sort().join("\0") !== [...keys].sort().join("\0")) throw new Error(`${label} contains unsupported fields.`);
}

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${label} must be an object.`);
  return value as Record<string, unknown>;
}

export function validatedTouchBarState(value: unknown): TouchBarState {
  const state = record(value, "Touch Bar state");
  const keys = ["format", "version", "revision", "brainId", "instanceName", "phase"];
  for (const optional of ["submit", "stop", "call"] as const) if (Object.hasOwn(state, optional)) keys.push(optional);
  exactKeys(state, keys, "Touch Bar state");
  if (state.format !== TOUCH_BAR_STATE_FORMAT || state.version !== 1 || state.phase !== "conversation" ||
      !Number.isSafeInteger(state.revision) || Number(state.revision) < 1 || typeof state.brainId !== "string" ||
      !SAFE_ID.test(state.brainId) || typeof state.instanceName !== "string" || !state.instanceName.trim() ||
      state.instanceName.length > 128 || state.instanceName.includes("\0")) throw new Error("Touch Bar state identity is invalid.");
  let submit: TouchBarState["submit"], stop: TouchBarState["stop"], call: TouchBarState["call"];
  if (state.submit !== undefined) {
    const item = record(state.submit, "Touch Bar submit state"); exactKeys(item, ["mode", "enabled"], "Touch Bar submit state");
    if (!['send', 'queue'].includes(String(item.mode)) || typeof item.enabled !== "boolean") throw new Error("Touch Bar submit state is invalid.");
    submit = { mode: item.mode as "send" | "queue", enabled: item.enabled };
  }
  if (state.stop !== undefined) {
    const item = record(state.stop, "Touch Bar stop state"); exactKeys(item, ["turnId"], "Touch Bar stop state");
    if (typeof item.turnId !== "string" || !SAFE_ID.test(item.turnId)) throw new Error("Touch Bar stop state is invalid.");
    stop = { turnId: item.turnId };
  }
  if (state.call !== undefined) {
    const item = record(state.call, "Touch Bar call state"); exactKeys(item, ["mode", "enabled"], "Touch Bar call state");
    if (!['start', 'end'].includes(String(item.mode)) || typeof item.enabled !== "boolean") throw new Error("Touch Bar call state is invalid.");
    call = { mode: item.mode as "start" | "end", enabled: item.enabled };
  }
  return { format: TOUCH_BAR_STATE_FORMAT, version: 1, revision: Number(state.revision), brainId: state.brainId,
    instanceName: state.instanceName.trim(), phase: "conversation", ...(submit ? { submit } : {}),
    ...(stop ? { stop } : {}), ...(call ? { call } : {}) };
}

/** Main-owned native mirror. Button callbacks contain no prompt/text and must
 * pass the current focused-window and exact state-revision checks. */
export class MacTouchBarController {
  private state?: TouchBarState;
  private senderId?: number;

  constructor(
    private readonly host: TouchBarWindowHost,
    private readonly factory: NativeTouchBarFactory,
    private readonly publish: (event: TouchBarActionEvent) => void,
    private readonly platform = process.platform
  ) {}

  update(senderId: number, value: TouchBarState | null): TouchBarUpdateResult {
    if (this.platform !== "darwin") return { supported: false, accepted: false };
    if (!Number.isSafeInteger(senderId) || senderId < 1) throw new Error("Touch Bar sender identity is invalid.");
    if (value === null) {
      this.state = undefined; this.senderId = senderId; this.render(); return { supported: true, accepted: true };
    }
    const state = validatedTouchBarState(value);
    if (this.senderId === senderId && this.state && state.revision <= this.state.revision) {
      return { supported: true, accepted: false, revision: this.state.revision };
    }
    this.senderId = senderId; this.state = state; this.render();
    return { supported: true, accepted: true, revision: state.revision };
  }

  resetOwner(): void {
    this.state = undefined; this.senderId = undefined; this.render();
  }

  refresh(): void { this.render(); }

  private render(): void {
    if (this.platform !== "darwin" || this.host.isDestroyed()) return;
    const state = this.state;
    if (!state || !this.host.isFocused()) { this.host.setTouchBar(null); return; }
    const ownerRevision = state.revision, items: unknown[] = [];
    if (state.submit) items.push(this.factory.button({ label: state.submit.mode === "queue" ? "Queue" : "Send",
      accessibilityLabel: `${state.submit.mode === "queue" ? "Queue" : "Send"} the current message`, enabled: state.submit.enabled,
      click: () => this.activate("submit", ownerRevision) }));
    if (state.stop) items.push(this.factory.button({ label: "Stop", accessibilityLabel: "Stop the active response",
      enabled: true, click: () => this.activate("stop", ownerRevision) }));
    if (state.call) items.push(this.factory.button({ label: state.call.mode === "end" ? "End Call" : "Call",
      accessibilityLabel: state.call.mode === "end" ? "End the current call" : "Start a call with this instance",
      enabled: state.call.enabled, click: () => this.activate("call", ownerRevision) }));
    if (!items.length) { this.host.setTouchBar(null); return; }
    if (items.length > 1) items.splice(1, 0, this.factory.spacer());
    this.host.setTouchBar(this.factory.bar(items));
  }

  private activate(action: TouchBarActionEvent["action"], revision: number): void {
    const state = this.state;
    if (this.platform !== "darwin" || !state || state.revision !== revision || this.host.isDestroyed() || !this.host.isFocused()) return;
    const base = { format: TOUCH_BAR_ACTION_FORMAT, version: 1 as const, brainId: state.brainId, stateRevision: state.revision };
    if (action === "submit" && state.submit?.enabled) this.publish({ ...base, action, submitMode: state.submit.mode });
    else if (action === "stop" && state.stop) this.publish({ ...base, action, turnId: state.stop.turnId });
    else if (action === "call" && state.call?.enabled) this.publish({ ...base, action, callMode: state.call.mode });
  }
}
