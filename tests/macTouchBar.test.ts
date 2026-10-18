import { describe, expect, it, vi } from "vitest";
import { MacTouchBarController, validatedTouchBarState, type NativeTouchBarFactory } from "../src/main/macTouchBar";
import { TOUCH_BAR_STATE_FORMAT, type TouchBarActionEvent, type TouchBarState } from "../src/shared/touchBar";

function state(revision = 1): TouchBarState {
  return { format: TOUCH_BAR_STATE_FORMAT, version: 1, revision, brainId: "brain-one", instanceName: "Ada",
    phase: "conversation", submit: { mode: "send", enabled: true }, stop: { turnId: "turn-one" },
    call: { mode: "start", enabled: true } };
}

describe("optional native Mac Touch Bar mirror", () => {
  it("normalizes only typed nonprompt state", () => {
    expect(validatedTouchBarState(state())).toEqual(state());
    expect(() => validatedTouchBarState({ ...state(), prompt: "hidden" })).toThrow(/unsupported fields/i);
    expect(() => validatedTouchBarState({ ...state(), stop: { turnId: "bad id" } })).toThrow(/stop state/i);
  });

  it("publishes only current focused actions and never keeps stale Stop authority", () => {
    let focused = true, destroyed = false; const bars: Array<unknown | null> = [], buttons: Array<{ label: string; enabled: boolean; click(): void }> = [];
    const factory: NativeTouchBarFactory = { button: options => { const button = { label: options.label, enabled: options.enabled, click: options.click }; buttons.push(button); return button; },
      spacer: () => ({ spacer: true }), bar: items => ({ items }) };
    const events: TouchBarActionEvent[] = [], host = { isDestroyed: () => destroyed, isFocused: () => focused,
      setTouchBar: (value: unknown | null) => { bars.push(value); } };
    const controller = new MacTouchBarController(host, factory, event => events.push(event), "darwin");
    expect(controller.update(7, state())).toMatchObject({ supported: true, accepted: true, revision: 1 });
    expect(buttons.map(button => button.label)).toEqual(["Send", "Stop", "Call"]);
    buttons[0]!.click(); buttons[1]!.click(); buttons[2]!.click();
    expect(events.map(event => event.action)).toEqual(["submit", "stop", "call"]);
    const staleStop = buttons[1]!;
    controller.update(7, { ...state(2), submit: { mode: "queue", enabled: false }, stop: undefined,
      call: { mode: "end", enabled: true } });
    staleStop.click(); expect(events).toHaveLength(3);
    expect(buttons.slice(-2).map(button => [button.label, button.enabled])).toEqual([["Queue", false], ["End Call", true]]);
    focused = false; controller.refresh(); buttons.at(-1)!.click(); expect(events).toHaveLength(3); expect(bars.at(-1)).toBeNull();
    focused = true; controller.refresh(); destroyed = true; buttons.at(-1)!.click(); expect(events).toHaveLength(3);
  });

  it("rejects stale revisions, resets navigation ownership, and stays inert off macOS", () => {
    const setTouchBar = vi.fn(), factory: NativeTouchBarFactory = { button: options => ({ label: options.label, enabled: options.enabled }), spacer: () => ({}), bar: items => items };
    const mac = new MacTouchBarController({ isDestroyed: () => false, isFocused: () => true, setTouchBar }, factory, vi.fn(), "darwin");
    mac.update(3, state(4)); expect(mac.update(3, state(4))).toMatchObject({ accepted: false, revision: 4 });
    mac.resetOwner(); expect(setTouchBar).toHaveBeenLastCalledWith(null); expect(mac.update(3, state(1))).toMatchObject({ accepted: true });
    const other = new MacTouchBarController({ isDestroyed: () => false, isFocused: () => true, setTouchBar }, factory, vi.fn(), "win32");
    expect(other.update(3, state())).toEqual({ supported: false, accepted: false });
  });
});
