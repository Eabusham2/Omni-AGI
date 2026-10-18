import { expect, it, vi } from "vitest";
import type { ReactElement } from "react";
import { ComposerSubmitControl } from "../src/renderer/src/ComposerSubmitControl";

// Inject a deliberately stale display modifier; only the real click may steer.
vi.mock("react", () => ({ useState: () => [true, vi.fn()], useEffect: vi.fn() }));

it("never treats stale display state as permission to steer a normal click", () => {
  const onSend = vi.fn(), onSteer = vi.fn();
  const root = ComposerSubmitControl({ active: true, stopping: false, canSend: true, canSteer: true, onSend, onSteer, onStop: vi.fn() });
  const children = root.props.children as ReactElement<{ className: string; onClick(event: { ctrlKey: boolean; metaKey: boolean }): void }>[];
  const submit = children.find(child => child?.props.className.includes("composer__submit"))!;
  submit.props.onClick({ ctrlKey: false, metaKey: false });
  expect(onSend).toHaveBeenCalledOnce(); expect(onSteer).not.toHaveBeenCalled();
  submit.props.onClick({ ctrlKey: true, metaKey: false });
  submit.props.onClick({ ctrlKey: false, metaKey: true });
  expect(onSteer).toHaveBeenCalledTimes(2); expect(onSend).toHaveBeenCalledOnce();
});
