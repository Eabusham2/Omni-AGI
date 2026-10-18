import { expect, it, vi } from "vitest";
import { McpAppDisplayGate, presentMcpAppDisplayMode } from "../src/renderer/src/mcpAppDisplayModes";
import type { McpAppDisplayEvent } from "../src/shared/mcpApps";

function dialogFixture(refuse = false) {
  let modal = false;
  const child = { identity: "same-existing-iframe" };
  const dialog = { isConnected: true, open: true, child,
    matches: (selector: string) => selector === ":modal" && modal,
    close: vi.fn(() => { dialog.open = false; modal = false; }),
    show: vi.fn(() => { dialog.open = true; modal = false; }),
    showModal: vi.fn(() => { if (refuse) throw new Error("browser declined"); dialog.open = true; modal = true; }) };
  return { dialog, child };
}
it("switches the existing surface between inline and top-layer modes without replacing its iframe", () => {
  const f = dialogFixture(); expect(presentMcpAppDisplayMode(f.dialog as unknown as HTMLDialogElement, "fullscreen")).toBe("fullscreen");
  expect(f.dialog.child).toBe(f.child); expect(f.dialog.showModal).toHaveBeenCalledOnce();
  expect(presentMcpAppDisplayMode(f.dialog as unknown as HTMLDialogElement, "fullscreen")).toBe("fullscreen"); expect(f.dialog.showModal).toHaveBeenCalledOnce();
  expect(presentMcpAppDisplayMode(f.dialog as unknown as HTMLDialogElement, "inline")).toBe("inline"); expect(f.dialog.open).toBe(true); expect(f.dialog.child).toBe(f.child);
});
it("reports actual inline mode when the browser refuses fullscreen or the surface has unmounted", () => {
  const f = dialogFixture(true); expect(presentMcpAppDisplayMode(f.dialog as unknown as HTMLDialogElement, "fullscreen")).toBe("inline");
  expect(f.dialog.show).toHaveBeenCalledOnce(); expect(f.dialog.open).toBe(true); expect(f.dialog.child).toBe(f.child);
  f.dialog.isConnected = false; expect(presentMcpAppDisplayMode(f.dialog as unknown as HTMLDialogElement, "fullscreen")).toBe("inline"); expect(f.dialog.showModal).toHaveBeenCalledOnce();
});
it("rejects foreign, delayed, malformed and retired view events before any presentation or acknowledgement", () => {
  const identity = { brainId: "brain", viewId: "view", viewToken: "a".repeat(64) }, gate = new McpAppDisplayGate(identity);
  const event: McpAppDisplayEvent = { ...identity, sequence: 1, action: "apply", operationId: "nonce", displayMode: "fullscreen", availableDisplayModes: ["inline", "fullscreen"] };
  for (const changed of [{ brainId: "other" }, { viewId: "other" }, { viewToken: "b".repeat(64) }, { operationId: undefined }]) expect(gate.receive({ ...event, ...changed })).toBeUndefined();
  expect(gate.receive(event)).toBe(event); expect(gate.current(1)).toBe(true); expect(gate.receive(event)).toBeUndefined();
  const closed = { ...event, action: "closed" as const, sequence: 2, displayMode: "inline" as const }; expect(gate.receive(closed)).toBe(closed);
  expect(gate.receive({ ...event, sequence: 3 })).toBeUndefined(); expect(gate.current(1)).toBe(false);
});
