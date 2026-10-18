import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({ expose: vi.fn(), on: vi.fn(), off: vi.fn(), invoke: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke }, webUtils: { getPathForFile: vi.fn() } }));
import "../src/preload/index";

it("retains panel history paging behind typed human API without leaking its live nonce or grants", async () => {
  const plugins = (bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi).plugins;
  const source = "b".repeat(64);
  await plugins.panelHistory("brain", { sourceSha256: source, afterSequence: 256, maxBytes: 4096 });
  await plugins.panelHistoryBytes("brain", 257, 0, 1024, source);
  await plugins.panelHistorySources("brain", { afterSha256: source, limit: 2 });
  expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([
    [IPC.plugins.panelHistory, "brain", { sourceSha256: source, afterSequence: 256, maxBytes: 4096 }],
    [IPC.plugins.panelHistoryBytes, "brain", 257, 0, 1024, source],
    [IPC.plugins.panelHistorySources, "brain", { afterSha256: source, limit: 2 }]
  ]));
  expect("nonce" in plugins).toBe(false); expect("approvalToken" in plugins).toBe(false);
});

it("exposes typed human runtime controls and bounded custody pages, not the private child transport", async () => {
  const runtime = (bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi).pluginRuntime;
  const hash = "a".repeat(64), selected = { brainId: "brain", sourceSha256: hash, pluginHash: hash, moduleId: "module", key: "data", expectedRevision: 0 };
  await runtime.list("brain", { afterRowid: 12, maxBytes: 4096 });
  await runtime.history("brain", { sourceSha256: hash, afterSequence: 42, maxBytes: 4096 });
  await runtime.readHistoryEvent("brain", 42, 0, 1024, hash);
  await runtime.stateSources("brain", { afterSha256: hash, limit: 2 });
  await runtime.stateConflicts("brain", { afterKey: "cursor", limit: 2 });
  await runtime.readSourceState("brain", hash, hash, "module", "data", 4096);
  await runtime.readSourceStateBytes("brain", hash, hash, "module", "data", 0, 1024);
  await runtime.selectSourceState(selected);
  const listener = vi.fn(), off = runtime.onEvent(listener);
  const wrapped = bridge.on.mock.calls.find(([channel]) => channel === IPC.pluginRuntime.event)?.[1];
  const event = { brainId: "brain", kind: "contributions" }; wrapped({}, event); off();
  expect(listener).toHaveBeenCalledWith(event); expect(bridge.off).toHaveBeenCalledWith(IPC.pluginRuntime.event, wrapped);
  expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([
    [IPC.pluginRuntime.list, "brain", { afterRowid: 12, maxBytes: 4096 }],
    [IPC.pluginRuntime.history, "brain", { sourceSha256: hash, afterSequence: 42, maxBytes: 4096 }],
    [IPC.pluginRuntime.readHistoryEvent, "brain", 42, 0, 1024, hash],
    [IPC.pluginRuntime.stateSources, "brain", { afterSha256: hash, limit: 2 }],
    [IPC.pluginRuntime.stateConflicts, "brain", { afterKey: "cursor", limit: 2 }],
    [IPC.pluginRuntime.readSourceState, "brain", hash, hash, "module", "data", 4096],
    [IPC.pluginRuntime.readSourceStateBytes, "brain", hash, hash, "module", "data", 0, 1024],
    [IPC.pluginRuntime.selectSourceState, selected]
  ]));
  expect("request" in runtime).toBe(false); expect("execute" in runtime).toBe(false);
  expect("nonce" in runtime).toBe(false); expect("credentials" in runtime).toBe(false);
});
