import { expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";
const bridge = vi.hoisted(() => ({ invoke: vi.fn(async () => undefined), expose: vi.fn() }));
vi.mock("electron", () => ({ contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { invoke: bridge.invoke, on: vi.fn(), removeListener: vi.fn() }, webUtils: { getPathForFile: vi.fn() } }));
it("exposes only reviewed imagery profile controls, never a renderer fetch or credential getter", async () => {
  await import("../src/preload/index");
  const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
  const maps = api.maps.basemaps!;
  await maps.list(); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.maps.basemapList);
  const profile = { id: "explicit-source", expectedRevision: 0 } as Parameters<typeof maps.save>[0];
  await maps.save(profile); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.maps.basemapSave, profile);
  await maps.remove("explicit-source", 1); expect(bridge.invoke).toHaveBeenLastCalledWith(IPC.maps.basemapRemove, "explicit-source", 1);
  expect(maps).not.toHaveProperty("execute"); expect(maps).not.toHaveProperty("fetch"); expect(maps).not.toHaveProperty("getCredential");
});
