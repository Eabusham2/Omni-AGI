import { describe, expect, it, vi } from "vitest";
import type { OmniApi } from "../src/shared/types";
import { IPC } from "../src/shared/ipc";

const bridge = vi.hoisted(() => ({
  expose: vi.fn(),
  on: vi.fn(),
  off: vi.fn(),
  invoke: vi.fn(),
  getPath: vi.fn((file: { name: string }) => `/trusted/${file.name}`)
}));
vi.mock("electron", () => ({
  contextBridge: { exposeInMainWorld: bridge.expose },
  ipcRenderer: { on: bridge.on, removeListener: bridge.off, invoke: bridge.invoke },
  webUtils: { getPathForFile: bridge.getPath }
}));
import "../src/preload/index";

describe("task attachment preload boundary", () => {
  it("exposes handles and converts dropped File objects through Electron's trusted path API", async () => {
    const api = bridge.expose.mock.calls.find(([name]) => name === "omni")?.[1] as OmniApi;
    const request = { brainId: "brain-one", requestId: "stage-one", selection: "files" as const };
    await api.chat.attachments.stage(request);
    await api.chat.attachments.stageDropped(
      { brainId: "brain-one", requestId: "drop-one" },
      [{ name: "one.txt" }, { name: "two.png" }]
    );
    await api.chat.attachments.cancelStage("brain-one", "stage-one");
    await api.chat.attachments.list("brain-one");
    await api.chat.attachments.list("brain-one", "staged");
    await api.chat.attachments.forTurn("brain-one", "turn-one");
    await api.chat.attachments.byIds("brain-one", ["attachment-one"]);
    await api.chat.attachments.remove("brain-one", "attachment-one");
    expect(bridge.getPath).toHaveBeenCalledTimes(2);
    expect(bridge.invoke.mock.calls).toEqual(expect.arrayContaining([
      [IPC.chat.stageAttachments, request],
      [IPC.chat.stageDroppedAttachments, { brainId: "brain-one", requestId: "drop-one" },
        ["/trusted/one.txt", "/trusted/two.png"]],
      [IPC.chat.cancelAttachmentStage, "brain-one", "stage-one"],
      [IPC.chat.listAttachments, "brain-one"],
      [IPC.chat.listAttachments, "brain-one", "staged"],
      [IPC.chat.attachmentsForTurn, "brain-one", "turn-one"],
      [IPC.chat.attachmentsByIds, "brain-one", ["attachment-one"]],
      [IPC.chat.removeAttachment, "brain-one", "attachment-one"]
    ]));
  });
});
