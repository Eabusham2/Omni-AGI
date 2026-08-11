import { describe, expect, it, vi } from "vitest";
import {
  createTextFrameBatcher,
  mergeChatActionEvent,
  patchChatActionPreview
} from "../src/renderer/src/chatStreaming";
import type { ActionEvent } from "../src/shared/types";

function actionEvent(overrides: Partial<ActionEvent> = {}): ActionEvent {
  return {
    id: "action-1",
    brainId: "brain-1",
    action: {
      kind: "imagine",
      source: "brain",
      toolId: "modality.imagine",
      action: "generate",
      arguments: { modality: "image" }
    },
    state: "running",
    createdAt: "2026-08-10T00:00:00.000Z",
    updatedAt: "2026-08-10T00:00:01.000Z",
    ...overrides
  };
}

describe("renderer chat streaming state", () => {
  it("does not rerender for a duplicate action envelope", () => {
    const event = actionEvent({ progress: 0.25, statusLabel: "Forming latents" });
    const current = [event];
    const next = mergeChatActionEvent(current, { ...event });
    expect(next).toBe(current);
  });

  it("applies ordered lightweight preview patches without losing media", () => {
    const initial = [actionEvent()];
    const withMedia = patchChatActionPreview(initial, "action-1", {
      revision: 0,
      progress: 0.2,
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,b25jZQ=="
    });
    const metadataOnly = patchChatActionPreview(withMedia, "action-1", {
      revision: 1,
      progress: 0.7,
      statusLabel: "Refining details"
    });

    expect(metadataOnly[0]?.preview).toEqual({
      revision: 1,
      progress: 0.7,
      statusLabel: "Refining details",
      mimeType: "image/png",
      dataUrl: "data:image/png;base64,b25jZQ=="
    });
    expect(metadataOnly[0]).toMatchObject({
      progress: 0.7,
      statusLabel: "Refining details"
    });
    expect(
      patchChatActionPreview(metadataOnly, "action-1", {
        revision: 1,
        dataUrl: "data:image/png;base64,c3RhbGU="
      })
    ).toBe(metadataOnly);

    const lightweightAction = actionEvent({
      progress: 0.9,
      updatedAt: "2026-08-10T00:00:02.000Z"
    });
    expect(mergeChatActionEvent(metadataOnly, lightweightAction)[0]?.preview)
      .toEqual(metadataOnly[0]?.preview);
  });

  it("commits many token deltas once per scheduled frame", () => {
    const commits = vi.fn();
    const callbacks = new Map<number, () => void>();
    let nextHandle = 0;
    const batcher = createTextFrameBatcher(
      commits,
      (callback) => {
        const handle = ++nextHandle;
        callbacks.set(handle, () => {
          callbacks.delete(handle);
          callback();
        });
        return handle;
      },
      (handle) => callbacks.delete(handle)
    );

    batcher.push("one");
    batcher.push(" two");
    batcher.push(" three");
    expect(callbacks).toHaveLength(1);
    expect(commits).not.toHaveBeenCalled();

    callbacks.get(1)?.();
    expect(commits).toHaveBeenCalledTimes(1);
    expect(commits).toHaveBeenLastCalledWith("one two three");

    batcher.push(" terminal");
    batcher.flush();
    expect(commits).toHaveBeenCalledTimes(2);
    expect(commits).toHaveBeenLastCalledWith(" terminal");
    expect(callbacks).toHaveLength(0);
  });
});
