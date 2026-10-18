import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { TaskAttachmentStrip } from "../src/renderer/src/TaskAttachmentStrip";
import { ChatQueueStack } from "../src/renderer/src/ChatQueueStack";
import { mergeTaskAttachments } from "../src/renderer/src/useTaskAttachmentDraft";
import type { ChatTaskAttachmentHandle } from "../src/shared/taskAttachments";
const handle: ChatTaskAttachmentHandle = { format: "omni-chat-task-attachment", version: 1,
  id: "task-file", brainId: "fixture", displayName: "A long selected document.pdf", kind: "file",
  mimeType: "application/pdf", bytes: 12345, state: "staged", contentSha256: "a".repeat(64),
  manifestSha256: "b".repeat(64), createdAt: "2026-10-11T12:00:00Z", updatedAt: "2026-10-11T12:00:00Z",
  ingestionStarted: false, parametersChanged: false };
describe("plain task attachment presentation", () => {
  it("keeps exact selected names/hash available without claiming training", () => {
    const html = renderToStaticMarkup(<TaskAttachmentStrip entries={[handle]} onRemove={() => undefined} />);
    expect(html).toContain("Remove attachment A long selected document.pdf"); expect(html).toContain(handle.contentSha256);
    expect(html).toContain("does not start training"); expect(html).not.toContain("relativePath");
  });
  it("shows queued attachment ownership without another upload or removable historical file", () => {
    const html = renderToStaticMarkup(<ChatQueueStack items={[{ id: "queued", text: "Review this", attachments: [handle] }]}
      busy onEdit={() => undefined} onRemove={() => undefined} onMove={() => undefined} onSendNow={() => undefined} />);
    expect(html).toContain(handle.displayName); expect(html).toContain("Steer with queued message 1");
    expect(html).not.toContain("Remove attachment"); expect(html).not.toContain("type=\"file\"");
  });
  it("deduplicates only equal attachment identities and preserves distinct selections", () => {
    expect(mergeTaskAttachments([handle], [{ ...handle, state: "bound" }, { ...handle, id: "other" }]))
      .toEqual([{ ...handle, state: "bound" }, { ...handle, id: "other" }]);
  });
});
