import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ChatDirectionStrip, chatDirectionStatus, type ChatDirectionStripItem } from "../src/renderer/src/ChatDirectionStrip";
const direction: ChatDirectionStripItem = { id: "successor", content: "Keep the current task, but use the second source.",
  state: "pending", inputReceived: false, dormant: false, dismissed: false };
describe("receipt-based steer presentation", () => {
  it("does not equate started/pending or copied records with native receipt", () => {
    expect(chatDirectionStatus(direction)).toBe("Waiting");
    expect(chatDirectionStatus({ ...direction, state: "failed" })).toBe("Not received");
    expect(chatDirectionStatus({ ...direction, state: "failed", inputReceived: true })).toBe("Received · reply did not finish");
    expect(chatDirectionStatus({ ...direction, dormant: true })).toBe("Saved copy · not resent");
  });
  it("renders a compact expandable strip and dismisses display, not recorded learning", () => {
    const html = renderToStaticMarkup(<ChatDirectionStrip items={[direction]} total={1} busy={false} canSteer
      hasOlder={false} onOlder={() => undefined} onDismiss={async () => undefined} onRevise={async () => undefined} />);
    expect(html).toContain("Steering directions"); expect(html).toContain(direction.content);
    expect(html).toContain("recorded input and learning remain saved"); expect(html).toContain("Waiting");
    expect(html).not.toContain("textarea"); expect(html).not.toContain("successor</");
  });
  it("does not silently show dismissed records again or remove pagination for retained history", () => {
    const html = renderToStaticMarkup(<ChatDirectionStrip items={[{ ...direction, dismissed: true }]} total={10} busy={false} canSteer={false}
      hasOlder onOlder={() => undefined} onDismiss={async () => undefined} onRevise={async () => undefined} />);
    expect(html).not.toContain(direction.content); expect(html).toContain("Earlier directions");
  });
});
