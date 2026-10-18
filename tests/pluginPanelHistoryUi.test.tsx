import React from "react";
import { Buffer } from "node:buffer";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { PluginsApi } from "../src/shared/plugins";
import {
  PluginPanelHistory,
  PluginPanelHistoryRecord,
  appendPanelHistoryBytes,
  panelHistoryReadText
} from "../src/renderer/src/PluginPanelHistory";

const SHA = "a".repeat(64);

describe("plugin panel history UI", () => {
  it("does not query durable panel history while its disclosure remains closed", () => {
    const api = { panelHistory: vi.fn(), panelHistoryBytes: vi.fn(), panelHistorySources: vi.fn() } as unknown as PluginsApi;
    const html = renderToStaticMarkup(<PluginPanelHistory brainId="brain-a" api={api} />);
    expect(html).toContain("<summary>Panel history</summary>");
    expect(api.panelHistory).not.toHaveBeenCalled();
    expect(api.panelHistorySources).not.toHaveBeenCalled();
  });

  it("renders oversized custody as explicit bounded pages without clipping or exposing it in the summary", () => {
    const body = '{"result":"complete"}', encoded = Buffer.from(body).toString("base64");
    const record = { sequence: 7, viewId: "view-under-details", requestId: "request-under-details", kind: "actual-host-result",
      sha256: SHA, bytes: Buffer.byteLength(body) };
    const read = appendPanelHistoryBytes(undefined, record, 0, { text: encoded, bytes: record.bytes, sha256: SHA });
    expect(panelHistoryReadText(read)).toBe(body);
    const html = renderToStaticMarkup(<PluginPanelHistoryRecord record={record} read={read} busy={false} onRead={vi.fn()} />);
    expect(html).toContain("<summary><strong>actual-host-result</strong><span>Event 7</span></summary>");
    expect(html.indexOf("view-under-details")).toBeGreaterThan(html.indexOf("</summary>"));
    expect(html).toContain(body.replace(/"/g, "&quot;"));
    expect(html).toContain("Complete record");
  });
});
