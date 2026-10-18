/** New inert read-status branches; no API, renderer, engine or model is run. */
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { workspaceReadStatus } from "../src/renderer/src/workspaceReadStatus";

describe("initial workspace reads without a snapshot", () => {
  const text = { pending: "Reading…", unavailable: "Read unavailable. Refresh to retry.", disconnected: "Desktop connection unavailable." };
  it("does not mistake a present connection for an in-flight read after failure", () => {
    const message = workspaceReadStatus(true, true, text);
    expect(message).toBe(text.unavailable); expect(message).not.toBe(text.pending);
    expect(renderToStaticMarkup(<p role="status">{message}</p>)).toBe('<p role="status">Read unavailable. Refresh to retry.</p>');
  });
  it("keeps disconnected status truthful before and after the failed refresh settles", () => {
    expect(workspaceReadStatus(false, false, text)).toBe(text.disconnected);
    expect(workspaceReadStatus(false, true, text)).toBe(text.disconnected);
  });
  it("retains the genuine initial pending branch until data or a failure arrives", () => {
    expect(workspaceReadStatus(true, false, text)).toBe(text.pending);
  });
});
