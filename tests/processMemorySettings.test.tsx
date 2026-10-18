import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { ProcessMemorySettings } from "../src/renderer/src/ProcessMemorySettings";
describe("process memory optional setting", () => {
  it("starts disabled and cannot arm or probe a process while rendering", () => {
    const api = { preferences: vi.fn(), update: vi.fn() };
    const html = renderToStaticMarkup(<ProcessMemorySettings brainId="brain-a" api={api} />);
    expect(api.preferences).not.toHaveBeenCalled(); expect(api.update).not.toHaveBeenCalled();
    expect(html).toContain('type="checkbox" disabled=""'); expect(html).not.toContain("checked=");
    expect(html).toContain("system.memory"); expect(html).toContain("not unrestricted physical RAM");
  });
  it("does not imply a browser preview has OS memory access", () => {
    const html = renderToStaticMarkup(<ProcessMemorySettings brainId="brain-a" />);
    expect(html).toContain("Available in the desktop app"); expect(html).not.toContain("pid");
  });
});
