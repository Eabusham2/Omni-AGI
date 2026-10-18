import React, { isValidElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { RendererStartupGate, rendererStartupMode } from "../src/renderer/src/RendererStartupGate";

describe("real renderer startup versus explicit inert preview", () => {
  it("requires the desktop bridge normally and refuses partial bridge objects even with the preview flag", () => {
    for (const bridge of [undefined, null, {}, false, { window: { platform: "not callable" }, brain: { list: () => [] } }]) {
      expect(rendererStartupMode(bridge, "")).toBe("unavailable");
    }
    expect(rendererStartupMode({}, "?omni-preview=1")).toBe("unavailable");
    expect(rendererStartupMode({ get window() { throw new Error("broken bridge"); } }, "?omni-preview=1")).toBe("unavailable");
    const bridge = { window: { platform: vi.fn() }, brain: { list: vi.fn() } };
    expect(rendererStartupMode(bridge, "")).toBe("desktop");
    expect(rendererStartupMode(bridge, "?omni-preview=1")).toBe("desktop");
    expect(bridge.window.platform).not.toHaveBeenCalled(); expect(bridge.brain.list).not.toHaveBeenCalled();
  });

  it("accepts only the exact explicit no-bridge preview flag and does not create a bridge or permission", () => {
    expect(rendererStartupMode(undefined, "?omni-preview=1")).toBe("preview");
    for (const search of ["?preview=1", "?omni-preview=true", "?omni-preview=0", "?omni-preview=1&omni-preview=0", "?omni-preview=1&omni-preview=1"]) {
      expect(rendererStartupMode(undefined, search)).toBe("unavailable");
    }
  });

  it("renders a truthful failure/reload surface without ever rendering synthetic App children", () => {
    const child = vi.fn(() => <div>Design preview · synthetic Aster</div>), reload = vi.fn();
    const tree = RendererStartupGate({ bridge: undefined, search: "", onReload: reload, children: React.createElement(child) });
    const markup = renderToStaticMarkup(tree);
    expect(markup).toContain("Desktop connection unavailable");
    expect(markup).toContain('role="alert"'); expect(markup).toContain("Reload window");
    expect(markup).not.toContain("Design preview"); expect(markup).not.toContain("Aster"); expect(child).not.toHaveBeenCalled();
    const section = isValidElement(tree) ? (tree.props as { children: ReactElement }).children : undefined;
    const button = (section?.props as { children: ReactElement[] }).children.find(element => isValidElement(element) && element.type === "button")!;
    (button.props as { onClick(): void }).onClick(); expect(reload).toHaveBeenCalledOnce();
    const preview = renderToStaticMarkup(<RendererStartupGate bridge={undefined} search="?omni-preview=1" onReload={reload}><div>Explicit fixture preview</div></RendererStartupGate>);
    expect(preview).toContain("Explicit fixture preview");
  });
});
