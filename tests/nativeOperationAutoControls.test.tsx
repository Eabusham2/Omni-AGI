import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { NativeOperationAutoControls, nativeAutoStatusText, nativeAutoPendingText, nativeAutoWorkKindLabel } from "../src/renderer/src/NativeOperationAutoControls";
import type { NativeOperationAutoApi, NativeOperationAutoStatus } from "../src/shared/nativeOperationAuto";
it("renders collapsed metadata controls without executing any model/calibration/Load call", () => {
  const api = new Proxy({}, { get() { throw new Error("Rendering must not call the host"); } }) as NativeOperationAutoApi;
  const html = renderToStaticMarkup(<NativeOperationAutoControls api={api} brainId="a" loaded={false} foregroundTurnActive={false} dirty={false} />);
  expect(html).toContain("automatic operation placement"); expect(html).toContain("Load this instance"); expect(html).not.toContain("<details open");
  expect(html).toContain("does not run a benchmark"); expect(html).not.toContain("whole brain accelerated");
});
it("shows an acknowledged Stop while old metadata drains instead of claiming Auto is still pending", () => {
  expect(nativeAutoPendingText({ phase: "cancelled" } as NativeOperationAutoStatus, true)).toContain("Auto stopped.");
  expect(nativeAutoPendingText(undefined, true)).toContain("Waiting");
});
it("keeps route counts distinct from whole-model speed and allows an honest fallback status", () => {
  expect(nativeAutoStatusText({ phase: "active", qualifiedPlanCount: 2 } as NativeOperationAutoStatus)).toBe("2 qualified operation routes.");
  expect(nativeAutoStatusText({ phase: "failed-fallback" } as NativeOperationAutoStatus)).toContain("kept its ordinary route");
  expect(nativeAutoStatusText({ phase: "awaiting-operation" } as NativeOperationAutoStatus)).toContain("supported normal operation");
});
it("labels the three media artifact routes without exposing internal operation identifiers", () => {
  expect(nativeAutoWorkKindLabel("native.media.image-artifact")).toBe("Image artifact");
  expect(nativeAutoWorkKindLabel("native.media.audio-artifact")).toBe("Audio artifact");
  expect(nativeAutoWorkKindLabel("native.media.video-artifact")).toBe("Video artifact");
});
