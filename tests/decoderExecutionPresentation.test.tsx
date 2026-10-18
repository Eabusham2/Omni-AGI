import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import { DecoderExecutionControls, decoderExecutionNextAction, decoderMilliseconds } from "../src/renderer/src/DecoderExecutionControls";
import type { DecoderExecutionApi } from "../src/shared/decoderExecution";

it("shows only the correct next user action rather than an automatic quiz or measurement", () => {
  expect(decoderExecutionNextAction(undefined, false).kind).toBe("capture");
  expect(decoderExecutionNextAction("awaiting-normal-input", false).kind).toBe("wait");
  expect(decoderExecutionNextAction("captured", false).kind).toBe("prepare");
  expect(decoderExecutionNextAction("prepared", false).kind).toBe("prepare");
  expect(decoderExecutionNextAction("prepared", true).kind).toBe("calibrate");
  expect(decoderExecutionNextAction("calibrating", true).kind).toBe("wait");
  expect(decoderExecutionNextAction("proposal", false).kind).toBe("apply");
});
it("keeps operation timings precise without implying complete-brain speed", () => {
  expect(decoderMilliseconds("123456789")).toBe("123.456 ms");
  expect(decoderMilliseconds("9007199254740993")).toBe(`${(9_007_199_254n).toLocaleString()}.740 ms`);
  expect(decoderMilliseconds("NaN")).toBe("unavailable");
  expect(decoderMilliseconds("1e99")).toBe("unavailable");
});
it("renders unloaded controls disabled without calling any native API", () => {
  const forbidden = vi.fn(async () => { throw new Error("No native call on initial render"); });
  const api: DecoderExecutionApi = { captureNext: forbidden, prepare: forbidden, calibrate: forbidden, apply: forbidden, status: forbidden, stop: forbidden };
  const html = renderToStaticMarkup(<DecoderExecutionControls api={api} brainId="fixture" loaded={false} foregroundTurnActive={false} dirty={false} />);
  expect(html).toContain("Load this instance before measuring"); expect(html).toContain("does not measure the whole brain");
  expect(html).toMatch(/<button[^>]+disabled=""[^>]*>Use next normal turn/);
  expect(forbidden).not.toHaveBeenCalled();
});
