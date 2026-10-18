import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { CachedDecoderExecutionControls, cachedDecoderAction, cachedDecoderDeviceLabel, cachedDecoderStatusText } from "../src/renderer/src/CachedDecoderExecutionControls";
import type { CachedDecoderExecutionStatus } from "../src/shared/cachedDecoderExecution";

it("arms only after a current review and does not offer another arm during an active frontier", () => {
  expect(cachedDecoderAction("prepared", false).kind).toBe("prepare");
  expect(cachedDecoderAction("prepared", true).kind).toBe("arm");
  expect(cachedDecoderAction("awaiting-retained-frontier", true).kind).toBe("wait");
  expect(cachedDecoderAction("calibrating", true).kind).toBe("wait");
});
it("does not call fallback or an unmatched completion an adopted cached step", () => {
  const status = { phase: "completed", actualOperationConsumed: true, exactCacheAndSamplingMatched: false } as CachedDecoderExecutionStatus;
  expect(cachedDecoderStatusText(status)).toContain("without adopting");
  expect(cachedDecoderStatusText({ ...status, exactCacheAndSamplingMatched: true })).toBe("One matching cached step completed.");
  expect(cachedDecoderStatusText({ ...status, phase: "failed-fallback" })).not.toContain("Chat kept");
  expect(cachedDecoderDeviceLabel("cuda:2")).toBe("CUDA device 2");
});
it("does not initiate comparison or pretend to measure an unloaded instance", () => {
  const html = renderToStaticMarkup(<CachedDecoderExecutionControls brainId="fixture" loaded={false} foregroundTurnActive={false} dirty={false} />);
  expect(html).toContain("Load this instance before reviewing devices");
  expect(html).toContain("No retained-context comparison requested");
  expect(html).not.toContain("0/0"); expect(html).not.toContain("Matched step:");
});
