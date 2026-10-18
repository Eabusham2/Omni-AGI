import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { readFileSync } from "node:fs";
import { expect, it } from "vitest";
import { BuildProgressStatus, StorageOperationTiming } from "../src/renderer/src/BuildProgressStatus";
import { ResponseOutputCounter } from "../src/renderer/src/ResponseOutputCounter";
import type { BrainStorageOperationEvent } from "../src/shared/types";

it("renders nonterminal 99 percent with measured elapsed and a phase estimate instead of false completion", () => {
  const html = renderToStaticMarkup(<BuildProgressStatus progress={{ sequence: 9, phase: "packing", progress: 1, label: "Packing", elapsedMs: 30000 }} />);
  expect(html).toContain("99%"); expect(html).toContain("30s elapsed"); expect(html).toContain("Estimating remaining time · Packing");
  expect(html).not.toContain("Durably complete");
  const storage = { state: "running", phase: "finalizing", elapsedMs: 60000 } as BrainStorageOperationEvent;
  expect(renderToStaticMarkup(<dl><StorageOperationTiming operation={storage} /></dl>)).toContain("Estimating remaining time · Finalizing");
});
it("exposes precise output counts and speed on hover without changing compact visible output", () => {
  const detail = "Live response output: 12,345,678 output tokens used. Average delivered speed since first output: 12.34 tokens per second.";
  const html = renderToStaticMarkup(<ResponseOutputCounter presentation={{ phase: "streaming", label: "12.346M tokens · 12.3 tok/s", ariaLabel: detail, outputTokens: 12345678, tokensPerSecond: 12.34 }} />);
  expect(html).toContain(`title="${detail}"`); expect(html).toContain(`aria-label="${detail}"`); expect(html).toContain("12.346M tokens · 12.3 tok/s");
});
it("keeps checksum diagnostics out of basic Build while retaining the authoritative parameter total", () => {
  const app = readFileSync(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
  const start = app.indexOf('{buildProgress && (!initializationRunningActive || page === "library") ? (');
  const end = app.indexOf("{confirmingInitializationCancel", start);
  expect(start >= 0 && end > start).toBe(true);
  const basicBuild = app.slice(start, end);
  expect(basicBuild.includes("Composite neural checksum")).toBe(false);
  expect(basicBuild.includes("parameterChecksumChanged")).toBe(false);
  expect(app.includes("Total neural parameters")).toBe(true); expect(basicBuild.includes("<BuildProgressStatus progress={buildProgress}")).toBe(true);
  expect(app.includes("<ResponseOutputCounter presentation={responseOutputPresentation}")).toBe(true);
});
