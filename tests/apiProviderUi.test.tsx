import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it } from "vitest";
import { ExternalAdvicePanel } from "../src/renderer/src/ExternalAdvicePanel";
import { ApiTeacherModelPicker, ApiTeacherExcludedModels, modelDiscoveryExclusions } from "../src/renderer/src/ApiTeacherModelPicker";
import type { ApiTeacherModelDiscoveryResult, OmniApi } from "../src/shared/types";
import { ExternalAdviceModelPicker, adviceDiscoveryIds } from "../src/renderer/src/ExternalAdviceModelPicker";
import type { ExternalAdviceChannelConfiguration, ExternalAdviceModelDiscoveryResult } from "../src/shared/externalAdvice";

it("shows explicit Ollama alongside OpenRouter and keeps local model inventory keyless without invoking API while rendering", () => {
  const advice = renderToStaticMarkup(<ExternalAdvicePanel brainId="synthetic-owner" hidden={false} onClose={() => undefined} />);
  expect(advice).toContain('value="ollama"'); expect(advice).toContain("Ollama (local server)"); expect(advice).toContain('value="openrouter"');
  const api = new Proxy({}, { get: () => { throw new Error("SSR invoked a provider"); } }) as OmniApi["teacher"];
  const picker = renderToStaticMarkup(<ApiTeacherModelPicker provider="ollama" model="local:latest" configured={false} credentialEpoch={0} api={api} onChange={() => { throw new Error("manual choice changed"); }} />);
  expect(picker).toContain('value="local:latest"'); expect(picker).toContain("Refresh model list"); expect(picker).toContain("no key or automatic model pull");
  expect(picker).not.toContain("Add your provider key");
});

it("keeps every excluded cloud ID/reason inspectable with bounded visible rows instead of silent truncation", () => {
  const rows = Array.from({ length: 205 }, (_, index) => ({ id: `model-${index}:cloud`, reason: "Separate cloud consent is required." }));
  const result = { excludedModels: rows, excludedModelCount: 205 } as ApiTeacherModelDiscoveryResult;
  expect(modelDiscoveryExclusions(result)).toHaveLength(205);
  expect(() => modelDiscoveryExclusions({ ...result, excludedModelCount: 1 })).toThrow(/inconsistent/);
  const html = renderToStaticMarkup(<ApiTeacherExcludedModels rows={rows} />);
  expect(html).toContain("205 models require separate cloud consent"); expect(html.match(/<li>/g)).toHaveLength(100);
  expect(html).toContain("More excluded models"); expect(html).toContain("model-0:cloud");
});

it("advice model catalog refuses wrong owner/channel/revision/key/endpoint and renders explicit-click-only controls", () => {
  const configuration: ExternalAdviceChannelConfiguration = { enabled: false, provider: "ollama", model: "local:latest", endpoint: "http://127.0.0.1:19999/v1",
    maxOutputTokens: 50, context: { includeTask: false, includeActivity: false, maxCharacters: 0 } };
  const expected = { brainId: "one", channel: "assistant" as const, requestId: "catalog", settingsRevision: 2, credentialRevision: 0, configuration };
  const result: ExternalAdviceModelDiscoveryResult = { format: "omni-api-teacher-model-discovery", version: 1, brainId: "one", channel: "assistant",
    requestId: "catalog", settingsRevision: 2, credentialRevision: 0, provider: "ollama", endpoint: configuration.endpoint,
    sourceEndpoint: "http://127.0.0.1:19999/api/tags", discoveredAt: "2026-10-15T00:00:00.000Z", pageCount: 1,
    complete: true, generationCompatibilityVerified: false, models: [{ provider: "ollama", id: "local:latest" }] };
  expect(adviceDiscoveryIds(result, expected)).toEqual(["local:latest"]);
  for (const changed of [{ brainId: "other" }, { channel: "reviewer" }, { settingsRevision: 3 }, { credentialRevision: 1 },
    { endpoint: "http://127.0.0.1:11434/v1" }, { sourceEndpoint: "https://untrusted.invalid/api/tags" }, { provider: "openrouter" }]) {
    expect(() => adviceDiscoveryIds({ ...result, ...changed } as ExternalAdviceModelDiscoveryResult, expected)).toThrow();
  }
  const api = new Proxy({}, { get: () => { throw new Error("Rendering auto-dispatched metadata"); } });
  const html = renderToStaticMarkup(<ExternalAdviceModelPicker {...expected} model="local:latest" canDiscover={true} api={api as never} onChange={() => { throw new Error("Model changed"); }} />);
  expect(html).toContain("Refresh advice model list"); expect(html).not.toContain("Stop model discovery");
});
