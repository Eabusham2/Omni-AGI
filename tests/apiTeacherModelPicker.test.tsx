import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { ApiTeacherModelPicker, modelChoicePage, modelDiscoveryIds, modelDiscoveryRequestIsCurrent } from "../src/renderer/src/ApiTeacherModelPicker";
import type { ApiTeacherModelDiscoveryResult, OmniApi } from "../src/shared/types";
const result: ApiTeacherModelDiscoveryResult = { format: "omni-api-teacher-model-discovery", version: 1, provider: "openai", requestId: "request-a", sourceEndpoint: "https://api.openai.com/v1/models", discoveredAt: "2026-10-13T00:00:00Z", pageCount: 1, complete: true, generationCompatibilityVerified: false, models: [{ provider: "openai", id: "synthetic-b" }, { provider: "openai", id: "synthetic-a" }, { provider: "openai", id: "synthetic-a" }] };
describe("provider model picker keeps manual choice and exact request ownership", () => {
  it("lists only complete exact-owner models and never treats listing as compatibility verification", () => {
    expect(modelDiscoveryIds(result, "openai", "request-a")).toEqual(["synthetic-a", "synthetic-b"]);
    for (const changed of [{ ...result, provider: "gemini" }, { ...result, requestId: "other" }, { ...result, complete: false }, { ...result, generationCompatibilityVerified: true }]) expect(() => modelDiscoveryIds(changed as ApiTeacherModelDiscoveryResult, "openai", "request-a")).toThrow();
  });
  it("rejects cross-provider or invalid model entries rather than silently substituting a different choice", () => {
    for (const model of [{ provider: "gemini", id: "synthetic-a" }, { provider: "openai", id: "<script>" }]) expect(() => modelDiscoveryIds({ ...result, models: [model] } as ApiTeacherModelDiscoveryResult, "openai", "request-a")).toThrow();
  });
  it("renders the exact manual value while unconfigured without requesting, creating, saving or exposing any key", () => {
    const api = new Proxy({}, { get: () => { throw new Error("Unexpected API operation"); } }) as OmniApi["teacher"];
    const html = renderToStaticMarkup(<ApiTeacherModelPicker provider="openai" model="my-manual-model" configured={false} credentialEpoch={0} api={api} onChange={() => { throw new Error("Model changed without user action"); }} />);
    expect(html).toContain('value="my-manual-model"'); expect(html).toContain("Add your provider key"); expect(html).not.toContain("Refresh model list"); expect(html).not.toContain("password");
  });
  it("windows the explicit chooser without a total-model ceiling and searches the full returned catalog", () => {
    const ids = Array.from({ length: 205 }, (_, index) => `synthetic-${String(index).padStart(3, "0")}`);
    expect(modelChoicePage(ids, "", 0)).toMatchObject({ total: 205, start: 0 }); expect(modelChoicePage(ids, "", 0).ids).toHaveLength(100);
    expect(modelChoicePage(ids, "", 2).ids).toEqual(ids.slice(200));
    expect(modelChoicePage(ids, "SYNTHETIC-204", 99)).toEqual({ ids: ["synthetic-204"], total: 1, start: 0 });
  });
  it("retires a late Stop reply once a later request owns the same mounted picker", () => {
    const owner = {}, nextOwner = {};
    expect(modelDiscoveryRequestIsCurrent(owner, owner, "request-a", "request-a")).toBe(true);
    // A settled first discovery can be refreshed while its Stop transport is
    // still finishing. Its late status/error/finally must not change request-b.
    expect(modelDiscoveryRequestIsCurrent(owner, owner, "request-b", "request-a")).toBe(false);
    expect(modelDiscoveryRequestIsCurrent(owner, owner, undefined, "request-a")).toBe(false);
    expect(modelDiscoveryRequestIsCurrent(nextOwner, owner, "request-a", "request-a")).toBe(false);
    expect(modelDiscoveryRequestIsCurrent(null, owner, "request-a", "request-a")).toBe(false);
  });
});
