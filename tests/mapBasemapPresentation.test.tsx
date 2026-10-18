import React from "react";
import { createHash } from "node:crypto";
import { renderToStaticMarkup } from "react-dom/server";
import { expect, it, vi } from "vitest";
import { MapBasemapPanel } from "../src/renderer/src/MapBasemapPanel";
import { MapRasterView } from "../src/renderer/src/MapRasterView";
import { checkedMapRasterResult, mapRasterFromAction } from "../src/renderer/src/mapRasterPresentation";
import { declaredMapBasemapCapabilityPack } from "../src/main/mapBasemapCapabilityPack";
import type { MapRasterResult } from "../src/shared/mapBasemaps";
import type { ActionEvent } from "../src/shared/types";

const bytes = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=", "base64");
const result: MapRasterResult = { format: "omni-map-raster", version: 1, provider: "mapbox-static-image", profileId: "fixture", profileRevision: 1, label: "Supplied map fixture",
  width: 1, height: 1, bounds: [0, 0, 1, 1], coordinateReferenceSystem: "EPSG:3857", images: [{ dataUrl: `data:image/png;base64,${bytes.toString("base64")}`, sha256: createHash("sha256").update(bytes).digest("hex"),
    bytes: bytes.length, mime: "image/png", width: 1, height: 1, bounds: [0, 0, 1, 1], extent: "provider-fit-to-request", cached: false, sourceObservedAt: "2026-10-17T00:00:00.000Z" }],
  attribution: "Synthetic attribution", attributionUrl: "https://example.invalid/attribution", attributionLinks: [{ label: "Synthetic attribution", url: "https://example.invalid/attribution" }],
  rights: { retentionLearningAndRedistributionAuthorized: true, authorizationReference: "Synthetic fixture, not a license grant", licenseUrl: "https://example.invalid/license" },
  provenance: { source: "https://api.mapbox.com/styles/v1/fixture/style/static", observedAt: "2026-10-17T00:00:00.000Z", requestSha256: "a".repeat(64), brandingRequested: true, automaticRetry: false },
  basemapDownloaded: true, untrustedTaskData: true };
const event = (): ActionEvent => ({ id: "action", brainId: "fixture", state: "complete", createdAt: "2026-10-17T00:00:00.000Z", updatedAt: "2026-10-17T00:00:00.000Z",
  action: { kind: "tool", source: "brain", toolId: "maps.display", action: "basemap", arguments: {} }, execution: { id: "result", toolId: "maps.display", action: "basemap", state: "complete", startedAt: "2026-10-17T00:00:00.000Z", output: result } });

it("keeps optional provider setup collapsed and inert, without automatic source selection or provider calls", () => {
  const api = { list: vi.fn(), save: vi.fn(), remove: vi.fn() }, html = renderToStaticMarkup(<MapBasemapPanel api={api} />);
  expect(html).toContain("Optional basemaps"); expect(html).toContain("offline coordinate view remains the default"); expect(html).not.toContain("<details open"); expect(html).not.toContain("<img");
  expect(api.list).not.toHaveBeenCalled(); expect(api.save).not.toHaveBeenCalled(); expect(renderToStaticMarkup(<MapBasemapPanel />)).toBe("");
});
it("renders retained bytes only and labels provider-fitted bounds honestly rather than fabricating exact geographic edges", () => {
  const html = renderToStaticMarkup(<MapRasterView result={result} />);
  expect(html).toContain("Requested area"); expect(html).toContain("visible edges may extend"); expect(html).toContain("does not determine");
  expect(html).toContain("<canvas"); expect(html).not.toContain("<img"); expect(html).not.toContain("<iframe"); expect(html).not.toContain("src=\"https:");
  expect(checkedMapRasterResult({ ...result, images: [{ ...result.images[0]!, dataUrl: "data:image/svg+xml;base64,PHN2Zy8+" }] })).toBeUndefined();
  expect(checkedMapRasterResult({ ...result, attributionLinks: [{ label: "bad", url: "javascript:alert(1)" }] })).toBeUndefined();
});
it("mounts only completed matching map results, leaving original retained source data unchanged", () => {
  const action = event(), before = JSON.stringify(action); expect(mapRasterFromAction(action)).toBe(result); expect(JSON.stringify(action)).toBe(before);
  expect(mapRasterFromAction({ ...action, state: "running" })).toBeUndefined();
  expect(mapRasterFromAction({ ...action, execution: { ...action.execution!, action: "show" } })).toBeUndefined();
  expect(checkedMapRasterResult({ ...result, images: [{ ...result.images[0]!, extent: "exact-xyz-tile" }] })).toBeUndefined();
});
it("adds only declared map schemas/examples with no credential field, hidden prompt or fabricated execution", () => {
  const pack = declaredMapBasemapCapabilityPack(); expect(pack.schemas[0]).toMatchObject({ id: "maps.display", actions: ["basemaps", "basemap"] });
  expect(pack.records).toHaveLength(2); expect(pack.executedHostActions).toBe(false); expect(pack.rewardModel).toBe(false);
  expect(JSON.stringify(pack.schemas)).not.toMatch(/credential|access_token|bearer|systemPrompt/); expect(pack.sha256).toMatch(/^[a-f0-9]{64}$/);
});
