import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { MapConnectionPanel } from "../src/renderer/src/MapConnectionPanel";
import { MapResultView } from "../src/renderer/src/MapResultView";
import { McpPresetPicker } from "../src/renderer/src/McpPresetPicker";
import { mapDisplayFromAction, safeMapAttributionUrl, transformedMapView } from "../src/renderer/src/mapPresentation";
import type { ActionEvent } from "../src/shared/types";
import type { MapDisplayResult } from "../src/shared/maps";

const display: MapDisplayResult = { format: "omni-map-display", version: 1, displayMode: "offline-coordinate-canvas", coordinateReferenceSystem: "EPSG:4326",
  dataOrigin: "supplied-geojson", geoJson: { type: "FeatureCollection", features: [{ type: "Feature", properties: { label: "<script>raw</script>" }, geometry: { type: "Point", coordinates: [13, 52] } }] },
  bounds: [13, 52, 13, 52], attribution: "<script>source</script>", attributionUrl: "https://example.invalid/licence", basemapDownloaded: false, untrustedTaskData: true };
describe("map/template presentation (static structure, not actual canvas/visual acceptance)", () => {
  it("labels offline coordinates honestly, preserves attribution and makes no remote iframe, tile or script", () => {
    const html = renderToStaticMarkup(<MapResultView result={display} />);
    expect(html).toContain("Offline coordinate canvas"); expect(html).toContain("No street basemap was downloaded"); expect(html).toContain("&lt;script&gt;source&lt;/script&gt;");
    expect(html).toContain("Pan west"); expect(html).not.toContain("<iframe"); expect(html).not.toContain("<script>"); expect(html).not.toContain("<img");
    expect(safeMapAttributionUrl("javascript:alert(1)")).toBeUndefined(); expect(safeMapAttributionUrl("https://user:password@example.invalid")).toBeUndefined();
    const tiny: [number, number, number, number] = [13, 52, 13 + 1e-14, 52 + 1e-14];
    expect(transformedMapView(tiny, 1e-6)).toBe(tiny); expect(transformedMapView([-180, -90, 180, 90], Number.MAX_VALUE)).toEqual([-180, -90, 180, 90]);
  });
  it("accepts only completed matching map effects and never changes original returned data", () => {
    const event: ActionEvent = { id: "action", brainId: "brain", state: "complete", createdAt: "2026-10-11T00:00:00Z", updatedAt: "2026-10-11T00:00:00Z",
      action: { kind: "tool", source: "brain", toolId: "maps.display", action: "show", arguments: {} },
      execution: { id: "effect", toolId: "maps.display", action: "show", state: "complete", startedAt: "2026-10-11T00:00:00Z", output: display } };
    const before = JSON.stringify(event.execution!.output); expect(mapDisplayFromAction(event)?.geoJson).toEqual(display.geoJson);
    expect(JSON.stringify(event.execution!.output)).toBe(before);
    expect(mapDisplayFromAction({ ...event, state: "running" })).toBeUndefined();
    expect(mapDisplayFromAction({ ...event, execution: { ...event.execution!, toolId: "mcp.unrelated.tool" } })).toBeUndefined();
    const found = { format: "omni-map-search", version: 1, provider: "private-photon", connectionId: "private", connectionRevision: 1, query: "supplied place",
      geoJson: display.geoJson, bounds: display.bounds, attribution: display.attribution, attributionUrl: display.attributionUrl, untrustedTaskData: true, cacheStored: true,
      provenance: { endpoint: "http://127.0.0.1:2322/api/", observedAt: event.createdAt, sourceObservedAt: event.createdAt, responseSha256: "a".repeat(64), cached: false, sourceEnvelope: "geojson" } };
    const searchEvent = { ...event, action: { ...event.action, toolId: "maps.search", action: "search" }, execution: { ...event.execution!, toolId: "maps.search", action: "search", output: found } };
    expect(mapDisplayFromAction(searchEvent)?.search?.provenance).toEqual(found.provenance);
  });
  it("hides unavailable preview configuration and keeps template selection inert", () => {
    expect(renderToStaticMarkup(<MapConnectionPanel />)).toBe("");
    const api = { list: vi.fn(), save: vi.fn(), remove: vi.fn() }, selected = vi.fn();
    expect(renderToStaticMarkup(<MapConnectionPanel api={api} />)).toContain("Add private connection"); expect(api.list).not.toHaveBeenCalled(); expect(api.save).not.toHaveBeenCalled();
    const html = renderToStaticMarkup(<McpPresetPicker onChoose={selected} />); expect(html).toContain("Reviewed connection templates"); expect(html).toContain("Adobe Express Developer"); expect(selected).not.toHaveBeenCalled();
  });
});
