import type { ActionEvent } from "../../shared/types";
import type { MapDisplayResult, MapSearchResult } from "../../shared/maps";
import { checkedMapGeoJson } from "../../shared/mapGeoJson";

/** A retained result is display data, never a live connection or permission. */
export function mapDisplayFromAction(event: ActionEvent): MapDisplayResult | undefined {
  if (event.state !== "complete" || event.execution?.state !== "complete" || event.action.kind !== "tool" ||
    typeof event.action.toolId !== "string" || !["maps.search", "maps.display"].includes(event.action.toolId) || event.execution.toolId !== event.action.toolId) return undefined;
  const output = event.execution.output;
  const value = output && typeof output === "object" && !Array.isArray(output) ? output as Record<string, unknown> : undefined;
  if (!value || value.version !== 1 || value.untrustedTaskData !== true || typeof value.attribution !== "string") return undefined;
  try {
    const checked = checkedMapGeoJson(value.geoJson);
    if (value.format === "omni-map-display" && event.action.toolId === "maps.display" && event.action.action === "show" &&
      value.displayMode === "offline-coordinate-canvas" && value.basemapDownloaded === false && value.coordinateReferenceSystem === "EPSG:4326") return { ...(value as unknown as MapDisplayResult), ...checked };
    if (value.format === "omni-map-search" && event.action.toolId === "maps.search" && event.action.action === "search" && value.provider === "private-photon" &&
      value.provenance && typeof value.connectionId === "string" && typeof value.connectionRevision === "number" && typeof value.query === "string") {
      const source = value as unknown as MapSearchResult;
      return { format: "omni-map-display", version: 1, displayMode: "offline-coordinate-canvas", coordinateReferenceSystem: "EPSG:4326", dataOrigin: "search-result",
        ...checked, attribution: source.attribution, attributionUrl: source.attributionUrl, basemapDownloaded: false, untrustedTaskData: true,
        search: { connectionId: source.connectionId, connectionRevision: source.connectionRevision, query: source.query, provenance: source.provenance } };
    }
  } catch { /* malformed imported data never becomes an active control */ }
  return undefined;
}

export function safeMapAttributionUrl(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  try { const url = new URL(value); return url.protocol === "https:" && !url.username && !url.password ? url.href : undefined; } catch { return undefined; }
}

export function transformedMapView(current: [number, number, number, number], factor: number, dx = 0, dy = 0): [number, number, number, number] {
  const width = (current[2] - current[0]) * factor, height = (current[3] - current[1]) * factor;
  if (!Number.isFinite(width + height) || width <= 0 || height <= 0) return current;
  const x = current[0] / 2 + current[2] / 2 + dx * width, y = current[1] / 2 + current[3] / 2 + dy * height;
  const result: [number, number, number, number] = [x - width / 2, y - height / 2, x + width / 2, y + height / 2];
  return result.every(Number.isFinite) && result[2] > result[0] && result[3] > result[1] ? result : current;
}
