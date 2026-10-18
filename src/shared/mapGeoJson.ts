import type { MapGeoJson } from "./maps";

const object = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);
/** Validate every supplied coordinate. Foreign GeoJSON members stay inert data. */
export function checkedMapGeoJson(value: unknown): { geoJson: MapGeoJson; bounds: [number, number, number, number] | null } {
  if (!object(value) || value.type !== "FeatureCollection" || !Array.isArray(value.features)) throw new Error("Map data must be a GeoJSON FeatureCollection.");
  let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  const position = (point: unknown): void => {
    if (!Array.isArray(point) || point.length < 2 || point.some(part => typeof part !== "number" || !Number.isFinite(part)) ||
      typeof point[0] !== "number" || typeof point[1] !== "number" || point[0] < -180 || point[0] > 180 || point[1] < -90 || point[1] > 90) throw new Error("Map coordinates must be finite WGS84 longitude/latitude values.");
    minLon = Math.min(minLon, point[0]); maxLon = Math.max(maxLon, point[0]); minLat = Math.min(minLat, point[1]); maxLat = Math.max(maxLat, point[1]);
  };
  const coordinates = (raw: unknown, depth: number): void => {
    const pending: Array<[unknown, number]> = [[raw, depth]];
    while (pending.length) { const [part, remaining] = pending.pop()!;
      if (remaining === 0) { position(part); continue; }
      if (!Array.isArray(part)) throw new Error("Map geometry has a malformed coordinate array.");
      for (const child of part) pending.push([child, remaining - 1]);
    }
  };
  for (const feature of value.features) {
    if (!object(feature) || feature.type !== "Feature" || !Object.hasOwn(feature, "geometry") ||
      feature.properties !== null && !object(feature.properties) || feature.id !== undefined && typeof feature.id !== "string" && typeof feature.id !== "number") throw new Error("Map feature metadata is invalid.");
    const pending = [feature.geometry];
    while (pending.length) { const geometry = pending.pop(); if (geometry === null) continue;
      if (!object(geometry)) throw new Error("Map geometry is invalid.");
      if (geometry.type === "GeometryCollection") { if (!Array.isArray(geometry.geometries)) throw new Error("Map geometry collection is invalid."); for (const child of geometry.geometries) pending.push(child); continue; }
      const depth = { Point: 0, MultiPoint: 1, LineString: 1, MultiLineString: 2, Polygon: 2, MultiPolygon: 3 }[String(geometry.type)];
      if (depth === undefined) throw new Error("Unsupported GeoJSON geometry."); coordinates(geometry.coordinates, depth);
    }
  }
  return { geoJson: value as unknown as MapGeoJson, bounds: Number.isFinite(minLon) ? [minLon, minLat, maxLon, maxLat] : null };
}
