/** Device-owned configuration. Neither secrets nor a permission grant are portable. */
export type MapRasterSource =
  | { kind: "raster-xyz"; template: string; tileSize: number; minimumZoom: number; maximumZoom: number }
  | { kind: "mapbox-static-image"; username: string; styleId: string };
export type MapRasterAuthentication =
  | { kind: "none" }
  | { kind: "bearer" }
  | { kind: "query"; parameter: string };
export interface MapRasterRights {
  /** A user declaration, not a legal determination by Omni. */
  retentionLearningAndRedistributionAuthorized: true;
  authorizationReference: string;
  licenseUrl: string;
}
export interface MapBasemapProfile {
  id: string;
  revision: number;
  label: string;
  enabled: boolean;
  source: MapRasterSource;
  authentication: MapRasterAuthentication;
  credentialConfigured: boolean;
  credentialPersistence: "encrypted" | "session";
  attribution: string;
  attributionUrl: string;
  rights: MapRasterRights;
  minimumIntervalMs: number;
  cacheTtlMs: number;
  timeoutMs: number;
}
export interface MapBasemapSave extends Omit<MapBasemapProfile, "revision" | "credentialConfigured" | "credentialPersistence"> {
  expectedRevision: number;
  /** Omit to keep, null to clear; never returned or included in a tool schema. */
  credential?: string | null;
}
export interface MapBasemapsApi {
  list(): Promise<MapBasemapProfile[]>;
  save(value: MapBasemapSave): Promise<MapBasemapProfile>;
  remove(id: string, expectedRevision: number): Promise<void>;
}
export type MapBounds = [west: number, south: number, east: number, north: number];
export interface MapRasterRequest {
  profileId: string;
  expectedRevision: number;
  bounds: MapBounds;
  width: number;
  height: number;
  /** XYZ only. Omission chooses a geometric fit, not a provider/hardware benchmark. */
  zoom?: number;
}
export interface MapRasterImage {
  /** Exact received raster bytes, never a remote URL, SVG or executable document. */
  dataUrl: string;
  sha256: string;
  bytes: number;
  mime: "image/png" | "image/jpeg" | "image/webp";
  width: number;
  height: number;
  /** XYZ bounds are exact; static provider fitting may include a larger area. */
  bounds: MapBounds;
  extent: "exact-xyz-tile" | "provider-fit-to-request";
  tile?: { z: number; x: number; y: number };
  cached: boolean;
  sourceObservedAt: string;
}
export interface MapRasterResult {
  format: "omni-map-raster";
  version: 1;
  provider: MapRasterSource["kind"];
  profileId: string;
  profileRevision: number;
  label: string;
  /** User-requested viewport, not an assertion about a provider-fitted image edge. */
  bounds: MapBounds;
  width: number;
  height: number;
  coordinateReferenceSystem: "EPSG:3857";
  images: MapRasterImage[];
  attribution: string;
  attributionUrl: string;
  attributionLinks: Array<{ label: string; url: string }>;
  rights: MapRasterRights;
  provenance: {
    /** Redacted template/endpoint: no token, query credential or owner grant. */
    source: string;
    observedAt: string;
    requestSha256: string;
    brandingRequested: boolean;
    automaticRetry: false;
  };
  basemapDownloaded: true;
  untrustedTaskData: true;
}

export const MAP_MERCATOR_LATITUDE = 85.0511287798066;
export function checkedMapBounds(value: unknown): MapBounds {
  if (!Array.isArray(value) || value.length !== 4 || value.some(n => typeof n !== "number" || !Number.isFinite(n)) ||
    value[0] < -180 || value[2] > 180 || value[1] < -MAP_MERCATOR_LATITUDE || value[3] > MAP_MERCATOR_LATITUDE ||
    value[0] >= value[2] || value[1] >= value[3]) throw new Error("Choose west,south,east,north bounds within Web Mercator, with east > west and north > south; split antimeridian views explicitly.");
  return [...value] as MapBounds;
}
export function mercatorY(latitude: number): number {
  const radians = latitude * Math.PI / 180;
  return (1 - Math.log(Math.tan(radians) + 1 / Math.cos(radians)) / Math.PI) / 2;
}
export function mercatorLatitude(y: number): number {
  return Math.atan(Math.sinh(Math.PI * (1 - 2 * y))) * 180 / Math.PI;
}
