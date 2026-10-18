export interface MapConnection {
  id: string;
  label: string;
  provider: "private-photon";
  endpoint: string;
  attribution: string;
  attributionUrl: string;
  enabled: boolean;
  revision: number;
  credentialConfigured: boolean;
  credentialPersistence: "encrypted" | "session";
  minimumIntervalMs: number;
  cacheTtlMs: number;
  timeoutMs: number;
  /** Results participate in ordinary tool history, learning and full exports. */
  retainedResultsAccepted: true;
}
export interface MapConnectionSave {
  id: string;
  expectedRevision: number;
  label: string;
  endpoint: string;
  attribution: string;
  attributionUrl: string;
  enabled: boolean;
  retainedResultsAccepted: true;
  /** Omit to keep; null to clear. Never returned in status or model tools. */
  bearerToken?: string | null;
  minimumIntervalMs?: number;
  cacheTtlMs?: number;
  timeoutMs?: number;
}
export interface MapConnectionsApi {
  basemaps?: import("./mapBasemaps").MapBasemapsApi;
  list(): Promise<MapConnection[]>;
  save(value: MapConnectionSave): Promise<MapConnection>;
  remove(id: string, expectedRevision: number): Promise<void>;
}
export interface MapGeoJson {
  type: "FeatureCollection";
  features: Array<{ type: "Feature"; geometry: Record<string, unknown> | null; properties: Record<string, unknown> | null; [key: string]: unknown }>;
  [key: string]: unknown;
}
export interface MapSearchResult {
  format: "omni-map-search";
  version: 1;
  provider: "private-photon";
  connectionId: string;
  connectionRevision: number;
  query: string;
  geoJson: MapGeoJson;
  bounds: [number, number, number, number] | null;
  attribution: string;
  attributionUrl: string;
  provenance: { endpoint: string; observedAt: string; responseSha256: string; cached: boolean; sourceObservedAt: string; sourceEnvelope: "geojson" | "photon-features" };
  searchResultId?: string;
  cacheStored: boolean;
  warning?: string;
  untrustedTaskData: true;
}
export interface MapDisplayResult {
  format: "omni-map-display";
  version: 1;
  displayMode: "offline-coordinate-canvas";
  coordinateReferenceSystem: "EPSG:4326";
  dataOrigin: "search-result" | "supplied-geojson";
  geoJson: MapGeoJson;
  bounds: [number, number, number, number] | null;
  attribution: string;
  attributionUrl?: string;
  search?: Pick<MapSearchResult, "connectionId" | "connectionRevision" | "query" | "provenance">;
  /** No street tiles or geography are invented or downloaded. */
  basemapDownloaded: false;
  untrustedTaskData: true;
}
