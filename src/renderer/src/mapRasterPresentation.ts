import { checkedMapBounds, type MapRasterResult } from "../../shared/mapBasemaps";
import type { ActionEvent } from "../../shared/types";
import { safeMapAttributionUrl } from "./mapPresentation";

/** Retained data cannot select a provider, make a request, or grant device access. */
export function checkedMapRasterResult(raw: unknown): MapRasterResult | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const value = raw as MapRasterResult;
  if (value.format !== "omni-map-raster" || value.version !== 1 || value.basemapDownloaded !== true || value.untrustedTaskData !== true ||
    !["raster-xyz", "mapbox-static-image"].includes(value.provider) || value.coordinateReferenceSystem !== "EPSG:3857" ||
    !Number.isSafeInteger(value.width) || value.width <= 0 || !Number.isSafeInteger(value.height) || value.height <= 0 ||
    typeof value.profileId !== "string" || !Number.isSafeInteger(value.profileRevision) || value.profileRevision <= 0 || typeof value.label !== "string" ||
    typeof value.attribution !== "string" || !safeMapAttributionUrl(value.attributionUrl) || !Array.isArray(value.attributionLinks) || !value.attributionLinks.length ||
    !value.rights || value.rights.retentionLearningAndRedistributionAuthorized !== true || typeof value.rights.authorizationReference !== "string" || !safeMapAttributionUrl(value.rights.licenseUrl) ||
    !value.provenance || typeof value.provenance.source !== "string" || !/^[a-f0-9]{64}$/.test(value.provenance.requestSha256) || !Number.isFinite(Date.parse(value.provenance.observedAt)) ||
    value.provenance.automaticRetry !== false || !Array.isArray(value.images) || !value.images.length) return undefined;
  try {
    checkedMapBounds(value.bounds);
    if (value.attributionLinks.some(link => !link || typeof link.label !== "string" || !safeMapAttributionUrl(link.url))) return undefined;
    for (const image of value.images) {
      if (!image || !["image/png", "image/jpeg", "image/webp"].includes(image.mime) || !Number.isSafeInteger(image.width) || image.width <= 0 ||
        !Number.isSafeInteger(image.height) || image.height <= 0 || !Number.isSafeInteger(image.width * image.height * 4) || !Number.isSafeInteger(image.bytes) || image.bytes <= 0 ||
        !/^[a-f0-9]{64}$/.test(image.sha256) || typeof image.dataUrl !== "string" || !image.dataUrl.startsWith(`data:${image.mime};base64,`) ||
        !Number.isFinite(Date.parse(image.sourceObservedAt)) || typeof image.cached !== "boolean" ||
        image.extent !== (value.provider === "raster-xyz" ? "exact-xyz-tile" : "provider-fit-to-request")) return undefined;
      const encoded = image.dataUrl.slice(image.dataUrl.indexOf(",") + 1);
      const padding = (3 - image.bytes % 3) % 3;
      if (encoded.length !== 4 * Math.ceil(image.bytes / 3) || !/^[A-Za-z0-9+/]+={0,2}$/.test(encoded) ||
        (padding ? encoded.indexOf("=") !== encoded.length - padding : encoded.includes("="))) return undefined;
      checkedMapBounds(image.bounds);
      if (value.provider === "raster-xyz" && (!image.tile || ![image.tile.z, image.tile.x, image.tile.y].every(Number.isSafeInteger) || image.tile.z < 0 || image.tile.z > 52 ||
        image.tile.x < 0 || image.tile.y < 0 || image.tile.x >= 2 ** image.tile.z || image.tile.y >= 2 ** image.tile.z)) return undefined;
    }
    if (value.provider === "mapbox-static-image" && (value.images.length !== 1 || value.provenance.brandingRequested !== true)) return undefined;
    return value;
  } catch { return undefined; }
}
export function mapRasterFromAction(event: ActionEvent): MapRasterResult | undefined {
  if (event.state !== "complete" || event.execution?.state !== "complete" || event.action.kind !== "tool" || event.action.toolId !== "maps.display" ||
    event.action.action !== "basemap" || event.execution.toolId !== "maps.display" || event.execution.action !== "basemap") return undefined;
  return checkedMapRasterResult(event.execution.output);
}
