import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { deflateSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MapBasemapService } from "../src/main/mapBasemapService";
import { mapRasterImageInfo } from "../src/main/mapRasterImage";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import type { MapBasemapSave, MapRasterResult } from "../src/shared/mapBasemaps";

const roots: string[] = [], services: MapBasemapService[] = [];
afterEach(async () => { for (const service of services.splice(0)) service.close(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
function png(): Buffer {
  const crc = (data: Buffer): number => { let n = 0xffffffff; for (const byte of data) { n ^= byte; for (let i = 0; i < 8; i++) n = n & 1 ? (n >>> 1) ^ 0xedb88320 : n >>> 1; } return (n ^ 0xffffffff) >>> 0; };
  const chunk = (type: string, bytes: Buffer): Buffer => { const body = Buffer.concat([Buffer.from(type), bytes]), out = Buffer.alloc(body.length + 8); out.writeUInt32BE(bytes.length); body.copy(out, 4); out.writeUInt32BE(crc(body), body.length + 4); return out; };
  const header = Buffer.alloc(13); header.writeUInt32BE(1, 0); header.writeUInt32BE(1, 4); header[8] = 8; header[9] = 6;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", header), chunk("IDAT", deflateSync(Buffer.from([0, 255, 0, 0, 255]))), chunk("IEND", Buffer.alloc(0))]);
}
const pixel = png(), pixelBody = (): ArrayBuffer => Uint8Array.from(pixel).buffer, settings = (): MapBasemapSave => ({ id: "fixture", expectedRevision: 0, label: "Fixture map", enabled: true,
  source: { kind: "raster-xyz", template: "http://127.0.0.1:2322/{z}/{x}/{y}.png", tileSize: 1, minimumZoom: 0, maximumZoom: 0 }, authentication: { kind: "none" },
  attribution: "Fixture data", attributionUrl: "https://example.invalid/attribution", rights: { retentionLearningAndRedistributionAuthorized: true, authorizationReference: "Synthetic fixture permission, not a provider license", licenseUrl: "https://example.invalid/license" },
  minimumIntervalMs: 0, cacheTtlMs: 60000, timeoutMs: 30000 });
async function fixture(options: ConstructorParameters<typeof MapBasemapService>[2] = {}) {
  const root = await mkdtemp(join(tmpdir(), "omni-map-basemap-")); roots.push(root); const path = join(root, "maps.sqlite3");
  const secrets = new SecureSecretStore(join(root, "secrets.json"), { available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
  const fetcher = vi.fn(async (_url: unknown, _init?: RequestInit) => new Response(pixelBody(), { headers: { "Content-Type": "image/png", "Cache-Control": "max-age=60" } }));
  const service = new MapBasemapService(path, secrets, { fetch: fetcher as typeof fetch, ...options }); services.push(service); await service.initialize(); return { service, path, secrets, fetcher };
}
const request = { profileId: "fixture", expectedRevision: 1, bounds: [-180, -85, 180, 85], width: 1, height: 1, zoom: 0 };
const allowed = async (): Promise<void> => undefined;
const render = (service: MapBasemapService, args = request, signal = new AbortController().signal, current = allowed) => service.execute("brain", "basemap", args, signal, current) as Promise<MapRasterResult>;

describe("optional retained basemaps — synthetic HTTP, no model or public provider", () => {
  it("has no active default or provider request and rejects unsuited public services and absent rights", async () => {
    const f = await fixture(); expect(await f.service.list()).toEqual([]);
    expect(await f.service.execute("brain", "basemaps", {}, new AbortController().signal, allowed)).toMatchObject({ defaultProvider: null, publicServiceSelected: false, profiles: [] });
    await expect(f.service.save({ ...settings(), rights: { ...settings().rights, retentionLearningAndRedistributionAuthorized: false } } as never)).rejects.toThrow(/authorization/);
    await expect(f.service.save({ ...settings(), source: { ...settings().source, kind: "raster-xyz", template: "https://tile.openstreetmap.org/{z}/{x}/{y}.png", tileSize: 1, minimumZoom: 0, maximumZoom: 0 } })).rejects.toThrow(/Public OSM/);
    await expect(f.service.save({ ...settings(), source: { kind: "raster-xyz", template: "https://{z}.example.invalid/{x}/{y}.png", tileSize: 1, minimumZoom: 0, maximumZoom: 0 } })).rejects.toThrow(/origin/);
    expect(f.fetcher).not.toHaveBeenCalled();
  });
  it("returns exact licensed raster bytes, exact tile extent, source hashes and cached observations without a second dispatch", async () => {
    const f = await fixture(); await f.service.save(settings()); const first = await render(f.service), again = await render(f.service);
    expect(first).toMatchObject({ provider: "raster-xyz", coordinateReferenceSystem: "EPSG:3857", bounds: request.bounds, images: [{ extent: "exact-xyz-tile", tile: { z: 0, x: 0, y: 0 }, cached: false, width: 1, height: 1 }] });
    expect(Buffer.from(first.images[0]!.dataUrl.split(",")[1]!, "base64")).toEqual(pixel);
    expect(again.images[0]).toMatchObject({ sha256: first.images[0]!.sha256, sourceObservedAt: first.images[0]!.sourceObservedAt, cached: true });
    expect(f.fetcher).toHaveBeenCalledOnce(); expect(String(f.fetcher.mock.calls[0]?.[0])).toBe("http://127.0.0.1:2322/0/0/0.png");
    expect(first.attributionLinks).toEqual([{ label: settings().attribution, url: settings().attributionUrl }]);
  });
  it("keeps query keys main-only, has no geocoding branch, and requests Mapbox branding without pretending its fitted extent is exact", async () => {
    const f = await fixture(), key = "synthetic-long-mapbox-fixture-key";
    const profile = await f.service.save({ ...settings(), source: { kind: "mapbox-static-image", username: "owner", styleId: "reviewed-style" }, authentication: { kind: "query", parameter: "access_token" }, credential: key });
    const { zoom: _zoom, ...args } = request; const result = await render(f.service, args as typeof request), url = new URL(String(f.fetcher.mock.calls[0]?.[0]));
    expect(url.origin).toBe("https://api.mapbox.com"); expect(url.searchParams.get("access_token")).toBe(key); expect(url.searchParams.get("logo")).toBe("true"); expect(url.searchParams.get("attribution")).toBe("true"); expect(url.searchParams.get("padding")).toBe("0");
    expect(result.images[0]!.extent).toBe("provider-fit-to-request"); expect(result.attributionLinks.map(link => link.label)).toContain("© Mapbox");
    expect(JSON.stringify([profile, result, await f.service.list()])).not.toContain(key); expect((await readFile(f.path)).includes(Buffer.from(key))).toBe(false);
    await expect(f.service.execute("brain", "geocode", {}, new AbortController().signal, allowed)).rejects.toThrow(/Unsupported/);
  });
  it("does not forward a credential to a changed source, and tombstones removed profile revisions", async () => {
    const f = await fixture(), base = { ...settings(), authentication: { kind: "bearer" } as const, credential: "synthetic-map-bearer-fixture" };
    await f.service.save(base);
    await expect(f.service.save({ ...base, credential: undefined, expectedRevision: 1, source: { kind: "raster-xyz", template: "https://different.invalid/{z}/{x}/{y}.png", tileSize: 1, minimumZoom: 0, maximumZoom: 0 } })).rejects.toThrow(/replacing or clearing/);
    await f.service.remove("fixture", 1); const next = await f.service.save(settings()); expect(next.revision).toBe(3);
    await expect(render(f.service)).rejects.toThrow(/current revision/); expect(f.fetcher).not.toHaveBeenCalled();
  });
  it("rechecks the current action owner immediately after provider headers and does not return or cache a revoked result", async () => {
    let current = true; const f = await fixture({ fetch: (async () => { current = false; return new Response(pixelBody(), { headers: { "Content-Type": "image/png" } }); }) as typeof fetch }); await f.service.save(settings());
    await expect(render(f.service, request, undefined, async () => { if (!current) throw new Error("Exact action owner retired"); })).rejects.toThrow(/owner retired/);
    const db = new DatabaseSync(f.path); try { expect(db.prepare("SELECT COUNT(*) AS n FROM cache").get()?.n).toBe(0); } finally { db.close(); }
  });
  it("aborts only the forgotten instance and also cancels an in-flight operation when its profile is edited", async () => {
    const began: Array<() => void> = []; let releaseSecond!: () => void; let count = 0;
    const f = await fixture({ fetch: ((_url: unknown, init?: RequestInit) => new Promise<Response>((resolve, reject) => {
      count++; init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      if (count === 2) releaseSecond = () => resolve(new Response(pixelBody(), { headers: { "Content-Type": "image/png" } })); began.shift()?.();
    })) as typeof fetch }); await f.service.save(settings());
    const start = (): Promise<void> => new Promise(resolve => began.push(resolve)); const firstBegan = start();
    const a = f.service.execute("brain-a", "basemap", request, new AbortController().signal, allowed), rejectedA = expect(a).rejects.toThrow(/authority retired/); await firstBegan;
    const secondBegan = start(), b = f.service.execute("brain-b", "basemap", request, new AbortController().signal, allowed); await secondBegan;
    f.service.forget("brain-a"); await rejectedA; releaseSecond(); expect((await b as MapRasterResult).images).toHaveLength(1);
    const thirdBegan = start(), c = f.service.execute("brain-c", "basemap", { ...request, width: 2 }, new AbortController().signal, allowed);
    // A distinct request at the same tile is cached; remove only the synthetic cache to exercise a new in-flight fetch.
    const db = new DatabaseSync(f.path); db.exec("DELETE FROM cache"); db.close();
    const rejectedC = expect(c).rejects.toThrow(/profile changed/); await thirdBegan;
    await f.service.save({ ...settings(), expectedRevision: 1, enabled: false }); await rejectedC;
  });
  it("honors Stop during Retry-After pacing without retry, redirect following or credential leakage", async () => {
    const fetcher = vi.fn(async (_url: unknown, _init?: RequestInit) => new Response(null, { status: 429, headers: { "Retry-After": "120" } }));
    const f = await fixture({ fetch: fetcher as typeof fetch }); await f.service.save(settings());
    await expect(render(f.service)).rejects.toMatchObject({ output: { status: 429, retryAfterMs: 120000, automaticRetry: false } });
    const controller = new AbortController(), pending = render(f.service, request, controller.signal), rejected = expect(pending).rejects.toThrow("Stop fixture"); controller.abort(new Error("Stop fixture")); await rejected; expect(fetcher).toHaveBeenCalledOnce();
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ redirect: "manual", credentials: "omit" });
  });
  it("admits visible decoded pixels before any fetch and refuses corrupt caches or non-raster provider content", async () => {
    const refused = await fixture({ admit: async (_bytes, purpose) => { if (purpose === "visible raster pixels and tile plan") throw new Error("Current app RAM cap"); } }); await refused.service.save(settings());
    await expect(render(refused.service)).rejects.toThrow(/RAM cap/); expect(refused.fetcher).not.toHaveBeenCalled();
    const f = await fixture(); await f.service.save(settings()); await render(f.service); const db = new DatabaseSync(f.path); db.exec("UPDATE cache SET sha='corrupt'"); db.close();
    await expect(render(f.service)).rejects.toThrow(/checksum/); expect(f.fetcher).toHaveBeenCalledOnce();
    const html = await fixture({ fetch: (async () => new Response("<script>bad()</script>", { headers: { "Content-Type": "image/svg+xml" } })) as typeof fetch }); await html.service.save(settings()); await expect(render(html.service)).rejects.toThrow(/SVG/);
    expect(mapRasterImageInfo(pixel)).toEqual({ mime: "image/png", width: 1, height: 1 }); expect(() => mapRasterImageInfo(Buffer.from("<svg/>"))).toThrow(/supported/);
  });
});
