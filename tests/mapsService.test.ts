import { createServer, type Server } from "node:http";
import { mkdir, mkdtemp, rm, stat } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MapsService, MapAdapterError } from "../src/main/mapsService";
import { SecureSecretStore } from "../src/main/secureSecretStore";
import type { MapConnectionSave, MapDisplayResult, MapSearchResult } from "../src/shared/maps";

describe("private Photon adapter (local HTTP/synthetic provider fixtures, no accounts/models)", () => {
  let root: string; const servers: Server[] = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-maps-")); });
  afterEach(async () => { for (const server of servers.splice(0)) { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())); }
    await rm(root, { recursive: true, force: true }); });
  const body = { features: [{ type: "Feature", properties: { name: "<script>Untrusted place</script>", userId: "opaque-source" }, geometry: { type: "Point", coordinates: [13.3888599, 52.5170365] } }] };
  const settings = (endpoint = "http://127.0.0.1:2322/api/"): MapConnectionSave => ({ id: "private", expectedRevision: 0, label: "Private fixture", endpoint,
    attribution: "© OpenStreetMap contributors", attributionUrl: "https://www.openstreetmap.org/copyright", enabled: true, retainedResultsAccepted: true,
    minimumIntervalMs: 0, cacheTtlMs: 60000, timeoutMs: 5000 });
  async function fixture(fetcher?: typeof fetch, checkDisk?: (bytes: number) => Promise<void>) {
    const path = join(root, "map-connections.sqlite3"), secretStore = new SecureSecretStore(join(root, "secrets.json"), {
      available: () => false, encrypt: value => Buffer.from(value), decrypt: value => value.toString() });
    const service = new MapsService(path, secretStore, { fetch: fetcher, checkDisk }); await service.initialize(); return { service, secretStore, path };
  }
  const search = (service: MapsService, revision = 1, query = "Berlin", signal = new AbortController().signal, extra: object = {}) =>
    service.execute("brain-one", "maps.search", "search", { connectionId: "private", expectedRevision: revision, query, ...extra }, signal) as Promise<MapSearchResult>;
  it("does not select a public service, install/connect/test anything or accept unreviewed retention", async () => {
    const fetcher = vi.fn(), f = await fixture(fetcher as typeof fetch); expect(await f.service.list()).toEqual([]);
    await expect(f.service.save({ ...settings(), retainedResultsAccepted: false } as unknown as MapConnectionSave)).rejects.toThrow(/retention/);
    for (const endpoint of ["https://nominatim.openstreetmap.org/search", "https://photon.komoot.io/api/", "https://api.mapbox.com/search/geocode/v6/forward", "https://user:secret@private.invalid/api/"]) {
      await expect(f.service.save(settings(endpoint))).rejects.toThrow();
    }
    await f.service.save(settings()); expect(fetcher).not.toHaveBeenCalled();
  });
  it("keeps credentials main-only, rejects retargeting/stale revisions and permits explicit replacement or clear", async () => {
    const f = await fixture(), secret = "synthetic-private-fixture-bearer";
    const saved = await f.service.save({ ...settings(), bearerToken: secret }); expect(saved).toMatchObject({ revision: 1, credentialConfigured: true, credentialPersistence: "session" });
    const db = new DatabaseSync(f.path, { readOnly: true }); try { expect(JSON.stringify(db.prepare("SELECT * FROM connections").all())).not.toContain(secret); } finally { db.close(); }
    expect(JSON.stringify(await f.service.list())).not.toContain(secret);
    await expect(f.service.save({ ...settings("https://other-private.invalid/api/"), expectedRevision: 1 })).rejects.toThrow(/replacing or clearing/);
    await expect(search(f.service, 0)).rejects.toThrow(/current revision/);
    const next = await f.service.save({ ...settings("https://other-private.invalid/api/"), expectedRevision: 1, bearerToken: null });
    expect(next).toMatchObject({ revision: 2, credentialConfigured: false });
    await expect(f.service.remove("private", 1)).rejects.toThrow(/current revision/); await f.service.remove("private", 2); expect(await f.service.list()).toEqual([]);
    const replacement = await f.service.save(settings("https://third-private.invalid/api/")); expect(replacement.revision).toBeGreaterThan(2);
    await expect(search(f.service, 1)).rejects.toThrow(/current revision/);
  });
  it("uses the actual local HTTP Photon schema, caches exact source bytes, and displays only owner-bound coordinates with attribution", async () => {
    const requests: Array<{ url: string; authorization: string | undefined; agent: string | undefined }> = [];
    const server = createServer((request, response) => { requests.push({ url: request.url!, authorization: request.headers.authorization, agent: request.headers["user-agent"] });
      response.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "max-age=60" }); response.end(JSON.stringify(body)); }); servers.push(server);
    await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
    const address = server.address(); if (!address || typeof address === "string") throw new Error("Local fixture address missing");
    const f = await fixture(); await f.service.save({ ...settings(`http://127.0.0.1:${address.port}/api/`), bearerToken: "synthetic-private-fixture-bearer" });
    const first = await search(f.service, 1, "Berlin + café");
    expect(first.geoJson.features).toEqual(body.features); expect(first.provenance).toMatchObject({ cached: false, sourceEnvelope: "photon-features" });
    expect(first.bounds).toEqual([13.3888599, 52.5170365, 13.3888599, 52.5170365]); expect(first.searchResultId).toBeTruthy();
    const cached = await search(f.service, 1, "Berlin + café"); expect(cached.provenance.cached).toBe(true); expect(requests).toHaveLength(1);
    expect(new URL(requests[0]!.url, "http://fixture").searchParams.get("q")).toBe("Berlin + café"); expect(requests[0]!.agent).toContain("Omni-AGI-Studio");
    expect(requests[0]!.authorization).toBe("Bearer synthetic-private-fixture-bearer"); expect(JSON.stringify(first)).not.toContain("synthetic-private-fixture-bearer");
    const display = await f.service.execute("brain-one", "maps.display", "show", { searchResultId: first.searchResultId }, new AbortController().signal) as MapDisplayResult;
    expect(display).toMatchObject({ displayMode: "offline-coordinate-canvas", basemapDownloaded: false, attribution: "© OpenStreetMap contributors", dataOrigin: "search-result" });
    expect(display.geoJson.features).toEqual(body.features);
    await expect(f.service.execute("other-brain", "maps.display", "show", { searchResultId: first.searchResultId }, new AbortController().signal)).rejects.toThrow(/not available to this instance/);
  });
  it("never follows bearer redirects or retries provider errors and honors a cancelled backoff wait", async () => {
    const fetcher = vi.fn(async (_url: unknown, _init?: RequestInit) => new Response(null, { status: 302, headers: { Location: "https://other.invalid/steal" } })), f = await fixture(fetcher as typeof fetch);
    await f.service.save({ ...settings(), bearerToken: "synthetic-private-fixture-bearer" }); await expect(search(f.service)).rejects.toThrow(/not forwarded/);
    expect(fetcher).toHaveBeenCalledOnce(); expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ redirect: "manual", credentials: "omit" });
    fetcher.mockResolvedValueOnce(new Response("not retained error body", { status: 429, headers: { "Retry-After": "120" } }));
    await expect(search(f.service)).rejects.toMatchObject({ output: { status: 429, retryAfterMs: 120000, automaticRetry: false } });
    const controller = new AbortController(), waiting = search(f.service, 1, "Different", controller.signal); const rejected = expect(waiting).rejects.toThrow("fixture stop");
    controller.abort(new Error("fixture stop")); await rejected; expect(fetcher).toHaveBeenCalledTimes(2);
  });
  it("cancels an in-flight read on settings revision and does not expose a reflected credential", async () => {
    let started!: () => void; const readStarted = new Promise<void>(resolve => { started = resolve; });
    const fetcher = vi.fn((_url: unknown, init?: RequestInit) => new Promise<Response>((_resolve, reject) => { started(); const cancel = () => reject(init?.signal?.reason);
      init?.signal?.addEventListener("abort", cancel, { once: true }); })); const f = await fixture(fetcher as typeof fetch);
    await f.service.save(settings()); const pending = search(f.service), rejected = expect(pending).rejects.toThrow(/settings changed/); await readStarted;
    await f.service.save({ ...settings(), expectedRevision: 1, enabled: false }); await rejected;
    const secret = "synthetic-private-fixture-bearer"; fetcher.mockResolvedValueOnce(new Response(JSON.stringify({ ...body, leaked: secret })));
    await f.service.save({ ...settings(), expectedRevision: 2, bearerToken: secret }); await expect(search(f.service, 3)).rejects.toThrow(/reflected its private credential/);
  });
  it("returns actual source data if cache storage cannot commit, and refuses corrupted cache without an explicit refresh", async () => {
    let disk = true; const fetcher = vi.fn(async () => new Response(JSON.stringify(body))), f = await fixture(fetcher as typeof fetch, async () => { if (!disk) throw new Error("Synthetic reserve pressure"); });
    await f.service.save(settings()); disk = false; const actual = await search(f.service); expect(actual).toMatchObject({ cacheStored: false, geoJson: { features: body.features } }); expect(actual.searchResultId).toBeUndefined();
    disk = true; const retained = await search(f.service); expect(retained.cacheStored).toBe(true);
    const db = new DatabaseSync(f.path); db.exec("UPDATE cache SET body='corrupt'"); db.close();
    await expect(search(f.service)).rejects.toThrow(/explicitly reviewed refresh/); expect(fetcher).toHaveBeenCalledTimes(2);
    await search(f.service, 1, "Berlin", undefined, { refresh: true }); expect(fetcher).toHaveBeenCalledTimes(3);
  });
  it("keeps supplied geometry and unknown labels inert and rejects bad coordinate shapes rather than faking a basemap", async () => {
    const f = await fixture(), signal = new AbortController().signal;
    const result = await f.service.execute("brain-one", "maps.display", "show", { geoJson: { type: "FeatureCollection", ...body }, attribution: "Explicitly supplied geometry" }, signal) as MapDisplayResult;
    expect(result.geoJson.features).toEqual(body.features); expect(result.basemapDownloaded).toBe(false);
    await expect(f.service.execute("brain-one", "maps.display", "show", { geoJson: { type: "FeatureCollection", features: [{ type: "Feature", properties: null, geometry: { type: "Point", coordinates: [181, 20] } }] }, attribution: "source" }, signal)).rejects.toThrow(/WGS84/);
    await expect(search(f.service, 1, "Berlin", signal, { bearerToken: "spoofed" })).rejects.toThrow(/credential operands/);
    expect(MapAdapterError).toBeDefined();
  });
  it("keeps configuration/cache storage private and refuses non-regular database targets", async () => {
    const f = await fixture(); if (process.platform !== "win32") expect((await stat(f.path)).mode & 0o777).toBe(0o600);
    const path = join(root, "not-a-database.sqlite3"); await mkdir(path);
    await expect(new MapsService(path, f.secretStore).initialize()).rejects.toThrow(/owned regular database/);
  });
});
