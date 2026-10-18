import { createHash, randomUUID } from "node:crypto";
import { lstatSync } from "node:fs";
import { chmod, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import type { SecureSecretStore } from "./secureSecretStore";
import { checkedMapBounds, mercatorLatitude, mercatorY, type MapBasemapProfile, type MapBasemapSave, type MapBounds,
  type MapRasterAuthentication, type MapRasterImage, type MapRasterRequest, type MapRasterResult, type MapRasterRights, type MapRasterSource } from "../shared/mapBasemaps";
import { mapRasterImageInfo } from "./mapRasterImage";
import packageMetadata from "../../package.json";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const hash = (value: string | Uint8Array): string => createHash("sha256").update(value).digest("hex");
const record = (value: unknown): Record<string, unknown> => { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Map basemap options must be a typed object."); return value as Record<string, unknown>; };
const clean = (value: unknown, label: string): string => { if (typeof value !== "string" || !value.trim() || /[\r\n\0]/.test(value)) throw new Error(`Invalid map ${label}.`); return value.trim(); };
function integer(value: unknown, label: string, minimum = 0, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum || Number(value) > maximum) throw new Error(`Invalid map ${label}.`); return Number(value);
}
function exactFields(value: Record<string, unknown>, fields: string[]): void {
  if (Object.keys(value).some(field => !fields.includes(field))) throw new Error("Map basemap contains unsupported or credential-bearing operands.");
}
function httpsLink(value: unknown): string {
  const url = new URL(clean(value, "attribution/license link"));
  if (url.protocol !== "https:" || url.username || url.password) throw new Error("Map attribution and license links must be credential-free HTTPS URLs."); return url.href;
}
function rights(value: unknown): MapRasterRights {
  const raw = record(value); exactFields(raw, ["retentionLearningAndRedistributionAuthorized", "authorizationReference", "licenseUrl"]);
  if (raw.retentionLearningAndRedistributionAuthorized !== true) throw new Error("Review authorization for retention, learning and complete redistribution before enabling a basemap. A declaration is not a license grant by Omni.");
  return { retentionLearningAndRedistributionAuthorized: true, authorizationReference: clean(raw.authorizationReference, "authorization reference"), licenseUrl: httpsLink(raw.licenseUrl) };
}
function source(value: unknown): MapRasterSource {
  const raw = record(value);
  if (raw.kind === "mapbox-static-image") {
    exactFields(raw, ["kind", "username", "styleId"]);
    const username = clean(raw.username, "Mapbox username"), styleId = clean(raw.styleId, "Mapbox style id");
    if (![username, styleId].every(part => /^[A-Za-z0-9_-]+$/.test(part))) throw new Error("Use an exact Mapbox style owner/id, not a URL or credential.");
    return { kind: raw.kind, username, styleId };
  }
  if (raw.kind !== "raster-xyz") throw new Error("Choose a reviewed XYZ source or an explicitly licensed Mapbox Static Images source.");
  exactFields(raw, ["kind", "template", "tileSize", "minimumZoom", "maximumZoom"]);
  const template = clean(raw.template, "XYZ template");
  if (["z", "x", "y"].some(axis => template.split(`{${axis}}`).length !== 2) || template.replace(/\{[zxy]\}/g, "").includes("{")) throw new Error("XYZ templates require exactly one each of {z}, {x}, {y} and no other substitutions.");
  if (/^https?:\/\/[^/]*\{/.test(template)) throw new Error("XYZ placeholders belong only in the path; the reviewed request origin must remain exact.");
  const url = new URL(template.replace(/\{[zxy]\}/g, "0"));
  if (url.username || url.password || url.search || url.hash || !["https:", "http:"].includes(url.protocol) ||
    url.protocol === "http:" && !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) throw new Error("Use an exact HTTPS or explicit loopback HTTP XYZ endpoint without embedded credentials, query or fragment.");
  const hostname = url.hostname.toLowerCase().replace(/\.$/, "");
  if (hostname === "tile.openstreetmap.org" || hostname.endsWith(".tile.openstreetmap.org") || hostname === "api.mapbox.com") throw new Error("Public OSM tiles forbid offline/bulk archives; Mapbox requires its separate contract-gated adapter. Use a source licensed for retention, learning and complete export.");
  const minimumZoom = integer(raw.minimumZoom, "minimum zoom", 0, 52), maximumZoom = integer(raw.maximumZoom, "maximum zoom", minimumZoom, 52);
  return { kind: raw.kind, template, tileSize: integer(raw.tileSize, "tile dimensions", 1), minimumZoom, maximumZoom };
}
function authentication(value: unknown, provider: MapRasterSource): MapRasterAuthentication {
  const raw = record(value);
  if (provider.kind === "mapbox-static-image") {
    exactFields(raw, ["kind", "parameter"]);
    if (raw.kind !== "query" || raw.parameter !== "access_token") throw new Error("Mapbox uses its main-only access_token credential, never a renderer token."); return { kind: "query", parameter: "access_token" };
  }
  if (raw.kind === "none" || raw.kind === "bearer") { exactFields(raw, ["kind"]); return { kind: raw.kind }; }
  if (raw.kind === "query" && typeof raw.parameter === "string" && /^[A-Za-z][A-Za-z0-9_.-]*$/.test(raw.parameter)) { exactFields(raw, ["kind", "parameter"]); return { kind: raw.kind, parameter: raw.parameter }; }
  throw new Error("Choose no credential, a main-only bearer, or a named main-only query credential.");
}
interface StoredProfile extends Omit<MapBasemapProfile, "credentialConfigured" | "credentialPersistence"> { credentialId?: string }
interface CachedImage { bytes: Uint8Array; observedAt: string }
interface RasterPart { bounds: MapBounds; url: string; expectedWidth: number; expectedHeight: number; tile?: { z: number; x: number; y: number } }
export interface MapBasemapServiceOptions {
  fetch?: typeof fetch;
  now?: () => number;
  admit?(bytes: number, purpose: string): Promise<void>;
  checkDisk?(bytes: number): Promise<void>;
}
export class MapBasemapError extends Error {
  constructor(message: string, readonly output: { format: "omni-map-raster-error"; version: 1; profileId: string; status?: number; retryAfterMs?: number; automaticRetry: false }) { super(message); this.name = "MapBasemapError"; }
}

/** Optional device-owned map sources. There is no active/public default. */
export class MapBasemapService {
  private mutations: Promise<unknown> = Promise.resolve();
  private readonly pending = new Map<string, Set<AbortController>>();
  private readonly owners = new Map<string, Set<AbortController>>();
  private readonly pacing = new Map<string, Promise<void>>();
  constructor(private readonly databasePath: string,
    private readonly secrets: Pick<SecureSecretStore, "get" | "set" | "remove" | "has" | "persistence">,
    private readonly options: MapBasemapServiceOptions = {}) {}
  async initialize(): Promise<void> {
    await mkdir(dirname(this.databasePath), { recursive: true, mode: 0o700 }); const db = this.database();
    try { await chmod(this.databasePath, 0o600); db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE IF NOT EXISTS revisions(id TEXT PRIMARY KEY,revision INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,bytes BLOB NOT NULL,sha TEXT NOT NULL,observed_at TEXT NOT NULL,expires_ms INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS pacing(id TEXT PRIMARY KEY,next_ms INTEGER NOT NULL);"); } finally { db.close(); }
  }
  private database(): DatabaseSync {
    try { const info = lstatSync(this.databasePath); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Map basemap storage must be an owned regular file, not a link."); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    const db = new DatabaseSync(this.databasePath); db.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=1000; PRAGMA cache_size=-512"); return db;
  }
  private now(): number { return this.options.now?.() ?? Date.now(); }
  private async admit(bytes: number, purpose: string): Promise<void> {
    if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Map raster allocation exceeds exact resource planning.");
    if (this.options.admit) return this.options.admit(bytes, purpose);
    if (bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error(`Map ${purpose} awaits available physical RAM.`);
  }
  private due(delay: number): number { return integer(this.now() + delay, "exact request deadline"); }
  private async profile(id: string): Promise<StoredProfile> {
    if (!ID.test(id)) throw new Error("Invalid map basemap profile id.");
    const db = this.database();
    try {
      const bytes = db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM profiles WHERE id=?").get(id)?.bytes;
      if (typeof bytes !== "number") throw new Error("Choose an explicitly reviewed map basemap profile first.");
      await this.admit(bytes * 12 + 8192, "profile metadata");
      const raw = db.prepare("SELECT body FROM profiles WHERE id=? AND length(CAST(body AS BLOB))=?").get(id, bytes)?.body;
      if (typeof raw !== "string") throw new Error("The map profile changed while its metadata was admitted."); return JSON.parse(raw) as StoredProfile;
    } finally { db.close(); }
  }
  private async summary(value: StoredProfile): Promise<MapBasemapProfile> {
    const { credentialId, ...visible } = value;
    return { ...visible, credentialConfigured: credentialId ? await this.secrets.has(credentialId) : false, credentialPersistence: this.secrets.persistence() };
  }
  async list(): Promise<MapBasemapProfile[]> {
    const db = this.database(); const ids: string[] = [];
    try { for (const row of db.prepare("SELECT id FROM profiles ORDER BY id").iterate()) { await this.admit((ids.length + 1) * 1024, "profile list"); ids.push(String(row.id)); } } finally { db.close(); }
    const result: MapBasemapProfile[] = [];
    for (const id of ids) { result.push(await this.summary(await this.profile(id))); await this.admit(Buffer.byteLength(JSON.stringify(result)) * 12 + 8192, "profile list"); } return result;
  }
  async save(value: MapBasemapSave): Promise<MapBasemapProfile> {
    const operation = this.mutations.then(async () => {
      const row = record(value); exactFields(row, ["id", "expectedRevision", "label", "enabled", "source", "authentication", "credential", "attribution", "attributionUrl", "rights", "minimumIntervalMs", "cacheTtlMs", "timeoutMs"]);
      if (typeof row.id !== "string" || !ID.test(row.id) || typeof row.enabled !== "boolean") throw new Error("Review the map profile id and enabled state.");
      let prior: StoredProfile | undefined;
      try { prior = await this.profile(row.id); } catch (error) { if (!String(error).includes("explicitly reviewed map basemap profile")) throw error; }
      const expected = integer(row.expectedRevision, "expected revision"); if (expected !== (prior?.revision ?? 0)) throw new Error("Map profile changed; review its current revision.");
      const provider = source(row.source), auth = authentication(row.authentication, provider), db = this.database(); let last: number;
      try { last = Number(db.prepare("SELECT revision FROM revisions WHERE id=?").get(row.id)?.revision ?? 0); } finally { db.close(); }
      const next: StoredProfile = { id: row.id, revision: integer(Math.max(last, expected) + 1, "profile revision", 1), label: clean(row.label, "label"), enabled: row.enabled,
        source: provider, authentication: auth, attribution: clean(row.attribution, "attribution"), attributionUrl: httpsLink(row.attributionUrl), rights: rights(row.rights),
        minimumIntervalMs: integer(row.minimumIntervalMs, "minimum interval"), cacheTtlMs: integer(row.cacheTtlMs, "cache duration"), timeoutMs: integer(row.timeoutMs, "timeout", 1, 2147483647),
        ...(prior?.credentialId ? { credentialId: prior.credentialId } : {}) };
      this.due(next.minimumIntervalMs); this.due(next.cacheTtlMs);
      if (prior?.credentialId && (JSON.stringify(prior.source) !== JSON.stringify(provider) || JSON.stringify(prior.authentication) !== JSON.stringify(auth)) && row.credential === undefined) throw new Error("Changing a source/authentication requires explicitly replacing or clearing its credential; no key is forwarded to a new source.");
      if (row.credential === null || auth.kind === "none") delete next.credentialId;
      let created: string | undefined;
      if (row.credential !== undefined && row.credential !== null) {
        if (auth.kind === "none") throw new Error("A keyless profile cannot receive a credential."); created = `map-basemap:${row.id}:${randomUUID()}`;
        await this.secrets.set(created, clean(row.credential, "credential")); next.credentialId = created;
      }
      if (next.enabled && auth.kind !== "none" && !next.credentialId) throw new Error("Add the required credential before enabling this profile.");
      try {
        const body = JSON.stringify(next); await this.admit(Buffer.byteLength(body) * 12 + 8192, "profile update"); await this.options.checkDisk?.(Buffer.byteLength(body) * 4 + 8192);
        const saved = this.database();
        try { saved.exec("BEGIN IMMEDIATE"); saved.prepare("INSERT OR REPLACE INTO profiles VALUES(?,?)").run(next.id, body); saved.prepare("INSERT OR REPLACE INTO revisions VALUES(?,?)").run(next.id, next.revision); saved.exec("COMMIT"); }
        catch (error) { try { saved.exec("ROLLBACK"); } catch { /* original error */ } throw error; } finally { saved.close(); }
      } catch (error) { if (created) await this.secrets.remove(created).catch(() => undefined); throw error; }
      for (const controller of this.pending.get(next.id) ?? []) controller.abort(new Error("Map basemap profile changed."));
      if (prior?.credentialId && prior.credentialId !== next.credentialId) await this.secrets.remove(prior.credentialId);
      return this.summary(next);
    }); this.mutations = operation.catch(() => undefined); return operation;
  }
  async remove(id: string, expectedRevision: number): Promise<void> {
    const operation = this.mutations.then(async () => {
      const prior = await this.profile(id); if (prior.revision !== expectedRevision) throw new Error("Map profile changed; review its current revision.");
      const revision = integer(prior.revision + 1, "profile revision", 1), db = this.database();
      try { db.exec("BEGIN IMMEDIATE"); db.prepare("DELETE FROM profiles WHERE id=?").run(id); db.prepare("INSERT OR REPLACE INTO revisions VALUES(?,?)").run(id, revision); db.exec("COMMIT"); }
      catch (error) { try { db.exec("ROLLBACK"); } catch { /* original error */ } throw error; } finally { db.close(); }
      for (const controller of this.pending.get(id) ?? []) controller.abort(new Error("Map basemap profile removed."));
      if (prior.credentialId) await this.secrets.remove(prior.credentialId);
    }); this.mutations = operation.catch(() => undefined); return operation;
  }
  private async current(profile: StoredProfile, signal: AbortSignal, requireCurrentPermission: () => Promise<void>, credential?: string): Promise<void> {
    signal.throwIfAborted(); await requireCurrentPermission(); signal.throwIfAborted();
    const current = await this.profile(profile.id);
    if (!current.enabled || current.revision !== profile.revision || current.credentialId !== profile.credentialId) throw new Error("The reviewed map profile, credential or enabled state changed.");
    if (profile.credentialId && await this.secrets.get(profile.credentialId) !== credential) throw new Error("The reviewed map credential changed or is unavailable in this OS session.");
    signal.throwIfAborted(); await requireCurrentPermission(); signal.throwIfAborted();
  }
  private async pace(profile: StoredProfile, signal: AbortSignal): Promise<void> {
    const previous = this.pacing.get(profile.id) ?? Promise.resolve(); let release!: () => void;
    const turn = new Promise<void>(resolve => { release = resolve; }); this.pacing.set(profile.id, turn); let acquired = false;
    const interrupted = async <T>(work: Promise<T>): Promise<T> => {
      let abort!: () => void;
      try { signal.throwIfAborted(); return await Promise.race([work, new Promise<never>((_resolve, reject) => { abort = () => reject(signal.reason); signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort(); })]); }
      finally { signal.removeEventListener("abort", abort); }
    };
    try {
      await interrupted(previous); acquired = true;
      const db = this.database(); let due: number;
      try { due = Number(db.prepare("SELECT next_ms FROM pacing WHERE id=?").get(profile.id)?.next_ms ?? 0); } finally { db.close(); }
      while (due > this.now()) { let timer: ReturnType<typeof setTimeout> | undefined;
        try { await interrupted(new Promise<void>(resolve => { timer = setTimeout(resolve, Math.min(due - this.now(), 2147483647)); })); } finally { if (timer) clearTimeout(timer); }
      }
      signal.throwIfAborted(); const reserve = this.database(); try { reserve.prepare("INSERT OR REPLACE INTO pacing VALUES(?,?)").run(profile.id, this.due(profile.minimumIntervalMs)); } finally { reserve.close(); }
    } finally {
      const finish = (): void => { release(); if (this.pacing.get(profile.id) === turn) this.pacing.delete(profile.id); };
      if (acquired) finish(); else void previous.then(finish, finish);
    }
  }
  private plan(request: MapRasterRequest, profile: StoredProfile): { count: number; parts: Generator<RasterPart> } {
    const { bounds, width, height } = request;
    if (profile.source.kind === "mapbox-static-image") {
      if (width > 1280 || height > 1280 || request.zoom !== undefined) throw new Error("Mapbox Static Images supports dimensions through 1280 per side and an exact bounding-box request; choose smaller dimensions or a different reviewed source.");
      const endpoint = `https://api.mapbox.com/styles/v1/${profile.source.username}/${profile.source.styleId}/static/[${bounds.join(",")}]/${width}x${height}.png`;
      return { count: 1, parts: (function* () { yield { bounds, url: endpoint, expectedWidth: width, expectedHeight: height }; })() };
    }
    const provider = profile.source, north = Math.max(0, Math.min(1, mercatorY(bounds[3]))), south = Math.max(0, Math.min(1, mercatorY(bounds[1])));
    const fitted = Math.floor(Math.log2(Math.min(width / ((bounds[2] - bounds[0]) / 360 * provider.tileSize), height / ((south - north) * provider.tileSize))));
    const z = integer(request.zoom ?? Math.max(provider.minimumZoom, Math.min(provider.maximumZoom, fitted)), "zoom", provider.minimumZoom, provider.maximumZoom), scale = 2 ** z;
    const minX = Math.max(0, Math.floor((bounds[0] + 180) / 360 * scale)), maxX = Math.min(scale - 1, Math.ceil((bounds[2] + 180) / 360 * scale) - 1);
    const minY = Math.max(0, Math.floor(north * scale)), maxY = Math.min(scale - 1, Math.ceil(south * scale) - 1), count = (maxX - minX + 1) * (maxY - minY + 1);
    integer(count, "visible tile count", 1);
    return { count, parts: (function* () { for (let y = minY; y <= maxY; y++) for (let x = minX; x <= maxX; x++) yield {
      bounds: [x / scale * 360 - 180, mercatorLatitude((y + 1) / scale), (x + 1) / scale * 360 - 180, mercatorLatitude(y / scale)] as MapBounds,
      url: provider.template.replace("{z}", String(z)).replace("{x}", String(x)).replace("{y}", String(y)), expectedWidth: provider.tileSize, expectedHeight: provider.tileSize, tile: { z, x, y }
    }; })() };
  }
  private async cached(key: string): Promise<CachedImage | undefined> {
    const db = this.database();
    try {
      const bytes = db.prepare("SELECT length(bytes) AS bytes FROM cache WHERE key=? AND expires_ms>?").get(key, this.now())?.bytes;
      if (typeof bytes !== "number") return undefined;
      await this.admit(bytes * 4 + 65536, "cached raster");
      const row = db.prepare("SELECT bytes,sha,observed_at FROM cache WHERE key=? AND expires_ms>? AND length(bytes)=?").get(key, this.now(), bytes);
      if (!row || !(row.bytes instanceof Uint8Array) || hash(row.bytes) !== row.sha) throw new Error("Retained map cache failed its exact checksum; no provider request was silently replayed.");
      return { bytes: row.bytes, observedAt: String(row.observed_at) };
    } finally { db.close(); }
  }
  /** Called only after host ToolExecutor approval; its exact owner/policy fence remains live through every await. */
  async execute(brainId: string, action: string, args: Record<string, unknown>, signal: AbortSignal, requireCurrentPermission: () => Promise<void>): Promise<unknown> {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(brainId) || typeof requireCurrentPermission !== "function") throw new Error("Current instance/tool authority is required for optional map access.");
    signal.throwIfAborted(); await requireCurrentPermission(); signal.throwIfAborted();
    if (action === "basemaps") { exactFields(args, []); const profiles = await this.list(); await requireCurrentPermission(); signal.throwIfAborted(); return { profiles, defaultProvider: null, publicServiceSelected: false }; }
    if (action !== "basemap") throw new Error("Unsupported optional map operation.");
    exactFields(args, ["profileId", "expectedRevision", "bounds", "width", "height", "zoom"]);
    const request: MapRasterRequest = { profileId: clean(args.profileId, "profile id"), expectedRevision: integer(args.expectedRevision, "expected revision"),
      bounds: checkedMapBounds(args.bounds), width: integer(args.width, "viewport width", 1), height: integer(args.height, "viewport height", 1), ...(args.zoom !== undefined ? { zoom: integer(args.zoom, "zoom") } : {}) };
    const profile = await this.profile(request.profileId);
    if (!profile.enabled || profile.revision !== request.expectedRevision) throw new Error("Review the enabled map profile and its exact current revision first.");
    const controller = new AbortController(), stop = (): void => controller.abort(signal.reason); signal.addEventListener("abort", stop, { once: true }); if (signal.aborted) stop();
    const pending = this.pending.get(profile.id) ?? new Set<AbortController>(); pending.add(controller); this.pending.set(profile.id, pending);
    const owners = this.owners.get(brainId) ?? new Set<AbortController>(); owners.add(controller); this.owners.set(brainId, owners);
    try {
      const credential = profile.credentialId ? await this.secrets.get(profile.credentialId) : undefined;
      if (profile.authentication.kind !== "none" && !credential) throw new Error("The reviewed map credential is unavailable in this OS account/session.");
      await this.current(profile, controller.signal, requireCurrentPermission, credential);
      const plan = this.plan(request, profile), pixels = profile.source.kind === "raster-xyz" ? plan.count * profile.source.tileSize ** 2 : request.width * request.height;
      await this.admit(pixels * 4 + plan.count * 2048, "visible raster pixels and tile plan");
      const images: MapRasterImage[] = []; let retainedBytes = 0;
      for (const part of plan.parts) {
        await this.current(profile, controller.signal, requireCurrentPermission, credential);
        const key = hash(JSON.stringify([profile.id, profile.revision, part.url])), cached = await this.cached(key);
        let bytes: Uint8Array, observedAt: string, cacheTtl = profile.cacheTtlMs;
        if (cached) { bytes = cached.bytes; observedAt = cached.observedAt; }
        else {
          await this.pace(profile, controller.signal); await this.current(profile, controller.signal, requireCurrentPermission, credential);
          const url = new URL(part.url);
          if (profile.authentication.kind === "query" && credential) url.searchParams.set(profile.authentication.parameter, credential);
          if (profile.source.kind === "mapbox-static-image") { url.searchParams.set("attribution", "true"); url.searchParams.set("logo", "true"); url.searchParams.set("padding", "0"); }
          const timeout = setTimeout(() => controller.abort(new Error("Map raster request timed out.")), profile.timeoutMs);
          try {
            let response: Response;
            try { response = await (this.options.fetch ?? fetch)(url, { redirect: "manual", credentials: "omit", signal: controller.signal, headers: {
              Accept: "image/png,image/jpeg,image/webp", "User-Agent": `Omni-AGI-Studio/${packageMetadata.version} (+https://github.com/Eabusham2/Omni-AGI)`,
              ...(profile.authentication.kind === "bearer" && credential ? { Authorization: `Bearer ${credential}` } : {}) } }); }
            catch { controller.signal.throwIfAborted(); throw new MapBasemapError("Map raster service could not be reached; no automatic retry occurred.", { format: "omni-map-raster-error", version: 1, profileId: profile.id, automaticRetry: false }); }
            try { await this.current(profile, controller.signal, requireCurrentPermission, credential); }
            catch (error) { await response.body?.cancel().catch(() => undefined); throw error; }
            if (!response.ok) {
              const retry = response.headers.get("Retry-After"), delay = retry && /^\d+$/.test(retry) ? Number(retry) * 1000 : retry ? Math.max(0, Date.parse(retry) - this.now()) : undefined;
              if (delay !== undefined && Number.isSafeInteger(delay)) { const paced = this.database(); try { paced.prepare("INSERT INTO pacing VALUES(?,?) ON CONFLICT(id) DO UPDATE SET next_ms=MAX(next_ms,excluded.next_ms)").run(profile.id, this.due(delay)); } finally { paced.close(); } }
              await response.body?.cancel(); throw new MapBasemapError(response.status >= 300 && response.status < 400 ? "Map source redirected; review it explicitly. No credential was forwarded." : `Map source returned HTTP ${response.status}; no automatic retry occurred.`,
                { format: "omni-map-raster-error", version: 1, profileId: profile.id, status: response.status, ...(delay !== undefined && Number.isSafeInteger(delay) ? { retryAfterMs: delay } : {}), automaticRetry: false });
            }
            if (!response.body) throw new Error("Map service returned no raster body.");
            const contentType = (response.headers.get("content-type") ?? "").split(";", 1)[0]!.trim().toLowerCase();
            if (!["image/png", "image/jpeg", "image/webp"].includes(contentType)) { await response.body.cancel(); throw new Error("Map service did not declare a supported raster image; HTML and SVG are not accepted."); }
            const reader = response.body.getReader(), pieces: Uint8Array[] = []; let length = 0;
            try { for (;;) { controller.signal.throwIfAborted(); const next = await reader.read(); await this.current(profile, controller.signal, requireCurrentPermission, credential); if (next.done) break; length += next.value.byteLength;
              await this.admit(retainedBytes * 4 + length * 5 + pixels * 4 + 65536, "streaming raster bytes"); pieces.push(next.value); } }
            finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
            bytes = Buffer.concat(pieces); const info = mapRasterImageInfo(bytes); if (info.mime !== contentType) throw new Error("Map raster signature does not match its declared content type.");
            if (credential && Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).includes(Buffer.from(credential))) throw new Error("Map source reflected its credential; this image was not retained or returned.");
            observedAt = new Date(this.now()).toISOString();
            const control = response.headers.get("cache-control") ?? "", maxAge = /(?:^|,)\s*max-age=(\d+)/i.exec(control);
            if (/no-store|no-cache/i.test(control)) cacheTtl = 0; else if (maxAge) cacheTtl = Math.min(cacheTtl, Number(maxAge[1]) * 1000);
          } finally { clearTimeout(timeout); }
        }
        await this.current(profile, controller.signal, requireCurrentPermission, credential);
        const info = mapRasterImageInfo(bytes);
        if (info.width !== part.expectedWidth || info.height !== part.expectedHeight) throw new Error("Map raster dimensions do not match the reviewed tile/viewport request.");
        await this.admit(retainedBytes * 4 + bytes.byteLength * 5 + pixels * 4 + 65536, "retained raster and decoded pixels");
        retainedBytes += bytes.byteLength;
        const sha256 = hash(bytes);
        images.push({ ...info, bytes: bytes.byteLength, sha256, dataUrl: `data:${info.mime};base64,${Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("base64")}`,
          bounds: part.bounds, extent: part.tile ? "exact-xyz-tile" : "provider-fit-to-request", ...(part.tile ? { tile: part.tile } : {}), cached: !!cached, sourceObservedAt: observedAt });
        if (!cached && cacheTtl > 0) {
          try { await this.options.checkDisk?.(bytes.byteLength * 2 + 8192); await this.current(profile, controller.signal, requireCurrentPermission, credential);
            const db = this.database(); try { db.prepare("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?)").run(key, bytes, sha256, observedAt, this.due(cacheTtl)); } finally { db.close(); }
          } catch (error) { controller.signal.throwIfAborted(); await this.current(profile, controller.signal, requireCurrentPermission, credential); /* Cache failure must not fabricate or discard the actual received image. */ }
        }
      }
      await this.current(profile, controller.signal, requireCurrentPermission, credential);
      return { format: "omni-map-raster", version: 1, provider: profile.source.kind, profileId: profile.id, profileRevision: profile.revision, label: profile.label,
        bounds: request.bounds, width: request.width, height: request.height, coordinateReferenceSystem: "EPSG:3857", images, attribution: profile.attribution, attributionUrl: profile.attributionUrl, rights: profile.rights,
        attributionLinks: [...(profile.source.kind === "mapbox-static-image" ? [{ label: "© Mapbox", url: "https://www.mapbox.com/about/maps/" },
          { label: "© OpenStreetMap contributors", url: "https://www.openstreetmap.org/copyright" }] : []), { label: profile.attribution, url: profile.attributionUrl }],
        provenance: { source: profile.source.kind === "raster-xyz" ? profile.source.template : `https://api.mapbox.com/styles/v1/${profile.source.username}/${profile.source.styleId}/static`,
          observedAt: new Date(this.now()).toISOString(), requestSha256: hash(JSON.stringify(request)), brandingRequested: profile.source.kind === "mapbox-static-image", automaticRetry: false }, basemapDownloaded: true, untrustedTaskData: true } satisfies MapRasterResult;
    } finally { signal.removeEventListener("abort", stop); pending.delete(controller); if (!pending.size) this.pending.delete(profile.id); owners.delete(controller); if (!owners.size) this.owners.delete(brainId); }
  }
  forget(brainId: string): void { for (const controller of this.owners.get(brainId) ?? []) controller.abort(new Error("Map instance authority retired.")); this.owners.delete(brainId); }
  close(): void { for (const requests of this.pending.values()) for (const controller of requests) controller.abort(new Error("Map adapter closed.")); this.pending.clear(); this.owners.clear(); }
}
