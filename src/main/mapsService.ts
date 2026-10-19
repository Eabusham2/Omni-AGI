import { createHash, randomUUID } from "node:crypto";
import { chmod, mkdir } from "node:fs/promises";
import { lstatSync } from "node:fs";
import { dirname } from "node:path";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import { DatabaseSync } from "node:sqlite";
import type { MapConnection, MapConnectionSave, MapDisplayResult, MapSearchResult } from "../shared/maps";
import type { SecureSecretStore } from "./secureSecretStore";
import { checkedMapGeoJson } from "../shared/mapGeoJson";
import packageMetadata from "../../package.json";
import { sqliteFilename } from "./sqliteFilename";

const ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/, SHA = /^[a-f0-9]{64}$/;
const hash = (value: string | Uint8Array) => createHash("sha256").update(value).digest("hex");
const object = (value: unknown): Record<string, unknown> => { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Map options must be a typed object."); return value as Record<string, unknown>; };
function text(value: unknown, label: string): string { if (typeof value !== "string" || !value.trim() || /[\0\r\n]/.test(value)) throw new Error(`Invalid map ${label}.`); return value.trim(); }
function number(value: unknown, fallback: number, label: string, minimum = 0): number { const n = value === undefined ? fallback : value; if (!Number.isSafeInteger(n) || Number(n) < minimum) throw new Error(`Invalid map ${label}.`); return Number(n); }
function endpoint(value: unknown): string {
  const url = new URL(text(value, "endpoint"));
  if (url.username || url.password || url.search || url.hash || !["http:", "https:"].includes(url.protocol) ||
    url.protocol === "http:" && !["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)) throw new Error("Choose an HTTPS private Photon API endpoint or explicit loopback HTTP endpoint without embedded credentials/query parameters.");
  if (["photon.komoot.io", "nominatim.openstreetmap.org", "api.mapbox.com"].includes(url.hostname.toLowerCase().replace(/\.$/, ""))) throw new Error("This adapter requires a private Photon-compatible endpoint. Public demo/Nominatim policy and Mapbox retained-export licensing are not silently assumed.");
  return url.href;
}
function attributionUrl(value: unknown): string { const url = new URL(text(value, "attribution link")); if (url.protocol !== "https:" || url.username || url.password) throw new Error("Map attribution must link to an HTTPS source/licence page."); return url.href; }
function deadline(now: number, delay: number): number { const result = now + delay; if (!Number.isSafeInteger(result)) throw new Error("Map timing value exceeds exact host date storage."); return result; }
async function waitForTurn(turn: Promise<void>, signal: AbortSignal): Promise<void> {
  signal.throwIfAborted(); let stop!: () => void;
  try { await Promise.race([turn, new Promise<never>((_resolve, reject) => { stop = () => reject(signal.reason); signal.addEventListener("abort", stop, { once: true }); if (signal.aborted) stop(); })]); }
  finally { signal.removeEventListener("abort", stop); }
}
interface StoredConnection extends Omit<MapConnection, "credentialConfigured" | "credentialPersistence"> { credentialId?: string }
export interface MapsServiceOptions {
  fetch?: typeof fetch;
  now?: () => number;
  admit?(bytes: number): Promise<void>;
  checkDisk?(bytes: number): Promise<void>;
}
export class MapAdapterError extends Error {
  constructor(message: string, readonly output: { format: "omni-map-error"; version: 1; connectionId: string; endpoint: string; status?: number; retryAfterMs?: number; automaticRetry: false }) { super(message); this.name = "MapAdapterError"; }
}

/** Installation-private configuration/cache; only tool outputs enter a brain.
 * No public geocoder is selected and no credential is part of a tool argument. */
export class MapsService {
  private mutations: Promise<unknown> = Promise.resolve();
  private readonly requests = new Map<string, Set<AbortController>>();
  private readonly pacing = new Map<string, Promise<void>>();
  constructor(private readonly databasePath: string, private readonly secrets: Pick<SecureSecretStore, "get" | "set" | "remove" | "has" | "persistence">,
    private readonly options: MapsServiceOptions = {}) {}
  async initialize(): Promise<void> {
    await mkdir(dirname(this.databasePath), { recursive: true, mode: 0o700 }); const db = this.database();
    try { await chmod(this.databasePath, 0o600); db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; CREATE TABLE IF NOT EXISTS connections(id TEXT PRIMARY KEY,body TEXT NOT NULL); CREATE TABLE IF NOT EXISTS connection_revisions(id TEXT PRIMARY KEY,revision INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS cache(cache_key TEXT PRIMARY KEY,body TEXT NOT NULL,response_sha TEXT NOT NULL,observed_at TEXT NOT NULL,expires_ms INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,brain_id TEXT NOT NULL,body TEXT NOT NULL,body_sha TEXT NOT NULL); CREATE TABLE IF NOT EXISTS pacing(id TEXT PRIMARY KEY,next_ms INTEGER NOT NULL);"); } finally { db.close(); }
  }
  private database(): DatabaseSync {
    try { const info = lstatSync(this.databasePath); if (!info.isFile() || info.isSymbolicLink()) throw new Error("Map configuration must be an owned regular database, not a filesystem link."); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    const db = new DatabaseSync(sqliteFilename(this.databasePath)); db.exec("PRAGMA trusted_schema=OFF; PRAGMA busy_timeout=1000; PRAGMA cache_size=-512"); return db;
  }
  private now(): number { return this.options.now?.() ?? Date.now(); }
  private async admit(bytes: number): Promise<void> { if (this.options.admit) return this.options.admit(bytes); if (!Number.isSafeInteger(bytes) || bytes > freemem() || bytes > getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed) throw new Error("Map transfer awaits physical RAM."); }
  private connection(id: string): StoredConnection {
    if (!ID.test(id)) throw new Error("Invalid map connection id."); const db = this.database();
    try { const raw = db.prepare("SELECT body FROM connections WHERE id=?").get(id)?.body; if (typeof raw !== "string") throw new Error("Choose an explicitly configured private Photon connection first."); return JSON.parse(raw) as StoredConnection; } finally { db.close(); }
  }
  private async summary(value: StoredConnection): Promise<MapConnection> { const { credentialId, ...visible } = value; return { ...visible,
    credentialConfigured: credentialId ? await this.secrets.has(credentialId) : false, credentialPersistence: this.secrets.persistence() }; }
  async list(): Promise<MapConnection[]> { const db = this.database(); let values: StoredConnection[];
    try { values = db.prepare("SELECT body FROM connections ORDER BY id").all().map(row => JSON.parse(String(row.body)) as StoredConnection); } finally { db.close(); }
    return Promise.all(values.map(value => this.summary(value)));
  }
  async save(value: MapConnectionSave): Promise<MapConnection> {
    const operation = this.mutations.then(async () => {
      const row = object(value), allowed = ["id", "expectedRevision", "label", "endpoint", "attribution", "attributionUrl", "enabled", "retainedResultsAccepted", "bearerToken", "minimumIntervalMs", "cacheTtlMs", "timeoutMs"];
      if (Object.keys(row).some(key => !allowed.includes(key)) || typeof row.id !== "string" || !ID.test(row.id) || typeof row.enabled !== "boolean" || row.retainedResultsAccepted !== true) throw new Error("Review map configuration and accept ordinary result retention before saving.");
      let prior: StoredConnection | undefined; try { prior = this.connection(String(row.id)); } catch (error) { if (!String(error).includes("configured private Photon")) throw error; }
      const revision = number(row.expectedRevision, -1, "expected revision"); if (revision !== (prior?.revision ?? 0)) throw new Error("Map connection changed; review its current revision.");
      const revisions = this.database(); let lastRevision: number;
      try { lastRevision = Number(revisions.prepare("SELECT revision FROM connection_revisions WHERE id=?").get(row.id)?.revision ?? prior?.revision ?? 0); } finally { revisions.close(); }
      const next: StoredConnection = { id: String(row.id), label: text(row.label, "label"), provider: "private-photon", endpoint: endpoint(row.endpoint),
        attribution: text(row.attribution, "attribution"), attributionUrl: attributionUrl(row.attributionUrl), enabled: row.enabled,
        revision: Math.max(revision, lastRevision) + 1, minimumIntervalMs: number(row.minimumIntervalMs, prior?.minimumIntervalMs ?? 1000, "pacing"),
        cacheTtlMs: number(row.cacheTtlMs, prior?.cacheTtlMs ?? 86400000, "cache duration"), timeoutMs: number(row.timeoutMs, prior?.timeoutMs ?? 30000, "timeout", 1),
        retainedResultsAccepted: true, ...(prior?.credentialId ? { credentialId: prior.credentialId } : {}) };
      if (next.timeoutMs > 2147483647) throw new Error("Map timeout exceeds the host timer range.");
      if (!Number.isSafeInteger(next.revision)) throw new Error("Map connection revision exceeds exact storage.");
      deadline(this.now(), next.minimumIntervalMs); deadline(this.now(), next.cacheTtlMs);
      if (prior && next.endpoint !== prior.endpoint && row.bearerToken === undefined && prior.credentialId) throw new Error("Changing the map endpoint requires explicitly replacing or clearing its credential; no bearer is forwarded to a new origin.");
      let createdSecret: string | undefined;
      if (row.bearerToken === null) delete next.credentialId;
      else if (row.bearerToken !== undefined) { createdSecret = `maps:${next.id}:${randomUUID()}`; await this.secrets.set(createdSecret, text(row.bearerToken, "bearer credential")); next.credentialId = createdSecret; }
      try { await this.options.checkDisk?.(Buffer.byteLength(JSON.stringify(next)) * 4 + 8192); const db = this.database();
        try { db.exec("BEGIN IMMEDIATE"); db.prepare("INSERT OR REPLACE INTO connections VALUES(?,?)").run(next.id, JSON.stringify(next));
          db.prepare("INSERT OR REPLACE INTO connection_revisions VALUES(?,?)").run(next.id, next.revision); db.exec("COMMIT");
        } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original error */ } throw error; } finally { db.close(); }
      } catch (error) { if (createdSecret) await this.secrets.remove(createdSecret).catch(() => undefined); throw error; }
      for (const controller of this.requests.get(next.id) ?? []) controller.abort(new Error("Map connection settings changed."));
      if (prior?.credentialId && prior.credentialId !== next.credentialId) await this.secrets.remove(prior.credentialId);
      return this.summary(next);
    }); this.mutations = operation.catch(() => undefined); return operation;
  }
  async remove(id: string, expectedRevision: number): Promise<void> {
    const operation = this.mutations.then(async () => { const prior = this.connection(id);
      if (prior.revision !== expectedRevision) throw new Error("Map connection changed; review its current revision.");
      if (!Number.isSafeInteger(prior.revision + 1)) throw new Error("Map connection revision exceeds exact storage.");
      const db = this.database(); try { db.exec("BEGIN IMMEDIATE"); db.prepare("DELETE FROM connections WHERE id=?").run(id);
        db.prepare("INSERT OR REPLACE INTO connection_revisions VALUES(?,?)").run(id, prior.revision + 1); db.exec("COMMIT");
      } catch (error) { try { db.exec("ROLLBACK"); } catch { /* original error */ } throw error; } finally { db.close(); }
      for (const controller of this.requests.get(id) ?? []) controller.abort(new Error("Map connection was removed."));
      if (prior.credentialId) await this.secrets.remove(prior.credentialId);
    }); this.mutations = operation.catch(() => undefined); return operation;
  }
  private async pace(connection: StoredConnection, signal: AbortSignal): Promise<void> {
    const previous = this.pacing.get(connection.id) ?? Promise.resolve(); let release!: () => void;
    const turn = new Promise<void>(yes => { release = yes; }); this.pacing.set(connection.id, turn);
    let acquired = false;
    try { await waitForTurn(previous, signal); acquired = true; signal.throwIfAborted(); const db = this.database(); let delay: number, due: number;
      try { const row = db.prepare("SELECT next_ms FROM pacing WHERE id=?").get(connection.id); due = Number(row?.next_ms ?? 0); delay = Math.max(0, due - this.now()); } finally { db.close(); }
      while (delay) { await new Promise<void>((resolve, reject) => { const cancel = () => { clearTimeout(timer); signal.removeEventListener("abort", cancel); reject(signal.reason); };
        const timer = setTimeout(() => { signal.removeEventListener("abort", cancel); resolve(); }, Math.min(delay, 2147483647)); signal.addEventListener("abort", cancel, { once: true }); if (signal.aborted) cancel(); });
        delay = Math.max(0, due - this.now()); }
      signal.throwIfAborted(); const reserve = this.database(); try { reserve.prepare("INSERT OR REPLACE INTO pacing VALUES(?,?)").run(connection.id, deadline(this.now(), connection.minimumIntervalMs)); } finally { reserve.close(); }
    } finally { const finish = (): void => { release(); if (this.pacing.get(connection.id) === turn) this.pacing.delete(connection.id); };
      if (acquired) finish(); else void previous.then(finish, finish); }
  }
  private current(connection: StoredConnection): void { const actual = this.connection(connection.id); if (!actual.enabled || actual.revision !== connection.revision) throw new Error("The reviewed map connection changed or was disabled."); }
  async execute(brainId: string, toolId: string, action: string, args: Record<string, unknown>, signal: AbortSignal): Promise<unknown> {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(brainId)) throw new Error("Invalid map brain identity."); signal.throwIfAborted();
    if (toolId === "maps.search" && action === "connections") { if (Object.keys(args).length) throw new Error("Map connection listing accepts no operands."); return { connections: await this.list(), publicServiceSelected: false }; }
    if (toolId === "maps.search" && action === "search") return this.search(brainId, args, signal);
    if (toolId === "maps.display" && action === "show") return this.display(brainId, args, signal);
    throw new Error("Unsupported map action.");
  }
  private async search(brainId: string, args: Record<string, unknown>, signal: AbortSignal): Promise<MapSearchResult> {
    if (Object.keys(args).some(key => !["connectionId", "expectedRevision", "query", "limit", "language", "bbox", "refresh"].includes(key)) || args.refresh !== undefined && typeof args.refresh !== "boolean") throw new Error("Map search contains unsupported/credential operands.");
    const connection = this.connection(text(args.connectionId, "connection id"));
    if (!connection.enabled || args.expectedRevision !== connection.revision) throw new Error("Review the enabled map connection and its current revision before search.");
    const query = text(args.query, "query"), url = new URL(connection.endpoint); url.searchParams.set("q", query);
    if (args.limit !== undefined) url.searchParams.set("limit", String(number(args.limit, 0, "requested result count", 1)));
    if (args.language !== undefined) url.searchParams.set("lang", text(args.language, "language"));
    if (args.bbox !== undefined) { if (!Array.isArray(args.bbox) || args.bbox.length !== 4 || args.bbox.some(value => typeof value !== "number" || !Number.isFinite(value)) ||
      Number(args.bbox[0]) < -180 || Number(args.bbox[2]) > 180 || Number(args.bbox[1]) < -90 || Number(args.bbox[3]) > 90 || Number(args.bbox[0]) > Number(args.bbox[2]) || Number(args.bbox[1]) > Number(args.bbox[3])) throw new Error("Map bounds must be west,south,east,north WGS84 coordinates."); url.searchParams.set("bbox", args.bbox.join(",")); }
    const cacheKey = hash(JSON.stringify([connection.id, connection.revision, url.href])), db = this.database();
    let cached: { body: string; responseSha256: string; observedAt: string } | undefined;
    try { const info = args.refresh ? undefined : db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM cache WHERE cache_key=? AND expires_ms>?").get(cacheKey, this.now());
      if (info) { await this.admit(Number(info.bytes) * 16 + 65536); const row = db.prepare("SELECT * FROM cache WHERE cache_key=? AND expires_ms>?").get(cacheKey, this.now());
        if (!row || typeof row.body !== "string" || typeof row.response_sha !== "string" || !SHA.test(row.response_sha) || hash(row.body) !== row.response_sha) throw new Error("Cached map response failed integrity. An explicitly reviewed refresh is required; no request was silently replayed.");
        cached = { body: row.body, responseSha256: row.response_sha, observedAt: String(row.observed_at) }; }
    } finally { db.close(); }
    const controller = new AbortController(), stop = () => controller.abort(signal.reason); signal.addEventListener("abort", stop, { once: true }); if (signal.aborted) stop();
    const requests = this.requests.get(connection.id) ?? new Set<AbortController>(); requests.add(controller); this.requests.set(connection.id, requests);
    let timer: ReturnType<typeof setTimeout> | undefined, raw = cached?.body, responseSha256 = cached?.responseSha256, observedAt = cached?.observedAt, ttl = connection.cacheTtlMs;
    try {
      if (!cached) {
        const bearer = connection.credentialId ? await this.secrets.get(connection.credentialId) : undefined;
        if (connection.credentialId && !bearer) throw new Error("The map credential is unavailable in this OS account/session.");
        await this.pace(connection, controller.signal); this.current(connection); controller.signal.throwIfAborted();
        timer = setTimeout(() => controller.abort(new Error("Map service request timed out.")), connection.timeoutMs);
        let response: Response;
        try { response = await (this.options.fetch ?? fetch)(url, { method: "GET", redirect: "manual", credentials: "omit", signal: controller.signal,
          headers: { Accept: "application/json", "User-Agent": `Omni-AGI-Studio/${packageMetadata.version} (+https://github.com/Eabusham2/Omni-AGI)`, ...(bearer ? { Authorization: `Bearer ${bearer}` } : {}) } }); }
        catch (error) { if (controller.signal.aborted) throw controller.signal.reason; throw new MapAdapterError("Private map service could not be reached; no automatic request retry occurred.", { format: "omni-map-error", version: 1, connectionId: connection.id, endpoint: connection.endpoint, automaticRetry: false }); }
        if (!response.ok) {
          const retry = response.headers.get("Retry-After"), retryMs = retry && /^\d+$/.test(retry) ? Number(retry) * 1000 : retry ? Math.max(0, Date.parse(retry) - this.now()) : undefined;
          if (retryMs !== undefined && Number.isSafeInteger(retryMs)) { const wait = this.database(); try { wait.prepare("INSERT INTO pacing VALUES(?,?) ON CONFLICT(id) DO UPDATE SET next_ms=MAX(next_ms,excluded.next_ms)").run(connection.id, deadline(this.now(), retryMs)); } finally { wait.close(); } }
          await response.body?.cancel(); throw new MapAdapterError(response.status >= 300 && response.status < 400 ? "Map endpoint redirected; reconfigure explicitly. Credentials were not forwarded." : `Private map service returned HTTP ${response.status}. No automatic retry occurred.`,
            { format: "omni-map-error", version: 1, connectionId: connection.id, endpoint: connection.endpoint, status: response.status, ...(Number.isSafeInteger(retryMs) ? { retryAfterMs: retryMs } : {}), automaticRetry: false });
        }
        if (!response.body) throw new Error("Map service returned no response body.");
        const reader = response.body.getReader(), pieces: Uint8Array[] = []; let bytes = 0;
        try { for (;;) { controller.signal.throwIfAborted(); const piece = await reader.read(); if (piece.done) break; bytes += piece.value.byteLength; await this.admit(bytes * 16 + 65536); pieces.push(piece.value); } }
        finally { await reader.cancel().catch(() => undefined); reader.releaseLock(); }
        raw = new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(pieces));
        if (bearer && (raw.includes(bearer) || raw.includes(JSON.stringify(bearer).slice(1, -1)))) throw new Error("Map service reflected its private credential; the response was not retained or sent to the brain.");
        responseSha256 = hash(raw); observedAt = new Date(this.now()).toISOString();
        const control = response.headers.get("Cache-Control") ?? "", maxAge = /(?:^|,)\s*max-age=(\d+)/i.exec(control);
        if (/no-store|no-cache/i.test(control)) ttl = 0; else if (maxAge) ttl = Math.min(ttl, Number(maxAge[1]) * 1000);
      }
      controller.signal.throwIfAborted(); this.current(connection); await this.admit(Buffer.byteLength(raw!) * 16 + 65536);
      const original = object(JSON.parse(raw!)), sourceEnvelope = original.type === undefined ? "photon-features" as const : "geojson" as const;
      const parsed = checkedMapGeoJson(original.type === undefined ? { ...original, type: "FeatureCollection" } : original);
      const result: MapSearchResult = { format: "omni-map-search", version: 1, provider: "private-photon", connectionId: connection.id,
        connectionRevision: connection.revision, query, ...parsed, attribution: connection.attribution, attributionUrl: connection.attributionUrl,
        provenance: { endpoint: connection.endpoint, observedAt: new Date(this.now()).toISOString(), responseSha256: responseSha256!, cached: !!cached, sourceObservedAt: observedAt!, sourceEnvelope }, cacheStored: false, untrustedTaskData: true };
      try { await this.options.checkDisk?.(Buffer.byteLength(raw!) * 3 + 16384); controller.signal.throwIfAborted(); const id = randomUUID(), saved = this.database();
        try { saved.exec("BEGIN IMMEDIATE"); if (!cached && ttl > 0) saved.prepare("INSERT OR REPLACE INTO cache VALUES(?,?,?,?,?)").run(cacheKey, raw!, responseSha256!, observedAt!, deadline(this.now(), ttl));
          result.searchResultId = id; result.cacheStored = !!cached || ttl > 0; const body = JSON.stringify(result); saved.prepare("INSERT INTO results VALUES(?,?,?,?)").run(id, brainId, body, hash(body)); saved.exec("COMMIT");
        } catch (error) { try { saved.exec("ROLLBACK"); } catch { /* original failure */ } throw error; } finally { saved.close(); }
      } catch (error) { controller.signal.throwIfAborted(); delete result.searchResultId; result.cacheStored = false; result.warning = "The actual map response is available, but local lookup/cache storage could not be committed. Use its supplied GeoJSON to display it."; }
      return result;
    } finally { if (timer) clearTimeout(timer); signal.removeEventListener("abort", stop); requests.delete(controller); if (!requests.size) this.requests.delete(connection.id); }
  }
  private async display(brainId: string, args: Record<string, unknown>, signal: AbortSignal): Promise<MapDisplayResult> {
    if (Object.keys(args).some(key => !["searchResultId", "geoJson", "attribution", "attributionUrl"].includes(key)) || (args.searchResultId === undefined) === (args.geoJson === undefined)) throw new Error("Choose one owner-bound map result or supplied GeoJSON, not both.");
    let source: MapSearchResult | undefined;
    if (args.searchResultId !== undefined) { const db = this.database(); try { const id = text(args.searchResultId, "result id"), info = db.prepare("SELECT length(CAST(body AS BLOB)) AS bytes FROM results WHERE id=? AND brain_id=?").get(id, brainId);
        if (info) await this.admit(Number(info.bytes) * 16 + 65536);
        const row = db.prepare("SELECT body,body_sha FROM results WHERE id=? AND brain_id=?").get(id, brainId);
        if (typeof row?.body !== "string") throw new Error("That map lookup is not available to this instance. Use retained supplied GeoJSON or an explicitly permitted new search.");
        if (hash(row.body) !== row.body_sha) throw new Error("The retained map lookup failed its exact content checksum.");
        source = JSON.parse(row.body) as MapSearchResult;
      } finally { db.close(); }
    }
    signal.throwIfAborted(); const data = source?.geoJson ?? args.geoJson;
    await this.admit(Buffer.byteLength(JSON.stringify(data)) * 16 + 65536); const checked = checkedMapGeoJson(data);
    return { format: "omni-map-display", version: 1, displayMode: "offline-coordinate-canvas", coordinateReferenceSystem: "EPSG:4326",
      dataOrigin: source ? "search-result" : "supplied-geojson", ...checked, attribution: source?.attribution ?? text(args.attribution, "supplied-data attribution"),
      ...(source ? { attributionUrl: source.attributionUrl, search: { connectionId: source.connectionId, connectionRevision: source.connectionRevision, query: source.query, provenance: source.provenance } }
        : args.attributionUrl !== undefined ? { attributionUrl: attributionUrl(args.attributionUrl) } : {}), basemapDownloaded: false, untrustedTaskData: true };
  }
}
