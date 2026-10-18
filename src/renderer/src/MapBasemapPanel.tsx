import { useEffect, useRef, useState } from "react";
import type { MapBasemapProfile, MapBasemapSave, MapBasemapsApi, MapRasterAuthentication, MapRasterSource } from "../../shared/mapBasemaps";
import "./maps.css";

const blank = (): MapBasemapSave => ({ id: "my-map", expectedRevision: 0, label: "My map", enabled: false,
  source: { kind: "raster-xyz", template: "http://127.0.0.1:8080/{z}/{x}/{y}.png", tileSize: 256, minimumZoom: 0, maximumZoom: 22 }, authentication: { kind: "none" },
  attribution: "© OpenStreetMap contributors", attributionUrl: "https://www.openstreetmap.org/copyright", rights: {
    retentionLearningAndRedistributionAuthorized: true, authorizationReference: "", licenseUrl: "https://www.openstreetmap.org/copyright" }, minimumIntervalMs: 1000, cacheTtlMs: 86400000, timeoutMs: 30000 });
type Flight = { api: MapBasemapsApi };
/** Explicit settings only. Opening this panel never contacts any map provider. */
export function MapBasemapPanel({ api }: { api?: MapBasemapsApi }) {
  const [entries, setEntries] = useState<MapBasemapProfile[]>([]), [draft, setDraft] = useState<MapBasemapSave>();
  const [busy, setBusy] = useState(false), [error, setError] = useState(""), [accepted, setAccepted] = useState(false);
  const [keyMode, setKeyMode] = useState<"keep" | "replace" | "clear">("keep"), [key, setKey] = useState("");
  const currentApi = useRef(api), flight = useRef<Flight | undefined>(undefined); currentApi.current = api;
  useEffect(() => { flight.current = undefined; setBusy(false); setEntries([]); setDraft(undefined); setKey(""); setError(""); return () => { flight.current = undefined; }; }, [api]);
  const run = async (operation: (owner: MapBasemapsApi, current: () => boolean) => Promise<void>): Promise<void> => {
    if (!api || flight.current) return; const owner = { api }; flight.current = owner; setBusy(true); setError("");
    try { await operation(api, () => flight.current === owner && currentApi.current === api); } catch (failure) { if (flight.current === owner && currentApi.current === api) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (flight.current === owner && currentApi.current === api) { flight.current = undefined; setBusy(false); } }
  };
  const refresh = (): void => { void run(async (owner, current) => { const values = await owner.list(); if (current()) setEntries(values); }); };
  const edit = (entry?: MapBasemapProfile): void => {
    if (entry) { const { revision, credentialConfigured: _configured, credentialPersistence: _persistence, ...body } = entry; setDraft({ ...body, expectedRevision: revision }); }
    else setDraft(blank()); setAccepted(false); setKeyMode("keep"); setKey(""); setError("");
  };
  const chooseSource = (kind: MapRasterSource["kind"]): void => {
    if (!draft) return; setAccepted(false); setKey(""); setKeyMode("replace");
    setDraft({ ...draft, source: kind === "raster-xyz" ? blank().source : { kind, username: "", styleId: "" }, authentication: kind === "raster-xyz" ? { kind: "none" } : { kind: "query", parameter: "access_token" } });
  };
  if (!api) return null;
  return <details className="map-connections map-basemaps" onToggle={event => { if (event.currentTarget.open && !entries.length && !busy) refresh(); }}><summary>Optional basemaps</summary>
    <p>The offline coordinate view remains the default. Add a source only if you want permitted map requests to contact it.</p>
    {error ? <p role="alert" className="map-connections__error">{error}</p> : null}
    <div className="map-connections__list">{entries.map(entry => <div key={entry.id}><span><strong>{entry.label}</strong><small>{entry.source.kind === "raster-xyz" ? entry.source.template : `Mapbox · ${entry.source.username}/${entry.source.styleId}`}</small>
      <small>{entry.enabled ? "Enabled" : "Disabled"} · revision {entry.revision} · {entry.credentialConfigured ? `${entry.credentialPersistence} credential` : "No credential"}</small></span>
      <div className="map-connections__actions"><button type="button" className="button button--ghost" disabled={busy} onClick={() => edit(entry)}>Review or edit</button>
        <button type="button" className="button button--ghost" disabled={busy} onClick={() => void run(async (owner, current) => { await owner.remove(entry.id, entry.revision); if (current()) setEntries(values => values.filter(value => value.id !== entry.id)); })}>Remove profile</button></div></div>)}</div>
    <div className="map-connections__actions"><button type="button" className="button button--secondary" disabled={busy} onClick={() => edit()}>Add basemap</button><button type="button" className="button button--ghost" disabled={busy} onClick={refresh}>Refresh profiles</button></div>
    {draft ? <fieldset disabled={busy}><legend>{draft.expectedRevision ? "Review basemap" : "New basemap"}</legend>
      <label>Profile ID<input value={draft.id} readOnly={draft.expectedRevision > 0} onChange={event => setDraft({ ...draft, id: event.currentTarget.value })} /></label>
      <label>Label<input value={draft.label} onChange={event => setDraft({ ...draft, label: event.currentTarget.value })} /></label>
      <label>Source<select value={draft.source.kind} onChange={event => chooseSource(event.currentTarget.value as MapRasterSource["kind"])}><option value="raster-xyz">Authorized XYZ service — keyless or keyed</option><option value="mapbox-static-image">Mapbox Static Images — compatible contract required</option></select></label>
      {draft.source.kind === "raster-xyz" ? <>
        <label>Exact XYZ template<input type="url" value={draft.source.template} onChange={event => { if (draft.source.kind === "raster-xyz") setDraft({ ...draft, source: { ...draft.source, template: event.currentTarget.value } }); setAccepted(false); }} /></label>
        <p>Use a service you operate or are authorized to retain, learn from and redistribute. Public OpenStreetMap tile servers are not an offline/export source.</p>
        <details><summary>Tile format and zoom range</summary>{([["tileSize", "Tile side (pixels)"], ["minimumZoom", "Minimum zoom"], ["maximumZoom", "Maximum zoom"]] as const).map(([field, label]) => <label key={field}>{label}<input type="number" min={field === "tileSize" ? 1 : 0} step="1" value={draft.source.kind === "raster-xyz" ? draft.source[field] : 0}
          onChange={event => { if (draft.source.kind === "raster-xyz") setDraft({ ...draft, source: { ...draft.source, [field]: Number(event.currentTarget.value) } }); }} /></label>)}</details>
      </> : <>
        <label>Mapbox style owner<input value={draft.source.username} onChange={event => { if (draft.source.kind === "mapbox-static-image") setDraft({ ...draft, source: { ...draft.source, username: event.currentTarget.value } }); setAccepted(false); }} /></label>
        <label>Exact style ID<input value={draft.source.styleId} onChange={event => { if (draft.source.kind === "mapbox-static-image") setDraft({ ...draft, source: { ...draft.source, styleId: event.currentTarget.value } }); setAccepted(false); }} /></label>
        <p>Ordinary API access is not permission for model learning or complete redistribution. Enable only under your compatible agreement; logo and attribution are requested. Geocoding is not enabled.</p>
      </>}
      {draft.source.kind === "raster-xyz" ? <label>Authentication<select value={draft.authentication.kind} onChange={event => { const kind = event.currentTarget.value as MapRasterAuthentication["kind"]; setDraft({ ...draft, authentication: kind === "query" ? { kind, parameter: "key" } : { kind } }); setKey(""); setKeyMode(kind === "none" ? "clear" : "replace"); setAccepted(false); }}>
        <option value="none">Keyless</option><option value="bearer">Bearer credential</option><option value="query">Query credential</option></select></label> : null}
      {draft.authentication.kind === "query" ? <label>Credential parameter<input value={draft.authentication.parameter} readOnly={draft.source.kind === "mapbox-static-image"} onChange={event => setDraft({ ...draft, authentication: { kind: "query", parameter: event.currentTarget.value } })} /></label> : null}
      {draft.authentication.kind !== "none" ? <><label>Credential action<select value={keyMode} onChange={event => { setKeyMode(event.currentTarget.value as typeof keyMode); setKey(""); }}><option value="keep">Keep current credential</option><option value="replace">Replace credential</option><option value="clear">Clear credential</option></select></label>
        {keyMode === "replace" ? <label>Credential<input type="password" autoComplete="off" value={key} onChange={event => setKey(event.currentTarget.value)} /></label> : null}</> : null}
      <label>Attribution<input value={draft.attribution} onChange={event => setDraft({ ...draft, attribution: event.currentTarget.value })} /></label>
      <label>Attribution link<input type="url" value={draft.attributionUrl} onChange={event => setDraft({ ...draft, attributionUrl: event.currentTarget.value })} /></label>
      <label>Source license / agreement link<input type="url" value={draft.rights.licenseUrl} onChange={event => { setDraft({ ...draft, rights: { ...draft.rights, licenseUrl: event.currentTarget.value } }); setAccepted(false); }} /></label>
      <label>Authorization reference<input placeholder="License or your compatible agreement reference" value={draft.rights.authorizationReference} onChange={event => { setDraft({ ...draft, rights: { ...draft.rights, authorizationReference: event.currentTarget.value } }); setAccepted(false); }} /></label>
      <details><summary>Request timing and cache</summary>{([["minimumIntervalMs", "Minimum interval (seconds)"], ["cacheTtlMs", "Cache duration (seconds)"], ["timeoutMs", "Timeout (seconds)"]] as const).map(([field, label]) => <label key={field}>{label}<input type="number" min={field === "timeoutMs" ? .001 : 0} step=".001" value={draft[field] / 1000} onChange={event => setDraft({ ...draft, [field]: Number(event.currentTarget.value) * 1000 })} /></label>)}</details>
      <label className="map-connections__check"><input type="checkbox" checked={draft.enabled} onChange={event => setDraft({ ...draft, enabled: event.currentTarget.checked })} />Enable this source. Each request still requires this instance’s map permission.</label>
      <label className="map-connections__check"><input type="checkbox" checked={accepted} onChange={event => setAccepted(event.currentTarget.checked)} />I have authority to retain, learn from and redistribute these outputs, including complete instance exports. This declaration does not grant rights or replace the provider’s terms.</label>
      <div className="map-connections__actions"><button type="button" className="button button--primary" disabled={!accepted || !draft.rights.authorizationReference.trim()} onClick={() => void run(async (owner, current) => {
        const value = await owner.save({ ...draft, ...(draft.authentication.kind === "none" || keyMode === "clear" ? { credential: null } : keyMode === "replace" ? { credential: key } : {}) });
        if (current()) { setEntries(values => [...values.filter(entry => entry.id !== value.id), value]); setDraft(undefined); setKey(""); }
      })}>Save reviewed basemap</button><button type="button" className="button button--ghost" onClick={() => { setDraft(undefined); setKey(""); setAccepted(false); }}>Discard draft</button></div>
    </fieldset> : null}
  </details>;
}
