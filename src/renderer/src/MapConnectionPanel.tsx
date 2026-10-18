import { useEffect, useState } from "react";
import type { MapConnection, MapConnectionSave, MapConnectionsApi } from "../../shared/maps";
import "./maps.css";

function availableApi(): MapConnectionsApi | undefined {
  const api = typeof window === "undefined" ? undefined : window.omni?.maps;
  return api && [api.list, api.save, api.remove].every(value => typeof value === "function") ? api : undefined;
}
const blank = (): MapConnectionSave => ({ id: "private-photon", expectedRevision: 0, label: "Private Photon", endpoint: "http://127.0.0.1:2322/api/",
  attribution: "© OpenStreetMap contributors", attributionUrl: "https://www.openstreetmap.org/copyright", enabled: true,
  retainedResultsAccepted: true, minimumIntervalMs: 1000, cacheTtlMs: 86400000, timeoutMs: 30000 });

export function MapConnectionPanel({ api = availableApi() }: { api?: MapConnectionsApi }) {
  const [entries, setEntries] = useState<MapConnection[]>([]), [draft, setDraft] = useState<MapConnectionSave>();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [accepted, setAccepted] = useState(false), [credential, setCredential] = useState<"keep" | "replace" | "clear">("keep"), [token, setToken] = useState("");
  useEffect(() => { if (!api) return; let active = true;
    void api.list().then(value => { if (active) setEntries(value); }, failure => { if (active) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { active = false; };
  }, [api]);
  if (!api) return null;
  const edit = (value?: MapConnection): void => {
    setDraft(value ? { id: value.id, expectedRevision: value.revision, label: value.label, endpoint: value.endpoint,
      attribution: value.attribution, attributionUrl: value.attributionUrl, enabled: value.enabled, retainedResultsAccepted: true,
      minimumIntervalMs: value.minimumIntervalMs, cacheTtlMs: value.cacheTtlMs, timeoutMs: value.timeoutMs } : blank());
    setAccepted(false); setCredential("keep"); setToken(""); setError("");
  };
  const save = async (): Promise<void> => {
    if (!draft || !accepted || busy) return; setBusy(true); setError("");
    try { const result = await api.save({ ...draft, ...(credential === "replace" ? { bearerToken: token } : credential === "clear" ? { bearerToken: null } : {}) });
      setEntries(current => [...current.filter(value => value.id !== result.id), result]); setDraft(undefined); setToken("");
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure));
      await api.list().then(setEntries, () => undefined); }
    finally { setBusy(false); }
  };
  return <details className="map-connections"><summary>Private map connections</summary>
    <p>Search a Photon service you operate or are authorized to use. No public geocoder, account or paid plan is selected automatically.</p>
    {error ? <p role="alert" className="map-connections__error">{error}</p> : null}
    <div className="map-connections__list">{entries.map(value => <div key={value.id}><span><strong>{value.label}</strong><small>{value.endpoint}</small><small>{value.enabled ? "Enabled" : "Disabled"} · revision {value.revision} · {value.credentialConfigured ? `${value.credentialPersistence} credential configured` : "No credential"}</small></span>
      <button type="button" className="button button--ghost" disabled={busy} onClick={() => edit(value)}>Review or edit</button>
      <button type="button" className="button button--ghost" disabled={busy} onClick={() => { setBusy(true); void api.remove(value.id, value.revision)
        .then(() => setEntries(current => current.filter(item => item.id !== value.id)), failure => setError(failure instanceof Error ? failure.message : String(failure)))
        .finally(() => setBusy(false)); }}>Remove connection</button></div>)}</div>
    <button type="button" className="button button--secondary" disabled={busy} onClick={() => edit()}>Add private connection</button>
    <button type="button" className="button button--ghost" disabled={busy} onClick={() => { setBusy(true); void api.list().then(setEntries,
      failure => setError(failure instanceof Error ? failure.message : String(failure))).finally(() => setBusy(false)); }}>Refresh connections</button>
    {draft ? <fieldset disabled={busy}><legend>{draft.expectedRevision ? "Review connection" : "New private connection"}</legend>
      <label>Connection ID<input value={draft.id} readOnly={draft.expectedRevision > 0} onChange={event => setDraft({ ...draft, id: event.currentTarget.value })} /></label>
      <label>Label<input value={draft.label} onChange={event => setDraft({ ...draft, label: event.currentTarget.value })} /></label>
      <label>Exact Photon search endpoint<input type="url" value={draft.endpoint} onChange={event => { setDraft({ ...draft, endpoint: event.currentTarget.value }); setAccepted(false); }} /></label>
      <label>Data attribution<input value={draft.attribution} onChange={event => setDraft({ ...draft, attribution: event.currentTarget.value })} /></label>
      <label>Attribution or licence link<input type="url" value={draft.attributionUrl} onChange={event => setDraft({ ...draft, attributionUrl: event.currentTarget.value })} /></label>
      <label>Credential action<select value={credential} onChange={event => { setCredential(event.currentTarget.value as typeof credential); setToken(""); }}>
        <option value="keep">Keep current / no credential</option><option value="replace">Replace bearer credential</option><option value="clear">Clear credential</option></select></label>
      {credential === "replace" ? <label>Bearer credential<input type="password" autoComplete="off" value={token} onChange={event => setToken(event.currentTarget.value)} /></label> : null}
      <details><summary>Request timing and cache</summary>{([
        ["minimumIntervalMs", "Minimum request interval (seconds)"], ["cacheTtlMs", "Query cache duration (seconds)"], ["timeoutMs", "Request timeout (seconds)"]
      ] as const).map(([key, label]) => <label key={key}>{label}<input type="number" min={key === "timeoutMs" ? .001 : 0} step=".001" value={(draft[key] ?? 0) / 1000}
        onChange={event => setDraft({ ...draft, [key]: Number(event.currentTarget.value) * 1000 })} /></label>)}</details>
      <label className="map-connections__check"><input type="checkbox" checked={draft.enabled} onChange={event => setDraft({ ...draft, enabled: event.currentTarget.checked })} />Enable this connection; each instance still uses its own map-tool permissions.</label>
      <label className="map-connections__check"><input type="checkbox" checked={accepted} onChange={event => setAccepted(event.currentTarget.checked)} />I am authorized to use this service and retain its results in ordinary learning, history and complete exports.</label>
      <div className="map-connections__actions"><button type="button" className="button button--primary" disabled={!accepted} onClick={() => void save()}>Save reviewed connection</button>
        <button type="button" className="button button--ghost" onClick={() => { setDraft(undefined); setToken(""); }}>Discard draft</button></div>
    </fieldset> : null}
  </details>;
}
