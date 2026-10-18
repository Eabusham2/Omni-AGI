import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ApiTeacherModelDiscoveryResult, ApiTeacherProvider, OmniApi } from "../../shared/types";
import { conciseUiMessage } from "./uiPresentation";

type DiscoveryApi = Pick<OmniApi["teacher"], "discoverModels" | "cancelModelDiscovery">;
interface Props { provider: ApiTeacherProvider; model: string; onChange(value: string): void; configured: boolean; credentialEpoch: number; api?: DiscoveryApi }
export function modelDiscoveryIds(value: ApiTeacherModelDiscoveryResult, provider: ApiTeacherProvider, requestId: string): string[] {
  if (value.format !== "omni-api-teacher-model-discovery" || value.version !== 1 || value.provider !== provider || value.requestId !== requestId || value.complete !== true || value.generationCompatibilityVerified !== false) throw new Error("Model list does not belong to this complete provider request.");
  return [...new Set(value.models.map(item => {
    if (item.provider !== provider || !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/.test(item.id)) throw new Error("Provider model identifier is invalid.");
    return item.id;
  }))].sort((a, b) => a.localeCompare(b));
}
export function modelChoicePage(ids: string[], filter: string, page: number): { ids: string[]; total: number; start: number } {
  const query = filter.trim().toLocaleLowerCase(), matches = ids.filter(id => id.toLocaleLowerCase().includes(query));
  const start = Math.min(Math.max(0, Math.floor(page)) * 100, Math.max(0, Math.floor((matches.length - 1) / 100) * 100));
  return { ids: matches.slice(start, start + 100), total: matches.length, start };
}
export function modelDiscoveryRequestIsCurrent(scope: object | null, owner: object, pending: string | undefined, requestId: string): boolean {
  return scope === owner && pending === requestId;
}
export function modelDiscoveryExclusions(value: ApiTeacherModelDiscoveryResult): Array<{ id: string; reason: string }> {
  const rows = value.excludedModels ?? [];
  if (!Array.isArray(rows) || value.excludedModelCount !== undefined && value.excludedModelCount !== rows.length || rows.some(row =>
    !row || !/^[a-zA-Z0-9][a-zA-Z0-9._:/-]{0,199}$/.test(row.id) || typeof row.reason !== "string" || !row.reason || row.reason.length > 2_048)) {
    throw new Error("Provider model exclusions are inconsistent.");
  }
  return rows;
}
export function ApiTeacherExcludedModels({ rows }: { rows: Array<{ id: string; reason: string }> }) {
  const [page, setPage] = useState(0), start = Math.min(page * 100, Math.max(0, Math.floor((rows.length - 1) / 100) * 100));
  if (!rows.length) return null;
  return <details><summary>{rows.length.toLocaleString()} {rows.length === 1 ? "model requires" : "models require"} separate cloud consent</summary>
    <ul>{rows.slice(start, start + 100).map((row, index) => <li key={`${row.id}:${start + index}`}><code>{row.id}</code> · {row.reason}</li>)}</ul>
    {start > 0 ? <button type="button" className="button button--ghost" onClick={() => setPage(Math.max(0, page - 1))}>Previous excluded models</button> : null}
    {start + 100 < rows.length ? <button type="button" className="button button--ghost" onClick={() => setPage(page + 1)}>More excluded models</button> : null}
  </details>;
}
export function ApiTeacherModelPicker(props: Props) { return <PickerOwner key={`${props.provider}:${props.credentialEpoch}:${props.configured}`} {...props} />; }

function PickerOwner({ provider, model, onChange, configured, api }: Props) {
  const listId = useId(), scope = useRef<object | null>(null), pending = useRef<string | undefined>(undefined), stopRequested = useRef(false);
  const [ids, setIds] = useState<string[]>([]), [busy, setBusy] = useState(false), [stopping, setStopping] = useState(false), [error, setError] = useState(""), [notice, setNotice] = useState("");
  const [filter, setFilter] = useState(""), [page, setPage] = useState(0);
  const [excluded, setExcluded] = useState<Array<{ id: string; reason: string }>>([]);
  const available = configured || provider === "ollama";
  const choices = useMemo(() => modelChoicePage(ids, filter, page), [ids, filter, page]);
  useLayoutEffect(() => { const owner = {}; scope.current = owner; return () => { if (scope.current === owner) scope.current = null; }; }, [api, provider]);
  const refresh = useCallback(async () => {
    const owner = scope.current; if (!api || !available || !owner || pending.current) return;
    const requestId = crypto.randomUUID(); pending.current = requestId; stopRequested.current = false; setBusy(true); setStopping(false); setError(""); setNotice("");
    try {
      const result = await api.discoverModels({ provider, requestId });
      if (!modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) return;
      const models = modelDiscoveryIds(result, provider, requestId), excludedRows = modelDiscoveryExclusions(result); setIds(models); setExcluded(excludedRows);
      setNotice(`${models.length.toLocaleString()} provider-listed models. Generation compatibility is not verified.`);
    } catch (reason) {
      if (modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) {
        if (stopRequested.current) setNotice(conciseUiMessage(reason, "Discovery ended. Manual model is unchanged."));
        else setError(conciseUiMessage(reason, "Provider models could not load. Manual model is unchanged."));
      }
    } finally { if (modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) { pending.current = undefined; setBusy(false); setStopping(false); } }
  }, [api, available, provider]);
  useEffect(() => {
    void refresh();
    return () => { const requestId = pending.current; if (pending.current === requestId) pending.current = undefined; if (api && requestId) void api.cancelModelDiscovery({ provider, requestId }).catch(() => undefined); };
  }, [api, provider, refresh]);
  const stop = async () => {
    const owner = scope.current, requestId = pending.current; if (!api || !owner || !requestId || stopping) return;
    stopRequested.current = true; setStopping(true);
    try { const result = await api.cancelModelDiscovery({ provider, requestId }); if (modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) setNotice(result.cancelled ? "Waiting for discovery to acknowledge Stop…" : "Discovery already settled."); }
    catch (reason) { if (modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) setError(conciseUiMessage(reason, "Discovery Stop was not acknowledged.")); }
    finally { if (modelDiscoveryRequestIsCurrent(scope.current, owner, pending.current, requestId)) setStopping(false); }
  };
  return <div className="api-model-picker">
    <label><span>Model</span><input value={model} list={listId} autoComplete="off" spellCheck={false} onChange={event => onChange(event.target.value)} /><datalist id={listId}>{choices.ids.map(id => <option key={id} value={id} />)}</datalist></label>
    {available && api ? <button type="button" className="button button--ghost" disabled={stopping} onClick={() => void (busy ? stop() : refresh())}>{stopping ? "Awaiting discovery Stop…" : busy ? "Stop model discovery" : "Refresh model list"}</button> : <small>Add your provider key to discover models.</small>}
    {provider === "ollama" ? <small>Local Ollama inventory · no key or automatic model pull. Server availability and fully offline execution are not verified by listing.</small> : null}
    <ApiTeacherExcludedModels rows={excluded} />
    {ids.length ? <details className="api-model-picker__choices"><summary>Choose from provider models</summary>
      <label>Find a model<input value={filter} onChange={event => { setFilter(event.target.value); setPage(0); }} /></label>
      <label>Listed models<select value={choices.ids.includes(model) ? model : ""} onChange={event => { if (event.target.value) onChange(event.target.value); }}><option value="">Choose a model explicitly</option>{choices.ids.map(id => <option key={id} value={id}>{id}</option>)}</select></label>
      <div className="api-model-picker__pages">{choices.start > 0 ? <button type="button" className="button button--ghost" onClick={() => setPage(Math.max(0, Math.floor(choices.start / 100) - 1))}>Previous models</button> : null}{choices.start + 100 < choices.total ? <button type="button" className="button button--ghost" onClick={() => setPage(Math.floor(choices.start / 100) + 1)}>More models</button> : null}</div>
      <small>{choices.total.toLocaleString()} matching models{choices.total > 100 ? ` · showing ${choices.start + 1}–${Math.min(choices.start + 100, choices.total)}` : ""}</small>
    </details> : null}
    {notice ? <small role="status">{notice}</small> : null}{error ? <small role="alert">{error}</small> : null}
  </div>;
}
