import { useRef, useState, type SyntheticEvent } from "react";
import type { PluginPanelHistoryPage, PluginPanelHistorySourcePage } from "../../shared/pluginInteractivePanels";
import type { PluginsApi } from "../../shared/plugins";

const HISTORY_PAGE_BYTES = 64 * 1024;
const RECORD_PAGE_BYTES = 64 * 1024;
const SOURCE_PAGE_SIZE = 32;

type HistoryRecord = PluginPanelHistoryPage["records"][number];

interface HistorySnapshot extends PluginPanelHistoryPage { scope: string }
interface SourceSnapshot extends PluginPanelHistorySourcePage { brainId: string }
export interface PanelHistoryRead {
  chunks: string[];
  loadedBytes: number;
  bytes: number;
  sha256: string;
  nextOffset?: number;
}

function scopeKey(brainId: string, sourceSha256: string): string {
  return `${brainId}\u0000${sourceSha256}`;
}

function errorText(value: unknown): string {
  return value instanceof Error ? value.message : String(value);
}

function base64Bytes(value: string): Uint8Array {
  if (value.length % 4 !== 0 || !/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/.test(value)) {
    throw new Error("Panel history returned invalid base64 data.");
  }
  const decoded = globalThis.atob(value), bytes = new Uint8Array(decoded.length);
  for (let index = 0; index < decoded.length; index++) bytes[index] = decoded.charCodeAt(index);
  return bytes;
}

export function appendPanelHistoryBytes(
  current: PanelHistoryRead | undefined,
  expected: Pick<HistoryRecord, "sha256" | "bytes">,
  offset: number,
  page: { text: string; bytes: number; sha256: string; nextOffset?: number }
): PanelHistoryRead {
  if (expected.bytes === undefined || page.sha256 !== expected.sha256 || page.bytes !== expected.bytes ||
      current?.sha256 !== undefined && current.sha256 !== expected.sha256 || offset !== (current?.loadedBytes ?? 0)) {
    throw new Error("Panel history byte page changed its record identity.");
  }
  const decoded = base64Bytes(page.text), loadedBytes = offset + decoded.byteLength;
  if (decoded.byteLength === 0 && loadedBytes < page.bytes || loadedBytes > page.bytes ||
      page.nextOffset !== undefined && page.nextOffset !== loadedBytes ||
      page.nextOffset === undefined && loadedBytes !== page.bytes) {
    throw new Error("Panel history byte page did not cover the declared range.");
  }
  return {
    chunks: [...(current?.chunks ?? []), page.text], loadedBytes, bytes: page.bytes, sha256: page.sha256,
    ...(page.nextOffset === undefined ? {} : { nextOffset: page.nextOffset })
  };
}

export function panelHistoryReadText(value: PanelHistoryRead): string {
  const chunks = value.chunks.map(base64Bytes), bytes = new Uint8Array(value.loadedBytes);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return new TextDecoder("utf-8").decode(bytes, { stream: value.nextOffset !== undefined });
}

function appendedRecords(current: HistoryRecord[], next: HistoryRecord[]): HistoryRecord[] {
  const seen = new Set(current.map(record => record.sequence)), result = [...current];
  for (const record of next) if (!seen.has(record.sequence)) { seen.add(record.sequence); result.push(record); }
  return result;
}

function appendedSources(
  current: PluginPanelHistorySourcePage["sources"],
  next: PluginPanelHistorySourcePage["sources"]
): PluginPanelHistorySourcePage["sources"] {
  const seen = new Set(current.map(source => source.sha256)), result = [...current];
  for (const source of next) if (!seen.has(source.sha256)) { seen.add(source.sha256); result.push(source); }
  return result;
}

export function PluginPanelHistoryRecord({ record, read, busy, onRead }: {
  record: HistoryRecord;
  read?: PanelHistoryRead;
  busy: boolean;
  onRead(record: HistoryRecord): void;
}) {
  return <details className="plugin-panel-history__record">
    <summary><strong>{record.kind}</strong><span>Event {record.sequence}</span></summary>
    <dl>
      <div><dt>View ID</dt><dd>{record.viewId}</dd></div>
      {record.requestId ? <div><dt>Request ID</dt><dd>{record.requestId}</dd></div> : null}
      <div><dt>Content hash</dt><dd>{record.sha256}</dd></div>
    </dl>
    {record.value !== undefined ? <pre>{JSON.stringify(record.value, null, 2)}</pre> : null}
    {record.bytes !== undefined ? <>
      {read ? <pre>{panelHistoryReadText(read)}</pre> : <p>{record.bytes.toLocaleString()} bytes · content available in bounded pages.</p>}
      {read?.nextOffset !== undefined || !read ? <button type="button" className="button" disabled={busy} onClick={() => onRead(record)}>
        {read ? "Read next 64 KiB page" : "Read first 64 KiB page"}
      </button> : <p role="status">Complete record · {read.loadedBytes.toLocaleString()} bytes read.</p>}
    </> : null}
  </details>;
}

export function PluginPanelHistory({ brainId, api }: { brainId: string; api: PluginsApi }) {
  const [open, setOpen] = useState(false);
  const [selectedSource, setSelectedSource] = useState("");
  const [history, setHistory] = useState<HistorySnapshot>();
  const [sources, setSources] = useState<SourceSnapshot>();
  const [reads, setReads] = useState<Record<string, PanelHistoryRead>>({});
  const [historyBusy, setHistoryBusy] = useState(false);
  const [sourcesBusy, setSourcesBusy] = useState(false);
  const [recordBusy, setRecordBusy] = useState("");
  const [error, setError] = useState<{ scope: string; message: string }>();
  const openRef = useRef(false), brainRef = useRef(brainId), sourceRef = useRef(selectedSource);
  const historyEpoch = useRef(0), sourcesEpoch = useRef(0), bytesEpoch = useRef(0), readsRef = useRef(reads);
  brainRef.current = brainId; sourceRef.current = selectedSource; readsRef.current = reads;
  const currentScope = scopeKey(brainId, selectedSource);
  const currentHistory = history?.scope === currentScope ? history : undefined;
  const currentSources = sources?.brainId === brainId ? sources : undefined;
  const visibleError = error?.scope === currentScope || error?.scope === brainId ? error.message : "";

  const current = () => openRef.current && brainRef.current === brainId;
  const refreshHistory = async (sourceSha256: string): Promise<void> => {
    const request = ++historyEpoch.current, scope = scopeKey(brainId, sourceSha256);
    setHistoryBusy(true); setError(undefined); setReads({}); bytesEpoch.current++;
    try {
      const page = await api.panelHistory(brainId, { ...(sourceSha256 ? { sourceSha256 } : {}), maxBytes: HISTORY_PAGE_BYTES });
      if (!current() || sourceRef.current !== sourceSha256 || request !== historyEpoch.current) return;
      setHistory({ ...page, scope });
    } catch (failure) {
      if (current() && sourceRef.current === sourceSha256 && request === historyEpoch.current) setError({ scope, message: errorText(failure) });
    } finally {
      if (current() && sourceRef.current === sourceSha256 && request === historyEpoch.current) setHistoryBusy(false);
    }
  };

  const refreshAll = async (): Promise<void> => {
    const sourceSha256 = sourceRef.current, historyRequest = ++historyEpoch.current, sourceRequest = ++sourcesEpoch.current;
    const scope = scopeKey(brainId, sourceSha256);
    setHistoryBusy(true); setSourcesBusy(true); setError(undefined); setReads({}); bytesEpoch.current++;
    try {
      const [historyPage, sourcePage] = await Promise.all([
        api.panelHistory(brainId, { ...(sourceSha256 ? { sourceSha256 } : {}), maxBytes: HISTORY_PAGE_BYTES }),
        api.panelHistorySources(brainId, { limit: SOURCE_PAGE_SIZE })
      ]);
      if (!current() || sourceRef.current !== sourceSha256 || historyRequest !== historyEpoch.current || sourceRequest !== sourcesEpoch.current) return;
      setHistory({ ...historyPage, scope }); setSources({ ...sourcePage, brainId });
    } catch (failure) {
      if (current() && sourceRef.current === sourceSha256 && historyRequest === historyEpoch.current && sourceRequest === sourcesEpoch.current) {
        setError({ scope, message: errorText(failure) });
      }
    } finally {
      if (current() && sourceRef.current === sourceSha256 && historyRequest === historyEpoch.current) setHistoryBusy(false);
      if (current() && sourceRequest === sourcesEpoch.current) setSourcesBusy(false);
    }
  };

  const loadMoreHistory = async (): Promise<void> => {
    if (!currentHistory?.nextSequence || historyBusy) return;
    const sourceSha256 = selectedSource, afterSequence = currentHistory.nextSequence, request = ++historyEpoch.current;
    setHistoryBusy(true); setError(undefined);
    try {
      const page = await api.panelHistory(brainId, { ...(sourceSha256 ? { sourceSha256 } : {}), afterSequence, maxBytes: HISTORY_PAGE_BYTES });
      if (!current() || sourceRef.current !== sourceSha256 || request !== historyEpoch.current) return;
      setHistory(snapshot => snapshot?.scope === currentScope ? { ...page, scope: currentScope,
        records: appendedRecords(snapshot.records, page.records) } : snapshot);
    } catch (failure) {
      if (current() && sourceRef.current === sourceSha256 && request === historyEpoch.current) setError({ scope: currentScope, message: errorText(failure) });
    } finally { if (current() && sourceRef.current === sourceSha256 && request === historyEpoch.current) setHistoryBusy(false); }
  };

  const loadMoreSources = async (): Promise<void> => {
    if (!currentSources?.nextSha256 || sourcesBusy) return;
    const afterSha256 = currentSources.nextSha256, request = ++sourcesEpoch.current;
    setSourcesBusy(true); setError(undefined);
    try {
      const page = await api.panelHistorySources(brainId, { afterSha256, limit: SOURCE_PAGE_SIZE });
      if (!current() || request !== sourcesEpoch.current) return;
      setSources(snapshot => snapshot?.brainId === brainId ? { ...page, brainId,
        sources: appendedSources(snapshot.sources, page.sources) } : snapshot);
    } catch (failure) {
      if (current() && request === sourcesEpoch.current) setError({ scope: brainId, message: errorText(failure) });
    } finally { if (current() && request === sourcesEpoch.current) setSourcesBusy(false); }
  };

  const readRecord = async (record: HistoryRecord): Promise<void> => {
    if (record.bytes === undefined || recordBusy) return;
    const sourceSha256 = selectedSource, key = `${sourceSha256}/${record.sequence}/${record.sha256}`;
    const prior = readsRef.current[key], offset = prior?.nextOffset ?? prior?.loadedBytes ?? 0;
    if (prior && prior.nextOffset === undefined) return;
    const request = ++bytesEpoch.current;
    setRecordBusy(key); setError(undefined);
    try {
      const page = await api.panelHistoryBytes(brainId, record.sequence, offset, RECORD_PAGE_BYTES, sourceSha256 || undefined);
      if (!current() || sourceRef.current !== sourceSha256 || request !== bytesEpoch.current) return;
      const next = appendPanelHistoryBytes(prior, record, offset, page);
      setReads(values => ({ ...values, [key]: next }));
    } catch (failure) {
      if (current() && sourceRef.current === sourceSha256 && request === bytesEpoch.current) setError({ scope: currentScope, message: errorText(failure) });
    } finally { if (current() && sourceRef.current === sourceSha256 && request === bytesEpoch.current) setRecordBusy(""); }
  };

  const toggle = (event: SyntheticEvent<HTMLDetailsElement>): void => {
    const next = event.currentTarget.open;
    openRef.current = next; setOpen(next);
    if (!next) {
      historyEpoch.current++; sourcesEpoch.current++; bytesEpoch.current++;
      setHistoryBusy(false); setSourcesBusy(false); setRecordBusy("");
    } else if (!currentHistory) void refreshAll();
  };
  const selectSource = (sourceSha256: string): void => {
    sourceRef.current = sourceSha256; setSelectedSource(sourceSha256); setHistory(undefined); setReads({}); setRecordBusy("");
    historyEpoch.current++; bytesEpoch.current++;
    if (openRef.current) void refreshHistory(sourceSha256);
  };

  return <details className="plugin-panel-history" onToggle={toggle}>
    <summary>Panel history</summary>
    {open ? <div className="plugin-panel-history__body">
      <p>Durable panel requests and actual results. History is read only in bounded pages while this section is open.</p>
      <div className="plugin-manager__actions">
        <label>History source<select value={selectedSource} disabled={historyBusy || sourcesBusy}
          onChange={event => selectSource(event.currentTarget.value)}>
          <option value="">Current instance</option>
          {(currentSources?.sources ?? []).map(source => <option key={source.sha256} value={source.sha256}>
            {source.owner} · {source.sha256.slice(0, 12)}… · {source.bytes.toLocaleString()} bytes
          </option>)}
        </select></label>
        <button type="button" className="button" disabled={historyBusy || sourcesBusy} onClick={() => void refreshAll()}>Refresh</button>
      </div>
      {currentSources && currentSources.total > currentSources.sources.length ? <div className="plugin-manager__actions">
        <span>Showing {currentSources.sources.length.toLocaleString()} of {currentSources.total.toLocaleString()} archived sources.</span>
        {currentSources.nextSha256 ? <button type="button" className="button" disabled={sourcesBusy} onClick={() => void loadMoreSources()}>Load more sources</button> : null}
      </div> : null}
      {currentHistory ? <>
        <p role="status">Showing {currentHistory.records.length.toLocaleString()} of {currentHistory.total.toLocaleString()} history records.</p>
        <div className="plugin-panel-history__records">{currentHistory.records.map(record => {
          const key = `${selectedSource}/${record.sequence}/${record.sha256}`;
          return <PluginPanelHistoryRecord key={key} record={record} read={reads[key]} busy={recordBusy === key} onRead={readRecord} />;
        })}</div>
        {currentHistory.nextSequence ? <button type="button" className="button" disabled={historyBusy} onClick={() => void loadMoreHistory()}>Load next history page</button> : null}
        {!currentHistory.records.length ? <p>No panel history is retained for this source.</p> : null}
      </> : historyBusy ? <p role="status">Loading one bounded history page…</p> : null}
      {visibleError ? <p role="alert" className="plugin-manager__error">{visibleError}</p> : null}
    </div> : null}
  </details>;
}
