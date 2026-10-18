import { useEffect, useRef, useState, type CSSProperties } from "react";
import type { CortexPage, CortexElement, CortexBoundary, CortexActivity, CortexQuery } from "@shared/cortexInspection";
import { cortexActivityViewport, cortexInventoryLabel, cortexRecordLabel, cortexRequestKey } from "./cortexPresentation";
import "./brainMapCortex.css";

export function BrainMapCortex({ brainId, updatedAt, onBack }: { brainId: string; updatedAt: string; onBack: () => void }) {
  const [result, setResult] = useState<{ key: string; page: CortexPage } | null>(null);
  const [query, setQuery] = useState<CortexQuery>({ entity: "modules", pageSize: 64 });
  const [history, setHistory] = useState<Array<string | undefined>>([]);
  const [scroll, setScroll] = useState(0);
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null);
  const [pendingKey, setPendingKey] = useState("");
  const [captureChoice, setCaptureChoice] = useState({ brainId, enabled: false });
  const [activityResult, setActivityResult] = useState<{ key: string; activity: CortexActivity } | null>(null);
  const scrollViewport = useRef<HTMLDivElement>(null);
  const requestKey = cortexRequestKey(brainId, updatedAt, query);
  const page = result?.key === requestKey ? result.page : null;
  const selected = page?.selectedModule ?? null;
  const error = failure?.key === requestKey ? failure.message : "";
  const loading = pendingKey === requestKey;
  const capture = captureChoice.brainId === brainId && captureChoice.enabled;
  const activityKey = JSON.stringify([requestKey, selected?.id, capture]);
  const activity = activityResult?.key === activityKey ? activityResult.activity : null;
  const viewportKey = cortexRequestKey(brainId, "", query);
  useEffect(() => { setScroll(0); if (scrollViewport.current) scrollViewport.current.scrollTop = 0; }, [viewportKey]);
  useEffect(() => {
    let current = true;
    if (!window.omni?.brain.queryCortex) { setFailure({ key: requestKey, message: "Cortical inspection requires the native engine." }); return; }
    setPendingKey(requestKey); setFailure(null);
    void window.omni.brain.queryCortex(brainId, query).then((value) => {
      if (!current) return;
      if (value.brainId !== brainId || value.entity !== (query.entity ?? "modules")) throw new Error("Cortical inspection returned a different owner or query.");
      setResult({ key: requestKey, page: value });
    }).catch((cause) => {
      if (!current) return;
      setResult(null); setFailure({ key: requestKey, message: String(cause) });
      if (query.cursor) { setQuery((value) => ({ ...value, cursor: undefined, offset: 0 })); setHistory([]); }
    }).finally(() => { if (current) setPendingKey(""); });
    return () => { current = false; };
  }, [brainId, query, requestKey]);
  useEffect(() => {
    if (!selected || !page || !window.omni?.brain.cortexActivity) return;
    let current = true;
    let timer: number | undefined;
    const captureId = crypto.randomUUID();
    let configured: ReturnType<typeof cortexActivityViewport> = null;
    const api = window.omni.brain.cortexActivity;
    const release = () => { if (configured) void api(brainId, { ...configured, enabled: false }).catch(() => undefined); };
    const poll = async (request: NonNullable<typeof configured>) => {
      try {
        const value = await api(brainId, request);
        if (!current) { release(); return; }
        setActivityResult({ key: activityKey, activity: value });
        if (capture) timer = window.setTimeout(() => { void poll(request); }, 1500);
      } catch (cause) {
        if (current) setActivityResult({ key: activityKey, activity: { available: false, observed: false, observation: null, reason: String(cause) } });
      }
    };
    void api(brainId, { module: selected.activityOwner ?? selected.module, enabled: false, captureId }).then((descriptor) => {
      if (!current) return;
      const viewport = cortexActivityViewport(selected, query, page, descriptor, capture);
      configured = viewport ? { ...viewport, captureId } : null;
      if (configured) void poll(configured);
      else setActivityResult({ key: activityKey, activity: descriptor });
    }).catch((cause) => {
      if (current) setActivityResult({ key: activityKey, activity: { available: false, observed: false, observation: null, reason: String(cause) } });
    });
    return () => { current = false; if (timer !== undefined) window.clearTimeout(timer); release(); };
  }, [brainId, selected, page, capture, activityKey, query]);
  const choose = (next: CortexQuery) => { setQuery({ pageSize: 64, ...next }); setHistory([]); };
  const records = page?.records ?? [];
  const rowHeight = 38;
  const first = Math.max(0, Math.floor(scroll / rowHeight) - 3);
  const visible = records.slice(first, first + 16);
  const label = cortexRecordLabel;
  const open = (value: typeof records[number]) => {
    if ("module" in value) choose({ entity: "rows", moduleId: value.id });
    else if ("row" in value && !("column" in value) && selected) choose({ entity: "elements", moduleId: selected.id, row: value.row });
    else if ("tokenId" in value) {
      choose({ entity: "modules", search: "decoder.embedding._packed_forward_weight" });
    }
  };
  return <div className="content-page cortex-viewer">
    <div className="content-page__title content-page__title--compact"><div>
      <button className="content-page__back" onClick={onBack}>← Conversation</button>
      <span className="eyebrow-text">PACKED NATIVE CORTEX</span><h1>Model elements</h1>
      <p>Exact committed ternary bytes. Links identify computation—not a semantic explanation or hidden reasoning trace.</p>
    </div></div>
    <div className="cortex-viewer__toolbar">
      <button onClick={() => choose({ entity: "modules" })}>All modules</button>
      <button onClick={() => choose({ entity: "boundaries" })}>Observed token boundaries</button>
      <select aria-label="Cortical module group" value={query.group ?? ""} onChange={(event) => choose({ entity: "modules", group: event.target.value || undefined })}>
        <option value="">Every group</option>{page?.groups.map((group) => <option key={group}>{group}</option>)}
      </select>
      <input aria-label="Find cortical module" placeholder="Module path" value={query.search ?? ""} onChange={(event) => choose({ entity: "modules", search: event.target.value })} />
    </div>
    <div className="cortex-viewer__summary"><span>{page?.moduleCount.toLocaleString() ?? "…"} packed tensor owners</span>
      <span>{page ? cortexInventoryLabel(page) : "…"}</span>
      {page?.inventory?.repeatedContactPages ? <span>{page.inventory.repeatedContactPages.toLocaleString()} repeated-contact pages · {page.inventory.allocatedRepeatedContactSlots.toLocaleString()} allocated slots, not a live-contact count</span> : null}
      <span>{page ? (page.packedBytes + (page.connectionStateBytes ?? 0) + (page.structuralBytes ?? 0)).toLocaleString() : "…"} stored connection bytes</span>
      <span>Generation {page?.revision.slice(0, 12) ?? "…"}</span></div>
    {error ? <p role="alert">{error}</p> : null}
    <div className="cortex-viewer__layout"><section className="surface">
      <div className="cortex-viewer__toolbar"><strong>{query.entity ?? "modules"}</strong>
        {selected ? <span>{selected.module} · {selected.field}</span> : null}
        {selected ? <button onClick={() => choose({ entity: "links", moduleId: selected.id, row: query.row ?? 0 })}>Tensor-axis links</button> : null}
      </div>
      <div ref={scrollViewport} className="cortex-viewer__scroll" onScroll={(event) => setScroll(event.currentTarget.scrollTop)} role="list" aria-label="Virtualized cortical viewport" aria-busy={loading}>
        <div style={{ height: records.length * rowHeight, position: "relative" }}>
          {visible.map((value, index) => <button key={label(value)} title={label(value)} role="listitem" className="cortex-viewer__row"
            style={{ position: "absolute", top: (first + index) * rowHeight, height: rowHeight, width: "100%" } as CSSProperties}
            onClick={() => open(value)}>{label(value)}{"module" in value ? <small>{value.rows.toLocaleString()} × {value.columns?.toLocaleString() ?? "unknown logical width"}</small> : null}</button>)}
        </div>
      </div>
      <div className="cortex-viewer__toolbar"><button disabled={!history.length || loading} onClick={() => {
        const next = [...history]; const cursor = next.pop(); setHistory(next); setQuery((value) => ({ ...value, cursor }));
      }}>Previous</button><span>{page ? `${page.offset.toLocaleString()}–${(page.scannedThrough ?? page.offset + page.returned).toLocaleString()} of ${page.total.toLocaleString()}` : "Loading…"}</span>
      <button disabled={!page?.hasMore || loading} onClick={() => { setHistory((value) => [...value, query.cursor]); setQuery((value) => ({ ...value, cursor: page?.nextCursor ?? undefined })); }}>Next</button>
      <label>Go to {query.entity === "elements" ? "column" : "row/page offset"}<input type="number" min={0} aria-label="Exact cortical offset" onKeyDown={(event) => { if (event.key === "Enter") choose({ ...query, offset: Number(event.currentTarget.value), cursor: undefined }); }} /></label>
      </div>
      {query.entity === "elements" ? <div className="cortex-viewer__trits" aria-label="Exact selected ternary values">
        {(records as CortexElement[]).map((element) => <span key={element.ordinal ?? element.column} data-trit={element.value} title={`input ${element.column} → output ${element.row}; strength ${element.strength ?? "unreported"}; resistance ${element.resistance ?? "unreported"}; contribution ${element.effectiveContribution ?? "unreported"}${element.isRepeatedContact ? `; repeated contact ${element.ordinal}` : ""}`}>
          <small>{element.column}</small>{element.value > 0 ? "+1" : element.value}{element.strength !== undefined ? ` × ${element.strength}` : ""}</span>)}
      </div> : null}
      {query.entity === "boundaries" ? <p>{(records as CortexBoundary[]).filter((value) => value.kind.endsWith("boundary")).length} boundary markers on this page. Byte tokens address decoder embedding/output rows; that is not evidence those rows fired.</p> : null}
    </section><aside className="surface cortex-viewer__detail"><h2>{selected?.module ?? "Select a module"}</h2>
      {selected ? <dl><dt>Logical shape</dt><dd>{selected.logicalShape?.join(" × ") ?? "Not available"}</dd><dt>Storage layout</dt><dd>{selected.layout}</dd>
        <dt>Evidence</dt><dd>{selected.shapeEvidence}</dd><dt>Activity</dt><dd>Unobserved unless explicitly captured below</dd></dl> : null}
      <label><input type="checkbox" checked={capture} disabled={!selected} onChange={(event) => setCaptureChoice({ brainId, enabled: event.target.checked })} /> Observe this owner on actual forwards</label>
      {activity?.observation ? <><p>Observed {activity.observation.axis} {activity.observation.start}–{activity.observation.end}. {activity.observation.sample}. Not a full firing map.</p>
        <ol start={activity.observation.start}>{activity.observation.values.map((value, index) => <li key={index}>{value === null ? "Unobserved / nonfinite" : value.toPrecision(5)}</li>)}</ol></>
        : <p>{activity?.reason ?? "No actual forward observation for this viewport. We do not infer activation from confidence or weight sign."}</p>}
      <p>Idea owners present: {page?.relationships.ideaInputs.join(" · ") || "None reported in this inventory"}. Token owners present: {page?.relationships.tokenInputs.join(" · ") || "None reported"}. Stored coordinates do not establish semantic meaning.</p>
      <small>Selected codes are validated on read. Generation manifest is verified; the whole multi-GiB role payload is not rescanned per viewport.</small>
    </aside></div>
  </div>;
}
