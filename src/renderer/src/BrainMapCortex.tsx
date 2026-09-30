import { useEffect, useState, type CSSProperties } from "react";
import type { CortexPage, CortexModule, CortexElement, CortexBoundary, CortexActivity, CortexQuery } from "@shared/cortexInspection";
import "./brainMapCortex.css";

export function BrainMapCortex({ brainId, updatedAt, onBack }: { brainId: string; updatedAt: string; onBack: () => void }) {
  const [page, setPage] = useState<CortexPage | null>(null);
  const [selected, setSelected] = useState<CortexModule | null>(null);
  const [query, setQuery] = useState<CortexQuery>({ entity: "modules", pageSize: 64 });
  const [history, setHistory] = useState<Array<string | undefined>>([]);
  const [scroll, setScroll] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [capture, setCapture] = useState(false);
  const [activity, setActivity] = useState<CortexActivity | null>(null);
  useEffect(() => {
    let current = true;
    if (!window.omni?.brain.queryCortex) { setError("Cortical inspection requires the native engine."); return; }
    setLoading(true); setError("");
    void window.omni.brain.queryCortex(brainId, query).then((value) => {
      if (!current) return;
      setPage(value); setScroll(0);
      if (value.selectedModule) setSelected(value.selectedModule);
    }).catch((cause) => {
      if (!current) return;
      setError(String(cause));
      if (query.cursor) { setQuery((value) => ({ ...value, cursor: undefined, offset: 0 })); setHistory([]); }
    }).finally(() => { if (current) setLoading(false); });
    return () => { current = false; };
  }, [brainId, updatedAt, query]);
  useEffect(() => {
    if (!selected || !window.omni?.brain.cortexActivity) return;
    let current = true;
    const embedding = /embedding|latent_table|scale_delta|\.levels$/.test(selected.module);
    const start = query.entity === "rows" ? (page?.offset ?? 0)
      : query.entity === "elements" || query.entity === "links" ? (embedding ? page?.offset ?? 0 : query.row ?? 0) : 0;
    const request = { module: selected.module, enabled: capture, start, count: query.entity === "elements" && !embedding ? 1 : 64 };
    const poll = () => { void window.omni!.brain.cortexActivity(brainId, request).then((value) => { if (current) setActivity(value); }).catch(() => { if (current) setActivity(null); }); };
    poll();
    const timer = capture ? window.setInterval(poll, 1500) : undefined;
    return () => { current = false; if (timer !== undefined) window.clearInterval(timer);
      void window.omni?.brain.cortexActivity(brainId, { ...request, enabled: false }).catch(() => undefined); };
  }, [brainId, selected?.module, capture, query.entity, query.row, page?.offset]);
  const choose = (next: CortexQuery) => { setQuery({ pageSize: 64, ...next }); setHistory([]); };
  const records = page?.records ?? [];
  const rowHeight = 38;
  const first = Math.max(0, Math.floor(scroll / rowHeight) - 3);
  const visible = records.slice(first, first + 16);
  const label = (value: typeof records[number]) => {
    if ("module" in value) return `${value.module} · ${value.field}`;
    if ("tokenId" in value) return `${value.position}: token ${value.tokenId} · ${value.kind}`;
    if ("value" in value) return `row ${value.row}, column ${value.column}: ${value.value > 0 ? "+1" : value.value}`;
    if ("sourceIndex" in value) return `input ${value.sourceIndex} → output ${value.targetIndex}`;
    return `Output row ${value.row}`;
  };
  const open = (value: typeof records[number]) => {
    if ("module" in value) { setSelected(value); choose({ entity: "rows", moduleId: value.id }); }
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
      <button onClick={() => { setSelected(null); choose({ entity: "modules" }); }}>All modules</button>
      <button onClick={() => choose({ entity: "boundaries" })}>Observed token boundaries</button>
      <select aria-label="Cortical module group" value={query.group ?? ""} onChange={(event) => choose({ entity: "modules", group: event.target.value || undefined })}>
        <option value="">Every group</option>{page?.groups.map((group) => <option key={group}>{group}</option>)}
      </select>
      <input aria-label="Find cortical module" placeholder="Module path" value={query.search ?? ""} onChange={(event) => choose({ entity: "modules", search: event.target.value })} />
    </div>
    <div className="cortex-viewer__summary"><span>{page?.moduleCount.toLocaleString() ?? "…"} packed tensor owners</span>
      <span>{page?.logicalInventoryComplete ? page.logicalParameters.toLocaleString() : "Incomplete layout inventory"} logical synapses</span>
      <span>{page?.packedBytes.toLocaleString() ?? "…"} authoritative packed bytes</span>
      <span>Generation {page?.revision.slice(0, 12) ?? "…"}</span></div>
    {error ? <p role="alert">{error}</p> : null}
    <div className="cortex-viewer__layout"><section className="surface">
      <div className="cortex-viewer__toolbar"><strong>{query.entity ?? "modules"}</strong>
        {selected ? <span>{selected.module} · {selected.field}</span> : null}
        {selected ? <button onClick={() => choose({ entity: "links", moduleId: selected.id, row: query.row ?? 0 })}>Tensor-axis links</button> : null}
      </div>
      <div className="cortex-viewer__scroll" onScroll={(event) => setScroll(event.currentTarget.scrollTop)} role="list" aria-label="Virtualized cortical viewport" aria-busy={loading}>
        <div style={{ height: records.length * rowHeight, position: "relative" }}>
          {visible.map((value, index) => <button key={label(value)} role="listitem" className="cortex-viewer__row"
            style={{ position: "absolute", top: (first + index) * rowHeight, height: rowHeight, width: "100%" } as CSSProperties}
            onClick={() => open(value)}>{label(value)}{"module" in value ? <small>{value.rows.toLocaleString()} × {value.columns?.toLocaleString() ?? "unknown logical width"}</small> : null}</button>)}
        </div>
      </div>
      <div className="cortex-viewer__toolbar"><button disabled={!history.length || loading} onClick={() => {
        const next = [...history]; const cursor = next.pop(); setHistory(next); setQuery((value) => ({ ...value, cursor }));
      }}>Previous</button><span>{page ? `${page.offset.toLocaleString()}–${(page.offset + page.returned).toLocaleString()} of ${page.total.toLocaleString()}` : "Loading…"}</span>
      <button disabled={!page?.hasMore || loading} onClick={() => { setHistory((value) => [...value, query.cursor]); setQuery((value) => ({ ...value, cursor: page?.nextCursor ?? undefined })); }}>Next</button>
      <label>Go to {query.entity === "elements" ? "column" : "row/page offset"}<input type="number" min={0} aria-label="Exact cortical offset" onKeyDown={(event) => { if (event.key === "Enter") choose({ ...query, offset: Number(event.currentTarget.value), cursor: undefined }); }} /></label>
      </div>
      {query.entity === "elements" ? <div className="cortex-viewer__trits" aria-label="Exact selected ternary values">
        {(records as CortexElement[]).map((element) => <span key={element.column} data-trit={element.value} title={`input ${element.column} → output ${element.row}`}>
          <small>{element.column}</small>{element.value > 0 ? "+1" : element.value}</span>)}
      </div> : null}
      {query.entity === "boundaries" ? <p>{(records as CortexBoundary[]).filter((value) => value.kind.endsWith("boundary")).length} boundary markers on this page. Byte tokens address decoder embedding/output rows; that is not evidence those rows fired.</p> : null}
    </section><aside className="surface cortex-viewer__detail"><h2>{selected?.module ?? "Select a module"}</h2>
      {selected ? <dl><dt>Logical shape</dt><dd>{selected.logicalShape?.join(" × ") ?? "Not available"}</dd><dt>Storage layout</dt><dd>{selected.layout}</dd>
        <dt>Evidence</dt><dd>{selected.shapeEvidence}</dd><dt>Activity</dt><dd>Unobserved unless explicitly captured below</dd></dl> : null}
      <label><input type="checkbox" checked={capture} disabled={!selected} onChange={(event) => setCapture(event.target.checked)} /> Observe this viewport on actual forwards</label>
      {activity?.observation ? <><p>Observed {activity.observation.axis} {activity.observation.start}–{activity.observation.end}. {activity.observation.sample}. Not a full firing map.</p>
        <ol start={activity.observation.start}>{activity.observation.values.map((value, index) => <li key={index}>{value === null ? "Unobserved / nonfinite" : value.toPrecision(5)}</li>)}</ol></>
        : <p>{activity?.reason ?? "No actual forward observation for this viewport. We do not infer activation from confidence or weight sign."}</p>}
      <p>Idea pathways: memory_bridge → idea_adapter / decoder.memory_projection. These are declared computational dependencies, not proof of a remembered semantic answer.</p>
      <small>Selected codes are validated on read. Generation manifest is verified; the whole multi-GiB role payload is not rescanned per viewport.</small>
    </aside></div>
  </div>;
}
