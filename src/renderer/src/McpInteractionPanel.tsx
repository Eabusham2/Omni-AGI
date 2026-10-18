import { useEffect, useId, useRef, useState } from "react";
import type { McpFormProperty, McpFormValues, McpInteractionRecord, McpInteractionResponse, McpInteractionsApi } from "../../shared/mcpInteractions";
import type { NativeMcpSamplingInput, NativeMcpSamplingContent } from "../../shared/nativeMcpSampling";
import type { McpAppDataReview } from "../../shared/mcpAppTaskData";
import "./mcpInteractions.css";

function requiredMcpReviewApi() {
  const mcp = typeof window !== "undefined" ? window.omni?.mcp : undefined;
  if (!mcp || typeof mcp.interactions !== "function" || typeof mcp.respondInteraction !== "function" || typeof mcp.onInteractionEvent !== "function") {
    throw new Error("Connected-app review is unavailable in this preview.");
  }
  return mcp;
}
const hostApi: McpInteractionsApi = {
  list: id => requiredMcpReviewApi().interactions(id),
  respond: response => requiredMcpReviewApi().respondInteraction(response),
  onEvent: listener => requiredMcpReviewApi().onInteractionEvent(listener)
};

export function McpInteractionPanel({ brainId, api }: { brainId: string; api?: McpInteractionsApi }) {
  const available = api ?? (typeof window !== "undefined" && typeof window.omni?.mcp?.interactions === "function" &&
    typeof window.omni?.mcp?.respondInteraction === "function" && typeof window.omni?.mcp?.onInteractionEvent === "function" ? hostApi : undefined);
  if (!available) return null;
  return <McpInteractionPanelInstance key={brainId} brainId={brainId} api={available} />;
}
function McpInteractionPanelInstance({ brainId, api }: { brainId: string; api: McpInteractionsApi }) {
  const [records, setRecords] = useState<McpInteractionRecord[]>([]);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const resolved = useRef(new Set<string>());
  const heading = useId();
  useEffect(() => {
    let active = true;
    const receive = (values: McpInteractionRecord[]) => {
      if (!active) return;
      setRecords(current => {
        const map = new Map(current.map(record => [record.id, record]));
        for (const record of values) {
          if (record.origin.brainId !== brainId) continue;
          if (record.status !== "pending") { resolved.current.add(record.id); map.delete(record.id); }
          else if (!resolved.current.has(record.id)) map.set(record.id, record);
        }
        return [...map.values()].sort((a, b) => a.createdAt.localeCompare(b.createdAt));
      });
    };
    let unsubscribe: () => void = () => undefined;
    try { unsubscribe = api.onEvent(event => receive([event.record])); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    void Promise.resolve().then(() => api.list(brainId)).then(receive).catch(failure => { if (active) setError(failure instanceof Error ? failure.message : String(failure)); });
    return () => { active = false; unsubscribe(); };
  }, [brainId, api, reload]);
  if (!records.length && !error) return null;
  return <section className="mcp-requests" aria-labelledby={heading}>
    <header><h3 id={heading}>Connected app requests <span>{records.length}</span></h3><span role="status">Review before sharing</span></header>
    {error ? <div role="alert"><details><summary>Requests could not update</summary><p>{error}</p></details>
      <button type="button" className="button button--ghost" onClick={() => { setError(""); setReload(value => value + 1); }}>Retry</button></div> : null}
    <div className="mcp-requests__list">{records.map(record => <McpInteractionCard key={record.id} record={record} onRespond={async response => {
      const saved = await api.respond({ brainId, id: record.id, ...response });
      if (saved.status !== "pending") { resolved.current.add(saved.id); setRecords(current => current.filter(item => item.id !== saved.id)); }
    }} />)}</div>
  </section>;
}

type RawFormValues = Record<string, string | boolean | string[]>;
function initialValues(record: McpInteractionRecord): RawFormValues {
  if (record.kind !== "form") return {};
  return Object.fromEntries(Object.entries(record.request.requestedSchema.properties).map(([name, property]) => [name,
    property.default !== undefined ? typeof property.default === "number" ? String(property.default) : property.default
      : property.type === "boolean" ? false : property.type === "array" ? [] : ""]));
}
function formValues(record: Extract<McpInteractionRecord, { kind: "form" }>, raw: RawFormValues): McpFormValues {
  const values: McpFormValues = Object.create(null) as McpFormValues;
  for (const [name, property] of Object.entries(record.request.requestedSchema.properties)) {
    const value = raw[name];
    if (value === "" && !record.request.requestedSchema.required?.includes(name)) continue;
    if (property.type === "number" || property.type === "integer") {
      if (typeof value !== "string" || !value.trim() || !Number.isFinite(Number(value))) throw new Error(`Enter a number for ${property.title ?? name}.`);
      values[name] = Number(value);
    } else if (value !== undefined) values[name] = value;
  }
  return values;
}

function McpFormField({ name, property, required, value, onChange }: {
  name: string; property: McpFormProperty; required: boolean; value: RawFormValues[string] | undefined;
  onChange(value: RawFormValues[string]): void;
}) {
  const id = useId(); const label = property.title ?? name;
  const options = property.enum?.map(value => ({ value, title: value })) ?? property.oneOf?.map(item => ({ value: item.const, title: item.title ?? item.const }));
  const selections = property.items?.enum?.map(value => ({ value, title: value })) ?? property.items?.anyOf?.map(item => ({ value: item.const, title: item.title ?? item.const }));
  if (property.type === "array") return <fieldset className="mcp-request__choices"><legend>{label}{required ? " *" : ""}</legend>
    {property.description ? <p>{property.description}</p> : null}
    {selections?.map(option => <label key={option.value}><input type="checkbox" checked={Array.isArray(value) && value.includes(option.value)}
      onChange={event => { const previous = Array.isArray(value) ? value : []; onChange(event.currentTarget.checked ? [...previous, option.value] : previous.filter(item => item !== option.value)); }} />{option.title}</label>)}
    {property.minItems !== undefined || property.maxItems !== undefined ? <small>Selections: {property.minItems ?? 0} minimum{property.maxItems !== undefined ? `, ${property.maxItems} maximum` : ""}</small> : null}
  </fieldset>;
  if (property.type === "boolean") return <label className="mcp-request__boolean"><input type="checkbox" checked={value === true} onChange={event => onChange(event.currentTarget.checked)} />
    <span>{label}{required ? " *" : ""}{property.description ? <small>{property.description}</small> : null}</span></label>;
  return <label className="mcp-request__field" htmlFor={id}><span>{label}{required ? " *" : ""}</span>
    {property.description ? <small>{property.description}</small> : null}
    {options ? <select id={id} value={typeof value === "string" ? value : ""} required={required} onChange={event => onChange(event.currentTarget.value)}>
      <option value="">Choose…</option>{options.map(option => <option key={option.value} value={option.value}>{option.title}</option>)}
    </select> : <input id={id} type={property.type === "number" || property.type === "integer" ? "number" : property.format === "email" ? "email" : property.format === "date" ? "date" : "text"}
      value={typeof value === "string" ? value : ""} required={required} step={property.type === "integer" ? 1 : "any"}
      min={property.minimum} max={property.maximum} autoComplete="off" spellCheck={false}
      placeholder={property.format === "date-time" ? "2026-10-10T12:00:00Z" : undefined} onChange={event => onChange(event.currentTarget.value)} />}
  </label>;
}

function McpAppDataContents({ request }: { request: McpAppDataReview }) {
  const content = request.data.params.content, blocks = Array.isArray(content) ? content : content ? [content] : [];
  return <>
    <p className="mcp-request__notice">This content comes from the named App, not from you or a hidden system prompt. Approval saves it as pending task data; it does not claim the brain has already learned it.</p>
    <p>Source resource: <code>{request.view.resourceUri}</code></p>
    {blocks.map((block, index) => <section className="mcp-request__message" key={index} aria-label={`App data block ${index + 1}`}>
      {block.type === "text" ? <pre>{block.text}</pre> : block.type === "image" || block.type === "audio" ? <>
        <strong>{block.type === "image" ? "Image" : "Audio"} · {block.mimeType}</strong><p>{block.data.length.toLocaleString()} encoded characters, preserved unchanged.</p>
        <small>Untrusted media is not decoded or played in this review.</small>
      </> : block.type === "resource" ? <><strong>Embedded resource</strong><p>{block.resource.uri}</p>{"text" in block.resource ? <pre>{block.resource.text}</pre> : <small>Inert binary resource preserved; no automatic opening.</small>}</>
        : block.type === "resource_link" ? <><strong>Resource link</strong><p>{block.uri}</p><small>This link is data; review does not fetch it or grant filesystem access.</small></> : null}
    </section>)}
    {request.data.method === "ui/update-model-context" && request.data.params.structuredContent !== undefined ? <details open><summary>Structured App data</summary><pre>{JSON.stringify(request.data.params.structuredContent, null, 2)}</pre></details> : null}
    <details><summary>Exact source and content fingerprints</summary><p>Completed call: <code>{request.captureSha256}</code></p><p>Task data: <code>{request.dataSha256}</code></p></details>
  </>;
}

export function McpInteractionCard({ record, onRespond }: { record: McpInteractionRecord;
  onRespond(response: Omit<McpInteractionResponse, "brainId" | "id">): Promise<void> }) {
  const [values, setValues] = useState(() => initialValues(record));
  const [messages, setMessages] = useState<NativeMcpSamplingInput["messages"]>(() => record.kind === "sampling-input" ? structuredClone(record.request.messages) : []);
  const [maxTokens, setMaxTokens] = useState(() => record.kind === "sampling-input" ? String(record.request.maxTokens) : "");
  const [output, setOutput] = useState(() => record.kind === "sampling-output" ? record.request.content.text : "");
  const [busy, setBusy] = useState(false);
  const sending = useRef(false);
  const [error, setError] = useState("");
  const prefix = useId();
  const act = async (action: McpInteractionResponse["action"]) => {
    if (sending.current) return; sending.current = true; setBusy(true); setError("");
    try {
      await onRespond({ action, ...(action !== "accept" ? {} : record.kind === "form" ? { values: formValues(record, values) }
        : record.kind === "sampling-input" ? { sampling: { messages, maxTokens: Number(maxTokens),
          ...(record.request.temperature === undefined ? {} : { temperature: record.request.temperature }),
          ...(record.request.stopSequences === undefined ? {} : { stopSequences: record.request.stopSequences }) } }
          : record.kind === "sampling-output" ? { editedText: output } : {}) });
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { sending.current = false; setBusy(false); }
  };
  const title = record.kind === "form" || record.kind === "url" || record.kind === "app-permission" || record.kind === "app-data" ? record.request.message
    : record.kind === "sampling-input" ? "Allow this app to ask your brain?" : "Share this response with the app?";
  const target = record.kind === "url" ? new URL(record.request.url) : undefined;
  return <article className="mcp-request" data-kind={record.kind}>
    <form aria-labelledby={`${prefix}-title`} onSubmit={event => { event.preventDefault(); void act("accept"); }}>
      <header><span className="mcp-request__source">{record.origin.serverLabel}</span><h4 id={`${prefix}-title`}>{title}</h4></header>
      <details className="mcp-request__source-details"><summary>Connection</summary><p>{record.origin.endpoint}</p></details>
      <fieldset className="mcp-request__body" disabled={busy}>
        {record.kind === "form" ? <><p className="mcp-request__notice">Only the values you submit are shared with this server. Do not enter passwords, keys, or payment credentials.</p>
          {Object.entries(record.request.requestedSchema.properties).map(([name, property]) => <McpFormField key={name} name={name} property={property}
            required={record.request.requestedSchema.required?.includes(name) ?? false} value={values[name]}
            onChange={value => setValues(current => ({ ...current, [name]: value }))} />)}</> : null}
        {record.kind === "url" ? <><p>Open <strong>{target!.hostname}</strong> in your browser. The app cannot read what you enter there.</p>
          <pre className="mcp-request__url" aria-label="Full destination URL">{record.request.url}</pre>
          {target!.hostname.includes("xn--") ? <p className="mcp-request__notice">This address uses an internationalized domain. Examine the full hostname before opening.</p> : null}
          <p className="mcp-request__notice">Opening the link is not a claim that sign-in or the requested action is complete.</p></> : null}
        {record.kind === "app-permission" ? <>
          <p className="mcp-request__notice">This approval applies only to the reviewed connected-app request. It does not change the instance’s tool policy or grant a timeout approval.</p>
          {record.request.toolId ? <p>Tool: <strong>{record.request.toolId}</strong></p> : null}
          {record.request.arguments ? <details open><summary>Exact action arguments</summary><pre>{JSON.stringify(record.request.arguments, null, 2)}</pre></details> : null}
          {record.request.csp ? <><p>The app may contact only these exact declared origins while this view is open. No Studio cookies, authorization headers or local system permissions are granted.</p>
            <pre aria-label="Exact network origins">{JSON.stringify(record.request.csp, null, 2)}</pre></> : null}
          {record.request.url ? <><p>Browser destination: <strong>{new URL(record.request.url).hostname}</strong></p><pre className="mcp-request__url" aria-label="Full destination URL">{record.request.url}</pre></> : null}
        </> : null}
        {record.kind === "app-data" ? <McpAppDataContents request={record.request} /> : null}
        {record.kind === "sampling-input" ? <><p>This request uses <strong>this instance’s own OmniCortex</strong>. Review the exact task messages before allowing private inference.</p>
          {record.systemPromptOmitted !== undefined ? <details className="mcp-request__notice" open><summary>Server system prompt will not be used</summary>
            <p>This app requested separate instructions. They are shown below for review, but will not become hidden persona instructions. You may edit the task messages yourself.</p><pre>{record.systemPromptOmitted}</pre></details> : null}
          {messages.map((message, index) => <section className="mcp-request__message" key={index} aria-label={`Message ${index + 1}, ${message.role}`}>
            <h5>Message {index + 1} · {message.role}</h5>
            {(Array.isArray(message.content) ? message.content : [message.content]).map((block, blockIndex) => block.type === "text"
              ? <label className="mcp-request__field" key={blockIndex}><span>Task text {blockIndex + 1}</span><textarea rows={4} value={block.text}
                onChange={event => { const text = event.currentTarget.value; setMessages(current => current.map((item, position) => {
                  if (position !== index) return item;
                  const replace = (value: NativeMcpSamplingContent, i: number): NativeMcpSamplingContent => i === blockIndex ? { type: "text", text } : value;
                  return { ...item, content: Array.isArray(item.content) ? item.content.map(replace) : replace(item.content, 0) };
                })); }} /></label>
              : <div className="mcp-request__media" key={blockIndex}><strong>{block.type === "image" ? "Image" : "Audio"} · {block.mimeType}</strong>
                <p>{record.media?.find(item => item.messageIndex === index && item.blockIndex === blockIndex)?.decodedBytes.toLocaleString() ?? "Unmeasured"} source bytes</p>
                <details><summary>Exact media fingerprint</summary><code>{record.media?.find(item => item.messageIndex === index && item.blockIndex === blockIndex)?.encodedSha256 ?? "Not available"}</code></details>
                <small>The submitted media is preserved unchanged. It is not decoded or played in this review panel.</small></div>)}
          </section>)}
          <label className="mcp-request__field"><span>Maximum response tokens</span><input type="number" min="1" step="1" required value={maxTokens} onChange={event => setMaxTokens(event.currentTarget.value)} /></label>
          {record.request.temperature !== undefined || record.request.stopSequences !== undefined ? <details><summary>Requested sample controls</summary><pre>{JSON.stringify({
            ...(record.request.temperature === undefined ? {} : { temperature: record.request.temperature }),
            ...(record.request.stopSequences === undefined ? {} : { stopSequences: record.request.stopSequences }) }, null, 2)}</pre></details> : null}
          {record.modelPreferences ? <details><summary>Server model preferences</summary><pre>{JSON.stringify(record.modelPreferences, null, 2)}</pre><p>No external model will be substituted.</p></details> : null}</> : null}
        {record.kind === "sampling-output" ? <><p>Generated by {record.request.model}. You can edit or withhold the response before this server receives it.</p>
          <label className="mcp-request__field"><span>Response to share</span><textarea rows={8} value={output} onChange={event => setOutput(event.currentTarget.value)} /></label></> : null}
      </fieldset>
      {error ? <div role="alert" className="mcp-request__error"><details open><summary>Review could not be sent</summary><p>{error}</p></details></div> : null}
      <footer><button type="submit" className="button button--primary mcp-request__share" disabled={busy}>{busy ? "Sending…" : record.kind === "url" ? "Open reviewed URL" : record.kind === "sampling-input" ? "Allow private inference" : record.kind === "app-permission" ? "Allow reviewed request" : record.kind === "app-data" ? "Queue reviewed App data" : "Share reviewed response"}</button>
        <button type="button" className="button button--ghost" disabled={busy} onClick={() => void act("decline")}>Decline</button><button type="button" className="button button--ghost" disabled={busy} onClick={() => void act("cancel")}>Cancel request</button></footer>
    </form>
  </article>;
}
