import { useEffect, useRef, useState } from "react";
import type { ExternalAdviceApi, ExternalAdviceChannel, ExternalAdviceChannelConfiguration, ExternalAdviceCredentialStatus, ExternalAdvicePreview, ExternalAdviceProvider, ExternalAdviceRequest, ExternalAdviceSettings } from "../../shared/externalAdvice";
import { conciseUiMessage } from "./uiPresentation";
import { advicePanelActive, emptyAdvicePanel, mergeAdvicePanel } from "./externalAdvicePresentation";
import { ExternalAdviceModelPicker } from "./ExternalAdviceModelPicker";
import "./externalAdvice.css";

const providers: Array<{ id: ExternalAdviceProvider; label: string }> = [
  { id: "openai", label: "OpenAI" }, { id: "anthropic", label: "Anthropic" },
  { id: "gemini", label: "Gemini" }, { id: "openrouter", label: "OpenRouter" },
  { id: "ollama", label: "Ollama (local server)" },
  { id: "openai-compatible-local", label: "Local OpenAI-compatible server" }
];
const defaultConfig: ExternalAdviceChannelConfiguration = { enabled: false, provider: "openai", model: "", maxOutputTokens: 2048,
  context: { includeTask: true, includeActivity: true, maxCharacters: 16000 } };
const pending = (entry: ExternalAdviceRequest) => entry.state === "queued" || entry.state === "streaming";

/** Optional, separate advice. This is not an alternate OmniCortex chat backend. */
export function ExternalAdvicePanel({ brainId, api, hidden, onClose }: {
  brainId: string; api?: ExternalAdviceApi; hidden: boolean; onClose(): void;
}) {
  return <PanelOwner key={brainId} brainId={brainId} api={api} hidden={hidden} onClose={onClose} />;
}
function PanelOwner({ brainId, api, hidden, onClose }: {
  brainId: string; api?: ExternalAdviceApi; hidden: boolean; onClose(): void;
}) {
  const [settings, setSettings] = useState<ExternalAdviceSettings>();
  const [channel, setChannel] = useState<ExternalAdviceChannel>("assistant");
  const [draft, setDraft] = useState(defaultConfig);
  const [credentials, setCredentials] = useState<ExternalAdviceCredentialStatus[]>([]);
  const [key, setKey] = useState("");
  const [prompt, setPrompt] = useState("");
  const [preview, setPreview] = useState<ExternalAdvicePreview>();
  const [panel, setPanel] = useState(emptyAdvicePanel);
  const entries = panel.entries;
  const [next, setNext] = useState<string>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const alive = useRef(false);
  const operationPending = useRef(false);
  const append = (incoming: ExternalAdviceRequest[], activeIds: readonly string[] = [], persisted?: number) =>
    setPanel(previous => mergeAdvicePanel(previous, incoming.map(request => ({ request, active: activeIds.includes(request.id), persistedCharacters: persisted }))));
  useEffect(() => {
    alive.current = true;
    if (!api) return () => { alive.current = false; };
    let current = true;
    void Promise.all([api.settings(brainId), api.credentialStatus(), api.history(brainId)]).then(([value, keys, history]) => {
      if (!current) return; setSettings(value); setDraft({ ...value.assistant, enabled: value.assistant.enabled && value.deviceAuthority?.assistant === true }); setCredentials(keys);
      append(history.entries, history.activeRequestIds); setNext(history.nextAfter);
      if (history.retirementBlocked) setError(history.retirementBlocked.error);
    }).catch(reason => { if (current) setError(conciseUiMessage(reason, "Assistant settings could not load.")); });
    const unsubscribe = api.onEvent(event => { if (current && event.brainId === brainId) append([event.request], event.active ? [event.request.id] : [], event.persistence?.persistedCharacters); });
    return () => { current = false; alive.current = false; unsubscribe(); };
  }, [api, brainId]);
  const run = async (operation: () => Promise<void>) => {
    if (operationPending.current || !api) return; operationPending.current = true; setBusy(true); setError("");
    try { await operation(); } catch (reason) { if (alive.current) setError(conciseUiMessage(reason, "The advice request could not finish.")); }
    finally { operationPending.current = false; if (alive.current) setBusy(false); }
  };
  const selectChannel = (value: ExternalAdviceChannel) => {
    if (busy) return;
    setChannel(value); setDraft(settings ? { ...settings[value], enabled: settings[value].enabled && settings.deviceAuthority?.[value] === true } : defaultConfig); setKey(""); setPreview(undefined); setError("");
  };
  const update = (value: Partial<ExternalAdviceChannelConfiguration>) => { setDraft(previous => ({ ...previous, ...value })); setPreview(undefined); };
  const configured = credentials.find(item => item.provider === draft.provider);
  const active = advicePanelActive(panel, "assistant");
  return <aside hidden={hidden} className="external-advice-panel" aria-label="Separate assistant and action reviewer" id="external-advice-panel">
    <header><div><strong>Assistant &amp; reviewer</strong><small>Separate from {"OmniCortex"}</small></div>
      <button type="button" className="button button--ghost" aria-label="Close assistant panel" onClick={onClose}>Close</button></header>
    <div className="external-advice-panel__body">
      <div className="external-advice-panel__tabs" role="group" aria-label="Advice channel">
        <button type="button" className="button" aria-pressed={channel === "assistant"} onClick={() => selectChannel("assistant")}>Assistant</button>
        <button type="button" className="button" aria-pressed={channel === "reviewer"} onClick={() => selectChannel("reviewer")}>Action reviewer</button>
      </div>
      {!api ? <p role="status">The desktop advice service is unavailable.</p> : null}
      <details className="external-advice-panel__settings" open={!settings || !settings[channel].model}>
        <summary>{channel === "assistant" ? "Assistant" : "Reviewer"} settings</summary>
        <label className="external-advice-panel__check"><input type="checkbox" checked={draft.enabled} onChange={event => update({ enabled: event.target.checked })} />
          <span>Enable {channel === "assistant" ? "assistant" : "pre-action review"} on this device</span></label>
        <label>Provider<select value={draft.provider} onChange={event => { update({ provider: event.target.value as ExternalAdviceProvider, model: "", endpoint: undefined }); setKey(""); }}>
          {providers.map(provider => <option key={provider.id} value={provider.id}>{provider.label}</option>)}</select></label>
        <label>Model ID<input value={draft.model} autoComplete="off" spellCheck={false} onChange={event => update({ model: event.target.value })} placeholder={draft.provider === "ollama" ? "Choose an installed local model" : "Choose a model available to your account"} /></label>
        {draft.provider === "openai-compatible-local" || draft.provider === "ollama" ? <label>Local endpoint<input value={draft.endpoint ?? ""} onChange={event => update({ endpoint: event.target.value })} placeholder={draft.provider === "ollama" ? "http://127.0.0.1:11434/v1" : "http://127.0.0.1:8080/v1"} /></label> : null}
        {settings ? <ExternalAdviceModelPicker key={`${draft.provider}:${draft.endpoint ?? ""}`} brainId={brainId} channel={channel} settingsRevision={settings.revision} credentialRevision={configured?.revision ?? 0}
          configuration={settings[channel]} model={draft.model} api={api} onChange={model => update({ model })}
          canDiscover={settings[channel].provider === draft.provider && settings[channel].endpoint === draft.endpoint &&
            (draft.provider === "ollama" || configured?.configured === true)} /> : null}
        <label>Output-token budget<input type="number" min={1} max={65536} value={draft.maxOutputTokens} onChange={event => update({ maxOutputTokens: Number(event.target.value) })} /></label>
        <fieldset><legend>Shared context</legend>
          <label className="external-advice-panel__check"><input type="checkbox" checked={draft.context.includeTask} onChange={event => update({ context: { ...draft.context, includeTask: event.target.checked } })} /><span>Recent user task</span></label>
          <label className="external-advice-panel__check"><input type="checkbox" checked={draft.context.includeActivity} onChange={event => update({ context: { ...draft.context, includeActivity: event.target.checked } })} /><span>Recent recorded replies and action status</span></label>
          <label>Context characters<input type="number" min={0} value={draft.context.maxCharacters} onChange={event => update({ context: { ...draft.context, maxCharacters: Number(event.target.value) } })} /></label>
        </fieldset>
        <button type="button" className="button button--primary" disabled={busy || !api || !settings} onClick={() => void run(async () => {
          const result = await api!.configure(brainId, { expectedRevision: settings!.revision, channel, configuration: draft });
          if (alive.current) { setSettings(result); setDraft(result[channel]); setPreview(undefined); }
        })}>Save {channel === "assistant" ? "assistant" : "reviewer"} settings</button>
        {draft.provider === "ollama" ? <small>No API key is needed. Start your own trusted Ollama server and choose an installed local model. Omni does not start it or pull models; known cloud selections require separate consent. A loopback address alone cannot guarantee that a server never forwards data.</small> : <details><summary>Your provider key · {configured?.configured ? configured.persistence : draft.provider === "openai-compatible-local" ? "optional" : "not added"}</summary>
          <label>API key<input type="password" autoComplete="new-password" value={key} onChange={event => setKey(event.target.value)} /></label>
          <div className="external-advice-panel__row"><button type="button" className="button" disabled={busy || !api || !key} onClick={() => void run(async () => {
            const result = await api!.saveCredential({ provider: draft.provider, credential: key }); if (alive.current) { setCredentials(result); setKey(""); setPreview(undefined); }
          })}>Save key</button><button type="button" className="button button--ghost" disabled={busy || !api || !configured?.configured} onClick={() => void run(async () => {
            const result = await api!.removeCredential(draft.provider); if (alive.current) { setCredentials(result); setKey(""); setPreview(undefined); }
          })}>Remove key</button></div>
          <small>Keys stay on this device, outside the brain and its exports. Provider requests may incur charges.</small>
        </details>}
      </details>
      {channel === "reviewer" ? <p>The reviewer holds new tool actions for at least one second and until review finishes. Flagged or failed reviews ask you; existing tool permissions still apply. It cannot inspect hidden intentions or guarantee safety.</p> : <>
        <label>Ask the separate assistant<textarea rows={4} value={prompt} onChange={event => { setPrompt(event.target.value); setPreview(undefined); }} placeholder="Ask for help with the task or recorded activity…" /></label>
        <div className="external-advice-panel__row"><button type="button" className="button" disabled={busy || !api || !prompt.trim() || active} onClick={() => void run(async () => {
          const value = await api!.preview(brainId, { channel: "assistant", prompt }); if (alive.current) setPreview(value);
        })}>Preview shared context</button>
        <button type="button" className="button button--primary" disabled={busy || !api || !preview || active} onClick={() => void run(async () => {
          const value = await api!.start(brainId, { channel: "assistant", prompt, requestId: crypto.randomUUID(), expectedSettingsRevision: preview!.settingsRevision, previewSha256: preview!.sha256 });
          if (alive.current) { append([value], pending(value) ? [value.id] : []); setPreview(undefined); setPrompt(""); }
        })}>Send to assistant</button></div>
        {preview ? <details open><summary>Exact request · {preview.requestText.length.toLocaleString()} characters</summary><pre tabIndex={0}>{preview.requestText}</pre>
          <small>{preview.context.items.length} recorded items · {preview.context.omittedItems.toLocaleString()} omitted{preview.context.truncated ? " · partial history" : ""}. No automatic OmniCortex learning.</small></details> : null}
      </>}
      {error ? <p role="alert" className="external-advice-panel__error">{error}</p> : null}
      <section aria-label={`${channel === "assistant" ? "Assistant" : "Reviewer"} history`} className="external-advice-panel__history">
        {entries.filter(item => item.channel === channel).map(item => <article key={item.id} aria-label={`${item.provider} ${item.state}`}>
          <header><strong>{item.provider} · {item.model}</strong><span role={pending(item) && panel.live[item.id] ? "status" : undefined}>{pending(item) && !panel.live[item.id] ? "Saved prefix" : item.state}</span></header>
          <details><summary>{channel === "assistant" ? "Your request and shared context" : "Reviewed proposal and context"}</summary><pre tabIndex={0}>{item.requestText}</pre></details>
          <div className="external-advice-panel__response">{item.response || (pending(item) && panel.live[item.id] ? "Waiting for provider…" : "No text returned.")}</div>
          {(panel.persisted[item.id] ?? item.response.length) < item.response.length ? <small role="status">Latest text has not reached its saved checkpoint.</small> : null}
          {pending(item) && !panel.live[item.id] ? <small>No provider request is running for this saved prefix.</small> : null}
          {item.error ? <p role="alert">{item.error}</p> : null}
          {pending(item) && panel.live[item.id] ? <button type="button" className="button button--ghost" disabled={busy || !api} onClick={() => void run(async () => { await api!.cancel(brainId, item.id); })}>Stop advice request</button> : null}
        </article>)}
        {next ? <button type="button" className="button" disabled={busy || !api} onClick={() => void run(async () => {
          const value = await api!.history(brainId, { after: next }); if (alive.current) { append(value.entries, value.activeRequestIds); setNext(value.nextAfter); if (value.retirementBlocked) setError(value.retirementBlocked.error); }
        })}>More saved advice</button> : null}
      </section>
    </div>
  </aside>;
}
