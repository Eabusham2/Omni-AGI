import { useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { ExternalAdviceApi, ExternalAdviceChannel, ExternalAdviceChannelConfiguration, ExternalAdviceModelDiscoveryResult } from "../../shared/externalAdvice";
import { ApiTeacherExcludedModels, modelChoicePage, modelDiscoveryExclusions, modelDiscoveryIds, modelDiscoveryRequestIsCurrent } from "./ApiTeacherModelPicker";
import { conciseUiMessage } from "./uiPresentation";

const endpoints = { openai: "https://api.openai.com/v1/models", anthropic: "https://api.anthropic.com/v1/models",
  gemini: "https://generativelanguage.googleapis.com/v1beta/models", openrouter: "https://openrouter.ai/api/v1/models/user" };
export function adviceDiscoveryIds(result: ExternalAdviceModelDiscoveryResult, expected: {
  brainId: string; channel: ExternalAdviceChannel; requestId: string; settingsRevision: number; credentialRevision: number;
  configuration: ExternalAdviceChannelConfiguration;
}): string[] {
  const { configuration } = expected;
  if (configuration.provider === "openai-compatible-local") throw new Error("Generic local catalog discovery is unavailable; keep a manual model ID.");
  if (result.brainId !== expected.brainId || result.channel !== expected.channel || result.settingsRevision !== expected.settingsRevision ||
      result.credentialRevision !== expected.credentialRevision) throw new Error("Advice catalog belongs to a stale owner, channel, settings or credential revision.");
  let endpoint: string | undefined;
  if (configuration.provider === "ollama") {
    const url = new URL(configuration.endpoint ?? "http://127.0.0.1:11434/v1");
    if (!["127.0.0.1", "localhost", "[::1]", "::1"].includes(url.hostname) || !["http:", "https:"].includes(url.protocol) ||
        url.username || url.password || url.search || url.hash || url.pathname.replace(/\/+$/, "") !== "/v1") throw new Error("Saved Ollama discovery endpoint is invalid.");
    endpoint = url.toString().replace(/\/$/, "");
    if (result.endpoint !== endpoint || result.sourceEndpoint !== `${url.origin}/api/tags`) throw new Error("Advice catalog came from another local endpoint.");
  } else if (result.endpoint !== undefined || result.sourceEndpoint !== endpoints[configuration.provider]) throw new Error("Advice catalog came from another provider endpoint.");
  return modelDiscoveryIds(result, configuration.provider, expected.requestId);
}

interface Props {
  brainId: string; channel: ExternalAdviceChannel; settingsRevision: number; credentialRevision: number;
  configuration: ExternalAdviceChannelConfiguration; model: string; canDiscover: boolean;
  api?: Pick<ExternalAdviceApi, "discoverModels" | "cancelModelDiscovery">;
  onChange(model: string): void;
}
/** Explicit click only: imported/saved settings never trigger a provider request on mount. */
export function ExternalAdviceModelPicker(props: Props) {
  return <AdvicePicker key={`${props.brainId}:${props.channel}:${props.settingsRevision}:${props.credentialRevision}:${props.configuration.provider}:${props.configuration.endpoint ?? ""}`} {...props} />;
}
function AdvicePicker({ brainId, channel, settingsRevision, credentialRevision, configuration, model, canDiscover, api, onChange }: Props) {
  const filterId = useId(), scope = useRef<object | null>(null), pending = useRef<string | undefined>(undefined);
  const [ids, setIds] = useState<string[]>([]), [excluded, setExcluded] = useState<Array<{ id: string; reason: string }>>([]);
  const [filter, setFilter] = useState(""), [page, setPage] = useState(0), [busy, setBusy] = useState(false), [stopping, setStopping] = useState(false);
  const [error, setError] = useState(""), [notice, setNotice] = useState("");
  const choices = useMemo(() => modelChoicePage(ids, filter, page), [ids, filter, page]);
  useLayoutEffect(() => {
    const owner = {}; scope.current = owner;
    return () => {
      if (scope.current === owner) scope.current = null;
      const requestId = pending.current; pending.current = undefined;
      if (api && requestId) void api.cancelModelDiscovery(brainId, { channel, requestId }).catch(() => undefined);
    };
  }, [api, brainId, channel]);
  const current = (owner: object | null, requestId: string) => modelDiscoveryRequestIsCurrent(scope.current, owner!, pending.current, requestId);
  const refresh = async () => {
    const owner = scope.current; if (!api || !canDiscover || !owner || pending.current) return;
    const requestId = crypto.randomUUID(); pending.current = requestId; setBusy(true); setError(""); setNotice("");
    try {
      const result = await api.discoverModels(brainId, { channel, requestId, expectedSettingsRevision: settingsRevision, expectedCredentialRevision: credentialRevision });
      if (!current(owner, requestId)) return;
      const models = adviceDiscoveryIds(result, { brainId, channel, requestId, settingsRevision, credentialRevision, configuration });
      setIds(models); setExcluded(modelDiscoveryExclusions(result)); setPage(0);
      setNotice(`${models.length.toLocaleString()} provider-listed models. No model was selected; generation compatibility is not verified.`);
    } catch (reason) { if (current(owner, requestId)) setError(conciseUiMessage(reason, "Model discovery ended; manual model is unchanged.")); }
    finally { if (current(owner, requestId)) { pending.current = undefined; setBusy(false); setStopping(false); } }
  };
  const stop = async () => {
    const owner = scope.current, requestId = pending.current; if (!api || !owner || !requestId || stopping) return;
    setStopping(true);
    try { await api.cancelModelDiscovery(brainId, { channel, requestId }); }
    catch (reason) { if (current(owner, requestId)) setError(conciseUiMessage(reason, "Discovery Stop was not acknowledged.")); }
    finally { if (current(owner, requestId)) setStopping(false); }
  };
  if (configuration.provider === "openai-compatible-local") return <small>Enter a manual model ID for this generic local server.</small>;
  return <div className="api-model-picker">
    <button type="button" className="button button--ghost" disabled={stopping || !api || !busy && !canDiscover} onClick={() => void (busy ? stop() : refresh())}>
      {stopping ? "Awaiting model discovery Stop…" : busy ? "Stop model discovery" : "Refresh advice model list"}
    </button>
    {!canDiscover ? <small>Save the provider/endpoint settings and add its key if required. Manual model remains available.</small> : null}
    {ids.length ? <details><summary>Choose a provider-listed model</summary>
      <label htmlFor={filterId}>Find a model</label><input id={filterId} value={filter} onChange={event => { setFilter(event.target.value); setPage(0); }} />
      <label>Listed models<select value={choices.ids.includes(model) ? model : ""} onChange={event => { if (event.target.value) onChange(event.target.value); }}>
        <option value="">Choose explicitly</option>{choices.ids.map(id => <option key={id} value={id}>{id}</option>)}
      </select></label>
      {choices.start > 0 ? <button type="button" className="button button--ghost" onClick={() => setPage(Math.max(0, page - 1))}>Previous advice models</button> : null}
      {choices.start + 100 < choices.total ? <button type="button" className="button button--ghost" onClick={() => setPage(page + 1)}>More advice models</button> : null}
    </details> : null}
    <ApiTeacherExcludedModels rows={excluded} />
    {notice ? <small role="status">{notice}</small> : null}{error ? <small role="alert">{error}</small> : null}
  </div>;
}
