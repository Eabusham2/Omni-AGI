import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ApiTeacherProvider, ApiTeacherProviderStatus, ApiTeacherTrainingReport, ApiTeacherTrainingRequest, OmniApi, RuntimeJob } from "../../shared/types";
import { ApiTeacherModelPicker } from "./ApiTeacherModelPicker";
import { Icon } from "./icons";
import { conciseUiMessage } from "./uiPresentation";
import type { ApiTeacherMediaKind } from "../../shared/apiTeacherMedia";

type TeacherApi = OmniApi["teacher"];
export const teacherProviderLabels: Record<ApiTeacherProvider, string> = {
  openai: "OpenAI", anthropic: "Claude", gemini: "Gemini", openrouter: "OpenRouter", ollama: "Ollama · local"
};
export function teacherNeedsKey(provider: ApiTeacherProvider): boolean { return provider !== "ollama"; }
export function teacherMediaOutputChoices(provider: ApiTeacherProvider): ApiTeacherMediaKind[] {
  return provider === "gemini" ? ["image", "audio", "video"] : provider === "openai" || provider === "openrouter" ? ["image"] : [];
}
export function teacherDurableReport(job: RuntimeJob | null): ApiTeacherTrainingReport | undefined {
  const value = job?.output as Partial<ApiTeacherTrainingReport> | undefined;
  if (!value || !["queued", "running", "paused", "review-required", "stopped", "complete", "failed", "dormant"].includes(value.durableState ?? "") ||
      typeof value.resumable !== "boolean" || !Array.isArray(value.uncertainItemIndexes) ||
      value.uncertainItemIndexes.some(index => !Number.isSafeInteger(index) || index < 0) ||
      !Number.isSafeInteger(value.learned) || value.learned! < 0 || !Number.isSafeInteger(value.requested) || value.requested! < 1) return undefined;
  return value as ApiTeacherTrainingReport;
}
export interface TeacherSubmit { signature: string; requestId: string }
/** An explicit retry after a lost IPC acknowledgement keeps the identical host request ID. */
export function teacherSubmitIdentity(previous: TeacherSubmit | undefined, input: Omit<ApiTeacherTrainingRequest, "requestId">,
  newId: () => string): TeacherSubmit {
  const signature = JSON.stringify(input);
  return previous?.signature === signature ? previous : { signature, requestId: newId() };
}
export function teacherCurrentJob(previous: RuntimeJob | null, incoming: RuntimeJob, brainId: string): RuntimeJob | null {
  if (incoming.brainId !== brainId) return previous;
  if (previous?.id === incoming.id) {
    if (incoming.updatedAt < previous.updatedAt) return previous;
    const terminal = (value: RuntimeJob) => ["complete", "failed", "cancelled"].includes(value.state);
    if (incoming.updatedAt === previous.updatedAt && (terminal(previous) && !terminal(incoming) || previous.state === "running" && incoming.state === "queued")) return previous;
  } else if (previous && incoming.createdAt < previous.createdAt) return previous;
  return incoming;
}

interface Props { brainId: string; api?: TeacherApi; onNotice(message: string): void; initialProvider?: ApiTeacherProvider }
export function ApiTeacherPanel(props: Props) { return <TeacherOwner key={props.brainId} {...props} />; }
function TeacherOwner({ brainId, api, onNotice, initialProvider = "openai" }: Props) {
  const [provider, setProvider] = useState<ApiTeacherProvider>(initialProvider);
  const [statuses, setStatuses] = useState<ApiTeacherProviderStatus[]>([]), [credentialEpoch, setCredentialEpoch] = useState(0);
  const [key, setKey] = useState(""), [model, setModel] = useState(""), [prompts, setPrompts] = useState("");
  const [mediaOutputs, setMediaOutputs] = useState<ApiTeacherMediaKind[]>([]);
  const [job, setJob] = useState<RuntimeJob | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [uncertainDetails, setUncertainDetails] = useState<string>("");
  const [receivedReview, setReceivedReview] = useState<{ jobId: string; indexes: number[] }>();
  const owner = useRef<object | null>(null), submit = useRef<TeacherSubmit | undefined>(undefined), notice = useRef(onNotice), busyRef = useRef(false);
  notice.current = onNotice;
  useLayoutEffect(() => { const scope = {}; owner.current = scope; return () => { if (owner.current === scope) owner.current = null; }; }, [api, brainId]);
  useEffect(() => {
    const scope = owner.current; if (!api || !scope) return;
    void Promise.all([api.status(), api.list(brainId)]).then(([next, jobs]) => {
      if (owner.current !== scope) return;
      setStatuses(next);
      const latest = jobs.find(value => value.state === "queued" || value.state === "running") ?? jobs[0];
      if (latest) setJob(previous => teacherCurrentJob(previous, latest, brainId));
    }).catch(reason => { if (owner.current === scope) setError(conciseUiMessage(reason, "Teacher settings could not load.")); });
    return api.onEvent(({ job: incoming }) => {
      if (owner.current === scope) setJob(previous => teacherCurrentJob(previous, incoming, brainId));
    });
  }, [api, brainId]);
  const required = teacherNeedsKey(provider), configured = !required || statuses.some(value => value.provider === provider && value.configured);
  const status = statuses.find(value => value.provider === provider), report = teacherDurableReport(job);
  const active = job?.state === "queued" || job?.state === "running" || job?.state === "cancelling";
  const operate = async (work: () => Promise<void>) => {
    const scope = owner.current; if (!api || !scope || busyRef.current) return;
    busyRef.current = true; setBusy(true); setError("");
    try { await work(); } catch (reason) { if (owner.current === scope) setError(conciseUiMessage(reason, "Teacher action could not finish.")); }
    finally { if (owner.current === scope) { busyRef.current = false; setBusy(false); } }
  };
  const saveKey = () => operate(async () => {
    if (!api || !required || !key.trim()) return;
    const scope = owner.current, next = await api.saveCredential({ provider, apiKey: key });
    if (owner.current !== scope) return;
    setStatuses(next); setKey(""); setCredentialEpoch(value => value + 1);
    notice.current(`${teacherProviderLabels[provider]} key stored in device storage, not this brain.`);
  });
  const removeKey = () => operate(async () => {
    if (!api || !required || !configured || !window.confirm(`Remove the saved ${teacherProviderLabels[provider]} teacher key? Learned neural state is not changed.`)) return;
    const scope = owner.current, next = await api.removeCredential(provider);
    if (owner.current !== scope) return;
    setStatuses(next); setKey(""); setCredentialEpoch(value => value + 1);
  });
  const start = () => operate(async () => {
    if (!api || active || !configured) return;
    const input: Omit<ApiTeacherTrainingRequest, "requestId"> = { brainId, provider, model: model.trim(), prompts: prompts.split(/\n+/).map(value => value.trim()).filter(Boolean),
      ...(mediaOutputs.length ? { responseModalities: ["text", ...mediaOutputs] } : {}) };
    const pending = teacherSubmitIdentity(submit.current, input, () => crypto.randomUUID()); submit.current = pending;
    const scope = owner.current, started = await api.start({ ...input, requestId: pending.requestId });
    if (owner.current !== scope) return;
    if (started.brainId !== brainId) throw new Error("Teacher result belongs to another instance.");
    setJob(previous => teacherCurrentJob(previous, started, brainId)); submit.current = undefined;
    notice.current("Teacher learning submitted. Progress reports committed responses and neural learning receipts.");
  });
  const control = (action: "pause" | "resume" | "stop") => operate(async () => {
    if (!api || !job) return; const scope = owner.current;
    const updated = await api[action]({ brainId, jobId: job.id });
    if (owner.current === scope) setJob(previous => teacherCurrentJob(previous, updated, brainId));
  });
  const resolve = (itemIndex: number, decision: "retry" | "skip") => operate(async () => {
    if (!api || !job) return;
    if (decision === "retry" && !window.confirm(receivedReview?.jobId === job.id && receivedReview.indexes.includes(itemIndex)
      ? "Recheck the already received media response? This does not contact or bill the provider. It will remain paused until you press Resume."
      : "The previous provider request or neural update may already have completed. Another provider attempt can be billed; retrying a lost learning acknowledgement can repeat neural learning. Allow another attempt for this exact question? It will remain paused until you press Resume.")) return;
    const scope = owner.current, updated = await api.resolveUncertain({ brainId, jobId: job.id, itemIndex, decision });
    if (owner.current === scope) { setJob(previous => teacherCurrentJob(previous, updated, brainId)); setUncertainDetails(""); }
  });
  const inspect = () => operate(async () => {
    if (!api || !job) return; const scope = owner.current, details = await api.details({ brainId, jobId: job.id });
    if (owner.current !== scope) return;
    if (details.brainId !== brainId || details.jobId !== job.id) throw new Error("Teacher history belongs to another job.");
    setReceivedReview({ jobId: details.jobId, indexes: details.items.filter(item => item.state === "response-review").map(item => item.index) });
    setUncertainDetails(details.items.filter(item => item.state === "uncertain-review" || item.state === "response-review" || item.state === "learning-uncertain-review" || item.state === "learning").map(item =>
      `Question ${item.index + 1} · ${item.state === "response-review" ? "received media response review (retry does not contact provider)" : item.state === "learning-uncertain-review" || item.state === "learning" ? "unproven neural learning acknowledgement" : "provider request"}\n${item.prompt}\n${item.error ?? "Outcome is uncertain."}`).join("\n\n"));
  });
  return <div className="integration-card api-teacher-panel">
    <div className="integration-card__head"><span><Icon name="brain" size={18} /></span><div><strong>Teacher learning</strong><small>Optional API or local teacher · OmniCortex remains your brain</small></div></div>
    <div className="integration-fields integration-fields--three">
      <label><span>Provider</span><select aria-label="Teacher provider" value={provider} disabled={busy} onChange={event => {
        setProvider(event.target.value as ApiTeacherProvider); setModel(""); setKey(""); setMediaOutputs([]); setError(""); submit.current = undefined;
      }}>{Object.entries(teacherProviderLabels).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></label>
      <ApiTeacherModelPicker provider={provider} model={model} onChange={setModel} configured={configured} credentialEpoch={credentialEpoch} api={api} />
      {required ? <label><span>API key</span><input type="password" autoComplete="off" disabled={busy} value={key} onChange={event => setKey(event.target.value)} placeholder={status?.keyHint ?? "Paste your key"} /></label>
        : <div><strong>Local connection</strong><small>127.0.0.1:11434 · no API key. Choose an installed model; no model is pulled or started for you.</small></div>}
    </div>
    {required ? <div className="integration-actions">
      <button type="button" className="button button--secondary" disabled={!api || busy || !key.trim()} onClick={() => void saveKey()}>Store key</button>
      <button type="button" className="button button--danger" disabled={!api || busy || !configured} onClick={() => void removeKey()}>Remove key</button>
      <small>{configured ? `Key configured · ${status?.persistence ?? "device storage"}` : "Add your provider key"}</small>
    </div> : null}
    <label><span>Learning questions · one per line</span><textarea rows={5} value={prompts} onChange={event => setPrompts(event.target.value)} placeholder={"Explain a concept with evidence…\nShow a correct tool trajectory for…"} /></label>
    {teacherMediaOutputChoices(provider).length ? <fieldset className="api-teacher-panel__media"><legend>Include generated media in these lessons</legend>
      {teacherMediaOutputChoices(provider).map(kind => <label key={kind}><input type="checkbox" checked={mediaOutputs.includes(kind)}
        onChange={event => setMediaOutputs(previous => event.target.checked ? [...previous, kind] : previous.filter(value => value !== kind))} />{kind === "image" ? "Images" : kind === "audio" ? "Audio" : "Video"}</label>)}
      <small>Choose a provider model that supports the selected outputs. Actual media bytes are learned by this mind; descriptions alone do not count.</small>
    </fieldset> : null}
    <div className="integration-actions">
      <button type="button" className="button button--primary" disabled={!api || busy || active || !configured || !model.trim() || !prompts.trim()} onClick={() => void start()}>{submit.current ? "Retry same submission" : "Learn from teacher"}</button>
      {active ? <><button type="button" className="button button--secondary" disabled={busy} onClick={() => void control("pause")}>Pause learning</button><button type="button" className="button button--danger" disabled={busy} onClick={() => void control("stop")}>Stop learning</button></> : null}
      {report?.resumable ? <button type="button" className="button button--secondary" disabled={busy} onClick={() => void control("resume")}>Resume learning</button> : null}
    </div>
    {job ? <div role="status"><strong>{job.label}</strong><small>{Math.round(job.progress * 100)}%{report ? ` · ${report.learned.toLocaleString()} of ${report.requested.toLocaleString()} learned` : ""}</small>{job.error ? <small>{conciseUiMessage(job.error)}</small> : null}</div> : null}
    {report?.durableState === "review-required" ? <div className="api-teacher-panel__review">
      <p>A provider request or neural update may already have completed. It was not repeated automatically. Inspect the uncertain question before deciding.</p>
      <button type="button" className="button button--ghost" disabled={busy} onClick={() => void inspect()}>Inspect uncertain questions</button>
      {uncertainDetails ? <pre>{uncertainDetails}</pre> : null}
      {report.uncertainItemIndexes.map(index => <div key={index} className="integration-actions"><span>Question {index + 1}</span>
        <button type="button" className="button button--secondary" disabled={busy || !job} onClick={() => void resolve(index, "skip")}>Skip question</button>
        <button type="button" className="button button--secondary" disabled={busy || !job} onClick={() => void resolve(index, "retry")}>{receivedReview?.jobId === job?.id && receivedReview?.indexes.includes(index) ? "Recheck received response" : "Allow another attempt"}</button>
      </div>)}
    </div> : null}
    {error ? <small role="alert">{error}</small> : null}
  </div>;
}
