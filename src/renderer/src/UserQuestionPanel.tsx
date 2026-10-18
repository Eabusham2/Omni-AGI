import React, { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  DEFAULT_USER_QUESTION_PREFERENCES,
  type UserQuestionPreferences,
  type UserQuestionRecord,
  type UserQuestionResponse,
  type UserQuestionsApi
} from "../../shared/userQuestions";
import {
  mergeUserQuestionRecords,
  userQuestionAnswerText,
  userQuestionCountdown,
  userQuestionDraftError,
  userQuestionDraftResponse,
  userQuestionInPanel,
  userQuestionResolutionLabel,
  userQuestionTimeoutSeconds,
  type UserQuestionDraft
} from "./userQuestionPresentation";
import "./userQuestions.css";

export interface UserQuestionPanelProps {
  brainId: string;
  api: UserQuestionsApi;
  /** New durable resolutions only; inspect source before describing an assumption as a user reply. */
  onResolved?: (question: UserQuestionRecord) => void;
  /** Optional task-local view; the durable question service remains shared. */
  questionIds?: string[];
  /** Conversation reuses the cards without duplicating Learn/Fact Check trays. */
  presentation?: "task" | "conversation";
}

const apiOwners = new WeakMap<UserQuestionsApi, number>();
let nextApiOwner = 0;
function apiOwner(api: UserQuestionsApi): number {
  let owner = apiOwners.get(api);
  if (owner === undefined) { owner = ++nextApiOwner; apiOwners.set(api, owner); }
  return owner;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function UserQuestionPanel(props: UserQuestionPanelProps) {
  // A replaced API is a new live owner too. Unmounting retires its subscriptions
  // and late answers without resetting the panel on ordinary chat output.
  return <UserQuestionPanelInstance key={JSON.stringify([props.brainId, apiOwner(props.api), props.presentation ?? "task"])} {...props} />;
}

function UserQuestionPanelInstance({ brainId, api, onResolved, questionIds, presentation = "task" }: UserQuestionPanelProps) {
  const [records, setRecords] = useState<UserQuestionRecord[]>([]);
  const [preferences, setPreferences] = useState<UserQuestionPreferences>({ ...DEFAULT_USER_QUESTION_PREFERENCES });
  const [preferencesLoaded, setPreferencesLoaded] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const callback = useRef(onResolved);
  const visibleScope = useRef({ brainId, questionIds, presentation });
  visibleScope.current = { brainId, questionIds, presentation };
  const notified = useRef(new Set<string>());
  const alive = useRef(true);
  const headingId = useId();

  useEffect(() => { callback.current = onResolved; }, [onResolved]);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);

  const notifyResolution = useCallback((record: UserQuestionRecord) => {
    const scope = visibleScope.current;
    if (!userQuestionInPanel(record, scope.brainId, scope.questionIds, scope.presentation)) return;
    if (!record.resolution || record.suspended || notified.current.has(record.resolution.id)) return;
    notified.current.add(record.resolution.id);
    callback.current?.(record);
  }, []);

  useEffect(() => {
    let active = true;
    let scanning = true;
    const duringScan = new Map<string, UserQuestionRecord>();
    let policyDuringScan: UserQuestionPreferences | undefined;
    setLoading(true);
    setError("");
    let unsubscribe = () => {};
    try {
      // Subscribe before reading disk so an answer committed during the scan is not lost.
      unsubscribe = api.onEvent((event) => {
        if (!active || event.brainId !== brainId) return;
        if ("question" in event) {
          if (event.question.brainId !== brainId) { setError("A question update belongs to another instance."); return; }
          if (scanning) duringScan.set(event.question.id, event.question);
          setRecords((current) => mergeUserQuestionRecords(current, [event.question]));
          if (event.type === "resolved" || event.type === "cancelled") notifyResolution(event.question);
        } else if (event.type === "preferences") {
          if (scanning) policyDuringScan = event.preferences;
          setPreferences(event.preferences);
          setPreferencesLoaded(true);
        } else setError(event.message);
      });
    } catch (failure) { setError(errorMessage(failure)); }
    void Promise.allSettled([api.list(brainId), api.preferences(brainId)]).then(([saved, policy]) => {
      if (!active) return;
      if (saved.status === "fulfilled") {
        if (saved.value.some(record => record.brainId !== brainId)) setError("Saved questions belong to another instance.");
        else setRecords(mergeUserQuestionRecords(saved.value, [...duringScan.values()]));
      }
      if (policy.status === "fulfilled") {
        setPreferences(policyDuringScan ?? policy.value);
        setPreferencesLoaded(true);
      }
      const failures = [saved, policy].filter((result) => result.status === "rejected");
      if (failures.length) setError(failures.map((result) => errorMessage(result.reason)).join("\n"));
      scanning = false;
      duringScan.clear();
      setLoading(false);
    });
    return () => { active = false; unsubscribe(); };
  }, [api, brainId, notifyResolution, reload]);

  const visible = records.filter(record => userQuestionInPanel(record, brainId, questionIds, presentation));
  const pending = visible.filter((record) => record.status === "pending");
  const history = visible.filter((record) => record.status !== "pending");
  const activePendingIds = pending.filter(record => !record.suspended).map(record => record.id).join("\0");
  useEffect(() => { if (presentation === "conversation" && (activePendingIds || error)) setExpanded(true); }, [activePendingIds, error, presentation]);
  const hasDeadline = pending.some((record) => !record.suspended && record.timeout.deadlineAt);
  useEffect(() => {
    if (!hasDeadline) return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [hasDeadline]);

  const receive = (saved: UserQuestionRecord[]) => {
    if (!alive.current) return;
    if (saved.some(record => record.brainId !== brainId)) throw new Error("The question response belongs to another instance.");
    setRecords((current) => mergeUserQuestionRecords(current, saved));
  };
  const refreshAfterFailure = () => { if (alive.current) setReload((current) => current + 1); };
  const respond = async (record: UserQuestionRecord, response: UserQuestionResponse) => {
    try {
      const answered = await api.respond({ brainId, questionId: record.id, response });
      if (!alive.current) return;
      if (answered.id !== record.id) throw new Error("The answer receipt belongs to another question.");
      receive([answered]);
      if (alive.current) notifyResolution(answered);
    } catch (failure) { refreshAfterFailure(); throw failure; }
  };
  const cancel = async (record: UserQuestionRecord) => {
    try {
      const skipped = await api.cancel(brainId, record.id);
      if (!alive.current) return;
      if (skipped.id !== record.id) throw new Error("The skipped receipt belongs to another question.");
      receive([skipped]);
      if (alive.current) notifyResolution(skipped);
    } catch (failure) { refreshAfterFailure(); throw failure; }
  };
  const resume = async (record: UserQuestionRecord) => {
    try { receive(await api.resume(brainId, record.id)); }
    catch (failure) { refreshAfterFailure(); throw failure; }
  };

  // Keep the subscription alive, but add no permanent empty row to chat.
  // A real question/history event or a service error reveals the existing UI.
  if (presentation === "conversation" && !error && visible.length === 0) return null;
  const stateLabel = error ? "Needs attention" : loading ? "Loading…" : pending.some(record => !record.suspended) ? "Waiting for you" : pending.length ? "Paused copy" : "No open questions";
  const contents = <>
    {error ? <div className="user-questions__error" role="alert">
      <details><summary>Questions could not update</summary><p>{error}</p></details>
      <button type="button" onClick={() => setReload((current) => current + 1)}>Retry</button>
    </div> : null}
    {pending.length ? <div className="user-questions__pending">
      {pending.map((record) => <UserQuestionCard key={record.id} question={record} now={now}
        onRespond={(response) => respond(record, response)} onSkip={() => cancel(record)} onResume={() => resume(record)} />)}
    </div> : null}
    {history.length ? <details className="user-questions__history" onToggle={(event) => setHistoryOpen(event.currentTarget.open)}>
      <summary>Past questions <span>{history.length}</span></summary>
      {historyOpen ? <div className="user-questions__history-list">{[...history].reverse().map((record) => <UserQuestionHistory key={record.id} question={record} />)}</div> : null}
    </details> : null}
    <details className="user-questions__settings">
      <summary>Question settings</summary>
      {preferencesLoaded ? <UserQuestionSettings preferences={preferences} onSave={async (value) => {
        const saved = await api.setPreferences({ brainId, ...value });
        if (alive.current) setPreferences(saved);
        return saved;
      }} /> : <p className="user-questions__hint">{loading ? "Loading settings…" : "Retry to load saved settings."}</p>}
    </details>
  </>;
  return <section className={`user-questions${presentation === "conversation" ? " user-questions--conversation" : ""}`} aria-labelledby={headingId} aria-busy={loading || undefined}>
    {presentation === "conversation" ? <details open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
      <summary className="user-questions__header">
        <span id={headingId} className="user-questions__heading">Questions{pending.length ? <span className="user-questions__count">{pending.length}</span> : null}</span>
        <span className="user-questions__state" role="status">{stateLabel}</span>
      </summary>
      <div className="user-questions__content">{contents}</div>
    </details> : <>
      <header className="user-questions__header">
        <h3 id={headingId}>Questions{pending.length ? <span className="user-questions__count">{pending.length}</span> : null}</h3>
        <span className="user-questions__state" role="status">{stateLabel}</span>
      </header>
      {contents}
    </>}
  </section>;
}

export function UserQuestionCard({ question, now, onRespond, onSkip, onResume }: {
  question: UserQuestionRecord;
  now: number;
  onRespond: (response: UserQuestionResponse) => Promise<void>;
  onSkip: () => Promise<void>;
  onResume: () => Promise<void>;
}) {
  const [draft, setDraft] = useState<UserQuestionDraft>({ optionIds: [], text: "", other: false });
  const [busy, setBusy] = useState<"answer" | "skip" | "resume" | undefined>();
  const [error, setError] = useState("");
  const prefix = useId();
  const promptId = `${prefix}-prompt`, hintId = `${prefix}-hint`, errorId = `${prefix}-error`;
  const validation = userQuestionDraftError(question, draft);
  const countdown = userQuestionCountdown(question, now);
  const disabled = Boolean(busy || question.suspended);
  const select = (optionId: string, checked: boolean) => {
    setError("");
    setDraft((current) => ({ ...current, other: false, optionIds: question.mode === "single" ? [optionId]
      : checked ? [...current.optionIds, optionId] : current.optionIds.filter((id) => id !== optionId) }));
  };
  const act = async (kind: "answer" | "skip" | "resume", operation: () => Promise<void>) => {
    if (busy) return;
    setError("");
    setBusy(kind);
    try { await operation(); } catch (failure) { setError(errorMessage(failure)); }
    finally { setBusy(undefined); }
  };
  return <article className="user-question" data-question-status={question.status}>
    <form aria-labelledby={promptId} onSubmit={(event) => {
      event.preventDefault();
      if (validation) { setError(validation); return; }
      void act("answer", () => onRespond(userQuestionDraftResponse(question, draft)));
    }}>
      <fieldset disabled={disabled} aria-describedby={`${hintId}${error ? ` ${errorId}` : ""}`}>
        <legend id={promptId}>{question.prompt}</legend>
        {question.detail ? <details className="user-question__detail"><summary>Details</summary><div>{question.detail}</div></details> : null}
        <p id={hintId} className="user-questions__hint">{question.mode === "text" ? "Write your answer." : question.mode === "single" ? "Choose one option." : question.minSelections === question.maxSelections
          ? `Choose ${question.minSelections} options.` : `Choose ${question.minSelections}–${question.maxSelections} options.`}</p>
        {question.mode !== "text" ? <div className="user-question__options">
          {question.options.map((option) => <label key={option.id} className={`user-question__option${option.recommended ? " is-recommended" : ""}${draft.optionIds.includes(option.id) ? " is-selected" : ""}`}>
            <input type={question.mode === "single" ? "radio" : "checkbox"} name={`${prefix}-choice`} value={option.id}
              checked={!draft.other && draft.optionIds.includes(option.id)} onChange={(event) => select(option.id, event.currentTarget.checked)} />
            <span><span className="user-question__option-title">{option.label}{option.recommended ? <span className="user-question__recommended">Recommended</span> : null}</span>
              {option.description ? <span className="user-question__description">{option.description}</span> : null}</span>
          </label>)}
          {question.allowOther ? <label className={`user-question__option${draft.other ? " is-selected" : ""}`}>
            <input type={question.mode === "single" ? "radio" : "checkbox"} name={`${prefix}-choice`} checked={draft.other}
              onChange={(event) => { const other = event.currentTarget.checked; setError(""); setDraft((current) => ({ ...current, other, optionIds: [] })); }} />
            <span>Other answer</span>
          </label> : null}
        </div> : null}
        {question.mode === "text" || draft.other ? <label className="user-question__written">
          <span>{draft.other ? "Your alternative" : "Your answer"}</span>
          <textarea rows={3} value={draft.text} aria-invalid={Boolean(error && !draft.text.trim()) || undefined} aria-describedby={error ? errorId : undefined}
            onChange={(event) => { const value = event.currentTarget.value; setError(""); setDraft((current) => ({ ...current, text: value })); }} />
        </label> : null}
      </fieldset>
      {error ? <p id={errorId} className="user-question__error" role="alert">{error}</p> : null}
      <div className="user-question__actions">
        {question.suspended ? <>
          <span className="user-questions__hint">Paused copy</span>
          <button type="button" disabled={Boolean(busy)} onClick={() => void act("resume", onResume)}>{busy === "resume" ? "Resuming…" : "Resume question"}</button>
        </> : <>
          <button type="submit" className="user-question__send" disabled={Boolean(busy || validation)}>{busy === "answer" ? "Saving…" : "Send answer"}</button>
          <button type="button" disabled={Boolean(busy)} onClick={() => void act("skip", onSkip)}>{busy === "skip" ? "Skipping…" : "Skip"}</button>
          {countdown ? <time className="user-question__countdown" dateTime={question.timeout.deadlineAt} title={new Date(question.timeout.deadlineAt!).toLocaleString()}>{countdown}</time> : null}
        </>}
      </div>
    </form>
  </article>;
}

export function UserQuestionHistory({ question }: { question: UserQuestionRecord }) {
  return <details className="user-question-history" data-question-status={question.status}>
    <summary><span className="user-question-history__label">{userQuestionResolutionLabel(question)}</span><span className="user-question-history__prompt">{question.prompt}</span></summary>
    <div className="user-question-history__body">
      <p className="user-question-history__full-prompt">{question.prompt}</p>
      <p>{userQuestionAnswerText(question)}</p>
      {question.resolution ? <time dateTime={question.resolution.resolvedAt}>{new Date(question.resolution.resolvedAt).toLocaleString()}</time> : null}
      {question.detail ? <details className="user-question__detail"><summary>Question details</summary><div>{question.detail}</div></details> : null}
    </div>
  </details>;
}

export function UserQuestionSettings({ preferences, onSave }: {
  preferences: UserQuestionPreferences;
  onSave: (preferences: UserQuestionPreferences) => Promise<UserQuestionPreferences>;
}) {
  const [enabled, setEnabled] = useState(preferences.timeoutEnabled);
  const [minutes, setMinutes] = useState(String(Math.floor(preferences.timeoutSeconds / 60)));
  const [seconds, setSeconds] = useState(String(preferences.timeoutSeconds % 60));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  const editing = useRef(false);
  const prefix = useId();
  useEffect(() => {
    if (editing.current) return;
    setEnabled(preferences.timeoutEnabled);
    setMinutes(String(Math.floor(preferences.timeoutSeconds / 60)));
    setSeconds(String(preferences.timeoutSeconds % 60));
  }, [preferences.timeoutEnabled, preferences.timeoutSeconds]);
  const changed = () => { editing.current = true; setSaved(false); setError(""); };
  return <form className="user-question-settings" onSubmit={(event) => {
    event.preventDefault();
    if (busy) return;
    let value: UserQuestionPreferences;
    try { value = { timeoutEnabled: enabled, timeoutSeconds: userQuestionTimeoutSeconds(minutes, seconds) }; }
    catch (failure) { setError(errorMessage(failure)); return; }
    setBusy(true); setError(""); setSaved(false);
    void onSave(value).then((result) => {
      editing.current = false;
      setEnabled(result.timeoutEnabled);
      setMinutes(String(Math.floor(result.timeoutSeconds / 60)));
      setSeconds(String(result.timeoutSeconds % 60));
      setSaved(true);
    }).catch((failure) => setError(errorMessage(failure))).finally(() => setBusy(false));
  }}>
    <label className="user-question-settings__toggle"><input type="checkbox" checked={enabled} disabled={busy}
      onChange={(event) => { changed(); setEnabled(event.currentTarget.checked); }} />Use a suggested answer after a wait</label>
    <div className="user-question-settings__duration">
      <label>Minutes<input type="number" inputMode="numeric" min="0" step="1" value={minutes} disabled={busy || !enabled}
        aria-describedby={`${prefix}-scope`} onChange={(event) => { changed(); setMinutes(event.currentTarget.value); }} /></label>
      <label>Seconds<input type="number" inputMode="numeric" min="0" max="59" step="1" value={seconds} disabled={busy || !enabled}
        aria-describedby={`${prefix}-scope`} onChange={(event) => { changed(); setSeconds(event.currentTarget.value); }} /></label>
    </div>
    <p id={`${prefix}-scope`} className="user-questions__hint">Task choices only. Permissions always use their own approval.</p>
    {error ? <p className="user-question__error" role="alert">{error}</p> : null}
    <div className="user-question__actions"><button type="submit" disabled={busy}>{busy ? "Saving…" : "Save question settings"}</button>
      {saved ? <span className="user-questions__hint" role="status">Saved</span> : null}</div>
  </form>;
}
