import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { UserQuestionPanel } from "../../src/renderer/src/UserQuestionPanel";
import type { UserQuestionEvent, UserQuestionMode, UserQuestionRecord, UserQuestionSource, UserQuestionsApi } from "../../src/shared/userQuestions";
import "../../src/renderer/src/styles.css";

const preferences = { timeoutEnabled: false, timeoutSeconds: 1200 };
let sequence = 0;
type FixtureHost = ReturnType<typeof makeHost>;
function makeHost(notice: (message: string) => void, holds: Array<() => void>) {
  const rows = new Map<string, UserQuestionRecord>(), listeners = new Set<(event: UserQuestionEvent) => void>();
  const control = { hold: false };
  const emit = (event: UserQuestionEvent) => { for (const listener of listeners) listener(event); };
  const api: UserQuestionsApi = {
    request: async () => { throw new Error("Use the explicit synthetic question controls."); },
    list: async brainId => [...rows.values()].filter(row => row.brainId === brainId),
    preferences: async () => ({ ...preferences }),
    setPreferences: async request => { notice("Synthetic timeout preference saved; no timer or permission runs.");
      return { timeoutEnabled: request.timeoutEnabled, timeoutSeconds: request.timeoutSeconds }; },
    respond: async request => {
      const row = rows.get(request.questionId);
      if (!row || row.brainId !== request.brainId) throw new Error("Synthetic question owner changed.");
      const apply = () => {
        const now = new Date().toISOString();
        const answer: UserQuestionRecord = { ...row, status: "answered", updatedAt: now, resolution: {
          id: crypto.randomUUID(), source: "user", resolvedAt: now, response: request.response,
          authority: "task-choice-only", permissionsGranted: false } };
        rows.set(answer.id, answer); emit({ type: "resolved", brainId: answer.brainId, question: answer });
        notice("Synthetic answer receipt returned. No native model or learning consumed it."); return answer;
      };
      if (!control.hold) return apply();
      control.hold = false; notice("Synthetic answer acknowledgement held; replace the API owner to test retirement.");
      return new Promise<UserQuestionRecord>(resolve => holds.push(() => resolve(apply())));
    },
    cancel: async (brainId, id) => {
      const row = rows.get(id); if (!row || row.brainId !== brainId) throw new Error("Synthetic question unavailable.");
      const now = new Date().toISOString(), saved: UserQuestionRecord = { ...row, status: "cancelled", updatedAt: now,
        resolution: { id: crypto.randomUUID(), source: "cancellation", resolvedAt: now, authority: "task-choice-only", permissionsGranted: false } };
      rows.set(id, saved); emit({ type: "cancelled", brainId, question: saved }); return saved;
    },
    resume: async (brainId, id) => {
      for (const [key, row] of rows) if (row.brainId === brainId && (!id || id === key) && row.suspended) {
        const saved = { ...row, suspended: undefined, resumedAt: new Date().toISOString() }; rows.set(key, saved);
        emit({ type: "updated", brainId, question: saved });
      }
      return [...rows.values()].filter(row => row.brainId === brainId);
    },
    onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
  };
  const ask = (brainId: string, source: UserQuestionSource["kind"], mode: UserQuestionMode, paused = false) => {
    const serial = ++sequence, id = serial.toString(16).padStart(64, "0"), now = new Date().toISOString();
    const row: UserQuestionRecord = { format: "omni-user-question", version: 1, id, brainId, requestId: `synthetic-${serial}`,
      kind: "task-choice", requestSha256: "f".repeat(64), prompt: mode === "text" ? "How should this synthetic story end?" : "Which sections should this synthetic draft use?",
      detail: "This is a synthetic UI question. Expand the detail to inspect wrapping and scroll behavior. ".repeat(6), mode,
      options: mode === "text" ? [] : [{ id: "intro", label: "Introduction", recommended: true, rank: 1, description: "Start with the current premise." },
        { id: "examples", label: "Examples", rank: 2, description: "Include concrete cases for the reader." },
        { id: "appendix", label: "Appendix", rank: 3, description: "Keep supporting details separately." }],
      minSelections: mode === "text" ? 0 : mode === "multiple" ? 2 : 1, maxSelections: mode === "text" ? 0 : mode === "multiple" ? 3 : 1,
      allowOther: mode !== "text", source: { kind: source, turnId: "synthetic-turn", actionId: `synthetic-action-${serial}` },
      status: "pending", createdAt: now, updatedAt: now, timeout: { enabled: false, seconds: 1200, policyRevision: "synthetic" },
      ...(paused ? { suspended: { reason: "copied-instance" as const, fromBrainId: "synthetic-parent", copiedAt: now } } : {}) };
    rows.set(id, row); emit({ type: "created", brainId, question: row });
    notice(`Synthetic ${source} question emitted for ${brainId}. No model called the question tool.`);
  };
  return { api, ask, control, fail: (brainId: string) => emit({ type: "error", brainId, message: "Synthetic question service error. Retry reads the current fixture owner only." }) };
}
function Fixture() {
  const [notice, setNotice] = useState("Empty conversation: no permanent question row should appear.");
  const held = useRef<Array<() => void>>([]), [host, setHost] = useState<FixtureHost>(() => makeHost(setNotice, held.current));
  const [brainId, setBrainId] = useState("synthetic-one"), [source, setSource] = useState<UserQuestionSource["kind"]>("brain");
  const [compact, setCompact] = useState(false), [dark, setDark] = useState(false);
  return <main style={{ maxWidth: compact ? 390 : 960, margin: "0 auto", padding: 16, minWidth: 0,
    height: "100dvh", overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Conversation questions</h1><p>Actual question component with synthetic in-memory APIs. No brain, Load, training, provider, host permission or real chat runs.</p>
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
      <label>Question source <select value={source} onChange={event => setSource(event.target.value as UserQuestionSource["kind"])}>
        <option value="brain">Native chat · synthetic</option><option value="learn">Learn task · synthetic</option><option value="fact-check">Fact Check task · synthetic</option>
      </select></label>
      <button className="button" onClick={() => host.ask(brainId, source, "single")}>Emit single choice</button>
      <button className="button" onClick={() => host.ask(brainId, source, "multiple")}>Emit multiple choice</button>
      <button className="button" onClick={() => host.ask(brainId, source, "text")}>Emit written question</button>
      <button className="button" onClick={() => host.ask(brainId, source, "single", true)}>Emit paused copy</button>
      <button className="button" onClick={() => { host.control.hold = true; setNotice("The next synthetic answer acknowledgement will be held."); }}>Hold next answer</button>
      <button className="button" onClick={() => { setHost(makeHost(setNotice, held.current)); setNotice("New API owner is empty; old pending receipts must not appear here."); }}>Replace API owner</button>
      <button className="button" onClick={() => { held.current.splice(0).forEach(release => release()); }}>Release held answer</button>
      <button className="button" onClick={() => setBrainId(current => current === "synthetic-one" ? "synthetic-two" : "synthetic-one")}>Switch brain</button>
      <button className="button" onClick={() => host.fail(brainId)}>Emit service error</button>
      <button className="button" onClick={() => setCompact(value => !value)}>Compact content</button>
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
    </div>
    <p>Current synthetic owner: {brainId}</p>
    <div className="surface" style={{ padding: 12 }}>
      <p>Static conversation placement marker · not a generated reply</p>
      <UserQuestionPanel brainId={brainId} api={host.api} presentation="conversation" />
      <p>Static composer placement marker · empty question UI must leave no row here</p>
    </div>
    <p role="status">{notice}</p>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
document.documentElement.dataset.workspaceArrangement = "split";
createRoot(document.getElementById("root")!).render(<Fixture />);
