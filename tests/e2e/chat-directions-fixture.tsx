import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { ChatDirectionStrip } from "../../src/renderer/src/ChatDirectionStrip";
import { useChatDirections } from "../../src/renderer/src/useChatDirections";
import { chatDirectionStripItem, isSteerDirectionMessage } from "../../src/renderer/src/chatDirectionPresentation";
import type { SteerDirectionRecord, SteerDirectionsApi, SteerDirectionEvent } from "../../src/shared/steerDirections";
import type { ChatMessage } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// Real presentation and hook; synthetic transport only. No model, learning,
// actual source file, provider, tool, permission or persistent user data runs.
const rows = new Map<string, SteerDirectionRecord[]>();
const listeners = new Set<(event: SteerDirectionEvent) => void>();
let lookups = 0, largestLookup = 0;
function seed(brainId: string): SteerDirectionRecord[] {
  if (!rows.has(brainId)) rows.set(brainId, Array.from({ length: brainId === "A" ? 43 : 1 }, (_, index) => {
    const turnId = `${brainId}-direction-${index}`, createdAt = new Date(Date.UTC(2026, 9, 13, 12, 0, index)).toISOString();
    return { format: "omni-steer-direction", version: 1, id: turnId, brainId, sequence: index + 1,
      turnId, predecessorTurnId: `${brainId}-original-task`, bindingSha256: String(index % 10).repeat(64),
      source: { brainId, turnId, predecessorTurnId: `${brainId}-original-task`, createdAt,
        content: index === 42 ? "Use the second source, preserve the first draft, and continue checking the appendix before the conclusion." : `${brainId} saved direction ${index + 1}`,
        attachmentIds: [], inputKind: "text" }, state: index === 42 ? "pending" : "complete", revision: 1,
      createdAt, updatedAt: createdAt, humanMessageId: `human-${turnId}` };
  }));
  return rows.get(brainId)!;
}
function changed(value: SteerDirectionRecord) {
  listeners.forEach(listener => listener({ brainId: value.brainId, id: value.id, sequence: value.sequence,
    revision: value.revision, state: value.state, type: value.dismissedAt ? "dismissed" : "changed", at: value.updatedAt }));
}
const api: SteerDirectionsApi = {
  list: async (brainId, query = {}) => {
    const all = seed(brainId), selected = all.filter(row => (query.includeDismissed || !row.dismissedAt) &&
      (query.beforeSequence === undefined || row.sequence < query.beforeSequence)).sort((a, b) => b.sequence - a.sequence);
    const entries = selected.slice(0, query.limit ?? 40), nextBeforeSequence = selected.length > entries.length ? entries.at(-1)!.sequence : undefined;
    return structuredClone({ entries, total: all.length, nextBeforeSequence });
  },
  get: async (brainId, id) => structuredClone(seed(brainId).find(row => row.id === id) ?? null),
  forTurns: async (brainId, ids) => { lookups++; largestLookup = Math.max(largestLookup, ids.length);
    if (ids.length > 100) throw new Error("Synthetic host enforces the real 100-turn query boundary.");
    return structuredClone(seed(brainId).filter(row => ids.includes(row.turnId))); },
  dismiss: async request => { const row = seed(request.brainId).find(value => value.id === request.id);
    if (!row || row.revision !== request.expectedRevision) throw new Error("Stale direction revision; reopen the latest entry.");
    row.dismissedAt = request.dismissed ? new Date().toISOString() : undefined; row.revision++; row.updatedAt = new Date().toISOString();
    changed(row); return structuredClone(row); },
  onEvent: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("A"), [active, setActive] = useState(true), [narrow, setNarrow] = useState(false);
  const [draft, setDraft] = useState("Unsent draft stays here"), [notice, setNotice] = useState("");
  const direction = seed(brainId).at(-1)!;
  const messages: ChatMessage[] = [
    { id: "original-human", turnId: `${brainId}-original-task`, role: "human", content: "Review both sources and write a draft.", createdAt: direction.createdAt },
    { id: "original-brain", turnId: `${brainId}-original-task`, role: "brain", content: "The existing reply remains here while steering changes the task.", createdAt: direction.createdAt },
    { id: direction.humanMessageId!, turnId: direction.turnId, role: "human", content: direction.source.content, createdAt: direction.createdAt }
  ];
  const directions = useChatDirections(brainId, messages.filter(value => value.role === "human").map(value => value.turnId!), api);
  return <main style={{ padding: 18, maxWidth: 950, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Existing steering controls · UI check</h1>
    <p>Synthetic transport only. No real brain, model, learning, file or tool runs.</p>
    <div className="automation-buttons">
      <button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      <button className="button" onClick={() => { setBrainId(value => value === "A" ? "B" : "A"); setNotice(""); }}>Switch synthetic instance</button>
      <button className="button" onClick={() => setActive(value => !value)}>{active ? "Simulate reply finished" : "Simulate responding"}</button>
      <button className="button" onClick={() => { const row = seed(brainId).at(-1)!; row.state = "complete"; row.revision++; row.updatedAt = new Date().toISOString(); changed(row); }}>Complete synthetic direction</button>
    </div>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%" }}>
      <h2>Instance {brainId}</h2>
      <section aria-label="Existing chat messages">{messages.filter(value => !isSteerDirectionMessage(value, directions.records)).map(value => <p key={value.id}><strong>{value.role}: </strong>{value.content}</p>)}</section>
      <ChatDirectionStrip key={brainId} items={directions.records.map(chatDirectionStripItem)} total={directions.total} busy={directions.busy}
        canSteer={active} hasOlder={directions.hasOlder} onOlder={() => void directions.older()} onDismiss={directions.dismiss}
        onRevise={async text => { if (active) setNotice(`Synthetic new direction: ${text}. Accepted source not edited.`);
          else setDraft(current => current.trim() ? `${current}\n\n${text}` : text); }} />
      {directions.error ? <p role="alert">{directions.error}</p> : null}
      <label>Existing composer<textarea style={{ width: "100%", minHeight: 90 }} value={draft} onChange={event => setDraft(event.currentTarget.value)} /></label>
      <p role="status">{notice}</p>
      <p>Exact lookups: {lookups}. Largest batch: {largestLookup}. Model submissions: 0. Neural changes: 0.</p>
    </div>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
