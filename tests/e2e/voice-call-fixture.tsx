import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { VoiceRoutingControls } from "../../src/renderer/src/VoiceRoutingControls";
import { useInstanceVoice } from "../../src/renderer/src/useInstanceVoice";
import { LiveVoiceController } from "../../src/renderer/src/liveVoiceController";
import { DEFAULT_INSTANCE_VOICE_SETTINGS, type InstanceVoiceApi, type InstanceVoiceSnapshot } from "../../src/shared/voiceRouting";
import { TaskAttachmentStrip } from "../../src/renderer/src/TaskAttachmentStrip";
import { useTaskAttachmentDraft } from "../../src/renderer/src/useTaskAttachmentDraft";
import type { ChatTaskAttachmentHandle, ChatTaskAttachmentsApi } from "../../src/shared/taskAttachments";
import "../../src/renderer/src/styles.css";

// Synthetic host edges: no real instance, microphone, neural model, file, tool,
// capture grant, training, provider or external side effect is created.
const noEffects = (): never => { throw new Error("Capture/generation must not run in this UI fixture."); };
const snapshots = new Map<string, InstanceVoiceSnapshot>();
let measured = true;
function snapshot(brainId: string): InstanceVoiceSnapshot {
  let value = snapshots.get(brainId);
  if (!value) { value = { brainId, revision: 1, updatedAt: null, preferences: { ...DEFAULT_INSTANCE_VOICE_SETTINGS }, initialized: true,
    legacyMigrationAllowed: false, inheritedVariants: 0, parameterEvidence: null, parameterDetail: "Synthetic 2B count for controls only. No neural instance exists." }; snapshots.set(brainId, value); }
  return { ...structuredClone(value), parameterEvidence: measured ? { source: "committed-native-parameter-accounting", totalNeuralParameters: 2_000_000_000,
    countingRule: "synthetic-ui-count-only", metadataSha256: "a".repeat(64), substrateGenerationSha256: "b".repeat(64), mutableGenerationSha256: "c".repeat(64),
    parameterChecksum: "d".repeat(64), measuredAt: new Date().toISOString() } : null,
    parameterDetail: measured ? value.parameterDetail : "Synthetic missing count; automatic output must not guess." };
}
const voice: InstanceVoiceApi = { preferences: async id => snapshot(id), migrateLegacy: noEffects,
  update: async request => { const current = snapshot(request.brainId); if (request.expectedRevision !== current.revision) throw new Error("Synthetic stale voice revision");
    snapshots.set(request.brainId, { ...current, preferences: request.preferences, revision: current.revision + 1 }); return snapshot(request.brainId); } };
let handles: ChatTaskAttachmentHandle[] = [];
let effects = 0;
const attachments: ChatTaskAttachmentsApi = {
  stage: async request => { const now = new Date().toISOString(); const row: ChatTaskAttachmentHandle = { format: "omni-chat-task-attachment", version: 1, id: crypto.randomUUID(), brainId: request.brainId,
    displayName: "A very long selected document for the current task.pdf", kind: "file", mimeType: "application/pdf", bytes: 12345, state: "staged",
    contentSha256: "a".repeat(64), manifestSha256: "b".repeat(64), createdAt: now, updatedAt: now, ingestionStarted: false, parametersChanged: false };
    handles.push(row); return [structuredClone(row)]; }, stageDropped: noEffects, cancelStage: async () => true,
  list: async (id, state) => structuredClone(handles.filter(value => value.brainId === id && (!state || value.state === state))), forTurn: async (id, turn) => structuredClone(handles.filter(value => value.brainId === id && value.turnId === turn)),
  byIds: async (id, ids) => structuredClone(handles.filter(value => value.brainId === id && ids.includes(value.id))),
  remove: async (id, attachmentId) => { const row = handles.find(value => value.brainId === id && value.id === attachmentId); if (!row || row.state !== "staged") return false;
    handles = handles.filter(value => value.id !== attachmentId); return true; }
};
function Fixture() {
  const [brainId, setBrainId] = useState("synthetic-a"), [narrow, setNarrow] = useState(false), [active, setActive] = useState(false), [utterance, setUtterance] = useState(false);
  const [open, setOpen] = useState(true), [notice, setNotice] = useState("");
  const controller = useMemo(() => new LiveVoiceController({ brainId, capture: { available: true, request: noEffects },
    recognition: { available: true, start: noEffects }, synthesis: { available: true, speak: noEffects, cancel: () => undefined },
    chat: { send: noEffects, cancel: noEffects } }), [brainId]);
  const controllerRef = useRef<LiveVoiceController | null>(controller); controllerRef.current = controller;
  useEffect(() => () => { void controller.dispose(); }, [controller]);
  const instance = useInstanceVoice(brainId, true, open, controllerRef, voice);
  const draft = useTaskAttachmentDraft(brainId, attachments);
  return <main style={{ padding: 18, maxWidth: 1000, margin: "0 auto", height: "100vh", overflow: "auto", boxSizing: "border-box" }}>
    <h1>Voice & task attachments · UI check</h1><p>Synthetic host and capability data only. No real brain, microphone, file, training or external action runs.</p>
    <div className="automation-buttons"><button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Full surface" : "320px surface"}</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      <button className="button" onClick={() => { setBrainId(value => value === "synthetic-a" ? "synthetic-b" : "synthetic-a"); setNotice(""); }}>Switch synthetic instance</button>
      <button className="button" onClick={() => setActive(value => !value)}>{active ? "End synthetic call" : "Simulate active call"}</button>
      <button className="button" onClick={() => setUtterance(value => !value)}>{utterance ? "End synthetic utterance" : "Simulate active utterance"}</button>
      <button className="button" onClick={() => { measured = !measured; setOpen(value => !value); }}>Toggle synthetic count</button></div>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%" }}>
      <h2>{brainId === "synthetic-a" ? "Instance A" : "Instance B"} voice settings</h2>
      {instance.error ? <p role="alert">{instance.error}</p> : null}
      <VoiceRoutingControls key={brainId} snapshot={instance.snapshot} capabilities={controller.getState().capabilities}
        callActive={active} utteranceActive={utterance} onSave={instance.save} />
      <button className="button" onClick={() => void instance.prepareCall().then(() => setNotice("Selected routes are available in synthetic metadata; capture was NOT started."), failure => setNotice(failure.message))}>Check selected route without capture</button>
      <p role="status">{notice}</p><hr /><h2>Task attachments</h2>
      <button className="button" disabled={draft.busy} onClick={() => void draft.stage()}>Attach synthetic document</button>
      <TaskAttachmentStrip entries={draft.entries} onRemove={id => void draft.remove(id)} /><p role="status">{draft.status}</p>
      <p>Ingestion jobs: {effects}. Neural mutations: {effects}. Capture attempts: {effects}.</p>
    </div>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
