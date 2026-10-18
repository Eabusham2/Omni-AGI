import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import { LivePerceptionPanel } from "../../src/renderer/src/LivePerceptionPanel";
import { CallLayout } from "../../src/renderer/src/CallLayout";
import type { BrainDocument, LiveObservationControl, LiveObservationEvent, LiveObservationSession, OmniApi } from "../../src/shared/types";
import "../../src/renderer/src/styles.css";

// Synthetic capture/acceptance only: real canvas tracks and encoded images, but
// no device, main service, neural worker, learned mutation or permission grant.
const listeners = new Set<(event: LiveObservationEvent) => void>();
const streams: MediaStream[] = [], receipts: string[] = [];
let refresh = () => {}, denied = false, current: LiveObservationSession | undefined;
const now = () => new Date().toISOString();
const record = (value: string) => { receipts.push(value); refresh(); };
async function syntheticStream(kind: string): Promise<MediaStream> {
  record(`${kind} request`);
  if (denied) throw new DOMException("Synthetic fixture: system capture permission denied.", "NotAllowedError");
  const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 360;
  const context = canvas.getContext("2d")!, image = new Image(); image.src = "/build/icon.svg";
  await image.decode();
  context.fillStyle = "#172232"; context.fillRect(0, 0, canvas.width, canvas.height);
  context.drawImage(image, 235, 50, 170, 170);
  context.fillStyle = "white"; context.font = "20px system-ui"; context.textAlign = "center";
  context.fillText(`Synthetic ${kind} · product icon fixture`, 320, 280);
  const stream = canvas.captureStream(8); streams.push(stream); refresh(); return stream;
}
Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: {
  getUserMedia: () => syntheticStream("camera"), getDisplayMedia: () => syntheticStream("screen")
} });
const modality = {
  startObservation: async request => {
    current = { id: crypto.randomUUID(), brainId: request.brainId, state: "active", modalities: request.modalities,
      permission: request.permission, retention: request.retention ?? "working", maxInFlight: 2, maxPacketBytes: 8 * 1024 * 1024,
      inFlight: 0, packetsReceived: 0, packetsAccepted: 0, packetsDroppedBackpressure: 0, bytesAccepted: 0, lastSequence: -1,
      createdAt: now(), updatedAt: now(), capabilities: { imageNeural: true, audioNeural: false, videoNeural: true }, capture: { ...request.capture!, revision: 0 } };
    record(`synthetic session ${request.brainId}`); return { ...current };
  },
  pushObservation: async packet => {
    if (!current || current.state !== "active" || current.id !== packet.sessionId) throw new Error("Synthetic session is closed.");
    current = { ...current, packetsReceived: current.packetsReceived + 1, packetsAccepted: current.packetsAccepted + 1,
      bytesAccepted: current.bytesAccepted + packet.data.byteLength, lastSequence: packet.sequence, updatedAt: now() };
    if (packet.settings?.observationControlId) record(`accepted actual encoded snapshot ${packet.data.byteLength} bytes`);
    return { accepted: true, session: { ...current } };
  },
  stopObservation: async id => { if (!current || current.id !== id) throw new Error("Unknown fixture session"); current = { ...current, state: "stopped" }; record("stop receipt"); return { ...current }; },
  cancelObservation: async id => { if (!current || current.id !== id) throw new Error("Unknown fixture session"); current = { ...current, state: "cancelled" }; record("cancel receipt"); return { ...current }; },
  requestObservationControl: async request => {
    const control: LiveObservationControl = { id: crypto.randomUUID(), sessionId: request.sessionId, source: "human", temporary: true,
      kind: request.kind, state: "requested", requested: request.requested, createdAt: now(), updatedAt: now() };
    queueMicrotask(() => { if (current) listeners.forEach(listener => listener({ type: "control", session: { ...current! }, control })); });
    return control;
  },
  resolveObservationControl: async request => ({ id: request.controlId, sessionId: request.sessionId, source: "human", temporary: true,
    kind: "snapshot", requested: {}, state: request.state, actual: request.actual, reason: request.reason, createdAt: now(), updatedAt: now() }),
  onObservation: listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }
} satisfies Pick<OmniApi["modality"], "startObservation" | "pushObservation" | "stopObservation" | "cancelObservation" | "requestObservationControl" | "resolveObservationControl" | "onObservation">;
window.omni = { modality, catalog: { hardwareProfile: async () => ({ recommendedTier: "personal" }), resourcePlan: async () => null } } as unknown as OmniApi;

function Fixture() {
  const [open, setOpen] = useState(true), [loaded, setLoaded] = useState(true), [identity, setIdentity] = useState(1), [narrow, setNarrow] = useState(false);
  const [, update] = useState(0); refresh = () => update(value => value + 1);
  const brain = { id: `capture-fixture-${identity}`, config: { workingMemoryMode: "auto", workingMemorySlots: 4096, systemRamMode: "auto" },
    readiness: {}, traces: [] } as unknown as BrainDocument;
  const active = streams.flatMap(stream => stream.getTracks()).filter(track => track.readyState === "live").length;
  return <main style={{ maxWidth: 1000, height: "100vh", margin: "auto", padding: 16, overflowY: "auto", boxSizing: "border-box" }}>
    <h1>Call camera/screen UI check</h1><p>Synthetic canvas streams and synthetic acceptance receipts. Your camera, microphone, screen, and models are not accessed.</p>
    <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBlock: 12 }}>
      <button className="button" onClick={() => setOpen(value => !value)}>{open ? "Collapse Call" : "Open Call"}</button>
      <button className="button" onClick={() => setNarrow(value => !value)}>{narrow ? "Wide surface" : "320px surface"}</button>
      <button className="button" onClick={() => { document.documentElement.dataset.colorScheme = document.documentElement.dataset.colorScheme === "dark" ? "light" : "dark"; }}>Switch appearance</button>
      <button className="button" onClick={() => setLoaded(value => !value)}>{loaded ? "Unload fixture" : "Load fixture"}</button>
      <button className="button" onClick={() => setIdentity(value => value + 1)}>Switch fixture instance</button>
      <button className="button" onClick={() => { denied = !denied; record(`deny next request: ${denied}`); }}>Toggle synthetic permission denial</button>
      <button className="button" onClick={() => { streams.forEach(stream => stream.getTracks().forEach(track => { track.stop(); track.dispatchEvent(new Event("ended")); })); record("synthetic device-ended event"); }}>End synthetic source</button>
    </div>
    <p role="status">Instance {identity} · {loaded ? "loaded" : "unloaded"} · active synthetic tracks: {active}</p>
    <div style={{ width: narrow ? 320 : "100%", maxWidth: "100%", containerType: "inline-size" }}><CallLayout open={open} controls={<section className="chat-call-panel" hidden={!open}>
      <LivePerceptionPanel brain={brain} runtimeLoaded={loaded} />
    </section>}><section className="fixture-chat"><h2>Conversation stays beside Call</h2><p>No neural chat runs in this UI fixture.</p></section></CallLayout></div>
    <details><summary>Capture receipts</summary><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipts.join("\n")}</pre></details>
  </main>;
}
document.documentElement.dataset.colorScheme = "dark"; document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
