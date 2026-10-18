import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { NativeOperationAutoControls } from "../../src/renderer/src/NativeOperationAutoControls";
import { PonderSelfReportView } from "../../src/renderer/src/PonderSelfReportView";
import { PonderPresentationLedger } from "../../src/renderer/src/ponderSelfReportPresentation";
import { NATIVE_OPERATION_AUTO_STAGE_COVERAGE, NATIVE_OPERATION_AUTO_WORK_KINDS,
  type NativeOperationAutoApi, type NativeOperationAutoPrepared, type NativeOperationAutoStatus } from "../../src/shared/nativeOperationAuto";
import type { ActionEvent } from "../../src/shared/types";
import type { PonderSelfReportDelta } from "../../src/shared/ponderSelfReport";
import "../../src/renderer/src/styles.css";
const candidate = { planSha256: "a".repeat(64), executionDevice: "cpu", deviceId: "cpu:0", memoryGroupId: "ram:0",
  backendMode: "native-cpu" as const, asyncDevices: [], supported: true, fallbackAcknowledged: false };
const initial = (id: string): NativeOperationAutoStatus => ({ format: "omni-native-operation-auto-status", version: 2,
  ownerId: id, ownerSessionId: `${id}-session`, phase: "idle", operationId: null, reason: "synthetic-only",
  progress: null, cancellationRequested: false, workKind: null, qualifiedPlanCount: 0, lastPlan: null,
  supportedWorkKinds: [...NATIVE_OPERATION_AUTO_WORK_KINDS], unsupportedRoutes: ["training", "whole-brain", "retained-context"],
  globalPlacementChanged: false, calibrationOnAppOpen: false, wholeBrainOrTrainingMeasured: false,
  qualificationIsExact: true, reusableRouteIsInferenceForUnseenInputs: true });
const prepare = (id: string): NativeOperationAutoPrepared => ({ format: "omni-native-operation-auto-review", version: 2, ownerId: id,
  reviewId: `review-${id}`, binding: { ownerSessionId: `${id}-session`, policySha256: "b".repeat(64), topologySha256: "c".repeat(64),
    registrySha256: "d".repeat(64), budgetSha256: "e".repeat(64), homeExecutionDevice: "cpu" }, currentCandidate: candidate,
  candidates: [candidate], minimumSamples: 3, stageCoverageByWorkKind: NATIVE_OPERATION_AUTO_WORK_KINDS.map(workKind => ({ workKind,
    stages: [...NATIVE_OPERATION_AUTO_STAGE_COVERAGE[workKind]] })), supportedWorkKinds: [...NATIVE_OPERATION_AUTO_WORK_KINDS],
  connectedPerceptionSupported: true, status: { ...initial(id), phase: "prepared" } });
const actionId = "a".repeat(32), reportId = "b".repeat(64), prefix = "This is synthetic published Ponder text. ";
const suffix = "Only display is slowed; computation and reply delivery are independent. This deliberately long excerpt checks wrapping, selection, contrast and interrupted publication without inventing a model result. ".repeat(6);
function Fixture() {
  const [brainId, setBrainId] = useState("original"), [dark, setDark] = useState(false), [compact, setCompact] = useState(false);
  const [dirty, setDirty] = useState(false), [loaded, setLoaded] = useState(true), [receipts, setReceipts] = useState<string[]>([]);
  const [held, setHeld] = useState(false), hold = useRef(false), wrong = useRef(false);
  const late = useRef<(() => void) | undefined>(undefined), states = useRef(new Map<string, NativeOperationAutoStatus>());
  const [readable, setReadable] = useState(false), [hidden, setHidden] = useState(false), [revision, update] = useState(0);
  const ledger = useMemo(() => new PonderPresentationLedger("fixture"), []), reportStarted = useRef(false);
  const log = (text: string) => setReceipts(rows => [...rows.slice(-12), text]);
  const api = useMemo<NativeOperationAutoApi>(() => ({
    prepare: async request => {
      log(`Synthetic Prepare ${request.brainId}`); const value = prepare(request.brainId);
      if (wrong.current) { wrong.current = false; value.ownerId = "wrong-owner"; }
      if (hold.current) { hold.current = false; setHeld(true); return new Promise(resolve => { late.current = () => { resolve(value); }; }); }
      states.current.set(request.brainId, value.status); return value;
    },
    arm: async request => { log(`Synthetic Arm ${request.brainId}`); const value = { ...initial(request.brainId), phase: "awaiting-operation" as const, operationId: `operation-${request.brainId}` };
      states.current.set(request.brainId, value); return value; },
    status: async id => structuredClone(states.current.get(id) ?? initial(id)),
    stop: async request => { log(`Synthetic Stop ${request.brainId}`); const value = { ...(states.current.get(request.brainId) ?? initial(request.brainId)),
      phase: "cancelled" as const, cancellationRequested: true }; states.current.set(request.brainId, value); return value; },
    plans: async request => { log(`Synthetic plan page ${request.after ? 2 : 1}`); return { ownerId: request.brainId, ownerSessionId: `${request.brainId}-session`, revision: 1,
      total: 17, plans: Array.from({ length: request.after ? 1 : 16 }, (_, index) => ({ id: (request.after ? "f" : index.toString(16)).repeat(64),
        workKind: (["native.perception.image-embedding", "native.perception.audio-embedding", "native.perception.video-embedding"] as const)[index % 3]!, geometrySha256: "d".repeat(64), qualifiedNativeStateSha256: "e".repeat(64),
        candidate, qualifiedSamples: 3, qualificationWorkloads: 1 as const, numericGate: "exact-embedding-equality" as const,
        reuseBasis: "same-validated-operator-geometry-current-owner-policy-inference" as const, unseenInputOrFrontierWasPreviouslyMeasured: false as const, wholeBrainOrTrainingCovered: false as const })),
      ...(request.after ? {} : { nextAfter: "synthetic-next-page" }) }; }
  }), []);
  const publish = (append: boolean) => {
    if (!append && reportStarted.current || append && !reportStarted.current) return;
    ledger.recognize({ id: "fixture-host-action", neuralActionId: actionId, brainId: "fixture", action: { kind: "ponder" } } as ActionEvent, "turn");
    const frame: PonderSelfReportDelta = { format: "omni-ponder-self-report-delta", version: 1, reportId, actionId, phase: "pre-speech",
      chunkIndex: append ? 1 : 0, byteOffset: append ? new TextEncoder().encode(prefix).length : 0, delta: append ? suffix : prefix,
      publicationComplete: false, modelEmitted: true, selfReportNotChainOfThought: true, hiddenPromptUsed: false, inferenceCompletedBeforePublication: true };
    ledger.receive("turn", frame); reportStarted.current = true; update(value => value + 1); log(append ? "Synthetic second publication frame" : "Synthetic first publication frame");
  };
  const rows = useMemo(() => ledger.snapshot(), [ledger, revision]);
  return <main style={{ maxWidth: compact ? 360 : 940, height: "100dvh", overflow: "auto", margin: "0 auto", padding: 16, boxSizing: "border-box" }}>
    <h1>Native operation controls</h1><p>Synthetic UI only. Zero neural, provider, calibration, training or host-effect calls.</p>
    <div className="decoder-execution__actions">
      <button className="button" onClick={() => { setDark(!dark); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(!compact)}>Compact content</button>
      <button className="button" onClick={() => setDirty(!dirty)}>Toggle unsaved settings</button>
      <button className="button" onClick={() => setLoaded(!loaded)}>Toggle loaded state</button>
      <button className="button" onClick={() => setBrainId(brainId === "original" ? "copy" : "original")}>Switch fixture instance</button>
      <button className="button" onClick={() => { hold.current = true; }}>Hold next review</button>
      <button className="button" disabled={!held} onClick={() => { late.current?.(); late.current = undefined; setHeld(false); log("Delivered late synthetic review"); }}>Deliver held review</button>
      <button className="button" onClick={() => { wrong.current = true; }}>Wrong owner next</button>
    </div>
    <NativeOperationAutoControls api={api} brainId={brainId} loaded={loaded} dirty={dirty} foregroundTurnActive={false} />
    <h2>Ponder publication</h2><div className="decoder-execution__actions">
      <button className="button" onClick={() => publish(false)}>Publish synthetic prefix</button>
      <button className="button" onClick={() => publish(true)}>Append synthetic text</button>
      <button className="button" onClick={() => { ledger.finish("turn"); update(value => value + 1); }}>Stop publication</button>
      <button className="button" onClick={() => setHidden(!hidden)}>Toggle chat hidden</button>
    </div>
    <details className="chat-action-card" open><summary>Ponder</summary>
      <div className="chat-action-card__body">{rows.map(row => <PonderSelfReportView key={row.report.id} row={row} readable={readable} onReadableChange={setReadable} flush={hidden} />)}</div>
    </details>
    <h2>Synthetic receipts</h2><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipts.join("\n") || "No synthetic action"}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
