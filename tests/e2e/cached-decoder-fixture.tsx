import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { CachedDecoderExecutionControls } from "../../src/renderer/src/CachedDecoderExecutionControls";
import { CACHED_DECODER_EXECUTION_WORK_KIND, type CachedDecoderExecutionApi, type CachedDecoderExecutionPrepared, type CachedDecoderExecutionStatus } from "../../src/shared/cachedDecoderExecution";
import type { DecoderExecutionCandidate } from "../../src/shared/decoderExecution";
import "../../src/renderer/src/styles.css";

const candidate: DecoderExecutionCandidate = { planSha256: "a".repeat(64), executionDevice: "cpu", deviceId: "fixture-cpu", memoryGroupId: "fixture-ram",
  backendMode: "native-cpu", asyncDevices: [], supported: true, fallbackAcknowledged: false };
const initial = (brainId: string): CachedDecoderExecutionStatus => ({ format: "omni-cached-decoder-execution-status", version: 1, ownerId: brainId,
  ownerSessionId: `${brainId}-session`, phase: "idle", workKind: CACHED_DECODER_EXECUTION_WORK_KIND, operationId: null,
  reason: "synthetic-only-no-model", progress: null, cancellationRequested: false, selectedCandidate: null,
  actualOperationConsumed: false, exactCacheAndSamplingMatched: false, globalPlacementChanged: false,
  connectedDecoderSupported: false, wholeBrainOrMediaMeasured: false, reusableAcrossWorkloads: false });
function prepared(brainId: string): CachedDecoderExecutionPrepared {
  return { format: "omni-cached-decoder-execution-review", version: 1, ownerId: brainId, reviewId: `fixture-review-${brainId}`,
    binding: { ownerSessionId: `${brainId}-session`, policySha256: "b".repeat(64), topologySha256: "c".repeat(64), registrySha256: "d".repeat(64),
      budgetSha256: "e".repeat(64), homeExecutionDevice: "cpu" }, workKind: CACHED_DECODER_EXECUTION_WORK_KIND,
    currentCandidate: candidate, candidates: [candidate], minimumSamples: 3,
    stageCoverage: ["cache-cow-snapshot", "cached-workspace", "routed-decoder-blocks", "home-readout", "same-home-sampling", "retained-cache-return"],
    status: { ...initial(brainId), phase: "prepared" } };
}
function Fixture() {
  const [brainId, setBrainId] = useState("original"), [loaded, setLoaded] = useState(true), [dirty, setDirty] = useState(false);
  const [foreground, setForeground] = useState(false), [dark, setDark] = useState(false), [compact, setCompact] = useState(false);
  const [receipts, setReceipts] = useState<string[]>([]), [held, setHeld] = useState(false);
  const states = useRef(new Map<string, CachedDecoderExecutionStatus>()), holdNext = useRef(false), wrongNext = useRef(false);
  const delayed = useRef<{ owner: string; resolve(value: CachedDecoderExecutionPrepared): void; result: CachedDecoderExecutionPrepared } | undefined>(undefined);
  const log = (message: string) => setReceipts(values => [...values.slice(-19), message]);
  const api = useMemo<CachedDecoderExecutionApi>(() => ({
    prepare: async request => {
      log(`Synthetic Prepare ${JSON.stringify(request)}`);
      const result = prepared(request.brainId);
      if (wrongNext.current) { wrongNext.current = false; result.status.ownerId = "wrong-owner"; return result; }
      if (holdNext.current) {
        holdNext.current = false; setHeld(true);
        return new Promise(resolve => { delayed.current = { owner: request.brainId, resolve, result }; });
      }
      states.current.set(request.brainId, result.status); return result;
    },
    arm: async request => {
      log(`Synthetic Arm ${JSON.stringify(request)}`);
      const value: CachedDecoderExecutionStatus = { ...initial(request.brainId), phase: "awaiting-retained-frontier", operationId: `fixture-operation-${request.brainId}` };
      states.current.set(request.brainId, value); return value;
    },
    status: async owner => structuredClone(states.current.get(owner) ?? initial(owner)),
    stop: async request => {
      log(`Synthetic Stop ${JSON.stringify(request)}`);
      const value: CachedDecoderExecutionStatus = { ...(states.current.get(request.brainId) ?? initial(request.brainId)), phase: "cancelled",
        progress: null, cancellationRequested: true, reason: "synthetic-stop" };
      states.current.set(request.brainId, value); return value;
    }
  }), []);
  const advance = () => {
    const current = states.current.get(brainId) ?? initial(brainId);
    if (current.phase === "awaiting-retained-frontier") states.current.set(brainId, { ...current, phase: "calibrating",
      progress: { stage: "measuring", candidate: 1, candidates: 1, sample: 1, samplesPerCandidate: 3 } });
    else if (current.phase === "calibrating") states.current.set(brainId, { ...current, phase: "completed", progress: null,
      selectedCandidate: candidate, actualOperationConsumed: true, exactCacheAndSamplingMatched: true });
    log(`Synthetic state advanced from ${current.phase}; no native execution`);
  };
  return <main style={{ maxWidth: compact ? 360 : 960, height: "100dvh", overflow: "auto", margin: "0 auto", padding: 16, boxSizing: "border-box" }}>
    <h1>Cached computation controls</h1><p>Explicit synthetic UI fixture. No model, neural work, preloading, training, measurement or provider calls.</p>
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button className="button" onClick={() => setLoaded(value => !value)}>Toggle loaded state</button>
      <button className="button" onClick={() => setDirty(value => !value)}>Toggle unsaved settings</button>
      <button className="button" onClick={() => setForeground(value => !value)}>Toggle foreground turn</button>
      <button className="button" onClick={() => setBrainId(value => value === "original" ? "copy" : "original")}>Switch fixture instance</button>
      <button className="button" onClick={() => { holdNext.current = true; log("Next synthetic Prepare will be held"); }}>Hold next review</button>
      <button className="button" disabled={!held} onClick={() => { const value = delayed.current; delayed.current = undefined; setHeld(false);
        if (value) { log(`Late synthetic review delivered for ${value.owner}`); value.resolve(value.result); } }}>Deliver held review</button>
      <button className="button" onClick={() => { wrongNext.current = true; log("Next synthetic review will have a wrong owner"); }}>Wrong owner on next review</button>
      <button className="button" onClick={advance}>Advance synthetic computation</button>
      <button className="button" onClick={() => { setDark(value => !value); document.documentElement.dataset.colorScheme = dark ? "light" : "dark"; }}>Switch theme</button>
      <button className="button" onClick={() => setCompact(value => !value)}>Compact content</button>
    </div>
    <p>Fixture instance: {brainId} · {loaded ? "loaded" : "unloaded"} · {dirty ? "unsaved settings" : "saved settings"} · {foreground ? "turn active" : "turn idle"}</p>
    <CachedDecoderExecutionControls api={api} brainId={brainId} loaded={loaded} dirty={dirty} foregroundTurnActive={foreground} />
    <h2>Synthetic receipts</h2><pre style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{receipts.join("\n") || "No synthetic action called"}</pre>
  </main>;
}
document.documentElement.dataset.colorScheme = "light";
document.documentElement.dataset.layout = "glass";
createRoot(document.getElementById("root")!).render(<Fixture />);
