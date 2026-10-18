import { useLayoutEffect, useRef, useState } from "react";
import type { McpAppLaunch, McpAppsApi, McpAppHostDisplayMode } from "../../shared/mcpApps";
import type { McpAppArtifactSummary } from "../../shared/mcpAppArtifacts";
import type { McpAppObservationReceipt } from "../../shared/mcpAppTaskData";
import { subscribeAppObservations } from "./mcpAppObservationChannel";
import { McpAppNotificationGate } from "./mcpAppServerNotifications";
import { McpAppDisplayGate, presentMcpAppDisplayMode } from "./mcpAppDisplayModes";
import "./mcpApps.css";

function appApi(): McpAppsApi | undefined {
  const api = typeof window === "undefined" ? undefined : window.omni?.mcp?.apps;
  return api && [api.describe, api.open, api.request, api.close].every(value => typeof value === "function") ? api : undefined;
}
const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
function newerObservation(current: McpAppObservationReceipt | undefined, incoming: McpAppObservationReceipt): McpAppObservationReceipt {
  if (current?.recordSha256 === incoming.recordSha256 && current.nativeReceipt && !incoming.nativeReceipt) return current;
  if (current && Date.parse(incoming.observedAt) < Date.parse(current.observedAt)) return current;
  return incoming;
}

export function McpAppObservationCard({ observation }: { observation: McpAppObservationReceipt }) {
  const learning = observation.nativeReceipt?.head.learning, coverage = observation.nativeReceipt?.head.content?.summary;
  return <details className="mcp-app-view__observation" open><summary>{observation.state === "dormant" ? "Retained App data — no automatic admission" : observation.neuralAdmission === "admitted" && observation.nativeReceipt ? "App data admitted to the brain" : observation.state === "superseded" ? "Older App context retained in audit" : "App task data queued — not yet admitted to the brain"}</summary>
    <p>{observation.method === "ui/message" ? "App-proposed task" : "App context data"} · {observation.observedAt}</p>
    {learning ? <details><summary>Recorded admission details</summary>
      <p>Experience count: {learning.experienceCountBefore.toLocaleString()} → {learning.experienceCountAfter.toLocaleString()}.</p>
      {coverage ? <><p>Input coverage: {coverage.blockCount.toLocaleString()} {coverage.blockCount === 1 ? "block" : "blocks"}{coverage.imageBlocks ? ` · ${coverage.imageFrames.toLocaleString()} image ${coverage.imageFrames === 1 ? "frame" : "frames"}` : ""}{coverage.audioBlocks ? ` · ${coverage.audioModelSamples.toLocaleString()} audio ${coverage.audioModelSamples === 1 ? "sample" : "samples"}` : ""}.</p>
        {coverage.referenceBlocks ? <p>Linked resource references were retained, not fetched.</p> : null}</> : null}
    </details> : <p>{observation.neuralAdmission === "advanced-without-observation" ? "Progress advanced without learning superseded data." : "No completed neural-learning receipt was created."}</p>}
  </details>;
}

/** Only Studio relays reviewed RPC. Neither iframe is given a preload bridge. */
export function McpAppView(props: { brainId: string; viewId: string; api?: McpAppsApi }) {
  return <McpAppViewInstance key={`${props.brainId}:${props.viewId}`} {...props} />;
}
function McpAppViewInstance({ brainId, viewId, api = appApi() }: { brainId: string; viewId: string; api?: McpAppsApi }) {
  const [launch, setLaunch] = useState<McpAppLaunch>();
  const [error, setError] = useState("");
  const [opening, setOpening] = useState(false);
  const [ready, setReady] = useState(false);
  const [height, setHeight] = useState(320);
  const [mapping, setMapping] = useState<McpAppArtifactSummary>();
  const [capture, setCapture] = useState("");
  const [reviewedMapping, setReviewedMapping] = useState(false);
  const [observation, setObservation] = useState<McpAppObservationReceipt>();
  const [displayMode, setDisplayMode] = useState<McpAppHostDisplayMode>("inline");
  const [expandable, setExpandable] = useState(false);
  const [displayBusy, setDisplayBusy] = useState(false);
  const frame = useRef<HTMLIFrameElement>(null);
  const surface = useRef<HTMLDialogElement>(null);
  const changeDisplay = useRef<(mode: McpAppHostDisplayMode) => Promise<void>>(async () => undefined);
  const busy = useRef(false);
  useLayoutEffect(() => {
    if (!api?.onObservation) return;
    let active = true;
    let unsubscribe: () => void = () => undefined;
    try { unsubscribe = subscribeAppObservations(api, event => {
        if (!active || event.brainId !== brainId || event.artifactId !== viewId) return;
        if (event.receipt?.artifactId === viewId) { setObservation(current => newerObservation(current, event.receipt!)); if (event.receipt.nativeReceipt) setError(""); }
        if (event.error) setError(event.error);
      }); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    return () => { active = false; unsubscribe(); };
  }, [api, brainId, viewId]);
  useLayoutEffect(() => {
    if (!launch || !api) return;
    let active = true, phase: "proxy" | "initializing" | "initialized" | "ready" | "closed" = "proxy", sequence = 0;
    let currentMode: McpAppHostDisplayMode = "inline", declaredModes: unknown[] = [], displayChanging = false;
    const forward = (message: Record<string, unknown>): void => {
      if (active) frame.current?.contentWindow?.postMessage({ omniMcpApp: launch.viewToken, message }, launch.proxyOrigin);
    };
    const notifications = new McpAppNotificationGate(launch);
    const display = new McpAppDisplayGate({ brainId, viewId, viewToken: launch.viewToken });
    const present = (mode: McpAppHostDisplayMode): McpAppHostDisplayMode => {
      currentMode = presentMcpAppDisplayMode(surface.current, mode); setDisplayMode(currentMode); return currentMode;
    };
    setExpandable(false); setDisplayBusy(false); setDisplayMode("inline");
    const unsubscribeDisplay = api.onDisplayMode?.(incoming => {
      const event = display.receive(incoming); if (!active || !event) return;
      if (event.action === "closed") {
        phase = "closed"; present("inline"); setDisplayBusy(false); setReady(false); setLaunch(undefined); if (event.reason) setError(event.reason); return;
      }
      if (phase !== "ready") return;
      if (event.action === "apply") {
        setDisplayBusy(true);
        const previous = currentMode;
        const mode = event.displayMode === "inline" || declaredModes.includes(event.displayMode) ? present(event.displayMode) : currentMode;
        void api.acknowledgeDisplayMode?.({ brainId, viewId, viewToken: launch.viewToken, operationId: event.operationId!, mode }).catch(failure => {
          if (!active || !display.current(event.sequence)) return;
          present(previous); setDisplayBusy(false); setError(failure instanceof Error ? failure.message : String(failure));
        });
      } else {
        setDisplayBusy(false);
        const mode = present(event.displayMode);
        forward({ jsonrpc: "2.0", method: "ui/notifications/host-context-changed", params: { displayMode: mode,
          availableDisplayModes: event.availableDisplayModes } });
      }
    });
    changeDisplay.current = async mode => {
      if (!active || phase !== "ready" || displayChanging) return;
      displayChanging = true; setDisplayBusy(true);
      try {
        const result = await api.request({ brainId, viewId, viewToken: launch.viewToken, id: `host-display-${crypto.randomUUID()}`,
          method: "ui/request-display-mode", params: { mode } });
        if (!active) return;
        if (result.error) throw new Error(result.error.message);
      } catch (failure) { if (active) setError(failure instanceof Error ? failure.message : String(failure)); }
      finally { displayChanging = false; if (active) setDisplayBusy(false); }
    };
    const unsubscribeNotifications = api.onServerNotification?.(event => {
      const message = notifications.receive(event, active && phase === "ready");
      if (message) forward(message);
    });
    const receive = (event: MessageEvent): void => {
      if (!active || event.source !== frame.current?.contentWindow || event.origin !== launch.proxyOrigin) return;
      const envelope = record(event.data), message = record(envelope.message);
      if (envelope.omniMcpApp !== launch.viewToken || message.jsonrpc !== "2.0") return;
      const method = typeof message.method === "string" ? message.method : undefined;
      if (!method) return;
      if (method === "ui/notifications/sandbox-proxy-ready") {
        if (phase !== "proxy") return; phase = "initializing";
        forward({ jsonrpc: "2.0", method: "ui/notifications/sandbox-resource-ready", params: { html: launch.resourceHtml,
          resourceUrl: launch.resourceUrl, csp: launch.csp } }); return;
      }
      if (method.startsWith("ui/notifications/sandbox-")) return;
      const hasId = typeof message.id === "string" || typeof message.id === "number";
      if (phase !== "ready" && method !== "ui/initialize" && method !== "ui/notifications/initialized") {
        if (hasId) forward({ jsonrpc: "2.0", id: message.id, error: { code: -32000, message: "Finish initialization before requesting host actions." } }); return;
      }
      const id = hasId ? message.id as string | number : `host-notification-${++sequence}`;
      void api.request({ brainId, viewId, viewToken: launch.viewToken, id, method, params: record(message.params), ...(!hasId ? { notification: true as const } : {}) }).then(result => {
        if (!active) return;
        if (method === "ui/initialize" && !result.error) {
          phase = "initialized";
          const modes = record(record(message.params).appCapabilities).availableDisplayModes;
          declaredModes = Array.isArray(modes) ? modes : [];
          const supported = record(record(result.result).hostContext).availableDisplayModes;
          setExpandable(Boolean(api.onDisplayMode && api.acknowledgeDisplayMode && declaredModes.includes("inline") && declaredModes.includes("fullscreen") &&
            Array.isArray(supported) && supported.includes("fullscreen")));
        }
        if (hasId) forward({ jsonrpc: "2.0", id: message.id, ...result });
        if (method === "ui/notifications/initialized" && !result.error && phase === "initialized") {
          phase = "ready"; setReady(true);
          forward({ jsonrpc: "2.0", method: "ui/notifications/tool-input", params: { arguments: launch.toolInput } });
          forward({ jsonrpc: "2.0", method: "ui/notifications/tool-result", params: launch.toolResult });
        }
        if (method === "ui/notifications/size-changed" && !result.error) {
          const size = record(message.params).height;
          if (typeof size === "number" && Number.isFinite(size)) setHeight(Math.max(160, Math.min(size, window.innerHeight * .75)));
        }
        if ((method === "ui/message" || method === "ui/update-model-context") && !result.error) {
          const receipt = record(record(record(result.result)._meta)["omni/appObservation"]);
          if (receipt.artifactId === viewId && receipt.neuralAdmission === "not-yet-admitted" && ["pending-native-admission", "processing-native-admission", "superseded", "dormant"].includes(String(receipt.state))) setObservation(current => newerObservation(current, receipt as unknown as McpAppObservationReceipt));
        }
      }).catch(failure => { if (active) setError(failure instanceof Error ? failure.message : String(failure)); });
    };
    window.addEventListener("message", receive);
    return () => {
      forward({ jsonrpc: "2.0", id: `host-teardown-${++sequence}`, method: "ui/resource-teardown", params: {} });
      active = false; phase = "closed"; window.removeEventListener("message", receive);
      changeDisplay.current = async () => undefined; display.close(); unsubscribeDisplay?.();
      presentMcpAppDisplayMode(surface.current, "inline");
      notifications.close(); unsubscribeNotifications?.();
      void api.close(brainId, viewId, launch.viewToken).catch(() => undefined);
    };
  }, [launch, api, brainId, viewId]);
  if (!api) return null;
  const open = async (selection?: { captureSha256: string; reviewedMapping: true }): Promise<void> => {
    if (busy.current) return; busy.current = true; setOpening(true); setError("");
    try {
      if (!selection) {
        const summary = await api.describe(brainId, viewId);
        setObservation(summary?.observationQueue?.latest);
        if (summary?.needsMappingReview) { setMapping(summary); setCapture(summary.selectedCapture ?? ""); setReviewedMapping(false); return; }
      }
      const opened = await api.open(brainId, viewId, selection); if (opened.view.brainId !== brainId || opened.view.id !== viewId) throw new Error("Connected-app launch belongs to another instance."); setMapping(undefined); setLaunch(opened);
    }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { busy.current = false; setOpening(false); }
  };
  const close = (): void => { setLaunch(undefined); setReady(false); setDisplayMode("inline"); };
  return <section className="mcp-app-view" aria-label="Interactive connected app">
    <dialog ref={surface} open className="mcp-app-view__surface" aria-label={launch ? `${launch.view.serverLabel} connected app` : "Connected app"}
      onCancel={event => { event.preventDefault(); if (expandable && !displayBusy) void changeDisplay.current("inline"); else close(); }}>
    <header><span>{launch?.view.serverLabel ?? "Connected app"}</span><span className="mcp-app-view__controls">{launch && ready && expandable ?
      <button type="button" className="button button--ghost" disabled={displayBusy} onClick={() => void changeDisplay.current(displayMode === "fullscreen" ? "inline" : "fullscreen")}>
        {displayMode === "fullscreen" ? "Return to chat" : "Expand app"}</button> : null}{launch ? <button type="button" className="button button--ghost" onClick={close}>Close app</button>
      : <button type="button" className="button button--ghost" disabled={opening} onClick={() => void open()}>{opening ? "Reviewing app access…" : "Open interactive app"}</button>}</span></header>
    {mapping ? <fieldset className="mcp-app-view__mapping"><legend>Review retained source mapping</legend><p>Saved branches contain different source or resource versions. Every prior artifact remains saved. Reopening reads a fresh resource through the selected source’s current instance permissions; it does not rerun the original tool.</p>
      <label>Original source<select value={capture} onChange={event => { setCapture(event.currentTarget.value); setReviewedMapping(false); }}><option value="">Choose a retained source…</option>
        {mapping.captures.map(item => <option key={item.sha256} value={item.sha256}>{item.source.serverLabel} · {item.source.endpoint} · {item.observedAt}</option>)}</select></label>
      {capture ? <p className="mcp-app-view__source">{mapping.captures.find(item => item.sha256 === capture)?.source.endpoint}</p> : null}
      <label><input type="checkbox" checked={reviewedMapping} onChange={event => setReviewedMapping(event.currentTarget.checked)} />I reviewed this source mapping and want a fresh permitted resource read.</label>
      <button type="button" className="button button--primary" disabled={!capture || !reviewedMapping || opening} onClick={() => void open({ captureSha256: capture, reviewedMapping: true })}>Reopen reviewed source</button>
    </fieldset> : null}
    {launch ? <><details><summary>Resource and access</summary><p>{launch.view.resourceUri}</p><p>Untrusted UI. Actions still use this instance’s permissions.</p></details>
      {!ready ? <p role="status">Connecting to the sandbox…</p> : null}
      <iframe ref={frame} src={launch.proxyUrl} title={`${launch.view.serverLabel} · sandboxed app`} sandbox="allow-scripts allow-same-origin"
        onLoad={() => frame.current?.contentWindow?.postMessage({ omniMcpApp: launch.viewToken,
          message: { jsonrpc: "2.0", method: "ui/notifications/sandbox-proxy-ping" } }, launch.proxyOrigin)}
        referrerPolicy="no-referrer" allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-write 'none'" style={{ height }} />
    </> : null}
    {error ? <details className="mcp-app-view__error" role="alert" open><summary>App could not continue</summary><p>{error}</p></details> : null}
    </dialog>
    {observation ? <McpAppObservationCard observation={observation} /> : null}
  </section>;
}
