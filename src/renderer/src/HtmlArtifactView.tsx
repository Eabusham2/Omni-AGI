import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ActionEvent } from "../../shared/types";
import type {
  HtmlArtifactsApi,
  HtmlArtifactLaunch,
  HtmlArtifactOpenRequest,
  HtmlArtifactToolReceipt
} from "../../shared/htmlArtifacts";
import {
  chatHtmlArtifactRequest,
  htmlArtifactError,
  htmlArtifactHostReady,
  htmlArtifactIdentity,
  htmlArtifactRequestFromMessage,
  htmlArtifactResponse,
  isCurrentHtmlArtifactOperation,
  nextHtmlArtifactReviews
} from "./htmlArtifactPresentation";
import "./htmlArtifacts.css";

function currentApi(): HtmlArtifactsApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.artifacts;
}

function Review({ receipt, busy, onDecision }: {
  receipt: HtmlArtifactToolReceipt;
  busy: boolean;
  onDecision(value: "approve" | "deny"): void;
}) {
  if (!receipt.review || !receipt.reviewId) return null;
  return <section className="plugin-approval-review" aria-label={`HTML artifact action review ${receipt.requestId}`}>
    <strong>Artifact requests {receipt.review.resolved.toolId}.{receipt.review.resolved.action}</strong>
    <pre tabIndex={0}>{JSON.stringify(receipt.review.resolved.arguments, null, 2)}</pre>
    <small>Exact saved artifact {receipt.review.contentSha256.slice(0, 12)}… · current tool permission is checked again before dispatch.</small>
    <div className="plugin-manager__actions">
      <button type="button" className="button button--primary" disabled={busy} onClick={() => onDecision("approve")}>Approve exact action</button>
      <button type="button" className="button" disabled={busy} onClick={() => onDecision("deny")}>Deny</button>
    </div>
  </section>;
}

/** Saved chat HTML remains in an opaque frame; raw markup never enters Studio. */
export function HtmlArtifactView({ event, api = currentApi() }: { event: ActionEvent; api?: HtmlArtifactsApi }) {
  const request = chatHtmlArtifactRequest(event);
  if (!request || !api) return null;
  return <HtmlArtifactViewInstance key={htmlArtifactIdentity(request)} request={request} api={api} />;
}

function HtmlArtifactViewInstance({ request, api }: { request: HtmlArtifactOpenRequest; api: HtmlArtifactsApi }) {
  const identity = htmlArtifactIdentity(request);
  const [launch, setLaunch] = useState<HtmlArtifactLaunch>();
  const [error, setError] = useState("");
  const [opening, setOpening] = useState(false);
  const [activating, setActivating] = useState(false);
  const [reviews, setReviews] = useState<HtmlArtifactToolReceipt[]>([]);
  const [reviewing, setReviewing] = useState<string[]>([]);
  const [lastReceipt, setLastReceipt] = useState<HtmlArtifactToolReceipt>();
  const frame = useRef<HTMLIFrameElement>(null), mounted = useRef(false), identityRef = useRef(identity);
  const launchRef = useRef<HtmlArtifactLaunch | undefined>(undefined), openEpoch = useRef(0);
  const relays = useRef(new Map<string, { operandsJson: string; promise: Promise<HtmlArtifactToolReceipt> }>());
  const reviewMap = useRef(new Map<string, HtmlArtifactToolReceipt>()), reviewingSet = useRef(new Set<string>());
  identityRef.current = identity; launchRef.current = launch;

  useLayoutEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const replaceReview = useCallback((receipt: HtmlArtifactToolReceipt) => {
    const next = nextHtmlArtifactReviews([...reviewMap.current.values()], receipt);
    reviewMap.current = new Map(next.map((item) => [item.requestId, item]));
    if (mounted.current) setReviews(next);
  }, []);
  const clearReviews = useCallback(() => {
    reviewMap.current.clear(); reviewingSet.current.clear();
    if (mounted.current) { setReviews([]); setReviewing([]); }
  }, []);
  const post = useCallback((owner: HtmlArtifactLaunch, message: Record<string, unknown>) => {
    if (launchRef.current?.viewId !== owner.viewId) return;
    frame.current?.contentWindow?.postMessage(message, "*");
  }, []);

  const open = useCallback(async () => {
    const ownerIdentity = identity, epoch = ++openEpoch.current;
    setOpening(true); setError(""); clearReviews(); setLastReceipt(undefined);
    let opened: HtmlArtifactLaunch | undefined;
    try {
      opened = await api.open(request);
      if (!isCurrentHtmlArtifactOperation({ mounted: mounted.current,
        currentIdentity: identityRef.current, expectedIdentity: ownerIdentity,
        currentOpenEpoch: openEpoch.current, expectedOpenEpoch: epoch })) {
        await api.close(opened.artifact.brainId, opened.viewId, opened.channelNonce).catch(() => undefined);
        return;
      }
      if (opened.artifact.brainId !== request.brainId || opened.artifact.originalBrainId !== request.originalBrainId ||
          opened.artifact.artifactId !== request.artifactId || opened.artifact.sourceActionId !== request.sourceActionId ||
          opened.artifact.contentSha256 !== request.contentSha256) {
        await api.close(opened.artifact.brainId, opened.viewId, opened.channelNonce).catch(() => undefined);
        throw new Error("Saved HTML artifact launch differs from the completed chat result.");
      }
      launchRef.current = opened; setLaunch(opened);
    } catch (failure) {
      if (isCurrentHtmlArtifactOperation({ mounted: mounted.current,
        currentIdentity: identityRef.current, expectedIdentity: ownerIdentity,
        currentOpenEpoch: openEpoch.current, expectedOpenEpoch: epoch })) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    } finally {
      if (isCurrentHtmlArtifactOperation({ mounted: mounted.current,
        currentIdentity: identityRef.current, expectedIdentity: ownerIdentity,
        currentOpenEpoch: openEpoch.current, expectedOpenEpoch: epoch })) setOpening(false);
    }
  }, [api, clearReviews, identity, request]);

  useEffect(() => {
    void open();
  }, [api, identity, open, request]);

  useLayoutEffect(() => {
    if (!api || !launch) return;
    let active = true;
    const receive = (event: MessageEvent): void => {
      const parsed = htmlArtifactRequestFromMessage(event, frame.current?.contentWindow ?? null, launch);
      if (!parsed) return;
      const prior = relays.current.get(parsed.request.requestId);
      if (prior && prior.operandsJson !== parsed.operandsJson) {
        post(launch, htmlArtifactError(launch, parsed.request.requestId, new Error("This artifact reused a request id with different operands.")));
        return;
      }
      const promise = prior?.promise ?? api.request(parsed.request);
      if (!prior) relays.current.set(parsed.request.requestId, { operandsJson: parsed.operandsJson, promise });
      void promise.then((receipt) => {
        if (!active || !mounted.current || identityRef.current !== identity || launchRef.current?.viewId !== launch.viewId) return;
        setLastReceipt(receipt); replaceReview(receipt); post(launch, htmlArtifactResponse(launch, receipt));
      }, (failure) => { if (active) post(launch, htmlArtifactError(launch, parsed.request.requestId, failure)); });
    };
    window.addEventListener("message", receive);
    return () => {
      active = false; window.removeEventListener("message", receive); relays.current.clear();
      void api.close(launch.artifact.brainId, launch.viewId, launch.channelNonce).catch(() => undefined);
    };
  }, [api, identity, launch, post, replaceReview]);

  const decide = async (pending: HtmlArtifactToolReceipt, decision: "approve" | "deny") => {
    const owner = launchRef.current;
    if (!api || !owner || !pending.reviewId || reviewingSet.current.has(pending.requestId)) return;
    const ownerIdentity = identityRef.current, viewId = owner.viewId;
    reviewingSet.current.add(pending.requestId); setReviewing([...reviewingSet.current]); setError("");
    try {
      const receipt = await api.review({ viewId, channelNonce: owner.channelNonce, requestId: pending.requestId,
        reviewId: pending.reviewId, decision });
      if (!mounted.current || identityRef.current !== ownerIdentity || launchRef.current?.viewId !== viewId ||
          reviewMap.current.get(pending.requestId)?.reviewId !== pending.reviewId) return;
      const relay = relays.current.get(pending.requestId);
      if (relay) relays.current.set(pending.requestId, { operandsJson: relay.operandsJson, promise: Promise.resolve(receipt) });
      setLastReceipt(receipt); replaceReview(receipt); post(owner, htmlArtifactResponse(owner, receipt));
    } catch (failure) {
      if (mounted.current && identityRef.current === ownerIdentity && launchRef.current?.viewId === viewId) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    } finally {
      reviewingSet.current.delete(pending.requestId); if (mounted.current) setReviewing([...reviewingSet.current]);
    }
  };

  const activate = async () => {
    if (!launch || activating) return;
    const ownerIdentity = identityRef.current, ownerViewId = launch.viewId, ownerEpoch = openEpoch.current;
    const stillOwned = (): boolean => isCurrentHtmlArtifactOperation({ mounted: mounted.current,
      currentIdentity: identityRef.current, expectedIdentity: ownerIdentity,
      currentOpenEpoch: openEpoch.current, expectedOpenEpoch: ownerEpoch,
      currentViewId: launchRef.current?.viewId, expectedViewId: ownerViewId });
    setActivating(true); setError("");
    try {
      await api.activate({ ...request, expectedRevision: launch.artifact.revision, acknowledged: true });
      if (!stillOwned()) return;
      setActivating(false);
      launchRef.current = undefined; setLaunch(undefined); clearReviews(); await open();
    } catch (failure) { if (stillOwned()) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (stillOwned()) setActivating(false); }
  };

  return <section className="html-artifact-view" aria-label="Saved interactive HTML artifact">
    {opening && !launch ? <p role="status">Opening saved artifact…</p> : null}
    {launch ? <>
      <header><strong>{launch.artifact.title}</strong>
        <button type="button" className="button button--ghost" onClick={() => { launchRef.current = undefined; setLaunch(undefined); clearReviews(); }}>Close</button></header>
      {launch.artifact.operationalState === "dormant" && launch.artifact.interactive ? <div role="status">
        <p>Saved imported/copied interaction is dormant. Its exact presentation is visible, but scripts and host calls stay off until reviewed.</p>
        <button type="button" className="button" disabled={activating} onClick={() => void activate()}>{activating ? "Enabling reviewed interaction…" : "Enable this exact artifact interaction"}</button>
      </div> : null}
      <iframe ref={frame} className="html-artifact-view__frame" title={launch.artifact.title} src={launch.url}
        sandbox={launch.scriptEnabled ? "allow-scripts" : ""} referrerPolicy="no-referrer"
        allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'; display-capture 'none'"
        style={{ height: launch.height }} onLoad={() => post(launch, htmlArtifactHostReady(launch))} />
    </> : !opening ? <button type="button" className="button" onClick={() => void open()}>Reopen saved artifact</button> : null}
    {reviews.length ? <section aria-label="Pending HTML artifact reviews"><p role="status">{reviews.length} exact artifact {reviews.length === 1 ? "action awaits" : "actions await"} review.</p>
      {reviews.map((receipt) => <Review key={receipt.requestId} receipt={receipt} busy={reviewing.includes(receipt.requestId)} onDecision={(value) => void decide(receipt, value)} />)}
    </section> : null}
    {lastReceipt && lastReceipt.state !== "approval-required" ? <details><summary>Artifact host action · {lastReceipt.state}</summary>
      <p>{lastReceipt.toolId}.{lastReceipt.action} · {lastReceipt.executionId}</p>
      {lastReceipt.error ? <p role="status">{lastReceipt.error}</p> : null}</details> : null}
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
