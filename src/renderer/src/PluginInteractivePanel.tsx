import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import type { PluginsApi } from "../../shared/plugins";
import type {
  PluginInteractivePanelLaunch,
  PluginInteractiveToolReceipt
} from "../../shared/pluginInteractivePanels";
import {
  interactivePanelErrorResponse,
  interactivePanelHostReady,
  interactivePanelIdentity,
  interactivePanelRequestFromMessage,
  interactivePanelResponse,
  isCurrentInteractivePanelOperation,
  nextPendingInteractivePanelReviews
} from "./pluginInteractivePanelBridge";

function currentApi(): PluginsApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.plugins;
}

interface PendingRelay {
  operandsJson: string;
  promise: Promise<PluginInteractiveToolReceipt>;
}

function PanelReview({ receipt, busy, onReview }: {
  receipt: PluginInteractiveToolReceipt;
  busy: boolean;
  onReview(decision: "approve" | "deny"): void;
}) {
  const review = receipt.review;
  if (!review || !receipt.reviewId) return null;
  return <section className="plugin-approval-review" aria-label={`Interactive plugin operation review ${receipt.requestId}`}>
    <strong>{review.pluginName} requests {review.resolved.toolId}.{review.resolved.action}</strong>
    <pre tabIndex={0}>{JSON.stringify(review.resolved.arguments, null, 2)}</pre>
    <small>Panel request {receipt.requestId} · package {review.contentSha256.slice(0, 12)}… · both permissions are checked again before dispatch.</small>
    <div className="plugin-manager__actions">
      <button type="button" className="button button--primary" disabled={busy} onClick={() => onReview("approve")}>Approve exact action</button>
      <button type="button" className="button" disabled={busy} onClick={() => onReview("deny")}>Deny</button>
    </div>
  </section>;
}

/**
 * Trusted relay around one opaque scripted child. The child has no preload;
 * every requested effect returns through the main-owned panel handle.
 */
export interface PluginInteractivePanelProps {
  brainId: string;
  pluginId: string;
  contentSha256: string;
  panelId: string;
  title: string;
  api?: PluginsApi;
}

export function PluginInteractivePanel(props: PluginInteractivePanelProps) {
  // A new identity gets fresh local Open/review state. In particular, a late
  // Open for the previous package must not leave the next identity disabled.
  return <PluginInteractivePanelInstance key={interactivePanelIdentity(props)} {...props} />;
}

function PluginInteractivePanelInstance({ brainId, pluginId, contentSha256, panelId, title, api = currentApi() }: PluginInteractivePanelProps) {
  const identity = interactivePanelIdentity({ brainId, pluginId, contentSha256, panelId });
  const [launch, setLaunch] = useState<PluginInteractivePanelLaunch>();
  const [error, setError] = useState("");
  const [opening, setOpening] = useState(false);
  const [pendingReviews, setPendingReviews] = useState<PluginInteractiveToolReceipt[]>([]);
  const [lastReceipt, setLastReceipt] = useState<PluginInteractiveToolReceipt>();
  const [reviewingIds, setReviewingIds] = useState<string[]>([]);
  const frame = useRef<HTMLIFrameElement>(null);
  const relays = useRef(new Map<string, PendingRelay>());
  const mounted = useRef(false);
  const identityOwner = useRef(identity);
  const launchOwner = useRef<PluginInteractivePanelLaunch | undefined>(undefined);
  const openingOwner = useRef(false);
  const pendingOwner = useRef(new Map<string, PluginInteractiveToolReceipt>());
  const reviewingOwner = useRef(new Set<string>());
  identityOwner.current = identity;
  launchOwner.current = launch;

  useLayoutEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const replacePending = useCallback((receipt: PluginInteractiveToolReceipt): void => {
    const next = nextPendingInteractivePanelReviews([...pendingOwner.current.values()], receipt);
    pendingOwner.current = new Map(next.map((item) => [item.requestId, item]));
    if (mounted.current) setPendingReviews(next);
  }, []);

  const clearPending = useCallback((): void => {
    pendingOwner.current.clear();
    reviewingOwner.current.clear();
    if (mounted.current) {
      setPendingReviews([]);
      setReviewingIds([]);
    }
  }, []);

  const post = useCallback((owner: PluginInteractivePanelLaunch, value: Record<string, unknown>): void => {
    if (launchOwner.current?.viewId !== owner.viewId ||
        identityOwner.current !== interactivePanelIdentity(owner)) return;
    frame.current?.contentWindow?.postMessage(value, "*");
  }, []);

  useLayoutEffect(() => {
    if (!launch || interactivePanelIdentity(launch) === identity) return;
    launchOwner.current = undefined;
    clearPending();
    setLaunch(undefined);
  }, [clearPending, identity, launch]);

  useLayoutEffect(() => {
    if (!launch || !api || interactivePanelIdentity(launch) !== identity) return;
    let active = true;
    const receive = (event: MessageEvent): void => {
      const parsed = interactivePanelRequestFromMessage(event, frame.current?.contentWindow ?? null, launch);
      if (!parsed) return;
      const prior = relays.current.get(parsed.request.requestId);
      if (prior && prior.operandsJson !== parsed.operandsJson) {
        post(launch, interactivePanelErrorResponse(launch, parsed.request.requestId,
          new Error("This panel reused a request id with different operands.")));
        return;
      }
      const promise = prior?.promise ?? api.panelRequest(parsed.request);
      if (!prior) relays.current.set(parsed.request.requestId, { operandsJson: parsed.operandsJson, promise });
      void promise.then((receipt) => {
        if (!active || !isCurrentInteractivePanelOperation({ mounted: mounted.current,
          currentIdentity: identityOwner.current, expectedIdentity: identity,
          currentViewId: launchOwner.current?.viewId, expectedViewId: launch.viewId })) return;
        setLastReceipt(receipt);
        replacePending(receipt);
        post(launch, interactivePanelResponse(launch, receipt));
      }, (failure) => {
        if (active) post(launch, interactivePanelErrorResponse(launch, parsed.request.requestId, failure));
      });
    };
    window.addEventListener("message", receive);
    return () => {
      active = false;
      window.removeEventListener("message", receive);
      relays.current.clear();
      void api.closePanel(launch.brainId, launch.viewId, launch.channelNonce).catch(() => undefined);
    };
  }, [api, identity, launch, post, replacePending]);

  useEffect(() => {
    if (!api?.onEvent || !launch || interactivePanelIdentity(launch) !== identity) return;
    return api.onEvent((event) => {
      if (event.type === "semantic-learning" || event.pluginId !== launch.pluginId ||
          event.type === "instance" && event.brainId !== launch.brainId ||
          launchOwner.current?.viewId !== launch.viewId) return;
      setError("Plugin access changed. Close and reopen this panel after reviewing the current version and permissions.");
      launchOwner.current = undefined;
      setLaunch(undefined);
      clearPending();
    });
  }, [api, clearPending, identity, launch]);

  const open = async (): Promise<void> => {
    if (!api || openingOwner.current) return;
    const expectedIdentity = identity;
    openingOwner.current = true;
    setOpening(true); setError(""); clearPending(); setLastReceipt(undefined);
    let opened: PluginInteractivePanelLaunch | undefined;
    try {
      opened = await api.openPanel(brainId, pluginId, contentSha256, panelId);
      const responseMatches = opened.brainId === brainId && opened.pluginId === pluginId &&
        opened.contentSha256 === contentSha256 && opened.panelId === panelId;
      const stillOwned = isCurrentInteractivePanelOperation({ mounted: mounted.current,
        currentIdentity: identityOwner.current, expectedIdentity });
      if (!responseMatches || !stillOwned) {
        await api.closePanel(opened.brainId, opened.viewId, opened.channelNonce).catch(() => undefined);
        if (!responseMatches && stillOwned) {
          throw new Error("Interactive panel launch belongs to another reviewed package or brain.");
        }
        return;
      }
      launchOwner.current = opened;
      setLaunch(opened);
    } catch (failure) {
      if (isCurrentInteractivePanelOperation({ mounted: mounted.current,
        currentIdentity: identityOwner.current, expectedIdentity })) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    } finally {
      openingOwner.current = false;
      if (mounted.current && identityOwner.current === expectedIdentity) setOpening(false);
    }
  };

  const review = async (pending: PluginInteractiveToolReceipt, decision: "approve" | "deny"): Promise<void> => {
    const owner = launchOwner.current;
    if (!api || !owner || !pending.reviewId || reviewingOwner.current.has(pending.requestId)) return;
    const expectedIdentity = identityOwner.current;
    const expectedViewId = owner.viewId;
    reviewingOwner.current.add(pending.requestId);
    setReviewingIds([...reviewingOwner.current]);
    setError("");
    try {
      const receipt = await api.reviewPanelRequest({
        viewId: owner.viewId,
        channelNonce: owner.channelNonce,
        requestId: pending.requestId,
        reviewId: pending.reviewId,
        decision
      });
      const currentPending = pendingOwner.current.get(pending.requestId);
      if (!isCurrentInteractivePanelOperation({ mounted: mounted.current,
        currentIdentity: identityOwner.current, expectedIdentity,
        currentViewId: launchOwner.current?.viewId, expectedViewId }) ||
        currentPending?.reviewId !== pending.reviewId) return;
      const relay = relays.current.get(pending.requestId);
      if (relay) relays.current.set(pending.requestId, {
        operandsJson: relay.operandsJson,
        promise: Promise.resolve(receipt)
      });
      setLastReceipt(receipt);
      replacePending(receipt);
      post(owner, interactivePanelResponse(owner, receipt));
    } catch (failure) {
      if (isCurrentInteractivePanelOperation({ mounted: mounted.current,
        currentIdentity: identityOwner.current, expectedIdentity,
        currentViewId: launchOwner.current?.viewId, expectedViewId })) {
        setError(failure instanceof Error ? failure.message : String(failure));
      }
    } finally {
      reviewingOwner.current.delete(pending.requestId);
      if (mounted.current) setReviewingIds([...reviewingOwner.current]);
    }
  };

  if (!api) return <p role="status">Interactive plugin host is unavailable.</p>;
  return <section aria-label={`${title} interactive plugin panel`}>
    <p>Sandboxed package UI. Every host action still uses this brain’s current plugin and nested tool permissions.</p>
    {!launch || interactivePanelIdentity(launch) !== identity ?
      <button type="button" className="button" disabled={opening} onClick={() => void open()}>
        {opening ? "Opening reviewed panel…" : "Open interactive panel"}
      </button> : <>
        <button type="button" className="button button--ghost" onClick={() => {
          launchOwner.current = undefined;
          setLaunch(undefined);
          clearPending();
        }}>Close panel</button>
        <iframe ref={frame} title={`${title} · interactive plugin panel`} src={launch.url}
          sandbox="allow-scripts" referrerPolicy="no-referrer"
          allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'; display-capture 'none'"
          onLoad={() => post(launch, interactivePanelHostReady(launch))} />
      </>}
    {pendingReviews.length ? <section aria-label="Pending interactive plugin reviews">
      <p role="status">{pendingReviews.length} exact panel {pendingReviews.length === 1 ? "action awaits" : "actions await"} review.</p>
      {pendingReviews.map((pending) => <PanelReview key={pending.requestId} receipt={pending}
        busy={reviewingIds.includes(pending.requestId)} onReview={(decision) => void review(pending, decision)} />)}
    </section> : null}
    {lastReceipt && lastReceipt.state !== "approval-required" ? <details>
      <summary>Host action · {lastReceipt.state}</summary>
      <p>{lastReceipt.toolId}.{lastReceipt.action} · execution {lastReceipt.executionId}</p>
      <p>{lastReceipt.dispatchStarted ? "Dispatch started; the actual result is retained in the ordinary tool trace." : "No external dispatch was recorded."}</p>
      {lastReceipt.error ? <p role="status">{lastReceipt.error}</p> : null}
    </details> : null}
    {error ? <p role="alert">{error}</p> : null}
  </section>;
}
