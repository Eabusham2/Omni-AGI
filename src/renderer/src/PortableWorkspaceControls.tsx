import { useEffect, useRef, useState } from "react";
import type { BrainSummary } from "../../shared/types";
import { checkedPortablePresentation, type PortableDestinationReview, type PortablePresentationSettings,
  type PortableWorkspaceApi, type PortableWorkspaceCreateResult, type PortableWorkspaceStatus } from "../../shared/portableWorkspace";
import "./portableWorkspace.css";

const message = (error: unknown) => error instanceof Error ? error.message : String(error);
const gib = (bytes: number) => `${(bytes / 1024 ** 3).toLocaleString(undefined, { maximumFractionDigits: 2 })} GiB`;

export function PortableWorkspaceControls({ api, summaries, presentation, onCreated, onEditSettings }: {
  api?: PortableWorkspaceApi; summaries: BrainSummary[]; presentation(): PortablePresentationSettings;
  onCreated(result: PortableWorkspaceCreateResult): void; onEditSettings(brainId: string): void;
}) {
  const [selected, setSelected] = useState("");
  const [status, setStatus] = useState<PortableWorkspaceStatus>();
  const [review, setReview] = useState<PortableDestinationReview>();
  const [busy, setBusy] = useState(""); const [error, setError] = useState("");
  const alive = useRef(false), generation = useRef(0), flight = useRef<object | undefined>(undefined);
  useEffect(() => {
    alive.current = true; flight.current = undefined; setBusy(""); setError("");
    setStatus(undefined); setReview(undefined); const owner = ++generation.current;
    if (api) void api.status().then(value => { if (alive.current && generation.current === owner) setStatus(value); })
      .catch(reason => { if (alive.current && generation.current === owner) setError(message(reason)); });
    return () => { alive.current = false; generation.current++; };
  }, [api]);
  if (!api) return null;
  const perform = async (label: string, action: (current: () => boolean) => Promise<void>) => {
    if (flight.current) return; const owner = generation.current, token = {}; flight.current = token;
    const current = () => alive.current && generation.current === owner && flight.current === token;
    setBusy(label); setError("");
    try { await action(current); }
    catch (reason) { if (alive.current && generation.current === owner) setError(message(reason)); }
    finally { if (current()) { flight.current = undefined; setBusy(""); } }
  };
  const acceptStatus = (value: PortableWorkspaceStatus, current: () => boolean) => { if (current()) { setStatus(value); setReview(undefined); } };
  const creating = () => perform("Creating portable folder…", async current => {
    const result = await api.create({ brainId: selected, requestId: crypto.randomUUID(), presentation: checkedPortablePresentation(presentation()) });
    if (result && current()) onCreated(result);
  });
  return <section className="portable-workspace" aria-label="Portable Windows workspace">
    <h2>Portable Windows workspace</h2>
    {status?.portable && status.state !== "ready" ? <div className="portable-workspace__destination">
      <p>{status.state === "preparation-required" ? "Prepare the saved files before reviewing this PC." : status.state === "preparing" ? status.label ?? "Preparing saved files…" : "Review this PC before loading the portable instance."}</p>
      {status.error ? <p role="alert">{status.error}</p> : null}
      <div className="portable-workspace__actions">
        {status.state === "preparation-required" ? <button type="button" className="button button--primary" disabled={Boolean(busy)}
          onClick={() => void perform("Preparing saved files…", async current => acceptStatus(await api.prepare({ requestId: crypto.randomUUID() }), current))}>Prepare saved files</button> : status.state === "preparing" ? null : <>
          <button type="button" className="button" disabled={Boolean(busy)} onClick={() => void perform("Checking destination…", async current => {
            const value = await api.reviewDestination(); if (current()) setReview(value);
          })}>Review destination</button>
          {status.brainId ? <button type="button" className="button button--ghost" disabled={Boolean(busy)} onClick={() => { setReview(undefined); onEditSettings(status.brainId!); }}>Edit resource settings</button> : null}
        </>}
      </div>
      {review ? <div className="portable-workspace__review">
        <dl><div><dt>RAM ceiling</dt><dd>{gib(review.plan.resources.systemRamBudgetBytes)}</dd></div>
          <div><dt>Shared storage pool</dt><dd>{gib(review.plan.resources.sharedStoragePoolBytes)}</dd></div>
          <div><dt>Context</dt><dd>{review.plan.context.selectedTokens.toLocaleString()} tokens</dd></div></dl>
        <p>Device credentials and copied live grants stay inactive. Set tool access separately in Tools & permissions.</p>
        {review.plan.blockers.length ? <ul role="alert">{review.plan.blockers.map((text, index) => <li key={index}>{text}</li>)}</ul> : null}
        {review.plan.warnings.length ? <details><summary>Resource warnings</summary><ul>{review.plan.warnings.map((text, index) => <li key={index}>{text}</li>)}</ul></details> : null}
        <details><summary>Exact review</summary><code>{review.reviewId}</code><code>{review.reviewSha256}</code></details>
        <button type="button" className="button button--primary" disabled={Boolean(busy) || !review.plan.allowed}
          onClick={() => void perform("Saving destination review…", async current => acceptStatus(await api.approveDestination({ reviewId: review.reviewId, reviewSha256: review.reviewSha256 }), current))}>Approve this resource fit</button>
      </div> : null}
    </div> : null}
    {status?.portable && status.state === "ready" ? <p role="status">This destination is ready. Load the instance when you want to use it.</p> : null}
    <details><summary>Create a portable copy</summary>
      <p>Copy an unloaded instance and ordinary settings into a Windows app folder.</p>
      <label>Instance<select value={selected} disabled={Boolean(busy)} onChange={event => setSelected(event.target.value)}>
        <option value="">Choose an instance</option>{summaries.map(brain => <option key={brain.id} value={brain.id}>{brain.name}</option>)}
      </select></label>
      <button type="button" className="button" disabled={Boolean(busy) || !summaries.some(brain => brain.id === selected)} onClick={() => void creating()}>Create portable folder</button>
    </details>
    {busy ? <p role="status">{busy}</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    <button type="button" className="button button--ghost" disabled={Boolean(busy)} onClick={() => void perform("Refreshing workspace…", async current => acceptStatus(await api.status(), current))}>Refresh workspace</button>
  </section>;
}
