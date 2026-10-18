import { useEffect, useRef, useState } from "react";
import type { RouterSpikeActivity, RouterSpikeFrame, RouterSpikeQuery } from "../../shared/routerActivity";
import { RouterActivityBrainMap, routerChannelIdentity } from "./RouterActivityBrainMap";
import "./routerSpikeView.css";

type Observer = (brainId: string, query: RouterSpikeQuery) => Promise<RouterSpikeActivity>;
interface Capture { id: string; start: number; count: number; epoch?: string; afterSequence: number;
  contactOffset: number; contactPageSize: number; transmissionEpoch?: string;
  afterTransmissionSequence: number; transmissionPageSize: number; contactRevision: number; stopped: boolean }
export { routerChannelIdentity };
/** Retain actual last observations, never turn an empty transport poll into a
 * fabricated silent neural step. Raw cursors still advance only from the RPC. */
export function retainRouterObservation(previous: RouterSpikeActivity | undefined, incoming: RouterSpikeActivity): RouterSpikeActivity {
  if (!previous || !incoming.available || !incoming.enabled || previous.brainId !== incoming.brainId ||
      previous.captureId !== incoming.captureId || previous.epoch !== incoming.epoch ||
      previous.channelStart !== incoming.channelStart || previous.admittedCount !== incoming.admittedCount) return incoming;
  const latest = previous.frames.at(-1);
  const before = previous.transmissionPage, next = incoming.transmissionPage;
  const retainTransmissions = next?.available && next.enabled && before?.available && before.enabled &&
    next.captureEpoch === before.captureEpoch && next.epoch === before.epoch && next.layoutDigest === before.layoutDigest &&
    next.selectedContactOffset === before.selectedContactOffset && next.selectedContactCount === before.selectedContactCount &&
    next.events.length === 0;
  return { ...incoming, frames: incoming.frames.length || !latest ? incoming.frames : [latest],
    ...(retainTransmissions ? { transmissionPage: { ...next, events: before.events } } : {}) };
}
export function checkedRouterViewport(value: RouterSpikeActivity, brainId: string, capture: Pick<Capture, "id" | "epoch" | "afterSequence">): RouterSpikeActivity {
  if (value.brainId !== brainId || value.captureId !== capture.id || value.source !== "actual-lif-hard-spikes" || value.owner !== "root.router.population"
    || value.semanticNeuronMapping !== false || value.synapseFiringObserved !== false || (value.available && capture.epoch && value.epoch !== capture.epoch)) throw new Error("The router observer owner changed.");
  let sequence = capture.afterSequence;
  for (const frame of value.frames) {
    if (frame.sequence <= sequence || frame.channelStart !== value.channelStart || frame.spikes.length !== value.admittedCount || frame.spikes.some(spike => spike !== 0 && spike !== 1)) throw new Error("The router observer returned an invalid frame.");
    sequence = frame.sequence;
  }
  return value;
}

/** Reads an existing LIF observer only; it cannot Load or create a brain. */
export function RouterSpikeView({ brainId, observe, showMap = true, onObservation }: { brainId: string; observe?: Observer;
  showMap?: boolean; onObservation?(activity: RouterSpikeActivity | undefined): void }) {
  return <RouterSpikeOwner key={brainId} brainId={brainId} observe={observe} showMap={showMap} onObservation={onObservation} />;
}
function RouterSpikeOwner({ brainId, observe, showMap, onObservation }: { brainId: string; observe?: Observer;
  showMap: boolean; onObservation?(activity: RouterSpikeActivity | undefined): void }) {
  const [start, setStart] = useState("0"), [count, setCount] = useState("64");
  const [running, setRunning] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const [activity, setActivity] = useState<RouterSpikeActivity>(), [frame, setFrame] = useState<RouterSpikeFrame>();
  const [selected, setSelected] = useState<number>();
  const owner = useRef<object | null>(null), capture = useRef<Capture | undefined>(undefined);
  const pollTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => { onObservation?.(activity); }, [activity, onObservation]);
  const stopExact = async (value: Capture) => {
    if (!observe) return;
    // A missing epoch means first configuration is still in flight. Its owner
    // performs this exact stop immediately after that receipt arrives.
    if (value.epoch) await observe(brainId, { enabled: false, captureId: value.id, epoch: value.epoch, afterSequence: value.afterSequence });
  };
  useEffect(() => {
    const scope = {}; owner.current = scope;
    return () => {
      owner.current = null; if (pollTimer.current) clearTimeout(pollTimer.current);
      const current = capture.current; if (current) { current.stopped = true; void stopExact(current).catch(() => undefined); }
    };
  }, [brainId, observe]);

  const poll = async (current: Capture, scope: object, configure = false) => {
    if (!observe || current.stopped || owner.current !== scope || capture.current !== current) return;
    const contactRevision = current.contactRevision;
    try {
      const value = checkedRouterViewport(await observe(brainId, { enabled: true, captureId: current.id, start: current.start, count: current.count,
        ...(configure ? {} : { epoch: current.epoch, afterSequence: current.afterSequence }), pageSize: 32,
        contactOffset: current.contactOffset, contactPageSize: current.contactPageSize,
        ...(!configure && current.transmissionEpoch ? { transmissionEpoch: current.transmissionEpoch,
          afterTransmissionSequence: current.afterTransmissionSequence } : {}),
        transmissionPageSize: current.transmissionPageSize }), brainId, current);
      if (value.epoch) current.epoch = value.epoch;
      if (current.stopped || owner.current !== scope || capture.current !== current) { await stopExact(current); return; }
      if (current.contactRevision !== contactRevision) {
        pollTimer.current = setTimeout(() => { void poll(current, scope); }, 0); return;
      }
      setBusy(false); setActivity(previous => retainRouterObservation(previous, value));
      if (!value.available || !value.enabled || !value.epoch) { setRunning(false); setFrame(undefined); current.stopped = true; return; }
      current.afterSequence = value.nextSequence;
      if (value.transmissionPage?.epoch) {
        current.transmissionEpoch = value.transmissionPage.epoch;
        current.afterTransmissionSequence = value.transmissionPage.nextSequence;
      }
      const latest = value.frames.at(-1); if (latest) setFrame(latest);
      pollTimer.current = setTimeout(() => { void poll(current, scope); }, value.hasMore || value.transmissionPage?.hasMore ? 0 : 250);
    } catch (reason) {
      if (owner.current === scope && capture.current === current && !current.stopped) {
        current.stopped = true; setRunning(false); setBusy(false); setFrame(undefined); setActivity(undefined); onObservation?.(undefined);
        setError(reason instanceof Error ? reason.message : "Router observation stopped."); void stopExact(current).catch(() => undefined);
      }
    }
  };
  const begin = () => {
    if (!observe || running || busy || !owner.current) return;
    const first = Number(start), size = Number(count);
    if (!/^\d+$/.test(start) || !Number.isSafeInteger(first) || first < 0 || !/^\d+$/.test(count) || !Number.isSafeInteger(size) || size < 1 || size > 256) {
      setError("Choose a nonnegative channel offset and a viewport of 1–256 channels."); return;
    }
    const current: Capture = { id: crypto.randomUUID(), start: first, count: size, afterSequence: 0,
      contactOffset: 0, contactPageSize: 128, afterTransmissionSequence: 0,
      transmissionPageSize: 128, contactRevision: 0, stopped: false };
    capture.current = current; setActivity(undefined); setFrame(undefined); setSelected(undefined); setError(""); setRunning(true); setBusy(true);
    void poll(current, owner.current, true);
  };
  const stop = async () => {
    const current = capture.current, scope = owner.current; if (!current || !scope) return;
    current.stopped = true; if (pollTimer.current) clearTimeout(pollTimer.current); setBusy(true); setFrame(undefined); onObservation?.(undefined);
    try { await stopExact(current); }
    catch (reason) { if (owner.current === scope) setError(reason instanceof Error ? reason.message : "The observer stop was not acknowledged."); }
    finally { if (owner.current === scope) { setBusy(false); setRunning(false); } }
  };
  const total = activity?.totalChannels, lastIndex = frame ? frame.channelStart + frame.spikes.length : activity ? activity.channelStart + activity.admittedCount : 0;
  const moveContacts = (offset: number) => {
    const current = capture.current; if (!current || current.stopped) return;
    current.contactOffset = offset; current.contactRevision++; current.transmissionEpoch = undefined; current.afterTransmissionSequence = 0;
    setActivity(undefined); onObservation?.(undefined);
  };
  return <section className="router-spike-view surface" aria-label="Actual spiking router activity">
    <div><h2>Spiking router</h2><p>Exact LIF channels. Connections to idea assemblies are not mapped here.</p></div>
    <div className="router-spike-view__controls">
      <label>First channel<input aria-label="First router channel" inputMode="numeric" value={start} disabled={running || busy} onChange={event => setStart(event.target.value)} /></label>
      <label>Channels on page<input aria-label="Router channels on page" inputMode="numeric" value={count} disabled={running || busy} onChange={event => setCount(event.target.value)} /></label>
      {running ? <button type="button" className="button" onClick={() => void stop()} disabled={busy && Boolean(capture.current?.stopped)}>Stop observing</button>
        : <button type="button" className="button button--primary" onClick={begin} disabled={!observe || busy}>Observe loaded brain</button>}
    </div>
    {error ? <p role="alert">{error}</p> : null}
    <p role="status">{busy ? capture.current?.stopped ? "Stopping observer…" : "Configuring observer…" : activity?.available === false ? activity.reason ?? "No loaded router is available." : running ? frame ? "Latest recorded spike frame" : "Waiting for an actual LIF step…" : "Observation off"}</p>
    {activity ? <p className="router-spike-view__scope">{activity.admittedCount ? `${activity.channelStart.toLocaleString()}–${(lastIndex - 1).toLocaleString()}` : "No admitted channels"} of {total?.toLocaleString() ?? "unknown"} channels · {activity.droppedFrames.toLocaleString()} ring frames overwritten · {activity.missedFrames.toLocaleString()} frames missed by this reader</p> : null}
    {frame ? <>
      <div className="router-spike-view__channels" role="group" aria-label="Latest measured router frame">
        {frame.spikes.map((spike, index) => { const channel = frame.channelStart + index; return <button type="button" key={channel} data-spike={spike}
          aria-pressed={selected === channel} onClick={() => setSelected(channel)} title={`${routerChannelIdentity(brainId, channel)} · measured spike ${spike}`}
          aria-label={`Channel ${channel}: ${spike === 1 ? "spiked" : "no spike"}`}>{channel.toLocaleString()}</button>; })}
      </div>
      <p className="router-spike-view__scope">Frame {frame.sequence.toLocaleString()} · {new Date(frame.observedAtUnix * 1000).toLocaleString()} · lit channels emitted a measured spike.</p>
      {selected !== undefined ? <code className="router-spike-view__identity">{routerChannelIdentity(brainId, selected)}</code> : null}
      <details><summary>Observer details</summary><dl>
        <div><dt>Exact monotonic timestamp (ns)</dt><dd>{frame.monotonicNs}</dd></div>
        <div><dt>Ring allocation</dt><dd>{activity?.ringBytes.toLocaleString()} bytes</dd></div>
        <div><dt>Epoch</dt><dd>{activity?.epoch}</dd></div>
      </dl></details>
    </> : null}
    {showMap && activity?.available ? <RouterActivityBrainMap activity={activity} onContactOffset={running ? moveContacts : undefined} /> : null}
    {!showMap && activity?.contactSnapshot?.available ? <div className="router-spike-view__controls">
      <span>{activity.contactSnapshot.returned ? `${activity.contactSnapshot.offset + 1}–${activity.contactSnapshot.offset + activity.contactSnapshot.returned}` : "0"} of {activity.contactSnapshot.totalContacts?.toLocaleString() ?? "unknown"} captured contacts</span>
      <button type="button" className="button" disabled={!running || activity.contactSnapshot.offset === 0}
        onClick={() => moveContacts(Math.max(0, activity.contactSnapshot!.offset - activity.contactSnapshot!.pageSize))}>Previous contacts</button>
      <button type="button" className="button" disabled={!running || !activity.contactSnapshot.hasMore}
        onClick={() => moveContacts(activity.contactSnapshot!.offset + activity.contactSnapshot!.returned)}>Next contacts</button>
    </div> : null}
    {total !== null && total !== undefined ? <div className="router-spike-view__controls">
      <button type="button" className="button" disabled={running || busy || Number(start) === 0} onClick={() => setStart(String(Math.max(0, Number(start) - Number(count))))}>Previous channels</button>
      <button type="button" className="button" disabled={running || busy || lastIndex >= total} onClick={() => setStart(String(lastIndex))}>Next channels</button>
    </div> : null}
  </section>;
}
