import { useEffect, useState } from "react";
import type { LiveVoiceCapabilities } from "../../shared/liveVoice";
import { DEFAULT_INSTANCE_VOICE_SETTINGS, resolveInstanceVoice, type InstanceVoiceSettings, type InstanceVoiceSnapshot } from "../../shared/voiceRouting";
import "./voiceRouting.css";

export function VoiceRoutingControls({ snapshot, capabilities, callActive, utteranceActive, onSave }: {
  snapshot?: InstanceVoiceSnapshot; capabilities?: LiveVoiceCapabilities; callActive: boolean; utteranceActive: boolean;
  onSave(preferences: InstanceVoiceSettings): Promise<void>;
}) {
  const [draft, setDraft] = useState<InstanceVoiceSettings>(snapshot?.preferences ?? { ...DEFAULT_INSTANCE_VOICE_SETTINGS });
  const [busy, setBusy] = useState(false), [status, setStatus] = useState("");
  useEffect(() => { setDraft(snapshot?.preferences ?? { ...DEFAULT_INSTANCE_VOICE_SETTINGS }); setStatus(""); }, [snapshot?.brainId, snapshot?.revision]);
  if (!snapshot) return <p role="status">Reading this instance’s saved voice settings…</p>;
  const routesChanged = draft.input !== snapshot.preferences.input || draft.output !== snapshot.preferences.output;
  const dirty = routesChanged || draft.deliveryMode !== snapshot.preferences.deliveryMode || draft.pace !== snapshot.preferences.pace;
  const route = capabilities ? resolveInstanceVoice(draft, snapshot.parameterEvidence, capabilities) : undefined;
  return <form className="voice-routing" onSubmit={event => {
    event.preventDefault(); if (busy || !dirty || utteranceActive || callActive && routesChanged) return;
    setBusy(true); setStatus("Saving voice settings…");
    void onSave(draft).then(() => setStatus("Voice settings saved for this instance."), failure => setStatus(failure instanceof Error ? failure.message : String(failure))).finally(() => setBusy(false));
  }} aria-label="Instance voice settings">
    <div className="voice-routing__pair"><label>Input<select value={draft.input} disabled={busy || callActive} onChange={event => { setDraft({ ...draft, input: event.target.value as InstanceVoiceSettings["input"] }); setStatus(""); }}>
      <option value="auto">Automatic · platform recognition</option><option value="platform">Platform speech recognition</option><option value="native">This brain’s raw audio input</option>
    </select></label><label>Output<select value={draft.output} disabled={busy || callActive} onChange={event => { setDraft({ ...draft, output: event.target.value as InstanceVoiceSettings["output"] }); setStatus(""); }}>
      <option value="auto">Automatic · by measured size</option><option value="platform">Platform text-to-speech</option><option value="native">This brain’s audio output</option>
    </select></label></div>
    <div className="voice-routing__pair"><label>Delivery<select value={draft.deliveryMode} disabled={busy || utteranceActive} onChange={event => { setDraft({ ...draft, deliveryMode: event.target.value as InstanceVoiceSettings["deliveryMode"] }); setStatus(""); }}><option value="live">Live · interruptible</option><option value="buffered">Wait for the full reply</option></select></label>
      <label>Pace<select value={draft.pace} disabled={busy || utteranceActive} onChange={event => { setDraft({ ...draft, pace: event.target.value as InstanceVoiceSettings["pace"] }); setStatus(""); }}><option value="slow">Slow</option><option value="normal">Normal</option><option value="fast">Fast</option></select></label></div>
    {callActive ? <small>End the current call before changing its input or output route.</small> : null}
    <details><summary>Selected routes & availability</summary><p>Automatic output uses platform speech through 200B measured parameters and this brain’s audio above 200B. Native audio is never silently replaced.</p>
      <p>{snapshot.parameterDetail}</p>{route ? <><p>{route.input.detail}</p><p>{route.output.detail}</p></> : <p>Current audio capability has not been checked yet.</p>}
      {snapshot.inheritedVariants ? <p>{snapshot.inheritedVariants} parent preference record(s) remain in this instance’s history; they do not override these choices.</p> : null}</details>
    <button type="submit" className="button" disabled={busy || !dirty || utteranceActive || callActive && routesChanged}>{busy ? "Saving…" : "Save voice settings"}</button>
    {status ? <p role="status" aria-live="polite">{status}</p> : null}
  </form>;
}
