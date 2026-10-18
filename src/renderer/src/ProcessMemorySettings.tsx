import { useEffect, useRef, useState } from "react";
import type { ProcessMemoryApi, ProcessMemoryPreferences } from "../../shared/processMemory";

export function ProcessMemorySettings({ brainId, api }: { brainId: string; api?: ProcessMemoryApi }) {
  return <ProcessMemorySettingsOwner key={brainId} brainId={brainId} api={api} />;
}
function ProcessMemorySettingsOwner({ brainId, api }: { brainId: string; api?: ProcessMemoryApi }) {
  const [preferences, setPreferences] = useState<ProcessMemoryPreferences>();
  const [busy, setBusy] = useState(false), [error, setError] = useState("");
  const owner = useRef<object | null>(null), pending = useRef(false);
  useEffect(() => {
    const scope = {}; owner.current = scope;
    if (api) void api.preferences(brainId).then(value => {
      if (owner.current === scope && value.brainId === brainId) setPreferences(value);
    }).catch(reason => { if (owner.current === scope) setError(reason instanceof Error ? reason.message : "Process-memory settings could not load."); });
    return () => { if (owner.current === scope) owner.current = null; };
  }, [api, brainId]);
  const update = async (enabled: boolean) => {
    if (!api || !preferences || !owner.current || pending.current) return;
    const scope = owner.current; pending.current = true; setBusy(true); setError("");
    try { const next = await api.update({ brainId, expectedRevision: preferences.revision, enabled });
      if (owner.current === scope && next.brainId === brainId) setPreferences(next);
    } catch (reason) {
      if (owner.current !== scope) return;
      setError(reason instanceof Error ? reason.message : "Process-memory settings were not saved.");
      // Never retry a rejected grant. Re-read current host authority instead.
      const current = await api.preferences(brainId).catch(() => undefined);
      if (owner.current === scope && current?.brainId === brainId) setPreferences(current);
    } finally { if (owner.current === scope) { pending.current = false; setBusy(false); } }
  };
  return <section aria-label="DMA process-memory setting" className="process-memory-settings">
    <label><input type="checkbox" checked={preferences?.enabled ?? false}
      disabled={!api || !preferences || busy || !preferences.capabilities.implemented}
      onChange={event => void update(event.target.checked)} />Enable DMA / process-memory access</label>
    <p>Allows permitted process inspection and modification. The <code>system.memory</code> tool’s access policy still applies.</p>
    {busy ? <p role="status">Saving process-memory setting…</p> : null}
    {!api ? <p>Available in the desktop app.</p> : null}
    {error ? <p role="alert">{error}</p> : null}
    <details><summary>OS access and resource considerations</summary>
      <p>Advanced access for capable machines. This is OS-permitted process memory, not unrestricted physical RAM. Protected programs and Omni’s own authority and neural processes cannot be changed through this tool.</p>
      {preferences ? <p>{preferences.capabilities.detail}</p> : null}
      <p>The switch and live target permissions belong to this installation; exports do not grant another machine access.</p>
    </details>
  </section>;
}
