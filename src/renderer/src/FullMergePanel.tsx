import React, { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_CONFIG, type BrainConfig, type BrainSummary, type HardwareTier, type OmniApi, type ToolPermissionLevel } from "../../shared/types";
import type { FullMergeProgress, FullMergeStartRequest } from "../../shared/fullMerge";
import { DEFAULT_DEVICE_RESOURCE_SETTINGS } from "../../shared/deviceResources";
import "./fullMerge.css";

const TOOLS = [
  ["system.files", "System files"], ["system.shell", "System shell"], ["code.execute", "Code runner"],
  ["web.search", "Web search"], ["web.fetch", "Web fetch"], ["browser.automation", "Browser automation"],
  ["device.input", "Device input"], ["modality.imagine", "Imagination"], ["brain.history", "Conversation history"],
  ["agent.fork", "Subagent fork"], ["source.self-modify", "Source evolution"]
] as const;
const PHASE: Record<FullMergeProgress["state"], string> = {
  preparing: "Preparing parent checkpoints", copying: "Copying complete snapshots", assembling: "Connecting native regions",
  verifying: "Verifying saved state", paused: "Paused", complete: "Complete", cancelled: "Stopped — incomplete output removed"
};
const bytes = (value: number): string => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GiB`
  : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(1)} MiB` : `${value.toLocaleString()} bytes`;
const serial = (config: BrainConfig): string => JSON.stringify(config, null, 2);

export function reviewedMergeConfig(value: unknown): BrainConfig {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Settings must be a JSON object.");
  const config = value as BrainConfig;
  if (config.nativeArchitecture) throw new Error("Parent neural architecture is preserved; remove the single-stack architecture override.");
  if (config.idleCognition !== false) throw new Error("Start Active mode separately after the new instance is loaded.");
  if (!config.name?.trim() || config.runtime !== "adaptive-core") throw new Error("Choose a name and the native adaptive-core runtime.");
  if (!Number.isSafeInteger(config.workingMemorySlots) || config.workingMemorySlots < 1 ||
    !Number.isSafeInteger(config.contextWindowTokens) || config.contextWindowTokens < 8 ||
    !Number.isSafeInteger(config.memoryResidentItems) || config.memoryResidentItems < 1 || config.memoryResidentItems > config.workingMemorySlots) {
    throw new Error("Review the context and resident working-memory capacities.");
  }
  if (config.systemRamMode === "manual" && (!Number.isFinite(config.systemRamSharePercent) || config.systemRamSharePercent < 30 || config.systemRamSharePercent > 100)) throw new Error("Manual RAM must be 30–100% of the safe pool.");
  if (!Number.isSafeInteger(config.storagePoolBytes) || config.storagePoolBytes < 0 || config.storagePoolMode === "manual" && config.storagePoolBytes === 0) throw new Error("A manual storage pool needs a positive byte size.");
  return { ...config, name: config.name.trim() };
}

export function FullMergeReceipt({ operation, pending, onCommand, onOpen }: {
  operation: FullMergeProgress; pending?: boolean;
  onCommand: (operationId: string, command: "pause" | "resume" | "cancel") => void;
  onOpen?: (id: string) => void;
}) {
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const active = !["complete", "cancelled", "paused"].includes(operation.state);
  return <article className="full-merge-receipt" aria-label={`Merge ${operation.name ?? "new instance"}`}>
    <div className="full-merge-receipt__heading"><strong>{operation.name ?? "Merged instance"}</strong><span role="status">{PHASE[operation.state]}</span></div>
    <p>{operation.parents.map((parent) => parent.name).join(" + ") || "Selected parent instances"}</p>
    {operation.state === "copying" ? <progress aria-label="Parent snapshot files copied" value={operation.filesCopied} max={Math.max(1, operation.filesDiscovered)} />
      : active ? <progress aria-label={PHASE[operation.state]} /> : null}
    <p className="full-merge-muted">{operation.filesDiscovered > 0
      ? `${operation.filesCopied.toLocaleString()} of ${operation.filesDiscovered.toLocaleString()} snapshot files · ${bytes(operation.copiedBytes)} captured`
      : operation.state === "cancelled" ? "Stopped before copying snapshot files."
        : "Parent inventory has not finished yet."}</p>
    {operation.error && operation.state !== "cancelled" ? <p className="full-merge-error">{operation.error}</p> : null}
    <details><summary>Reviewed settings and receipt</summary>
      <p>Operation: <code>{operation.operationId}</code></p><p>Destination: <code>{operation.targetBrainId}</code></p>
      <p>Parent instances are preserved. Stored history is retained; old approvals and credentials are not inherited.</p>
      {operation.reviewedSettings ? <pre>{serial(operation.reviewedSettings)}</pre> : null}
      {operation.reviewedAccess ? <pre>{JSON.stringify(operation.reviewedAccess, null, 2)}</pre> : null}
    </details>
    <div className="full-merge-actions">
      {active ? <button type="button" className="button" disabled={pending} onClick={() => onCommand(operation.operationId, "pause")}>Pause</button> : null}
      {operation.state === "paused" ? <button type="button" className="button button--primary" disabled={pending} onClick={() => onCommand(operation.operationId, "resume")}>Resume same merge</button> : null}
      {operation.state !== "complete" && operation.state !== "cancelled" ? <button type="button" className="button" disabled={pending} onClick={() => setConfirmDiscard(true)}>Stop &amp; discard…</button> : null}
      {operation.state === "complete" && onOpen ? <button type="button" className="button button--primary" onClick={() => onOpen(operation.targetBrainId)}>Open merged instance</button> : null}
    </div>
    {confirmDiscard && !["complete", "cancelled"].includes(operation.state) ? <div className="full-merge-confirm" role="group" aria-label="Confirm stopping this merge">
      <p>Discard only this incomplete merged output? All parent instances remain unchanged.</p>
      <button type="button" className="button" disabled={pending} onClick={() => { setConfirmDiscard(false); onCommand(operation.operationId, "cancel"); }}>Discard incomplete output</button>
      <button type="button" className="button" onClick={() => setConfirmDiscard(false)}>Keep it</button>
    </div> : null}
  </article>;
}

export function FullMergePanel({ summaries, api, onPublished, onOpen, initialOperations = [] }: {
  summaries: BrainSummary[]; api?: OmniApi; onPublished?: () => void; onOpen?: (id: string) => void;
  initialOperations?: FullMergeProgress[];
}) {
  const [expanded, setExpanded] = useState(false);
  const [parents, setParents] = useState<string[]>([]);
  const [config, setConfig] = useState<BrainConfig>(() => ({ ...DEFAULT_CONFIG, name: "Merged instance", idleCognition: false }));
  const [json, setJson] = useState(() => serial({ ...DEFAULT_CONFIG, name: "Merged instance", idleCognition: false }));
  const [tier, setTier] = useState<HardwareTier>("personal");
  const [permissions, setPermissions] = useState<Record<string, ToolPermissionLevel>>(() => Object.fromEntries(TOOLS.map(([id]) => [id, "off"])));
  const [operations, setOperations] = useState(initialOperations);
  const [creating, setCreating] = useState(false);
  const [loadingSettings, setLoadingSettings] = useState(false);
  const [commands, setCommands] = useState<Set<string>>(() => new Set());
  const [reviewed, setReviewed] = useState(false);
  const [error, setError] = useState("");
  const [spillSelection, setSpillSelection] = useState<{ selectionId: string; directories: string[] }>();
  const alive = useRef(true), submitted = useRef(false), completed = useRef(new Set<string>()), resuming = useRef(new Set<string>());
  const callback = useRef(onPublished); callback.current = onPublished;
  const capable = Boolean(api?.brain.startFullMerge);
  const upsert = useCallback((operation: FullMergeProgress): void => {
    if (!alive.current) return;
    setOperations((current) => [operation, ...current.filter((entry) => entry.operationId !== operation.operationId)]);
    if (operation.state !== "paused" && resuming.current.delete(operation.operationId)) {
      setCommands((current) => { const next = new Set(current); next.delete(operation.operationId); return next; });
    }
    if (operation.state === "complete" && !completed.current.has(operation.operationId)) {
      completed.current.add(operation.operationId); callback.current?.();
    }
  }, []);
  useEffect(() => {
    alive.current = true;
    if (!api?.brain.listFullMerges || !api.brain.onFullMerge) return () => { alive.current = false; };
    let cancelled = false;
    const unsubscribe = api.brain.onFullMerge(upsert);
    void api.brain.listFullMerges().then((values) => {
      if (cancelled) return;
      for (const value of values) if (value.state === "complete") completed.current.add(value.operationId);
      setOperations((current) => [...current, ...values.filter((value) => !current.some((entry) => entry.operationId === value.operationId))]);
    }).catch((reason: unknown) => { if (!cancelled) setError(reason instanceof Error ? reason.message : "Merge receipts could not be loaded."); });
    return () => { cancelled = true; alive.current = false; unsubscribe(); };
  }, [api, upsert]);
  const update = (next: BrainConfig): void => { setConfig(next); setJson(serial(next)); setReviewed(false); setError(""); };
  const loadParentSettings = async (): Promise<void> => {
    if (!api || !parents[0] || loadingSettings) return;
    setLoadingSettings(true); setError("");
    try {
      const brain = await api.brain.get(parents[0]);
      if (!alive.current) return;
      const { nativeArchitecture, deviceResources: _grants, ...base } = brain.config;
      setTier(nativeArchitecture?.hardwareTier ?? "personal"); setSpillSelection(undefined);
      update({ ...DEFAULT_CONFIG, ...base, name: config.name, idleCognition: false });
      // Labels may be imported; authority always starts Off for this new identity.
      setPermissions((current) => ({ ...Object.fromEntries((brain.toolPermissions ?? []).map((entry) => [entry.toolId, "off" as const])), ...current }));
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : "Parent settings could not be loaded."); }
    finally { if (alive.current) setLoadingSettings(false); }
  };
  const chooseDrives = async (): Promise<void> => {
    if (!api) return;
    try {
      const chosen = await api.catalog.selectSpillPools();
      if (!chosen || !alive.current) return;
      setSpillSelection(chosen);
      update({ ...config, deviceResources: { ...(config.deviceResources ?? DEFAULT_DEVICE_RESOURCE_SETTINGS), storageDirectories: chosen.directories } });
    } catch (reason) { if (alive.current) setError(reason instanceof Error ? reason.message : "Drive selection failed."); }
  };
  const start = async (event: React.FormEvent): Promise<void> => {
    event.preventDefault(); if (!api || submitted.current || parents.length < 2 || !reviewed) return;
    setError("");
    let operationId: string | undefined;
    try {
      if (json !== serial(config)) throw new Error("Apply the edited settings JSON before starting.");
      const selected = reviewedMergeConfig(config);
      if (selected.deviceResources?.storageDirectories.length && JSON.stringify(selected.deviceResources.storageDirectories) !== JSON.stringify(spillSelection?.directories)) throw new Error("Choose the reviewed destination drives with the native picker.");
      const request: FullMergeStartRequest = { operationId: crypto.randomUUID(), parentIds: [...parents], name: selected.name,
        config: selected, hardwareTier: tier, accessPolicy: { credentials: "none", mcp: "disabled",
          toolPermissions: Object.entries(permissions).map(([toolId, level]) => ({ toolId, level })) },
        ...(spillSelection ? { spillPoolSelectionId: spillSelection.selectionId } : {}) };
      operationId = request.operationId;
      submitted.current = true; setCreating(true); setExpanded(false);
      upsert(await api.brain.startFullMerge(request)); if (alive.current) setReviewed(false);
    } catch (reason) {
      const receipt = operationId ? await api.brain.fullMergeStatus(operationId).catch(() => undefined) : undefined;
      if (receipt) upsert(receipt);
      else if (alive.current) setError(reason instanceof Error ? reason.message : "The merge paused before completion.");
    }
    finally { submitted.current = false; if (alive.current) { setCreating(false); setReviewed(false); } }
  };
  const command = async (id: string, action: "pause" | "resume" | "cancel"): Promise<void> => {
    if (!api || commands.has(id)) return; setError("");
    setCommands((current) => new Set(current).add(id));
    if (action === "resume") resuming.current.add(id);
    try {
      upsert(await (action === "pause" ? api.brain.pauseFullMerge(id) : action === "resume" ? api.brain.resumeFullMerge(id) : api.brain.cancelFullMerge(id)));
    } catch (reason) {
      const receipt = await api.brain.fullMergeStatus(id).catch(() => undefined);
      if (receipt) upsert(receipt);
      else if (alive.current) setError(reason instanceof Error ? reason.message : "The merge could not continue.");
    }
    finally { resuming.current.delete(id); if (alive.current) setCommands((current) => { const next = new Set(current); next.delete(id); return next; }); }
  };
  return <section className="full-merge-panel surface" aria-labelledby="full-merge-heading">
    <div className="full-merge-panel__heading"><div><h2 id="full-merge-heading">Merge native instances</h2><p>Create one connected instance from two or more parents. Originals stay in your library.</p></div>
      <button type="button" className="button" aria-expanded={expanded} disabled={!capable || creating || summaries.length < 2} onClick={() => setExpanded((value) => !value)}>{expanded ? "Close review" : "Review a new merge"}</button>
    </div>
    {!capable ? <p className="full-merge-muted">Full merging requires the desktop native runtime.</p> : summaries.length < 2 ? <p className="full-merge-muted">Select this after at least two native instances are available.</p> : null}
    {expanded ? <form onSubmit={(event) => void start(event)}>
      <fieldset disabled={creating || loadingSettings}><legend>Parent instances</legend><div className="full-merge-parent-list">
        {summaries.map((brain) => <label key={brain.id}><input type="checkbox" checked={parents.includes(brain.id)} onChange={(event) => { setParents((current) => event.target.checked ? [...current, brain.id] : current.filter((id) => id !== brain.id)); setReviewed(false); }} /><span>{brain.name}</span></label>)}
      </div><p>{parents.length} selected · at least two required</p>
        <button type="button" className="button" disabled={!parents[0] || loadingSettings} onClick={() => void loadParentSettings()}>{loadingSettings ? "Reading settings…" : "Use first selected parent's settings"}</button>
      </fieldset>
      <fieldset disabled={creating || loadingSettings}><legend>New instance settings</legend><div className="full-merge-fields">
        <label>Name<input required maxLength={120} value={config.name} onChange={(event) => update({ ...config, name: event.target.value })} /></label>
        <label>Hardware policy<select value={tier} onChange={(event) => { setTier(event.target.value as HardwareTier); setReviewed(false); }}>{["micro", "personal", "gpu", "workstation"].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
        <label className="full-merge-wide">Description<textarea rows={2} value={config.description} onChange={(event) => update({ ...config, description: event.target.value })} /></label>
        <label>Context tokens<input type="number" min={8} step={1} value={config.contextWindowTokens} onChange={(event) => update({ ...config, contextWindowTokens: Number(event.target.value) })} /></label>
        <label>Working patterns<input type="number" min={1} step={1} value={config.workingMemorySlots} onChange={(event) => update({ ...config, workingMemorySlots: Number(event.target.value), memoryResidentItems: Math.min(config.memoryResidentItems, Number(event.target.value)) })} /></label>
        <label>Resident working patterns<input type="number" min={1} max={config.workingMemorySlots} step={1} value={config.memoryResidentItems} onChange={(event) => update({ ...config, memoryResidentItems: Number(event.target.value) })} /></label>
        <label>Working-memory policy<select value={config.workingMemoryMode} onChange={(event) => update({ ...config, workingMemoryMode: event.target.value as BrainConfig["workingMemoryMode"], extendedWorkingMemory: event.target.value === "extended" })}>{["auto", "extended", "manual"].map((value) => <option key={value}>{value}</option>)}</select></label>
        <label>RAM selection<select value={config.systemRamMode} onChange={(event) => update({ ...config, systemRamMode: event.target.value as BrainConfig["systemRamMode"], systemRamSharePercent: event.target.value === "auto" ? 0 : 65 })}><option value="auto">Auto</option><option value="manual">Manual</option></select></label>
        {config.systemRamMode === "manual" ? <label>Safe RAM pool %<input type="number" min={30} max={100} value={config.systemRamSharePercent} onChange={(event) => update({ ...config, systemRamSharePercent: Number(event.target.value) })} /></label> : null}
        <label>Storage pool<select value={config.storagePoolMode} onChange={(event) => update({ ...config, storagePoolMode: event.target.value as BrainConfig["storagePoolMode"], storagePoolBytes: event.target.value === "manual" ? Math.max(1024 ** 3, config.storagePoolBytes) : config.storagePoolBytes })}><option value="auto">Auto</option><option value="manual">Manual</option></select></label>
        {config.storagePoolMode === "manual" ? <label>Storage reservation (GiB)<input type="number" min={0.01} step="any" value={config.storagePoolBytes / 1024 ** 3} onChange={(event) => update({ ...config, storagePoolBytes: Math.round(Number(event.target.value) * 1024 ** 3) })} /></label> : null}
        <label><input type="checkbox" checked={config.onlineLearning} onChange={(event) => update({ ...config, onlineLearning: event.target.checked })} /> Background parameter learning</label>
        <label><input type="checkbox" checked={config.recursiveImprovement} onChange={(event) => update({ ...config, recursiveImprovement: event.target.checked })} /> Recursive improvement</label>
        <label><input type="checkbox" checked={config.bestWorkEnabled === true} onChange={(event) => update({ ...config, bestWorkEnabled: event.target.checked })} /> Best Work inference</label>
        <label>Maximum extra candidates<input type="number" min={1} max={16} value={config.bestWorkExplorationBranches ?? 16} onChange={(event) => update({ ...config, bestWorkExplorationBranches: Number(event.target.value) })} /></label>
      </div>
      <p className="full-merge-muted">Native geometry comes from the parents. Stored parent windows must fit the reviewed capacity; actual native allocation is checked before publication. Active mode starts separately after loading.</p>
      <button type="button" className="button" onClick={() => void chooseDrives()}>Choose destination spill drives…</button>
      {spillSelection ? <p>{spillSelection.directories.join(" · ")}</p> : <p className="full-merge-muted">No external drive authority inherited.</p>}
      <details><summary>All settings — complete JSON review</summary><p>Includes learning rate, history retention, trace detail, placement and stored performance settings. Changing future retention does not remove any parent's saved data.</p>
        <label>Destination configuration<textarea className="full-merge-json" spellCheck={false} rows={14} value={json} onChange={(event) => { setJson(event.target.value); setReviewed(false); }} /></label>
        <button type="button" className="button" onClick={() => { try { update(reviewedMergeConfig(JSON.parse(json))); } catch (reason) { setError(reason instanceof Error ? reason.message : "Invalid settings JSON."); } }}>Apply edited JSON</button>
      </details></fieldset>
      <fieldset disabled={creating}><legend>Destination access policy</legend><p>MCP and credentials start disconnected. Each tool below defaults to Off; existing parent grants and approval receipts are not inherited.</p>
        <div className="full-merge-fields">{Object.entries(permissions).map(([id, level]) => <label key={id}>{TOOLS.find(([tool]) => tool === id)?.[1] ?? id}<select aria-label={TOOLS.find(([tool]) => tool === id)?.[1] ?? id} value={level} onChange={(event) => { setPermissions((current) => ({ ...current, [id]: event.target.value as ToolPermissionLevel })); setReviewed(false); }}><option value="off">Off</option><option value="ask">Ask every time</option><option value="auto">Automatic policy</option><option value="full">Full access</option></select></label>)}</div>
      </fieldset>
      <label className="full-merge-review"><input type="checkbox" checked={reviewed} onChange={(event) => setReviewed(event.target.checked)} /> I reviewed the new instance's settings and access policy. The selected parents will be preserved.</label>
      <button type="submit" className="button button--primary" disabled={creating || parents.length < 2 || !reviewed || json !== serial(config)}>Create merged instance</button>
    </form> : null}
    {creating ? <p role="status">Merge running. Progress and safe-boundary controls are below; you can continue using the library.</p> : null}
    {error ? <p className="full-merge-error" role="alert">{error}</p> : null}
    <div className="full-merge-receipts">{operations.map((operation) => <FullMergeReceipt key={operation.operationId} operation={operation} pending={commands.has(operation.operationId)} onCommand={(id, action) => void command(id, action)} onOpen={onOpen} />)}</div>
  </section>;
}
