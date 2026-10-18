import { useCallback, useEffect, useRef, useState } from "react";
import type {
  OmniPluginPackage,
  PluginCatalogEntry,
  PluginInstanceAccess,
  PluginsApi
} from "../../shared/plugins";
import type {
  PluginRuntimeApi,
  PluginRuntimeHistoryPage,
  PluginRuntimeModuleDefinition,
  PluginRuntimePendingApproval,
  PluginRuntimeSession,
  PluginRuntimeStateConflictPage,
  PluginRuntimeStateSourcePage,
  PluginRuntimeStateValue
} from "../../shared/pluginRuntime";
import { PluginRuntimeApprovalCard } from "./PluginRuntimeApprovals";

interface RuntimeSnapshot {
  scopeKey?: string;
  entry?: PluginCatalogEntry;
  access?: PluginInstanceAccess | null;
  sessions: PluginRuntimeSession[];
  sessionTotal: number;
  sessionNextRowid?: number;
  approvals: PluginRuntimePendingApproval[];
  history: PluginRuntimeHistoryPage;
  conflicts: PluginRuntimeStateConflictPage;
  sources: PluginRuntimeStateSourcePage;
}

const EMPTY_HISTORY: PluginRuntimeHistoryPage = { events: [], total: 0 };
const EMPTY_CONFLICTS: PluginRuntimeStateConflictPage = { conflicts: [], total: 0 };
const EMPTY_SOURCES: PluginRuntimeStateSourcePage = { sources: [], total: 0 };
const EMPTY_SNAPSHOT: RuntimeSnapshot = {
  sessions: [],
  sessionTotal: 0,
  approvals: [],
  history: EMPTY_HISTORY,
  conflicts: EMPTY_CONFLICTS,
  sources: EMPTY_SOURCES
};

function currentRuntimeApi(): PluginRuntimeApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.pluginRuntime;
}

function runtimeScopeKey(brainId: string, pluginId: string, contentSha256: string): string {
  return `${brainId}\u0000${pluginId}\u0000${contentSha256}`;
}

function secureUuid(): string {
  if (!globalThis.crypto?.randomUUID) throw new Error("Secure runtime request identity is unavailable.");
  return globalThis.crypto.randomUUID();
}

export function pluginRuntimeActivationReview(input: {
  brainId: string;
  expectedContentSha256: string;
  entry?: PluginCatalogEntry;
  access?: PluginInstanceAccess | null;
  packageValue: OmniPluginPackage;
  moduleId: string;
}): { allowed: boolean; reason: string; catalogRevision?: number; instanceRevision?: number } {
  const { brainId, expectedContentSha256, entry, access, packageValue, moduleId } = input;
  const module = packageValue.manifest.runtimeModules?.find(candidate => candidate.id === moduleId);
  if (!entry || entry.pluginId !== packageValue.manifest.id || !module) {
    return { allowed: false, reason: "This module is not part of the current immutable package." };
  }
  if (!entry.enabled || entry.activeContentSha256 !== expectedContentSha256 || entry.activeContentSha256 !== access?.contentSha256) {
    return { allowed: false, reason: "Enable and review this exact active plugin version first." };
  }
  if (!access?.enabled || access.pluginId !== entry.pluginId || access.brainId !== brainId || access.contentSha256 !== entry.activeContentSha256) {
    return { allowed: false, reason: "Enable this exact plugin version for the current instance first." };
  }
  return { allowed: true, reason: "Ready for explicit activation.", catalogRevision: entry.revision, instanceRevision: access.revision };
}

function liveSession(
  sessions: readonly PluginRuntimeSession[],
  pluginId: string,
  contentSha256: string,
  moduleId: string
): PluginRuntimeSession | undefined {
  return sessions.find(session => session.purpose !== "training-parser" && session.pluginId === pluginId && session.contentSha256 === contentSha256 &&
    session.moduleId === moduleId && ["starting", "active", "stopping"].includes(session.state));
}

function appendedUnique<T>(current: readonly T[], next: readonly T[], identity: (value: T) => string): T[] {
  const result = [...current], seen = new Set(current.map(identity));
  for (const value of next) if (!seen.has(identity(value))) { seen.add(identity(value)); result.push(value); }
  return result;
}

export function pluginRuntimeHistoryCount(total: number): string {
  return `${total.toLocaleString()} ${total === 1 ? "event" : "events"}`;
}

export function PluginRuntimeConflictReview({ brainId, conflict, sources, api, onResolved }: {
  brainId: string;
  conflict: PluginRuntimeStateConflictPage["conflicts"][number];
  sources: PluginRuntimeStateSourcePage["sources"];
  api: PluginRuntimeApi;
  onResolved(): Promise<void>;
}) {
  const [sourceSha256, setSourceSha256] = useState("");
  const [inspected, setInspected] = useState<PluginRuntimeStateValue>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const inspect = async () => {
    if (!sourceSha256 || busy) return;
    setBusy(true); setError(""); setInspected(undefined);
    try { setInspected(await api.readSourceState(brainId, sourceSha256, conflict.pluginHash, conflict.moduleId, conflict.key)); }
    catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setBusy(false); }
  };
  const select = async () => {
    if (!sourceSha256 || !inspected || busy) return;
    setBusy(true); setError("");
    try {
      await api.selectSourceState({ brainId, sourceSha256, pluginHash: conflict.pluginHash,
        moduleId: conflict.moduleId, key: conflict.key, expectedRevision: 0 });
      await onResolved();
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { setBusy(false); }
  };
  return <article className="plugin-runtime-conflict">
    <strong>{conflict.moduleId} · {conflict.key}</strong>
    <code>{conflict.pluginHash}</code>
    <label>Inherited source
      <select value={sourceSha256} disabled={busy} onChange={event => { setSourceSha256(event.currentTarget.value); setInspected(undefined); setError(""); }}>
        <option value="">Choose a source to inspect…</option>
        {sources.map(source => <option key={source.sha256} value={source.sha256}>{source.owner} · {source.sha256.slice(0, 12)}… · {source.bytes.toLocaleString()} bytes</option>)}
      </select>
    </label>
    <div className="plugin-manager__actions">
      <button type="button" className="button" disabled={!sourceSha256 || busy} onClick={() => void inspect()}>Inspect exact source</button>
      <button type="button" className="button button--primary" disabled={!sourceSha256 || !inspected || busy} onClick={() => void select()}>Use reviewed source</button>
    </div>
    {inspected ? <pre>{JSON.stringify(inspected.value ?? { receipt: inspected.receipt, revision: inspected.revision }, null, 2)}</pre> : null}
    {error ? <p role="alert">{error}</p> : null}
  </article>;
}

export function PluginRuntimeModulePresentation({ module, session, approvals, busy, activation, onActivate, onStop, onReview }: {
  module: PluginRuntimeModuleDefinition;
  session?: PluginRuntimeSession;
  approvals: PluginRuntimePendingApproval[];
  busy: string;
  activation: ReturnType<typeof pluginRuntimeActivationReview>;
  onActivate(): void;
  onStop(): void;
  onReview(review: PluginRuntimePendingApproval, decision: "approve" | "deny"): void;
}) {
  return <article className="plugin-runtime-module">
    <header><span><strong>{module.title}</strong><small>{module.description}</small></span>
      <em>{session?.state ?? "inactive"}</em></header>
    <dl>
      <div><dt>Events</dt><dd>{module.events.join(", ") || "None"}</dd></div>
      <div><dt>UI kinds</dt><dd>{module.ui.kinds.join(", ") || "None"}</dd></div>
      <div><dt>Slots</dt><dd>{module.ui.slots.join(", ") || "None"}</dd></div>
      <div><dt>Dynamic tools</dt><dd>{module.dynamicTools ? "Declared" : "Not declared"}</dd></div>
    </dl>
    <p role="status">{session ? `Module ${session.state}.` : activation.reason}</p>
    {session ? <details><summary>Session details</summary><dl>
      <div><dt>Session ID</dt><dd>{session.sessionId}</dd></div>
      <div><dt>Package</dt><dd>{session.pluginId} · {session.contentSha256}</dd></div>
    </dl></details> : null}
    <div className="plugin-manager__actions">
      {session ? <button type="button" className="button" disabled={busy !== "" || session.state === "stopping"} onClick={onStop}>Stop</button>
        : <button type="button" className="button button--primary" disabled={busy !== "" || !activation.allowed} onClick={onActivate}>Activate</button>}
    </div>
    {approvals.map(item => <PluginRuntimeApprovalCard key={`${item.sessionId}/${item.rpcId}/${item.reviewId}`} item={item} busy={busy !== ""} onReview={onReview} />)}
  </article>;
}

export function PluginRuntimeControls({ brainId, entry, packageValue, pluginsApi, runtimeApi = currentRuntimeApi(), onNotice }: {
  brainId: string;
  entry: PluginCatalogEntry;
  packageValue: OmniPluginPackage;
  pluginsApi: PluginsApi;
  runtimeApi?: PluginRuntimeApi;
  onNotice?: (message: string) => void;
}) {
  const modules = packageValue.manifest.runtimeModules ?? [];
  const [snapshot, setSnapshot] = useState<RuntimeSnapshot>(EMPTY_SNAPSHOT);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [feedbackScope, setFeedbackScope] = useState("");
  const scopeEpoch = useRef(0), readEpoch = useRef(0);
  const notice = useRef(onNotice); notice.current = onNotice;
  const pluginId = entry.pluginId, contentSha256 = entry.activeContentSha256;
  const currentScopeKey = runtimeScopeKey(brainId, pluginId, contentSha256);
  const renderedScope = useRef(currentScopeKey); renderedScope.current = currentScopeKey;
  const currentSnapshot = snapshot.scopeKey === currentScopeKey ? snapshot : EMPTY_SNAPSHOT;
  const currentBusy = feedbackScope === currentScopeKey ? busy : "";
  const currentError = feedbackScope === currentScopeKey ? error : "";

  const refresh = useCallback(async () => {
    if (!runtimeApi) return;
    const scope = scopeEpoch.current, request = ++readEpoch.current;
    try {
      const [catalog, access, sessions, approvals, history, conflicts, sources] = await Promise.all([
        pluginsApi.list(), pluginsApi.instance(brainId, pluginId), runtimeApi.list(brainId, { maxBytes: 256 * 1024 }),
        runtimeApi.approvals(brainId), runtimeApi.history(brainId, { maxBytes: 64 * 1024 }),
        runtimeApi.stateConflicts(brainId, { limit: 64 }), runtimeApi.stateSources(brainId, { limit: 64 })
      ]);
      if (scope !== scopeEpoch.current || request !== readEpoch.current || renderedScope.current !== currentScopeKey) return;
      setSnapshot({ scopeKey: currentScopeKey, entry: catalog.find(candidate => candidate.pluginId === pluginId), access,
        sessions: sessions.sessions, sessionTotal: sessions.total, ...(sessions.nextRowid ? { sessionNextRowid: sessions.nextRowid } : {}),
        approvals, history, conflicts, sources });
      setFeedbackScope(currentScopeKey); setError("");
    } catch (failure) {
      if (scope !== scopeEpoch.current || request !== readEpoch.current || renderedScope.current !== currentScopeKey) return;
      setSnapshot(EMPTY_SNAPSHOT);
      setFeedbackScope(currentScopeKey); setError(failure instanceof Error ? failure.message : String(failure));
    }
  }, [brainId, contentSha256, currentScopeKey, pluginId, pluginsApi, runtimeApi]);

  useEffect(() => {
    scopeEpoch.current++; readEpoch.current++; setSnapshot(EMPTY_SNAPSHOT); setFeedbackScope(currentScopeKey); setError(""); setBusy("");
    if (!runtimeApi) return;
    void refresh();
    const runtimeEvents = runtimeApi.onEvent(event => { if (event.brainId === brainId) void refresh(); });
    const pluginEvents = pluginsApi.onEvent(event => {
      if (event.pluginId === pluginId && (event.type === "catalog" || event.brainId === brainId)) void refresh();
    });
    return () => { scopeEpoch.current++; readEpoch.current++; runtimeEvents(); pluginEvents(); };
  }, [brainId, pluginId, contentSha256, currentScopeKey, pluginsApi, runtimeApi, refresh]);

  if (!modules.length) return null;
  if (!runtimeApi) return <fieldset><legend>Feature runtime</legend><p>Plugin runtime is unavailable in this renderer.</p></fieldset>;
  const run = async (key: string, operation: () => Promise<string>) => {
    if (currentBusy) return;
    const owner = scopeEpoch.current, operationScope = currentScopeKey; setFeedbackScope(operationScope); setBusy(key); setError("");
    try {
      const message = await operation();
      if (owner === scopeEpoch.current && renderedScope.current === operationScope) { notice.current?.(message); setBusy(""); await refresh(); }
    }
    catch (failure) {
      if (owner === scopeEpoch.current && renderedScope.current === operationScope) setError(failure instanceof Error ? failure.message : String(failure));
    }
    finally { if (owner === scopeEpoch.current && renderedScope.current === operationScope) setBusy(""); }
  };
  const activate = (module: PluginRuntimeModuleDefinition) => void run(`activate:${module.id}`, async () => {
    const [catalog, access, exactPackage] = await Promise.all([
      pluginsApi.list(), pluginsApi.instance(brainId, pluginId), pluginsApi.package(pluginId, contentSha256)
    ]);
    const current = catalog.find(candidate => candidate.pluginId === pluginId);
    const reviewed = pluginRuntimeActivationReview({ brainId, expectedContentSha256: contentSha256,
      entry: current, access, packageValue: exactPackage, moduleId: module.id });
    if (!reviewed.allowed || reviewed.catalogRevision === undefined || reviewed.instanceRevision === undefined ||
        exactPackage.manifest.id !== pluginId || current?.activeContentSha256 !== contentSha256) throw new Error(reviewed.reason);
    await runtimeApi.activate({ brainId, pluginId, contentSha256, moduleId: module.id,
      expectedCatalogRevision: reviewed.catalogRevision, expectedInstanceRevision: reviewed.instanceRevision, requestId: secureUuid() });
    return `${module.title} activated in its isolated runtime.`;
  });
  const stop = (module: PluginRuntimeModuleDefinition, session: PluginRuntimeSession) => void run(`stop:${session.sessionId}`, async () => {
    await runtimeApi.stop(brainId, session.sessionId); return `${module.title} stopped.`;
  });
  const review = (item: PluginRuntimePendingApproval, decision: "approve" | "deny") => void run(`review:${item.reviewId}`, async () => {
    await runtimeApi.review({ brainId, sessionId: item.sessionId, rpcId: item.rpcId, reviewId: item.reviewId, decision });
    return decision === "approve" ? "The exact runtime host action was reviewed." : "The runtime host action was denied.";
  });
  const page = async (key: string, operation: (owner: number) => Promise<void>) => {
    if (currentBusy) return;
    const owner = scopeEpoch.current, operationScope = currentScopeKey; setFeedbackScope(operationScope); setBusy(key); setError("");
    try { await operation(owner); }
    catch (failure) {
      if (owner === scopeEpoch.current && renderedScope.current === operationScope) setError(failure instanceof Error ? failure.message : String(failure));
    }
    finally { if (owner === scopeEpoch.current && renderedScope.current === operationScope) setBusy(""); }
  };
  const olderSessions = () => void page("page:sessions", async owner => {
    if (!currentSnapshot.sessionNextRowid) return;
    const next = await runtimeApi.list(brainId, { afterRowid: currentSnapshot.sessionNextRowid, maxBytes: 256 * 1024 });
    if (owner !== scopeEpoch.current || renderedScope.current !== currentScopeKey) return;
    setSnapshot(current => ({ ...current, sessions: appendedUnique(current.sessions, next.sessions, value => value.sessionId), sessionTotal: next.total,
      ...(next.nextRowid ? { sessionNextRowid: next.nextRowid } : { sessionNextRowid: undefined }) }));
  });
  const olderHistory = () => void page("page:history", async owner => {
    if (!currentSnapshot.history.nextSequence) return;
    const next = await runtimeApi.history(brainId, { afterSequence: currentSnapshot.history.nextSequence, maxBytes: 64 * 1024 });
    if (owner !== scopeEpoch.current || renderedScope.current !== currentScopeKey) return;
    setSnapshot(current => ({ ...current, history: { ...next,
      events: appendedUnique(current.history.events, next.events, value => String(value.sequence)) } }));
  });
  const moreConflicts = () => void page("page:conflicts", async owner => {
    if (!currentSnapshot.conflicts.nextKey) return;
    const next = await runtimeApi.stateConflicts(brainId, { afterKey: currentSnapshot.conflicts.nextKey, limit: 64 });
    if (owner !== scopeEpoch.current || renderedScope.current !== currentScopeKey) return;
    setSnapshot(current => ({ ...current, conflicts: { ...next,
      conflicts: appendedUnique(current.conflicts.conflicts, next.conflicts, value => `${value.pluginHash}/${value.moduleId}/${value.key}`) } }));
  });
  const moreSources = () => void page("page:sources", async owner => {
    if (!currentSnapshot.sources.nextSha256) return;
    const next = await runtimeApi.stateSources(brainId, { afterSha256: currentSnapshot.sources.nextSha256, limit: 64 });
    if (owner !== scopeEpoch.current || renderedScope.current !== currentScopeKey) return;
    setSnapshot(current => ({ ...current, sources: { ...next,
      sources: appendedUnique(current.sources.sources, next.sources, value => value.sha256) } }));
  });

  return <fieldset className="plugin-runtime-controls">
    <legend>Feature runtime</legend>
    <p>Modules run only after explicit activation in an isolated renderer. Package or access changes stop them and require another review.</p>
    {modules.map(module => {
      const session = liveSession(currentSnapshot.sessions, pluginId, contentSha256, module.id);
      const activation = pluginRuntimeActivationReview({ brainId, expectedContentSha256: contentSha256,
        entry: currentSnapshot.entry, access: currentSnapshot.access, packageValue, moduleId: module.id });
      const approvals = currentSnapshot.approvals.filter(item => item.pluginId === pluginId && item.contentSha256 === contentSha256 && item.moduleId === module.id);
      return <PluginRuntimeModulePresentation key={module.id} module={module} session={session} approvals={approvals} busy={currentBusy}
        activation={activation} onActivate={() => activate(module)} onStop={() => session && stop(module, session)} onReview={review} />;
    })}
    {currentSnapshot.conflicts.conflicts.filter(conflict => conflict.pluginHash === contentSha256 && modules.some(module => module.id === conflict.moduleId)).map(conflict =>
      <PluginRuntimeConflictReview key={`${brainId}/${conflict.pluginHash}/${conflict.moduleId}/${conflict.key}`} brainId={brainId} conflict={conflict}
        sources={currentSnapshot.sources.sources} api={runtimeApi} onResolved={refresh} />)}
    {currentSnapshot.conflicts.nextKey || currentSnapshot.sources.nextSha256 ? <div className="plugin-manager__actions">
      {currentSnapshot.conflicts.nextKey ? <button type="button" className="button" disabled={Boolean(currentBusy)} onClick={moreConflicts}>Load more conflicts</button> : null}
      {currentSnapshot.sources.nextSha256 ? <button type="button" className="button" disabled={Boolean(currentBusy)} onClick={moreSources}>Load more inherited sources</button> : null}
      <span>No parent is selected automatically.</span>
    </div> : null}
    {currentSnapshot.history.events.length ? <details><summary>Runtime history · {pluginRuntimeHistoryCount(currentSnapshot.history.total)}</summary>
      <ol>{currentSnapshot.history.events.map(event => <li key={event.sequence}><strong>{event.kind}</strong> · #{event.sequence} · {event.sha256.slice(0, 12)}…
        {event.receipt ? ` · ${event.receipt.bytes.toLocaleString()} bytes (paged)` : null}</li>)}</ol>
      {currentSnapshot.history.nextSequence ? <button type="button" className="button" disabled={Boolean(currentBusy)} onClick={olderHistory}>Load more history</button> : null}
    </details> : null}
    {currentSnapshot.sessionTotal > currentSnapshot.sessions.length ? <div className="plugin-manager__actions"><span role="status">Showing the newest {currentSnapshot.sessions.length} of {currentSnapshot.sessionTotal} retained sessions.</span>
      {currentSnapshot.sessionNextRowid ? <button type="button" className="button" disabled={Boolean(currentBusy)} onClick={olderSessions}>Load older sessions</button> : null}</div> : null}
    {currentError ? <p className="plugin-manager__error" role="alert">{currentError}</p> : null}
  </fieldset>;
}
