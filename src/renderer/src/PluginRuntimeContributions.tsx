import { createContext, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type {
  PluginRuntimeApi,
  PluginRuntimeMountedContribution
} from "../../shared/pluginRuntime";
import "./pluginRuntimeContributions.css";

export const PLUGIN_RUNTIME_STUDIO_SLOTS = ["chat.toolbar", "chat.before", "chat.after", "tools"] as const;
export type PluginRuntimeStudioSlot = typeof PLUGIN_RUNTIME_STUDIO_SLOTS[number];
const SLOT_SET = new Set<string>(PLUGIN_RUNTIME_STUDIO_SLOTS);
const PANEL_CSP = "default-src 'none'; script-src 'none'; connect-src 'none'; img-src 'none'; media-src 'none'; font-src 'none'; frame-src 'none'; worker-src 'none'; object-src 'none'; form-action 'none'; base-uri 'none'; navigate-to 'none'; style-src 'unsafe-inline'; sandbox";

interface RuntimeContributionContextValue {
  brainId?: string;
  api?: PluginRuntimeApi;
  contributions: PluginRuntimeMountedContribution[];
  error: string;
  onNotice?: (message: string) => void;
}

const EMPTY_CONTEXT: RuntimeContributionContextValue = { contributions: [], error: "" };
const RuntimeContributionContext = createContext<RuntimeContributionContextValue>(EMPTY_CONTEXT);

function currentRuntimeApi(): PluginRuntimeApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.pluginRuntime;
}

function secureUuid(): string {
  if (!globalThis.crypto?.randomUUID) throw new Error("Secure runtime event identity is unavailable.");
  return globalThis.crypto.randomUUID();
}

function escapeHtml(value: string): string {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** Main has already validated inert HTML/CSS; this keeps it inside an opaque deny-all child. */
export function pluginRuntimePanelDocument(contribution: Pick<PluginRuntimeMountedContribution, "title" | "html" | "css">): string {
  const title = escapeHtml(contribution.title ?? "Plugin panel");
  // A literal '<' cannot terminate this owned style element after CSS validation.
  const css = (contribution.css ?? "").replace(/</g, "\\3c ");
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="${PANEL_CSP}"><meta name="referrer" content="no-referrer"><title>${title}</title><style>html{color-scheme:light dark;font:13px/1.45 system-ui,sans-serif}body{margin:0;padding:12px;overflow-wrap:anywhere}${css}</style></head><body>${contribution.html ?? ""}</body></html>`;
}

export function pluginRuntimeStudioCss(value: string): string {
  return value.replace(/</g, "\\3c ");
}

export function PluginRuntimePanelFrame({ contribution }: { contribution: PluginRuntimeMountedContribution }) {
  return <iframe title={contribution.title ?? `${contribution.pluginId} panel`} sandbox="" referrerPolicy="no-referrer"
    allow="camera 'none'; microphone 'none'; geolocation 'none'; clipboard-read 'none'; clipboard-write 'none'; display-capture 'none'"
    srcDoc={pluginRuntimePanelDocument(contribution)} style={{ boxSizing: "border-box", width: "100%", minHeight: 160, border: 0 }} />;
}

export function checkedPluginRuntimeContributions(brainId: string, values: PluginRuntimeMountedContribution[]): PluginRuntimeMountedContribution[] {
  return values.filter(value => value.brainId === brainId && SLOT_SET.has(value.slot) &&
    ["style", "menu", "button", "panel"].includes(value.kind) && Number.isSafeInteger(value.revision) && value.revision > 0 &&
    /^[a-f0-9]{64}$/.test(value.contentSha256) && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(value.sessionId));
}

export function ownedPluginRuntimeContributions(
  brainId: string | undefined,
  loadedBrainId: string | undefined,
  values: PluginRuntimeMountedContribution[]
): PluginRuntimeMountedContribution[] {
  return brainId && loadedBrainId === brainId ? checkedPluginRuntimeContributions(brainId, values) : [];
}

export function pluginRuntimeSlotError(providerError: string, localError: string, showProviderError: boolean): string {
  return localError || (showProviderError ? providerError : "");
}

export function PluginRuntimeContributionsProvider({ brainId, api = currentRuntimeApi(), onNotice, children }: {
  brainId?: string;
  api?: PluginRuntimeApi;
  onNotice?: (message: string) => void;
  children?: ReactNode;
}) {
  const [contributions, setContributions] = useState<PluginRuntimeMountedContribution[]>([]);
  const [error, setError] = useState("");
  const [loadedBrainId, setLoadedBrainId] = useState<string>();
  const epoch = useRef(0);
  useEffect(() => {
    const owner = ++epoch.current;
    let current = true;
    setContributions([]); setError("");
    if (!api || !brainId) return;
    const refresh = async () => {
      const request = ++epoch.current;
      try {
        const next = checkedPluginRuntimeContributions(brainId, await api.contributions(brainId));
        if (current && request === epoch.current) { setContributions(next); setError(""); setLoadedBrainId(brainId); }
      } catch (failure) {
        if (current && request === epoch.current) {
          setContributions([]); setError(failure instanceof Error ? failure.message : String(failure)); setLoadedBrainId(brainId);
        }
      }
    };
    void refresh();
    const unsubscribe = api.onEvent(event => { if (event.brainId === brainId) void refresh(); });
    return () => { current = false; epoch.current = Math.max(epoch.current + 1, owner + 1); unsubscribe(); };
  }, [api, brainId]);
  const currentContributions = useMemo(() => ownedPluginRuntimeContributions(brainId, loadedBrainId, contributions),
    [brainId, contributions, loadedBrainId]);
  const currentError = loadedBrainId === brainId ? error : "";
  const styles = currentContributions.filter(value => value.kind === "style" && value.css);
  const context = useMemo<RuntimeContributionContextValue>(() => ({ brainId, api, contributions: currentContributions, error: currentError, onNotice }),
    [api, brainId, currentContributions, currentError, onNotice]);
  return <RuntimeContributionContext.Provider value={context}>
    {styles.map(value => <style key={`${value.sessionId}/${value.id}/${value.revision}`} data-plugin-runtime-style={`${value.pluginId}/${value.id}`}>{pluginRuntimeStudioCss(value.css!)}</style>)}
    {children}
  </RuntimeContributionContext.Provider>;
}

export function PluginRuntimeSlot({ slot, showProviderError = false }: {
  slot: PluginRuntimeStudioSlot;
  /** Designate one existing surface (normally Tools) for provider-wide failures. */
  showProviderError?: boolean;
}) {
  const state = useContext(RuntimeContributionContext);
  const [pending, setPending] = useState("");
  const [error, setError] = useState("");
  const brainOwner = useRef(state.brainId); brainOwner.current = state.brainId;
  useEffect(() => { setPending(""); setError(""); }, [state.brainId]);
  const values = state.contributions.filter(value => value.slot === slot && value.kind !== "style");
  const visibleError = pluginRuntimeSlotError(state.error, error, showProviderError);
  const invoke = async (value: PluginRuntimeMountedContribution) => {
    if (!state.api || !state.brainId || !value.event || pending) return;
    const brainId = state.brainId;
    const pendingId = `${value.sessionId}/${value.id}/${value.revision}`;
    setPending(pendingId); setError("");
    try {
      await state.api.invokeContribution(brainId, value.sessionId, value.id, secureUuid(), {
        slot: value.slot,
        kind: value.kind,
        revision: value.revision
      });
      if (brainOwner.current === brainId) state.onNotice?.(`${value.title ?? value.id} sent to its isolated plugin runtime.`);
    } catch (failure) { if (brainOwner.current === brainId) setError(failure instanceof Error ? failure.message : String(failure)); }
    finally { if (brainOwner.current === brainId) setPending(""); }
  };
  return <PluginRuntimeContributionItems slot={slot} values={values} pending={pending} visibleError={visibleError} onInvoke={invoke} />;
}

export function PluginRuntimeContributionItems({ slot, values, pending, visibleError, onInvoke }: {
  slot: PluginRuntimeStudioSlot;
  values: PluginRuntimeMountedContribution[];
  pending: string;
  visibleError: string;
  onInvoke(value: PluginRuntimeMountedContribution): void;
}) {
  if (!values.length && !visibleError) return null;
  const buttons = values.filter(value => value.kind === "button");
  const menus = values.filter(value => value.kind === "menu");
  const panels = values.filter(value => value.kind === "panel");
  return <section className="plugin-runtime-slot" data-runtime-slot={slot} aria-label={`Plugin additions for ${slot}`}>
    {buttons.map(value => value.event
      ? <button type="button" className="button" key={`${value.sessionId}/${value.id}/${value.revision}`}
          disabled={Boolean(pending)} onClick={() => onInvoke(value)}>{value.title ?? value.id}</button>
      : <span key={`${value.sessionId}/${value.id}/${value.revision}`}>{value.title ?? value.id}</span>)}
    {menus.length ? <details className="plugin-runtime-menu"><summary>Plugin actions</summary>
      <div role="group" aria-label="Plugin runtime actions">{menus.map(value => value.event
      ? <button type="button" className="button" key={`${value.sessionId}/${value.id}/${value.revision}`}
          disabled={Boolean(pending)} onClick={event => {
            onInvoke(value);
            const menu = event.currentTarget.closest("details");
            if (menu) { menu.open = false; menu.querySelector("summary")?.focus(); }
          }}>{value.title ?? value.id}</button>
      : <span key={`${value.sessionId}/${value.id}/${value.revision}`}>{value.title ?? value.id}</span>)}</div>
    </details> : null}
    {panels.map(value => <article className="plugin-runtime-panel" key={`${value.sessionId}/${value.id}/${value.revision}`}>
      <PluginRuntimePanelFrame contribution={value} />
      {value.event ? <button type="button" className="button" disabled={Boolean(pending)} onClick={() => onInvoke(value)}>
        {value.title ? `Use ${value.title}` : "Use plugin panel"}
      </button> : null}
    </article>)}
    {visibleError ? <p role="alert">{visibleError}</p> : null}
  </section>;
}
