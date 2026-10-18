import { useEffect, useState, useSyncExternalStore } from "react";
import type {
  OmniPluginPackage,
  PluginCatalogEntry,
  PluginThemeDefinition,
  PluginsApi
} from "../../shared/plugins";

export const PLUGIN_THEME_STORAGE_KEY = "omni.plugins.theme.v1";
export const PLUGIN_THEME_CHANGE_EVENT = "omni-plugin-theme-change";

export interface PluginThemeSelection {
  schemaVersion: 1;
  pluginId: string;
  contentSha256: string;
  themeId: string;
}

export interface PluginThemeStorage {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface PluginThemeRuntimeState {
  selection: PluginThemeSelection | null;
  state: "none" | "checking" | "applied" | "inactive" | "error";
  message: string;
}

export interface ResolvedPluginTheme {
  selection: PluginThemeSelection;
  entry: PluginCatalogEntry;
  theme: PluginThemeDefinition;
}

const EMPTY_STATE: PluginThemeRuntimeState = {
  selection: null,
  state: "none",
  message: "No plugin theme is applied."
};
const PLUGIN_ID = /^[a-z0-9][a-z0-9._-]{0,127}$/;
const LOCAL_ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const TOKEN_NAME = /^--[a-z][a-z0-9-]{0,63}$/;
const UNSAFE_TOKEN = /[{};]|@(?:import|namespace|supports|media|layer|property)\b|(?:url|image|image-set|cross-fade|element|paint|src)\s*\(|(?:https?|file|data|blob|javascript)\s*:/i;

let runtimeState = EMPTY_STATE;
const runtimeListeners = new Set<() => void>();

function publishRuntimeState(value: PluginThemeRuntimeState): void {
  runtimeState = value;
  for (const listener of runtimeListeners) listener();
}

function currentApi(): PluginsApi | undefined {
  return typeof window === "undefined" ? undefined : window.omni?.plugins;
}

function currentStorage(): PluginThemeStorage | undefined {
  try { return typeof window === "undefined" ? undefined : window.localStorage; }
  catch { return undefined; }
}

function notifySelectionChanged(): void {
  if (typeof window !== "undefined") window.dispatchEvent(new Event(PLUGIN_THEME_CHANGE_EVENT));
}

export function parsePluginThemeSelection(value: unknown): PluginThemeSelection | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Record<string, unknown>;
  if (Object.keys(row).sort().join(",") !== "contentSha256,pluginId,schemaVersion,themeId" ||
      row.schemaVersion !== 1 || typeof row.pluginId !== "string" || !PLUGIN_ID.test(row.pluginId) ||
      typeof row.contentSha256 !== "string" || !SHA256.test(row.contentSha256) ||
      typeof row.themeId !== "string" || !LOCAL_ID.test(row.themeId)) return null;
  return { schemaVersion: 1, pluginId: row.pluginId, contentSha256: row.contentSha256, themeId: row.themeId };
}

export function loadPluginThemeSelection(storage: PluginThemeStorage): PluginThemeSelection | null {
  try {
    const raw = storage.getItem(PLUGIN_THEME_STORAGE_KEY);
    return raw ? parsePluginThemeSelection(JSON.parse(raw) as unknown) : null;
  } catch { return null; }
}

export function savePluginThemeSelection(storage: PluginThemeStorage, selection: PluginThemeSelection): void {
  const checked = parsePluginThemeSelection(selection);
  if (!checked) throw new Error("Invalid immutable plugin theme selection.");
  storage.setItem(PLUGIN_THEME_STORAGE_KEY, JSON.stringify(checked));
  publishRuntimeState({ selection: checked, state: "checking", message: "Checking the selected immutable plugin theme…" });
  notifySelectionChanged();
}

export function clearPluginThemeSelection(storage: PluginThemeStorage): void {
  storage.removeItem(PLUGIN_THEME_STORAGE_KEY);
  publishRuntimeState(EMPTY_STATE);
  notifySelectionChanged();
}

export function checkedPluginThemeTokens(value: unknown): Record<string, string> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Plugin theme tokens are invalid.");
  const result: Record<string, string> = {};
  for (const [name, token] of Object.entries(value as Record<string, unknown>)) {
    if (!TOKEN_NAME.test(name) || typeof token !== "string" || !token.trim() || token.length > 1_000 || UNSAFE_TOKEN.test(token)) {
      throw new Error("Plugin theme contains an unsafe CSS token.");
    }
    result[name] = token;
  }
  if (!Object.keys(result).length) throw new Error("Plugin theme must contain at least one token.");
  return result;
}

export async function resolvePluginTheme(
  api: Pick<PluginsApi, "list" | "package">,
  selection: PluginThemeSelection
): Promise<ResolvedPluginTheme> {
  const checked = parsePluginThemeSelection(selection);
  if (!checked) throw new Error("Invalid immutable plugin theme selection.");
  const entry = (await api.list()).find(candidate => candidate.pluginId === checked.pluginId);
  if (!entry) throw new Error("The selected plugin is no longer installed.");
  if (!entry.enabled) throw new Error("The selected plugin is disabled.");
  if (entry.activeContentSha256 !== checked.contentSha256) {
    throw new Error("The selected plugin version changed. Apply a theme from the active version to continue.");
  }
  const packageValue = await api.package(checked.pluginId, checked.contentSha256);
  if (packageValue.manifest.id !== checked.pluginId) throw new Error("Plugin theme package ownership changed.");
  const theme = packageValue.manifest.themes.find(candidate => candidate.id === checked.themeId);
  if (!theme) throw new Error("The selected theme is not declared by this immutable plugin version.");
  checkedPluginThemeTokens(theme.tokens);
  return { selection: checked, entry, theme };
}

/**
 * Applies only validated custom properties and restores the exact prior inline
 * values. Cleanup leaves a property untouched if another renderer owner has
 * changed it since this theme wrote it.
 */
export function applyPluginThemeTokens(target: HTMLElement, resolved: ResolvedPluginTheme): () => void {
  const tokens = checkedPluginThemeTokens(resolved.theme.tokens);
  const owner = `${resolved.selection.pluginId}:${resolved.selection.contentSha256}:${resolved.selection.themeId}`;
  const prior = new Map<string, { value: string; priority: string }>();
  for (const [name, value] of Object.entries(tokens)) {
    prior.set(name, { value: target.style.getPropertyValue(name), priority: target.style.getPropertyPriority(name) });
    target.style.setProperty(name, value);
  }
  const priorDataset = {
    owner: target.dataset.pluginThemeOwner,
    plugin: target.dataset.pluginThemePlugin,
    hash: target.dataset.pluginThemeHash,
    theme: target.dataset.pluginTheme
  };
  target.dataset.pluginThemeOwner = owner;
  target.dataset.pluginThemePlugin = resolved.selection.pluginId;
  target.dataset.pluginThemeHash = resolved.selection.contentSha256;
  target.dataset.pluginTheme = resolved.selection.themeId;
  return () => {
    if (target.dataset.pluginThemeOwner !== owner) return;
    for (const [name, applied] of Object.entries(tokens)) {
      if (target.style.getPropertyValue(name) !== applied || target.style.getPropertyPriority(name)) continue;
      const previous = prior.get(name)!;
      if (previous.value) target.style.setProperty(name, previous.value, previous.priority);
      else target.style.removeProperty(name);
    }
    for (const [key, value] of Object.entries(priorDataset) as Array<[keyof typeof priorDataset, string | undefined]>) {
      const datasetKey = key === "plugin" ? "pluginThemePlugin" : key === "hash" ? "pluginThemeHash" : key === "theme" ? "pluginTheme" : "pluginThemeOwner";
      if (value === undefined) delete target.dataset[datasetKey];
      else target.dataset[datasetKey] = value;
    }
  };
}

export function createPluginThemeController(input: {
  api: Pick<PluginsApi, "list" | "package">;
  target: HTMLElement;
  onState?(state: PluginThemeRuntimeState): void;
}): {
  reconcile(selection: PluginThemeSelection | null): Promise<PluginThemeRuntimeState>;
  dispose(): void;
} {
  let generation = 0;
  let cleanup: (() => void) | undefined;
  let disposed = false;
  const setState = (state: PluginThemeRuntimeState) => { publishRuntimeState(state); input.onState?.(state); return state; };
  const restore = () => { cleanup?.(); cleanup = undefined; };
  return {
    async reconcile(selection) {
      const owner = ++generation;
      restore();
      if (!selection) return setState(EMPTY_STATE);
      setState({ selection, state: "checking", message: "Checking the selected immutable plugin theme…" });
      try {
        const resolved = await resolvePluginTheme(input.api, selection);
        if (disposed || owner !== generation) return runtimeState;
        cleanup = applyPluginThemeTokens(input.target, resolved);
        return setState({ selection, state: "applied",
          message: `${resolved.theme.title} is applied from ${resolved.entry.name} ${selection.contentSha256.slice(0, 8)}.` });
      } catch (error) {
        if (disposed || owner !== generation) return runtimeState;
        return setState({ selection, state: "inactive",
          message: error instanceof Error ? error.message : "The selected plugin theme is unavailable." });
      }
    },
    dispose() { disposed = true; generation++; restore(); }
  };
}

export function usePluginThemeRuntimeState(): PluginThemeRuntimeState {
  return useSyncExternalStore(
    listener => { runtimeListeners.add(listener); return () => { runtimeListeners.delete(listener); }; },
    () => runtimeState,
    () => EMPTY_STATE
  );
}

/** Headless Studio-wide consumer. Plugin packages remain data, never renderer code. */
export function PluginThemeHost({ api = currentApi(), target }: { api?: PluginsApi; target?: HTMLElement }) {
  useEffect(() => {
    const storage = currentStorage();
    const root = target ?? (typeof document === "undefined" ? undefined : document.documentElement);
    if (!api || !storage || !root) { publishRuntimeState(EMPTY_STATE); return; }
    const controller = createPluginThemeController({ api, target: root });
    const reconcile = () => { void controller.reconcile(loadPluginThemeSelection(storage)); };
    reconcile();
    const changed = () => reconcile();
    const storageChanged = (event: StorageEvent) => { if (event.key === PLUGIN_THEME_STORAGE_KEY) reconcile(); };
    window.addEventListener(PLUGIN_THEME_CHANGE_EVENT, changed);
    window.addEventListener("storage", storageChanged);
    const unsubscribe = api.onEvent(event => {
      const selection = loadPluginThemeSelection(storage);
      if (selection?.pluginId === event.pluginId) reconcile();
    });
    return () => {
      window.removeEventListener(PLUGIN_THEME_CHANGE_EVENT, changed);
      window.removeEventListener("storage", storageChanged);
      unsubscribe();
      controller.dispose();
    };
  }, [api, target]);
  return null;
}

export function PluginThemeChooser({ entry, packageValue, onNotice }: {
  entry: PluginCatalogEntry;
  packageValue: OmniPluginPackage;
  onNotice?: (message: string) => void;
}) {
  const runtime = usePluginThemeRuntimeState();
  const [error, setError] = useState("");
  if (!packageValue.manifest.themes.length) return null;
  const apply = (theme: PluginThemeDefinition) => {
    const storage = currentStorage();
    if (!storage) { setError("Theme preferences are unavailable in this renderer."); return; }
    try {
      savePluginThemeSelection(storage, { schemaVersion: 1, pluginId: entry.pluginId,
        contentSha256: entry.activeContentSha256, themeId: theme.id });
      setError(""); onNotice?.(`Applying ${theme.title} after verifying this exact plugin version.`);
    } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)); }
  };
  const clear = () => {
    const storage = currentStorage();
    if (!storage) { setError("Theme preferences are unavailable in this renderer."); return; }
    clearPluginThemeSelection(storage); setError(""); onNotice?.("Plugin theme cleared; Studio appearance restored.");
  };
  return <fieldset className="plugin-manager__themes" aria-labelledby="plugin-theme-heading">
    <legend id="plugin-theme-heading">Studio themes</legend>
    <p>Data-only color and surface tokens. Applying a theme never runs package code.</p>
    <div className="plugin-manager__versions">{packageValue.manifest.themes.map(theme => {
      const selected = runtime.selection?.pluginId === entry.pluginId &&
        runtime.selection.contentSha256 === entry.activeContentSha256 && runtime.selection.themeId === theme.id;
      return <button type="button" className="button" key={theme.id} aria-pressed={selected}
        disabled={!entry.enabled} onClick={() => apply(theme)}>{theme.title} · {selected ? "selected" : "Apply"}</button>;
    })}</div>
    {!entry.enabled ? <p role="status">Enable this exact plugin version globally before applying its theme.</p> : null}
    <p role="status">{error || runtime.message}</p>
    {runtime.selection ? <button type="button" className="button" onClick={clear}>Clear plugin theme</button> : null}
  </fieldset>;
}
