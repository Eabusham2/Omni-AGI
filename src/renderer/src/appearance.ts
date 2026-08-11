import {
  APPEARANCE_SCHEMA_VERSION,
  type AppearanceLayout,
  type AppearanceMode,
  type AppearancePalette,
  type AppearancePreferences,
  type ResolvedColorScheme
} from "../../shared/types";

export const APPEARANCE_STORAGE_KEY = "omni.appearance.v1";

export interface AppearancePack {
  id: "standard" | "classic" | "colorful" | "liquid-glass";
  name: string;
  description: string;
  palette: AppearancePalette;
  layout: AppearanceLayout;
}

export const APPEARANCE_MODES: ReadonlyArray<AppearanceMode> = [
  "system",
  "light",
  "dark"
];
export const APPEARANCE_PALETTES: ReadonlyArray<AppearancePalette> = [
  "violet",
  "graphite",
  "spectrum",
  "aqua"
];
export const APPEARANCE_LAYOUTS: ReadonlyArray<AppearanceLayout> = [
  "standard",
  "classic",
  "expressive",
  "glass"
];

export const APPEARANCE_PACKS: ReadonlyArray<AppearancePack> = [
  {
    id: "standard",
    name: "Standard",
    description: "Balanced spacing and familiar violet surfaces.",
    palette: "violet",
    layout: "standard"
  },
  {
    id: "classic",
    name: "Classic Blocky",
    description: "Denser square panels with a restrained graphite palette.",
    palette: "graphite",
    layout: "classic"
  },
  {
    id: "colorful",
    name: "Colorful",
    description: "Expressive cards, stronger hierarchy, and spectrum accents.",
    palette: "spectrum",
    layout: "expressive"
  },
  {
    id: "liquid-glass",
    name: "Liquid Glass",
    description: "Layered translucent surfaces with fluid aqua light.",
    palette: "aqua",
    layout: "glass"
  }
] as const;

export const DEFAULT_APPEARANCE: AppearancePreferences = {
  schemaVersion: APPEARANCE_SCHEMA_VERSION,
  mode: "system",
  palette: "violet",
  layout: "standard"
};

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function includes<T extends string>(values: ReadonlyArray<T>, value: unknown): value is T {
  return typeof value === "string" && values.includes(value as T);
}

export function parseAppearancePreferences(value: unknown): AppearancePreferences {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return { ...DEFAULT_APPEARANCE };
  }
  const candidate = value as Partial<AppearancePreferences>;
  if (candidate.schemaVersion !== APPEARANCE_SCHEMA_VERSION) {
    return { ...DEFAULT_APPEARANCE };
  }
  return {
    schemaVersion: APPEARANCE_SCHEMA_VERSION,
    mode: includes(APPEARANCE_MODES, candidate.mode)
      ? candidate.mode
      : DEFAULT_APPEARANCE.mode,
    palette: includes(APPEARANCE_PALETTES, candidate.palette)
      ? candidate.palette
      : DEFAULT_APPEARANCE.palette,
    layout: includes(APPEARANCE_LAYOUTS, candidate.layout)
      ? candidate.layout
      : DEFAULT_APPEARANCE.layout
  };
}

export function loadAppearancePreferences(storage: StorageLike): AppearancePreferences {
  try {
    const raw = storage.getItem(APPEARANCE_STORAGE_KEY);
    return raw
      ? parseAppearancePreferences(JSON.parse(raw) as unknown)
      : { ...DEFAULT_APPEARANCE };
  } catch {
    return { ...DEFAULT_APPEARANCE };
  }
}

export function saveAppearancePreferences(
  storage: StorageLike,
  preferences: AppearancePreferences
): void {
  storage.setItem(
    APPEARANCE_STORAGE_KEY,
    JSON.stringify(parseAppearancePreferences(preferences))
  );
}

export function resolveColorScheme(
  mode: AppearanceMode,
  systemUsesDark: boolean
): ResolvedColorScheme {
  return mode === "system" ? (systemUsesDark ? "dark" : "light") : mode;
}

/** Packs update visual character while preserving the independently chosen mode. */
export function preferencesForPack(
  current: AppearancePreferences,
  packId: AppearancePack["id"]
): AppearancePreferences {
  const pack = APPEARANCE_PACKS.find((candidate) => candidate.id === packId);
  if (!pack) return current;
  return {
    schemaVersion: APPEARANCE_SCHEMA_VERSION,
    mode: current.mode,
    palette: pack.palette,
    layout: pack.layout
  };
}

export function activeAppearancePack(
  preferences: AppearancePreferences
): AppearancePack["id"] | "custom" {
  return APPEARANCE_PACKS.find(
    (pack) => pack.palette === preferences.palette && pack.layout === preferences.layout
  )?.id ?? "custom";
}

export function applyAppearanceAttributes(
  root: HTMLElement,
  preferences: AppearancePreferences,
  resolvedColorScheme: ResolvedColorScheme
): void {
  root.dataset.appearanceVersion = String(APPEARANCE_SCHEMA_VERSION);
  root.dataset.appearanceMode = preferences.mode;
  root.dataset.colorScheme = resolvedColorScheme;
  root.dataset.palette = preferences.palette;
  root.dataset.layout = preferences.layout;
  root.dataset.appearancePack = activeAppearancePack(preferences);
  root.style.colorScheme = resolvedColorScheme;
}
