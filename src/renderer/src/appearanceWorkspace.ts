import type { AppearancePreferences } from "../../shared/types";
import {
  APPEARANCE_LAYOUTS,
  APPEARANCE_PALETTES,
  parseAppearancePreferences,
  type StorageLike
} from "./appearance";

export const APPEARANCE_WORKSPACE_STORAGE_KEY = "omni.appearance.workspace.v1";

export type WorkspaceArrangement = "split" | "focus" | "stacked" | "mixed";

export interface AppearanceProfile {
  id: string;
  name: string;
  appearance: AppearancePreferences;
  arrangement: WorkspaceArrangement;
  createdAt: string;
  updatedAt: string;
}

export interface AppearanceWorkspacePreferences {
  schemaVersion: 1;
  arrangement: WorkspaceArrangement;
  notificationsEnabled: boolean;
  profiles: AppearanceProfile[];
}

export const WORKSPACE_ARRANGEMENTS: ReadonlyArray<WorkspaceArrangement> = [
  "split",
  "focus",
  "stacked",
  "mixed"
];

export const DEFAULT_APPEARANCE_WORKSPACE: AppearanceWorkspacePreferences = {
  schemaVersion: 1,
  arrangement: "split",
  notificationsEnabled: false,
  profiles: []
};

function isArrangement(value: unknown): value is WorkspaceArrangement {
  return typeof value === "string" && WORKSPACE_ARRANGEMENTS.includes(value as WorkspaceArrangement);
}

function safeProfile(value: unknown): AppearanceProfile | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = value as Partial<AppearanceProfile>;
  const name = typeof candidate.name === "string" ? candidate.name.trim().slice(0, 48) : "";
  const id = typeof candidate.id === "string" && /^[a-zA-Z0-9_-]{1,80}$/.test(candidate.id)
    ? candidate.id
    : "";
  if (!id || !name || !isArrangement(candidate.arrangement)) return null;
  const createdAt = typeof candidate.createdAt === "string" ? candidate.createdAt : new Date(0).toISOString();
  const updatedAt = typeof candidate.updatedAt === "string" ? candidate.updatedAt : createdAt;
  return {
    id,
    name,
    appearance: parseAppearancePreferences(candidate.appearance),
    arrangement: candidate.arrangement,
    createdAt,
    updatedAt
  };
}

export function parseAppearanceWorkspace(value: unknown): AppearanceWorkspacePreferences {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return { ...DEFAULT_APPEARANCE_WORKSPACE, profiles: [] };
  }
  const candidate = value as Partial<AppearanceWorkspacePreferences>;
  if (candidate.schemaVersion !== 1) {
    return { ...DEFAULT_APPEARANCE_WORKSPACE, profiles: [] };
  }
  const seen = new Set<string>();
  const profiles = (Array.isArray(candidate.profiles) ? candidate.profiles : [])
    .map(safeProfile)
    .filter((profile): profile is AppearanceProfile => {
      if (!profile || seen.has(profile.id)) return false;
      seen.add(profile.id);
      return true;
    });
  return {
    schemaVersion: 1,
    arrangement: isArrangement(candidate.arrangement)
      ? candidate.arrangement
      : DEFAULT_APPEARANCE_WORKSPACE.arrangement,
    notificationsEnabled: candidate.notificationsEnabled === true,
    profiles
  };
}

export function loadAppearanceWorkspace(storage: StorageLike): AppearanceWorkspacePreferences {
  try {
    const raw = storage.getItem(APPEARANCE_WORKSPACE_STORAGE_KEY);
    return raw
      ? parseAppearanceWorkspace(JSON.parse(raw) as unknown)
      : { ...DEFAULT_APPEARANCE_WORKSPACE, profiles: [] };
  } catch {
    return { ...DEFAULT_APPEARANCE_WORKSPACE, profiles: [] };
  }
}

export function saveAppearanceWorkspace(
  storage: StorageLike,
  preferences: AppearanceWorkspacePreferences
): void {
  storage.setItem(
    APPEARANCE_WORKSPACE_STORAGE_KEY,
    JSON.stringify(parseAppearanceWorkspace(preferences))
  );
}

export function createAppearanceProfile(
  name: string,
  appearance: AppearancePreferences,
  arrangement: WorkspaceArrangement,
  id: string,
  now = new Date().toISOString()
): AppearanceProfile {
  const safeName = name.trim().slice(0, 48);
  if (!safeName) throw new Error("Give this appearance profile a name.");
  if (!/^[a-zA-Z0-9_-]{1,80}$/.test(id)) throw new Error("Invalid appearance profile id.");
  return {
    id,
    name: safeName,
    appearance: parseAppearancePreferences(appearance),
    arrangement,
    createdAt: now,
    updatedAt: now
  };
}

export function upsertAppearanceProfile(
  preferences: AppearanceWorkspacePreferences,
  profile: AppearanceProfile
): AppearanceWorkspacePreferences {
  const existing = preferences.profiles.find((candidate) => candidate.id === profile.id);
  return {
    ...preferences,
    profiles: [
      { ...profile, createdAt: existing?.createdAt ?? profile.createdAt },
      ...preferences.profiles.filter((candidate) => candidate.id !== profile.id)
    ]
  };
}

export function removeAppearanceProfile(
  preferences: AppearanceWorkspacePreferences,
  id: string
): AppearanceWorkspacePreferences {
  return {
    ...preferences,
    profiles: preferences.profiles.filter((profile) => profile.id !== id)
  };
}

export function randomizedAppearance(
  current: AppearancePreferences,
  currentArrangement: WorkspaceArrangement,
  random: () => number = Math.random
): { appearance: AppearancePreferences; arrangement: WorkspaceArrangement } {
  const candidates = APPEARANCE_PALETTES.flatMap((palette) =>
    APPEARANCE_LAYOUTS.flatMap((layout) =>
      WORKSPACE_ARRANGEMENTS.map((arrangement) => ({
        appearance: { ...current, palette, layout },
        arrangement
      }))
    )
  ).filter(
    (candidate) =>
      candidate.appearance.palette !== current.palette ||
      candidate.appearance.layout !== current.layout ||
      candidate.arrangement !== currentArrangement
  );
  const unit = Math.max(0, Math.min(0.999999, random()));
  return candidates[Math.floor(unit * candidates.length)]!;
}

export function applyWorkspaceArrangement(root: HTMLElement, arrangement: WorkspaceArrangement): void {
  root.dataset.workspaceArrangement = arrangement;
}
