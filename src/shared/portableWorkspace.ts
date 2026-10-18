import type { AppearancePreferences, ToolRuntimePreferences, WorkingMemoryResourcePlan } from "./types";

export const PORTABLE_MARKER = "omni-portable.json";
export const PORTABLE_MANIFEST = "portable-workspace.json";
export interface PortablePresentationSettings {
  appearance: AppearancePreferences;
  workspace: { schemaVersion: 1; arrangement: "split" | "focus" | "stacked" | "mixed"; notificationsEnabled: boolean;
    profiles: Array<{ id: string; name: string; appearance: AppearancePreferences; arrangement: "split" | "focus" | "stacked" | "mixed"; createdAt: string; updatedAt: string }> };
  theme: { schemaVersion: 1; pluginId: string; contentSha256: string; themeId: string } | null;
}
export interface PortableWorkspaceSettings { version: 1; tools: ToolRuntimePreferences; presentation: PortablePresentationSettings }
export interface PortableFile { path: string; bytes: number; sha256: string }
export interface PortableWorkspaceManifest {
  format: "omni-portable-workspace"; version: 1; id: string; createdAt: string;
  runtime: { platform: "win32"; architecture: "x64" | "arm64"; workerArchitecture: "x64" | "arm64"; executable: "Omni AGI Studio.exe"; files: PortableFile[] };
  selected: { brainId: string; name: string; originChecksum: string; bundle: PortableFile };
  settings: PortableFile;
  themePackage?: PortableFile;
  authority: "destination-review-required";
  reservation: { policy: "destination-local-planner"; requestedPoolBytes: number; mode: "auto" | "manual"; shippedPaddingBytes: 0 };
  savedContent: "unsanitized-private-archive";
}
export interface PortableWorkspaceCreateRequest { brainId: string; requestId: string; presentation: PortablePresentationSettings }
export interface PortableWorkspaceCreateResult { destination: string; workspaceId: string; brainId: string; files: number; bytes: number; destinationReviewRequired: true }
export interface PortableWorkspaceStatus {
  portable: boolean;
  state: "installed" | "preparation-required" | "preparing" | "review-required" | "ready" | "blocked";
  workspaceId?: string; brainId?: string; name?: string; error?: string;
  presentation?: PortablePresentationSettings;
  settingsSha256?: string;
  phase?: string; label?: string;
  reservation?: { requestedPoolBytes: number; admittedPoolBytes: number | null; mandatoryFreeDiskBytes: number; kind: "logical-shared-pool"; allocatedPaddingBytes: 0 };
}
export interface PortableDestinationReview { reviewId: string; reviewSha256: string; workspaceId: string; brainId: string; plan: WorkingMemoryResourcePlan;
  sourceDriveSelectionsDiscarded: boolean; copiedExternalAuthorityDormant: true; credentialsTransferred: false }
export interface PortableDestinationApproval { reviewId: string; reviewSha256: string }
export interface PortableWorkspaceApi {
  create(request: PortableWorkspaceCreateRequest): Promise<PortableWorkspaceCreateResult | null>;
  /** Uses the ordinary import storage-operation progress/pause/resume/Stop channels. */
  prepare(request: { requestId: string }): Promise<PortableWorkspaceStatus>;
  status(): Promise<PortableWorkspaceStatus>;
  reviewDestination(): Promise<PortableDestinationReview>;
  approveDestination(request: PortableDestinationApproval): Promise<PortableWorkspaceStatus>;
}

function object(value: unknown): Record<string, unknown> { if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Portable settings must be typed data."); return value as Record<string, unknown>; }
function exact(row: Record<string, unknown>, keys: string[]): void { if (Object.keys(row).sort().join("\0") !== keys.sort().join("\0")) throw new Error("Portable settings contain unsupported fields; credentials and raw browser storage are not accepted."); }
function text(value: unknown, pattern: RegExp): string { if (typeof value !== "string" || pattern.exec(value)?.[0] !== value) throw new Error("Invalid portable settings identity."); return value; }
function appearance(value: unknown): AppearancePreferences {
  const row = object(value); exact(row, ["schemaVersion", "mode", "palette", "layout"]);
  if (row.schemaVersion !== 1 || !["system", "light", "dark"].includes(String(row.mode)) || !["violet", "graphite", "spectrum", "aqua"].includes(String(row.palette)) || !["standard", "classic", "expressive", "glass"].includes(String(row.layout))) throw new Error("Invalid portable appearance.");
  return structuredClone(row) as unknown as AppearancePreferences;
}
export function checkedPortablePresentation(value: unknown): PortablePresentationSettings {
  const row = object(value); exact(row, ["appearance", "workspace", "theme"]); const view = appearance(row.appearance), workspace = object(row.workspace);
  exact(workspace, ["schemaVersion", "arrangement", "notificationsEnabled", "profiles"]);
  const arrangement = (value: unknown): PortablePresentationSettings["workspace"]["arrangement"] => {
    if (!["split", "focus", "stacked", "mixed"].includes(String(value))) throw new Error("Invalid portable workspace arrangement.");
    return value as PortablePresentationSettings["workspace"]["arrangement"];
  };
  if (workspace.schemaVersion !== 1 || typeof workspace.notificationsEnabled !== "boolean" || !Array.isArray(workspace.profiles)) throw new Error("Invalid portable workspace settings.");
  const ids = new Set<string>();
  const profiles = workspace.profiles.map(value => { const item = object(value); exact(item, ["id", "name", "appearance", "arrangement", "createdAt", "updatedAt"]);
    const id = text(item.id, /^[A-Za-z0-9_-]{1,80}$/); if (ids.has(id)) throw new Error("Duplicate portable appearance profile."); ids.add(id);
    if (typeof item.name !== "string" || !item.name.trim() || item.name.length > 48 || item.name.includes("\0") || typeof item.createdAt !== "string" || !Number.isFinite(Date.parse(item.createdAt)) || typeof item.updatedAt !== "string" || !Number.isFinite(Date.parse(item.updatedAt))) throw new Error("Invalid portable appearance profile.");
    return { id, name: item.name, appearance: appearance(item.appearance), arrangement: arrangement(item.arrangement), createdAt: item.createdAt, updatedAt: item.updatedAt };
  });
  let theme: PortablePresentationSettings["theme"] = null;
  if (row.theme !== null) { const item = object(row.theme); exact(item, ["schemaVersion", "pluginId", "contentSha256", "themeId"]);
    if (item.schemaVersion !== 1) throw new Error("Invalid portable theme selection.");
    theme = { schemaVersion: 1, pluginId: text(item.pluginId, /^[a-z0-9][a-z0-9._-]{0,127}$/), contentSha256: text(item.contentSha256, /^[a-f0-9]{64}$/), themeId: text(item.themeId, /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/) };
  }
  return { appearance: view, workspace: { schemaVersion: 1, arrangement: arrangement(workspace.arrangement), notificationsEnabled: workspace.notificationsEnabled, profiles }, theme };
}
export function checkedPortableSettings(value: unknown): PortableWorkspaceSettings {
  const row = object(value); exact(row, ["version", "tools", "presentation"]); const tools = object(row.tools); exact(tools, ["approvalTimeoutSeconds"]);
  if (row.version !== 1 || typeof tools.approvalTimeoutSeconds !== "number" || !Number.isSafeInteger(tools.approvalTimeoutSeconds) || tools.approvalTimeoutSeconds < 1 || tools.approvalTimeoutSeconds > 3600) throw new Error("Invalid portable tool preferences.");
  return { version: 1, tools: { approvalTimeoutSeconds: tools.approvalTimeoutSeconds }, presentation: checkedPortablePresentation(row.presentation) };
}
