import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent
} from "react";
import {
  createPresetConfig,
  type ArchitecturePreset,
  type BrainConfig,
  type BrainDocument,
  type BrainExportMode,
  type BrainSummary,
  type BrainStorageOperationEvent,
  type BuildProgressEvent,
  type BuildResourceSelection,
  type CatalogEntry,
  type ChatCancellationState,
  type ChatDeliveryReceiptState,
  type ChatMessage,
  type ChatQueueState,
  type ConversationLedgerEntry,
  type ChatStreamEvent,
  type ChatTurnMetadata,
  type DeleteInstanceRequest,
  type DeleteInstanceResult,
  type DatasetPreviewProgress,
  type DatasetResumeCandidate,
  type GeneratedArtifact,
  type HardwareTier,
  type HardwareProfile as SystemHardwareProfile,
  type IngestResult,
  type MobileGatewayStatus,
  type MobilePairingSession,
  type ModalityGenerationSettings,
  type PersistedDatasetCoverageSummary,
  type PersistedSubstrateOverview,
  type RuntimeJob,
  type InstalledModalityPack,
  type ToolInvocation,
  type ToolPermissionRecord,
  type AgentMergePreview,
  type ActionEvent,
  type AppearanceLayout,
  type AppearanceMode,
  type AppearancePalette,
  type AppearancePreferences,
  type SubstratePage,
  type WorkspaceSnapshot,
  type StoragePoolMode,
  type SystemRamMode,
  type WorkingMemoryMode,
  type WorkingMemoryResourcePlan,
  type ApiTeacherProvider,
  type ApiTeacherProviderStatus,
  type McpTransport,
  type McpServerSummary
} from "@shared/types";
import type {
  BrainActiveModeResult,
  BrainSnapshotSummary
} from "@shared/types";
import { BRAIN_EXPORT_DISCLOSURE } from "@shared/brainExportDisclosure";
import { chatInputCapacity } from "@shared/chatInput";
import {
  EXPERIENCE_UPLOADS,
  type ExperienceUploadKind
} from "@shared/uploadSupport";
import {
  presentTrainingTelemetry,
  type TrainingTelemetryPresentation
} from "@shared/trainingTelemetry";
import type {
  LiveVoicePreferences,
  LiveVoiceState,
  LiveVoiceSynthesisAdapter,
  LiveVoiceSynthesisSession
} from "@shared/liveVoice";
import { demoSummaries, makeDemoBrain } from "./demo";
import { EvolutionWorkspace } from "./EvolutionWorkspace";
import { BrainMapCortex } from "./BrainMapCortex";
import {
  presentJournalKind,
  presentTraceEvidence,
  presentTraceSignalMeasures
} from "./memoryPresentation";
import { Icon, type IconName } from "./icons";
import {
  createTextFrameBatcher,
  mergeChatActionEvent,
  patchChatActionPreview,
  type TextFrameBatcher
} from "./chatStreaming";
import {
  actionImaginationMedia,
  imaginationRevisionCopy,
  imaginationSeedCopy,
  jobImaginationMedia,
  persistedArtifactMediaPresentation
} from "./imaginationPresentation";
import {
  anchoredTimelineScrollTop,
  latestChatTimelineWindow,
  newerChatTimelineWindow,
  olderChatTimelineWindow,
  reconcileChatTimelineWindow,
  type ChatTimelineWindow
} from "./chatTimelineWindow";
import { pendingChatOutputPresentation } from "./chatLoadingPresentation";
import { meanObservedTraceActivation } from "./observedTraceActivation";
import {
  replyCompleteLearningPresentation
} from "./chatPhasePresentation";
import {
  advanceChatGenerationPhase,
  reconcileUncommittedChatOutputs,
  retainUncommittedChatOutput,
  settleUncommittedChatOutput,
  type ChatGenerationPhase,
  type UncommittedChatOutput
} from "./chatGenerationLifecycle";
import {
  EMPTY_CHAT_WORKSPACE_ACTIVITY,
  dataActionAvailableDuringTurn,
  isChatVisibleLearningJob,
  mergeChatVisibleLearningJob,
  workspaceChatActivityPresentation,
  workspaceViewWaitsForForegroundTurn,
  type ChatWorkspaceActivitySnapshot
} from "./chatWorkspaceActivity";
import {
  APPEARANCE_LAYOUTS,
  APPEARANCE_MODES,
  APPEARANCE_PACKS,
  APPEARANCE_PALETTES,
  activeAppearancePack,
  applyAppearanceAttributes,
  defaultAppearanceForPlatform,
  loadAppearancePreferences,
  preferencesForPack,
  resolveColorScheme,
  saveAppearancePreferences
} from "./appearance";
import {
  createBrowserLiveVoiceAdapters,
  createOmniLiveVoiceChatAdapter
} from "./browserLiveVoice";
import { LiveVoiceController, LIVE_VOICE_PACE_RATES } from "./liveVoiceController";
import { liveVoiceStatusCopy } from "./liveVoicePresentation";
import {
  LIVE_VOICE_DELIVERY_MODES,
  LIVE_VOICE_PACES,
  loadLiveVoicePreferences,
  saveLiveVoicePreferences
} from "./liveVoicePreferences";
import { studioUiDestination } from "./studioUiActions";
import { LivePerceptionPanel } from "./LivePerceptionPanel";
import {
  clearSubmittedChatDraft,
  chatMessageDeliveryState,
  chatNoReplyPresentation,
  composerSubmitIntent,
  composerTurnCapabilities,
  isCurrentChatSubmission,
  mergeCommittedChatMessages,
  mergeChatMessagesForPresentation,
  preserveSteeredChatMessage,
  reconcileCompletedChatTurn,
  reconcileOptimisticChatMessages,
  recoverFailedChatDraft,
  receivedChatInputFailureStatus,
  settleOptimisticChatTurn,
  shouldFollowChatOutput
} from "./chatTurnPresentation";
import {
  chatActionCancellationTarget,
  chatActionSearchActivity,
  chatActionStateLabel,
  chatActionTitle,
  cleanChatActionStatus
} from "./chatActionPresentation";
import {
  WORKSPACE_ARRANGEMENTS,
  applyWorkspaceArrangement,
  createAppearanceProfile,
  loadAppearanceWorkspace,
  randomizedAppearance,
  removeAppearanceProfile,
  saveAppearanceWorkspace,
  upsertAppearanceProfile,
  type AppearanceWorkspacePreferences,
  type WorkspaceArrangement
} from "./appearanceWorkspace";
import {
  browserNotificationEnvironment,
  dispatchLocalNotification,
  requestLocalNotifications,
  type LocalNotificationOutcome
} from "./localNotifications";
import {
  MAX_SYSTEM_RAM_SHARE_PERCENT,
  MIN_SYSTEM_RAM_SHARE_PERCENT,
  STORAGE_BOUNDARY_COPY,
  clampSystemRamSharePercent,
  configWithResourceEnvelope,
  contextCapacityBandStyle,
  initialManualSystemRamSharePercent
} from "./resourceEnvelope";
import {
  compactParameterCount,
  neuralParameterAccounting,
  neuralResourceRuntimeRows,
  type NeuralParameterAccountingPresentation
} from "./neuralRuntimePresentation";
import { brainProvenancePresentation } from "./brainProvenancePresentation";
import {
  connectionCountPresentation,
  conciseUiMessage,
  isPristineBrainSummary,
  recoveryPointCreationErrorMessage,
  substrateOverviewCount,
  utf8DraftTokenCount
} from "./uiPresentation";
import {
  activeInitializationTrainingJob,
  initializationIsRunning,
  initializationRecoveryPresentation,
  initializationRunningPresentation
} from "./initializationRecoveryPresentation";
import {
  GIB_BYTES,
  INITIAL_WEB_CRAWL_RESERVE_BYTES,
  plannedTrainingSourceBytes
} from "./buildResourcePlanning";
import { nativeSystemToolExamples } from "./systemToolPresentation";
import { hydrateLibrarySubstrateTotals } from "./librarySubstrateTotals";
import {
  runtimeJobIsActive,
  runtimeJobProgressPresentation,
  runtimeJobScopedDatasetCoverage,
  runtimeJobStopPresentation
} from "./runtimeJobPresentation";
import { webLearningUrlAllowed } from "./remoteUrlPolicy";
import {
  AgentForkSubmissionGate,
  agentForkCompletionPresentation
} from "./agentForkPresentation";
import {
  chatCancellationStatus,
  chatQueueStatus
} from "./chatRuntimePresentation";
import {
  workspaceTelemetryDelta,
  workspaceTelemetryPresentation,
  type WorkspaceTelemetryDelta,
  type WorkspaceTelemetryPresentation
} from "./workspaceTelemetryPresentation";
import {
  ChatAttachmentOperationGate,
  queueChatAttachmentLearning
} from "./chatAttachmentLearning";

type AppPage = "library" | "build" | "workspace";
type InitializationStopIntent = "pause" | "cancel";
const WORKSPACE_HEALTH_REFRESH_MS = 5_000;
const WORKSPACE_TELEMETRY_REFRESH_MS = 1_250;
type WorkspaceView =
  | "chat"
  | "data"
  | "map"
  | "trace"
  | "imagine"
  | "tools"
  | "settings"
  | "agents"
  | "evolution";
type HardwareChoice = "auto" | "micro" | "personal" | "gpu" | "workstation";
type PermissionLevel = "off" | "ask" | "auto" | "full";
type ModalityId = "vision" | "image" | "audio" | "video";

interface BuilderExtras {
  hardware: HardwareChoice;
  initialTraining: boolean;
  initialResources: BuilderResource[];
  modalities: Record<ModalityId, boolean>;
  tools: Record<string, PermissionLevel>;
}

type BuilderResource =
  | BuildResourceSelection
  | {
      id: string;
      kind: "web";
      label: string;
      itemCount: 1;
      url: string;
    };

const recipeMeta: Array<{
  id: ArchitecturePreset;
  title: string;
  short: string;
  icon: IconName;
  color: string;
  features: string[];
}> = [
  {
    id: "whole-brain",
    title: "Whole Brain",
    short: "Ternary cortex, spikes, liquid dynamics, and idea memory in one adaptive system.",
    icon: "brain",
    color: "violet",
    features: ["Ternary forward", "STDP", "CfC", "VSA"]
  },
  {
    id: "ternary",
    title: "Ternary Cortex",
    short: "A compact decoder whose effective synapses settle at −1, 0, or +1.",
    icon: "memory",
    color: "blue",
    features: ["BitLinear", "Decoder", "Efficient"]
  },
  {
    id: "neuromorphic",
    title: "Neuromorphic Lab",
    short: "Leaky integrate-and-fire neurons with local connection learning.",
    icon: "pulse",
    color: "orange",
    features: ["LIF", "STDP", "Sparse"]
  },
  {
    id: "liquid",
    title: "Liquid Cortex",
    short: "A continuous-time recurrent mind that adapts its own time constants.",
    icon: "wave",
    color: "cyan",
    features: ["CfC", "LTC", "Temporal"]
  },
  {
    id: "symbolic",
    title: "VSA Idea Brain",
    short: "Compositional hypervectors store ideas and relationships above tokens.",
    icon: "sparkles",
    color: "pink",
    features: ["HDC", "Ideas", "Graph"]
  },
  {
    id: "custom",
    title: "Custom Architecture",
    short: "Start with a clean blueprint and choose every cognitive subsystem.",
    icon: "settings",
    color: "silver",
    features: ["Modular", "Advanced", "Yours"]
  }
];

const navItems: Array<{ id: WorkspaceView; label: string; icon: IconName }> = [
  { id: "chat", label: "Conversation", icon: "chat" },
  { id: "data", label: "Data & training", icon: "database" },
  { id: "map", label: "Brain map", icon: "brain" },
  { id: "trace", label: "Trace & journal", icon: "trace" },
  { id: "imagine", label: "Imagination", icon: "sparkles" },
  { id: "tools", label: "Tools & permissions", icon: "terminal" },
  { id: "settings", label: "Device & runtime", icon: "settings" },
  { id: "agents", label: "Forks & agents", icon: "fork" },
  { id: "evolution", label: "Evolution", icon: "pulse" }
];

const buildToolProtocolIds: Record<string, string[]> = {
  files: ["system.files"],
  shell: ["system.shell"],
  code: ["code.execute"],
  web: ["web.fetch", "web.search"],
  browser: ["browser.automation"],
  device: ["device.input"],
  imagination: ["modality.imagine"],
  agents: ["agent.fork"],
  evolution: ["source.self-modify"]
};

const standardBuildAccess: Record<string, PermissionLevel> = {
  files: "ask",
  shell: "ask",
  code: "ask",
  web: "ask",
  browser: "ask",
  device: "ask",
  imagination: "auto",
  agents: "ask",
  evolution: "ask"
};

function buildAccessLevels(
  enabled: boolean,
  fullAuthority = false
): Record<string, PermissionLevel> {
  return Object.fromEntries(
    Object.keys(standardBuildAccess).map((id) => [
      id,
      !enabled ? "off" : fullAuthority ? "full" : standardBuildAccess[id]!
    ])
  );
}

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function compactNumber(value: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatBytes(value: number) {
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KB`;
  if (value < 1_073_741_824) return `${(value / 1_048_576).toFixed(1)} MB`;
  if (value < 1_099_511_627_776) return `${(value / 1_073_741_824).toFixed(1)} GB`;
  return `${(value / 1_099_511_627_776).toFixed(2)} TB`;
}

function wholeGiBTextToBytes(value: string): string | undefined {
  const normalized = value.trim().replace(/^0+(?=\d)/, "");
  if (!/^\d+$/.test(normalized) || normalized === "0") return undefined;
  return (BigInt(normalized) * BigInt(GIB_BYTES)).toString();
}

function relativeTime(value: string) {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60_000));
  if (minutes < 2) return "just now";
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

function presetLabel(preset: ArchitecturePreset) {
  return (
    {
      "whole-brain": "Whole Brain",
      ternary: "Ternary Cortex",
      neuromorphic: "Neuromorphic",
      liquid: "Liquid Cortex",
      symbolic: "VSA Idea Brain",
      custom: "Custom Cortex"
    } satisfies Record<ArchitecturePreset, string>
  )[preset];
}

function instanceKindLabel(summary: BrainSummary): string {
  switch (summary.instanceKind) {
    case "duplicate":
      return "Duplicate instance";
    case "fork":
      return "Forked instance";
    case "imported":
      return "Imported instance";
    default:
      return summary.generation > 0 ? "Lineage instance" : "Original instance";
  }
}

function instanceOriginDescription(summary: BrainSummary): string {
  const sharedCount = summary.originInstanceCount ?? 1;
  if (sharedCount > 1) {
    return `Independent learning state · recovery origin shared by ${sharedCount} instances`;
  }
  return "One live identity · its stored origin is only a recovery point";
}

function brainAdaptationSummary(
  brain: BrainDocument
): BrainSummary["adaptation"] | undefined {
  if (brain.activity?.topAdaptation) {
    return {
      sourceLabel: brain.activity.topAdaptation.sourceLabel,
      learnedRecords: brain.activity.topAdaptation.learnedRecords
    };
  }
  const source = [...brain.trainingSources]
    .filter((candidate) =>
      Number.isSafeInteger(candidate.learnedRecords) &&
      Number(candidate.learnedRecords) > 0
    )
    .sort((left, right) =>
      Number(right.learnedRecords) - Number(left.learnedRecords) ||
      right.importedAt.localeCompare(left.importedAt)
    )[0];
  return source?.learnedRecords
    ? { sourceLabel: source.name, learnedRecords: source.learnedRecords }
    : undefined;
}

function brainTrainingSourceCount(brain: BrainDocument): number {
  return brain.activity?.trainingSourceCount ?? brain.trainingSources.length;
}

function brainSummaryProvenanceFields(
  brain: BrainDocument
): Pick<BrainSummary, "provenance" | "adaptation"> {
  const adaptation = brainAdaptationSummary(brain);
  return {
    ...(brain.provenance ? { provenance: brain.provenance } : {}),
    ...(adaptation ? { adaptation } : {})
  };
}

function Button({
  children,
  icon,
  kind = "secondary",
  className,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & {
  icon?: IconName;
  kind?: "primary" | "secondary" | "ghost" | "danger";
}) {
  return (
    <button type="button" className={cx("button", `button--${kind}`, className)} {...props}>
      {icon ? <Icon name={icon} size={16} /> : null}
      <span>{children}</span>
    </button>
  );
}

function BrandMark({ size = 30 }: { size?: number }) {
  return (
    <span className="brand-mark" style={{ width: size, height: size }} aria-hidden="true">
      <span className="brand-mark__orbit brand-mark__orbit--a" />
      <span className="brand-mark__orbit brand-mark__orbit--b" />
      <span className="brand-mark__core" />
    </span>
  );
}

const appearanceModeLabels: Record<AppearanceMode, string> = {
  system: "Auto",
  light: "Light",
  dark: "Dark"
};

const appearancePaletteLabels: Record<AppearancePalette, string> = {
  violet: "Violet",
  graphite: "Graphite",
  spectrum: "Spectrum",
  aqua: "Aqua"
};

const appearanceLayoutLabels: Record<AppearanceLayout, string> = {
  standard: "Standard",
  classic: "Blocky",
  expressive: "Expressive",
  glass: "Glass"
};

const appearanceLayoutDescriptions: Record<AppearanceLayout, string> = {
  standard: "Balanced",
  classic: "Compact",
  expressive: "Roomy",
  glass: "Layered"
};

const workspaceArrangementLabels: Record<WorkspaceArrangement, string> = {
  split: "Side by side",
  focus: "Full focus",
  stacked: "Top + below",
  mixed: "Mixed canvas"
};

const workspaceArrangementDescriptions: Record<WorkspaceArrangement, string> = {
  split: "Conversation beside the live cortex",
  focus: "Conversation fills the workspace",
  stacked: "Conversation above neural activity",
  mixed: "Cortex leads beside conversation"
};

function AppearanceMenu({
  preferences,
  workspacePreferences,
  resolvedColorScheme,
  onChange,
  onWorkspaceChange,
  onNotificationsChange
}: {
  preferences: AppearancePreferences;
  workspacePreferences: AppearanceWorkspacePreferences;
  resolvedColorScheme: "light" | "dark";
  onChange: (preferences: AppearancePreferences) => void;
  onWorkspaceChange: (preferences: AppearanceWorkspacePreferences) => void;
  onNotificationsChange: (enabled: boolean) => Promise<boolean>;
}) {
  const [open, setOpen] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [notificationBusy, setNotificationBusy] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const activePack = activeAppearancePack(preferences);
  const activePackName = activePack === "custom"
    ? "Custom mix"
    : APPEARANCE_PACKS.find((pack) => pack.id === activePack)?.name ?? "Custom mix";
  const selectedProfile = workspacePreferences.profiles.find(
    (profile) => profile.id === selectedProfileId
  );

  const saveNewProfile = (): void => {
    const name = profileName.trim();
    if (!name) return;
    const id = typeof globalThis.crypto?.randomUUID === "function"
      ? globalThis.crypto.randomUUID()
      : `profile-${Date.now().toString(36)}`;
    const profile = createAppearanceProfile(
      name,
      preferences,
      workspacePreferences.arrangement,
      id
    );
    onWorkspaceChange(upsertAppearanceProfile(workspacePreferences, profile));
    setSelectedProfileId(profile.id);
    setProfileName("");
  };

  const updateSelectedProfile = (): void => {
    if (!selectedProfile) return;
    onWorkspaceChange(
      upsertAppearanceProfile(workspacePreferences, {
        ...selectedProfile,
        appearance: preferences,
        arrangement: workspacePreferences.arrangement,
        updatedAt: new Date().toISOString()
      })
    );
  };

  const applyProfile = (profileId: string): void => {
    const profile = workspacePreferences.profiles.find((candidate) => candidate.id === profileId);
    if (!profile) return;
    setSelectedProfileId(profile.id);
    onChange(profile.appearance);
    onWorkspaceChange({ ...workspacePreferences, arrangement: profile.arrangement });
  };

  const randomize = (): void => {
    const next = randomizedAppearance(preferences, workspacePreferences.arrangement);
    setSelectedProfileId("");
    onChange(next.appearance);
    onWorkspaceChange({ ...workspacePreferences, arrangement: next.arrangement });
  };

  useEffect(() => {
    if (!open) return;
    const closeOnPointer = (event: globalThis.PointerEvent): void => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape") {
        setOpen(false);
        container.current?.querySelector<HTMLButtonElement>(".appearance-trigger")?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <div className="appearance-control" ref={container}>
      <button
        className="appearance-trigger"
        aria-expanded={open}
        aria-haspopup="dialog"
        aria-label="Appearance settings"
        title="Appearance settings"
        onClick={() => setOpen((current) => !current)}
      >
        <Icon name="settings" size={14} />
        <span>Appearance</span>
        <i className="appearance-trigger__swatch" aria-hidden="true" />
      </button>
      {open ? (
        <section className="appearance-panel" role="dialog" aria-label="Appearance settings">
          <header>
            <span>
              <strong>Appearance</strong>
              <small>Choose how the studio looks. Brain identity and learning never change.</small>
              <em>{activePackName} · {preferences.mode === "system" ? `Auto ${resolvedColorScheme}` : resolvedColorScheme}</em>
            </span>
            <button
              className="icon-button"
              aria-label="Close appearance settings"
              onClick={() => setOpen(false)}
            >
              <Icon name="close" size={14} />
            </button>
          </header>

          <fieldset className="appearance-fieldset appearance-profiles">
            <legend>
              <span>Profiles</span>
              <button
                className="appearance-randomizer"
                type="button"
                aria-label="Randomize appearance"
                title="Roll a new palette, surface style, and workspace layout"
                onClick={randomize}
              >
                <i aria-hidden="true"><b /><b /><b /><b /><b /></i>
                <span>Surprise me</span>
              </button>
            </legend>
            {workspacePreferences.profiles.length ? (
              <div className="appearance-profile-list" aria-label="Saved appearance profiles">
                {workspacePreferences.profiles.map((profile) => (
                  <button
                    key={profile.id}
                    className={selectedProfileId === profile.id ? "is-active" : ""}
                    aria-pressed={selectedProfileId === profile.id}
                    onClick={() => applyProfile(profile.id)}
                  >
                    <i aria-hidden="true" />
                    <span>{profile.name}</span>
                  </button>
                ))}
              </div>
            ) : (
              <small>Save combinations you want to return to.</small>
            )}
            <div className="appearance-profile-editor">
              <input
                value={profileName}
                maxLength={48}
                aria-label="New appearance profile name"
                placeholder="Profile name"
                onChange={(event) => setProfileName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") saveNewProfile();
                }}
              />
              <button disabled={!profileName.trim()} onClick={saveNewProfile}>
                <Icon name="plus" size={13} /> Save
              </button>
              {selectedProfile ? (
                <>
                  <button onClick={updateSelectedProfile}>Update</button>
                  <button
                    className="appearance-profile-remove"
                    aria-label={`Remove ${selectedProfile.name} appearance profile`}
                    onClick={() => {
                      onWorkspaceChange(
                        removeAppearanceProfile(workspacePreferences, selectedProfile.id)
                      );
                      setSelectedProfileId("");
                    }}
                  >
                    Remove
                  </button>
                </>
              ) : null}
            </div>
          </fieldset>

          <fieldset className="appearance-fieldset">
            <legend>Color</legend>
            <div className="appearance-segments">
              {APPEARANCE_MODES.map((mode) => (
                <button
                  key={mode}
                  aria-pressed={preferences.mode === mode}
                  className={preferences.mode === mode ? "is-active" : ""}
                  onClick={() => onChange({ ...preferences, mode })}
                >
                  {appearanceModeLabels[mode]}
                </button>
              ))}
            </div>
            <small>
              {preferences.mode === "system"
                ? `Following the OS · currently ${resolvedColorScheme}`
                : `Pinned to ${resolvedColorScheme}`}
            </small>
          </fieldset>

          <fieldset className="appearance-fieldset">
            <legend>One-click styles</legend>
            <div className="appearance-packs">
              {APPEARANCE_PACKS.map((pack) => (
                <button
                  key={pack.id}
                  className={activePack === pack.id ? "is-active" : ""}
                  aria-label={pack.name}
                  aria-pressed={activePack === pack.id}
                  onClick={() => onChange(preferencesForPack(preferences, pack.id))}
                >
                  <i data-pack-preview={pack.id} aria-hidden="true" />
                  <span><strong>{pack.name}</strong><small>{pack.description}</small></span>
                </button>
              ))}
            </div>
          </fieldset>

          <div className="appearance-independent">
            <fieldset className="appearance-fieldset">
              <legend>Palette</legend>
              <div className="appearance-options appearance-options--palette">
                {APPEARANCE_PALETTES.map((palette) => (
                  <button
                    key={palette}
                    aria-label={`${appearancePaletteLabels[palette]} palette`}
                    aria-pressed={preferences.palette === palette}
                    className={preferences.palette === palette ? "is-active" : ""}
                    onClick={() => onChange({ ...preferences, palette })}
                  >
                    <i data-palette-preview={palette} aria-hidden="true" />
                    <span>{appearancePaletteLabels[palette]}</span>
                  </button>
                ))}
              </div>
            </fieldset>
            <fieldset className="appearance-fieldset">
              <legend>Shape and density</legend>
              <div className="appearance-options appearance-options--layout">
                {APPEARANCE_LAYOUTS.map((layout) => (
                  <button
                    key={layout}
                    aria-label={appearanceLayoutLabels[layout]}
                    aria-pressed={preferences.layout === layout}
                    className={preferences.layout === layout ? "is-active" : ""}
                    onClick={() => onChange({ ...preferences, layout })}
                  >
                    <i data-layout-preview={layout} aria-hidden="true"><b /><b /><b /></i>
                    <span>
                      <strong>{appearanceLayoutLabels[layout]}</strong>
                      <small>{appearanceLayoutDescriptions[layout]}</small>
                    </span>
                  </button>
                ))}
              </div>
            </fieldset>
          </div>

          <fieldset className="appearance-fieldset appearance-workspace-layouts">
            <legend>Workspace layout</legend>
            <div className="appearance-layout-gallery">
              {WORKSPACE_ARRANGEMENTS.map((arrangement) => (
                <button
                  key={arrangement}
                  aria-label={workspaceArrangementLabels[arrangement]}
                  aria-pressed={workspacePreferences.arrangement === arrangement}
                  className={workspacePreferences.arrangement === arrangement ? "is-active" : ""}
                  onClick={() => onWorkspaceChange({ ...workspacePreferences, arrangement })}
                >
                  <i data-arrangement-preview={arrangement} aria-hidden="true">
                    <b /><b /><b />
                  </i>
                  <span>
                    <strong>{workspaceArrangementLabels[arrangement]}</strong>
                    <small>{workspaceArrangementDescriptions[arrangement]}</small>
                  </span>
                </button>
              ))}
            </div>
          </fieldset>

          <fieldset className="appearance-fieldset appearance-notifications">
            <legend>Notifications</legend>
            <button
              type="button"
              role="switch"
              aria-checked={workspacePreferences.notificationsEnabled}
              disabled={notificationBusy}
              onClick={() => {
                const next = !workspacePreferences.notificationsEnabled;
                setNotificationBusy(true);
                void onNotificationsChange(next).finally(() => setNotificationBusy(false));
              }}
            >
              <span>
                <strong>Desktop task alerts</strong>
                <small>Quiet completion and error alerts only while Omni is in the background.</small>
              </span>
              <i aria-hidden="true"><b /></i>
            </button>
          </fieldset>
        </section>
      ) : null}
    </div>
  );
}

function Titlebar({
  page,
  brain,
  demo,
  onLibrary,
  appearance,
  appearanceWorkspace,
  resolvedColorScheme,
  onAppearanceChange,
  onAppearanceWorkspaceChange,
  onNotificationsChange
}: {
  page: AppPage;
  brain: BrainDocument | null;
  demo: boolean;
  onLibrary: () => void;
  appearance: AppearancePreferences;
  appearanceWorkspace: AppearanceWorkspacePreferences;
  resolvedColorScheme: "light" | "dark";
  onAppearanceChange: (preferences: AppearancePreferences) => void;
  onAppearanceWorkspaceChange: (preferences: AppearanceWorkspacePreferences) => void;
  onNotificationsChange: (enabled: boolean) => Promise<boolean>;
}) {
  return (
    <header className="titlebar">
      <div className="titlebar__drag">
        <button className="brand" onClick={onLibrary} aria-label="Open brain library">
          <BrandMark />
          <span className="brand__name">Omni</span>
          <span className="brand__studio">AGI Studio</span>
        </button>
        <span className="titlebar__divider" />
        <div className="breadcrumbs" aria-label="Current location">
          <span>{page === "library" ? "Brain Library" : page === "build" ? "Build a brain" : "Brain Library"}</span>
          {page === "workspace" && brain ? (
            <>
              <Icon name="arrow" size={13} />
              <strong>{brain.name}</strong>
            </>
          ) : null}
        </div>
        <div className="titlebar__status">
          <span className={cx("status-dot", demo ? "status-dot--demo" : "status-dot--live")} />
          <span>{demo ? "Design preview" : "Local engine"}</span>
          <span className="titlebar__status-detail">{demo ? "No engine connected" : "Private"}</span>
        </div>
        <AppearanceMenu
          preferences={appearance}
          workspacePreferences={appearanceWorkspace}
          resolvedColorScheme={resolvedColorScheme}
          onChange={onAppearanceChange}
          onWorkspaceChange={onAppearanceWorkspaceChange}
          onNotificationsChange={onNotificationsChange}
        />
      </div>
    </header>
  );
}

function EmptyVisual() {
  return (
    <div className="empty-visual" aria-hidden="true">
      <span className="empty-visual__ring empty-visual__ring--one" />
      <span className="empty-visual__ring empty-visual__ring--two" />
      <span className="empty-visual__ring empty-visual__ring--three" />
      <span className="empty-visual__center">
        <BrandMark size={48} />
      </span>
      {Array.from({ length: 8 }).map((_, index) => (
        <span key={index} className={`empty-visual__node empty-visual__node--${index + 1}`} />
      ))}
    </div>
  );
}

function LibraryPage({
  summaries,
  loading,
  onOpen,
  onDuplicate,
  onDelete,
  onBuild,
  onImport,
  demo
}: {
  summaries: BrainSummary[];
  loading: boolean;
  onOpen: (summary: BrainSummary) => void;
  onDuplicate: (summary: BrainSummary) => void;
  onDelete: (summary: BrainSummary) => void;
  onBuild: () => void;
  onImport: () => void;
  demo: boolean;
}) {
  const [query, setQuery] = useState("");
  const visible = summaries.filter((brain) => brain.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));

  return (
    <main className="library-page">
      <section className="library-hero">
        <div className="library-hero__copy">
          <div className="eyebrow">
            <span className="eyebrow__line" />
            Persistent intelligence, locally grown
          </div>
          <h1>
            Build minds that
            <br />
            <span>keep becoming.</span>
          </h1>
          <p>
            Create a private, adaptive intelligence whose experiences become pathways—not prompt history.
          </p>
          <div className="library-hero__actions">
            <Button kind="primary" icon="plus" onClick={onBuild}>
              Build a new brain
            </Button>
            <Button icon="upload" onClick={onImport} title="Import a portable native OmniCortex brain">
              Import .omni
            </Button>
          </div>
          <div className="privacy-note">
            <Icon name="check" size={14} />
            <span>Local-first</span>
            <i />
            <span>No hidden behavioral prompt</span>
            <i />
            <span>Your data stays yours</span>
          </div>
        </div>
        <EmptyVisual />
      </section>

      <section className="library-content">
        <div className="section-heading">
          <div>
            <h2>Your instances</h2>
            <p>{loading ? "Looking for local instances…" : `${summaries.length} independent identities on this device`}</p>
          </div>
          <label className="search-field">
            <Icon name="search" size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search brains"
              aria-label="Search brains"
            />
            <kbd>Ctrl K</kbd>
          </label>
        </div>

        {visible.length > 0 ? (
          <div className="brain-grid">
            {visible.map((brain) => {
              const meta = recipeMeta.find((recipe) => recipe.id === brain.preset) ?? recipeMeta[0]!;
              const provenance = brainProvenancePresentation({
                provenance: brain.provenance,
                adaptation: brain.adaptation
              });
              const connections = connectionCountPresentation({
                persistedConnections: brain.substrateTotals?.synapses,
                mirroredConnections: brain.synapses,
                cumulativeUpdates: brain.neuralUpdates
              });
              return (
                <article
                  className="brain-card"
                  key={brain.id}
                  tabIndex={0}
                  onClick={() => onOpen(brain)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") onOpen(brain);
                  }}
                >
                  <div className="brain-card__top">
                    <div className={cx("brain-avatar", `brain-avatar--${meta.color}`)}>
                      <Icon name={meta.icon} size={25} />
                      {brain.activeMode ? <span className="brain-avatar__live" title="Active Mode" /> : null}
                    </div>
                    <div className="brain-card__instance-actions">
                      <button
                        className="brain-card__duplicate"
                        aria-label={`Duplicate instance ${brain.name}`}
                        title={`Duplicate instance ${brain.name} with copy-on-write neural storage`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onDuplicate(brain);
                        }}
                      >
                        <Icon name="copy" size={13} /> Duplicate instance
                      </button>
                      <button
                        className="brain-card__delete"
                        aria-label={`Delete instance ${brain.name}`}
                        title={`Permanently delete instance ${brain.name}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          onDelete(brain);
                        }}
                      >
                        <Icon name="close" size={13} />
                      </button>
                    </div>
                  </div>
                  <div className="brain-card__identity">
                    <h3>{brain.name}</h3>
                    <span>{instanceKindLabel(brain)}{brain.activeMode ? " · Active Mode" : ""}</span>
                  </div>
                  <p className="brain-card__thought">
                    {instanceOriginDescription(brain)}
                  </p>
                  {provenance ? (
                    <p
                      className="brain-card__provenance"
                      title={provenance.ariaLabel}
                      aria-label={provenance.ariaLabel}
                    >
                      <strong>{provenance.originLabel}</strong>
                      <span>{provenance.compactLabel}</span>
                    </p>
                  ) : null}
                  {isPristineBrainSummary(brain) ? (
                    <div className="brain-card__stats brain-card__stats--empty">
                      <span>
                        <strong>Ready to learn</strong>
                        <small>New origin · no learned pathways yet</small>
                      </span>
                    </div>
                  ) : (
                    <div className="brain-card__stats">
                      <span>
                        <strong>{compactNumber(brain.substrateTotals?.assemblies ?? brain.concepts)}</strong> assemblies
                      </span>
                      {connections.topologyPending ? (
                        <span title="Unique connection topology is still loading from durable substrate storage">
                          <strong>{compactNumber(connections.cumulativeUpdates)}</strong> learned updates
                        </span>
                      ) : (
                        <span
                          title={`${connections.uniqueConnections.toLocaleString("en-US")} unique live connections; ${connections.cumulativeUpdates.toLocaleString("en-US")} cumulative updates`}
                        >
                          <strong>{compactNumber(connections.uniqueConnections)}</strong> unique connections
                        </span>
                      )}
                      <span>
                        <strong>{compactNumber(brain.inferenceCount ?? 0)}</strong> completed turns
                      </span>
                    </div>
                  )}
                  <div className="brain-card__footer">
                    <span>{relativeTime(brain.updatedAt)}</span>
                    <span className="brain-card__open">
                      Open mind <Icon name="arrow" size={14} />
                    </span>
                  </div>
                </article>
              );
            })}
            <button className="new-brain-card" onClick={onBuild}>
              <span className="new-brain-card__plus">
                <Icon name="plus" size={23} />
              </span>
              <strong>Build another instance</strong>
              <span>Create one independent identity and continuous chat</span>
            </button>
          </div>
        ) : (
          <div className="library-empty">
            <BrandMark size={58} />
            <h3>{query ? "No matching instances" : "This library is waiting for its first instance"}</h3>
            <p>{query ? "Try a different name." : "Create one identity and let it keep learning."}</p>
            {!query ? (
              <Button kind="primary" icon="plus" onClick={onBuild}>
                Build a brain
              </Button>
            ) : null}
          </div>
        )}

      <div className="library-bottom">
          <div className="system-card">
            <div className="system-card__icon">
              <Icon name="pulse" size={20} />
            </div>
            <div>
              <strong>Compute is ready</strong>
              <span>Local runtime · Cross-platform · Hardware scaling enabled</span>
            </div>
            <span className={cx("system-card__pill", demo && "system-card__pill--demo")}>
              {demo ? "DEMO · no trained brain" : "Engine online"}
            </span>
          </div>
          <div className="tip-card">
            <Icon name="sparkles" size={18} />
            <span>
              <strong>Storage:</strong> Each card is one live identity. Origins are recovery points, shared by content hash—not extra running brains.
            </span>
          </div>
        </div>
      </section>
    </main>
  );
}

function DeleteInstanceDialog({
  target,
  busy,
  onCancel,
  onConfirm
}: {
  target: Pick<BrainSummary, "id" | "name">;
  busy: boolean;
  onCancel: () => void;
  onConfirm: (request: DeleteInstanceRequest) => Promise<void>;
}) {
  const [stage, setStage] = useState<1 | 2 | 3>(1);
  const [typedName, setTypedName] = useState("");
  const [finalConfirmation, setFinalConfirmation] = useState("");
  const finalPhrase = `PERMANENTLY DELETE ${target.name}`;

  return (
    <div className="instance-delete-backdrop" role="presentation">
      <section
        className="instance-delete-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="instance-delete-title"
      >
        <div className="instance-delete-dialog__icon"><Icon name="warning" size={22} /></div>
        <span className="instance-delete-dialog__step">Confirmation {stage} of 3</span>
        <h2 id="instance-delete-title">Delete instance {target.name}?</h2>
        {stage === 1 ? (
          <>
            <p>
              This permanently removes this exact live identity, its learned state, chat,
              journal, and local artifacts. Other instances and shared origins are not deleted.
            </p>
            <strong className="instance-delete-dialog__irreversible">
              This instance cannot be recovered after the final confirmation.
            </strong>
          </>
        ) : stage === 2 ? (
          <label>
            <span>Type the exact instance name to continue:</span>
            <code>{target.name}</code>
            <input
              autoFocus
              value={typedName}
              onChange={(event) => setTypedName(event.target.value)}
              aria-label="Exact instance name"
              disabled={busy}
            />
          </label>
        ) : (
          <label>
            <span>Final irreversible confirmation. Type exactly:</span>
            <code>{finalPhrase}</code>
            <input
              autoFocus
              value={finalConfirmation}
              onChange={(event) => setFinalConfirmation(event.target.value)}
              aria-label="Final permanent deletion phrase"
              disabled={busy}
            />
          </label>
        )}
        <div className="instance-delete-dialog__actions">
          <Button disabled={busy} onClick={onCancel}>Cancel</Button>
          {stage === 1 ? (
            <Button kind="primary" disabled={busy} onClick={() => setStage(2)}>
              I understand, continue
            </Button>
          ) : stage === 2 ? (
            <Button
              kind="primary"
              disabled={busy || typedName !== target.name}
              onClick={() => setStage(3)}
            >
              Name matches, continue
            </Button>
          ) : (
            <button
              className="instance-delete-dialog__delete"
              disabled={busy || finalConfirmation !== finalPhrase}
              onClick={() => void onConfirm({
                brainId: target.id,
                acknowledgedIrreversible: true,
                typedName,
                finalConfirmation
              })}
            >
              {busy ? "Deleting exact instance…" : "Permanently delete instance"}
            </button>
          )}
        </div>
      </section>
    </div>
  );
}

function InitializationCancelDialog({
  brainName,
  busy,
  onClose,
  onConfirm
}: {
  brainName: string;
  busy: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent): void => {
      if (event.key !== "Escape" || busy) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, onClose]);

  return (
    <div className="instance-delete-backdrop initialization-cancel-backdrop" role="presentation">
      <section
        className="instance-delete-dialog initialization-cancel-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="initialization-cancel-title"
        aria-describedby="initialization-cancel-detail"
      >
        <div className="instance-delete-dialog__icon"><Icon name="pause" size={22} /></div>
        <span className="instance-delete-dialog__step">Recoverable cancellation</span>
        <h2 id="initialization-cancel-title">Cancel training for {brainName}?</h2>
        <p id="initialization-cancel-detail">
          This stops the active worker job and returns this mind to a paused, resumable state.
          Work after the most recent durable checkpoint may be replayed when training resumes.
        </p>
        <strong className="initialization-cancel-dialog__recoverable">
          Nothing is deleted. The mind, selected sources, committed learning, and durable cursor stay on this device.
        </strong>
        <div className="instance-delete-dialog__actions initialization-cancel-dialog__actions">
          <Button autoFocus disabled={busy} onClick={onClose}>Keep training</Button>
          <Button
            kind="danger"
            icon="close"
            disabled={busy}
            onClick={() => void onConfirm()}
          >
            {busy ? "Cancelling training…" : "Confirm cancel training"}
          </Button>
        </div>
      </section>
    </div>
  );
}

function Toggle({
  checked,
  onChange,
  label,
  description,
  disabled
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  description?: string;
  disabled?: boolean;
}) {
  return (
    <label className={cx("toggle-row", disabled && "is-disabled")}>
      <span>
        <strong>{label}</strong>
        {description ? <small>{description}</small> : null}
      </span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        disabled={disabled}
      />
      <i aria-hidden="true">
        <b />
      </i>
    </label>
  );
}

const simpleBuildSteps = [
  ["Identity", "Name this persistent mind"],
  ["Senses & data", "Add its first experiences"],
  ["Memory & storage", "Size from this device and data"],
  ["Access", "Choose what it may use"]
] as const;

/**
 * The stable v1 build flow deliberately exposes intentions instead of neural
 * personality knobs. Hardware profiling owns scale; the substrate owns its
 * organic drive dynamics.
 */
function SimpleBuildWizard({
  onCancel,
  onCreate
}: {
  onCancel: () => void;
  onCreate: (config: BrainConfig, extras: BuilderExtras) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("Nova");
  const [building, setBuilding] = useState(false);
  const [detectedHardware, setDetectedHardware] = useState<SystemHardwareProfile | null>(null);
  const [workingMemoryMode, setWorkingMemoryMode] = useState<WorkingMemoryMode>(
    "auto"
  );
  const [manualContextTokens, setManualContextTokens] = useState(
    "8192"
  );
  const [systemRamMode, setSystemRamMode] = useState<SystemRamMode>(
    "auto"
  );
  const [systemRamSharePercent, setSystemRamSharePercent] = useState(
    65
  );
  const [storagePoolMode, setStoragePoolMode] = useState<StoragePoolMode>(
    "auto"
  );
  const [manualStoragePoolGiB, setManualStoragePoolGiB] = useState(
    "20"
  );
  const [memoryPlan, setMemoryPlan] = useState<WorkingMemoryResourcePlan | null>(null);
  const [memoryPlanPending, setMemoryPlanPending] = useState(false);
  const builderMainContentRef = useRef<HTMLDivElement>(null);
  const [webDraft, setWebDraft] = useState("");
  const [resourceRestoreError, setResourceRestoreError] = useState("");
  const [config] = useState<BrainConfig>(() => createPresetConfig("whole-brain", "Nova"));
  const [extras, setExtras] = useState<BuilderExtras>({
    hardware: "auto",
    initialTraining: false,
    initialResources: [],
    modalities: { vision: true, image: true, audio: true, video: true },
    tools: buildAccessLevels(true)
  });
  const selectedLocalTrainingBytes = extras.initialResources.reduce(
    (total, resource) =>
      total + (resource.kind === "web" ? 0 : resource.bytes ?? 0),
    0
  );
  const selectedWebResourceCount = extras.initialResources.filter(
    (resource) => resource.kind === "web"
  ).length;
  const selectedTrainingBytes = plannedTrainingSourceBytes(extras.initialResources);

  useEffect(() => {
    let active = true;
    if (!window.omni) return () => {
      active = false;
    };
    void window.omni.catalog.hardwareProfile().then((profile) => {
      if (active) setDetectedHardware(profile);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    builderMainContentRef.current?.querySelector<HTMLHeadingElement>("h2")?.focus();
  }, [step]);

  useEffect(() => {
    if (!window.omni || !detectedHardware) return;
    let active = true;
    const timer = window.setTimeout(() => {
      setMemoryPlanPending(true);
      void window.omni!.catalog.resourcePlan({
        mode: workingMemoryMode,
        hardwareTier: extras.hardware === "auto" ? detectedHardware.recommendedTier : extras.hardware,
        systemRamMode,
        acceleratorAvailable: detectedHardware.gpu.available,
        trainingSourceBytes: selectedTrainingBytes,
        storagePoolMode,
        ...(systemRamMode === "manual" ? { systemRamSharePercent } : {}),
        ...(storagePoolMode === "manual"
          ? { storagePoolBytes: wholeGiBTextToBytes(manualStoragePoolGiB) ?? "" }
          : {}),
        ...(workingMemoryMode === "manual"
          ? { requestedContextTokens: manualContextTokens }
          : {})
      }).then((plan) => {
        if (active) setMemoryPlan(plan);
      }).catch(() => {
        if (active) setMemoryPlan(null);
      }).finally(() => {
        if (active) setMemoryPlanPending(false);
      });
    }, workingMemoryMode === "manual" || systemRamMode === "manual" || storagePoolMode === "manual" ? 180 : 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [
    detectedHardware,
    manualContextTokens,
    selectedTrainingBytes,
    manualStoragePoolGiB,
    storagePoolMode,
    systemRamMode,
    systemRamSharePercent,
    workingMemoryMode,
    extras.hardware
  ]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void window.omni.data.listBuildResources().then((resources) => {
      if (!active || resources.length === 0) return;
      setExtras((current) => {
        const merged = new Map(
          current.initialResources.map((resource) => [resource.id, resource])
        );
        for (const resource of resources) merged.set(resource.id, resource);
        return { ...current, initialResources: [...merged.values()] };
      });
    }).catch((error: unknown) => {
      if (active) {
        setResourceRestoreError(
          error instanceof Error
            ? error.message
            : "Pending Build resources could not be restored."
        );
      }
    });
    return () => {
      active = false;
    };
  }, []);

  const selectInitialResource = async (
    kind: BuildResourceSelection["kind"],
    selection: ExperienceUploadKind = "files"
  ): Promise<void> => {
    if (!window.omni) {
      const demoSelection: BuildResourceSelection = {
        id: `demo-${kind}-${Date.now()}`,
        kind,
        label:
          kind === "folder"
            ? "Example dataset folder"
            : `Example selected ${EXPERIENCE_UPLOADS[selection].shortLabel}`,
        itemCount: kind === "folder" ? 1 : 3
      };
      setExtras((current) => ({
        ...current,
        initialResources: [...current.initialResources, demoSelection]
      }));
      return;
    }
    const selectedResource = await window.omni.data.selectBuildResources(
      kind,
      selection
    );
    if (!selectedResource) return;
    setExtras((current) => ({
      ...current,
      initialResources: [...current.initialResources, selectedResource]
    }));
  };

  const addWebResource = (): void => {
    const url = webDraft.trim();
    if (!webLearningUrlAllowed(url)) return;
    setExtras((current) => ({
      ...current,
      initialResources: [
        ...current.initialResources,
        {
          id: `web-${Date.now()}-${Math.random().toString(16).slice(2)}`,
          kind: "web",
          label: new URL(url).hostname,
          itemCount: 1,
          url
        }
      ]
    }));
    setWebDraft("");
  };

  const removeInitialResource = (resource: BuilderResource): void => {
    setExtras((current) => ({
      ...current,
      initialResources: current.initialResources.filter(
        (candidate) => candidate.id !== resource.id
      )
    }));
    if (resource.kind !== "web") {
      void window.omni?.data.discardBuildResource(resource.id);
    }
  };

  const submit = async () => {
    const extendedConfig = {
      ...config,
      name: name.trim(),
      preset: config.preset,
      runtime: "adaptive-core" as const,
      description: config.description,
      // Stable instances use adaptive retention. Recent activity is temporary,
      // useful experience changes neural state, and source provenance is
      // retained without keeping a routine verbatim archive. The persisted
      // recipe ID is a compatibility identifier, not public wording.
      onlineLearning: true,
      retainSourceText: false,
      memoryRecipe: config.memoryRecipe,
      recursiveImprovement: true,
      idleCognition: true,
      workingMemoryMode,
      workingMemorySlots: memoryPlan?.selectedItems ?? config.workingMemorySlots,
      contextWindowTokens:
        memoryPlan?.context.selectedTokens ?? config.contextWindowTokens,
      memoryOffloadBytes: memoryPlan?.resources.configuredMemorySpillBytes ?? 0,
      memoryResidentItems: memoryPlan?.offload.residentMemoryItems ?? config.memoryResidentItems,
      memoryOffloadSlowdownPercent: memoryPlan?.offload.estimatedSlowdownPercent ?? 0,
      systemRamMode,
      systemRamSharePercent: systemRamMode === "manual" ? systemRamSharePercent : 0,
      storagePoolMode: memoryPlan?.resources.storagePoolMode ?? "auto",
      storagePoolBytes: memoryPlan?.resources.sharedStoragePoolBytes ?? 0,
      storageBytesPerSecond: memoryPlan?.offload.benchmark.storageBytesPerSecond ?? 0,
      extendedWorkingMemory: workingMemoryMode === "extended"
    } as BrainConfig;
    setBuilding(true);
    try {
      await onCreate(extendedConfig, {
        ...extras,
        initialTraining: extras.initialResources.length > 0
      });
    } finally {
      setBuilding(false);
    }
  };

  const systemAccessEnabled = Object.values(extras.tools).some(
    (level) => level !== "off"
  );
  const fullAuthorityEnabled = systemAccessEnabled &&
    Object.values(extras.tools).every((level) => level === "full");
  const storageRequiredGiB = Math.max(
    1,
    Math.ceil((memoryPlan?.resources.requiredStoragePoolBytes ?? 20 * GIB_BYTES) / GIB_BYTES)
  );
  const storageMaximumGiB = Math.max(
    1,
    Math.floor((memoryPlan?.resources.maximumStoragePoolBytes ?? 20 * GIB_BYTES) / GIB_BYTES)
  );
  const storageSliderMinimumGiB = Math.min(storageRequiredGiB, storageMaximumGiB);
  const storageSliderUnavailable = Boolean(
    memoryPlan && storageRequiredGiB > storageMaximumGiB
  );
  const chooseStoragePoolMode = (mode: StoragePoolMode): void => {
    if (mode === "manual" && storagePoolMode !== "manual") {
      setManualStoragePoolGiB(
        String(Math.max(
          1,
          Math.ceil(
            (memoryPlan?.resources.sharedStoragePoolBytes ?? 20 * GIB_BYTES) / GIB_BYTES
          )
        ))
      );
    }
    setStoragePoolMode(mode);
  };
  const memoryPlanBlocksProgress = memoryPlanPending || (
    memoryPlan ? !memoryPlan.allowed : Boolean(window.omni)
  );

  const identityStage = (
    <div className="builder-stage simple-builder-stage" key="identity">
      <div className="builder-stage__intro">
        <span className="stage-number">01</span>
        <div>
          <h2 tabIndex={-1}>Who are you creating?</h2>
          <p>Each build becomes one persistent identity with an immutable origin and one continuous conversation.</p>
        </div>
      </div>
      <label className="name-field simple-name-field">
        <span>Name</span>
        <input
          value={name}
          maxLength={40}
          onChange={(event) => setName(event.target.value)}
          placeholder="Name this mind"
          autoFocus
        />
        <small>The display name can change later. Its lineage and origin cannot.</small>
      </label>
      <section className="build-foundation-note" aria-label="Neural origin">
        <span><Icon name="brain" size={20} /></span>
        <div>
          <strong>OmniCortex</strong>
          <p>
            Every new mind initializes its native core locally. Your selected experiences teach this
            same OmniCortex before chat opens.
          </p>
        </div>
      </section>
    </div>
  );

  const memoryStorageStage = (
    <div className="builder-stage simple-builder-stage" key="memory-storage">
      <div className="builder-stage__intro">
        <span className="stage-number">03</span>
        <div>
          <h2 tabIndex={-1}>Memory &amp; storage</h2>
          <p>
            One calculation now sizes active context, Omni-wide RAM, training scratch, and the
            shared drive spill pool from this device and the experiences you selected.
          </p>
        </div>
      </div>
      <div className="memory-storage-inputs" aria-label="Memory and storage calculation">
        <div>
          <small>Selected experiences</small>
          <strong>
            {extras.initialResources.length
              ? `${extras.initialResources.length} source${extras.initialResources.length === 1 ? "" : "s"}`
              : "Conversation first"}
          </strong>
          <span>
            {selectedLocalTrainingBytes > 0 ? `${formatBytes(selectedLocalTrainingBytes)} local data` : "No local source bytes"}
            {selectedWebResourceCount > 0
              ? ` · ${formatBytes(selectedWebResourceCount * INITIAL_WEB_CRAWL_RESERVE_BYTES)} initial web reserve`
              : ""}
          </span>
        </div>
        <div>
          <small>RAM allocation</small>
          <strong>
            {memoryPlan ? formatBytes(memoryPlan.resources.systemRamBudgetBytes) : "Measuring…"}
          </strong>
          <span>
            {memoryPlan
              ? `${memoryPlan.resources.systemRamSharePercent}% of the safe Omni pool`
              : "Device benchmark pending"}
          </span>
        </div>
        <div>
          <small>Shared drive spill pool</small>
          <strong>
            {memoryPlan ? formatBytes(memoryPlan.resources.sharedStoragePoolBytes) : "Measuring…"}
          </strong>
          <span>One reusable pool for training scratch, cold neural state, and growth</span>
        </div>
        <div>
          <small>Native core</small>
          <strong>
            {memoryPlan?.architecture
              ? `${memoryPlan.architecture.exactLogicalParameterCount.toLocaleString()} parameters`
              : "Calculating…"}
          </strong>
          <span>
            {memoryPlan?.architecture
              ? `All initial learned weights use 2-bit packed ternary codes · ${formatBytes(memoryPlan.architecture.packedTernaryWeightBytes)} weights; non-weight runtime state is budgeted separately`
              : "Versioned OmniCortex architecture profile"}
          </span>
        </div>
        <div>
          <small>Disk free now</small>
          <strong>
            {memoryPlan ? formatBytes(memoryPlan.diskSpace.diskFreeBytes) : "Measuring…"}
          </strong>
          <span>
            {memoryPlan
              ? `${formatBytes(memoryPlan.diskSpace.diskTotalBytes)} total · live device reading`
              : "Reading the selected storage volume"}
          </span>
        </div>
        <div>
          <small>Space left after plan</small>
          <strong>
            {memoryPlan
              ? formatBytes(memoryPlan.diskSpace.projectedRemainingBytes)
              : "Calculating…"}
          </strong>
          <span>
            {memoryPlan
              ? `${formatBytes(memoryPlan.diskSpace.projectedAboveReserveBytes)} above the mandatory ${formatBytes(memoryPlan.diskSpace.mandatoryReserveBytes)} reserve`
              : "Includes model, checkpoints, spill, scratch, and future growth"}
          </span>
        </div>
      </div>
      <div className="memory-storage-flow" aria-label="Runtime memory order">
        <span>
          <b>1</b>
          <span><strong>RAM first</strong><small>The active model, context, training step, and hottest pathways stay resident.</small></span>
        </span>
        <Icon name="arrow" size={16} />
        <span>
          <b>2</b>
          <span><strong>Drive spill only when needed</strong><small>Colder neural state, replay, and safe checkpoints page into the shared pool.</small></span>
        </span>
      </div>
      <section className="working-memory-explainer memory-story-card" aria-label="Memory & storage">
        <span><Icon name="memory" size={22} /></span>
        <div>
          <strong>Memory</strong>
          <p>
            Each whole experience changes what is active now, spreads through related pathways,
            and can be rehearsed into longer-lasting learning. Novelty, timing, reuse, importance,
            prediction, interference, stability, and decay continuously decide what strengthens or
            fades; there is no manual memory step or routine second source copy.
          </p>
          <div className="working-memory-planner">
            <div className="working-memory-planner__heading">
              <span>
                <strong>Active context &amp; neural memory</strong>
                <small>Measured from the live device, model, accelerator, and storage before Build and every start.</small>
              </span>
              <em>
                {memoryPlanPending
                  ? "Checking…"
                  : memoryPlan
                    ? `${memoryPlan.context.selectedTokens.toLocaleString()} tokens`
                    : "Detecting device"}
              </em>
            </div>
            {memoryPlan ? (
              <small className="working-memory-planner__context-floor">
                Context uses RAM first; cold attention can use the designated storage pool. Auto selected {memoryPlan.context.autoTokens.toLocaleString()} tokens.
              </small>
            ) : null}
            {memoryPlan ? (
              <dl
                className="working-memory-placement"
                aria-label="Core, active-context, and neural-memory placement"
              >
                <div>
                  <dt>Core + runtime</dt>
                  <dd>
                    <strong>
                      {memoryPlan.offload.modelSpillBytes > 0
                        ? memoryPlan.architecture
                          ? "Packed neural state uses storage assist"
                          : "Neural state needs storage assist"
                        : `${formatBytes(memoryPlan.resources.modelWorkingSetBytes + memoryPlan.resources.runtimeOverheadBytes)} resident`}
                    </strong>
                    <small>
                      {memoryPlan.offload.modelSpillBytes > 0
                        ? `${formatBytes(memoryPlan.offload.modelSpillBytes)} cold neural state · active layers remain in RAM`
                        : `${memoryPlan.architecture?.exactLogicalParameterCount.toLocaleString() ?? "Measured"} logical neural parameters · ${
                            memoryPlan.context.evidence.acceleratorAvailable
                              ? "RAM / accelerator"
                              : "RAM"
                          }`}
                    </small>
                  </dd>
                </div>
                <div>
                  <dt>Active context</dt>
                  <dd>
                    <strong>
                      {memoryPlan.context.selectedTokens <= memoryPlan.context.maximumTokens
                        ? `${formatBytes(memoryPlan.context.evidence.selectedContextResidentBytes)} resident`
                        : "Over the resident/model maximum"}
                    </strong>
                    <small>{memoryPlan.context.selectedTokens.toLocaleString()} tokens · hot activity in RAM, cold attention can use the designated pool</small>
                  </dd>
                </div>
                <div>
                  <dt>Neural memory</dt>
                  <dd>
                    <strong>
                      {memoryPlan.offload.required
                        ? `${memoryPlan.offload.residentMemoryItems.toLocaleString()} hot · ${memoryPlan.offload.pagedMemoryItems.toLocaleString()} paged`
                        : `${memoryPlan.selectedItems.toLocaleString()} resident`}
                    </strong>
                    <small>
                      {memoryPlan.offload.required
                        ? `Cold patterns use storage · about ${memoryPlan.offload.estimatedSlowdownPercent}% slower when paged`
                        : "No neural-memory storage paging"}
                    </small>
                  </dd>
                </div>
              </dl>
            ) : null}
            <div className="working-memory-planner__choices" role="group" aria-label="Working-memory size">
              {(
                [
                  ["auto", "Auto", "Best balance for this device"],
                  ["extended", "Extended", "More context, with storage spill when needed"],
                  ["manual", "Manual", "Choose a physically resident window"]
                ] as const
              ).map(([mode, label, copy]) => (
                <button
                  type="button"
                  key={mode}
                  className={cx(workingMemoryMode === mode && "is-active", `is-${mode}`)}
                  onClick={() => {
                    if (mode === "manual" && workingMemoryMode !== "manual" && memoryPlan) {
                      setManualContextTokens(String(memoryPlan.context.autoTokens));
                    }
                    setWorkingMemoryMode(mode);
                  }}
                  aria-pressed={workingMemoryMode === mode}
                >
                  <strong>{label}</strong>
                  <small>{copy}</small>
                </button>
              ))}
            </div>
            {workingMemoryMode === "manual" ? (
              <div className="working-memory-planner__manual">
                <label>
                  <span>Active context tokens</span>
                  <input
                    type="text"
                    inputMode="numeric"
                    value={manualContextTokens}
                    onChange={(event) => setManualContextTokens(event.target.value.replace(/[^0-9]/g, ""))}
                    aria-invalid={memoryPlan ? !memoryPlan.allowed : undefined}
                  />
                </label>
                <input
                  className="memory-capacity-slider"
                  type="range"
                  min={1}
                  max={Math.max(1, memoryPlan?.context.maximumTokens ?? 1)}
                  step={1}
                  value={Math.max(
                    1,
                    Math.min(
                      memoryPlan?.context.maximumTokens ?? 1,
                      Number.parseInt(manualContextTokens, 10) || 1
                    )
                  )}
                  onChange={(event) => setManualContextTokens(event.target.value)}
                  aria-label="Manual active-context size"
                  aria-describedby="manual-context-range-explanation"
                  style={memoryPlan ? contextCapacityBandStyle(memoryPlan) as React.CSSProperties : undefined}
                />
                <small id="manual-context-range-explanation">
                  Baseline {memoryPlan?.context.floorTokens.toLocaleString() ?? "checking…"} · Auto {memoryPlan?.context.autoTokens.toLocaleString() ?? "checking…"} · RAM + storage maximum {memoryPlan?.context.maximumTokens.toLocaleString() ?? "checking…"}.
                  Green marks the measured suitable range; darker or warmer ends need caution. Typed values pass the same live physical and model-limit check.
                </small>
              </div>
            ) : null}
            {memoryPlan ? (
              <div
                className={cx("working-memory-planner__status", !memoryPlan.allowed && "is-blocked", memoryPlan.offload.required && "uses-storage")}
                role={memoryPlan.allowed ? "status" : "alert"}
                aria-live="polite"
              >
                <Icon name={memoryPlan.allowed ? "check" : "warning"} size={15} />
                <span>
                  <strong>
                    {memoryPlan.allowed
                      ? memoryPlan.offload.required
                        ? "Cold memory may use storage"
                        : "Fits inside the Omni RAM envelope"
                      : "Build is locked until this fits"}
                  </strong>
                  <small>
                    {memoryPlan.allowed
                      ? memoryPlan.offload.required
                        ? `${formatBytes(memoryPlan.resources.configuredMemorySpillBytes)} on-demand spill · about ${memoryPlan.offload.estimatedSlowdownPercent}% slower for memory-heavy work`
                        : `Keeps ${formatBytes(memoryPlan.resources.mandatoryFreeDiskBytes)} free plus ${formatBytes(memoryPlan.resources.checkpointHeadroomBytes)} checkpoint headroom`
                      : memoryPlan.blockers.join(" ")}
                  </small>
                </span>
              </div>
            ) : null}
            <details className="system-ram-budget">
              <summary>
                <span>
                  <strong>Omni-wide RAM</strong>
                  <small>Auto by default · Advanced cap: 30–100% of the safe pool · affects the model, training, memory, media, caches, and agents.</small>
                </span>
                <em>
                  {memoryPlan
                    ? `${memoryPlan.resources.systemRamSharePercent}% · ${memoryPlan.training.storageClass.replaceAll("-", " ")}`
                    : "Auto"}
                </em>
              </summary>
              <div className="system-ram-budget__body">
                <div className="system-ram-budget__modes" role="group" aria-label="Omni-wide RAM policy">
                  <button
                    type="button"
                    className={cx(systemRamMode === "auto" && "is-active")}
                    aria-pressed={systemRamMode === "auto"}
                    onClick={() => setSystemRamMode("auto")}
                  >
                    Auto
                  </button>
                  <button
                    type="button"
                    className={cx(systemRamMode === "manual" && "is-active")}
                    aria-pressed={systemRamMode === "manual"}
                    onClick={() => setSystemRamMode("manual")}
                  >
                    Advanced cap (30–100%)
                  </button>
                </div>
                {systemRamMode === "manual" ? (
                  <div className="system-ram-budget__manual">
                    <label htmlFor="system-ram-percent">
                      <span>Omni share of safe pool (%)</span>
                      <input
                        id="system-ram-percent"
                        type="number"
                        min={MIN_SYSTEM_RAM_SHARE_PERCENT}
                        max={MAX_SYSTEM_RAM_SHARE_PERCENT}
                        step={1}
                        value={systemRamSharePercent}
                        onChange={(event) => setSystemRamSharePercent(Number(event.target.value))}
                        aria-describedby="build-system-ram-explanation"
                        aria-invalid={memoryPlan ? !memoryPlan.allowed : undefined}
                      />
                    </label>
                    <input
                      type="range"
                      min={MIN_SYSTEM_RAM_SHARE_PERCENT}
                      max={MAX_SYSTEM_RAM_SHARE_PERCENT}
                      step={1}
                      value={clampSystemRamSharePercent(systemRamSharePercent)}
                      onChange={(event) => setSystemRamSharePercent(
                        clampSystemRamSharePercent(Number(event.target.value))
                      )}
                      aria-label="Omni share of safely available memory"
                      aria-describedby="build-system-ram-explanation"
                    />
                  </div>
                ) : null}
                <small id="build-system-ram-explanation">
                  100% means all currently available memory inside Omni’s safe pool. The adaptive device/OS reserve is always excluded. {STORAGE_BOUNDARY_COPY}
                </small>
                {memoryPlan ? (
                  <dl className="resource-benchmark-grid" aria-label="Measured device throughput">
                    <div>
                      <dt>Measured RAM-copy throughput</dt>
                      <dd>{formatBytes(memoryPlan.offload.benchmark.memoryBytesPerSecond)}/s</dd>
                      <small>Bounded in-memory buffer copy; not full DRAM bandwidth.</small>
                    </div>
                    <div>
                      <dt>Measured durable storage-write throughput</dt>
                      <dd>{formatBytes(memoryPlan.offload.benchmark.storageBytesPerSecond)}/s</dd>
                      <small>Median sequential write plus sync · {memoryPlan.training.storageClass.replaceAll("-", " ")}</small>
                    </div>
                  </dl>
                ) : null}
                {memoryPlan?.warnings.length ? (
                  <div
                    className="system-ram-budget__warning"
                    role={systemRamMode === "manual" ? "alert" : "status"}
                    aria-live="polite"
                  >
                    <strong>{systemRamMode === "manual" ? "Custom cap warning" : "Device recommendation"}</strong>
                    <ul>
                      {memoryPlan.warnings.map((warning) => <li key={warning}>{warning}</li>)}
                    </ul>
                  </div>
                ) : null}
              </div>
            </details>
            <details className="system-ram-budget storage-pool-budget">
              <summary>
                <span>
                  <strong>Shared drive pool</strong>
                  <small>Auto by default · one device-wide pool is reused by every instance instead of multiplying storage per brain.</small>
                </span>
                <em>
                  {memoryPlan
                    ? `${storagePoolMode === "auto" ? "Auto" : "Custom"} · ${formatBytes(memoryPlan.resources.sharedStoragePoolBytes)}`
                    : storagePoolMode === "auto" ? "Auto" : `${manualStoragePoolGiB || "—"} GiB`}
                </em>
              </summary>
              <div className="system-ram-budget__body">
                <div className="system-ram-budget__modes" role="group" aria-label="Shared drive pool policy">
                  <button
                    type="button"
                    className={cx(storagePoolMode === "auto" && "is-active")}
                    aria-pressed={storagePoolMode === "auto"}
                    onClick={() => chooseStoragePoolMode("auto")}
                  >
                    Auto
                  </button>
                  <button
                    type="button"
                    className={cx(storagePoolMode === "manual" && "is-active")}
                    aria-pressed={storagePoolMode === "manual"}
                    onClick={() => chooseStoragePoolMode("manual")}
                  >
                    Custom size
                  </button>
                </div>
                {storagePoolMode === "manual" ? (
                  <div className="working-memory-planner__manual storage-pool-budget__manual">
                    <label htmlFor="build-storage-pool-gib">
                      <span>Shared pool capacity</span>
                      <input
                        id="build-storage-pool-gib"
                        type="text"
                        inputMode="numeric"
                        value={manualStoragePoolGiB}
                        onChange={(event) => setManualStoragePoolGiB(
                          event.target.value.replace(/[^0-9]/g, "")
                        )}
                        aria-label="Build shared pool capacity"
                        aria-describedby="build-storage-pool-explanation"
                        aria-invalid={memoryPlan ? !memoryPlan.allowed : undefined}
                      />
                    </label>
                    <input
                      type="range"
                      min={storageSliderMinimumGiB}
                      max={storageMaximumGiB}
                      step={1}
                      value={Math.max(
                        storageSliderMinimumGiB,
                        Math.min(
                          storageMaximumGiB,
                          Number.parseInt(manualStoragePoolGiB, 10) || 1
                        )
                      )}
                      disabled={storageSliderUnavailable}
                      onChange={(event) => setManualStoragePoolGiB(event.target.value)}
                      aria-label="Build shared storage pool in GiB"
                      aria-describedby="build-storage-pool-explanation"
                    />
                    <small id="build-storage-pool-explanation">
                      Required {memoryPlan ? formatBytes(memoryPlan.resources.requiredStoragePoolBytes) : "measuring…"} · physical maximum {memoryPlan ? formatBytes(memoryPlan.resources.maximumStoragePoolBytes) : "measuring…"}. The {memoryPlan ? formatBytes(memoryPlan.diskSpace.mandatoryReserveBytes) : "device-adaptive"} free-space reserve remains outside this pool.
                    </small>
                  </div>
                ) : (
                  <small>
                    Auto reports selected training data and budgets scratch space, model state, cold neural pages, checkpoints, and future growth while retaining the device-adaptive OS/recovery reserve.
                  </small>
                )}
              </div>
            </details>
          </div>
        </div>
      </section>
      <details className="research-diagnostics">
        <summary>
          <span><Icon name="pulse" size={16} /> Research diagnostics</span>
          <span>Measured internals <Icon name="chevron" size={14} /></span>
        </summary>
        <div className="research-diagnostics__grid">
          <span><small>Forward synapses</small><strong>Exact −1 · 0 · +1</strong></span>
          <span><small>Working activity</small><strong>Whole-experience recurrent integration</strong></span>
          <span><small>Fast temporal memory</small><strong>LIF + episodic STDP synapses</strong></span>
          <span><small>Spreading recall</small><strong>Distributed assemblies + signed pathways</strong></span>
          <span><small>Slow learning</small><strong>Replay into learned weights</strong></span>
          <span><small>Retention dynamics</small><strong>Salience + interference + stability + decay</strong></span>
          <span><small>Raw-source retention</small><strong>Off · no routine prompt retrieval</strong></span>
          <span><small>Initialization</small><strong>Locally initialized native core</strong></span>
          <span><small>Growth</small><strong>Resource-governed, no model cap</strong></span>
          <span>
            <small>Hardware profile</small>
            <strong>{detectedHardware ? `${detectedHardware.recommendedTier} · ${detectedHardware.logicalCpus} threads` : "Detected at build time"}</strong>
          </span>
        </div>
      </details>
    </div>
  );

  const dataStage = (
    <div className="builder-stage simple-builder-stage" key="senses">
      <div className="builder-stage__intro">
        <span className="stage-number">02</span>
        <div>
          <h2 tabIndex={-1}>What can it experience first?</h2>
          <p>Enable shared neural senses now. You can add whole datasets, folders, web sources, or media later from the same chat.</p>
        </div>
      </div>
      <h3 className="minor-heading">Perception and imagination</h3>
      <div className="modality-grid">
        {(
          [
            ["vision", "Vision", "Understand images in the shared workspace.", "eye"],
            ["image", "Image imagination", "Generate from internal idea assemblies.", "image"],
            ["audio", "Audio", "Encode and imagine sound or speech.", "volume"],
            ["video", "Video", "Learn and imagine temporal scenes.", "video"]
          ] as const
        ).map(([id, title, copy, icon]) => (
          <div
            key={id}
            className="modality-card is-selected"
          >
            <span className="modality-card__icon"><Icon name={icon} size={20} /></span>
            <span><strong>{title}</strong><p>{copy}</p></span>
            <span className="modality-card__check"><Icon name="check" size={12} /></span>
          </div>
        ))}
      </div>
      <h3 className="minor-heading minor-heading--tools">Initial learning source</h3>
      <div className="build-resource-actions">
        <Button icon="file" onClick={() => void selectInitialResource("files")}>
          Files & datasets
        </Button>
        <Button icon="image" onClick={() => void selectInitialResource("files", "images")}>
          Images
        </Button>
        <Button icon="volume" onClick={() => void selectInitialResource("files", "audio")}>
          Audio
        </Button>
        <Button icon="video" onClick={() => void selectInitialResource("files", "video")}>
          Video
        </Button>
        <Button icon="archive" onClick={() => void selectInitialResource("folder")}>
          Whole folder
        </Button>
        <label className="build-resource-url">
          <Icon name="search" size={15} />
          <input
            value={webDraft}
            onChange={(event) => setWebDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addWebResource();
              }
            }}
            placeholder="https://site.example or http://127.0.0.1:8000"
          />
          <button
            type="button"
            disabled={!webLearningUrlAllowed(webDraft)}
            onClick={addWebResource}
          >
            Add web
          </button>
        </label>
      </div>
      <div className="build-resource-list" aria-label="Initial learning resources">
        {extras.initialResources.map((resource) => (
          <div key={resource.id}>
            <span>
              <Icon
                name={
                  resource.kind === "web"
                    ? "search"
                    : resource.kind === "folder"
                      ? "archive"
                      : "file"
                }
                size={15}
              />
            </span>
            <span>
              <strong>{resource.label}</strong>
              <small>
                {resource.kind === "web"
                  ? "Continuous web crawl"
                  : resource.kind === "folder"
                    ? "Whole folder dataset"
                    : `${resource.itemCount} selected file${resource.itemCount === 1 ? "" : "s"}`}
              </small>
            </span>
            <button
              type="button"
              aria-label={`Remove ${resource.label} from this build`}
              title="Remove from this build; original files are not deleted"
              onClick={() => removeInitialResource(resource)}
            >
              <Icon name="close" size={13} /> Remove
            </button>
          </div>
        ))}
        {resourceRestoreError ? (
          <div className="builder-note builder-note--warning" role="alert">
            <Icon name="warning" size={16} />
            <span>{resourceRestoreError}</span>
          </div>
        ) : null}
        {!extras.initialResources.length ? (
          <p>No data selected. Build learns tool basics, but language starts primitive. Add text data for useful conversation.</p>
        ) : null}
      </div>
      <small className="build-resource-note">
        Remove only changes this build list; it never deletes or modifies the original files.
        Every selected document, dataset, image, audio clip, video, or folder is queued for
        neural encoding and training. Web sources crawl same-site in parallel, including linked
        image, audio, and video resources. Each web source contributes a 1 GiB initial rolling
        scratch reserve to sizing; an indefinite crawl expands the shared pool transactionally
        or pauses before its free-space boundary instead of pretending its final size is zero.
      </small>
    </div>
  );

  const accessStage = (
    <div className="builder-stage simple-builder-stage" key="review">
      <div className="builder-stage__intro">
        <span className="stage-number">04</span>
        <div>
          <h2 tabIndex={-1}>Choose system access.</h2>
          <p>Start with one safe system-access choice. Detailed per-tool permissions remain available inside this brain.</p>
        </div>
      </div>
      <div className="build-access-controls" aria-label="Initial system access">
        <Toggle
          checked={systemAccessEnabled}
          onChange={(enabled) =>
            setExtras((current) => ({
              ...current,
              tools: buildAccessLevels(enabled)
            }))
          }
          label="System access"
          description="Allow visible, cancellable use of files, code, web, browser, device input, imagination, agents, and evolution. Risky actions ask first."
        />
        <Toggle
          checked={fullAuthorityEnabled}
          disabled={!systemAccessEnabled}
          onChange={(enabled) =>
            setExtras((current) => ({
              ...current,
              tools: buildAccessLevels(true, enabled)
            }))
          }
          label="Full Authority"
          description="Master bypass for per-action confirmation. Trusted host validation, audit records, cancellation, and rollback boundaries still apply."
        />
        <small>
          Fine-tune Off, Ask, Auto, or Full separately for every tool later in Tools &amp; permissions.
        </small>
      </div>
      {fullAuthorityEnabled ? (
        <div className="builder-note builder-note--warning">
          <Icon name="warning" size={17} />
          <span>Full authority can act without confirmation. Actions stay traceable and cancellable, and source promotion keeps rollback points.</span>
        </div>
      ) : null}
      <div className="simple-review-card">
        <div>
          <span className="simple-review-card__mark"><BrandMark size={38} /></span>
          <span><small>READY TO CREATE</small><strong>{name || "Unnamed mind"}</strong></span>
        </div>
        <dl>
          <div><dt>Identity</dt><dd>One continuous adaptive mind</dd></div>
          <div><dt>Learning</dt><dd>Continuous · whole-experience adaptation</dd></div>
          <div><dt>Active context</dt><dd>{memoryPlan ? `${memoryPlan.context.selectedTokens.toLocaleString()} tokens · RAM resident` : "Hardware-sized automatically"}</dd></div>
          <div><dt>Omni RAM</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.resources.systemRamBudgetBytes)} · ${memoryPlan.resources.systemRamSharePercent}% safe pool` : "Measuring device"}</dd></div>
          <div><dt>Shared storage</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.resources.sharedStoragePoolBytes)} · ${storagePoolMode === "auto" ? "Auto" : "Custom"}` : "Measuring device"}</dd></div>
          <div><dt>Disk space left</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.diskSpace.projectedRemainingBytes)} · ${formatBytes(memoryPlan.diskSpace.projectedAboveReserveBytes)} above reserve` : "Measuring device"}</dd></div>
          <div><dt>Neural memory</dt><dd>{memoryPlan ? `${memoryPlan.selectedItems.toLocaleString()} recurrent/paged items${memoryPlan.offload.required ? " · storage assisted" : ""}` : "Hardware-sized automatically"}</dd></div>
          <div><dt>Native core</dt><dd>{memoryPlan?.architecture ? `${memoryPlan.architecture.exactLogicalParameterCount.toLocaleString()} logical neural parameters · locally initialized` : "Calculating exact architecture"}</dd></div>
          <div><dt>System access</dt><dd>{!systemAccessEnabled ? "Off" : fullAuthorityEnabled ? "Full Authority" : "On · risky actions ask"}</dd></div>
          <div><dt>Self-improvement</dt><dd>Available under chosen access</dd></div>
          <div><dt>First learning</dt><dd>{extras.initialResources.length ? `Tool basics + ${extras.initialResources.length} queued source${extras.initialResources.length === 1 ? "" : "s"} · ${formatBytes(selectedTrainingBytes)} sizing input` : "Tool basics only · language starts primitive"}</dd></div>
          <div><dt>Modalities</dt><dd>{Object.values(extras.modalities).filter(Boolean).length} enabled</dd></div>
          <div><dt>Behavioral prompt / RLHF</dt><dd>None</dd></div>
        </dl>
      </div>
    </div>
  );
  const stageContent = [identityStage, dataStage, memoryStorageStage, accessStage];
  return (
    <main className="builder-page simple-builder">
      <aside className="builder-sidebar">
        <button
          className="builder-back"
          aria-label="Brain Library"
          onClick={() => {
            for (const resource of extras.initialResources) {
              if (resource.kind !== "web") {
                void window.omni?.data.discardBuildResource(resource.id);
              }
            }
            onCancel();
          }}
        >
          <Icon name="arrow" size={15} /> <span>Brain Library</span>
        </button>
        <div className="builder-sidebar__intro">
          <span>NEW ORIGIN</span>
          <h1>Build a brain</h1>
          <p>Name it, add first experiences, and choose access. Neural details adapt automatically.</p>
        </div>
        <ol className="step-list">
          {simpleBuildSteps.map(([title, copy], index) => (
            <li key={title} className={cx(index === step && "is-active", index < step && "is-complete")}>
              <button
                type="button"
                onClick={() => setStep(index)}
                aria-current={index === step ? "step" : undefined}
                aria-label={`Step ${index + 1}: ${title}. ${copy}`}
              >
                <span>{index < step ? <Icon name="check" size={13} /> : index + 1}</span>
                <span><strong>{title}</strong><small>{copy}</small></span>
              </button>
            </li>
          ))}
        </ol>
        <div className="builder-sidebar__privacy">
          <Icon name="memory" size={17} />
          <span><strong>Local, persistent, inspectable</strong>Its state belongs to this device and this identity.</span>
        </div>
      </aside>
      <section className="builder-main">
        <span className="visually-hidden" role="status" aria-live="polite">
          Step {step + 1} of {simpleBuildSteps.length}: {simpleBuildSteps[step]?.[0] ?? "Build"}
        </span>
        <div ref={builderMainContentRef} className="builder-main__content">{stageContent[step]}</div>
        <footer className="builder-footer">
          <span>
            Step {step + 1} of {simpleBuildSteps.length}
            <i>{simpleBuildSteps.map((_, index) => <b key={index} className={index <= step ? "is-filled" : ""} />)}</i>
          </span>
          <div>
            {step > 0 ? <Button onClick={() => setStep((current) => current - 1)}>Back</Button> : null}
            {step < simpleBuildSteps.length - 1 ? (
              <Button
                kind="primary"
                disabled={!name.trim() || (step === 2 && memoryPlanBlocksProgress)}
                onClick={() => setStep((current) => current + 1)}
              >
                Continue <Icon name="arrow" size={14} />
              </Button>
            ) : (
              <Button
                kind="primary"
                icon={building ? "pulse" : "sparkles"}
                disabled={!name.trim() || building || memoryPlanBlocksProgress}
                onClick={() => void submit()}
              >
                {building ? "Creating origin…" : `Create ${name || "brain"}`}
              </Button>
            )}
          </div>
        </footer>
      </section>
    </main>
  );
}

function WorkspaceShell({
  brain,
  view,
  onView,
  onLibrary,
  onDuplicate,
  onDelete,
  onActiveModeChange,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  view: WorkspaceView;
  onView: (view: WorkspaceView) => void;
  onLibrary: () => void;
  onDuplicate: () => void;
  onDelete: () => void;
  onActiveModeChange: (result: BrainActiveModeResult) => Promise<void>;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [health, setHealth] = useState("Adaptive core ready");
  const [learningSwitchBusy, setLearningSwitchBusy] = useState(false);
  const [compactInspectorOpen, setCompactInspectorOpen] = useState(false);
  const [chatActivity, setChatActivity] = useState<ChatWorkspaceActivitySnapshot>(
    EMPTY_CHAT_WORKSPACE_ACTIVITY
  );
  const chatActivityStatus = workspaceChatActivityPresentation(chatActivity);
  const viewWaitsForTurn = workspaceViewWaitsForForegroundTurn(view, chatActivity);
  const explainForegroundWait = (): void => {
    onToast(
      chatActivityStatus?.waitingLabel ??
        "Waiting for the current reply. Return to Conversation to Queue or Steer."
    );
  };
  const runOutsideForegroundTurn = (action: () => void): void => {
    if (chatActivity.turnActive) {
      explainForegroundWait();
      return;
    }
    action();
  };
  const toggleLearning = async () => {
    if (learningSwitchBusy) return;
    if (chatActivity.turnActive) {
      explainForegroundWait();
      return;
    }
    const enabled = !brain.config.onlineLearning;
    setLearningSwitchBusy(true);
    try {
      const updated = window.omni
        ? await window.omni.brain.setOnlineLearning(brain.id, enabled)
        : { ...brain, config: { ...brain.config, onlineLearning: enabled }, updatedAt: new Date().toISOString() };
      onBrainChange(updated);
      onToast(enabled
        ? "Background parameter training resumed; queued updates are retained."
        : "Background parameter training paused or stopping. Fast conversation learning remains active.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Background training state could not be changed.");
    } finally {
      setLearningSwitchBusy(false);
    }
  };

  useEffect(() => {
    setChatActivity(EMPTY_CHAT_WORKSPACE_ACTIVITY);
  }, [brain.id]);

  useEffect(() => {
    let active = true;
    let refreshTimer: number | undefined;
    if (window.omni) {
      const refreshHealth = async (): Promise<void> => {
        try {
          const result = await window.omni!.brain.health(brain.id);
          if (!active) return;
          setHealth(
            result.ready
              ? `Python engine${result.pid ? ` · PID ${result.pid}` : ""}`
              : result.detail
          );
        } catch {
          if (active) setHealth("Python engine reconnecting…");
        } finally {
          if (active) {
            refreshTimer = window.setTimeout(
              () => void refreshHealth(),
              WORKSPACE_HEALTH_REFRESH_MS
            );
          }
        }
      };
      void refreshHealth();
    } else {
      setHealth("Demo preview · engine disconnected");
    }
    return () => {
      active = false;
      if (refreshTimer !== undefined) window.clearTimeout(refreshTimer);
    };
  }, [brain.id]);

  useEffect(() => {
    if (view !== "chat") setCompactInspectorOpen(false);
  }, [view]);

  useEffect(() => {
    if (!compactInspectorOpen) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape") setCompactInspectorOpen(false);
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [compactInspectorOpen]);

  return (
    <main className="workspace">
      <aside className="workspace-rail">
        <button
          className="rail-library"
          onClick={() => runOutsideForegroundTurn(onLibrary)}
          aria-label={chatActivity.turnActive ? "Brain library; waiting for current reply" : "Brain library"}
        >
          <Icon name="library" size={19} />
          <span className="rail-label">Brain library</span>
        </button>
        <div className="rail-avatar" title={brain.name}>
          <BrandMark size={32} />
          <span />
        </div>
        <nav aria-label="Workspace">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={view === item.id ? "is-active" : ""}
              onClick={() => onView(item.id)}
              aria-label={item.label}
            >
              <Icon name={item.icon} size={19} />
              <span className="rail-label">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="workspace-rail__bottom">
          <button
            aria-label="Open local data folder"
            onClick={() => void window.omni?.window.revealDataFolder()}
          >
            <Icon name="archive" size={19} />
            <span className="rail-label">Local data</span>
          </button>
        </div>
      </aside>
      <section
        className={cx(
          "workspace-body",
          compactInspectorOpen && "workspace-body--inspector-open"
        )}
      >
        <header className="workspace-header">
          <div className="workspace-header__identity">
            <span className="workspace-header__view">{navItems.find((item) => item.id === view)?.label}</span>
            <span className="workspace-header__slash">/</span>
            <strong>{brain.name}</strong>
            <span className="live-chip">
              <i /> Fast conversation learning active
            </span>
          </div>
          {view !== "chat" && chatActivityStatus ? (
            <button
              className="workspace-header__chat-status"
              aria-label={chatActivityStatus.headerAriaLabel}
              title={chatActivityStatus.headerAriaLabel}
              onClick={() => onView("chat")}
            >
              <i />
              <span>{chatActivityStatus.headerLabel}</span>
              <Icon name="chat" size={13} />
            </button>
          ) : null}
          <div className="workspace-header__activity">
            <span>
              <small>Runtime</small>
              <strong>{health}</strong>
            </span>
          </div>
          <div className="workspace-header__actions">
            {view === "chat" ? (
              <button
                className="icon-button workspace-inspector-toggle"
                aria-controls="chat-cortex-panel"
                aria-expanded={compactInspectorOpen}
                aria-label={compactInspectorOpen ? "Close cortex inspector" : "Open cortex inspector"}
                title={compactInspectorOpen ? "Close cortex inspector" : "Open live cortex and runtime card"}
                onClick={() => setCompactInspectorOpen((open) => !open)}
              >
                <Icon name={compactInspectorOpen ? "close" : "brain"} size={16} />
              </button>
            ) : null}
            <button
              className="workspace-duplicate"
              aria-label={chatActivity.turnActive
                ? `Duplicate instance ${brain.name}; waiting for current reply`
                : `Duplicate instance ${brain.name}`}
              title={chatActivity.turnActive
                ? "Waiting for the current reply; return to Conversation to Queue or Steer"
                : "Duplicate this instance as an independent copy-on-write identity"}
              onClick={() => runOutsideForegroundTurn(onDuplicate)}
            >
              <Icon name="copy" size={14} /> <span>Duplicate instance</span>
            </button>
            <button
              className="workspace-delete"
              aria-label={chatActivity.turnActive
                ? `Delete instance ${brain.name}; waiting for current reply`
                : `Delete instance ${brain.name}`}
              title={chatActivity.turnActive
                ? "Waiting for the current reply; return to Conversation to Queue or Steer"
                : "Permanently delete this exact instance"}
              onClick={() => runOutsideForegroundTurn(onDelete)}
            >
              <Icon name="close" size={14} />
            </button>
            <button
              className="icon-button"
              aria-label={brain.config.onlineLearning ? "Pause background training" : "Resume background training"}
              title={chatActivity.turnActive
                ? "Waiting for the current reply; return to Conversation to Queue or Steer"
                : brain.config.onlineLearning
                  ? "Pause slow parameter replay; fast conversation learning remains active"
                  : "Resume queued slow parameter replay"}
              disabled={learningSwitchBusy}
              onClick={() => void toggleLearning()}
            >
              <Icon name={brain.config.onlineLearning ? "pause" : "play"} size={16} />
            </button>
          </div>
        </header>
        <button
          className="workspace-inspector-backdrop"
          aria-label="Close cortex inspector"
          tabIndex={compactInspectorOpen ? 0 : -1}
          onClick={() => setCompactInspectorOpen(false)}
        />
        <ChatWorkspace
          brain={brain}
          hidden={view !== "chat"}
          onBrainChange={onBrainChange}
          onToast={onToast}
          onNavigate={onView}
          onActivityChange={setChatActivity}
        />
        {view !== "chat" ? (
          <div className={cx("workspace-view-host", viewWaitsForTurn && "is-waiting")}>
            <div
              className="workspace-view-host__content"
              inert={viewWaitsForTurn || undefined}
              aria-busy={viewWaitsForTurn || undefined}
            >
              {view === "data" ? (
                <DataWorkspace
                  brain={brain}
                  foregroundTurnActive={chatActivity.turnActive}
                  onBack={() => onView("chat")}
                  onBrainChange={onBrainChange}
                  onToast={onToast}
                />
              ) : view === "map" ? (
                <BrainMapWorkspace brain={brain} onBack={() => onView("chat")} />
              ) : view === "trace" ? (
                <TraceWorkspace brain={brain} />
              ) : view === "imagine" ? (
                <ImaginationWorkspace brain={brain} onToast={onToast} />
              ) : view === "tools" ? (
                <ToolsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
              ) : view === "settings" ? (
                <DeviceRuntimeWorkspace
                  brain={brain}
                  onActiveModeChange={onActiveModeChange}
                  onBrainChange={onBrainChange}
                  onToast={onToast}
                />
              ) : view === "agents" ? (
                <AgentsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
              ) : (
                <EvolutionWorkspace
                  brain={brain}
                  onOpenPermissions={() => onView("tools")}
                  onToast={onToast}
                />
              )}
            </div>
            {viewWaitsForTurn ? (
              <div className="workspace-view-host__waiting" role="status" aria-live="polite">
                <Icon name="chat" size={16} />
                <span>{chatActivityStatus?.waitingLabel}</span>
                <Button kind="primary" icon="chat" onClick={() => onView("chat")}>
                  Queue or Steer
                </Button>
              </div>
            ) : null}
          </div>
        ) : null}
      </section>
    </main>
  );
}

interface ChatToolCommand {
  label: string;
  invocation: Omit<ToolInvocation, "brainId" | "approvalToken">;
  source: "human" | "brain";
  /** Links a one-action approval back to its original visible action card. */
  actionEventId: string;
}

interface PendingChatTool extends ChatToolCommand {
  approvalToken: string;
  approvalExpiresAt?: string;
}

function pendingChatToolFromAction(event: ActionEvent): PendingChatTool | null {
  if (event.state !== "approval-required" || !event.execution?.approvalToken ||
      !event.action.toolId || !event.action.action) return null;
  return {
    label: `${chatActionTitle(event.action)} proposed in chat`,
    source: event.action.source === "human" ? "human" : "brain",
    actionEventId: event.id,
    invocation: {toolId: event.action.toolId, action: event.action.action,
      arguments: event.action.arguments},
    approvalToken: event.execution.approvalToken,
    approvalExpiresAt: event.execution.approvalExpiresAt
  };
}

interface QueuedChatTurn {
  id: string;
  text: string;
  createdAt: string;
}

interface AttachmentLearningReceipt {
  id: string;
  label: string;
  sourceCount: number;
  sourceKinds: string[];
  learnedIdeas: number;
  learnedConcepts: number;
  learnedSynapses: number;
  warnings: string[];
  createdAt: string;
}

function valueRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function ChatActionCard({ event, onCancel, onApprove }: {
  event: ActionEvent;
  onCancel?: () => Promise<void>;
  onApprove?: () => Promise<void>;
}) {
  const [mediaPlaybackFailed, setMediaPlaybackFailed] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [approving, setApproving] = useState(false);
  const iconByKind: Record<ActionEvent["action"]["kind"], IconName> = {
    talk: "chat",
    tool: "terminal",
    imagine: "sparkles",
    agent: "agents",
    ponder: "brain",
    learn: "database",
    evolve: "pulse",
    stop: "close"
  };
  const preview = event.preview;
  const imaginationMedia = actionImaginationMedia(event);
  const { sourceUrl, mimeType } = imaginationMedia;
  useEffect(() => setMediaPlaybackFailed(false), [sourceUrl]);
  const title = chatActionTitle(event.action);
  const stateLabel = event.cancellationRequested ? "Cancelling" : chatActionStateLabel(event.state);
  const searchActivity = chatActionSearchActivity(event.action, event.state);
  const argumentsText = Object.keys(event.action.arguments).length
    ? JSON.stringify(event.action.arguments)
    : "";
  const artifactPath = imaginationMedia.finalArtifactPath ||
    imaginationMedia.artifactPath;

  return (
    <details className={cx("chat-action-card", `chat-action-card--${event.state}`)}>
      <summary aria-label={`${title}: ${stateLabel}. Expand action details`}>
        <span className="chat-action-card__icon"><Icon name={iconByKind[event.action.kind]} size={15} /></span>
        <span className="chat-action-card__summary-copy">
          <strong>{title}</strong>
          <small>{searchActivity ? `${stateLabel} · ${searchActivity}` : stateLabel}</small>
        </span>
        <Icon name="chevron" size={13} />
      </summary>
      <div className="chat-action-card__body">
        <div className="chat-action-card__head">
          <span>
            <small>{event.action.source === "organic" ? "AROSE NATURALLY" : event.action.source === "human" ? "YOU ASKED" : "BRAIN ACTION"}</small>
            <strong>{title}</strong>
          </span>
          <em><i /> {stateLabel}</em>
        </div>
        {searchActivity ? (
          <span className="chat-action-card__activity">
            <Icon name="search" size={13} /> {searchActivity}
          </span>
        ) : null}
        {argumentsText ? <code>{argumentsText.length > 360 ? `${argumentsText.slice(0, 357)}…` : argumentsText}</code> : null}
        {event.error ? <p>{cleanChatActionStatus(event.error, event.action)}</p> : null}
        {event.state === "running" && event.progress !== undefined ? (
          <span
            className="chat-action-card__progress"
            aria-label={`${Math.round(event.progress * 100)} percent complete`}
          >
            <i style={{ width: `${Math.round(event.progress * 100)}%` }} />
            <small>{cleanChatActionStatus(event.statusLabel, event.action)}</small>
          </span>
        ) : null}
        {onCancel && event.state === "running" ? (
          <Button kind="ghost" icon={cancelling || event.cancellationRequested ? "pulse" : "close"}
            disabled={cancelling || event.cancellationRequested} onClick={() => {
              setCancelling(true);
              void onCancel().finally(() => setCancelling(false));
            }}>
            {cancelling || event.cancellationRequested ? "Cancelling this action…" : "Cancel this action"}
          </Button>
        ) : null}
        {onApprove && pendingChatToolFromAction(event) ? (
          <Button kind="primary" icon={approving ? "pulse" : "check"}
            disabled={approving || event.cancellationRequested}
            onClick={() => {
              setApproving(true);
              void onApprove().finally(() => setApproving(false));
            }}>
            {approving ? "Approving this action…" : "Approve exact action"}
          </Button>
        ) : null}
        {preview ? (
          <small
            className="chat-action-card__preview-label"
            role="status"
            aria-live="polite"
          >
            {imaginationRevisionCopy(imaginationMedia)}
          </small>
        ) : null}
        {sourceUrl && mimeType.startsWith("image/") ? (
          <img src={sourceUrl} alt="Artifact created by the local imagination action" />
        ) : sourceUrl && mimeType.startsWith("audio/") && !mediaPlaybackFailed ? (
          <audio src={sourceUrl} controls onError={() => setMediaPlaybackFailed(true)} />
        ) : sourceUrl && mimeType.startsWith("video/") && !mediaPlaybackFailed ? (
          <video src={sourceUrl} controls onError={() => setMediaPlaybackFailed(true)} />
        ) : mediaPlaybackFailed && imaginationMedia.downloadUrl ? (
          <span className="chat-action-card__artifact">
            <Icon name="warning" size={14} /> Inline playback is unavailable for this larger artifact. It remains saved in Imagination for download.
          </span>
        ) : artifactPath ? (
          <span className="chat-action-card__artifact"><Icon name="file" size={14} /> {artifactPath}</span>
        ) : null}
        {sourceUrl && imaginationMedia.finalArtifactPath ? (
          <span className="chat-action-card__artifact">
            <Icon name="file" size={14} /> Final artifact · {imaginationMedia.finalArtifactPath}
          </span>
        ) : null}
        {studioUiDestination(event) ? (
          <span className="chat-action-card__artifact">
            <Icon name={studioUiDestination(event) === "imagine" ? "sparkles" : "settings"} size={14} /> {studioUiDestination(event) === "imagine" ? "Creativity workspace" : "Permissions workspace"} opened locally · reversible
          </span>
        ) : null}
        <span className="chat-action-card__meta">
          {event.action.confidence !== undefined ? `${Math.round(event.action.confidence * 100)}% action confidence · ` : ""}
          {new Date(event.updatedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        </span>
      </div>
    </details>
  );
}

function AttachmentLearningCard({
  receipt
}: {
  receipt: AttachmentLearningReceipt;
}) {
  return (
    <article className="chat-action-card chat-action-card--complete chat-attachment-card">
      <span className="chat-action-card__icon">
        <Icon name="upload" size={18} />
      </span>
      <div className="chat-action-card__body">
        <div className="chat-action-card__head">
          <span>
            <small>CHAT ATTACHMENT · NEURAL LEARNING</small>
            <strong>{receipt.label}</strong>
          </span>
          <em><i /> complete</em>
        </div>
        <dl className="chat-attachment-card__metrics">
          <span><dt>Sources</dt><dd>{receipt.sourceCount}</dd></span>
          <span><dt>Ideas</dt><dd>+{receipt.learnedIdeas}</dd></span>
          <span><dt>Concepts</dt><dd>+{receipt.learnedConcepts}</dd></span>
          <span><dt>Synapses</dt><dd>+{receipt.learnedSynapses}</dd></span>
        </dl>
        <span className="chat-action-card__meta">
          {receipt.sourceKinds.join(" · ")} · encoded into neural state ·{" "}
          {new Date(receipt.createdAt).toLocaleTimeString([], {
            hour: "numeric",
            minute: "2-digit"
          })}
        </span>
        {receipt.warnings.length ? (
          <p>{receipt.warnings.join(" · ")}</p>
        ) : null}
      </div>
    </article>
  );
}

function DatasetActivityCard({
  preview,
  job,
  onPause,
  onCancel
}: {
  preview: DatasetPreviewProgress | null;
  job: RuntimeJob | null;
  onPause?: () => void;
  onCancel: () => void;
}) {
  const activePreview = preview && !["complete", "cancelled", "failed"].includes(preview.phase)
    ? preview
    : null;
  const jobStop = runtimeJobStopPresentation(job);
  const activeJob = jobStop.active ? job : null;
  const jobProgress = runtimeJobProgressPresentation(activeJob ?? job);
  const coverage = jobProgress.showCoverage
    ? valueRecord(valueRecord(job?.output)?.coverage)
    : undefined;
  const completedFiles =
    typeof coverage?.processedFiles === "number" && typeof coverage?.rejectedFiles === "number"
      ? coverage.processedFiles + coverage.rejectedFiles
      : null;
  const discoveredFiles =
    typeof coverage?.discoveredFiles === "number"
      ? coverage.discoveredFiles
      : preview?.discoveredFiles ?? 0;
  const completedRecords =
    typeof coverage?.processedRecords === "number" && typeof coverage?.rejectedRecords === "number"
      ? coverage.processedRecords + coverage.rejectedRecords
      : null;
  const discoveredRecords =
    typeof coverage?.discoveredRecords === "number" ? coverage.discoveredRecords : null;
  const fileHashProgress = activePreview?.currentFileBytes
    ? Math.max(
        0,
        Math.min(1, (activePreview.currentFileHashedBytes ?? 0) / activePreview.currentFileBytes)
      )
    : null;
  const progress = activeJob
    ? jobProgress.fraction
    : fileHashProgress;
  const state = activePreview?.phase ?? job?.state ?? preview?.phase ?? "queued";
  const label = activePreview?.message ?? job?.label ?? preview?.message ?? "Preparing dataset";
  const cancellable = Boolean(activePreview || activeJob);
  const error = job?.error ?? (preview?.phase === "failed" ? preview.message : undefined);
  const presentedError = error ? conciseUiMessage(error) : "";

  return (
    <article
      className={cx(
        "chat-action-card",
        "chat-dataset-card",
        `chat-action-card--${state === "complete" ? "complete" : state === "failed" ? "failed" : "running"}`
      )}
      aria-label="Long learning job progress"
    >
      <span className="chat-action-card__icon"><Icon name="database" size={18} /></span>
      <div className="chat-action-card__body">
        <div className="chat-action-card__head">
          <span>
            <small>{activePreview ? "PREPARING FILES" : "LEARNING FROM DATA"}</small>
            <strong>{label}</strong>
          </span>
          <em><i /> {state}</em>
        </div>
        <span
          className={cx("chat-action-card__progress", progress === null && "is-indeterminate")}
          aria-label={progress === null ? "Dataset progress is streaming" : `${Math.round(progress * 100)} percent complete`}
        >
          <i style={progress === null ? undefined : { width: `${Math.round(progress * 100)}%` }} />
          <small>
            {activePreview
              ? `${activePreview.hashedFiles} hashed · ${formatBytes(activePreview.hashedBytes)} read`
              : progress === null
                ? jobProgress.statusText
                : `${Math.round(progress * 100)}% complete`}
          </small>
        </span>
        {activePreview || jobProgress.showCoverage ? <dl className="chat-attachment-card__metrics">
          <span>
            <dt>Records visited</dt>
            <dd>{completedRecords === null || discoveredRecords === null ? "discovering" : `${completedRecords} / ${discoveredRecords}`}</dd>
          </span>
          <span title="A Parquet shard is one file even when it contains many records.">
            <dt>Dataset files</dt>
            <dd>{completedFiles === null ? discoveredFiles : `${completedFiles} / ${discoveredFiles}`}</dd>
          </span>
          <span>
            <dt>Read</dt>
            <dd>{formatBytes(activePreview?.hashedBytes ?? (typeof coverage?.processedBytes === "number" ? coverage.processedBytes : 0))}</dd>
          </span>
          <span>
            <dt>Snapshot</dt>
            <dd>{activePreview ? "hashing" : job?.state ?? preview?.phase ?? "ready"}</dd>
          </span>
        </dl> : (
          <span className="chat-action-card__artifact">
            <Icon name="pulse" size={14} /> Preparing neural state before record traversal
          </span>
        )}
        {activePreview?.currentFile ? (
          <span className="chat-action-card__artifact">
            <Icon name="file" size={14} /> {activePreview.currentFile}
          </span>
        ) : null}
        {presentedError ? <p className="ui-error-copy">{presentedError}</p> : null}
        {cancellable ? (
          <div className="chat-dataset-card__controls">
            {onPause && activeJob && activeJob.state !== "cancelling" ? (
              <Button kind="ghost" icon="pause" onClick={onPause}>Pause</Button>
            ) : null}
            <Button
              kind="ghost"
              icon={jobStop.waitingForAcknowledgement ? "pulse" : "close"}
              disabled={Boolean(activeJob && !jobStop.requestAllowed)}
              onClick={onCancel}
            >
              {activeJob ? jobStop.label : "Cancel safely"}
            </Button>
          </div>
        ) : null}
      </div>
    </article>
  );
}

function ChatWorkspace({
  brain,
  hidden,
  onBrainChange,
  onToast,
  onNavigate,
  onActivityChange
}: {
  brain: BrainDocument;
  hidden?: boolean;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
  onNavigate: (view: WorkspaceView) => void;
  onActivityChange: (activity: ChatWorkspaceActivitySnapshot) => void;
}) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [attachmentDragActive, setAttachmentDragActive] = useState(false);
  const [attachmentReceipts, setAttachmentReceipts] = useState<
    AttachmentLearningReceipt[]
  >([]);
  const [datasetPreview, setDatasetPreview] = useState<DatasetPreviewProgress | null>(null);
  const [learningJobs, setLearningJobs] = useState<RuntimeJob[]>([]);
  const [pendingTool, setPendingTool] = useState<PendingChatTool | null>(null);
  const approvingActionIdsRef = useRef(new Set<string>());
  const [approvalClock, setApprovalClock] = useState(() => Date.now());
  const [toolStatus, setToolStatus] = useState("");
  const [toolRunning, setToolRunning] = useState(false);
  const [actionEvents, setActionEvents] = useState<ActionEvent[]>([]);
  const [persistedConversation, setPersistedConversation] = useState<ConversationLedgerEntry[]>([]);
  const [conversationTotal, setConversationTotal] = useState(0);
  const [conversationHasOlder, setConversationHasOlder] = useState(false);
  const [conversationAtLatest, setConversationAtLatest] = useState(true);
  const [conversationPageRevision, setConversationPageRevision] = useState(0);
  const [partialText, setPartialText] = useState("");
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [uncommittedOutputs, setUncommittedOutputs] = useState<UncommittedChatOutput[]>([]);
  const [postReplyWork, setPostReplyWork] = useState<Array<{
    turnId: string; label: string; ariaLabel: string;
  }>>([]);
  const [activeTurnStartedAtMs, setActiveTurnStartedAtMs] = useState<number | null>(null);
  const [chatOutputClockMs, setChatOutputClockMs] = useState<number | null>(null);
  const [chatQueue, setChatQueue] = useState<ChatQueueState | null>(null);
  const [chatDeliveryStatus, setChatDeliveryStatus] = useState("");
  const [cancellingTurn, setCancellingTurn] = useState(false);
  const [optimisticHumans, setOptimisticHumans] = useState<ChatMessage[]>([]);
  const [sessionCommittedMessages, setSessionCommittedMessages] =
    useState<ChatMessage[]>([]);
  const [queuedTurns, setQueuedTurns] = useState<QueuedChatTurn[]>([]);
  const [followingOutput, setFollowingOutput] = useState(true);
  const [timelineWindow, setTimelineWindow] = useState<ChatTimelineWindow>(() =>
    latestChatTimelineWindow(brain.messages.length)
  );
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState<WorkspaceSnapshot | null>(null);
  const [workspaceDelta, setWorkspaceDelta] =
    useState<WorkspaceTelemetryDelta | undefined>();
  const [inspectorTab, setInspectorTab] = useState<"state" | "runtime">("state");
  const [liveVoiceState, setLiveVoiceState] = useState<LiveVoiceState | null>(null);
  const [savedReplyVoice, setSavedReplyVoice] = useState<{ messageId: string; phase: "preparing" | "playing" } | null>(null);
  const [liveVoicePreferences, setLiveVoicePreferences] =
    useState<LiveVoicePreferences>(() => {
      try {
        return loadLiveVoicePreferences(window.localStorage);
      } catch {
        return {
          schemaVersion: 1,
          deliveryMode: "live",
          pace: "normal",
          neuralListening: false,
          neuralVoice: false
        };
      }
    });

  useEffect(() => {
    if (!pendingTool?.approvalExpiresAt) return;
    const expiresAt = Date.parse(pendingTool.approvalExpiresAt);
    if (!Number.isFinite(expiresAt)) return;
    const update = () => {
      const now = Date.now();
      setApprovalClock(now);
      if (now >= expiresAt) {
        setPendingTool((current) => current?.approvalToken === pendingTool.approvalToken ? null : current);
        setToolStatus("The Ask approval window expired; the brain can continue and request the action again.");
      }
    };
    update();
    const timer = window.setInterval(update, 250);
    return () => window.clearInterval(timer);
  }, [pendingTool?.approvalExpiresAt, pendingTool?.approvalToken]);

  useEffect(() => {
    if (!activeTurnId || partialText) {
      setChatOutputClockMs(null);
      return;
    }
    const tick = (): void => setChatOutputClockMs(Date.now());
    tick();
    const timer = window.setInterval(tick, 1_000);
    return () => window.clearInterval(timer);
  }, [activeTurnId, partialText]);
  const messagesEnd = useRef<HTMLDivElement>(null);
  const messageStreamRef = useRef<HTMLDivElement>(null);
  const pendingTimelineAnchorRef = useRef<{
    id?: string;
    offset?: number;
    scrollTop?: number;
    latest?: boolean;
  } | null>(null);
  const timelineBrainIdRef = useRef(brain.id);
  const followOutputRef = useRef(true);
  const userTimelineScrollRef = useRef(false);
  const activeTurnIdRef = useRef<string | null>(null);
  const partialTextRef = useRef("");
  const generationPhasesRef = useRef(new Map<string, ChatGenerationPhase>());
  const committedTurnIdsRef = useRef(new Set<string>());
  const receivedInputTurnIdsRef = useRef(new Set<string>());
  const cancelledTurnIdsRef = useRef(new Set<string>());
  const steeredTurnIdsRef = useRef(new Set<string>());
  const actionTurnIdsRef = useRef(new Map<string, string>());
  const currentBrainIdRef = useRef(brain.id);
  currentBrainIdRef.current = brain.id;
  const chatQueueRef = useRef<ChatQueueState | null>(null);
  const streamSequenceRef = useRef(new Map<string, number>());
  const tokenBatcherRef = useRef<TextFrameBatcher | null>(null);
  const textTurnGenerationRef = useRef(0);
  const activeHumanTurnsRef = useRef(new Map<
    string,
    { content: string; createdAt: string }
  >());
  const activePreviewIdRef = useRef<string | null>(null);
  const attachmentOperationGateRef = useRef(
    new ChatAttachmentOperationGate()
  );
  const liveVoiceControllerRef = useRef<LiveVoiceController | null>(null);
  const savedReplySynthesisRef = useRef<LiveVoiceSynthesisAdapter | null>(null);
  const savedReplySessionRef = useRef<LiveVoiceSynthesisSession | null>(null);
  const savedReplyVoiceGenerationRef = useRef(0);
  const liveVoiceUtteranceRef = useRef<string | null>(null);
  const workspaceSnapshotRef = useRef<WorkspaceSnapshot | null>(null);
  const workspaceRequestRef = useRef(0);
  const onBrainChangeRef = useRef(onBrainChange);
  const onNavigateRef = useRef(onNavigate);
  const liveVoiceGenerationActive = Boolean(liveVoiceState?.activeUtterance &&
    generationPhasesRef.current.get(liveVoiceState.activeUtterance.turnId) === "responding");

  const refreshWorkspaceSnapshot = useCallback(async (): Promise<void> => {
    const workspace = window.omni?.brain.workspace;
    if (!workspace) {
      workspaceSnapshotRef.current = null;
      setWorkspaceSnapshot(null);
      setWorkspaceDelta(undefined);
      return;
    }
    const request = ++workspaceRequestRef.current;
    try {
      const snapshot = await workspace(brain.id);
      if (request !== workspaceRequestRef.current || snapshot.brainId !== brain.id) {
        return;
      }
      const previous = workspaceSnapshotRef.current;
      workspaceSnapshotRef.current = snapshot;
      setWorkspaceDelta(workspaceTelemetryDelta(previous, snapshot));
      setWorkspaceSnapshot(snapshot);
    } catch {
      // Keep the last committed measurement during a transient atomic file
      // replacement or worker restart. A later poll repairs it without making
      // the counter regress to configuration guesses.
    }
  }, [brain.id]);

  const persistDeliveryReceipt = (
    turnId: string,
    state: ChatDeliveryReceiptState,
    fallback?: { content: string; createdAt: string }
  ): Promise<void> => {
    const turn = activeHumanTurnsRef.current.get(turnId) ?? fallback;
    const recorder = window.omni?.chat.recordDeliveryReceipt;
    if (!turn || typeof recorder !== "function") return Promise.resolve();
    return recorder(brain.id, {
      schemaVersion: 1,
      turnId,
      content: turn.content,
      createdAt: turn.createdAt,
      state
    }).catch((error: unknown) => {
      onToast(
        conciseUiMessage(
          error,
          "The local delivery receipt could not be saved."
        )
      );
    });
  };

  const loadLatestConversation = async (): Promise<ChatMessage[]> => {
    if (!window.omni) return [];
    const page = await window.omni.chat.listPage(brain.id, undefined, 120);
    setPersistedConversation(page.entries);
    setConversationTotal(page.totalEntries);
    setConversationHasOlder(page.hasOlder);
    setConversationAtLatest(true);
    setConversationPageRevision((current) => current + 1);
    return page.entries.flatMap((entry) => entry.message ? [entry.message] : []);
  };

  const loadOlderConversation = async (): Promise<void> => {
    if (!window.omni || !conversationHasOlder) return;
    const before = persistedConversation[0]?.sequence;
    if (before === undefined) return;
    const viewport = messageStreamRef.current;
    const anchor = viewport?.querySelector<HTMLElement>("[data-timeline-id]");
    if (viewport && anchor) {
      pendingTimelineAnchorRef.current = {
        id: anchor.dataset.timelineId,
        offset: anchor.getBoundingClientRect().top -
          viewport.getBoundingClientRect().top,
        scrollTop: viewport.scrollTop
      };
    }
    const page = await window.omni.chat.listPage(brain.id, before, 40);
    const merged = new Map<number, ConversationLedgerEntry>();
    [...page.entries, ...persistedConversation].forEach((entry) =>
      merged.set(entry.sequence, entry)
    );
    const windowed = [...merged.values()]
      .sort((left, right) => left.sequence - right.sequence)
      .slice(0, 120);
    followOutputRef.current = false;
    setFollowingOutput(false);
    setPersistedConversation(windowed);
    setConversationTotal(page.totalEntries);
    setConversationHasOlder(page.hasOlder);
    setConversationAtLatest(false);
    setConversationPageRevision((current) => current + 1);
  };

  useEffect(() => {
    onBrainChangeRef.current = onBrainChange;
  }, [onBrainChange]);

  useEffect(() => {
    onNavigateRef.current = onNavigate;
  }, [onNavigate]);

  useEffect(() => {
    let active = true;
    if (!window.omni) {
      setPersistedConversation([]);
      setConversationTotal(brain.messages.length);
      setConversationHasOlder(false);
      setConversationAtLatest(true);
      return;
    }
    void window.omni.chat.listPage(brain.id, undefined, 120).then((page) => {
      if (!active) return;
      setPersistedConversation(page.entries);
      setConversationTotal(page.totalEntries);
      setConversationHasOlder(page.hasOlder);
      setConversationAtLatest(true);
      setConversationPageRevision((current) => current + 1);
    }).catch(() => undefined);
    return () => {
      active = false;
    };
  }, [brain.id]);

  const routeStudioAction = (event: ActionEvent): void => {
    const destination = studioUiDestination(event);
    // A brain-originated creativity action stays in the chronological chat as
    // an expandable artifact instead of unexpectedly replacing the page.
    if (destination && event.action.source === "human") {
      onNavigateRef.current(destination);
    }
  };

  useEffect(() => {
    if (!window.omni) {
      setLiveVoiceState(null);
      return;
    }
    const adapters = createBrowserLiveVoiceAdapters(undefined, brain.id);
    savedReplySynthesisRef.current = adapters.neural?.synthesis ?? null;
    const controller = new LiveVoiceController({
      brainId: brain.id,
      ...adapters,
      preferences: liveVoicePreferences,
      chat: createOmniLiveVoiceChatAdapter(window.omni.chat),
      onAcceptedReply: (reply) => {
        generationPhasesRef.current.delete(reply.turnId);
        committedTurnIdsRef.current.delete(reply.turnId);
        streamSequenceRef.current.delete(reply.turnId);
        setOptimisticHumans((current) =>
          current.filter((message) => message.id !== `pending-voice-${reply.utteranceId}`)
        );
        const result = reply.chatResult;
        if (!result) return;
        void loadLatestConversation();
        onBrainChangeRef.current(result.brain);
        if (result.actionEvents?.length) {
          setActionEvents((current) =>
            result.actionEvents!.reduce(
              (events, event) => mergeChatActionEvent(events, event),
              current
            )
          );
          result.actionEvents.forEach(routeStudioAction);
        }
      }
    });
    liveVoiceControllerRef.current = controller;
    const unsubscribe = controller.subscribe((state) => {
      setLiveVoiceState({ ...state });
      const utterance = state.activeUtterance;
      if (utterance && utterance.id !== liveVoiceUtteranceRef.current) {
        liveVoiceUtteranceRef.current = utterance.id;
        generationPhasesRef.current.set(utterance.turnId, "responding");
        const utteranceStartedAtMs = Date.parse(utterance.createdAt);
        const measuredStart = Number.isFinite(utteranceStartedAtMs)
          ? utteranceStartedAtMs
          : Date.now();
        setActiveTurnStartedAtMs(measuredStart);
        setChatOutputClockMs(measuredStart);
        tokenBatcherRef.current?.reset();
        partialTextRef.current = "";
        setPartialText("");
        setOptimisticHumans((current) => [
          ...current.filter((message) => !message.id.startsWith("pending-voice-")),
          {
            id: `pending-voice-${utterance.id}`,
            role: "human",
            content: utterance.transcript,
            createdAt: utterance.createdAt
          }
        ]);
      }
      if (utterance && generationPhasesRef.current.get(utterance.turnId) === "responding") {
        activeTurnIdRef.current = utterance.turnId;
        setActiveTurnId(utterance.turnId);
      } else if (activeTurnIdRef.current?.startsWith("voice-")) {
        streamSequenceRef.current.delete(activeTurnIdRef.current);
        activeTurnIdRef.current = null;
        setActiveTurnId(null);
        setActiveTurnStartedAtMs(null);
        setChatOutputClockMs(null);
        partialTextRef.current = "";
        setPartialText("");
      }
      if (
        !state.enabled ||
        state.phase === "error" ||
        state.phase === "unavailable"
      ) {
        liveVoiceUtteranceRef.current = null;
        setOptimisticHumans((current) =>
          current.filter((message) => !message.id.startsWith("pending-voice-"))
        );
      }
    });
    return () => {
      unsubscribe();
      savedReplyVoiceGenerationRef.current += 1;
      savedReplySessionRef.current?.cancel();
      savedReplySessionRef.current = null;
      savedReplySynthesisRef.current = null;
      setSavedReplyVoice(null);
      if (liveVoiceControllerRef.current === controller) {
        liveVoiceControllerRef.current = null;
      }
      liveVoiceUtteranceRef.current = null;
      void controller.dispose();
    };
  }, [brain.id]);

  const playSavedReplyWithOwnVoice = (message: ChatMessage): void => {
    if (savedReplyVoice?.messageId === message.id) {
      savedReplyVoiceGenerationRef.current += 1;
      savedReplySessionRef.current?.cancel();
      savedReplySessionRef.current = null;
      setSavedReplyVoice(null);
      return;
    }
    const synthesis = savedReplySynthesisRef.current;
    if (!synthesis?.available || liveVoiceControllerRef.current?.getState().enabled) {
      onToast("Own waveform playback is unavailable while live voice is active, or local playback is unavailable.");
      return;
    }
    const generation = ++savedReplyVoiceGenerationRef.current;
    savedReplySessionRef.current?.cancel();
    savedReplySessionRef.current = null;
    setSavedReplyVoice({ messageId: message.id, phase: "preparing" });
    const done = (): void => {
      if (savedReplyVoiceGenerationRef.current !== generation) return;
      savedReplyVoiceGenerationRef.current += 1;
      savedReplySessionRef.current = null;
      setSavedReplyVoice(null);
    };
    try {
      const session = synthesis.speak(message.content, {
        onStart: () => {
          if (savedReplyVoiceGenerationRef.current === generation) {
            setSavedReplyVoice({ messageId: message.id, phase: "playing" });
          }
        },
        onEnd: done,
        onError: (error) => {
          if (savedReplyVoiceGenerationRef.current !== generation) return;
          done();
          onToast(error || "Own waveform playback failed.");
        }
      }, { rate: LIVE_VOICE_PACE_RATES[liveVoicePreferences.pace] });
      if (savedReplyVoiceGenerationRef.current === generation) savedReplySessionRef.current = session;
      else session.cancel();
    } catch (error) {
      done();
      onToast(error instanceof Error ? error.message : "Own waveform playback failed.");
    }
  };

  useEffect(() => {
    const stalePreviewId = activePreviewIdRef.current;
    if (stalePreviewId) {
      void window.omni?.data.cancelPreview(stalePreviewId).catch(() => undefined);
    }
    attachmentOperationGateRef.current.reset();
    setAttaching(false);
    followOutputRef.current = true;
    setFollowingOutput(true);
    setOptimisticHumans([]);
    setSessionCommittedMessages([]);
    setQueuedTurns([]);
    setUncommittedOutputs([]);
    setPostReplyWork([]);
    generationPhasesRef.current.clear();
    committedTurnIdsRef.current.clear();
    receivedInputTurnIdsRef.current.clear();
    cancelledTurnIdsRef.current.clear();
    steeredTurnIdsRef.current.clear();
    actionTurnIdsRef.current.clear();
    activeTurnIdRef.current = null;
    setActiveTurnId(null);
    setSending(false);
    partialTextRef.current = "";
    setPartialText("");
    setActiveTurnStartedAtMs(null);
    setChatOutputClockMs(null);
    setChatQueue(null);
    chatQueueRef.current = null;
    setChatDeliveryStatus("");
    setCancellingTurn(false);
    setDatasetPreview(null);
    setLearningJobs([]);
    setTimelineWindow(latestChatTimelineWindow(brain.messages.length));
    pendingTimelineAnchorRef.current = null;
    activePreviewIdRef.current = null;
    activeHumanTurnsRef.current.clear();
    workspaceRequestRef.current += 1;
    workspaceSnapshotRef.current = null;
    setWorkspaceSnapshot(null);
    setWorkspaceDelta(undefined);
  }, [brain.id]);

  useLayoutEffect(() => {
    // A hidden chat has zero viewport geometry. Do not let off-chat layout
    // changes disturb the reader's follow state; returning to Conversation
    // performs the pending scroll once the viewport is measurable again.
    if (hidden || !followOutputRef.current) return;
    messagesEnd.current?.scrollIntoView({ block: "end" });
  }, [
    brain.messages,
    hidden,
    optimisticHumans,
    sending,
    partialText,
    actionEvents.length,
    attachmentReceipts,
    datasetPreview,
    learningJobs
  ]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void window.omni.train.list(brain.id).then((jobs) => {
      if (!active) return;
      setLearningJobs(
        jobs.filter(
          (job) => isChatVisibleLearningJob(job) &&
            runtimeJobIsActive(job)
        ).sort((left, right) => left.createdAt.localeCompare(right.createdAt))
      );
    });
    const unsubscribeJobs = window.omni.train.onEvent(({ job }) => {
      if (job.brainId !== brain.id || !isChatVisibleLearningJob(job)) return;
      setLearningJobs((current) => mergeChatVisibleLearningJob(current, job));
      if (job.state === "cancelling") {
        setToolStatus(
          job.error
            ? `${conciseUiMessage(job.error)} Retry stop to request acknowledgement again.`
            : `${job.label}. The control stays locked until the exact worker acknowledges termination.`
        );
      } else if (job.state === "complete") {
        setToolStatus("Whole-dataset learning completed; the committed coverage remains visible.");
        void window.omni?.brain.get(brain.id).then(onBrainChangeRef.current).catch(() => undefined);
      } else if (job.state === "failed" || job.state === "cancelled") {
        setToolStatus(
          job.state === "cancelled"
            ? "Whole-dataset learning paused safely; its committed cursor can resume."
            : `Whole-dataset learning failed${job.error ? `: ${job.error}` : "."}`
        );
      }
    });
    const unsubscribePreview = window.omni.data.onPreviewProgress((event) => {
      if (
        event.brainId !== brain.id ||
        event.requestId !== activePreviewIdRef.current
      ) return;
      setDatasetPreview(event);
    });
    return () => {
      active = false;
      unsubscribeJobs();
      unsubscribePreview();
    };
  }, [brain.id]);

  useEffect(() => {
    if (!window.omni) return;
    return window.omni.chat.onAction((event) => {
      if (event.brainId !== brain.id) return;
      setActionEvents((current) => mergeChatActionEvent(current, event));
      routeStudioAction(event);
      setToolRunning(event.state === "running");
      setToolStatus(
        event.state === "failed"
          ? `${chatActionTitle(event.action)} failed: ${cleanChatActionStatus(event.error, event.action)}`
          : `${chatActionTitle(event.action)}: ${chatActionStateLabel(event.state)}.`
      );
      const pending = pendingChatToolFromAction(event);
      if (pending) {
        setPendingTool(pending);
      } else if (["complete", "failed", "stopped"].includes(event.state)) {
        setPendingTool((current) =>
          current?.actionEventId === event.id ? null : current
        );
      }
      if (event.action.kind === "talk" && event.state === "complete") {
        void window.omni?.brain.get(brain.id).then(onBrainChange).catch(() => {
          // The action card remains the visible audit record if a refresh races
          // with shutdown; the persisted message is loaded on the next view.
        });
      }
    });
  }, [brain.id, onBrainChange]);

  useEffect(() => {
    if (!window.omni) return;
    const batcher = createTextFrameBatcher(
      (delta) => {
        partialTextRef.current += delta;
        setPartialText(partialTextRef.current);
      },
      (callback) => window.requestAnimationFrame(callback),
      (handle) => window.cancelAnimationFrame(handle)
    );
    tokenBatcherRef.current = batcher;
    const finishGeneration = (turnId: string, createdAt: string, provisional: boolean): void => {
      if (activeTurnIdRef.current !== turnId) return;
      batcher.flush();
      if (provisional) {
        const content = partialTextRef.current;
        setUncommittedOutputs((current) => retainUncommittedChatOutput(
          current, turnId, content, createdAt
        ));
      }
      batcher.reset();
      partialTextRef.current = "";
      activeTurnIdRef.current = null;
      chatQueueRef.current = null;
      setActiveTurnId(null);
      setActiveTurnStartedAtMs(null);
      setChatOutputClockMs(null);
      setPartialText("");
      setChatQueue(null);
      setCancellingTurn(false);
      setSending(false);
      setChatDeliveryStatus("");
    };
    const removeListener = window.omni.chat.onStream((event: ChatStreamEvent) => {
      if (event.brainId !== brain.id) return;
      const previousPhase = generationPhasesRef.current.get(event.turnId);
      if (!previousPhase) return;
      const lastSequence = streamSequenceRef.current.get(event.turnId) ?? -1;
      if (event.sequence <= lastSequence) return;
      streamSequenceRef.current.set(event.turnId, event.sequence);
      generationPhasesRef.current.set(event.turnId,
        advanceChatGenerationPhase(previousPhase, event));
      // Action/media work has its own lane, including updates from an older
      // completed reply while a newer text turn owns generation controls.
      if (event.type === "chat-input-accepted") {
        receivedInputTurnIdsRef.current.add(event.turnId);
        setSessionCommittedMessages(current => mergeCommittedChatMessages(current, [event.humanMessage]).slice(-240));
        setOptimisticHumans(current => reconcileCompletedChatTurn(current, event.turnId, event.humanMessage));
        // Input receipt never closes/reopens generation or fabricates a reply.
        return;
      }
      if (event.type === "chat-action") {
        actionTurnIdsRef.current.set(event.actionEvent.id, event.turnId);
        setActionEvents((current) => mergeChatActionEvent(current, event.actionEvent));
        routeStudioAction(event.actionEvent);
        setToolRunning(event.actionEvent.state === "running");
        const pending = pendingChatToolFromAction(event.actionEvent);
        if (pending) {
          setPendingTool(pending);
        } else if (["complete", "failed", "stopped"].includes(event.actionEvent.state)) {
          setPendingTool((current) => current?.actionEventId === event.actionEvent.id ? null : current);
        }
        setToolStatus(
          event.actionEvent.state === "failed"
            ? `${chatActionTitle(event.actionEvent.action)} failed: ${cleanChatActionStatus(event.actionEvent.error, event.actionEvent.action)}`
            : `${chatActionTitle(event.actionEvent.action)}: ${chatActionStateLabel(event.actionEvent.state)}.`
        );
        return;
      }
      if (event.type === "modality-preview") {
        setActionEvents((current) => patchChatActionPreview(current, event.actionId, event.preview));
        return;
      }
      if (event.type === "chat-phase") {
        if (previousPhase === "settled") return;
        const presentation = replyCompleteLearningPresentation(event);
        if (!presentation) return;
        // Freeze the final buffered output before allowing the next ordinary
        // Send. It remains explicitly uncommitted until the exact saved rows.
        finishGeneration(event.turnId, event.createdAt, !event.turnCommitted);
        setPostReplyWork((current) => [
          ...current.filter((work) => work.turnId !== event.turnId),
          { turnId: event.turnId, label: presentation.label, ariaLabel: presentation.ariaLabel }
        ]);
        void refreshWorkspaceSnapshot();
        return;
      }
      if (event.type === "chat-reply-committed") {
        committedTurnIdsRef.current.add(event.turnId);
        setSessionCommittedMessages((current) => mergeCommittedChatMessages(
          current, [event.humanMessage, event.brainMessage]
        ).slice(-240));
        setUncommittedOutputs((current) => reconcileUncommittedChatOutputs(current, event.turnId));
        setOptimisticHumans((current) => reconcileCompletedChatTurn(current, event.turnId, event.humanMessage));
        finishGeneration(event.turnId, event.createdAt, false);
        setPostReplyWork((current) => [
          ...current.filter((work) => work.turnId !== event.turnId),
          ...(event.pendingActions > 0 && previousPhase !== "settled" ? [{
            turnId: event.turnId,
            label: "Reply saved · action work continues",
            ariaLabel: "The exact chat turn is saved. Optional action and artifact work continues independently."
          }] : [])
        ]);
        return;
      }
      if (event.type === "chat-state" &&
          ["complete", "steered", "stopped", "no-reply", "failed", "cancelled"].includes(event.state)) {
        const priorQueue = event.turnId === activeTurnIdRef.current ? chatQueueRef.current : null;
        const committed = committedTurnIdsRef.current.has(event.turnId);
        const received = receivedInputTurnIdsRef.current.has(event.turnId);
        finishGeneration(event.turnId, event.createdAt, !committed && event.state !== "complete");
        setPostReplyWork((current) => current.filter((work) => work.turnId !== event.turnId));
        if (event.state === "no-reply" && committed && !activeTurnIdRef.current) {
          setChatDeliveryStatus("Input saved · generation produced no printable reply.");
        }
        if (event.state === "steered") {
          steeredTurnIdsRef.current.add(event.turnId);
          void persistDeliveryReceipt(event.turnId, "steered");
          setOptimisticHumans((current) => preserveSteeredChatMessage(current, event.turnId));
        }
        if (event.state === "stopped") {
          steeredTurnIdsRef.current.add(event.turnId);
          if (!committed) {
            void persistDeliveryReceipt(event.turnId, "stopped");
            setOptimisticHumans((current) => settleOptimisticChatTurn(current, event.turnId, "stopped"));
            if (!activeTurnIdRef.current) setChatDeliveryStatus("The brain chose to stop before replying; your message remains visible.");
          }
        }
        if (event.state === "failed" || event.state === "cancelled") {
          const terminalState = event.state;
          if (!committed && !received) {
            void persistDeliveryReceipt(event.turnId, terminalState);
            setOptimisticHumans((current) => settleOptimisticChatTurn(current, event.turnId, terminalState));
            setUncommittedOutputs((current) => settleUncommittedChatOutput(current, event.turnId, terminalState));
          }
          const terminalStatus = committed
            ? `Reply remains saved; its later action work ${terminalState === "cancelled" ? "was cancelled" : "failed"}.`
            : received ? receivedChatInputFailureStatus(terminalState)
            : event.state === "cancelled"
              ? chatCancellationStatus(event.cancellation, priorQueue)
              : `Message not sent: ${conciseUiMessage(event.error, "The local brain could not respond.")}`;
          // An older work receipt never replaces a newer response's delivery status.
          if (!activeTurnIdRef.current) setChatDeliveryStatus(terminalStatus);
          setToolStatus(terminalStatus);
        }
        void refreshWorkspaceSnapshot();
        return;
      }
      if (event.turnId !== activeTurnIdRef.current || previousPhase !== "responding") return;
      if (event.type === "chat-state" && event.state === "queued") {
        if (!event.queue) return;
        chatQueueRef.current = event.queue;
        setChatQueue(event.queue);
        setCancellingTurn(false);
        setChatDeliveryStatus(chatQueueStatus(event.queue));
      } else if (event.type === "chat-state" && event.state === "started") {
        chatQueueRef.current = null;
        setChatQueue(null);
        setCancellingTurn(false);
        setChatDeliveryStatus("");
        const startedAt = Date.now();
        setActiveTurnStartedAtMs(startedAt);
        setChatOutputClockMs(startedAt);
      } else if (event.type === "chat-token") {
        chatQueueRef.current = null;
        setChatQueue(null);
        batcher.push(event.delta);
      }
    });
    return () => {
      removeListener();
      batcher.reset();
      if (tokenBatcherRef.current === batcher) tokenBatcherRef.current = null;
      streamSequenceRef.current.clear();
    };
  }, [brain.id, refreshWorkspaceSnapshot]);

  useEffect(() => {
    void refreshWorkspaceSnapshot();
    const timer = window.setInterval(
      () => void refreshWorkspaceSnapshot(),
      WORKSPACE_TELEMETRY_REFRESH_MS
    );
    const invalidate = (event: Event): void => {
      const detail = (event as CustomEvent<{ brainId?: string }>).detail;
      if (detail?.brainId && detail.brainId !== brain.id) return;
      void refreshWorkspaceSnapshot();
    };
    window.addEventListener("omni:workspace-invalidated", invalidate);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("omni:workspace-invalidated", invalidate);
    };
  }, [brain.id, brain.updatedAt, refreshWorkspaceSnapshot]);

  const send = async (
    turnMetadata?: ChatTurnMetadata,
    queuedTurn?: QueuedChatTurn
  ) => {
    const text = (queuedTurn?.text ?? input).trim();
    const capacity = chatInputCapacity(text, brain.config.contextWindowTokens);
    if (text && !capacity.fits) {
      if (queuedTurn) setInput((current) => recoverFailedChatDraft(current, text));
      onToast(capacity.reason!);
      return;
    }
    const steering = turnMetadata?.kind === "steer";
    if (!text || (!steering && sending) ||
        (!steering && liveVoiceGenerationActive && liveVoiceState?.phase === "pondering")) return;
    if (!window.omni) {
      if (queuedTurn) {
        setInput((current) => recoverFailedChatDraft(current, text));
      }
      onToast("Neural engine unavailable in this design preview. Your draft was not sent.");
      return;
    }
    const submissionGeneration = ++textTurnGenerationRef.current;
    const turnId = queuedTurn?.id ?? globalThis.crypto?.randomUUID?.() ??
      `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const createdAt = queuedTurn?.createdAt ?? new Date().toISOString();
    generationPhasesRef.current.set(turnId, "responding");
    const optimisticHuman: ChatMessage = {
      id: queuedTurn ? `queued-${turnId}` : `pending-${turnId}`,
      role: "human",
      content: text,
      createdAt
    };
    activeHumanTurnsRef.current.set(turnId, { content: text, createdAt });
    setSending(true);
    setCancellingTurn(false);
    chatQueueRef.current = null;
    setChatQueue(null);
    setChatDeliveryStatus("");
    if (!queuedTurn) {
      setInput((current) => clearSubmittedChatDraft(current, text));
    }
    if (window.omni && !queuedTurn) {
      setOptimisticHumans((current) => [
        ...(turnMetadata
          ? preserveSteeredChatMessage(current, turnMetadata.replacesTurnId)
          : current),
        optimisticHuman
      ]);
    }
    tokenBatcherRef.current?.reset();
    streamSequenceRef.current.delete(turnId);
    activeTurnIdRef.current = turnId;
    setActiveTurnId(turnId);
    const turnStartedAtMs = Date.now();
    setActiveTurnStartedAtMs(turnStartedAtMs);
    setChatOutputClockMs(turnStartedAtMs);
    partialTextRef.current = "";
    setPartialText("");
    try {
      if (window.omni) {
        // Save the human bubble before neural work begins. A remounted chat can
        // render this presentation-only row even while the reply is pending.
        await persistDeliveryReceipt(turnId, "pending");
        if (liveVoiceControllerRef.current?.getState().enabled) {
          await liveVoiceControllerRef.current.stop();
        }
        // Steer is a warm-worker handoff. The main controller queues this
        // correlated turn at the current turn's atomic commit boundary;
        // cancelling here would intentionally terminate the serial Python
        // worker and force the loaded neural substrate to cold-start again.
        if (!isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) return;
        const result = await window.omni.chat.send(
          brain.id,
          text,
          turnId,
          turnMetadata
        );
        if (currentBrainIdRef.current !== brain.id) return;
        // The chat result is authoritative. The separate ledger page can lag
        // behind an atomic background save, so keep these exact messages in
        // this session until the page catches up.
        setSessionCommittedMessages((current) =>
          mergeCommittedChatMessages(
            current,
            [result.humanMessage, result.brainMessage]
          ).slice(-240)
        );
        committedTurnIdsRef.current.add(turnId);
        setUncommittedOutputs((current) => reconcileUncommittedChatOutputs(current, turnId));
        setOptimisticHumans((current) =>
          reconcileCompletedChatTurn(
            current,
            turnId,
            result.humanMessage
          )
        );
        if (isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) {
          onBrainChange(result.brain);
        } else {
          // Older artifact work may finish after a newer reply was saved. Its
          // exact messages are retained above, but its older whole document
          // must not replace the newer counters/trace in the workspace.
          void window.omni.brain.get(brain.id).then(onBrainChangeRef.current).catch(() => undefined);
        }
        // Refresh history for older turns and actions, but a transient page
        // read failure must never turn a committed reply into a failed send.
        await loadLatestConversation().catch(() => undefined);
        if (result.actionEvents?.length) {
          setActionEvents((current) => {
            const merged = new Map(current.map((event) => [event.id, event]));
            result.actionEvents?.forEach((event) => merged.set(event.id, event));
            return [...merged.values()];
          });
          result.actionEvents.forEach(routeStudioAction);
          const lastAction = result.actionEvents.at(-1);
          if (lastAction) {
            setToolStatus(
              lastAction.state === "failed"
                ? `${chatActionTitle(lastAction.action)} failed: ${cleanChatActionStatus(lastAction.error, lastAction.action)}`
                : `${chatActionTitle(lastAction.action)}: ${chatActionStateLabel(lastAction.state)}.`
            );
            if (
              lastAction.state === "approval-required" &&
              lastAction.execution?.approvalToken &&
              lastAction.action.toolId &&
              lastAction.action.action
            ) {
              setPendingTool({
                label: `${chatActionTitle(lastAction.action)} proposed in natural chat`,
                source: lastAction.action.source === "human" ? "human" : "brain",
                actionEventId: lastAction.id,
                invocation: {
                  toolId: lastAction.action.toolId,
                  action: lastAction.action.action,
                  arguments: lastAction.action.arguments
                },
                approvalToken: lastAction.execution.approvalToken,
                approvalExpiresAt: lastAction.execution.approvalExpiresAt
              });
            }
          }
        }
      }
    } catch (error) {
      if (currentBrainIdRef.current !== brain.id) return;
      if (steeredTurnIdsRef.current.has(turnId)) return;
      if (committedTurnIdsRef.current.has(turnId)) {
        setToolStatus(`Reply remains saved; later action work did not finish: ${conciseUiMessage(error)}`);
        return;
      }
      const terminalState = cancelledTurnIdsRef.current.has(turnId) ? "cancelled" : "failed";
      if (!receivedInputTurnIdsRef.current.has(turnId)) void persistDeliveryReceipt(turnId, terminalState, {
        content: text,
        createdAt
      });
      if (window.omni) {
        setOptimisticHumans((current) =>
          settleOptimisticChatTurn(current, turnId, terminalState)
        );
        setUncommittedOutputs((current) => settleUncommittedChatOutput(current, turnId, terminalState));
        // The main process commits the human and neural messages atomically.
        // Reload after failure so any independently completed neural activity
        // is reflected. A matching authoritative human turn reconciles the
        // local failure receipt; otherwise that receipt remains visibly failed.
        await window.omni.brain.get(brain.id).then(async (refreshed) => {
          const latestMessages = await loadLatestConversation();
          if (latestMessages.some(message => message.role === "human" && message.turnId === turnId && message.inputReceipt?.committed)) {
            receivedInputTurnIdsRef.current.add(turnId);
            if (isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) setChatDeliveryStatus(receivedChatInputFailureStatus(terminalState));
          }
          setOptimisticHumans((current) =>
            reconcileOptimisticChatMessages(current, latestMessages)
          );
          if (isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) {
            onBrainChange(refreshed);
          }
        }).catch(() => {
          // The existing authoritative document and visible failure receipt
          // remain valid if refresh itself fails.
        });
        // Preserve the newest unsent wording in the composer as well, so the
        // person can retry or edit it without reconstructing the message.
        if (terminalState === "failed" && !receivedInputTurnIdsRef.current.has(turnId) &&
            isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) {
          setInput((current) => recoverFailedChatDraft(current, text));
        }
      }
      if (terminalState === "failed" &&
          isCurrentChatSubmission(submissionGeneration, textTurnGenerationRef.current)) {
        onToast(error instanceof Error ? error.message : "The local brain could not respond.");
      }
    } finally {
      if (currentBrainIdRef.current !== brain.id) return;
      setPostReplyWork((current) => current.filter((work) => work.turnId !== turnId));
      // Late frames for retired turns are ignored by the stream's known-turn
      // guard; retain metadata only while its real work/result is outstanding.
      generationPhasesRef.current.delete(turnId);
      committedTurnIdsRef.current.delete(turnId);
      receivedInputTurnIdsRef.current.delete(turnId);
      cancelledTurnIdsRef.current.delete(turnId);
      steeredTurnIdsRef.current.delete(turnId);
      streamSequenceRef.current.delete(turnId);
      activeHumanTurnsRef.current.delete(turnId);
      // Only the matching generation may clear the live composer. Every older
      // reply still reconciles above, including late committed/cancelled work.
      if (activeTurnIdRef.current !== turnId) return;
      tokenBatcherRef.current?.reset();
      if (activeTurnIdRef.current) {
        streamSequenceRef.current.delete(activeTurnIdRef.current);
      }
      activeTurnIdRef.current = null;
      setActiveTurnId(null);
      setActiveTurnStartedAtMs(null);
      setChatOutputClockMs(null);
      partialTextRef.current = "";
      setPartialText("");
      setSending(false);
    }
  };

  const approvePendingTool = async (selected: PendingChatTool | null = pendingTool) => {
    if (!selected || cancellingTurn || !window.omni || approvingActionIdsRef.current.has(selected.actionEventId)) return;
    const approval = selected;
    if (approval.approvalExpiresAt && Date.now() >= Date.parse(approval.approvalExpiresAt)) {
      onToast("This action's approval expired. Ask the brain to propose it again.");
      return;
    }
    approvingActionIdsRef.current.add(approval.actionEventId);
    // The token is single-use. Disarm the visible control before awaiting so a
    // double click cannot dispatch the same approved action twice.
    setPendingTool((current) => current?.actionEventId === approval.actionEventId ? null : current);
    setToolRunning(true);
    setToolStatus(`${approval.label} is running through the approved action protocol.`);
    // Ask-approved executions use the action id (not its original chat turn)
    // as ToolExecutor correlation. Keep card cancellation exact for that lane.
    actionTurnIdsRef.current.set(approval.actionEventId, approval.actionEventId);
    try {
      const result = await window.omni.chat.approveAction({
        brainId: brain.id,
        actionEventId: approval.actionEventId,
        approvalToken: approval.approvalToken
      });
      setActionEvents((current) => mergeChatActionEvent(current, result.actionEvent));
      await loadLatestConversation();
      const refreshed = await window.omni.brain.get(brain.id);
      onBrainChange(refreshed);
      if (
        result.actionEvent.state === "approval-required" &&
        result.actionEvent.execution?.approvalToken
      ) {
        setPendingTool({
          ...approval,
          approvalToken: result.actionEvent.execution.approvalToken,
          approvalExpiresAt: result.actionEvent.execution.approvalExpiresAt
        });
        setToolStatus(
          `${approval.label} changed before execution and requires a fresh exact approval.`
        );
      } else if (result.actionEvent.state === "failed") {
        setToolStatus(
          `${approval.label} failed: ${cleanChatActionStatus(result.actionEvent.error)}`
        );
      } else if (result.learningError) {
        setToolStatus(
          `${approval.label} completed, but result learning failed: ${cleanChatActionStatus(result.learningError)}`
        );
      } else {
        setToolStatus(`${approval.label} completed and its structured result was learned.`);
      }
    } catch (error) {
      setPendingTool(approval);
      onToast(error instanceof Error ? error.message : "The approved tool could not run.");
    } finally {
      approvingActionIdsRef.current.delete(approval.actionEventId);
      setToolRunning(false);
    }
  };

  const steerCurrentTurn = async (): Promise<void> => {
    if (cancellingTurn) return;
    const capacity = chatInputCapacity(input, brain.config.contextWindowTokens);
    if (!capacity.fits) { onToast(capacity.reason!); return; }
    const replacesTurnId = activeTurnIdRef.current;
    if (!replacesTurnId) {
      await send();
      return;
    }
    await persistDeliveryReceipt(replacesTurnId, "steered");
    tokenBatcherRef.current?.flush();
    const emittedPrefix = partialTextRef.current;
    if (emittedPrefix) {
      setUncommittedOutputs((current) => retainUncommittedChatOutput(current, replacesTurnId,
        emittedPrefix, new Date().toISOString()).map((output) => output.turnId === replacesTurnId
          ? { ...output, state: "steering" } : output));
    }
    await send({
      kind: "steer",
      replacesTurnId,
      source: "human",
      createdAt: new Date().toISOString()
    });
  };

  const queueCurrentTurn = (): void => {
    const text = input.trim();
    if (!text) return;
    const capacity = chatInputCapacity(text, brain.config.contextWindowTokens);
    if (!capacity.fits) { onToast(capacity.reason!); return; }
    const id = globalThis.crypto?.randomUUID?.() ??
      `queued-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const createdAt = new Date().toISOString();
    const turn: QueuedChatTurn = { id, text, createdAt };
    setInput((current) => clearSubmittedChatDraft(current, text));
    void persistDeliveryReceipt(id, "queued", {
      content: text,
      createdAt
    });
    setQueuedTurns((current) => [...current, turn]);
    setOptimisticHumans((current) => [
      ...current,
      { id: `queued-${id}`, role: "human", content: text, createdAt }
    ]);
  };

  const submitOrdinaryTurn = (): void => {
    const active = sending || Boolean(activeTurnIdRef.current) ||
      liveVoiceGenerationActive;
    if (active) {
      queueCurrentTurn();
      return;
    }
    void send();
  };

  const discardQueuedTurns = (): void => {
    queuedTurns.forEach((turn) => {
      void persistDeliveryReceipt(turn.id, "cancelled", {
        content: turn.text,
        createdAt: turn.createdAt
      });
    });
    const ids = new Set(queuedTurns.map((turn) => `queued-${turn.id}`));
    setOptimisticHumans((current) =>
      current.filter((message) => !ids.has(message.id))
    );
    setQueuedTurns([]);
  };

  useEffect(() => {
    if (sending || liveVoiceGenerationActive || !queuedTurns.length) return;
    const next = queuedTurns[0]!;
    setQueuedTurns((current) =>
      current[0]?.id === next.id
        ? current.slice(1)
        : current.filter((turn) => turn.id !== next.id)
    );
    void send(undefined, next);
  }, [sending, liveVoiceGenerationActive, queuedTurns]);

  const cancelChatTool = async () => {
    if (!window.omni) return;
    const activeTurn = activeTurnIdRef.current ?? activeTurnId;
    if (!activeTurn) return;
    cancelledTurnIdsRef.current.add(activeTurn);
    let count = 0;
    if (activeTurn) {
      const priorQueue = chatQueueRef.current;
      setCancellingTurn(true);
      setChatDeliveryStatus(
        priorQueue
          ? `Cancelling only this queued message · ${priorQueue.queuedBehind.label} continues…`
          : "Stopping response · waiting for neural worker acknowledgement…"
      );
      void persistDeliveryReceipt(activeTurn, "cancelled");
      ++textTurnGenerationRef.current;
      const liveVoiceController = liveVoiceControllerRef.current;
      const liveVoiceActive = Boolean(liveVoiceController?.getState().enabled);
      try {
        if (liveVoiceActive) {
          await liveVoiceController!.stop();
          count = 1;
        } else {
          count = await window.omni.chat.cancel(brain.id, activeTurn);
        }
      } catch (error) {
        setCancellingTurn(false);
        const status = conciseUiMessage(error, "The current turn could not be stopped.");
        setChatDeliveryStatus(status);
        onToast(status);
        return;
      }
      if (!liveVoiceActive) {
        if (count < 1) {
          setCancellingTurn(false);
          setOptimisticHumans((current) =>
            settleOptimisticChatTurn(current, activeTurn, "cancelled")
          );
          tokenBatcherRef.current?.reset();
          streamSequenceRef.current.delete(activeTurn);
          activeTurnIdRef.current = null;
          setActiveTurnId(null);
          setActiveTurnStartedAtMs(null);
          setChatOutputClockMs(null);
          partialTextRef.current = "";
          setPartialText("");
          setSending(false);
          chatQueueRef.current = null;
          setChatQueue(null);
          activeHumanTurnsRef.current.delete(activeTurn);
          setChatDeliveryStatus(
            "Message cancelled before it entered the neural runtime queue."
          );
        }
        // The ordered terminal stream is the acknowledgement boundary. Keep
        // Stopping visible and preserve the active turn until that frame says a
        // queued request was removed or a running worker actually exited.
        setToolStatus(
          count > 0
            ? "Cancellation requested for this exact chat turn."
            : "Message cancelled before neural execution."
        );
        return;
      }
      setOptimisticHumans((current) =>
        settleOptimisticChatTurn(current, activeTurn, "cancelled")
      );
      tokenBatcherRef.current?.reset();
      streamSequenceRef.current.delete(activeTurn);
      activeTurnIdRef.current = null;
      setActiveTurnId(null);
      setActiveTurnStartedAtMs(null);
      setChatOutputClockMs(null);
      partialTextRef.current = "";
      setPartialText("");
      setSending(false);
      setCancellingTurn(false);
      chatQueueRef.current = null;
      setChatQueue(null);
      activeHumanTurnsRef.current.delete(activeTurn);
      setChatDeliveryStatus("Live voice stopped; the interrupted transcript remains visible.");
    }
    setToolStatus(
      count > 0
        ? "Cancellation requested. Partial text and artifacts remain visibly marked in this turn."
        : "No cancellable chat or tool execution is active."
    );
  };

  const cancelActionJob = async (event: ActionEvent): Promise<void> => {
    if (!window.omni) return;
    const target = chatActionCancellationTarget(event, actionTurnIdsRef.current.get(event.id));
    if (!target) return;
    try {
      if (target.kind === "modality-job") {
        await window.omni.modality.cancel(target.id);
      } else if (target.kind === "inline-action") {
        const result = await window.omni.chat.cancelInlineAction(brain.id, target.turnId, target.id);
        setActionEvents((current) => mergeChatActionEvent(current, result.actionEvent));
      } else {
        // ToolExecutor correlates this exact action execution with its turn;
        // this never aborts the newer text turn or any other turn's artifacts.
        await window.omni.tool.cancel(brain.id, target.id);
      }
    } catch (error) {
      onToast(conciseUiMessage(error, "This action could not be cancelled."));
    }
  };

  const toggleLiveVoice = async (): Promise<void> => {
    const controller = liveVoiceControllerRef.current;
    if (!controller) {
      onToast("Live voice requires the packaged desktop runtime.");
      return;
    }
    if (controller.getState().enabled) {
      await controller.stop();
    } else {
      // Saved-reply waveform output is an independent job, not a voice turn.
      // Stop that selected output before opening a microphone/recognizer.
      savedReplyVoiceGenerationRef.current += 1;
      savedReplySessionRef.current?.cancel();
      savedReplySessionRef.current = null;
      setSavedReplyVoice(null);
      await controller.start();
    }
  };

  const bargeInLiveVoice = (): void => {
    liveVoiceControllerRef.current?.interrupt();
  };

  const updateLiveVoicePreferences = (
    patch: Partial<Omit<LiveVoicePreferences, "schemaVersion">>
  ): void => {
    const next: LiveVoicePreferences = {
      ...liveVoicePreferences,
      ...patch,
      schemaVersion: 1
    };
    setLiveVoicePreferences(next);
    liveVoiceControllerRef.current?.setPreferences(next);
    try {
      saveLiveVoicePreferences(window.localStorage, next);
    } catch {
      // Ephemeral or hardened renderers can still use the choice this session.
    }
  };

  const recordAttachmentResults = (
    results: IngestResult[],
    label: string
  ): void => {
    if (!results.length) {
      setToolStatus("No attachment was selected; neural state was unchanged.");
      return;
    }
    const sourceKinds = [
      ...new Set(results.map((result) => result.source.kind.toLocaleUpperCase()))
    ];
    const receipt: AttachmentLearningReceipt = {
      id:
        globalThis.crypto?.randomUUID?.() ??
        `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      label,
      sourceCount: results.length,
      sourceKinds,
      learnedIdeas: results.reduce(
        (sum, result) => sum + result.source.learnedIdeas,
        0
      ),
      learnedConcepts: results.reduce(
        (sum, result) => sum + result.source.learnedConcepts,
        0
      ),
      learnedSynapses: results.reduce(
        (sum, result) => sum + result.source.learnedSynapses,
        0
      ),
      warnings: results.flatMap((result) => result.warnings),
      createdAt: new Date().toISOString()
    };
    setAttachmentReceipts((current) => [...current, receipt]);
    setToolStatus(
      `${receipt.sourceCount} ${label.toLocaleLowerCase()} encoded into neural state: ` +
      `+${receipt.learnedIdeas} ideas, +${receipt.learnedConcepts} concepts, ` +
      `+${receipt.learnedSynapses} synaptic changes.`
    );
  };

  const attachExperience = async (
    selection: ExperienceUploadKind | "folder" = "files"
  ) => {
    if (!window.omni) {
      onNavigate("data");
      return;
    }
    if (sending) return;
    const operationLease = attachmentOperationGateRef.current.begin();
    if (!operationLease) return;
    setAttaching(true);
    const label =
      selection === "folder"
        ? "Folder experience"
        : EXPERIENCE_UPLOADS[selection].shortLabel.replace(
            /^./,
            (character) => character.toLocaleUpperCase()
          );
    setToolStatus(
      `Choose ${selection === "folder" ? "a folder" : EXPERIENCE_UPLOADS[selection].shortLabel}; committed neural changes will appear in the training receipt.`
    );
    const requestId = globalThis.crypto?.randomUUID?.() ??
      `dataset-preview-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const startedAt = new Date().toISOString();
    activePreviewIdRef.current = requestId;
    setDatasetPreview({
      schemaVersion: 1,
      requestId,
      brainId: brain.id,
      phase: "discovering",
      discoveredFiles: 0,
      discoveredBytes: 0,
      hashedFiles: 0,
      hashedBytes: 0,
      message:
        `Choose ${selection === "folder" ? "a folder" : EXPERIENCE_UPLOADS[selection].shortLabel}; ` +
        "the selected snapshot will appear here before neural learning starts.",
      startedAt,
      updatedAt: startedAt
    });
    try {
      const started = await queueChatAttachmentLearning(window.omni.data, {
        brainId: brain.id,
        policy: "pretrain",
        requestId,
        selection
      }, () => attachmentOperationGateRef.current.isCurrent(operationLease));
      if (!attachmentOperationGateRef.current.isCurrent(operationLease)) return;
      activePreviewIdRef.current = null;
      if (!started) {
        const cancelledAt = new Date().toISOString();
        setDatasetPreview((current) => current?.requestId === requestId
          ? {
              ...current,
              phase: "cancelled",
              message: "Attachment selection cancelled; neural state was unchanged.",
              updatedAt: cancelledAt
            }
          : current);
        setToolStatus("No attachment was selected; neural state was unchanged.");
        return;
      }
      setDatasetPreview(null);
      setLearningJobs((current) =>
        mergeChatVisibleLearningJob(current, started.job)
      );
      setToolStatus(
        `${started.manifest.discoveredFiles} ${label.toLocaleLowerCase()} source${started.manifest.discoveredFiles === 1 ? "" : "s"} committed once and queued for adaptive neural retention.`
      );
    } catch (error) {
      if (!attachmentOperationGateRef.current.isCurrent(operationLease)) return;
      activePreviewIdRef.current = null;
      const cancelled = error instanceof Error && error.name === "AbortError";
      const message = cancelled
        ? "Attachment learning cancelled safely; no uncommitted source was learned."
        : error instanceof Error
          ? error.message
          : "The attachment could not be learned.";
      setDatasetPreview((current) => current?.requestId === requestId
        ? {
            ...current,
            phase: cancelled ? "cancelled" : "failed",
            message,
            updatedAt: new Date().toISOString()
          }
        : current);
      if (!cancelled) onToast(message);
      setToolStatus(message);
    } finally {
      activePreviewIdRef.current = activePreviewIdRef.current === requestId
        ? null
        : activePreviewIdRef.current;
      if (attachmentOperationGateRef.current.finish(operationLease)) {
        setAttaching(false);
      }
    }
  };

  const stopDatasetActivity = async (
    intent: "pause" | "cancel",
    jobId?: string
  ): Promise<void> => {
    if (!window.omni) return;
    const previewId = activePreviewIdRef.current;
    if (!jobId && previewId) {
      const cancelled = await window.omni.data.cancelPreview(previewId);
      if (cancelled) {
        setToolStatus("Cancelling folder discovery and hashing; partial manifest data will be removed.");
      }
      return;
    }
    const job = learningJobs.find((candidate) => candidate.id === jobId);
    const stop = runtimeJobStopPresentation(job);
    if (!job || !stop.requestAllowed) return;
    setLearningJobs((current) => mergeChatVisibleLearningJob(current, {
      ...job,
      state: "cancelling",
      label: intent === "pause"
        ? `Pausing ${job.kind} · waiting for acknowledgement`
        : `Cancelling ${job.kind} · waiting for acknowledgement`,
      error: undefined,
      updatedAt: new Date().toISOString()
    }));
    try {
      const stopped = intent === "pause" && ["ingestion", "crawl"].includes(job.kind)
        ? await window.omni.data.pause(job.id)
        : ["ingestion", "crawl"].includes(job.kind)
          ? await window.omni.data.cancel(job.id)
          : await window.omni.train.cancel(job.id);
      setLearningJobs((current) => mergeChatVisibleLearningJob(current, stopped));
      const status = intent === "pause"
        ? `${job.label} paused at its durable cursor after worker acknowledgement.`
        : `${job.label} cancelled after worker acknowledgement; completed checkpoints remain intact.`;
      setToolStatus(status);
      onToast(status);
    } catch (error) {
      const message = conciseUiMessage(
        error instanceof Error ? error.message : String(error),
        `${job.label} is still awaiting cancellation acknowledgement.`
      );
      const current = await window.omni.train
        .list(brain.id)
        .then((jobs) => jobs.find((candidate) => candidate.id === job.id))
        .catch(() => undefined);
      setLearningJobs((jobs) => mergeChatVisibleLearningJob(jobs, current ?? {
        ...job,
        state: "cancelling",
        label: `${job.kind} stop needs acknowledgement`,
        error: message,
        updatedAt: new Date().toISOString()
      }));
      setToolStatus(`${message} Retry stop when ready.`);
      onToast(message);
    }
  };

  const attachDroppedExperience = async (files: File[]): Promise<void> => {
    setAttachmentDragActive(false);
    if (!files.length || sending) return;
    if (!window.omni) {
      onNavigate("data");
      return;
    }
    const operationLease = attachmentOperationGateRef.current.begin();
    if (!operationLease) return;
    setAttaching(true);
    setToolStatus(
      `Learning ${files.length} dropped item${files.length === 1 ? "" : "s"} into parameters and synapses…`
    );
    try {
      const results = await window.omni.data.ingestDropped(
        { brainId: brain.id, policy: "pretrain" },
        files
      );
      if (results.length > 0) {
        onBrainChange(await window.omni.brain.get(brain.id));
      }
      recordAttachmentResults(results, "Dropped files and folders");
    } catch (error) {
      onToast(
        error instanceof Error
          ? error.message
          : "The dropped material could not be learned."
      );
      setToolStatus("Dropped attachment learning failed.");
    } finally {
      if (attachmentOperationGateRef.current.finish(operationLease)) {
        setAttaching(false);
      }
    }
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const currentTurnActive = sending || Boolean(activeTurnIdRef.current) ||
      liveVoiceGenerationActive;
    const intent = composerSubmitIntent({
      key: event.key,
      shiftKey: event.shiftKey,
      ctrlKey: event.ctrlKey,
      metaKey: event.metaKey,
      turnActive: currentTurnActive,
      queueAvailable: currentTurnActive,
      steerAvailable: currentTurnActive && !cancellingTurn
    });
    if (intent === "none" || intent === "newline") return;
    event.preventDefault();
    if (intent === "steer") {
      void steerCurrentTurn();
    } else {
      submitOrdinaryTurn();
    }
  };

  const persistedMessages = useMemo(
    () => persistedConversation.flatMap((entry) =>
      entry.message ? [entry.message] : []
    ),
    [persistedConversation]
  );
  const persistedActions = useMemo(
    () => persistedConversation.flatMap((entry) =>
      entry.action ? [entry.action] : []
    ),
    [persistedConversation]
  );
  const displayedMessages = useMemo(() => {
    const authoritative = window.omni
      ? mergeCommittedChatMessages(persistedMessages, sessionCommittedMessages)
      : brain.messages;
    const combined = mergeChatMessagesForPresentation(
      authoritative,
      optimisticHumans
    );
    const visible = [...combined, ...uncommittedOutputs.map((output) => output.message)];
    if (visible.length) return visible;
    return window.omni
      ? []
      : [{
          id: "empty-greeting",
          role: "brain" as const,
          content:
            "I am here, but almost nothing has happened to me yet. What should we explore first?",
          createdAt: brain.createdAt,
          runtime: brain.config.runtime
        }];
  }, [brain.createdAt, brain.messages, brain.config.runtime, optimisticHumans, persistedMessages, sessionCommittedMessages, uncommittedOutputs]);
  const displayedActions = useMemo(() => {
    const merged = new Map(persistedActions.map((event) => [event.id, event]));
    actionEvents.forEach((event) => merged.set(event.id, event));
    return [...merged.values()];
  }, [actionEvents, persistedActions]);
  const timelineEntries = useMemo(() => [
      ...displayedMessages.map((message, order) => ({
        kind: "message" as const,
        id: `message-${message.id}`,
        createdAt: message.createdAt,
        order,
        message
      })),
      ...displayedActions.map((event, order) => ({
        kind: "action" as const,
        id: `action-${event.id}`,
        createdAt: event.createdAt,
        order: displayedMessages.length + order,
        event
      })),
      ...attachmentReceipts.map((receipt, order) => ({
        kind: "attachment" as const,
        id: `attachment-${receipt.id}`,
        createdAt: receipt.createdAt,
        order: displayedMessages.length + displayedActions.length + order,
        receipt
      })),
      ...(datasetPreview
        ? [{
            kind: "dataset" as const,
            id: `dataset-preview-${datasetPreview.requestId}`,
            createdAt: datasetPreview.startedAt,
            order: displayedMessages.length + displayedActions.length + attachmentReceipts.length,
            preview: datasetPreview,
            job: null
          }]
        : []),
      ...learningJobs.map((job, order) => ({
        kind: "dataset" as const,
        id: `dataset-job-${job.id}`,
        createdAt: job.createdAt,
        order:
          displayedMessages.length + displayedActions.length +
          attachmentReceipts.length + (datasetPreview ? 1 : 0) + order,
        preview: null,
        job
      }))
    ].sort((left, right) => {
      const time = Date.parse(left.createdAt) - Date.parse(right.createdAt);
      return Number.isFinite(time) && time !== 0 ? time : left.order - right.order;
    }), [
      attachmentReceipts,
      datasetPreview,
      displayedActions,
      displayedMessages,
      learningJobs
    ]);
  useLayoutEffect(() => {
    setTimelineWindow((current) => {
      const next = timelineBrainIdRef.current !== brain.id
        ? latestChatTimelineWindow(timelineEntries.length)
        : reconcileChatTimelineWindow(
            current,
            timelineEntries.length,
            followingOutput
          );
      timelineBrainIdRef.current = brain.id;
      if (
        followingOutput &&
        (next.start !== current.start || next.end !== current.end)
      ) {
        pendingTimelineAnchorRef.current = { latest: true };
      }
      return next.start === current.start &&
        next.end === current.end &&
        next.total === current.total
          ? current
          : next;
    });
  }, [brain.id, followingOutput, timelineEntries.length]);

  useLayoutEffect(() => {
    const pending = pendingTimelineAnchorRef.current;
    if (!pending) return;
    pendingTimelineAnchorRef.current = null;
    if (pending.latest) {
      messagesEnd.current?.scrollIntoView({ block: "end", behavior: "smooth" });
      return;
    }
    const viewport = messageStreamRef.current;
    if (
      !viewport ||
      !pending.id ||
      pending.offset === undefined ||
      pending.scrollTop === undefined
    ) return;
    const anchor = [...viewport.querySelectorAll<HTMLElement>("[data-timeline-id]")]
      .find((element) => element.dataset.timelineId === pending.id);
    if (!anchor) return;
    const offsetAfter = anchor.getBoundingClientRect().top -
      viewport.getBoundingClientRect().top;
    viewport.scrollTop = anchoredTimelineScrollTop(
      pending.scrollTop,
      pending.offset,
      offsetAfter
    );
  }, [conversationPageRevision, timelineWindow.end, timelineWindow.start]);

  const windowedTimelineEntries = timelineEntries.slice(
    timelineWindow.start,
    timelineWindow.end
  );
  const timelineAtLatest = conversationAtLatest &&
    timelineWindow.end >= timelineEntries.length;
  const visibleTimelineEntries = useMemo(() => {
    if (timelineAtLatest) return windowedTimelineEntries;
    const pinnedIds = new Set([
      ...optimisticHumans.slice(-20).map((message) => `message-${message.id}`),
      ...sessionCommittedMessages.slice(-20).map((message) => `message-${message.id}`),
      ...actionEvents.slice(-20).map((event) => `action-${event.id}`),
      ...attachmentReceipts.slice(-10).map((receipt) => `attachment-${receipt.id}`),
      ...(datasetPreview ? [`dataset-preview-${datasetPreview.requestId}`] : []),
      ...learningJobs.slice(-10).map((job) => `dataset-job-${job.id}`)
    ]);
    const existing = new Set(windowedTimelineEntries.map((entry) => entry.id));
    const pinned = timelineEntries
      .filter((entry) => pinnedIds.has(entry.id) && !existing.has(entry.id))
      .slice(-20);
    return [
      ...windowedTimelineEntries.slice(0, Math.max(0, 120 - pinned.length)),
      ...pinned
    ];
  }, [
    actionEvents,
    attachmentReceipts,
    datasetPreview,
    learningJobs,
    optimisticHumans,
    sessionCommittedMessages,
    timelineAtLatest,
    timelineEntries,
    windowedTimelineEntries
  ]);
  const moveTimelineWindow = (direction: "older" | "newer"): void => {
    const viewport = messageStreamRef.current;
    const rows = viewport
      ? [...viewport.querySelectorAll<HTMLElement>("[data-timeline-id]")]
      : [];
    const anchor = direction === "older" ? rows[0] : rows.at(-1);
    if (viewport && anchor) {
      pendingTimelineAnchorRef.current = {
        id: anchor.dataset.timelineId,
        offset: anchor.getBoundingClientRect().top -
          viewport.getBoundingClientRect().top,
        scrollTop: viewport.scrollTop
      };
    }
    followOutputRef.current = false;
    setFollowingOutput(false);
    setTimelineWindow((current) => direction === "older"
      ? olderChatTimelineWindow(current)
      : newerChatTimelineWindow(current));
  };
  const recentTrace = brain.traces.at(-1);
  const measuredActivity = meanObservedTraceActivation(
    recentTrace?.activatedConcepts
  );
  const voicePondering = liveVoiceGenerationActive && liveVoiceState?.phase === "pondering";
  const textTurnVisiblyActive = Boolean(activeTurnId);
  const pondering = voicePondering || (
    textTurnVisiblyActive &&
    actionEvents.some(
      (event) =>
        event.action.kind === "ponder" &&
        (event.state === "proposed" || event.state === "running")
    )
  );
  const actionRunning =
    toolRunning ||
    (textTurnVisiblyActive && actionEvents.some((event) => event.state === "running"));
  const cortexState = cancellingTurn
    ? "Stopping response"
    : chatQueue
      ? `Queued behind ${chatQueue.queuedBehind.label}`
      : pondering
      ? "Pondering"
      : actionRunning
          ? "Running action"
          : textTurnVisiblyActive
            ? "Forming response"
            : postReplyWork[0]?.label
              ? postReplyWork[0].label
            : workspaceSnapshot?.learning?.backgroundParameters.lastError &&
                (workspaceSnapshot.learning.backgroundParameters.pending ?? 0) > 0
              ? "Idle · learning needs attention"
              : "Idle / ready";
  const showTurnActivity =
    textTurnVisiblyActive || voicePondering;
  const voiceStatus = liveVoiceState ? liveVoiceStatusCopy(liveVoiceState) : null;
  const showVoiceStatus = Boolean(
    liveVoiceState &&
    (liveVoiceState.enabled ||
      liveVoiceState.phase === "error" ||
      liveVoiceState.phase === "unavailable" ||
      liveVoiceState.phase === "starting")
  );
  const voiceSettingsLocked = Boolean(liveVoiceState?.activeUtterance);
  const turnActive = sending || textTurnVisiblyActive || liveVoiceGenerationActive;
  const composerResponsePhase = !turnActive
    ? "idle"
    : partialText ? "generating" : "loading";
  const composerTurn = composerTurnCapabilities(composerResponsePhase);
  const queuedLearningJobCount = learningJobs.filter((job) => job.state === "queued").length +
    (datasetPreview && !["complete", "cancelled", "failed"].includes(datasetPreview.phase) ? 1 : 0);
  const runningLearningJobCount = learningJobs.filter(
    (job) => job.state === "running" || job.state === "cancelling"
  ).length;
  useLayoutEffect(() => {
    onActivityChange({
      turnActive,
      phase: turnActive
        ? chatQueue
          ? "queued"
          : "responding"
        : "idle",
      ...(chatQueue ? { queue: chatQueue } : {}),
      queuedMessages: queuedTurns.length,
      queuedJobs: queuedLearningJobCount,
      runningJobs: runningLearningJobCount,
      postReplyWork: postReplyWork.length
    });
  }, [
    onActivityChange,
    chatQueue,
    queuedLearningJobCount,
    queuedTurns.length,
    postReplyWork.length,
    runningLearningJobCount,
    turnActive
  ]);
  const draftTokenCount = utf8DraftTokenCount(input);
  const draftCapacity = chatInputCapacity(input, brain.config.contextWindowTokens);
  const responseTokenCount = utf8DraftTokenCount(partialText);
  const responseOutputPresentation = pendingChatOutputPresentation({
    outputTokens: responseTokenCount,
    startedAtMs: activeTurnStartedAtMs,
    nowMs: chatOutputClockMs,
    queue: chatQueue
  });
  const workspaceTelemetry = workspaceSnapshot
    ? workspaceTelemetryPresentation({
        workspace: workspaceSnapshot,
        delta: workspaceDelta,
        currentExperience: turnActive
            ? "responding"
            : postReplyWork.length
              ? "saving"
            : "idle"
      })
    : undefined;
  const contextTokenCopy = workspaceTelemetry?.contextText ??
    "temporary context measuring…";
  const composerTelemetryCopy = workspaceTelemetry
    ? `${workspaceTelemetry.experienceText} · ${workspaceTelemetry.backgroundText}`
    : "neural learning state measuring…";
  const composerTelemetryAriaLabel = workspaceTelemetry
    ? `${workspaceTelemetry.contextAriaLabel} ${workspaceTelemetry.experienceAriaLabel} ` +
      `${workspaceTelemetry.backgroundAriaLabel} ${workspaceTelemetry.measuredText}.`
    : "Authoritative temporary context and neural learning measurements are loading.";
  const chatParameterAccounting = neuralParameterAccounting(
    workspaceSnapshot?.runtimeCard
  );
  const followLatest = (): void => {
    followOutputRef.current = true;
    setFollowingOutput(true);
    setConversationAtLatest(true);
    void loadLatestConversation();
    pendingTimelineAnchorRef.current = { latest: true };
    setTimelineWindow(latestChatTimelineWindow(timelineEntries.length));
    messagesEnd.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  };

  return (
    <div className="chat-layout" hidden={hidden}>
      <section
        className={cx(
          "conversation",
          attachmentDragActive && "conversation--attachment-drag"
        )}
        onDragEnter={(event) => {
          if (!event.dataTransfer.types.includes("Files")) return;
          event.preventDefault();
          setAttachmentDragActive(true);
        }}
        onDragOver={(event) => {
          if (!event.dataTransfer.types.includes("Files")) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        onDragLeave={(event) => {
          if (
            !event.currentTarget.contains(event.relatedTarget as Node | null)
          ) {
            setAttachmentDragActive(false);
          }
        }}
        onDrop={(event) => {
          event.preventDefault();
          void attachDroppedExperience(Array.from(event.dataTransfer.files));
        }}
      >
        {attachmentDragActive ? (
          <div className="chat-attachment-drop" role="status">
            <span><Icon name="upload" size={24} /></span>
            <strong>Drop files, folders, images, audio, or video</strong>
            <small>They will be encoded into this brain’s parameters and synapses.</small>
          </div>
        ) : null}
        <div className="conversation__date">
          <span />
          <time>Continuous conversation · started with this identity</time>
          <span />
        </div>
        <div className="message-stream-wrap">
          <div
            className="message-stream"
            ref={messageStreamRef}
            role="log"
            tabIndex={0}
            aria-label={`${brain.name} conversation`}
            aria-live="off"
            aria-busy={showTurnActivity}
            onWheel={() => {
              userTimelineScrollRef.current = true;
            }}
            onPointerDown={() => {
              userTimelineScrollRef.current = true;
            }}
            onKeyDown={(event) => {
              if (["ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End", " "].includes(event.key)) {
                userTimelineScrollRef.current = true;
              }
            }}
            onScroll={(event) => {
              // Flex/layout changes, hidden-view transitions, and our own
              // scrollIntoView calls can all emit scroll events. Only a person
              // interacting with the timeline may opt out of following output.
              if (hidden || !userTimelineScrollRef.current) return;
              const element = event.currentTarget;
              const next = timelineAtLatest && shouldFollowChatOutput(
                element.scrollTop,
                element.clientHeight,
                element.scrollHeight
              );
              followOutputRef.current = next;
              setFollowingOutput((current) => current === next ? current : next);
            }}
          >
          {!displayedMessages.length ? (
            <div className="chat-empty">
              <span><BrandMark size={42} /></span>
              <strong>{brain.name} has no conversation yet</strong>
              <p>This is one continuous chat. The first real exchange will begin its conversational history.</p>
            </div>
          ) : null}
          {conversationHasOlder || timelineWindow.start > 0 || !timelineAtLatest ? (
            <nav className="chat-window-controls" aria-label="Conversation history window">
              <button
                disabled={!conversationHasOlder && timelineWindow.start <= 0}
                onClick={() => conversationHasOlder
                  ? void loadOlderConversation()
                  : moveTimelineWindow("older")}
              >
                <Icon name="arrow" size={12} /> Load older
              </button>
              <span aria-live="polite">
                Showing {visibleTimelineEntries.length.toLocaleString()} of {conversationTotal.toLocaleString()} persisted entries
                <small>All turns remain persisted · Conversation history search reaches entries outside this window</small>
              </span>
              <button
                disabled={timelineAtLatest}
                onClick={followLatest}
              >
                Load newer <Icon name="arrow" size={12} />
              </button>
            </nav>
          ) : null}
          {visibleTimelineEntries.map((entry, visibleIndex) => (
            <div
              key={entry.id}
              className="chat-virtual-row"
              data-timeline-id={entry.id}
              role="article"
              aria-posinset={timelineWindow.start + visibleIndex + 1}
              aria-setsize={timelineEntries.length}
            >
            {
            entry.kind === "message" ? (
              <MessageBubble
                message={entry.message}
                brainName={brain.name}
                onToast={onToast}
                onTrace={() => onNavigate("trace")}
                onOwnVoice={() => playSavedReplyWithOwnVoice(entry.message)}
                ownVoicePhase={savedReplyVoice?.messageId === entry.message.id ? savedReplyVoice.phase : undefined}
                ownVoiceUnavailable={!liveVoiceState?.capabilities.neuralVoice || liveVoiceState.enabled}
                ownVoiceQuality={liveVoiceState?.capabilities.neuralVoiceQuality}
                uncommittedOutputState={uncommittedOutputs.find((output) =>
                  output.message.id === entry.message.id)?.state}
              />
            ) : entry.kind === "action" ? (
              <div className="chat-action-stream" aria-label="Visible chat action">
                <ChatActionCard event={entry.event}
                  onCancel={chatActionCancellationTarget(entry.event,
                    actionTurnIdsRef.current.get(entry.event.id))
                    ? () => cancelActionJob(entry.event) : undefined}
                  onApprove={pendingChatToolFromAction(entry.event)
                    ? () => approvePendingTool(pendingChatToolFromAction(entry.event)) : undefined} />
              </div>
            ) : entry.kind === "attachment" ? (
              <div className="chat-action-stream" aria-label="Learned chat attachment">
                <AttachmentLearningCard receipt={entry.receipt} />
              </div>
            ) : (
              <div className="chat-action-stream" aria-label="Dataset learning activity">
                <DatasetActivityCard
                  preview={entry.preview}
                  job={entry.job}
                  onPause={
                    entry.job && ["ingestion", "crawl"].includes(entry.job.kind)
                      ? () => void stopDatasetActivity("pause", entry.job!.id)
                      : undefined
                  }
                  onCancel={() => void stopDatasetActivity("cancel", entry.job?.id)}
                />
              </div>
            )
            }
            </div>
          ))}
          {showTurnActivity && timelineAtLatest ? (
            <div className="message message--brain">
              <div className="message__avatar">
                <BrandMark size={28} />
              </div>
              <div className="message__body">
                <div className="message__meta">
                  <strong>{brain.name}</strong>
                  <span
                    className="pondering-label"
                  >
                    <i /> {
                      cancellingTurn
                        ? "Stopping"
                        : chatQueue
                          ? "Queued"
                          : pondering
                            ? "Pondering"
                            : actionRunning
                              ? "Running action"
                              : "Responding"
                    }
                  </span>
                  <span
                    className={cx(
                      "response-token-counter",
                      `response-token-counter--${responseOutputPresentation.phase}`
                    )}
                    aria-label={responseOutputPresentation.ariaLabel}
                  >
                    {responseOutputPresentation.label}
                  </span>
                </div>
                {partialText ? (
                  <p
                    className="message__streaming-text"
                    aria-live="polite"
                  >
                    {partialText}
                  </p>
                ) : (
                  <div className="pondering">
                    <span />
                    <span />
                    <span />
                    <em>
                      {cancellingTurn
                          ? "Waiting for neural worker acknowledgement…"
                          : chatQueue
                            ? `Waiting behind ${chatQueue.queuedBehind.label}…`
                            : pondering
                          ? "Continuing a recurrent ponder cycle…"
                          : actionRunning
                            ? "Running the visible structured action…"
                            : "Forming a response…"}
                    </em>
                  </div>
                )}
              </div>
            </div>
          ) : null}
            <div ref={messagesEnd} />
          </div>
          {!followingOutput || !timelineAtLatest ? (
            <button className="chat-jump-latest" onClick={followLatest}>
              <Icon name="arrow" size={13} /> Latest activity
            </button>
          ) : null}
        </div>
        <div className="composer-wrap">
          {showVoiceStatus && liveVoiceState && voiceStatus ? (
            <div
              className={cx(
                "live-voice-status",
                `live-voice-status--${voiceStatus.tone}`
              )}
              role={voiceStatus.tone === "warning" ? "alert" : "status"}
              aria-live="polite"
            >
              <span className="live-voice-status__icon">
                <Icon name="microphone" size={16} />
                {liveVoiceState.enabled ? <i aria-hidden="true" /> : null}
              </span>
              <p>
                <strong>{voiceStatus.title}</strong>
                <small>{voiceStatus.detail}</small>
              </p>
              {(liveVoiceState.phase === "pondering" ||
                liveVoiceState.phase === "speaking") &&
                liveVoicePreferences.deliveryMode === "live" ? (
                <Button kind="ghost" icon="pause" onClick={bargeInLiveVoice}>
                  Barge in
                </Button>
              ) : null}
              {liveVoiceState.enabled ? (
                <Button kind="ghost" icon="close" onClick={() => void toggleLiveVoice()}>
                  Stop
                </Button>
              ) : liveVoiceState.error?.recoverable ? (
                <Button kind="ghost" icon="microphone" onClick={() => void toggleLiveVoice()}>
                  Try again
                </Button>
              ) : null}
            </div>
          ) : null}
          {liveVoiceState?.enabled ? (
            <>
              <div className="live-voice-settings" aria-label="Live voice settings">
                <span className="live-voice-settings__title">
                  <Icon name="microphone" size={13} /> Voice options
                </span>
                <fieldset>
                  <legend>Delivery</legend>
                  {LIVE_VOICE_DELIVERY_MODES.map((mode) => (
                    <button
                      key={mode}
                      aria-pressed={liveVoicePreferences.deliveryMode === mode}
                      disabled={voiceSettingsLocked}
                      onClick={() => updateLiveVoicePreferences({ deliveryMode: mode })}
                    >
                      {mode === "live" ? "Live" : "Buffered"}
                    </button>
                  ))}
                </fieldset>
                <fieldset>
                  <legend>Pace</legend>
                  {LIVE_VOICE_PACES.map((pace) => (
                    <button
                      key={pace}
                      aria-pressed={liveVoicePreferences.pace === pace}
                      disabled={voiceSettingsLocked}
                      onClick={() => updateLiveVoicePreferences({ pace })}
                    >
                      {pace[0]!.toLocaleUpperCase() + pace.slice(1)}
                    </button>
                  ))}
                </fieldset>
                <label
                  title={
                    liveVoiceState.capabilities.neuralListening
                      ? "Route microphone audio directly through a trained neural audio-input pack"
                      : "Requires a compatible trained neural audio-input pack"
                  }
                >
                  <input
                    type="checkbox"
                    checked={liveVoicePreferences.neuralListening}
                    disabled={
                      voiceSettingsLocked ||
                      (!liveVoiceState.capabilities.neuralListening &&
                        !liveVoicePreferences.neuralListening)
                    }
                    onChange={(event) =>
                      updateLiveVoicePreferences({ neuralListening: event.target.checked })
                    }
                  />
                  Neural listening
                </label>
                <label
                  title={
                    liveVoiceState.capabilities.neuralVoice
                      ? "Play waveform output from this brain's audio region; intelligible speech is not verified"
                      : "Requires this brain's audio region and local WAV playback"
                  }
                >
                  <input
                    type="checkbox"
                    checked={liveVoicePreferences.neuralVoice}
                    disabled={
                      voiceSettingsLocked ||
                      (!liveVoiceState.capabilities.neuralVoice &&
                        !liveVoicePreferences.neuralVoice)
                    }
                    onChange={(event) =>
                      updateLiveVoicePreferences({ neuralVoice: event.target.checked })
                    }
                  />
                  Own neural voice
                </label>
                <small>
                  Default uses platform STT + TTS. Own neural voice needs speech training; speech quality is not verified. No silent fallback.
                </small>
              </div>
            </>
          ) : null}
          <LivePerceptionPanel brain={brain} />
          {queuedTurns.length || queuedLearningJobCount ? (
            <div className="chat-queue-status" role="status" aria-live="polite">
              <span><Icon name="chat" size={14} /></span>
              <p>
                <strong>
                  {queuedTurns.length
                    ? `${queuedTurns.length} message${queuedTurns.length === 1 ? "" : "s"} queued`
                    : "Shared work queue"}
                  {queuedLearningJobCount
                    ? ` · ${queuedLearningJobCount} learning task${queuedLearningJobCount === 1 ? "" : "s"} waiting`
                    : ""}
                </strong>
                <small>Messages and long learning work run in visible order without replacing the active turn.</small>
              </p>
              {queuedTurns.length ? <button onClick={discardQueuedTurns}>Clear messages</button> : null}
            </div>
          ) : null}
          {chatDeliveryStatus ? (
            <div
              className="chat-tool-status chat-tool-status--delivery"
              role="status"
              aria-live="polite"
            >
              <span><Icon name={cancellingTurn ? "pulse" : chatQueue ? "chat" : "info"} size={15} /></span>
              <p>{chatDeliveryStatus}</p>
              {!turnActive ? (
                <button
                  aria-label="Dismiss chat delivery status"
                  onClick={() => setChatDeliveryStatus("")}
                >
                  <Icon name="close" size={13} />
                </button>
              ) : null}
            </div>
          ) : null}
          {postReplyWork.length ? (
            <div className="chat-tool-status" role="status" aria-live="polite"
              aria-label={postReplyWork.map((work) => work.ariaLabel).join(" ")}>
              <span><Icon name="activity" size={15} /></span>
              <p>{postReplyWork.length === 1 ? postReplyWork[0]!.label :
                `${postReplyWork.length} completed replies · save/action work continues`}</p>
            </div>
          ) : null}
          {toolStatus && (pendingTool || toolRunning || attaching) ? (
            <div className={cx("chat-tool-status", pendingTool && "chat-tool-status--approval")}>
              <span><Icon name={pendingTool ? "warning" : "activity"} size={15} /></span>
              <p>{toolStatus}{pendingTool?.approvalExpiresAt ? ` · ${Math.max(0, Math.ceil((Date.parse(pendingTool.approvalExpiresAt) - approvalClock) / 1_000))}s` : ""}</p>
              {pendingTool ? (
                <Button kind="primary" icon="check" disabled={cancellingTurn} onClick={() => void approvePendingTool()}>
                  Approve exact action
                </Button>
              ) : toolRunning ? (
                <small>Cancel the exact job in its action details.</small>
              ) : (
                <button aria-label="Dismiss tool status" onClick={() => setToolStatus("")}>
                  <Icon name="close" size={13} />
                </button>
              )}
            </div>
          ) : null}
          <div className="composer">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={onComposerKeyDown}
              placeholder={
                turnActive
                  ? composerTurn.queueAvailable
                    ? `Message ${brain.name} now — Enter queues it…`
                    : `Message ${brain.name} now — Enter sends it…`
                  : `Talk to ${brain.name}…`
              }
              rows={1}
              aria-label={`Message ${brain.name}`}
              aria-describedby={!draftCapacity.fits && input.trim() ? "chat-input-capacity-error" : undefined}
              aria-keyshortcuts="Enter Shift+Enter Control+Enter Meta+Enter"
            />
            {!draftCapacity.fits && input.trim() ? (
              <div id="chat-input-capacity-error" className="chat-tool-status" role="status">
                <span><Icon name="warning" size={15} /></span>
                <p>{draftCapacity.reason}</p>
              </div>
            ) : null}
            <div className="composer__bottom">
              <div>
                <button
                  aria-label="Upload files and datasets to learn"
                  title="Upload documents, code, datasets, or any supported file"
                  disabled={attaching || sending}
                  onClick={() => void attachExperience("files")}
                >
                  <Icon name={attaching ? "pulse" : "file"} size={18} />
                </button>
                <button
                  aria-label="Upload images to learn"
                  title="Upload one or more images into neural memory"
                  disabled={attaching || sending}
                  onClick={() => void attachExperience("images")}
                >
                  <Icon name="image" size={18} />
                </button>
                <button
                  aria-label="Upload audio to learn"
                  title="Upload one or more audio files into neural memory"
                  disabled={attaching || sending}
                  onClick={() => void attachExperience("audio")}
                >
                  <Icon name="volume" size={18} />
                </button>
                <button
                  aria-label="Upload video to learn"
                  title="Upload one or more videos into neural memory"
                  disabled={attaching || sending}
                  onClick={() => void attachExperience("video")}
                >
                  <Icon name="video" size={18} />
                </button>
                <button
                  aria-label="Upload a folder to learn"
                  title="Upload every supported file in a folder"
                  disabled={attaching || sending}
                  onClick={() => void attachExperience("folder")}
                >
                  <Icon name="archive" size={18} />
                </button>
                <button
                  className={cx(
                    "composer__voice-toggle",
                    liveVoiceState?.enabled && "is-active"
                  )}
                  aria-label={liveVoiceState?.enabled ? "Stop live voice" : "Start live voice"}
                  aria-pressed={liveVoiceState?.enabled ?? false}
                  title={
                    liveVoiceState?.enabled
                      ? "Stop continuous live voice"
                      : "Start continuous live voice (requests microphone permission)"
                  }
                  disabled={sending}
                  onClick={() => void toggleLiveVoice()}
                >
                  <Icon name="microphone" size={18} />
                  <span>Voice</span>
                </button>
              </div>
              <span className="composer__status-copy">
                <span className="composer__key-hint">
                  {turnActive
                    ? "Enter queues · Ctrl/Cmd Enter steers · empty input keeps Stop"
                    : "Enter to send · Shift Enter for a line break"}
                </span>
                <span
                  className="composer__token-counter"
                  aria-label={`Token counter and neural learning status. ${composerTelemetryAriaLabel}`}
                  title={composerTelemetryAriaLabel}
                >
                  {draftTokenCount.toLocaleString()} draft · {contextTokenCopy} · {composerTelemetryCopy}
                </span>
              </span>
              {turnActive ? (
                <div
                  className="composer__turn-choices"
                  role="group"
                  aria-label={composerTurn.queueAvailable && input.trim()
                    ? "Message choices and current turn controls"
                    : "Current turn controls"}
                >
                  {input.trim() && !cancellingTurn && composerTurn.queueAvailable ? (
                    <>
                      {composerTurn.steerAvailable ? (
                        <button
                          className="composer__steer-choice"
                          aria-label="Steer current turn"
                          title="Apply this direction at the next safe neural boundary; the prior request remains visible and the loaded mind stays warm"
                          disabled={!draftCapacity.fits}
                          onClick={() => void steerCurrentTurn()}
                        >
                          <Icon name="arrow" size={13} /> <span>Steer now</span>
                        </button>
                      ) : null}
                      <button
                        className="send-button composer__queue-choice"
                        aria-label="Queue message"
                        title="Run this message next without interrupting the active reply"
                        disabled={!draftCapacity.fits}
                        onClick={submitOrdinaryTurn}
                      >
                        <Icon name="chat" size={13} /> <span>Queue</span>
                      </button>
                    </>
                  ) : null}
                  <button
                    className="composer__stop-choice"
                    onClick={() => void cancelChatTool()}
                    disabled={cancellingTurn}
                    aria-label={cancellingTurn ? "Stopping current turn" : "Stop current turn"}
                  >
                    <Icon name={cancellingTurn ? "pulse" : "close"} size={13} />
                    <span>{cancellingTurn ? "Stopping" : "Stop"}</span>
                  </button>
                </div>
              ) : (
                <button
                  className="send-button"
                  onClick={() => void send()}
                  disabled={!input.trim() || !window.omni || !draftCapacity.fits}
                  aria-label="Send message"
                  title={!window.omni ? "Neural engine unavailable in design preview" : undefined}
                >
                  <Icon name="send" size={17} />
                </button>
              )}
            </div>
            <div className="composer__suggestions" aria-label="Natural chat examples">
              <button onClick={() => setInput("Search the web for ")}><Icon name="search" size={13} /> Search the web</button>
              <button onClick={() => setInput("Imagine an image of ")}><Icon name="image" size={13} /> Imagine</button>
              <button onClick={() => setInput("Fork an agent to ")}><Icon name="agents" size={13} /> Create an agent</button>
              <button onClick={() => setInput("Improve your own implementation by ")}><Icon name="pulse" size={13} /> Evolve</button>
            </div>
          </div>
          <p className="composer-disclaimer" role={!window.omni ? "status" : undefined}>
            {window.omni
              ? "Learned parameters, active state, and structured tool schemas drive each turn. Tool actions stay permissioned and visible."
              : "Neural engine unavailable in this design preview. Drafts stay in the composer; no chat or learning runs."}
          </p>
        </div>
      </section>

      <aside id="chat-cortex-panel" className="cortex-panel" aria-label="Cortex and runtime inspector">
        <div className="panel-tabs">
          <button className={inspectorTab === "state" ? "is-active" : ""} onClick={() => setInspectorTab("state")}>
            Live cortex
          </button>
          <button className={inspectorTab === "runtime" ? "is-active" : ""} onClick={() => setInspectorTab("runtime")}>
            Runtime card
          </button>
        </div>
        {inspectorTab === "state" ? (
          <>
            <CortexOrb activity={measuredActivity} />
            <div className="cortex-readout">
              <span>
                <i className="dot-violet" />
                <small>Current state</small>
                <strong>{cortexState}</strong>
              </span>
              <span>
                <small>Liquid τ</small>
                <strong>
                  {brain.liquidState.timeConstants.length
                    ? `${(
                        brain.liquidState.timeConstants.reduce((sum, value) => sum + value, 0) /
                        brain.liquidState.timeConstants.length
                      ).toFixed(2)}×`
                    : "—"}
                </strong>
              </span>
            </div>
            {chatParameterAccounting ? (
              <div className="cortex-parameter-count">
                <span>
                  <small>Total neural parameters</small>
                  <NeuralParameterCount accounting={chatParameterAccounting} />
                </span>
                <em>authoritative count</em>
              </div>
            ) : null}
            <div className="panel-section">
              <div className="panel-section__head">
                <span>Organic signals</span>
                <em>measured, not configured</em>
              </div>
              <div className="drive-grid">
                {(recentTrace ? [
                  ["Exploration", recentTrace.driveScores.curiosity],
                  ["Coherence", recentTrace.driveScores.coherence],
                  ["Novelty", recentTrace.driveScores.novelty]
                ] : []).map(([label, value]) => (
                  <div key={String(label)}>
                    <span>{label}</span>
                    <strong>{Math.round(Number(value) * 100)}%</strong>
                    <i>
                      <b style={{ width: `${Number(value) * 100}%` }} />
                    </i>
                  </div>
                ))}
                {!recentTrace ? <span className="organic-empty">Signals form after the first neural turn.</span> : null}
              </div>
            </div>
            <button className="trace-link" onClick={() => onNavigate("trace")}>
              <Icon name="trace" size={15} />
              Open latest operational trace
              <Icon name="arrow" size={13} />
            </button>
          </>
        ) : (
          <RuntimeCard
            brain={brain}
            workspace={workspaceSnapshot}
            telemetry={workspaceTelemetry}
          />
        )}
      </aside>
    </div>
  );
}

function MessageBubble({
  message,
  brainName,
  onToast,
  onTrace,
  onOwnVoice,
  ownVoicePhase,
  ownVoiceUnavailable,
  ownVoiceQuality,
  uncommittedOutputState
}: {
  message: ChatMessage;
  brainName: string;
  onToast: (message: string) => void;
  onTrace: () => void;
  onOwnVoice?: () => void;
  ownVoicePhase?: "preparing" | "playing";
  ownVoiceUnavailable?: boolean;
  ownVoiceQuality?: "needs-speech-training" | "unverified";
  uncommittedOutputState?: UncommittedChatOutput["state"];
}) {
  const isBrain = message.role === "brain";
  const deliveryState = chatMessageDeliveryState(message);
  const durableReceipt = message.deliveryReceipt?.presentationOnly === true;
  const queued = deliveryState === "queued";
  const steered = deliveryState === "steered";
  const stopped = deliveryState === "stopped";
  const noReply = isBrain && deliveryState === "no-reply";
  const noReplyPresentation = chatNoReplyPresentation(message);
  const failed = deliveryState === "failed";
  const cancelled = deliveryState === "cancelled";
  const pending = deliveryState === "pending" ||
    (queued && !durableReceipt);
  return (
    <article
      className={cx(
        "message",
        isBrain ? "message--brain" : "message--human",
        pending && "message--pending",
        queued && "message--queued",
        steered && "message--steered",
        failed && "message--failed",
        cancelled && "message--cancelled"
      )}
      data-turn-state={
        noReply ? "no-reply" : stopped ? "stopped" : !isBrain && message.inputReceipt?.committed ? "received" : uncommittedOutputState
          ? `output-uncommitted-${uncommittedOutputState}`
          : failed
          ? "failed"
          : cancelled
            ? "cancelled"
            : queued
              ? "queued"
              : steered
                ? "steered"
                : pending
                  ? "pending"
                  : "committed"
      }
      aria-busy={pending || undefined}
      aria-invalid={failed || undefined}
    >
      <div className="message__avatar">
        {isBrain ? <BrandMark size={28} /> : <span className="human-avatar">E</span>}
      </div>
      <div className="message__body">
        <div className="message__meta">
          <strong>{isBrain ? brainName : "You"}</strong>
          <time>
            {new Date(message.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
          </time>
          {!isBrain && message.inputReceipt?.committed ? <span className="message-queue-label">Received</span> : null}
          {uncommittedOutputState ? (
            <span className="message-queue-label">
              {uncommittedOutputState === "saving"
                ? "Reply complete · save pending, not committed"
                : uncommittedOutputState === "steering"
                  ? "Steering · emitted prefix awaiting safe save"
                : uncommittedOutputState === "cancelled"
                  ? "Stopped output · not committed"
                  : "Save failed · output not committed"}
            </span>
          ) : null}
          {queued ? (
            <span className="message-queue-label">
              {durableReceipt ? "Queued · not replayed" : "Queued"}
            </span>
          ) : null}
          {deliveryState === "pending" && durableReceipt ? (
            <span className="message-queue-label">Awaiting reply · status unconfirmed</span>
          ) : null}
          {steered ? <span className="message-queue-label message-steered-label">
            {isBrain && message.generationEnd === "steered" ? "Steered · partial reply saved" : "Steered"}
          </span> : null}
          {stopped ? <span className="message-queue-label">
            {message.generationEnd === "native-stop" ? "Brain stopped · partial reply saved" : "Brain stopped before replying"}
          </span> : null}
          {noReplyPresentation ? <span className="message-queue-label" aria-label={noReplyPresentation.ariaLabel}>
            {noReplyPresentation.label}
          </span> : null}
          {failed ? (
            <span className="message-queue-label message-failed-label">Not sent · retry ready</span>
          ) : null}
          {cancelled ? (
            <span className="message-queue-label message-cancelled-label">Cancelled</span>
          ) : null}
          {isBrain && message.runtime ? <span className="runtime-label">{message.runtime.replace("-", " ")}</span> : null}
        </div>
        {noReply ? null : <div className="message__content">{message.content}</div>}
        {isBrain ? (
          <div className="message__actions">
            {message.content ? <button aria-label="Copy response" onClick={() => void navigator.clipboard?.writeText(message.content)}>
              <Icon name="copy" size={14} />
            </button> : null}
            <button
              aria-label="Show trace"
              onClick={() => (message.traceId ? onTrace() : onToast("No trace is attached to this message."))}
            >
              <Icon name="trace" size={14} />
            </button>
            {onOwnVoice && !uncommittedOutputState && !noReply && Boolean(message.content.trim()) ? (
              <button
                aria-label={ownVoicePhase ? "Stop own waveform playback" : "Play reply with own neural voice"}
                title={`${ownVoicePhase === "preparing" ? "Producing waveform · click to cancel" : ownVoicePhase === "playing"
                  ? "Playing waveform · click to stop" : "Play using this brain's audio region"}. ${ownVoiceQuality === "unverified"
                  ? "Speech quality not verified" : "Needs speech training; the waveform may not match the words yet"}.`}
                disabled={ownVoiceUnavailable && !ownVoicePhase}
                onClick={onOwnVoice}
              >
                <Icon name={ownVoicePhase ? "close" : "volume"} size={14} />
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </article>
  );
}

function CortexOrb({ activity }: { activity: number | null }) {
  const measured = activity === null
    ? null
    : Math.max(0, Math.min(1, activity));
  return (
    <div
      className="cortex-orb"
      aria-label={
        measured === null
          ? "No observed activation measurement in the latest trace"
          : `${Math.round(measured * 100)} percent mean observed activation in the latest completed trace; not the fraction of the whole brain firing`
      }
    >
      <svg viewBox="0 0 240 170" role="img" aria-hidden="true">
        <defs>
          <radialGradient id="orbGlow" cx="50%" cy="46%" r="60%">
            <stop offset="0" stopColor="#d9ccff" stopOpacity=".42" />
            <stop offset=".45" stopColor="#8d6cff" stopOpacity=".2" />
            <stop offset="1" stopColor="#4b2f92" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="nodeLine" x1="0" y1="0" x2="1" y2="1">
            <stop stopColor="#a993ff" stopOpacity=".12" />
            <stop offset=".5" stopColor="#d2c7ff" stopOpacity=".7" />
            <stop offset="1" stopColor="#5ce0d8" stopOpacity=".16" />
          </linearGradient>
          <filter id="softGlow">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <ellipse cx="120" cy="84" rx="89" ry="73" fill="url(#orbGlow)" />
        {[
          [66, 81, 101, 44],
          [101, 44, 151, 55],
          [151, 55, 178, 92],
          [178, 92, 139, 122],
          [139, 122, 84, 129],
          [84, 129, 66, 81],
          [101, 44, 112, 85],
          [151, 55, 112, 85],
          [178, 92, 112, 85],
          [139, 122, 112, 85],
          [84, 129, 112, 85],
          [66, 81, 112, 85],
          [84, 129, 151, 55],
          [66, 81, 139, 122]
        ].map(([x1, y1, x2, y2], index) => (
          <line key={index} x1={x1} y1={y1} x2={x2} y2={y2} stroke="url(#nodeLine)" strokeWidth={index % 3 === 0 ? 1.3 : 0.8} />
        ))}
        {[
          [66, 81, 4],
          [101, 44, 3.4],
          [151, 55, 4.5],
          [178, 92, 3.4],
          [139, 122, 4],
          [84, 129, 3.1],
          [112, 85, 6.5],
          [128, 69, 2.4],
          [95, 98, 2.8]
        ].map(([cxValue, cy, radius], index) => (
          <circle
            key={index}
            cx={cxValue}
            cy={cy}
            r={radius}
            fill={index === 6 ? "#e8e3ff" : index % 3 === 0 ? "#6de3d9" : "#9c83ff"}
            opacity={0.55 + (index % 4) * 0.1}
            filter="url(#softGlow)"
          />
        ))}
        <path d="M53 68c-14 26-5 59 19 77M181 55c17 17 23 39 16 61" fill="none" stroke="#9d87ee" strokeOpacity=".18" />
      </svg>
      <span className="cortex-orb__caption">
        <i />
        {measured === null
          ? "No completed trace"
          : `${Math.round(measured * 100)}% last measured`}
      </span>
    </div>
  );
}

function NeuralParameterCount({
  accounting
}: {
  accounting: NeuralParameterAccountingPresentation;
}) {
  return (
    <span
      className="neural-parameter-count"
      tabIndex={0}
      role="note"
      title={accounting.exactLabel}
      aria-label={accounting.exactLabel}
    >
      <strong>{accounting.compactTotal}</strong>
      <span className="neural-parameter-count__detail" aria-hidden="true">
        <b>{accounting.totalNeuralParameters.toLocaleString("en-US")} logical weights and connections</b>
        <span>{accounting.mutableDenseParameters.toLocaleString("en-US")} core weights</span>
        {(accounting.substrateVectorParameters ?? 0) > 0 && (
          <span>{accounting.substrateVectorParameters!.toLocaleString("en-US")} learned memory weights</span>
        )}
        <span>{accounting.dynamicSparseSynapses.toLocaleString("en-US")} grown connections</span>
        <small>One brain; each logical parameter counted once. {accounting.countingRule}</small>
      </span>
    </span>
  );
}

function BrainSnapshotsPanel({
  brain,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const snapshotApiAvailable =
    typeof window.omni?.brain?.listSnapshots === "function";
  const [snapshots, setSnapshots] = useState<BrainSnapshotSummary[]>([]);
  const [label, setLabel] = useState("");
  const [busy, setBusy] = useState<"loading" | "creating" | `restore:${string}` | "">(
    snapshotApiAvailable ? "loading" : ""
  );
  const [status, setStatus] = useState(
    snapshotApiAvailable
      ? "Loading recovery points…"
      : "Recovery points require the current packaged desktop runtime."
  );

  const refreshSnapshots = useCallback(async (): Promise<void> => {
    const listSnapshots = window.omni?.brain?.listSnapshots;
    if (typeof listSnapshots !== "function") {
      setStatus("Recovery points require the current packaged desktop runtime.");
      return;
    }
    setBusy("loading");
    setStatus("Refreshing recovery points…");
    try {
      const next = await listSnapshots(brain.id);
      setSnapshots(next);
      setStatus(
        next.length
          ? `${next.length} recovery point${next.length === 1 ? "" : "s"} available.`
          : "No recovery points yet. Create one before a risky change."
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Recovery points could not be loaded.");
    } finally {
      setBusy("");
    }
  }, [brain.id]);

  useEffect(() => {
    let active = true;
    const listSnapshots = window.omni?.brain?.listSnapshots;
    if (typeof listSnapshots !== "function") return () => {
      active = false;
    };
    setBusy("loading");
    void listSnapshots(brain.id).then((next) => {
      if (!active) return;
      setSnapshots(next);
      setStatus(
        next.length
          ? `${next.length} recovery point${next.length === 1 ? "" : "s"} available.`
          : "No recovery points yet. Create one before a risky change."
      );
    }).catch((error: unknown) => {
      if (active) {
        setStatus(error instanceof Error ? error.message : "Recovery points could not be loaded.");
      }
    }).finally(() => {
      if (active) setBusy("");
    });
    return () => {
      active = false;
    };
  }, [brain.id]);

  const createSnapshot = async (): Promise<void> => {
    if (!window.omni || busy) return;
    setBusy("creating");
    setStatus("Saving the current document and neural state…");
    try {
      const operationId = globalThis.crypto?.randomUUID?.() ??
        `snapshot-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const created = await window.omni.brain.snapshot(
        brain.id,
        label.trim() || undefined,
        operationId
      );
      setSnapshots((current) => [created, ...current.filter((item) => item.id !== created.id)]);
      setLabel("");
      setStatus(`Recovery point “${created.label}” saved and verified.`);
      onToast(`Recovery point “${created.label}” created.`);
    } catch (error) {
      setStatus(recoveryPointCreationErrorMessage(error));
    } finally {
      setBusy("");
    }
  };

  const restoreSnapshot = async (snapshot: BrainSnapshotSummary): Promise<void> => {
    if (!window.omni || busy) return;
    const confirmed = window.confirm(
      `Restore “${snapshot.label}” from ${new Date(snapshot.createdAt).toLocaleString()}?\n\n` +
      "This replaces the current document and neural state for this identity. Changes made after that recovery point are not kept. Create a new recovery point first if you may need them."
    );
    if (!confirmed) {
      setStatus("Restore cancelled; the current brain was not changed.");
      return;
    }
    setBusy(`restore:${snapshot.id}`);
    setStatus(`Restoring “${snapshot.label}” and reloading the neural worker…`);
    try {
      const operationId = globalThis.crypto?.randomUUID?.() ??
        `restore-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const restored = await window.omni.brain.restoreSnapshot(
        brain.id,
        snapshot.id,
        operationId
      );
      onBrainChange(restored);
      setStatus(`Restored “${snapshot.label}”. The recovered neural state is active.`);
      onToast(`Restored recovery point “${snapshot.label}”.`);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "The recovery point could not be restored.");
    } finally {
      setBusy("");
    }
  };

  return (
    <section className="surface brain-snapshots" aria-labelledby="brain-snapshots-heading" aria-busy={Boolean(busy)}>
      <div className="resource-settings-section-heading brain-snapshots__heading">
        <span className="resource-settings-section-icon"><Icon name="archive" size={18} /></span>
        <span>
          <h2 id="brain-snapshots-heading">Recovery points</h2>
          <p>Save and restore this identity&apos;s exact document plus durable neural state.</p>
        </span>
      </div>
      <div className="brain-snapshots__create">
        <label>
          <span>Optional label</span>
          <input
            value={label}
            maxLength={120}
            disabled={Boolean(busy) || !window.omni}
            onChange={(event) => setLabel(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void createSnapshot();
            }}
            placeholder="Before tool changes"
          />
        </label>
        <Button
          kind="primary"
          icon={busy === "creating" ? "pulse" : "archive"}
          disabled={Boolean(busy) || !window.omni}
          onClick={() => void createSnapshot()}
        >
          {busy === "creating" ? "Saving…" : "Create recovery point"}
        </Button>
        <Button
          icon="pulse"
          disabled={Boolean(busy) || !window.omni}
          onClick={() => void refreshSnapshots()}
        >
          Refresh
        </Button>
      </div>
      <p className="brain-snapshots__status" role="status" aria-live="polite">{status}</p>
      <div className="brain-snapshots__list">
        {snapshots.map((snapshot) => {
          const restoring = busy === `restore:${snapshot.id}`;
          return (
            <article key={snapshot.id}>
              <span>
                <strong>{snapshot.label}</strong>
                <small>
                  {new Date(snapshot.createdAt).toLocaleString()} · {snapshot.metrics.messages.toLocaleString()} messages · {snapshot.metrics.synapses.toLocaleString()} connections · {formatBytes(snapshot.storage?.logicalBytes ?? snapshot.metrics.estimatedBytes)} logical{snapshot.storage ? ` · ${formatBytes(snapshot.storage.sharedBytes)} shared` : ""}
                </small>
              </span>
              <Button
                kind="ghost"
                icon={restoring ? "pulse" : "arrow"}
                disabled={Boolean(busy)}
                onClick={() => void restoreSnapshot(snapshot)}
              >
                {restoring ? "Restoring…" : "Restore"}
              </Button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function RuntimeCard({
  brain,
  workspace,
  telemetry
}: {
  brain: BrainDocument;
  workspace?: WorkspaceSnapshot | null;
  telemetry?: WorkspaceTelemetryPresentation;
}) {
  const stableConfig = brain.config as BrainConfig & {
    recursiveImprovement?: boolean;
  };
  const enabledTools = (brain.toolPermissions ?? [])
    .filter((permission) => permission.level !== "off")
    .map((permission) => permission.toolId);
  const evolutionPermission = (brain.toolPermissions ?? [])
    .find((permission) => permission.toolId === "source.self-modify")?.level ?? "off";
  const parameterAccounting = neuralParameterAccounting(workspace?.runtimeCard);
  const provenance = brainProvenancePresentation({
    provenance: brain.provenance,
    runtimeCard: workspace?.runtimeCard,
    trainingSources: brain.trainingSources
  });
  const rows = [
    ["Behavioral system prompt", workspace ? (workspace.hiddenBehavioralPrompt ? "Present" : "None") : "Pending measurement"],
    ["Long-term source injection", workspace ? (workspace.rawLongTermTextInjected ? "Present" : "None") : "Pending measurement"],
    ["Reward model / RLHF", "None"],
    ["Ternary forward paths", "Mandatory · −1 / 0 / +1"],
    ["Runtime", brain.config.runtime],
    [
      "Temporary context",
      telemetry?.contextText ?? "Measuring committed neural state…"
    ],
    [
      "Immediate neural changes",
      telemetry?.experienceText ?? "Measuring neural learning…"
    ],
    [
      "Deferred neural updates",
      telemetry?.backgroundText ?? "Measuring background neural work…"
    ],
    [
      "Core weight verification",
      telemetry?.corticalWeightsText ?? "Not independently verified"
    ],
    ...(telemetry?.backgroundErrorText
      ? [["Last slow-replay error", telemetry.backgroundErrorText]]
      : []),
    [
      "Change since last measurement",
      telemetry?.deltaText ?? "Waiting for the first committed measurement"
    ],
    ["Last measured", telemetry?.measuredText ?? "Waiting for neural state"],
    ...(telemetry?.freshAttentionText
      ? [["Fresh attention", telemetry.freshAttentionText]]
      : []),
    [
      "Baseline response budget",
      workspace?.contextWindow.generationBudgetTokens
        ? `${workspace.contextWindow.generationBudgetTokens} tokens · adjusted per turn from neural state`
        : "Hardware baseline · adjusted per turn from neural state"
    ],
    ["Storage assist · prior check", brain.config.memoryOffloadBytes > 0 ? `${formatBytes(brain.config.memoryOffloadBytes)} on demand · estimated ${brain.config.memoryOffloadSlowdownPercent}% slower for memory-heavy work` : "None then · see current errors above"],
    ...neuralResourceRuntimeRows(workspace?.runtimeCard),
    ["Liquid recurrent state", workspace ? `${workspace.liquidState.dimensions} dimensions · norm ${workspace.liquidState.norm.toFixed(2)}` : `${brain.liquidState.values.length} channels`],
    [
      "Recorded learning counters",
      workspace?.learning
        ? `${workspace.learning.fastNeuralMemory.connectionUpdatesTotal.toLocaleString()} local connection updates · ${workspace.learning.fastNeuralMemory.parameterStepsTotal.toLocaleString()} optimizer steps across all training`
        : "Measuring worker-owned neural counters…"
    ],
    ["Recursive improvement", evolutionPermission === "off" || stableConfig.recursiveImprovement === false ? "Off" : `${evolutionPermission} permission`],
    ["Tool schemas", enabledTools.length ? `${enabledTools.length} visible` : "None enabled"],
    ["Trace detail", brain.config.traceDetail]
  ];
  return (
    <div className="runtime-card">
      <div className="runtime-card__seal">
        <Icon name="check" size={20} />
      </div>
      <h3>Transparent runtime</h3>
      <p>Live temporary context, neural learning, tool availability, and runtime boundaries are measured here.</p>
      {provenance ? (
        <div
          className="runtime-card__provenance"
          role="note"
          title={provenance.ariaLabel}
          aria-label={provenance.ariaLabel}
        >
          <strong>{provenance.originLabel}</strong>
          <span>{provenance.compactLabel}</span>
        </div>
      ) : null}
      {parameterAccounting ? (
        <div className="runtime-card__parameter-count">
          <span>Total neural parameters</span>
          <NeuralParameterCount accounting={parameterAccounting} />
        </div>
      ) : null}
      <dl>
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
      <div className="runtime-card__note">
        <Icon name="info" size={15} />
        Immediate connection changes and deferred weight updates belong to the same brain. Core weights are shown as changed only after their own checksum verifies an update. Temporary context remains available until natural decay, capacity eviction, or Fresh attention. Tool schemas describe available actions, not a personality.
      </div>
    </div>
  );
}

function DeviceRuntimeWorkspace({
  brain,
  onActiveModeChange,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onActiveModeChange: (result: BrainActiveModeResult) => Promise<void>;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const persistedMode = brain.config.systemRamMode ?? "auto";
  const persistedShare = persistedMode === "manual"
    ? clampSystemRamSharePercent(brain.config.systemRamSharePercent)
    : 0;
  const persistedMemoryMode = brain.config.workingMemoryMode ?? "auto";
  const persistedContextTokens = brain.config.contextWindowTokens;
  const persistedStoragePoolMode = brain.config.storagePoolMode ?? "auto";
  const persistedStoragePoolBytes = brain.config.storagePoolBytes ?? 0;
  const [systemRamMode, setSystemRamMode] = useState<SystemRamMode>(persistedMode);
  const [systemRamSharePercent, setSystemRamSharePercent] = useState(
    initialManualSystemRamSharePercent(brain.config)
  );
  const [workingMemoryMode, setWorkingMemoryMode] = useState<WorkingMemoryMode>(
    persistedMemoryMode
  );
  const [manualContextTokens, setManualContextTokens] = useState(
    String(persistedContextTokens)
  );
  const [storagePoolMode, setStoragePoolMode] = useState<StoragePoolMode>(
    persistedStoragePoolMode
  );
  const [manualStoragePoolGiB, setManualStoragePoolGiB] = useState(
    String(Math.max(1, Math.ceil(persistedStoragePoolBytes / GIB_BYTES)))
  );
  const [detectedHardware, setDetectedHardware] = useState<SystemHardwareProfile | null>(null);
  const [resourcePlan, setResourcePlan] = useState<WorkingMemoryResourcePlan | null>(null);
  const [resourcePlanPending, setResourcePlanPending] = useState(Boolean(window.omni));
  const [resourceError, setResourceError] = useState("");
  const [saving, setSaving] = useState(false);
  const [mobileStatus, setMobileStatus] = useState<MobileGatewayStatus | null>(null);
  const [mobilePairing, setMobilePairing] = useState<MobilePairingSession | null>(null);
  const [mobileLan, setMobileLan] = useState(false);
  const [mobileBusy, setMobileBusy] = useState(false);
  const [mobileError, setMobileError] = useState("");
  const [freshAttentionBusy, setFreshAttentionBusy] = useState(false);
  const [activeModeBusy, setActiveModeBusy] = useState(false);
  const [activeModeStatus, setActiveModeStatus] = useState("");

  useEffect(() => {
    const nextMode = brain.config.systemRamMode ?? "auto";
    setSystemRamMode(nextMode);
    setSystemRamSharePercent(initialManualSystemRamSharePercent(brain.config));
    setWorkingMemoryMode(brain.config.workingMemoryMode ?? "auto");
    setManualContextTokens(String(brain.config.contextWindowTokens));
    setStoragePoolMode(brain.config.storagePoolMode ?? "auto");
    setManualStoragePoolGiB(
      String(Math.max(1, Math.ceil((brain.config.storagePoolBytes ?? 0) / GIB_BYTES)))
    );
  }, [
    brain.id,
    brain.config.contextWindowTokens,
    brain.config.storagePoolBytes,
    brain.config.storagePoolMode,
    brain.config.systemRamMode,
    brain.config.systemRamSharePercent,
    brain.config.workingMemoryMode
  ]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void window.omni.mobile.status().then((status) => {
      if (!active) return;
      setMobileStatus(status);
      setMobileLan(status.allowLan);
    }).catch((error: unknown) => {
      if (active) setMobileError(error instanceof Error ? error.message : "Mobile status is unavailable.");
    });
    return () => {
      active = false;
    };
  }, [brain.id]);

  useEffect(() => {
    if (!window.omni) {
      setResourcePlanPending(false);
      return;
    }
    let active = true;
    setResourceError("");
    void window.omni.catalog.hardwareProfile().then((profile) => {
      if (active) setDetectedHardware(profile);
    }).catch((error: unknown) => {
      if (!active) return;
      setResourceError(
        error instanceof Error ? error.message : "This device could not be measured."
      );
      setResourcePlanPending(false);
    });
    return () => {
      active = false;
    };
  }, [brain.id]);

  useEffect(() => {
    if (!window.omni || !detectedHardware) return;
    let active = true;
    const timer = window.setTimeout(() => {
      setResourcePlanPending(true);
      setResourceError("");
      void window.omni!.catalog.resourcePlan({
        mode: workingMemoryMode,
        brainId: brain.id,
        acceleratorAvailable: detectedHardware.gpu.available,
        systemRamMode,
        ...(systemRamMode === "manual" ? { systemRamSharePercent } : {}),
        storagePoolMode,
        ...(storagePoolMode === "manual"
          ? { storagePoolBytes: wholeGiBTextToBytes(manualStoragePoolGiB) ?? "" }
          : {}),
        ...(workingMemoryMode === "manual"
          ? {
              requestedContextTokens: manualContextTokens
            }
          : {})
      }).then((plan) => {
        if (active) setResourcePlan(plan);
      }).catch((error: unknown) => {
        if (!active) return;
        setResourcePlan(null);
        setResourceError(
          error instanceof Error ? error.message : "The resource preflight could not run."
        );
      }).finally(() => {
        if (active) setResourcePlanPending(false);
      });
    }, systemRamMode === "manual" || workingMemoryMode === "manual" || storagePoolMode === "manual" ? 180 : 0);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [
    brain.id,
    brain.config.workingMemorySlots,
    brain.config.nativeArchitecture?.sha256,
    detectedHardware,
    manualContextTokens,
    manualStoragePoolGiB,
    storagePoolMode,
    systemRamMode,
    systemRamSharePercent,
    workingMemoryMode
  ]);

  const dirty =
    systemRamMode !== persistedMode ||
    (systemRamMode === "manual" && systemRamSharePercent !== persistedShare) ||
    workingMemoryMode !== persistedMemoryMode ||
    (workingMemoryMode === "manual" &&
      Number.parseInt(manualContextTokens, 10) !== persistedContextTokens) ||
    storagePoolMode !== persistedStoragePoolMode ||
    (storagePoolMode === "manual" &&
      wholeGiBTextToBytes(manualStoragePoolGiB) !== String(persistedStoragePoolBytes));
  const planBlocked = Boolean(resourcePlan && !resourcePlan.allowed);
  const contextFloorTokens = Math.max(8, resourcePlan?.context.floorTokens ?? 8);
  const contextMaximumTokens = Math.max(8, resourcePlan?.context.maximumTokens ?? 8);
  const contextSliderMinimumTokens = Math.min(contextFloorTokens, contextMaximumTokens);
  const storageRequiredGiB = Math.max(
    1,
    Math.ceil((resourcePlan?.resources.requiredStoragePoolBytes ?? GIB_BYTES) / GIB_BYTES)
  );
  const storageMaximumGiB = Math.max(
    1,
    Math.floor((resourcePlan?.resources.maximumStoragePoolBytes ?? GIB_BYTES) / GIB_BYTES)
  );
  const storageSliderMinimumGiB = Math.min(storageRequiredGiB, storageMaximumGiB);
  const storageSliderUnavailable = Boolean(
    resourcePlan && storageRequiredGiB > storageMaximumGiB
  );

  const chooseSystemRamMode = (mode: SystemRamMode): void => {
    if (mode === "manual" && systemRamMode !== "manual") {
      setSystemRamSharePercent(
        clampSystemRamSharePercent(
          resourcePlan?.resources.autoSystemRamSharePercent ??
            initialManualSystemRamSharePercent(brain.config)
        )
      );
    }
    setSystemRamMode(mode);
  };

  const chooseWorkingMemoryMode = (mode: WorkingMemoryMode): void => {
    if (mode === "manual" && workingMemoryMode !== "manual") {
      setManualContextTokens(
        String(resourcePlan?.context.selectedTokens ?? persistedContextTokens)
      );
    }
    setWorkingMemoryMode(mode);
  };

  const chooseStoragePoolMode = (mode: StoragePoolMode): void => {
    if (mode === "manual" && storagePoolMode !== "manual") {
      const suggestedBytes = Math.max(
        resourcePlan?.resources.sharedStoragePoolBytes ?? 0,
        resourcePlan?.resources.requiredStoragePoolBytes ?? 0,
        persistedStoragePoolBytes
      );
      setManualStoragePoolGiB(
        String(Math.max(1, Math.ceil(suggestedBytes / GIB_BYTES)))
      );
    }
    setStoragePoolMode(mode);
  };

  const saveResourceEnvelope = async (): Promise<void> => {
    if (!resourcePlan?.allowed || resourcePlanPending || saving) return;
    setSaving(true);
    setResourceError("");
    try {
      const nextConfig = configWithResourceEnvelope(
        brain.config,
        resourcePlan,
        systemRamMode,
        systemRamSharePercent
      );
      const updated = window.omni
        ? await window.omni.brain.update(brain.id, nextConfig)
        : {
            ...brain,
            config: nextConfig,
            updatedAt: new Date().toISOString()
          };
      onBrainChange(updated);
      onToast("RAM, active context, and shared storage capacity saved.");
    } catch (error) {
      setResourceError(
        error instanceof Error ? error.message : "The RAM envelope could not be saved."
      );
    } finally {
      setSaving(false);
    }
  };

  const resetResourceEnvelope = (): void => {
    setSystemRamMode(persistedMode);
    setSystemRamSharePercent(initialManualSystemRamSharePercent(brain.config));
    setWorkingMemoryMode(persistedMemoryMode);
    setManualContextTokens(String(persistedContextTokens));
    setStoragePoolMode(persistedStoragePoolMode);
    setManualStoragePoolGiB(
      String(Math.max(1, Math.ceil(persistedStoragePoolBytes / GIB_BYTES)))
    );
  };

  const startFreshAttention = async (): Promise<void> => {
    if (!window.omni || freshAttentionBusy) return;
    const confirmed = window.confirm(
      "Start fresh attention?\n\n" +
      "This clears only temporary prompt and active neural state: recent context, working thoughts, fading scratch, and firing/spike/liquid activity. " +
      "It keeps visible chat history, learned weights and connections, long-term neural memory, replay, training receipts, tools, provenance, and origin.\n\n" +
      "Earlier chat text will not be supplied automatically to new responses. It remains visible and can re-enter only when you explicitly use the brain.history tool."
    );
    if (!confirmed) return;
    setFreshAttentionBusy(true);
    try {
      const result = await window.omni.brain.freshAttention(brain.id);
      window.dispatchEvent(new CustomEvent("omni:workspace-invalidated", {
        detail: { brainId: brain.id }
      }));
      onToast(
        `Fresh attention started. ${result.messagesPreserved.toLocaleString()} visible messages and learned neural memory were preserved.`
      );
    } catch (error) {
      onToast(
        error instanceof Error
          ? error.message
          : "Fresh attention could not be started."
      );
    } finally {
      setFreshAttentionBusy(false);
    }
  };

  const startMobilePairing = async (): Promise<void> => {
    if (!window.omni || mobileBusy) return;
    setMobileBusy(true);
    setMobileError("");
    try {
      const session = await window.omni.mobile.startPairing({ allowLan: mobileLan });
      setMobilePairing(session);
      setMobileStatus(session);
      onToast("A one-time mobile pairing code is ready for five minutes.");
    } catch (error) {
      setMobileError(error instanceof Error ? error.message : "Mobile pairing could not start.");
    } finally {
      setMobileBusy(false);
    }
  };

  const stopMobileGateway = async (): Promise<void> => {
    if (!window.omni || mobileBusy) return;
    setMobileBusy(true);
    setMobileError("");
    try {
      const status = await window.omni.mobile.stop();
      setMobileStatus(status);
      setMobilePairing(null);
      onToast("Mobile companion access stopped; the brain remained running locally.");
    } catch (error) {
      setMobileError(error instanceof Error ? error.message : "Mobile access could not stop.");
    } finally {
      setMobileBusy(false);
    }
  };

  const revokeMobileDevice = async (deviceId: string): Promise<void> => {
    if (!window.omni || mobileBusy) return;
    setMobileBusy(true);
    setMobileError("");
    try {
      const status = await window.omni.mobile.revoke(deviceId);
      setMobileStatus(status);
      onToast("That mobile device can no longer access this Studio runtime.");
    } catch (error) {
      setMobileError(error instanceof Error ? error.message : "The paired device could not be removed.");
    } finally {
      setMobileBusy(false);
    }
  };

  const setActiveModeEnabled = async (enabled: boolean): Promise<void> => {
    if (activeModeBusy) return;
    if (!window.omni) {
      const updated = {
        ...brain,
        config: { ...brain.config, idleCognition: enabled },
        updatedAt: new Date().toISOString()
      };
      onBrainChange(updated);
      setActiveModeStatus("Design preview only; no background scheduler is connected.");
      return;
    }
    setActiveModeBusy(true);
    setActiveModeStatus(enabled ? "Selecting this identity for Active Mode…" : "Turning Active Mode off…");
    try {
      const result = await window.omni.brain.setActiveMode(brain.id, enabled);
      await onActiveModeChange(result);
      const switched = result.deactivated.map((item) => item.name).join(", ");
      const message = enabled
        ? switched
          ? `${brain.name} is now active; ${switched} was switched off.`
          : `${brain.name} is now the Active Mode identity.`
        : `${brain.name} will no longer think or learn while Studio is idle.`;
      setActiveModeStatus(message);
      onToast(message);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Active Mode could not be changed.";
      setActiveModeStatus(message);
      onToast(message);
    } finally {
      setActiveModeBusy(false);
    }
  };

  return (
    <div className="content-page resource-settings-page">
      <div className="content-page__title resource-settings-page__title">
        <div>
          <span className="eyebrow-text">DEVICE RESOURCE POLICY</span>
          <h1>Memory &amp; shared storage</h1>
          <p>Change this instance&apos;s active context while one device-wide envelope governs its model, training, media, caches, and agents.</p>
        </div>
        <span
          className={cx("resource-settings-save-state", dirty && "is-dirty")}
          role="status"
          aria-live="polite"
        >
          <Icon name={dirty ? "warning" : "check"} size={14} />
          {dirty ? "Unsaved changes" : "Resource plan saved"}
        </span>
      </div>

      <div className="resource-settings-layout">
        <section className="surface active-mode-card" aria-labelledby="active-mode-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="pulse" size={18} /></span>
            <span>
              <h2 id="active-mode-heading">Active Mode</h2>
              <p>Choose the one identity allowed to think and learn during safe idle windows.</p>
            </span>
          </div>
          <Toggle
            checked={brain.config.idleCognition}
            disabled={activeModeBusy}
            onChange={(enabled) => void setActiveModeEnabled(enabled)}
            label={brain.config.idleCognition ? "Active for this identity" : "Inactive for this identity"}
            description="Enabling this identity atomically switches Active Mode off for every other brain. It does not change Ponder, personality, or foreground replies."
          />
          <p className="active-mode-card__status" role="status" aria-live="polite">
            {activeModeStatus || (brain.config.idleCognition
              ? "Selected · bounded idle cognition may run when no foreground work is active."
              : "Off · this identity remains stored and responds normally when you open it.")}
          </p>
        </section>

        <section className="surface resource-settings-policy" aria-labelledby="ram-policy-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="memory" size={19} /></span>
            <span>
              <h2 id="ram-policy-heading">RAM allocation</h2>
              <p>Auto is the default. Advanced limits are a share of memory left after the adaptive device/OS reserve.</p>
            </span>
          </div>

          <fieldset className="resource-settings-mode-picker">
            <legend>Omni-wide RAM policy</legend>
            <button
              type="button"
              className={systemRamMode === "auto" ? "is-active" : ""}
              aria-pressed={systemRamMode === "auto"}
              onClick={() => chooseSystemRamMode("auto")}
            >
              <Icon name="sparkles" size={16} />
              <span>
                <strong>Auto</strong>
                <small>Measures the device and adapts without pinning an old percentage.</small>
              </span>
            </button>
            <button
              type="button"
              className={systemRamMode === "manual" ? "is-active" : ""}
              aria-pressed={systemRamMode === "manual"}
              onClick={() => chooseSystemRamMode("manual")}
            >
              <Icon name="settings" size={16} />
              <span>
                <strong>Advanced cap · 30–100%</strong>
                <small>Pin Omni to a custom portion of the physical safe pool.</small>
              </span>
            </button>
          </fieldset>

          {systemRamMode === "manual" ? (
            <div className="resource-settings-custom-cap">
              <label htmlFor="settings-system-ram-percent">
                <span>
                  <strong>Omni share of safe pool</strong>
                  <small id="settings-system-ram-help">Minimum 30% for stable operation; maximum 100% of physical RAM after the adaptive OS reserve.</small>
                </span>
                <span className="resource-settings-custom-cap__value">
                  <input
                    id="settings-system-ram-percent"
                    type="number"
                    min={MIN_SYSTEM_RAM_SHARE_PERCENT}
                    max={MAX_SYSTEM_RAM_SHARE_PERCENT}
                    step={1}
                    value={systemRamSharePercent}
                    onChange={(event) => {
                      const next = Number.parseInt(event.target.value, 10);
                      if (Number.isFinite(next)) setSystemRamSharePercent(next);
                    }}
                    onBlur={() => setSystemRamSharePercent(
                      clampSystemRamSharePercent(systemRamSharePercent)
                    )}
                    aria-describedby="settings-system-ram-help"
                    aria-invalid={planBlocked || undefined}
                  />
                  <span aria-hidden="true">%</span>
                </span>
              </label>
              <input
                type="range"
                min={MIN_SYSTEM_RAM_SHARE_PERCENT}
                max={MAX_SYSTEM_RAM_SHARE_PERCENT}
                step={1}
                value={clampSystemRamSharePercent(systemRamSharePercent)}
                onChange={(event) => setSystemRamSharePercent(
                  clampSystemRamSharePercent(Number(event.target.value))
                )}
                aria-label="Advanced Omni RAM cap percentage"
                aria-describedby="settings-system-ram-help"
              />
              <div className="resource-settings-custom-cap__range" aria-hidden="true">
                <span>30% stability floor</span>
                <span>100% safe-pool maximum</span>
              </div>
            </div>
          ) : null}

          <div className="resource-settings-subsection">
            <div className="resource-settings-section-heading">
              <span className="resource-settings-section-icon"><Icon name="brain" size={18} /></span>
              <span>
                <h2>Active context</h2>
                <p>This is the live token window used during a turn. It remains in RAM; older neural activity can move through recurrent memory and storage-backed cold state.</p>
              </span>
            </div>
            <fieldset className="resource-settings-mode-picker resource-settings-mode-picker--three">
              <legend>Active-context policy</legend>
              {(
                [
                  ["auto", "Auto", "Measured speed and capability sweet spot", "sparkles"],
                  ["extended", "Extended", "Larger live window with less training headroom", "memory"],
                  ["manual", "Manual", "Choose up to the live physical/model maximum", "settings"]
                ] as const
              ).map(([mode, label, copy, icon]) => (
                <button
                  type="button"
                  key={mode}
                  className={workingMemoryMode === mode ? "is-active" : ""}
                  aria-pressed={workingMemoryMode === mode}
                  onClick={() => chooseWorkingMemoryMode(mode)}
                >
                  <Icon name={icon} size={16} />
                  <span><strong>{label}</strong><small>{copy}</small></span>
                </button>
              ))}
            </fieldset>
            {workingMemoryMode === "manual" ? (
              <div className="resource-settings-custom-cap">
                <label htmlFor="settings-context-tokens">
                  <span>
                    <strong>Active context tokens</strong>
                    <small id="settings-context-help">The green band is measured for this model and device. Typed values receive the same physical preflight.</small>
                  </span>
                  <span className="resource-settings-custom-cap__value resource-settings-custom-cap__value--wide">
                    <input
                      id="settings-context-tokens"
                      type="text"
                      inputMode="numeric"
                      value={manualContextTokens}
                      onChange={(event) => setManualContextTokens(
                        event.target.value.replace(/[^0-9]/g, "")
                      )}
                      aria-label="Active context tokens"
                      aria-describedby="settings-context-help"
                      aria-invalid={planBlocked || undefined}
                    />
                    <span aria-hidden="true">tokens</span>
                  </span>
                </label>
                <input
                  className="memory-capacity-slider"
                  type="range"
                  min={contextSliderMinimumTokens}
                  max={contextMaximumTokens}
                  step={256}
                  value={Math.max(
                    contextSliderMinimumTokens,
                    Math.min(
                      contextMaximumTokens,
                      Number.parseInt(manualContextTokens, 10) || 8
                    )
                  )}
                  disabled={contextFloorTokens > contextMaximumTokens}
                  onChange={(event) => setManualContextTokens(event.target.value)}
                  aria-label="Resident active-context tokens"
                  aria-describedby="settings-context-help"
                  style={resourcePlan ? contextCapacityBandStyle(resourcePlan) as React.CSSProperties : undefined}
                />
                <div className="resource-settings-custom-cap__range">
                  <span>Baseline {resourcePlan?.context.floorTokens.toLocaleString() ?? "measuring"}</span>
                  <span>Auto {resourcePlan?.context.autoTokens.toLocaleString() ?? "measuring"}</span>
                  <span>Maximum {resourcePlan?.context.maximumTokens.toLocaleString() ?? "measuring"}</span>
                </div>
              </div>
            ) : null}
          </div>

          <div className="resource-settings-subsection">
            <div className="resource-settings-section-heading">
              <span className="resource-settings-section-icon"><Icon name="archive" size={18} /></span>
              <span>
                <h2>Shared storage pool</h2>
                <p>One reusable pool covers the largest active instance&apos;s model offload, training scratch, cold neural pages, checkpoints, and growth—not one reservation per duplicate.</p>
              </span>
            </div>
            <fieldset className="resource-settings-mode-picker">
              <legend>Shared-storage policy</legend>
              <button
                type="button"
                className={storagePoolMode === "auto" ? "is-active" : ""}
                aria-pressed={storagePoolMode === "auto"}
                onClick={() => chooseStoragePoolMode("auto")}
              >
                <Icon name="sparkles" size={16} />
                <span><strong>Auto</strong><small>Keeps the largest required reservation and grows it when needed.</small></span>
              </button>
              <button
                type="button"
                className={storagePoolMode === "manual" ? "is-active" : ""}
                aria-pressed={storagePoolMode === "manual"}
                onClick={() => chooseStoragePoolMode("manual")}
              >
                <Icon name="settings" size={16} />
                <span><strong>Custom size</strong><small>Choose a physical ceiling while preserving the free-space reserve.</small></span>
              </button>
            </fieldset>
            {storagePoolMode === "manual" ? (
              <div className="resource-settings-custom-cap">
                <label htmlFor="settings-storage-pool-gib">
                  <span>
                    <strong>Shared pool capacity</strong>
                    <small id="settings-storage-pool-help">Must cover current model, learning scratch, checkpoint headroom, and future neural growth.</small>
                  </span>
                  <span className="resource-settings-custom-cap__value">
                    <input
                      id="settings-storage-pool-gib"
                      type="text"
                      inputMode="numeric"
                      value={manualStoragePoolGiB}
                      onChange={(event) => setManualStoragePoolGiB(
                        event.target.value.replace(/[^0-9]/g, "")
                      )}
                      aria-label="Shared pool capacity"
                      aria-describedby="settings-storage-pool-help"
                      aria-invalid={planBlocked || undefined}
                    />
                    <span aria-hidden="true">GiB</span>
                  </span>
                </label>
                <input
                  type="range"
                  min={storageSliderMinimumGiB}
                  max={storageMaximumGiB}
                  step={1}
                  value={Math.max(
                    storageSliderMinimumGiB,
                    Math.min(
                      storageMaximumGiB,
                      Number.parseInt(manualStoragePoolGiB, 10) || 1
                    )
                  )}
                  disabled={storageSliderUnavailable}
                  onChange={(event) => setManualStoragePoolGiB(event.target.value)}
                  aria-label="Shared storage pool in GiB"
                  aria-describedby="settings-storage-pool-help"
                />
                <div className="resource-settings-custom-cap__range">
                  <span>Required {resourcePlan ? formatBytes(resourcePlan.resources.requiredStoragePoolBytes) : "measuring"}</span>
                  <span>Physical maximum {resourcePlan ? formatBytes(resourcePlan.resources.maximumStoragePoolBytes) : "measuring"}</span>
                </div>
              </div>
            ) : null}
          </div>

          <div
            className={cx(
              "resource-settings-preflight",
              resourcePlanPending && "is-pending",
              planBlocked && "is-blocked"
            )}
            role={planBlocked || resourceError ? "alert" : "status"}
            aria-live="polite"
            aria-busy={resourcePlanPending}
          >
            <Icon
              name={resourcePlanPending ? "pulse" : planBlocked || resourceError ? "warning" : "check"}
              size={16}
            />
            <span>
              <strong>
                {resourcePlanPending
                  ? "Measuring this device…"
                  : resourceError
                    ? "Resource preflight unavailable"
                    : planBlocked
                      ? "This custom envelope is unsafe"
                      : resourcePlan
                        ? systemRamMode === "auto"
                          ? `Auto recommends ${resourcePlan.resources.systemRamSharePercent}% of the safe pool`
                          : `${resourcePlan.resources.systemRamSharePercent}% passes the live preflight`
                        : "Packaged runtime required for a live measurement"}
              </strong>
              <small>
                {resourceError ||
                  (resourcePlanPending
                    ? "Running bounded RAM-copy and durable storage-write checks."
                    : resourcePlan?.blockers.join(" ") ||
                      "The saved choice is checked again before every start.")}
              </small>
            </span>
          </div>

          {resourcePlan?.warnings.length ? (
            <div
              className="resource-settings-warnings"
              role={systemRamMode === "manual" ? "alert" : "status"}
              aria-live="polite"
            >
              <strong>{systemRamMode === "manual" ? "Custom cap warnings" : "Device recommendations"}</strong>
              <ul>
                {resourcePlan.warnings.map((warning) => <li key={warning}>{warning}</li>)}
              </ul>
            </div>
          ) : null}

          <div className="resource-settings-actions">
            <Button
              kind="primary"
              icon={saving ? "pulse" : "check"}
              disabled={!dirty || saving || resourcePlanPending || !resourcePlan?.allowed}
              onClick={() => void saveResourceEnvelope()}
            >
              {saving ? "Saving resources…" : "Save memory & storage"}
            </Button>
            <Button disabled={!dirty || saving} onClick={resetResourceEnvelope}>
              Discard change
            </Button>
          </div>
        </section>

        <aside className="resource-settings-sidebar">
          <section className="surface resource-settings-device" aria-labelledby="safe-pool-heading">
            <div className="resource-settings-section-heading">
              <span className="resource-settings-section-icon"><Icon name="activity" size={18} /></span>
              <span>
                <h2 id="safe-pool-heading">Live safe pool</h2>
                <p>{detectedHardware ? `${detectedHardware.platform} · ${detectedHardware.architecture} · ${detectedHardware.recommendedTier}` : "Detecting hardware…"}</p>
              </span>
            </div>
            {resourcePlan ? (
              <dl className="resource-settings-metrics">
                <div>
                  <dt>Physical RAM</dt>
                  <dd>{formatBytes(resourcePlan.resources.totalMemoryBytes)}</dd>
                </div>
                <div>
                  <dt>Free or reclaimable now</dt>
                  <dd>{formatBytes(resourcePlan.resources.availableMemoryBytes)}</dd>
                </div>
                <div>
                  <dt>Adaptive device/OS reserve</dt>
                  <dd>{formatBytes(resourcePlan.resources.ramReserveBytes)}</dd>
                </div>
                <div>
                  <dt>Safe pool after reserve</dt>
                  <dd>{formatBytes(resourcePlan.resources.safeRamPoolBytes)}</dd>
                </div>
                <div>
                  <dt>Saved Omni RAM capacity</dt>
                  <dd>{formatBytes(resourcePlan.resources.systemRamBudgetBytes)}</dd>
                </div>
                <div>
                  <dt>Available inside capacity now</dt>
                  <dd>{formatBytes(resourcePlan.resources.currentOmniAvailableBytes)}</dd>
                </div>
                <div>
                  <dt>Temporarily occupied by other apps</dt>
                  <dd>{formatBytes(resourcePlan.resources.currentOmniShortfallBytes)}</dd>
                </div>
                <div>
                  <dt>Shared storage capacity</dt>
                  <dd>{formatBytes(resourcePlan.resources.sharedStoragePoolBytes)}</dd>
                </div>
                <div>
                  <dt>Storage required now</dt>
                  <dd>{formatBytes(resourcePlan.resources.requiredStoragePoolBytes)}</dd>
                </div>
              </dl>
            ) : (
              <p className="resource-settings-empty">Live values appear after the packaged runtime completes its bounded probe.</p>
            )}
            <p className="resource-settings-note">
              <Icon name="info" size={14} />
              Saved context and cortex capacity are derived from physical RAM and do not shrink when another app opens. Under pressure, Omni waits and retries while asking you to close memory-heavy apps.
            </p>
          </section>
        </aside>

        <section className="surface resource-settings-benchmarks" aria-labelledby="throughput-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="pulse" size={18} /></span>
            <span>
              <h2 id="throughput-heading">Measured throughput</h2>
              <p>Bounded diagnostics guide Auto and warn when a custom choice would perform poorly.</p>
            </span>
          </div>
          {resourcePlan ? (
            <dl className="resource-benchmark-grid resource-benchmark-grid--settings">
              <div>
                <dt>Measured RAM-copy throughput</dt>
                <dd>{formatBytes(resourcePlan.offload.benchmark.memoryBytesPerSecond)}/s</dd>
                <small>Bounded in-memory buffer copy; not a full DRAM bandwidth claim.</small>
              </div>
              <div>
                <dt>Measured durable storage-write throughput</dt>
                <dd>{formatBytes(resourcePlan.offload.benchmark.storageBytesPerSecond)}/s</dd>
                <small>Median sequential write plus sync · {resourcePlan.training.storageClass.replaceAll("-", " ")}</small>
              </div>
              <div>
                <dt>Probe sample</dt>
                <dd>{formatBytes(resourcePlan.offload.benchmark.sampleBytes)}</dd>
                <small>{resourcePlan.offload.benchmark.cacheHit ? "Cached device result" : "Measured in this session"}</small>
              </div>
              <div>
                <dt>Measured at</dt>
                <dd>{new Date(resourcePlan.offload.benchmark.measuredAt).toLocaleString()}</dd>
                <small>Revalidated by the runtime on its bounded schedule.</small>
              </div>
            </dl>
          ) : (
            <p className="resource-settings-empty">Throughput results are not available in this renderer session.</p>
          )}
        </section>

        <section className="surface resource-settings-training" aria-labelledby="adaptive-training-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="database" size={18} /></span>
            <span>
              <h2 id="adaptive-training-heading">Lower-memory training</h2>
              <p>Training changes its working shape, not which source records it visits.</p>
            </span>
          </div>
          {resourcePlan ? (
            <dl className="resource-settings-training-grid">
              <div><dt>Physical batch</dt><dd>{resourcePlan.training.physicalBatchSize}</dd></div>
              <div><dt>Gradient accumulation</dt><dd>{resourcePlan.training.gradientAccumulation}×</dd></div>
              <div><dt>Training window</dt><dd>{resourcePlan.training.windowTokens.toLocaleString()} tokens</dd></div>
              <div><dt>Dataset coverage</dt><dd>{resourcePlan.training.allSourceBytesVisited ? "All source bytes" : "Incomplete"}</dd></div>
            </dl>
          ) : null}
          <p className="resource-settings-storage-boundary">
            <Icon name="archive" size={15} />
            <span><strong>Storage boundary</strong>{STORAGE_BOUNDARY_COPY}</span>
          </p>
        </section>

        <details className="research-diagnostics resource-settings-diagnostics">
          <summary>
            <span><Icon name="pulse" size={16} /> Research diagnostics</span>
            <span>Attention boundary <Icon name="chevron" size={14} /></span>
          </summary>
          <div className="research-diagnostics__grid">
            <span>
              <small>Fresh attention</small>
              <strong>Transient state only</strong>
            </span>
            <span>
              <small>Preserved</small>
              <strong>History + learned neural memory</strong>
            </span>
            <span>
              <small>Next response</small>
              <strong>No automatic prior chat text</strong>
            </span>
          </div>
          <p className="resource-settings-note">
            Start a new attention epoch without deleting visible conversation,
            weights, connections, replay, training receipts, tools, provenance,
            or origin. Earlier messages can re-enter only through an explicit
            brain.history tool action.
          </p>
          <div className="resource-settings-actions">
            <Button
              icon="pulse"
              disabled={freshAttentionBusy || saving}
              onClick={() => void startFreshAttention()}
            >
              {freshAttentionBusy ? "Starting fresh attention…" : "Start fresh attention"}
            </Button>
          </div>
        </details>

        <BrainSnapshotsPanel
          brain={brain}
          onBrainChange={onBrainChange}
          onToast={onToast}
        />

        <section className="surface mobile-companion" aria-labelledby="mobile-companion-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="agents" size={18} /></span>
            <span>
              <h2 id="mobile-companion-heading">Phone companion</h2>
              <p>The Android app opens this same persistent identity, conversation, neural state, tools, and action audit. It does not create a mobile model copy.</p>
            </span>
          </div>

          <div className="mobile-companion__controls">
            <label>
              <input
                type="checkbox"
                checked={mobileLan}
                disabled={mobileStatus?.state === "listening" || mobileBusy}
                onChange={(event) => setMobileLan(event.target.checked)}
              />
              <span>
                <strong>Allow phones on this trusted local network</strong>
                <small>Off keeps HTTP on loopback and the Android emulator only. LAN pairing uses TLS pinned to this Studio installation plus the saved bearer token.</small>
              </span>
            </label>
            <div className="resource-settings-actions">
              <Button
                kind="primary"
                icon={mobileBusy ? "pulse" : "agents"}
                disabled={mobileBusy}
                onClick={() => void startMobilePairing()}
              >
                {mobileBusy ? "Updating…" : mobileStatus?.state === "listening" ? "New pairing code" : "Pair Android app"}
              </Button>
              {mobileStatus?.state === "listening" ? (
                <Button disabled={mobileBusy} onClick={() => void stopMobileGateway()}>
                  Stop mobile access
                </Button>
              ) : null}
            </div>
          </div>

          {mobilePairing ? (
            <div className="mobile-pairing-card" role="status" aria-live="polite">
              <div>
                <small>ONE-TIME CODE</small>
                <strong>{mobilePairing.code}</strong>
                <span>Expires {new Date(mobilePairing.expiresAt).toLocaleTimeString()}</span>
              </div>
              <div className="mobile-pairing-card__addresses">
                <small>ENTER ONE ADDRESS IN THE APP</small>
                {mobilePairing.baseUrls.map((url) => (
                  <button
                    type="button"
                    key={url}
                    title="Copy companion address"
                    onClick={() => void navigator.clipboard.writeText(url).then(
                      () => onToast("Mobile companion address copied."),
                      () => onToast("Copy was unavailable; select the address manually.")
                    )}
                  >
                    <code>{url}</code><Icon name="copy" size={13} />
                  </button>
                ))}
                {mobilePairing.certificateSha256 ? (
                  <small>
                    TLS SHA-256 · {mobilePairing.certificateSha256.match(/.{1,8}/gu)?.join(" ")}
                  </small>
                ) : (
                  <small>HTTP is restricted to loopback/emulator transport.</small>
                )}
              </div>
            </div>
          ) : null}

          {mobileError ? <p className="mobile-companion__error" role="alert">{mobileError}</p> : null}

          {mobileStatus?.devices.length ? (
            <div className="mobile-device-list">
              <strong>Paired devices</strong>
              {mobileStatus.devices.map((device) => (
                <div key={device.id}>
                  <span>
                    <strong>{device.name}</strong>
                    <small>Last used {new Date(device.lastSeenAt).toLocaleString()}</small>
                  </span>
                  <button
                    type="button"
                    disabled={mobileBusy}
                    onClick={() => void revokeMobileDevice(device.id)}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p className="resource-settings-empty">No phone is paired. Pairing does not export weights or credentials.</p>
          )}
        </section>
      </div>
    </div>
  );
}

function TrainingTelemetryMetrics({
  telemetry
}: {
  telemetry: TrainingTelemetryPresentation;
}) {
  return (
    <div className="training-metrics training-metrics--telemetry" aria-label="Measured training telemetry">
      <span>
        <small>Stage</small>
        <strong>{telemetry.stageLabel}</strong>
        <em>{telemetry.throughput.length > 0 ? telemetry.throughput.join(" · ") : "Measuring throughput…"}</em>
      </span>
      <span>
        <small>Timing</small>
        <strong>{telemetry.timing.join(" · ")}</strong>
        <em>Measured from correlated progress</em>
      </span>
      <span>
        <small>Physical memory</small>
        <strong>{telemetry.memory}</strong>
        <em>Worker-reported only</em>
      </span>
      <span>
        <small>Disk space left</small>
        <strong>{telemetry.storage}</strong>
        <em>Live worker reading · mandatory reserve enforced</em>
      </span>
    </div>
  );
}

function DataWorkspace({
  brain,
  foregroundTurnActive,
  onBack,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  foregroundTurnActive: boolean;
  onBack: () => void;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [mode, setMode] = useState<"uploads" | "catalog" | "web">("uploads");
  const learningPolicy = "pretrain" as const;
  const [demoTraining, setDemoTraining] = useState(false);
  const [demoProgress, setDemoProgress] = useState(0);
  const [localIngestStatus, setLocalIngestStatus] = useState("");
  const [activeJob, setActiveJob] = useState<RuntimeJob | null>(null);
  const [resumableManifest, setResumableManifest] =
    useState<DatasetResumeCandidate | null>(null);
  const [activePreview, setActivePreview] = useState<DatasetPreviewProgress | null>(null);
  const [substrateTotals, setSubstrateTotals] = useState<SubstratePage["totals"] | null>(null);
  const [provenanceRuntimeCard, setProvenanceRuntimeCard] =
    useState<Record<string, unknown> | null>(null);
  const [parameterAccounting, setParameterAccounting] =
    useState<NeuralParameterAccountingPresentation | null>(null);
  const [persistedCoverage, setPersistedCoverage] =
    useState<PersistedDatasetCoverageSummary | null>(null);
  const [catalogEntries, setCatalogEntries] = useState<CatalogEntry[]>(
    window.omni
      ? []
      : [
          { id: "demo-1", name: "FineWeb-Edu recipe", description: "Curated language learning manifest", sourceUrl: "https://example.com/demo", license: "ODC-By", kind: "dataset" },
          { id: "demo-2", name: "Audio concepts pack", description: "Demo modality recipe", sourceUrl: "https://example.com/demo", license: "CC BY 4.0", kind: "modality-pack" }
        ]
  );
  const [crawlUrl, setCrawlUrl] = useState("");
  const crawlUrlAllowed = webLearningUrlAllowed(crawlUrl);
  const [respectRobots, setRespectRobots] = useState(true);
  const [followExternalLinks, setFollowExternalLinks] = useState(false);
  const [quarantine, setQuarantine] = useState(true);
  const [dragging, setDragging] = useState(false);
  const [displayedSources, setDisplayedSources] = useState(brain.trainingSources);
  const [sourceCursor, setSourceCursor] = useState<string | undefined>();
  const [sourcePageDepth, setSourcePageDepth] = useState(0);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const announcedJobs = useRef(new Set<string>());
  const previewRequestId = useRef<string | null>(null);

  const loadSourcePage = useCallback(async (cursor?: string, depth = 0) => {
    if (!window.omni) {
      setDisplayedSources(brain.trainingSources.slice(-60));
      setSourceCursor(undefined);
      setSourcePageDepth(0);
      return;
    }
    setSourcesLoading(true);
    try {
      const page = await window.omni.data.sources(brain.id, cursor, 60);
      setDisplayedSources(page.entries.map((entry) => entry.source));
      setSourceCursor(page.nextCursor);
      setSourcePageDepth(depth);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The source ledger could not be loaded.");
    } finally {
      setSourcesLoading(false);
    }
  }, [brain.id, brain.trainingSources, onToast]);

  useEffect(() => {
    void loadSourcePage(undefined, 0);
  }, [loadSourcePage, brain.activity?.trainingSourceHeadSha256]);

  const activeJobStop = runtimeJobStopPresentation(activeJob);
  const activeJobProgress = runtimeJobProgressPresentation(activeJob);
  const busy = demoTraining || activeJobStop.active;
  const previewFileProgress = activePreview?.currentFileBytes
    ? Math.round(
        Math.max(0, Math.min(1, (activePreview.currentFileHashedBytes ?? 0) / activePreview.currentFileBytes)) * 100
      )
    : null;
  const progress = demoTraining
    ? previewFileProgress ?? demoProgress
    : activeJob
      ? activeJobProgress.percent
      : persistedCoverage?.complete
        ? 100
        : 0;
  const jobOutput = valueRecord(activeJob?.output);
  const telemetryPresentation = activeJob?.telemetry && activeJobProgress.showTelemetry
    ? presentTrainingTelemetry(activeJob.telemetry)
    : undefined;
  const coverageValue = activeJobProgress.showCoverage
    ? valueRecord(jobOutput?.coverage)
    : undefined;
  const activeCoverage =
    coverageValue &&
    typeof coverageValue.discoveredRecords === "number" &&
    typeof coverageValue.processedRecords === "number" &&
    typeof coverageValue.rejectedRecords === "number" &&
    typeof coverageValue.discoveredFiles === "number" &&
    typeof coverageValue.processedFiles === "number" &&
    typeof coverageValue.rejectedFiles === "number"
      ? {
          discoveredRecords: coverageValue.discoveredRecords,
          processedRecords: coverageValue.processedRecords,
          rejectedRecords: coverageValue.rejectedRecords,
          discoveredFiles: coverageValue.discoveredFiles,
          processedFiles: coverageValue.processedFiles,
          rejectedFiles: coverageValue.rejectedFiles,
          complete: coverageValue.complete === true
        }
      : null;
  const displayedCoverage = runtimeJobScopedDatasetCoverage(
    activeJob,
    activeCoverage,
    !busy ? persistedCoverage : null
  );
  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    setPersistedCoverage(null);
    setResumableManifest(null);
    setSubstrateTotals(null);
    setParameterAccounting(null);
    setProvenanceRuntimeCard(null);
    void window.omni.brain.persistedSubstrateOverview(brain.id).then(
      (overview) => {
        if (!active || !overview) return;
        setSubstrateTotals(overview.totals);
        setPersistedCoverage(overview.latestCompletedCoverage ?? null);
        const persistedAccounting = overview.parameterAccounting
          ? neuralParameterAccounting({
              parameterAccounting: overview.parameterAccounting
            })
          : undefined;
        if (persistedAccounting) {
          setParameterAccounting(persistedAccounting);
        }
      },
      () => undefined
    );
    void Promise.all([
      window.omni.catalog.list(),
      window.omni.train.list(brain.id),
      window.omni.data.resumable(brain.id)
    ]).then(
      ([entries, jobs, resumable]) => {
        if (!active) return;
        setCatalogEntries(entries);
        setResumableManifest(resumable);
        setActiveJob(
          jobs
            .filter((job) => job.brainId === brain.id)
            .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))[0] ?? null
        );
      }
    );
    const unsubscribe = window.omni.train.onEvent(({ job }) => {
      if (job.brainId !== brain.id) return;
      setActiveJob(job);
      if (job.state === "complete") {
        void window.omni?.brain.get(brain.id).then(onBrainChange);
        void window.omni?.data.resumable(brain.id)
          .then(setResumableManifest)
          .catch(() => undefined);
        void window.omni?.brain
          .persistedSubstrateOverview(brain.id)
          .then((overview) => {
            if (!overview) return;
            setSubstrateTotals(overview.totals);
            setPersistedCoverage(overview.latestCompletedCoverage ?? null);
            const accounting = overview.parameterAccounting
              ? neuralParameterAccounting({
                  parameterAccounting: overview.parameterAccounting
                })
              : undefined;
            if (accounting) setParameterAccounting(accounting);
          });
      }
      if (
        (job.state === "complete" || job.state === "failed") &&
        !announcedJobs.current.has(job.id)
      ) {
        announcedJobs.current.add(job.id);
        onToast(
          job.state === "complete"
            ? `${job.label} completed.`
            : `${job.label} failed${job.error ? `: ${job.error}` : "."}`
        );
      }
      if (job.kind === "ingestion" && ["cancelled", "failed"].includes(job.state)) {
        void window.omni?.data.resumable(brain.id)
          .then(setResumableManifest)
          .catch(() => undefined);
      }
    });
    const unsubscribePreview = window.omni.data.onPreviewProgress((event) => {
      if (
        event.brainId !== brain.id ||
        event.requestId !== previewRequestId.current
      ) return;
      setActivePreview(event);
      setLocalIngestStatus(event.message);
    });
    return () => {
      active = false;
      unsubscribe();
      unsubscribePreview();
    };
  }, [brain.id, onBrainChange]);

  const applyResults = (results: Awaited<ReturnType<NonNullable<typeof window.omni>["data"]["ingestFiles"]>>) => {
    if (results.length > 0) {
      onBrainChange(results.at(-1)!.brain);
      onToast(`${results.length} source${results.length === 1 ? "" : "s"} encoded into ${brain.name}.`);
    }
  };

  const ingest = async (
    kind: "files" | "folder" = "files",
    selection: ExperienceUploadKind = "files"
  ) => {
    if (demoTraining) return;
    setDemoTraining(true);
    setDemoProgress(4);
    setLocalIngestStatus(
      kind === "folder"
        ? "Choosing a whole folder"
        : `Choosing ${EXPERIENCE_UPLOADS[selection].shortLabel}`
    );
    try {
      if (window.omni) {
        const requestId = globalThis.crypto?.randomUUID?.() ??
          `dataset-preview-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        previewRequestId.current = requestId;
        setActivePreview(null);
        const manifest = await window.omni.data.preview({
          brainId: brain.id,
          policy: learningPolicy,
          requestId,
          selection: kind === "folder" ? "folder" : selection
        });
        if (!manifest) {
          previewRequestId.current = null;
          setActivePreview(null);
          setLocalIngestStatus("Selection cancelled");
          return;
        }
        previewRequestId.current = null;
        setActivePreview(null);
        setDemoProgress(10);
        setLocalIngestStatus(
          `Committed ${manifest.discoveredFiles} source${manifest.discoveredFiles === 1 ? "" : "s"} to a resumable traversal manifest`
        );
        const job = await window.omni.data.start({
          brainId: brain.id,
          manifestId: manifest.id,
          policy: learningPolicy,
          epochs: 1,
          resume: true
        });
        setActiveJob(job);
        onToast(
          `${manifest.discoveredFiles} source${manifest.discoveredFiles === 1 ? "" : "s"} queued; live neural-learning progress is shown here.`
        );
      } else {
        for (const value of [18, 36, 57, 76, 100]) {
          await new Promise((resolve) => window.setTimeout(resolve, 230));
          setDemoProgress(value);
        }
        onToast("Demo source encoded: 42 ideas and 188 synaptic changes.");
      }
    } catch (error) {
      previewRequestId.current = null;
      onToast(error instanceof Error ? error.message : "Could not ingest the selected files.");
    } finally {
      setDemoTraining(false);
    }
  };

  const ingestDrop = async (files: File[]) => {
    setDragging(false);
    if (!files.length || busy) return;
    if (!dataActionAvailableDuringTurn("immediate-write", foregroundTurnActive)) {
      onToast(
        "Dropped-file learning is waiting for the current reply. Start a queued upload or return to Conversation to Queue or Steer."
      );
      return;
    }
    if (!window.omni) {
      onToast(`${files.length} dropped file${files.length === 1 ? "" : "s"} recognized in demo preview; no learning ran.`);
      return;
    }
    setDemoTraining(true);
    setDemoProgress(6);
    setLocalIngestStatus(
      `Encoding ${files.length} dropped file${files.length === 1 ? "" : "s"} into neural state`
    );
    try {
      const results = await window.omni.data.ingestDropped(
        { brainId: brain.id, policy: learningPolicy },
        files
      );
      setDemoProgress(100);
      applyResults(results);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Dropped files could not be ingested.");
    } finally {
      setDemoTraining(false);
    }
  };

  const startCrawl = async () => {
    if (!crawlUrlAllowed || demoTraining) return;
    if (!window.omni) {
      onToast("Web crawling is disabled in the browser design preview.");
      return;
    }
    setDemoTraining(true);
    setLocalIngestStatus("Adding the crawl to the shared learning queue");
    try {
      const job = await window.omni.data.crawlWeb({
        brainId: brain.id,
        url: crawlUrl,
        policy: learningPolicy,
        quarantine,
        respectRobots,
        sameOrigin: !followExternalLinks,
        followExternalLinks
      });
      setActiveJob(job);
      onToast(`${quarantine ? "Quarantined " : ""}continuous crawl started. It will run until stopped or resources pause it.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The web crawl could not start.");
    } finally {
      setDemoTraining(false);
    }
  };

  const ingestPage = async () => {
    if (!crawlUrlAllowed || busy) return;
    if (!dataActionAvailableDuringTurn("immediate-write", foregroundTurnActive)) {
      onToast(
        "Single-page learning is waiting for the current reply. Queue a crawl, or return to Conversation to Queue or Steer."
      );
      return;
    }
    if (!window.omni) {
      onToast("Single-page web ingestion is disabled in the browser design preview.");
      return;
    }
    try {
      const result = await window.omni.data.ingestWeb({
        brainId: brain.id,
        url: crawlUrl,
        policy: learningPolicy,
        quarantine
      });
      onBrainChange(result.brain);
      onToast(`Encoded ${result.source.name} with retained provenance.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The web page could not be ingested.");
    }
  };

  const cancelActiveJob = async () => {
    if (!window.omni) return;
    if (previewRequestId.current) {
      const cancelled = await window.omni.data.cancelPreview(previewRequestId.current);
      if (cancelled) {
        setLocalIngestStatus("Cancelling discovery and hashing; removing the partial manifest…");
        onToast("Dataset preview cancellation requested.");
      }
      return;
    }
    if (!activeJob || !activeJobStop.requestAllowed) return;
    const requestedJob = activeJob;
    setActiveJob({
      ...requestedJob,
      state: "cancelling",
      label: `Cancelling ${requestedJob.kind} · waiting for acknowledgement`,
      error: undefined,
      updatedAt: new Date().toISOString()
    });
    try {
      const cancelled = requestedJob.kind === "crawl" || requestedJob.kind === "ingestion"
        ? await window.omni.data.cancel(requestedJob.id)
        : await window.omni.train.cancel(requestedJob.id);
      setActiveJob(cancelled);
      onToast(
        requestedJob.kind === "crawl"
          ? "The crawler stopped after acknowledgement; its frontier is saved."
          : "The neural worker acknowledged cancellation."
      );
    } catch (error) {
      const reason = conciseUiMessage(
        error instanceof Error ? error.message : String(error),
        "Cancellation is still awaiting acknowledgement."
      );
      const current = await window.omni.train
        .list(brain.id)
        .then((jobs) => jobs.find((candidate) => candidate.id === requestedJob.id))
        .catch(() => undefined);
      setActiveJob(current ?? {
        ...requestedJob,
        state: "cancelling",
        label: `${requestedJob.kind} stop needs acknowledgement`,
        error: reason,
        updatedAt: new Date().toISOString()
      });
      onToast(`${reason} Retry stop when ready.`);
    }
  };

  const pauseActiveJob = async () => {
    if (
      !window.omni ||
      !activeJob ||
      !["ingestion", "crawl"].includes(activeJob.kind) ||
      !["queued", "running"].includes(activeJob.state)
    ) {
      return;
    }
    const paused = await window.omni.data.pause(activeJob.id);
    setActiveJob(paused);
    onToast(`${activeJob.label} paused at its durable cursor and can resume later.`);
  };

  const resumePersistedManifest = async (): Promise<void> => {
    if (!window.omni || !resumableManifest || activeJobStop.active) return;
    setLocalIngestStatus("Resuming the persisted dataset cursor");
    try {
      const job = await window.omni.data.resume({
        brainId: brain.id,
        manifestId: resumableManifest.manifestId,
        policy: learningPolicy,
        epochs: resumableManifest.requestedEpochs,
        resume: true
      });
      setResumableManifest(null);
      setActiveJob(job);
      onToast("Dataset learning resumed from its durable cursor.");
    } catch (error) {
      onToast(
        error instanceof Error
          ? error.message
          : "The persisted dataset could not resume."
      );
    }
  };
  const dataProvenance = brainProvenancePresentation({
    provenance: brain.provenance,
    runtimeCard: provenanceRuntimeCard,
    trainingSources: displayedSources
  });
  const sourceCount = brain.activity?.trainingSourceCount ?? brain.trainingSources.length;
  const learnedConceptTotal = brain.activity?.learnedConcepts ??
    brain.trainingSources.reduce((sum, source) => sum + source.learnedConcepts, 0);
  const learnedSynapseTotal = brain.activity?.learnedSynapses ??
    brain.trainingSources.reduce((sum, source) => sum + source.learnedSynapses, 0);

  return (
    <div className="content-page data-page">
      <div className="content-page__title">
        <div>
          <button
            className="content-page__back"
            aria-label="Back to conversation"
            onClick={onBack}
          >
            <Icon name="arrow" size={13} className="map-arrow-back" /> Conversation
          </button>
          <span className="eyebrow-text">CONTINUOUS EXPERIENCE LEARNING</span>
          <h1>Data & training</h1>
          <p>Each document, dataset, code sample, sound, image, video, or page enters the same continuous whole-experience learning flow.</p>
        </div>
        <Button icon="upload" kind="primary" onClick={() => void ingest("files")}>
          Add experience
        </Button>
      </div>
      {dataProvenance ? (
        <div
          className="data-provenance-note"
          role="note"
          title={dataProvenance.ariaLabel}
          aria-label={dataProvenance.ariaLabel}
        >
          <strong>{dataProvenance.originLabel}</strong>
          <span>{dataProvenance.compactLabel}</span>
        </div>
      ) : null}
      {foregroundTurnActive ? (
        <div className="data-foreground-queue-note" role="status" aria-live="polite">
          <Icon name="chat" size={15} />
          <span>
            <strong>Current reply continues in Conversation</strong>
            <small>Uploads, training, and crawls started here join the visible queue. Immediate learning waits; use Queue or Steer in Conversation.</small>
          </span>
        </div>
      ) : null}
      {resumableManifest && !activeJobStop.active ? (
        <div className="dataset-resume-note" role="status" aria-live="polite">
          <Icon name="database" size={16} />
          <span>
            <strong>
              {resumableManifest.state === "interrupted"
                ? "Interrupted dataset is ready to resume"
                : "Paused dataset is ready to resume"}
            </strong>
            <small>
              {resumableManifest.processedFiles.toLocaleString()} of {(resumableManifest.discoveredFiles * resumableManifest.requestedEpochs).toLocaleString()} file passes settled · manifest {resumableManifest.manifestId.slice(0, 8)}…
            </small>
          </span>
          <Button kind="primary" icon="play" onClick={() => void resumePersistedManifest()}>
            Resume dataset
          </Button>
        </div>
      ) : null}
      <div className="data-layout">
        <section>
          <div className="surface ingest-surface">
            <div className="surface-tabs">
              {(
                [
                  ["uploads", "Uploads", "upload"],
                  ["catalog", "Dataset catalog", "library"],
                  ["web", "Web crawler", "search"]
                ] as const
              ).map(([id, label, icon]) => (
                <button key={id} className={mode === id ? "is-active" : ""} onClick={() => setMode(id)}>
                  <Icon name={icon} size={15} /> {label}
                </button>
              ))}
            </div>
            {mode === "uploads" ? (
              <div className="upload-panel">
                <div
                  className={cx("drop-zone", dragging && "is-dragging")}
                  role="button"
                  tabIndex={0}
                  onClick={() => void ingest("files")}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") void ingest("files");
                  }}
                  onDragEnter={(event) => {
                    event.preventDefault();
                    setDragging(true);
                  }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={(event) => {
                    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
                  }}
                  onDrop={(event) => {
                    event.preventDefault();
                    void ingestDrop(Array.from(event.dataTransfer.files));
                  }}
                >
                  <span className="drop-zone__rings">
                    <Icon name="upload" size={25} />
                  </span>
                  <strong>Drop files or folders here</strong>
                  <p>Documents, datasets, code, images, audio, and video are traversed and encoded; rejected records stay visible.</p>
                  <span>Browse all supported material</span>
                </div>
                <div className="upload-kind-actions" aria-label="Upload by experience type">
                  <Button icon="file" disabled={demoTraining} onClick={() => void ingest("files", "files")}>
                    Files & datasets
                  </Button>
                  <Button icon="image" disabled={demoTraining} onClick={() => void ingest("files", "images")}>
                    Images
                  </Button>
                  <Button icon="volume" disabled={demoTraining} onClick={() => void ingest("files", "audio")}>
                    Audio
                  </Button>
                  <Button icon="video" disabled={demoTraining} onClick={() => void ingest("files", "video")}>
                    Video
                  </Button>
                  <Button icon="archive" disabled={demoTraining} onClick={() => void ingest("folder")}>
                    Whole folder
                  </Button>
                </div>
              </div>
            ) : mode === "catalog" ? (
              <div className="catalog-list">
                <p className="catalog-list__status">
                  Catalog entries are research and data references. New minds always initialize the native OmniCortex; a listing cannot change its architecture, origin, behavior, or tool permissions.
                </p>
                {catalogEntries.map((entry) => (
                  <div key={entry.id}>
                    <span className="catalog-list__icon">
                      <Icon name="database" size={17} />
                    </span>
                    <span>
                      <strong>{entry.name}</strong>
                      <small>{entry.kind} · {entry.description}</small>
                    </span>
                    <em>{entry.license}</em>
                    <Button onClick={() => void (window.omni?.window.openExternal(entry.sourceUrl))}>
                      Inspect source
                    </Button>
                  </div>
                ))}
                {!catalogEntries.length ? <div className="table-empty">No verified catalog sources are configured.</div> : null}
              </div>
            ) : (
              <div className="crawler-form">
                <label>
                  <span>Starting URL</span>
                  <div>
                    <Icon name="search" size={16} />
                    <input value={crawlUrl} onChange={(event) => setCrawlUrl(event.target.value)} placeholder="https://docs.example.com or http://127.0.0.1:8000" />
                  </div>
                  <small>Remote sources require HTTPS. HTTP is accepted only for verified loopback hosts.</small>
                </label>
                <div className="crawler-options">
                  <Toggle checked={respectRobots} onChange={setRespectRobots} label="Respect robots.txt" />
                  <Toggle checked={followExternalLinks} onChange={setFollowExternalLinks} label="Follow external links" />
                  <Toggle checked={quarantine} onChange={setQuarantine} label="Quarantine before learning" />
                </div>
                {!respectRobots ? (
                  <div className="crawler-warning">
                    <Icon name="warning" size={15} />
                    Disabling robots compliance may violate site terms or overload a server. Pacing and private-network protections still apply.
                  </div>
                ) : null}
                <div className="crawler-actions">
                  <Button
                    icon="download"
                    disabled={!crawlUrlAllowed || busy || foregroundTurnActive}
                    title={foregroundTurnActive
                      ? "Waiting for the current reply; use Queue or Steer in Conversation"
                      : "Learn this page now"}
                    onClick={() => void ingestPage()}
                  >
                    Learn this page
                  </Button>
                  <Button kind="primary" icon="play" disabled={!crawlUrlAllowed || demoTraining} onClick={() => void startCrawl()}>
                    Crawl until stopped
                  </Button>
                </div>
              </div>
            )}
            <div className="ingest-policy">
              <span>
                <strong>Learn each whole experience</strong>
                <small>Every valid record is visited. Timing, novelty, importance, interference, rehearsal, stability, and decay determine how it changes the same neural memory; provenance stays attached.</small>
              </span>
              <span className="representation-pill">Automatic</span>
            </div>
          </div>

          <div className="surface sources-surface">
            <div className="surface-title">
              <div>
                <h2>Encoded source ledger</h2>
                <p>Provenance and reported neural deltas stay attached to each successfully encoded source.</p>
              </div>
            </div>
            <div className="source-table">
              <div className="source-table__header">
                <span>Source</span>
                <span>Neural change</span>
                <span>Stored as</span>
                <span>Added</span>
                <span />
              </div>
              {displayedSources.map((source) => (
                <div className="source-row" key={source.id}>
                  <span className="source-identity">
                    <i className={`file-kind file-kind--${source.kind}`}>
                      <Icon name={source.kind === "code" ? "code" : "file"} size={17} />
                    </i>
                    <span>
                      <strong>{source.name}</strong>
                      <small>
                        {(source.bytes / 1_048_576).toFixed(source.bytes > 1_048_576 ? 1 : 2)} MB · {source.kind.toUpperCase()} · {source.license ?? "License undeclared"}
                      </small>
                    </span>
                  </span>
                  <span
                    className="source-neural-change"
                    aria-label={
                      `${source.learnedConcepts.toLocaleString("en-US")} concept or neuron changes, ` +
                      `${source.learnedSynapses.toLocaleString("en-US")} synaptic changes, ` +
                      (source.learnedParameterSteps === undefined
                        ? "Optimizer steps were not reported. "
                        : `${source.learnedParameterSteps.toLocaleString("en-US")} optimizer steps. `) +
                      (source.parametersChanged === undefined
                        ? "Composite neural checksum was not reported."
                        : source.parametersChanged
                          ? "Composite neural checksum changed; dense model weight change is not separately verified."
                          : "Composite neural checksum did not change.")
                    }
                  >
                    <strong
                      title={`${source.learnedConcepts.toLocaleString("en-US")} concept or neuron changes`}
                    >
                      {compactParameterCount(source.learnedConcepts)} concepts / neurons
                    </strong>
                    <small title={`${source.learnedSynapses.toLocaleString("en-US")} synaptic changes`}>
                      {compactParameterCount(source.learnedSynapses)} synaptic changes
                    </small>
                    <small
                      title={source.learnedParameterSteps === undefined
                        ? "Optimizer steps were not reported for this source"
                        : `${source.learnedParameterSteps.toLocaleString("en-US")} optimizer steps across this source`}
                    >
                      {source.learnedParameterSteps === undefined
                        ? "Optimizer steps unknown"
                        : `${compactParameterCount(source.learnedParameterSteps)} optimizer steps`} · {source.parametersChanged === undefined
                          ? "composite checksum unknown"
                          : source.parametersChanged
                            ? "composite neural checksum changed"
                            : "no composite checksum change"}
                    </small>
                  </span>
                  <span>
                    <i className="representation-pill">
                      {source.rawTextRetained ? "Neural memory + source" : "Neural memory"}
                    </i>
                  </span>
                  <span>{relativeTime(source.importedAt)}</span>
                  <span />
                </div>
              ))}
              {sourceCount === 0 ? (
                <div className="table-empty">No external experiences have been encoded yet.</div>
              ) : null}
            </div>
            {sourceCount > 0 ? (
              <div className="source-ledger-pager" aria-live="polite">
                <span>
                  Showing {displayedSources.length.toLocaleString("en-US")} of {sourceCount.toLocaleString("en-US")} sources
                  {sourcePageDepth > 0 ? ` · older page ${sourcePageDepth + 1}` : " · newest page"}
                </span>
                {sourcePageDepth > 0 ? (
                  <Button disabled={sourcesLoading} onClick={() => void loadSourcePage(undefined, 0)}>
                    Newest
                  </Button>
                ) : null}
                {sourceCursor ? (
                  <Button
                    disabled={sourcesLoading}
                    onClick={() => void loadSourcePage(sourceCursor, sourcePageDepth + 1)}
                  >
                    Older sources
                  </Button>
                ) : null}
              </div>
            ) : null}
          </div>
        </section>

        <aside className="training-sidebar">
          <div className="surface training-card">
            <div className="training-card__head">
              <span className="training-card__icon">
                <Icon name={busy ? "pulse" : activeJob?.state === "complete" || persistedCoverage ? "check" : "info"} size={20} />
              </span>
              <span>
                <small>{busy ? "ACTIVE JOB" : activeJob ? "LATEST JOB" : persistedCoverage ? "LATEST COMPLETED MANIFEST" : "TRAINING QUEUE"}</small>
                <strong>
                  {demoTraining
                    ? localIngestStatus || "Encoding experience"
                    : busy
                      ? activeJob?.label ?? "Encoding experience"
                      : activeJob?.label ?? (persistedCoverage ? "Completed dataset traversal" : "No jobs yet")}
                </strong>
              </span>
              {busy && (activeJob || activePreview) ? (
                <div className="training-card__controls">
                  {activeJob && ["ingestion", "crawl"].includes(activeJob.kind) && activeJob.state !== "cancelling" ? (
                    <button className="icon-button" onClick={() => void pauseActiveJob()} aria-label="Pause active job">
                      <Icon name="pause" size={15} />
                    </button>
                  ) : null}
                  <button
                    className="icon-button"
                    disabled={Boolean(activeJob && !activeJobStop.requestAllowed)}
                    onClick={() => void cancelActiveJob()}
                    aria-label={
                      activeJobStop.waitingForAcknowledgement
                        ? "Waiting for active job stop acknowledgement"
                        : activeJobStop.retryAvailable
                          ? "Retry stopping active job"
                          : "Cancel active job"
                    }
                    title={activeJob ? activeJobStop.label : "Cancel active job"}
                  >
                    <Icon name={activeJobStop.waitingForAcknowledgement ? "pulse" : "close"} size={15} />
                  </button>
                </div>
              ) : null}
            </div>
            <div className={cx("training-progress", progress === null && "is-indeterminate")}>
              <span>
                <strong>{progress === null ? "—" : `${progress}%`}</strong>
                <em>
                  {demoTraining
                    ? "processing"
                    : busy
                      ? activeJob?.state ?? "processing"
                      : activeJob?.state ?? (persistedCoverage ? "complete" : "idle")}
                </em>
              </span>
              <i>
                <b style={progress === null ? undefined : { width: `${progress}%` }} />
              </i>
            </div>
            {telemetryPresentation ? (
              <TrainingTelemetryMetrics telemetry={telemetryPresentation} />
            ) : null}
            <div className="training-metrics">
              <span>
                <small>Manifest coverage</small>
                <strong>
                  {displayedCoverage
                    ? `${displayedCoverage.processedRecords + displayedCoverage.rejectedRecords} / ${displayedCoverage.discoveredRecords}`
                    : activeJob && (activeJob.kind === "ingestion" || activeJob.kind === "crawl")
                      ? progress === null ? "—" : `${progress}%`
                      : "—"}
                </strong>
                <em>
                  {displayedCoverage
                    ? `${displayedCoverage.processedRecords} encoded · ${displayedCoverage.rejectedRecords} rejected · ${displayedCoverage.processedFiles + displayedCoverage.rejectedFiles}/${displayedCoverage.discoveredFiles} file visits${displayedCoverage.complete ? " · complete" : " · incomplete/resumable"}`
                    : activeJob && (activeJob.kind === "ingestion" || activeJob.kind === "crawl")
                      ? progress === null
                        ? activeJobProgress.statusText
                        : "committed traversal in progress"
                      : "no manifest run"}
                </em>
              </span>
              <span>
                <small>Concept / neuron deltas</small>
                <strong
                  title={`${learnedConceptTotal.toLocaleString("en-US")} total concept or neuron changes`}
                >
                  {sourceCount
                    ? compactParameterCount(learnedConceptTotal)
                    : "—"}
                </strong>
                <em>{sourceCount.toLocaleString("en-US")} encoded sources</em>
              </span>
              <span>
                <small>Cumulative learned updates</small>
                <strong
                  title={`${learnedSynapseTotal.toLocaleString("en-US")} cumulative learned connection updates`}
                >
                  {sourceCount
                    ? compactParameterCount(learnedSynapseTotal)
                    : "—"}
                </strong>
                <em>activity total · not unique connections</em>
              </span>
            </div>
            <div className="training-log">
              <Icon name="terminal" size={15} />
              <span className="training-log__message">
                {demoTraining
                  ? localIngestStatus
                  : activeJob?.error
                    ? conciseUiMessage(activeJob.error)
                    : activeJob?.label ?? (persistedCoverage ? "Completed manifest coverage verified from durable progress" : "No runtime log entries yet")}
              </span>
              <span className="training-log__time">{activeJob ? relativeTime(activeJob.updatedAt) : persistedCoverage ? relativeTime(persistedCoverage.updatedAt) : ""}</span>
            </div>
          </div>
          <div className="surface memory-health">
            <div className="surface-title">
              <div>
                <h2>Memory</h2>
                <p>Automatic, continuous learning</p>
              </div>
              <span
                className="memory-connection-count"
                title={substrateTotals
                  ? `${substrateTotals.synapses.toLocaleString("en-US")} unique live connections from validated persisted substrate`
                  : "Loading validated persisted connection total"}
                aria-label={substrateTotals
                  ? `${substrateTotals.synapses.toLocaleString("en-US")} unique live connections`
                  : "Loading unique live connection total"}
              >
                {substrateTotals?.synapses.toLocaleString() ?? "…"}
              </span>
            </div>
            <div className="memory-continuum-totals" aria-label="Current neural substrate totals">
              <span>
                <small>Neural parameters</small>
                {parameterAccounting ? (
                  <NeuralParameterCount accounting={parameterAccounting} />
                ) : (
                  <strong>…</strong>
                )}
              </span>
              <span><small>Neurons</small><strong>{substrateTotals?.neurons.toLocaleString() ?? "…"}</strong></span>
              <span><small>Assemblies</small><strong>{substrateTotals?.assemblies.toLocaleString() ?? "…"}</strong></span>
              <span><small>Unique connections</small><strong>{substrateTotals?.synapses.toLocaleString() ?? "…"}</strong></span>
            </div>
            <p className="memory-health__automatic">
              Each experience can affect immediate activity, related pathways, and longer-lasting
              learning together. Novelty, timing, reuse, importance, prediction, interference,
              rehearsal, stability, and decay continuously decide what strengthens or fades.
            </p>
          </div>
          <div className="surface provenance-card">
            <Icon name="archive" size={18} />
            <span>
              <strong>Research ledger</strong>
              Every architecture, dataset, and derived checkpoint keeps its source and license.
            </span>
            <Icon name="arrow" size={14} />
          </div>
        </aside>
      </div>
    </div>
  );
}

interface SubstrateMapNode {
  id: string;
  label: string;
  region?: string;
  activation: number;
  activationObserved?: boolean;
  importance: number;
  uncertainty: number;
  exposures: number;
  members: string[];
  clusterCount?: number;
  /** Activity can be real while a unique-node topology is not yet readable. */
  activityOnly?: boolean;
  lastActivatedAt?: string;
}

interface SubstrateMapEdge {
  id: string;
  sourceId: string;
  targetId: string;
  effectiveWeight: -1 | 0 | 1;
  /** Aggregate of displayed ternary connections, not a second learned weight. */
  signedStrength: number;
  stability: number;
  pathways: number;
}

function stableMapHash(value: string) {
  let hash = 2_166_136_261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

function BrainMapWorkspace({ brain, onBack }: { brain: BrainDocument; onBack: () => void }) {
  const [domain, setDomain] = useState<"substrate" | "cortex">("substrate");
  return <div>
    <div className="segmented" role="tablist" aria-label="Brain map domain" style={{ margin: "18px 28px 0" }}>
      <button role="tab" aria-selected={domain === "substrate"} className={domain === "substrate" ? "is-active" : ""} onClick={() => setDomain("substrate")}>Sparse neural substrate</button>
      <button role="tab" aria-selected={domain === "cortex"} className={domain === "cortex" ? "is-active" : ""} onClick={() => setDomain("cortex")}>Packed cortical model</button>
    </div>
    {domain === "cortex" ? <BrainMapCortex brainId={brain.id} updatedAt={brain.updatedAt} onBack={onBack} />
      : <SubstrateBrainMapWorkspace brain={brain} onBack={onBack} />}
  </div>;
}

function SubstrateBrainMapWorkspace({
  brain,
  onBack
}: {
  brain: BrainDocument;
  onBack: () => void;
}) {
  const [selected, setSelected] = useState("");
  const [filter, setFilter] = useState<"all" | "active" | "salient">("all");
  const [query, setQuery] = useState("");
  const [zoom, setZoom] = useState(0.72);
  const [offset, setOffset] = useState(0);
  const [region, setRegion] = useState<string | undefined>();
  const [cursor, setCursor] = useState<string | undefined>();
  const [cursorHistory, setCursorHistory] = useState<Array<string | undefined>>([]);
  const [substratePage, setSubstratePage] = useState<SubstratePage | null>(null);
  const [persistedOverview, setPersistedOverview] =
    useState<PersistedSubstrateOverview | null>(null);
  const [substrateLoading, setSubstrateLoading] = useState(false);
  const [pathwayCursor, setPathwayCursor] = useState<string | undefined>();
  const [pathwayCursorHistory, setPathwayCursorHistory] = useState<Array<string | undefined>>([]);
  const [pathwayOffset, setPathwayOffset] = useState(0);
  const [pathwayPage, setPathwayPage] = useState<SubstratePage | null>(null);
  const [pathwayLoading, setPathwayLoading] = useState(false);
  const concepts = useMemo(() => {
    const live = Object.values(brain.concepts);
    return live.length ? live : window.omni ? [] : Object.values(makeDemoBrain().concepts);
  }, [brain.concepts]);
  const filteredConcepts = useMemo(
    () =>
      concepts.filter((concept) => {
        const matchesQuery = concept.label.toLocaleLowerCase().includes(query.toLocaleLowerCase());
        const matchesFilter =
          filter === "all" ||
          (filter === "active" && concept.activation >= 0.5) ||
          (filter === "salient" && concept.importance >= 0.75);
        return matchesQuery && matchesFilter;
      }),
    [concepts, filter, query]
  );
  const clustered = zoom < 1;
  const pageSize = Math.max(48, Math.round(72 * zoom));
  const clampedOffset = Math.max(0, Math.min(offset, Math.max(0, filteredConcepts.length - pageSize)));
  const connectionCounts = connectionCountPresentation({
    queriedConnections:
      substratePage?.brainId === brain.id
        ? substratePage.totals.synapses
        : undefined,
    persistedConnections:
      persistedOverview?.brainId === brain.id
        ? persistedOverview.totals.synapses
        : undefined,
    mirroredConnections: Object.keys(brain.synapses).length,
    cumulativeUpdates: brain.counters.plasticityEvents
  });
  const connectionCountSourceLabel = connectionCounts.uniqueSource === "queried-substrate"
    ? "Paged from authoritative substrate"
    : connectionCounts.uniqueSource === "persisted-substrate"
      ? "Validated persisted substrate overview"
      : "Local compatibility mirror";

  useEffect(() => {
    if (!window.omni?.brain.persistedSubstrateOverview) return;
    let active = true;
    void window.omni.brain.persistedSubstrateOverview(brain.id).then(
      (overview) => {
        if (active) setPersistedOverview(overview);
      },
      () => {
        if (active) setPersistedOverview(null);
      }
    );
    return () => {
      active = false;
    };
  }, [brain.id, brain.updatedAt]);

  useEffect(() => {
    if (!window.omni?.brain.querySubstrate) {
      setSubstratePage(null);
      return;
    }
    let active = true;
    setSubstrateLoading(true);
    const timer = window.setTimeout(() => {
      void window.omni!.brain.querySubstrate(brain.id, {
        entity: clustered ? "overview" : zoom >= 1.8 ? "neurons" : "assemblies",
        cursor,
        pageSize,
        region,
        search: query.trim() || undefined,
        zoom
      }).then((page) => {
        if (active) setSubstratePage(page);
      }).catch(() => {
        if (!active) return;
        // A live STDP/activation update invalidates continuation cursors. Go
        // back to the first page instead of leaving the map in a dead local
        // fallback state with the same stale cursor.
        if (cursor) {
          setCursor(undefined);
          setCursorHistory([]);
        } else {
          setSubstratePage(null);
        }
      }).finally(() => {
        if (active) setSubstrateLoading(false);
      });
    }, 120);
    return () => {
      active = false;
      window.clearTimeout(timer);
    };
  }, [brain.id, brain.updatedAt, clustered, cursor, pageSize, query, region, zoom]);

  const mapNodes = useMemo<SubstrateMapNode[]>(() => {
    if (substratePage) {
      if (clustered && substratePage.clusters.length) {
        return substratePage.clusters
          .filter((cluster) => cluster.kind !== "pathway")
          .filter((cluster) =>
            filter === "all" ||
            (filter === "active" && cluster.activeCount > 0) ||
            (filter === "salient" && cluster.maxActivation >= 0.75)
          )
          .map((cluster) => ({
            id: cluster.id,
            label: cluster.label,
            region: cluster.region,
            activation: cluster.meanActivation,
            importance: cluster.maxActivation,
            uncertainty: 1 - cluster.meanActivation,
            exposures: cluster.activeCount,
            members: [cluster.id],
            clusterCount: cluster.count
          }));
      }
      if (!clustered && substratePage.entity === "assemblies") {
        return substratePage.assemblies
          .filter((assembly) =>
            filter === "all" ||
            (filter === "active" && assembly.activationObserved === true && (assembly.activation ?? 0) >= 0.5) ||
            (filter === "salient" && assembly.importance >= 0.75)
          )
          .map((assembly) => ({
            id: assembly.id,
            label: assembly.label,
            region: assembly.region,
            activation: assembly.activationObserved === true ? assembly.activation ?? 0 : 0,
            activationObserved: assembly.activationObserved === true,
            importance: assembly.importance,
            uncertainty: 1 - assembly.confidence,
            exposures: assembly.rehearsals,
            members: [assembly.id, ...assembly.neuronIds],
            lastActivatedAt: assembly.lastRecalledAt
          }));
      }
      if (!clustered && substratePage.entity === "neurons") {
        return substratePage.neurons
          .filter((neuron) =>
            filter === "all" ||
            (filter === "active" && neuron.activation >= 0.5) ||
            (filter === "salient" && neuron.importance >= 0.75)
          )
          .map((neuron) => ({
            id: neuron.id,
            label: neuron.label,
            region: neuron.region,
            activation: neuron.activation,
            importance: neuron.importance,
            uncertainty: neuron.uncertainty,
            exposures: neuron.exposures,
            members: [neuron.id],
            lastActivatedAt: neuron.lastActivatedAt
          }));
      }
    }
    if (!filteredConcepts.length && filter === "all" && !query.trim()) {
      const endpointIds = [...new Set(
        Object.values(brain.synapses).flatMap((synapse) => [synapse.sourceId, synapse.targetId])
      )].slice(0, pageSize);
      if (endpointIds.length) {
        return endpointIds.map((id) => {
          const mirror = brain.concepts[id];
          return {
            id,
            label: mirror?.label ?? `Neuron ${id.slice(0, 8)}`,
            activation: mirror?.activation ?? 0,
            importance: mirror?.importance ?? 0.35,
            uncertainty: mirror?.uncertainty ?? 0.5,
            exposures: mirror?.exposures ?? 0,
            members: [id],
            lastActivatedAt: mirror?.lastActivatedAt
          };
        });
      }
      const measuredCount = substrateOverviewCount({
        neurons:
          (substratePage?.brainId === brain.id
            ? substratePage.totals.neurons
            : undefined) ??
          (persistedOverview?.brainId === brain.id
            ? persistedOverview.totals.neurons
            : undefined),
        assemblies:
          (substratePage?.brainId === brain.id
            ? substratePage.totals.assemblies
            : undefined) ??
          (persistedOverview?.brainId === brain.id
            ? persistedOverview.totals.assemblies
            : undefined),
        plasticityEvents: 0
      });
      const activityOnly = measuredCount === 0 && connectionCounts.cumulativeUpdates > 0;
      const representedCount = measuredCount || (activityOnly ? 1 : 0);
      if (representedCount > 0) {
        return [{
          id: "substrate-overview",
          label: activityOnly ? "Cumulative learning activity" : "Persisted neural substrate",
          region: "whole brain",
          activation: 0,
          importance: 0.5,
          uncertainty: 0,
          exposures: connectionCounts.cumulativeUpdates,
          members: ["substrate-overview"],
          clusterCount: activityOnly ? undefined : representedCount,
          activityOnly
        }];
      }
    }
    if (!clustered) {
      return filteredConcepts
        .slice(clampedOffset, clampedOffset + pageSize)
        .map((concept) => ({
          id: concept.id,
          label: concept.label,
          activation: concept.activation,
          importance: concept.importance,
          uncertainty: concept.uncertainty,
          exposures: concept.exposures,
          members: [concept.id],
          lastActivatedAt: concept.lastActivatedAt
        }));
    }
    const bucketCount = Math.max(12, Math.min(64, Math.ceil(Math.sqrt(filteredConcepts.length || 1) * 1.6)));
    const buckets = new Map<number, typeof filteredConcepts>();
    filteredConcepts.forEach((concept) => {
      const bucket = stableMapHash(concept.id) % bucketCount;
      const members = buckets.get(bucket) ?? [];
      members.push(concept);
      buckets.set(bucket, members);
    });
    return [...buckets.entries()].map(([bucket, members]) => {
      const divisor = Math.max(1, members.length);
      const strongest = [...members].sort((a, b) => b.importance - a.importance)[0]!;
      return {
        id: `cluster-${bucket}`,
        label: members.length === 1 ? strongest.label : `${strongest.label} + ${members.length - 1}`,
        activation: members.reduce((sum, concept) => sum + concept.activation, 0) / divisor,
        importance: members.reduce((sum, concept) => sum + concept.importance, 0) / divisor,
        uncertainty: members.reduce((sum, concept) => sum + concept.uncertainty, 0) / divisor,
        exposures: members.reduce((sum, concept) => sum + concept.exposures, 0),
        members: members.map((concept) => concept.id)
      };
    });
  }, [
    brain.concepts,
    brain.counters.plasticityEvents,
    brain.synapses,
    clampedOffset,
    clustered,
    filter,
    filteredConcepts,
    pageSize,
    persistedOverview,
    query,
    substratePage
  ]);
  const visibleGraphNodes = useMemo<SubstrateMapNode[]>(() => {
    if (!pathwayPage?.synapses.length) return mapNodes;
    const present = new Set(mapNodes.map((node) => node.id));
    const connectedIds = new Set<string>();
    pathwayPage.synapses.forEach((synapse) => {
      connectedIds.add(synapse.sourceId);
      connectedIds.add(synapse.targetId);
    });
    const connected = [...connectedIds]
      .filter((id) => !present.has(id))
      .map((id) => {
        const mirror = brain.concepts[id];
        return {
          id,
          label: mirror?.label ?? id,
          region: "connected pathway",
          activation: mirror?.activation ?? 0,
          importance: mirror?.importance ?? 0.35,
          uncertainty: mirror?.uncertainty ?? 0.5,
          exposures: mirror?.exposures ?? 0,
          members: [id],
          lastActivatedAt: mirror?.lastActivatedAt
        };
      });
    return [...mapNodes, ...connected];
  }, [brain.concepts, mapNodes, pathwayPage]);
  const realToView = useMemo(() => {
    const lookup = new Map<string, string>();
    visibleGraphNodes.forEach((node) => {
      lookup.set(node.id, node.id);
      node.members.forEach((member) => lookup.set(member, node.id));
    });
    return lookup;
  }, [visibleGraphNodes]);
  const graphEdges = useMemo<SubstrateMapEdge[]>(() => {
    if (substratePage) {
      const authoritative = new Map(
        [...substratePage.synapses, ...(pathwayPage?.synapses ?? [])]
          .map((synapse) => [synapse.id, synapse] as const)
      );
      const directEdges = [...authoritative.values()].flatMap((synapse) => {
        const sourceId = realToView.get(synapse.sourceId);
        const targetId = realToView.get(synapse.targetId);
        if (!sourceId || !targetId || sourceId === targetId) return [];
        return [{
          id: synapse.id,
          sourceId,
          targetId,
          effectiveWeight: synapse.effectiveWeight,
          signedStrength: synapse.effectiveWeight,
          stability: synapse.stability,
          pathways: 1
        }];
      });
      if (directEdges.length) return directEdges;
      const regionNodes = new Map(
        mapNodes.flatMap((node) => node.region ? [[node.region, node.id] as const] : [])
      );
      const clusteredEdges = substratePage.clusters.flatMap((cluster) => {
        if (cluster.kind !== "pathway" || !cluster.sourceRegion || !cluster.targetRegion) return [];
        const sourceId = regionNodes.get(cluster.sourceRegion);
        const targetId = regionNodes.get(cluster.targetRegion);
        if (!sourceId || !targetId || sourceId === targetId) return [];
        const signed = cluster.effectiveWeights.positive - cluster.effectiveWeights.negative;
        const total = Math.max(1, cluster.effectiveWeights.negative + cluster.effectiveWeights.zero + cluster.effectiveWeights.positive);
        const signedStrength = signed / total;
        return [{
          id: cluster.id,
          sourceId,
          targetId,
          effectiveWeight: signedStrength > 0.15 ? 1 as const : signedStrength < -0.15 ? -1 as const : 0 as const,
          signedStrength,
          stability: cluster.activeCount / Math.max(1, cluster.count),
          pathways: cluster.count
        }];
      });
      if (clusteredEdges.length || clustered) return clusteredEdges;
    }
    const aggregated = new Map<string, {
      sourceId: string;
      targetId: string;
      signed: number;
      stability: number;
      pathways: number;
    }>();
    Object.values(brain.synapses).forEach((synapse) => {
      const sourceId = realToView.get(synapse.sourceId);
      const targetId = realToView.get(synapse.targetId);
      if (!sourceId || !targetId || sourceId === targetId) return;
      const key = `${sourceId}\u0000${targetId}`;
      const current = aggregated.get(key) ?? {
        sourceId,
        targetId,
        signed: 0,
        stability: 0,
        pathways: 0
      };
      current.signed += synapse.effectiveWeight;
      current.stability += synapse.stability;
      current.pathways += 1;
      aggregated.set(key, current);
    });
    return [...aggregated.entries()].map(([id, edge]) => {
      const signedStrength = edge.signed / edge.pathways;
      return {
        id,
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        effectiveWeight: signedStrength > 0.15 ? 1 : signedStrength < -0.15 ? -1 : 0,
        signedStrength,
        stability: edge.stability / edge.pathways,
        pathways: edge.pathways
      };
    });
  }, [brain.synapses, clustered, mapNodes, pathwayPage, realToView, substratePage]);
  const positions = useMemo(() => {
    const values = new Map<string, { x: number; y: number }>();
    const total = Math.max(1, visibleGraphNodes.length);
    visibleGraphNodes.forEach((node, index) => {
      const angle = index * 2.399963229728653;
      const radial = Math.sqrt((index + 0.65) / total);
      values.set(node.id, {
        x: 500 + Math.cos(angle) * radial * 420,
        y: 325 + Math.sin(angle) * radial * 265
      });
    });
    return values;
  }, [visibleGraphNodes]);
  const selectedMapNode =
    visibleGraphNodes.find((node) => node.id === selected || node.members.includes(selected)) ??
    visibleGraphNodes[0];
  // The authoritative worker page may contain nodes absent from the bounded
  // BrainDocument compatibility mirror. Inspect the selected map node itself.
  const selectedConcept = selectedMapNode?.clusterCount || selectedMapNode?.activityOnly
    ? undefined
    : selectedMapNode;
  const selectedNodeId = selectedConcept?.id ?? "";
  const pathwayPageSize = Math.max(12, Math.round(Math.sqrt(pageSize) * 2));
  const allLocalConnectedSynapses = useMemo(
    () =>
      selectedConcept
        ? Object.values(brain.synapses)
            .filter((synapse) => synapse.sourceId === selectedConcept.id || synapse.targetId === selectedConcept.id)
            .sort((a, b) => Math.abs(b.effectiveWeight) - Math.abs(a.effectiveWeight) || b.uses - a.uses)
        : [],
    [brain.synapses, selectedConcept]
  );
  const connectedSynapses = pathwayPage
    ? pathwayPage.synapses
    : allLocalConnectedSynapses.slice(pathwayOffset, pathwayOffset + pathwayPageSize);
  const connectedSynapseCount = pathwayPage?.matched ?? allLocalConnectedSynapses.length;
  const connectedSynapseCountLabel = pathwayLoading && !pathwayPage
    ? "Loading…"
    : `${compactNumber(connectedSynapseCount)} total`;
  const selectedStability = connectedSynapses.length
    ? connectedSynapses.reduce((sum, synapse) => sum + synapse.stability, 0) / connectedSynapses.length
    : 0;
  const recentSynapse = [...connectedSynapses].sort((a, b) =>
    (b.lastUpdatedAt ?? "").localeCompare(a.lastUpdatedAt ?? "")
  )[0];
  const persistedInitialCount =
    persistedOverview?.brainId === brain.id &&
    filter === "all" &&
    !query.trim() &&
    !region
      ? persistedOverview.totals.neurons
      : 0;
  const matchedCount =
    substratePage?.matched ?? (persistedInitialCount || filteredConcepts.length);
  const visibleStart = matchedCount
    ? (substratePage ? (substratePage.offset ?? cursorHistory.length * pageSize) + 1 : clampedOffset + 1)
    : 0;
  const visibleEnd = Math.min(matchedCount, visibleStart + mapNodes.length - 1);
  const showingSubstrateOverview =
    mapNodes.length === 1 && mapNodes[0]?.id === "substrate-overview";

  useEffect(() => {
    setPathwayCursor(undefined);
    setPathwayCursorHistory([]);
    setPathwayOffset(0);
    setPathwayPage(null);
  }, [selectedNodeId]);

  useEffect(() => {
    if (!selectedNodeId || !window.omni?.brain.querySubstrate) {
      setPathwayPage(null);
      setPathwayLoading(false);
      return;
    }
    let active = true;
    setPathwayLoading(true);
    void window.omni.brain.querySubstrate(brain.id, {
      entity: "synapses",
      connectedTo: selectedNodeId,
      cursor: pathwayCursor,
      pageSize: pathwayPageSize,
      zoom: 1
    }).then((page) => {
      if (active) setPathwayPage(page);
    }).catch(() => {
      if (!active) return;
      if (pathwayCursor) {
        setPathwayCursor(undefined);
        setPathwayCursorHistory([]);
      } else {
        setPathwayPage(null);
      }
    }).finally(() => {
      if (active) setPathwayLoading(false);
    });
    return () => {
      active = false;
    };
  }, [brain.id, brain.updatedAt, pathwayCursor, pathwayPageSize, selectedNodeId]);

  useEffect(() => {
    setOffset(0);
    setCursor(undefined);
    setCursorHistory([]);
  }, [filter, query]);

  const openNode = (node: SubstrateMapNode) => {
    const firstMember = node.members[0];
    if (!firstMember) return;
    setSelected(firstMember);
    if (clustered && (node.clusterCount ?? node.members.length) > 1) {
      if (node.region) setRegion(node.region);
      const index = filteredConcepts.findIndex((concept) => concept.id === firstMember);
      if (index >= 0) setOffset(index);
      setCursor(undefined);
      setCursorHistory([]);
      setZoom(2);
    }
  };

  return (
    <div className="content-page map-page">
      <div className="content-page__title content-page__title--compact">
        <div>
          <button className="content-page__back" onClick={onBack}>
            <Icon name="arrow" size={13} className="map-arrow-back" /> Conversation
          </button>
          <span className="eyebrow-text">MULTIRESOLUTION SUBSTRATE</span>
          <h1>Brain map</h1>
          <p>Zoom from whole-brain assembly clusters into individual neurons and ternary pathways.</p>
        </div>
        <div className="map-toolbar">
          {region ? (
            <button className="map-region-chip" onClick={() => {
              setRegion(undefined);
              setZoom(0.72);
              setCursor(undefined);
              setCursorHistory([]);
            }}>
              {region} <Icon name="close" size={11} />
            </button>
          ) : null}
          <label
            className={cx("search-field", "search-field--small", substrateLoading && "is-searching")}
            aria-busy={substrateLoading || undefined}
          >
            <Icon name={substrateLoading ? "pulse" : "search"} size={15} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Find an assembly"
              aria-label="Search neural assemblies"
            />
            <span className="visually-hidden" role="status" aria-live="polite">
              {substrateLoading
                ? persistedInitialCount > 0
                  ? "Loading detailed map"
                  : "Searching neural substrate"
                : query.trim()
                  ? `${matchedCount.toLocaleString()} matching neural assemblies`
                  : "Neural substrate ready"}
            </span>
          </label>
          <div className="segmented">
            {(["all", "active", "salient"] as const).map((item) => (
              <button
                key={item}
                className={filter === item ? "is-active" : ""}
                title={item === "salient" ? "High importance in detailed views; high peak activation in clustered views" : undefined}
                onClick={() => setFilter(item)}
              >
                {item[0]?.toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="map-layout scalable-map-layout">
        <section className="surface graph-surface">
          <div className="graph-legend">
            <span><i className="legend-dot legend-dot--active" /> Active now</span>
            <span><i className="legend-dot legend-dot--stable" /> Stable</span>
            <span><i className="legend-line" /> Excitatory</span>
            <span><i className="legend-line legend-line--negative" /> Inhibitory</span>
            <strong>
              {substrateLoading
                ? persistedInitialCount > 0
                  ? `Loading detailed map… ${compactNumber(
                      persistedOverview!.totals.neurons
                    )} persisted neurons`
                  : "Querying substrate…"
                : showingSubstrateOverview
                  ? mapNodes[0]?.activityOnly
                    ? `Cumulative activity · ${compactNumber(connectionCounts.cumulativeUpdates)} connection updates · ${connectionCounts.topologyPending ? "unique topology loading" : `${compactNumber(connectionCounts.uniqueConnections)} unique connections`}`
                    : `Persisted substrate · ${compactNumber(connectionCounts.uniqueConnections)} unique connections`
                  : clustered
                  ? `${mapNodes.length} clusters · ${compactNumber(
                      substratePage?.totals.neurons ??
                        persistedOverview?.totals.neurons ??
                        filteredConcepts.length
                    )} neurons`
                  : `${visibleStart}–${visibleEnd} of ${compactNumber(matchedCount)}`}
            </strong>
          </div>
          <div className="brain-graph brain-graph--scalable">
            {!mapNodes.length ? (
              <div className="graph-empty">
                <Icon name="brain" size={28} />
                <strong>No neural assemblies match this view</strong>
                <span>Clear the filter or begin a conversation to form connected assemblies.</span>
              </div>
            ) : null}
            <svg viewBox="0 0 1000 650" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Virtualized multiresolution neural substrate">
              <defs>
                <radialGradient id="scalableMapBackground">
                  <stop offset="0" stopColor="#7259d2" stopOpacity=".13" />
                  <stop offset="1" stopColor="#0e0d16" stopOpacity="0" />
                </radialGradient>
                <filter id="scalableMapGlow">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              <ellipse cx="500" cy="325" rx="430" ry="295" fill="url(#scalableMapBackground)" />
              {graphEdges.map((edge) => {
                const start = positions.get(edge.sourceId);
                const end = positions.get(edge.targetId);
                if (!start || !end) return null;
                return (
                  <line
                    key={edge.id}
                    className={cx(
                      "graph-edge",
                      edge.effectiveWeight < 0
                        ? "graph-edge--negative"
                        : edge.effectiveWeight === 0
                          ? "graph-edge--neutral"
                          : "graph-edge--positive"
                    )}
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    strokeOpacity={0.1 + Math.min(0.55, Math.abs(edge.signedStrength) * 0.42)}
                    strokeWidth={0.5 + Math.min(3, edge.stability * 1.4 + Math.log2(edge.pathways + 1) * 0.25)}
                    strokeDasharray={edge.effectiveWeight < 0 ? "5 5" : edge.effectiveWeight === 0 ? "2 7" : undefined}
                  />
                );
              })}
              {visibleGraphNodes.map((node, index) => {
                const position = positions.get(node.id)!;
                const isSelected = selected === node.id || node.members.includes(selectedConcept?.id ?? "");
                const representedCount = node.clusterCount ?? node.members.length;
                const clusterBoost = clustered ? Math.min(15, Math.log2(representedCount + 1) * 4) : 0;
                const radius = Math.max(7, 10 + node.importance * 10 + clusterBoost - Math.max(0, visibleGraphNodes.length - 90) * 0.025);
                return (
                  <g
                    key={node.id}
                    className="graph-node"
                    role="button"
                    tabIndex={0}
                    onClick={() => openNode(node)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") openNode(node);
                    }}
                    transform={`translate(${position.x} ${position.y})`}
                  >
                    <circle className="graph-node__hit-area" r="18" />
                    <circle
                      r={radius + (isSelected ? 9 : 3)}
                      fill={isSelected ? "#8f74ff" : index % 5 === 0 ? "#55d8cf" : "#8e76ef"}
                      opacity={isSelected ? ".16" : ".065"}
                    />
                    <circle
                      r={radius}
                      fill={isSelected ? "#a790ff" : index % 5 === 0 ? "#5bd8d0" : "#8069d5"}
                      opacity={0.45 + node.activation * 0.45}
                      stroke={isSelected ? "var(--graph-node-stroke-selected)" : "var(--graph-node-stroke)"}
                      strokeOpacity={isSelected ? ".9" : ".32"}
                      strokeWidth={isSelected ? "2.2" : "1"}
                      filter={isSelected ? "url(#scalableMapGlow)" : undefined}
                    />
                    {clustered && representedCount > 1 ? (
                      <text y="4" textAnchor="middle" fill="var(--graph-node-ink)" fontSize="11" fontWeight="700">
                        {compactNumber(representedCount)}
                      </text>
                    ) : <circle r={Math.max(2.5, radius * 0.22)} fill="var(--graph-node-ink)" opacity=".9" />}
                    {(clustered || visibleGraphNodes.length <= 90 || isSelected) ? (
                      <text y={radius + 17} textAnchor="middle" fill="var(--graph-label)" fontSize={clustered ? "11" : "10.5"} fontWeight={isSelected ? "650" : "500"}>
                        {node.label.length > 20 ? `${node.label.slice(0, 19)}…` : node.label}
                      </text>
                    ) : null}
                  </g>
                );
              })}
            </svg>
            <div className="map-viewport-controls">
              <button aria-label="Zoom out" onClick={() => setZoom((value) => Math.max(0.45, Number((value - 0.25).toFixed(2))))}>−</button>
              <span>{Math.round(zoom * 100)}% · {clustered ? "cluster view" : "assembly detail"}</span>
              <button aria-label="Zoom in" onClick={() => setZoom((value) => Math.min(3, Number((value + 0.25).toFixed(2))))}>+</button>
              <i />
              <button
                aria-label={clustered ? "Previous substrate clusters" : "Previous substrate page"}
                disabled={substratePage ? cursorHistory.length === 0 : clampedOffset === 0}
                onClick={() => {
                  if (substratePage) {
                    setCursorHistory((history) => {
                      const next = [...history];
                      setCursor(next.pop());
                      return next;
                    });
                  } else {
                    setOffset(Math.max(0, clampedOffset - pageSize));
                  }
                }}
              >
                <Icon name="arrow" size={13} className="map-arrow-back" />
              </button>
              <button
                aria-label={clustered ? "Next substrate clusters" : "Next substrate page"}
                disabled={substratePage ? !substratePage.hasMore || !substratePage.nextCursor : clampedOffset + pageSize >= filteredConcepts.length}
                onClick={() => {
                  if (substratePage?.nextCursor) {
                    setCursorHistory((history) => [...history, cursor]);
                    setCursor(substratePage.nextCursor);
                  } else {
                    setOffset(clampedOffset + pageSize);
                  }
                }}
              >
                <Icon name="arrow" size={13} />
              </button>
            </div>
          </div>
          <div className="graph-status">
            <span><i /> {compactNumber(connectionCounts.cumulativeUpdates)} cumulative connection updates</span>
            <span>
              {connectionCounts.uniqueConnections > 0
                ? `${compactNumber(connectionCounts.uniqueConnections)} unique live connections`
                : connectionCounts.topologyPending
                  ? "Unique live connection topology is still loading"
                  : "No unique live connections yet"}
            </span>
            <span>{connectionCountSourceLabel} · no display ceiling</span>
          </div>
        </section>
        <aside className="map-inspector">
          {selectedConcept ? (
            <>
              <div className="map-inspector__head">
                <span className="map-inspector__node"><i /></span>
                <span><small>{selectedConcept.region === "assembly" ? "SELECTED ASSEMBLY" : "SELECTED NEURON"}</small><h2>{selectedConcept.label}</h2></span>
              </div>
              <div className="activation-score">
                <div style={{ "--score": `${selectedConcept.activation * 360}deg` } as React.CSSProperties}>
                  <span>{selectedMapNode?.activationObserved === false ? "—" : Math.round(selectedConcept.activation * 100)}</span>
                </div>
                <span><strong>{selectedMapNode?.activationObserved === false ? "Activation unobserved" : "Current activation"}</strong><small>{selectedMapNode?.activationObserved === false ? "No measured firing for this inspection source" : selectedConcept.lastActivatedAt ? relativeTime(selectedConcept.lastActivatedAt) : selectedConcept.activation > 0.8 ? "Highly active" : "Available"}</small></span>
              </div>
              <dl className="inspector-stats">
                <div><dt>Importance</dt><dd>{Math.round(selectedConcept.importance * 100)}%</dd></div>
                <div><dt>Stability</dt><dd>{connectedSynapses.length ? `${Math.round(selectedStability * 100)}%` : "—"}</dd></div>
                <div><dt>Uncertainty</dt><dd>{Math.round(selectedConcept.uncertainty * 100)}%</dd></div>
                <div><dt>Exposures</dt><dd>{compactNumber(selectedConcept.exposures)}</dd></div>
              </dl>
              <div className="panel-section">
                <div className="panel-section__head"><span>Connected pathways</span><em>{connectedSynapseCountLabel}</em></div>
                <div className="pathway-list">
                  {connectedSynapses.map((synapse) => {
                    const otherId = synapse.sourceId === selectedConcept.id ? synapse.targetId : synapse.sourceId;
                    return (
                      <div key={synapse.id}>
                        <span><i /> {brain.concepts[otherId]?.label ?? otherId}</span>
                        <em>{synapse.effectiveWeight > 0 ? "+1" : synapse.effectiveWeight < 0 ? "−1" : "0"}</em>
                      </div>
                    );
                  })}
                  {pathwayLoading && !pathwayPage ? (
                    <span className="pathway-empty">Loading authoritative pathways…</span>
                  ) : !connectedSynapses.length ? (
                    <span className="pathway-empty">No synapses connect this assembly yet.</span>
                  ) : null}
                </div>
                {connectedSynapseCount > pathwayPageSize ? (
                  <div className="map-viewport-controls pathway-pagination">
                    <button
                      aria-label="Previous connected pathways"
                      disabled={pathwayPage ? pathwayCursorHistory.length === 0 : pathwayOffset === 0}
                      onClick={() => {
                        if (pathwayPage) {
                          setPathwayCursorHistory((history) => {
                            const next = [...history];
                            setPathwayCursor(next.pop());
                            return next;
                          });
                        } else {
                          setPathwayOffset((value) => Math.max(0, value - pathwayPageSize));
                        }
                      }}
                    >
                      <Icon name="arrow" size={13} className="map-arrow-back" />
                    </button>
                    <span>
                      {(pathwayPage?.offset ?? pathwayOffset) + 1}–
                      {Math.min(connectedSynapseCount, (pathwayPage?.offset ?? pathwayOffset) + connectedSynapses.length)}
                    </span>
                    <button
                      aria-label="Next connected pathways"
                      disabled={pathwayPage
                        ? !pathwayPage.hasMore || !pathwayPage.nextCursor
                        : pathwayOffset + pathwayPageSize >= connectedSynapseCount}
                      onClick={() => {
                        if (pathwayPage?.nextCursor) {
                          setPathwayCursorHistory((history) => [...history, pathwayCursor]);
                          setPathwayCursor(pathwayPage.nextCursor);
                        } else {
                          setPathwayOffset((value) => value + pathwayPageSize);
                        }
                      }}
                    >
                      <Icon name="arrow" size={13} />
                    </button>
                  </div>
                ) : null}
              </div>
              <div className="panel-section">
                <div className="panel-section__head"><span>Latest change on page</span><em>STDP</em></div>
                {recentSynapse ? (
                  <div className="change-note">
                    <Icon name="pulse" size={16} />
                    <span>
                      Latest connected synapse changed with <strong>{brain.concepts[recentSynapse.sourceId === selectedConcept.id ? recentSynapse.targetId : recentSynapse.sourceId]?.label ?? "another assembly"}</strong>.
                      <small>{recentSynapse.lastUpdatedAt ? relativeTime(recentSynapse.lastUpdatedAt) : "Time unavailable"} · ternary {recentSynapse.effectiveWeight > 0 ? "+1" : recentSynapse.effectiveWeight < 0 ? "−1" : "0"}</small>
                    </span>
                  </div>
                ) : <span className="pathway-empty">No connection change has been recorded for this assembly.</span>}
              </div>
            </>
          ) : selectedMapNode ? (
            <>
              <div className="map-inspector__head">
                <span className="map-inspector__node"><i /></span>
                <span>
                  <small>{selectedMapNode.activityOnly ? "ACTIVITY SUMMARY" : selectedMapNode.clusterCount ? "SELECTED CLUSTER" : substratePage?.entity === "neurons" ? "SELECTED NEURON" : "SELECTED ASSEMBLY"}</small>
                  <h2>{selectedMapNode.label}</h2>
                </span>
              </div>
              <div className="activation-score">
                <div style={{ "--score": `${selectedMapNode.activation * 360}deg` } as React.CSSProperties}>
                  <span>{selectedMapNode.activityOnly || selectedMapNode.activationObserved === false ? "—" : Math.round(selectedMapNode.activation * 100)}</span>
                </div>
                <span>
                  <strong>{selectedMapNode.activityOnly
                    ? connectionCounts.topologyPending
                      ? "Unique topology loading"
                      : "No unique live connections"
                    : "Measured activation"}</strong>
                  <small>{selectedMapNode.activityOnly ? "Updates are activity, not live edge count" : selectedMapNode.region ?? "Unified substrate"}</small>
                </span>
              </div>
              {selectedMapNode.activityOnly ? (
                <dl className="inspector-stats">
                  <div><dt>Cumulative updates</dt><dd>{compactNumber(connectionCounts.cumulativeUpdates)}</dd></div>
                  <div><dt>Unique connections</dt><dd>{connectionCounts.topologyPending ? "Loading" : compactNumber(connectionCounts.uniqueConnections)}</dd></div>
                </dl>
              ) : (
                <dl className="inspector-stats">
                  <div><dt>Importance</dt><dd>{Math.round(selectedMapNode.importance * 100)}%</dd></div>
                  <div><dt>Uncertainty</dt><dd>{Math.round(selectedMapNode.uncertainty * 100)}%</dd></div>
                  <div><dt>{selectedMapNode.clusterCount ? "Neurons" : "Exposures"}</dt><dd>{compactNumber(selectedMapNode.clusterCount ?? selectedMapNode.exposures)}</dd></div>
                  <div><dt>Pathways on page</dt><dd>{graphEdges.filter((edge) => edge.sourceId === selectedMapNode.id || edge.targetId === selectedMapNode.id).length}</dd></div>
                </dl>
              )}
              {selectedMapNode.clusterCount ? (
                <Button kind="primary" icon="expand" onClick={() => openNode(selectedMapNode)}>
                  Open this region
                </Button>
              ) : null}
              <div className="panel-section">
                <div className="panel-section__head">
                  <span>{selectedMapNode.activityOnly ? "Unique connection topology" : "Ternary pathways"}</span>
                  <em>{selectedMapNode.activityOnly ? "not inferred from updates" : selectedMapNode.clusterCount ? "aggregate page" : connectedSynapseCountLabel}</em>
                </div>
                <div className="pathway-list">
                  {(selectedMapNode.clusterCount
                    ? graphEdges.filter((edge) => edge.sourceId === selectedMapNode.id || edge.targetId === selectedMapNode.id)
                    : connectedSynapses
                  ).map((edge) => {
                      const referenceId = selectedNodeId || selectedMapNode.id;
                      const otherId = edge.sourceId === referenceId ? edge.targetId : edge.sourceId;
                      return (
                        <div key={edge.id}>
                          <span><i /> {brain.concepts[otherId]?.label ?? mapNodes.find((node) => node.id === otherId)?.label ?? otherId}</span>
                          <em>{edge.effectiveWeight > 0 ? "+1" : edge.effectiveWeight < 0 ? "−1" : "0"}</em>
                        </div>
                      );
                    })}
                  {!selectedMapNode.clusterCount && pathwayLoading && !pathwayPage ? (
                    <span className="pathway-empty">Loading authoritative pathways…</span>
                  ) : !selectedMapNode.clusterCount && !connectedSynapses.length ? (
                    <span className="pathway-empty">
                      {selectedMapNode.activityOnly
                        ? "Validated live connections will appear when the substrate topology is available."
                        : "No synapses connect this neural node yet."}
                    </span>
                  ) : null}
                </div>
                {!selectedMapNode.activityOnly && !selectedMapNode.clusterCount && connectedSynapseCount > pathwayPageSize ? (
                  <div className="map-viewport-controls pathway-pagination">
                    <button
                      aria-label="Previous connected pathways"
                      disabled={pathwayPage ? pathwayCursorHistory.length === 0 : pathwayOffset === 0}
                      onClick={() => {
                        if (pathwayPage) {
                          setPathwayCursorHistory((history) => {
                            const next = [...history];
                            setPathwayCursor(next.pop());
                            return next;
                          });
                        } else {
                          setPathwayOffset((value) => Math.max(0, value - pathwayPageSize));
                        }
                      }}
                    >
                      <Icon name="arrow" size={13} className="map-arrow-back" />
                    </button>
                    <span>
                      {(pathwayPage?.offset ?? pathwayOffset) + 1}–
                      {Math.min(connectedSynapseCount, (pathwayPage?.offset ?? pathwayOffset) + connectedSynapses.length)}
                    </span>
                    <button
                      aria-label="Next connected pathways"
                      disabled={pathwayPage
                        ? !pathwayPage.hasMore || !pathwayPage.nextCursor
                        : pathwayOffset + pathwayPageSize >= connectedSynapseCount}
                      onClick={() => {
                        if (pathwayPage?.nextCursor) {
                          setPathwayCursorHistory((history) => [...history, pathwayCursor]);
                          setPathwayCursor(pathwayPage.nextCursor);
                        } else {
                          setPathwayOffset((value) => value + pathwayPageSize);
                        }
                      }}
                    >
                      <Icon name="arrow" size={13} />
                    </button>
                  </div>
                ) : null}
              </div>
            </>
          ) : (
            <div className="graph-empty graph-empty--inspector">
              <Icon name="brain" size={25} />
              <strong>Select an assembly</strong>
              <span>Zoom or search to inspect its local ternary pathways.</span>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

function TraceWorkspace({ brain }: { brain: BrainDocument }) {
  const fallbackTraces = brain.traces.length > 0 ? brain.traces : window.omni ? [] : [makeDemoBrain().traces[0]!];
  const [traces, setTraces] = useState(fallbackTraces);
  const [selectedId, setSelectedId] = useState(fallbackTraces.at(-1)?.id ?? "");
  const [tab, setTab] = useState<"trace" | "journal">("trace");

  useEffect(() => {
    if (!window.omni) {
      setTraces(fallbackTraces);
      return;
    }
    void window.omni.trace.list(brain.id, { limit: 100 }).then(setTraces);
  }, [brain.id, brain.traces]);

  if (!traces.length) {
    const hasJournal = (brain.activity?.journalCount ?? brain.journal?.length ?? 0) > 0;
    return (
      <div className="content-page trace-page">
        <div className="content-page__title content-page__title--compact">
          <div>
            <span className="eyebrow-text">VERIFIABLE ACTIVITY</span>
            <h1>Trace & journal</h1>
            <p>
              {hasJournal
                ? "Browse the append-only operational journal while inference traces are still empty."
                : "Operational traces will appear after this brain performs inference or learning."}
            </p>
          </div>
        </div>
        {hasJournal ? (
          <JournalView brain={brain} />
        ) : (
          <div className="journal-empty surface">
            <span><Icon name="trace" size={24} /></span>
            <h2>No operational traces yet</h2>
            <p>Start a conversation or encode an experience. Real activation and mutation events will be recorded here.</p>
          </div>
        )}
      </div>
    );
  }

  const selected = traces.find((traceItem) => traceItem.id === selectedId) ?? traces.at(-1)!;
  const traceEvidence = presentTraceEvidence(selected);
  const traceSignalMeasures = presentTraceSignalMeasures(selected);
  const connectionChangeSteps = selected.steps.filter((stepItem) =>
    /plastic|synap|hebb|stdp|connection learning|strengthen/i.test(`${stepItem.stage} ${stepItem.detail}`)
  );
  const liquidStep = selected.steps.find((stepItem) => /liquid|time constant|integration/i.test(`${stepItem.stage} ${stepItem.detail}`));

  const exportTrace = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(selected, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${brain.name}-${selected.id}.trace.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="content-page trace-page">
      <div className="content-page__title content-page__title--compact">
        <div>
          <span className="eyebrow-text">VERIFIABLE ACTIVITY</span>
          <h1>Trace & journal</h1>
          <p>Follow activations and real mutations. Prose reflections remain clearly labeled self-reports.</p>
        </div>
        <div className="segmented segmented--large">
          <button className={tab === "trace" ? "is-active" : ""} onClick={() => setTab("trace")}>
            <Icon name="trace" size={15} /> Operational trace
          </button>
          <button className={tab === "journal" ? "is-active" : ""} onClick={() => setTab("journal")}>
            <Icon name="file" size={15} /> Inner journal
          </button>
        </div>
      </div>
      {tab === "trace" ? (
        <div className="trace-layout">
          <aside className="surface trace-list">
            <div className="trace-list__head">
              <strong>Recent inference</strong>
            </div>
            {[...traces].reverse().map((item, index) => (
              <button key={item.id} className={selected.id === item.id ? "is-active" : ""} onClick={() => setSelectedId(item.id)}>
                <span className="trace-list__pulse"><i /></span>
                <span>
                  <strong>{item.input}</strong>
                  <small>{relativeTime(item.createdAt)} · {item.steps.length} recorded operations</small>
                </span>
                <em>{index === 0 ? "latest" : ""}</em>
              </button>
            ))}
            {!window.omni ? (
              <>
                <div className="trace-list__day">DEMO EVENTS</div>
                {[1, 2, 3].map((index) => (
                  <button key={index} className="is-muted">
                    <span className="trace-list__pulse"><i /></span>
                    <span>
                      <strong>{index === 1 ? "Connections updated" : index === 2 ? "Document learned" : "Autonomous reflection"}</strong>
                      <small>demo preview · no live event</small>
                    </span>
                  </button>
                ))}
              </>
            ) : null}
          </aside>
          <section className="surface trace-detail">
            <div className="trace-detail__head">
              <div>
                <span className="eyebrow-text">TRACE · {selected.seed.toString(16).toUpperCase()}</span>
                <h2>{selected.input}</h2>
                <p>{new Date(selected.createdAt).toLocaleString()} · {selected.runtime}</p>
              </div>
              <Button icon="download" onClick={exportTrace}>Export trace</Button>
            </div>
            <div className="trace-summary">
              <div><span>Branches</span><strong>{selected.branches}</strong><small>selected #{selected.selectedBranch}</small></div>
              <div><span>Recalled idea previews</span><strong>{selected.recalledIdeas.length}</strong><small>{selected.activatedConcepts.length} labeled concept activations captured</small></div>
              {traceSignalMeasures.length ? (
                <div>
                  <span>Neural signal measures</span>
                  <strong>{traceSignalMeasures[0]}</strong>
                  <small>{traceSignalMeasures.slice(1).join(" · ") || "Worker-recorded signal/vector activity"}</small>
                </div>
              ) : null}
              <div><span>Connection changes</span><strong>{connectionChangeSteps.length}</strong><small>{connectionChangeSteps[0]?.value ?? "none recorded"}</small></div>
              <div><span>Timing adjustment</span><strong>{liquidStep?.value ?? "—"}</strong><small>{liquidStep ? "recorded" : "not recorded"}</small></div>
            </div>
            {traceEvidence.activatedConcepts.length ||
            traceEvidence.recalledIdeas.length ||
            traceEvidence.candidates.length ? (
              <div className="trace-evidence" aria-label="Measured trace evidence">
                {traceEvidence.activatedConcepts.length ? (
                  <section className="trace-evidence__group">
                    <span className="trace-evidence__head">
                      <small>ACTIVATION MEASUREMENT</small>
                      <strong>{traceEvidence.activatedConcepts.length} ideas observed</strong>
                    </span>
                    <div className="trace-concepts">
                      {traceEvidence.activatedConcepts.map((concept) => (
                        <span key={concept.id}>
                          {concept.label}<i>{Math.round(concept.activation * 100)}</i>
                        </span>
                      ))}
                    </div>
                  </section>
                ) : null}
                {traceEvidence.recalledIdeas.length ? (
                  <section className="trace-evidence__group">
                    <span className="trace-evidence__head">
                      <small>RECALL MEASUREMENT</small>
                      <strong>{traceEvidence.recalledIdeas.length} learned ideas recalled</strong>
                    </span>
                    <div className="trace-recalls">
                      {traceEvidence.recalledIdeas.map((idea) => (
                        <span key={idea.id}>
                          {idea.preview}<i>{Math.round(idea.score * 100)}</i>
                        </span>
                      ))}
                    </div>
                  </section>
                ) : null}
                {traceEvidence.candidates.length ? (
                  <section className="trace-evidence__group">
                    <span className="trace-evidence__head">
                      <small>CANDIDATE MEASUREMENT</small>
                      <strong>{traceEvidence.candidates.length} alternatives compared</strong>
                    </span>
                    <div className="branch-row">
                      {traceEvidence.candidates.map((candidate) => (
                        <span key={candidate.id} className={candidate.outcome === "selected" ? "is-selected" : ""}>
                          {candidate.label}<i>{candidate.outcome}</i>
                        </span>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            ) : null}
            <div className="trace-mechanisms__head">
              <span>
                <small>RECORDED MECHANISMS</small>
                <strong>{traceEvidence.mechanisms.length} observed operations</strong>
              </span>
              <p>Unordered evidence from this event; it is not a fixed memory sequence.</p>
            </div>
            <ul
              className="trace-mechanisms"
              aria-label="Recorded mechanisms; visual order does not imply a fixed sequence"
            >
              {traceEvidence.mechanisms.map((mechanism) => (
                <li
                  className={`trace-mechanism trace-mechanism--${mechanism.kind}`}
                  key={mechanism.id}
                >
                  <span className="trace-mechanism__marker" aria-hidden="true"><i /></span>
                  <div>
                    <span className="trace-mechanism__title">
                      <strong>{mechanism.title}</strong>
                      {mechanism.measure ? <em>{mechanism.measure}</em> : null}
                    </span>
                    <p>{mechanism.detail}</p>
                  </div>
                </li>
              ))}
              {!traceEvidence.mechanisms.length ? (
                <li className="trace-mechanisms__empty">No mechanism operations were recorded for this trace.</li>
              ) : null}
            </ul>
            <div className="trace-note">
              <Icon name="info" size={16} />
              <span>{selected.note}</span>
            </div>
          </section>
        </div>
      ) : (
        <JournalView brain={brain} />
      )}
    </div>
  );
}

function JournalView({ brain }: { brain: BrainDocument }) {
  const [entries, setEntries] = useState((brain.journal ?? []).slice(-60));
  const [cursor, setCursor] = useState<string | undefined>();
  const [pageDepth, setPageDepth] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [selectedId, setSelectedId] = useState(entries.at(-1)?.id ?? entries[0]?.id ?? "");

  const loadPage = useCallback(async (nextCursor?: string, depth = 0) => {
    if (!window.omni) {
      const fallback = (brain.journal ?? []).slice(-60);
      setEntries(fallback);
      setCursor(undefined);
      setPageDepth(0);
      setSelectedId(fallback.at(-1)?.id ?? fallback[0]?.id ?? "");
      return;
    }
    setLoading(true);
    try {
      const page = await window.omni.brain.journalPage(brain.id, nextCursor, 60);
      const nextEntries = page.entries.map((entry) => entry.entry);
      setEntries(nextEntries);
      setCursor(page.nextCursor);
      setPageDepth(depth);
      setSelectedId(nextEntries.at(-1)?.id ?? nextEntries[0]?.id ?? "");
      setLoadError("");
    } catch (error) {
      setLoadError(error instanceof Error ? error.message : "The journal page could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, [brain.id, brain.journal]);

  useEffect(() => {
    void loadPage(undefined, 0);
  }, [loadPage, brain.activity?.journalHeadSha256]);

  const selected = entries.find((entry) => entry.id === selectedId) ?? entries.at(-1);
  const selectedKind = selected ? presentJournalKind(selected.kind) : "";

  if (!selected) {
    return (
      <div className="journal-empty surface">
        <span><Icon name="file" size={24} /></span>
        <h2>No journal entries yet</h2>
        <p>
          {loadError || "Journal entries appear only after the brain records real learning, a tool, a fork, or a system event."}
        </p>
      </div>
    );
  }

  return (
    <div className="journal-layout">
      <section className="surface journal-entry">
        <div className="journal-entry__top">
          <span className="journal-date">
            <strong>{new Date(selected.createdAt).getDate()}</strong>
            <small>{new Date(selected.createdAt).toLocaleString("en", { month: "short" }).toUpperCase()}</small>
          </span>
          <span>
            <small>{selectedKind.toUpperCase()} EVENT · {new Date(selected.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
            <h2>{selected.summary}</h2>
          </span>
          <em>{selectedKind}</em>
        </div>
        <div className="journal-prose">
          {(selected.detail ?? selected.summary).split(/\n{2,}/).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
        </div>
        <div className="journal-entry__footer">
          <span><Icon name="brain" size={14} /> Recorded by {brain.name}</span>
          <span><Icon name="pulse" size={14} /> Verifiable {selectedKind} event</span>
          <span>{relativeTime(selected.createdAt)}</span>
        </div>
      </section>
      <aside className="surface journal-sidebar">
        <h3>Recorded entries</h3>
        <div className="journal-entry-list">
          {[...entries].reverse().map((entry) => (
            <button key={entry.id} className={entry.id === selected.id ? "is-active" : ""} onClick={() => setSelectedId(entry.id)}>
              <span>{entry.summary}</span>
              <small>{presentJournalKind(entry.kind)} · {relativeTime(entry.createdAt)}</small>
            </button>
          ))}
        </div>
        <div className="journal-page-controls">
          {pageDepth > 0 ? (
            <Button disabled={loading} onClick={() => void loadPage(undefined, 0)}>
              Newest
            </Button>
          ) : null}
          {cursor ? (
            <Button disabled={loading} onClick={() => void loadPage(cursor, pageDepth + 1)}>
              Older entries
            </Button>
          ) : null}
        </div>
        <dl className="journal-facts">
          <div>
            <dt>Recorded events</dt>
            <dd>{(brain.activity?.journalCount ?? entries.length).toLocaleString("en-US")}</dd>
          </div>
          <div>
            <dt>Measured novelty</dt>
            <dd>{brain.traces.at(-1) ? `${Math.round(brain.traces.at(-1)!.driveScores.novelty * 100)}%` : "Awaiting activity"}</dd>
          </div>
          <div><dt>Trace detail</dt><dd>{brain.config.traceDetail}</dd></div>
        </dl>
        <div className="journal-disclosure">
          <Icon name="info" size={16} />
          <p>
            Reflective prose is the brain’s interpretation of its state. Operational events are factual; prose is not a
            guaranteed transcript of a private recurrent scratchpad.
          </p>
        </div>
      </aside>
    </div>
  );
}

function PersistedArtifactMedia({ artifact }: { artifact: GeneratedArtifact }) {
  const presentation = persistedArtifactMediaPresentation(artifact);
  if (!artifact.available || !artifact.mediaUrl || presentation.kind === "unavailable") {
    return (
      <span className="imagination-gallery__unavailable" aria-label={presentation.ariaLabel}>
        <Icon name="warning" size={22} />
        {artifact.unavailableReason === "missing"
          ? "Artifact file is missing"
          : "Artifact integrity could not be verified"}
      </span>
    );
  }
  if (presentation.kind === "animated-video") {
    return (
      <span className="imagination-gallery__animated-video">
        <img src={artifact.mediaUrl} alt={presentation.ariaLabel} />
        <small>{presentation.animationCopy}</small>
      </span>
    );
  }
  if (presentation.kind === "image") {
    return <img src={artifact.mediaUrl} alt={presentation.ariaLabel} />;
  }
  if (presentation.kind === "video") {
    return (
      <video
        src={artifact.mediaUrl}
        controls
        muted
        preload="metadata"
        aria-label={presentation.ariaLabel}
      />
    );
  }
  return (
    <span className="imagination-gallery__audio">
      <Icon name="wave" size={24} />
      <audio
        src={artifact.mediaUrl}
        controls
        preload="metadata"
        aria-label={presentation.ariaLabel}
      />
    </span>
  );
}

function ImaginationWorkspace({ brain, onToast }: { brain: BrainDocument; onToast: (message: string) => void }) {
  const [mode, setMode] = useState<ModalityId>("image");
  const [prompt, setPrompt] = useState("A memory palace growing new luminous pathways after rain");
  const [outputMode, setOutputMode] = useState<"auto" | "exact">("auto");
  const [outputWidth, setOutputWidth] = useState("256");
  const [outputHeight, setOutputHeight] = useState("256");
  const [outputDurationMs, setOutputDurationMs] = useState("2000");
  const [outputSampleRate, setOutputSampleRate] = useState("16000");
  const [outputFps, setOutputFps] = useState("8");
  const [includeAudio, setIncludeAudio] = useState(true);
  const [mediaPlaybackFailed, setMediaPlaybackFailed] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [variation, setVariation] = useState(0);
  const [job, setJob] = useState<RuntimeJob | null>(null);
  const [installedPacks, setInstalledPacks] = useState<InstalledModalityPack[]>([]);
  const [artifacts, setArtifacts] = useState<GeneratedArtifact[]>([]);
  const [artifactTotal, setArtifactTotal] = useState(0);
  const [artifactCursor, setArtifactCursor] = useState<string | undefined>();
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [galleryLoading, setGalleryLoading] = useState(false);
  const [packUrl, setPackUrl] = useState("");
  const [packBusy, setPackBusy] = useState(false);
  const settledJobs = useRef(new Set<string>());
  const demo = !window.omni;

  const reloadPacks = async () => {
    if (!window.omni) return;
    setInstalledPacks(await window.omni.catalog.listModalityPacks(brain.id));
  };

  const reloadArtifacts = async () => {
    if (!window.omni) return;
    setGalleryLoading(true);
    try {
      const page = await window.omni.modality.artifacts(brain.id, undefined, 48);
      setArtifacts(page.artifacts);
      setArtifactTotal(page.totalArtifacts);
      setArtifactCursor(page.nextCursor);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The imagination gallery could not load.");
    } finally {
      setGalleryLoading(false);
    }
  };

  const loadMoreArtifacts = async () => {
    if (!window.omni || !artifactCursor || galleryLoading) return;
    setGalleryLoading(true);
    try {
      const page = await window.omni.modality.artifacts(brain.id, artifactCursor, 48);
      setArtifacts((current) => {
        const merged = new Map(current.map((artifact) => [artifact.id, artifact]));
        page.artifacts.forEach((artifact) => merged.set(artifact.id, artifact));
        return [...merged.values()];
      });
      setArtifactTotal(page.totalArtifacts);
      setArtifactCursor(page.nextCursor);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Older imagination artifacts could not load.");
    } finally {
      setGalleryLoading(false);
    }
  };

  useEffect(() => {
    setGalleryOpen(false);
    setArtifacts([]);
    setArtifactTotal(0);
    setArtifactCursor(undefined);
    void reloadPacks();
    void reloadArtifacts();
  }, [brain.id]);

  useEffect(() => {
    if (!galleryOpen) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape") setGalleryOpen(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [galleryOpen]);

  useEffect(() => {
    const trackedId = job?.id;
    if (!window.omni || !trackedId) return;
    const applyJob = (eventJob: RuntimeJob) => {
      if (eventJob.id !== trackedId) return;
      setJob(eventJob);
      if (["complete", "failed", "cancelled"].includes(eventJob.state)) {
        setGenerating(false);
        if (settledJobs.current.has(eventJob.id)) return;
        settledJobs.current.add(eventJob.id);
        if (eventJob.state === "complete") {
          setVariation((current) => current + 1);
          void reloadArtifacts();
          onToast("Local modality artifact completed.");
        } else if (eventJob.error) {
          onToast(eventJob.error);
        }
      }
    };
    const unsubscribe = window.omni.train.onEvent(({ job: eventJob }) => {
      applyJob(eventJob);
    });
    // A tiny local pack can finish before React commits the new job id and
    // installs the event subscription. The durable job list closes that race.
    void window.omni.train
      .list(brain.id)
      .then((jobs) => {
        const current = jobs.find((candidate) => candidate.id === trackedId);
        if (current) applyJob(current);
      })
      .catch(() => undefined);
    return unsubscribe;
  }, [brain.id, job?.id, onToast]);

  const generate = async () => {
    setGenerating(true);
    setJob(null);
    setMediaPlaybackFailed(false);
    try {
      if (window.omni) {
        const positiveInteger = (raw: string, label: string): number => {
          if (!/^\d+$/.test(raw.trim())) throw new Error(`${label} must be a positive whole number.`);
          const value = Number(raw);
          if (!Number.isSafeInteger(value) || value < 1) {
            throw new Error(`${label} must be a positive whole number.`);
          }
          return value;
        };
        const positiveNumber = (raw: string, label: string): number => {
          const value = Number(raw);
          if (!Number.isFinite(value) || value <= 0) throw new Error(`${label} must be positive.`);
          return value;
        };
        let settings: ModalityGenerationSettings | undefined;
        if (mode !== "vision") {
          settings = outputMode === "auto"
            ? { outputMode: "auto" }
            : mode === "image"
              ? {
                  outputMode: "exact",
                  width: positiveInteger(outputWidth, "Width"),
                  height: positiveInteger(outputHeight, "Height")
                }
              : mode === "audio"
                ? {
                    outputMode: "exact",
                    durationMs: positiveNumber(outputDurationMs, "Duration"),
                    sampleRate: positiveInteger(outputSampleRate, "Sample rate")
                  }
                : {
                    outputMode: "exact",
                    width: positiveInteger(outputWidth, "Width"),
                    height: positiveInteger(outputHeight, "Height"),
                    durationMs: positiveNumber(outputDurationMs, "Duration"),
                    sampleRate: positiveInteger(outputSampleRate, "Sample rate"),
                    fps: positiveInteger(outputFps, "FPS"),
                    includeAudio
                  };
        }
        const request = {
          brainId: brain.id,
          modality: mode,
          prompt,
          conceptIds: brain.workingMemory.map((item) => item.conceptId),
          ...(settings ? { settings } : {})
        };
        const started =
          mode === "vision"
            ? await window.omni.modality.selectInput(request)
            : await window.omni.modality.generate(request);
        if (!started) {
          setGenerating(false);
          return;
        }
        setJob(started);
        onToast(`${started.label} queued in the local Python engine.`);
      } else {
        await new Promise((resolve) => window.setTimeout(resolve, 1100));
        onToast("Demo preview rendered; no modality model ran.");
        setVariation((current) => current + 1);
        setGenerating(false);
      }
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The modality job could not start.");
      setGenerating(false);
    }
  };

  const cancelGeneration = async () => {
    if (!window.omni || !job) {
      setGenerating(false);
      return;
    }
    const stop = runtimeJobStopPresentation(job);
    if (!stop.requestAllowed) return;
    const requestedJob = job;
    setJob({
      ...requestedJob,
      state: "cancelling",
      label: `Cancelling ${requestedJob.kind} · waiting for acknowledgement`,
      error: undefined,
      updatedAt: new Date().toISOString()
    });
    setGenerating(true);
    try {
      const cancelled = await window.omni.train.cancel(requestedJob.id);
      setJob(cancelled);
      setGenerating(false);
      onToast("Imagination stopped after worker acknowledgement; the last real decoder revision remains visible.");
    } catch (error) {
      const reason = conciseUiMessage(
        error instanceof Error ? error.message : String(error),
        "The modality job is still awaiting stop acknowledgement."
      );
      const current = await window.omni.train
        .list(brain.id)
        .then((jobs) => jobs.find((candidate) => candidate.id === requestedJob.id))
        .catch(() => undefined);
      setJob(current ?? {
        ...requestedJob,
        state: "cancelling",
        label: `${requestedJob.kind} stop needs acknowledgement`,
        error: reason,
        updatedAt: new Date().toISOString()
      });
      onToast(`${reason} Retry stop when ready.`);
    }
  };

  const installPack = async (source: "url" | "file") => {
    if (!window.omni) {
      onToast("Pack installation is available in the packaged desktop app.");
      return;
    }
    setPackBusy(true);
    try {
      const installed =
        source === "url"
          ? await window.omni.catalog.installModalityPackUrl({
              brainId: brain.id,
              url: packUrl.trim()
            })
          : await window.omni.catalog.installModalityPackFile(brain.id);
      if (!installed) return;
      await reloadPacks();
      onToast(
        `Installed ${installed.name} for ${installed.modalities.join(", ")}; safe tensors are active.`
      );
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The modality pack could not be installed.");
    } finally {
      setPackBusy(false);
    }
  };

  const output =
    job && typeof job.output === "object" && job.output !== null
      ? (job.output as Record<string, unknown>)
      : null;
  const media = jobImaginationMedia(job, mode);
  useEffect(() => setMediaPlaybackFailed(false), [media.sourceUrl]);
  const generationStop = runtimeJobStopPresentation(job);
  const embedding = Array.isArray(output?.embedding) ? output.embedding : null;
  const qualityNote = typeof output?.qualityNote === "string"
    ? output.qualityNote
    : null;
  const finalMediaOutput = output?.mediaOutput && typeof output.mediaOutput === "object"
    ? output.mediaOutput as Record<string, unknown>
    : null;
  const mediaEvidence = finalMediaOutput ?? (
    job?.preview && typeof job.preview === "object"
      ? job.preview as unknown as Record<string, unknown>
      : null
  );
  const trainingState = mediaEvidence?.trainingState === "untrained-diagnostic" ||
    mediaEvidence?.trainingState === "trained-unverified-quality"
    ? mediaEvidence.trainingState
    : null;
  const finalMediaPlan = output?.mediaOutputPlan && typeof output.mediaOutputPlan === "object"
    ? output.mediaOutputPlan as Record<string, unknown>
    : null;
  const outputShape = finalMediaPlan
    ? mode === "image"
      ? `${String(finalMediaPlan.width)} × ${String(finalMediaPlan.height)} px`
      : mode === "audio"
        ? `${Math.round(Number(finalMediaPlan.durationMs))} ms · ${String(finalMediaPlan.sampleRate)} Hz`
        : mode === "video"
          ? `${String(finalMediaPlan.width)} × ${String(finalMediaPlan.height)} px · ${String(finalMediaPlan.totalFrames)} frames @ ${String(finalMediaPlan.fps)} fps`
          : null
    : null;
  const seedCopy = imaginationSeedCopy(
    output,
    job?.preview,
    brain.workingMemory.length
  );
  const downloadOutput = () => {
    if (!media.isFinal || !media.downloadUrl) return;
    const mimeType = typeof output?.mimeType === "string" ? output.mimeType : "application/octet-stream";
    const extension = mimeType.includes("mp4")
      ? "mp4"
      : mimeType.includes("png")
        ? "png"
        : mimeType.includes("wav")
          ? "wav"
          : "bin";
    const anchor = document.createElement("a");
    anchor.href = media.downloadUrl;
    anchor.download = `${brain.name}-${mode}-${job?.id ?? Date.now()}.${extension}`;
    anchor.click();
  };

  return (
    <div className="content-page imagine-page">
      <div className="content-page__title content-page__title--compact">
        <div>
          <span className="eyebrow-text">SHARED IDEA SPACE</span>
          <h1>Imagination</h1>
          <p>Let an internal idea become image, sound, or motion without routing it through long-term text.</p>
        </div>
        <span className="pack-status">
          <i /> Shared neural image · audio · video · vision pathways · {installedPacks.length} verified pack{installedPacks.length === 1 ? "" : "s"}
        </span>
      </div>
      <div className="imagine-layout">
        <section className="surface imagination-canvas">
          <div className={cx("generated-art", generating && "is-generating", `generated-art--${variation % 3}`)}>
            {media.sourceUrl && media.mimeType.startsWith("audio/") && !mediaPlaybackFailed ? (
              <div className="generated-media generated-media--audio">
                <Icon name="wave" size={42} />
                <audio
                  key={`audio-${job?.id}-${media.revision ?? "final"}`}
                  src={media.sourceUrl}
                  controls
                  onError={() => setMediaPlaybackFailed(true)}
                />
                <span>{media.isLiveRevision ? "Growing neural codec waveform" : "Final local neural audio artifact"}</span>
              </div>
            ) : media.sourceUrl && media.mimeType.startsWith("video/") && !mediaPlaybackFailed ? (
              <video
                key={`video-${job?.id}-${media.revision ?? "final"}`}
                className="generated-media-video"
                src={media.sourceUrl}
                controls
                autoPlay={!media.isLiveRevision}
                loop={!media.isLiveRevision}
                muted
                onError={() => setMediaPlaybackFailed(true)}
                aria-label={media.isLiveRevision ? "Live locally decoded video revision" : "Locally generated video artifact"}
              />
            ) : media.sourceUrl && media.mimeType.startsWith("image/") ? (
              <img
                className="generated-media-image"
                src={media.sourceUrl}
                alt={media.isLiveRevision ? `Live ${media.modality ?? mode} decoder revision` : `Locally generated ${media.modality ?? mode} artifact`}
              />
            ) : generating ? (
              <span className="generation-state">
                <Icon name={generationStop.waitingForAcknowledgement ? "pulse" : "sparkles"} size={22} />
                {job?.state === "cancelling"
                  ? generationStop.retryAvailable
                    ? "Stop acknowledgement failed · retry available"
                    : "Stopping generation · waiting for worker acknowledgement…"
                  : mode === "vision"
                    ? "Encoding selected image…"
                    : media.statusLabel}
              </span>
            ) : mediaPlaybackFailed && media.downloadUrl ? (
              <span className="imagination-empty" role="status">
                <Icon name="warning" size={25} />
                <strong>Inline playback unavailable for this output</strong>
                <small>The generated artifact is intact and remains available with Download output.</small>
              </span>
            ) : embedding ? (
              <div className="vision-result">
                <span><Icon name="eye" size={30} /></span>
                <strong>Vision encoding complete</strong>
                <p>{embedding.length}-dimension embedding mapped into {brain.name}’s shared idea space.</p>
              </div>
            ) : demo ? (
              <>
                <div className="generated-art__mist" />
                <div className="generated-art__structure">
                  {Array.from({ length: 7 }).map((_, index) => <i key={index} />)}
                </div>
                <div className="generated-art__path" />
                <span className="generated-art__label">
                  <small>DEMO PREVIEW · NO MODEL RAN</small>
                  <strong>Memory palace, visual concept</strong>
                </span>
              </>
            ) : (
              <span className="imagination-empty">
                <Icon name={mode === "vision" ? "eye" : "sparkles"} size={25} />
                <strong>{mode === "vision" ? "Choose an image to understand" : "No generated artifact yet"}</strong>
                <small>{mode === "vision" ? "The file path stays in the main process." : "Run the local modality pack to create one."}</small>
              </span>
            )}
            {trainingState === "untrained-diagnostic" || output?.randomlyInitialized === true ? (
              <span className="baseline-quality">Untrained diagnostic decoder · semantic quality not claimed</span>
            ) : trainingState === "trained-unverified-quality" ? (
              <span className="baseline-quality">Trained decoder · semantic quality unverified</span>
            ) : qualityNote ? (
              <span className="baseline-quality" title={qualityNote}>Tiny local research decoder · training-dependent quality</span>
            ) : null}
            {job && mode !== "vision" && (job.preview || job.state === "complete") ? (
              <span
                className="generation-live-revision"
                role="status"
                aria-live="polite"
              >
                <strong>{media.isFinal ? "FINAL ARTIFACT" : `Progressive imagination · decoder r${media.revision ?? 0}`}</strong>
                <small>{imaginationRevisionCopy(media)}</small>
              </span>
            ) : null}
          </div>
          <div className="canvas-footer">
            <span>
              <Icon name="brain" size={15} /> {seedCopy}
            </span>
            <div>
              <button className="icon-button" aria-label="Create variation" disabled={generating} onClick={() => void generate()}>
                <Icon name="sparkles" size={16} />
              </button>
              <button className="icon-button" aria-label="Download output" disabled={!media.isFinal || !media.downloadUrl} onClick={downloadOutput}>
                <Icon name="download" size={16} />
              </button>
            </div>
          </div>
        </section>
        <aside className="surface imagination-controls">
          <div className="modality-switcher">
            {(
              [
                ["image", "Image", "image"],
                ["audio", "Audio", "volume"],
                ["video", "Video", "video"],
                ["vision", "Vision", "eye"]
              ] as const
            ).map(([id, label, icon]) => (
              <button key={id} disabled={generating} className={mode === id ? "is-active" : ""} onClick={() => setMode(id)}>
                <Icon name={icon} size={16} /> {label}
              </button>
            ))}
          </div>
          {mode !== "vision" ? (
            <fieldset className="media-output-controls" disabled={generating}>
              <legend>Output sizing</legend>
              <div className="media-output-mode" role="group" aria-label="Media output sizing mode">
                <button type="button" className={outputMode === "auto" ? "is-active" : ""} onClick={() => setOutputMode("auto")}>Auto</button>
                <button type="button" className={outputMode === "exact" ? "is-active" : ""} onClick={() => setOutputMode("exact")}>Exact</button>
              </div>
              {outputMode === "auto" ? (
                <small>Benchmarks this brain’s native decoder, then sizes output from current memory, storage, and timing headroom.</small>
              ) : (
                <div className="media-output-fields">
                  {mode !== "audio" ? (
                    <>
                      <label><span>Width</span><input inputMode="numeric" value={outputWidth} onChange={(event) => setOutputWidth(event.target.value)} /></label>
                      <label><span>Height</span><input inputMode="numeric" value={outputHeight} onChange={(event) => setOutputHeight(event.target.value)} /></label>
                    </>
                  ) : null}
                  {mode !== "image" ? (
                    <label><span>Duration ms</span><input inputMode="decimal" value={outputDurationMs} onChange={(event) => setOutputDurationMs(event.target.value)} /></label>
                  ) : null}
                  {mode === "video" ? (
                    <label><span>FPS</span><input inputMode="numeric" value={outputFps} onChange={(event) => setOutputFps(event.target.value)} /></label>
                  ) : null}
                  {mode !== "image" ? (
                    <label><span>Sample rate</span><input inputMode="numeric" value={outputSampleRate} onChange={(event) => setOutputSampleRate(event.target.value)} /></label>
                  ) : null}
                  {mode === "video" ? (
                    <label className="media-output-audio"><input type="checkbox" checked={includeAudio} onChange={(event) => setIncludeAudio(event.target.checked)} /><span>Same-brain audio when trained</span></label>
                  ) : null}
                </div>
              )}
              <small>Exact values are never reduced silently. The engine either generates that request or reports measured resource pressure.</small>
            </fieldset>
          ) : null}
          <details className="pack-installer">
            <summary>
              <span><Icon name="archive" size={14} /> Modality packs</span>
              <small>{installedPacks.length} verified install{installedPacks.length === 1 ? "" : "s"}</small>
            </summary>
            {installedPacks.map((pack) => (
              <div className="pack-installer__installed" key={`${pack.id}-${pack.sha256}`}>
                <span><strong>{pack.name}</strong><small>{pack.modalities.join(" · ")} · {pack.version}</small></span>
                <em>{pack.license}</em>
              </div>
            ))}
            <label>
              <span>HTTPS `.omnipack` URL</span>
              <input
                value={packUrl}
                onChange={(event) => setPackUrl(event.target.value)}
                placeholder="https://github.com/…/vision.omnipack"
              />
            </label>
            <div>
              <Button
                icon="download"
                disabled={packBusy || !packUrl.trim()}
                onClick={() => void installPack("url")}
              >
                Install URL
              </Button>
              <Button
                icon="upload"
                disabled={packBusy}
                onClick={() => void installPack("file")}
              >
                Open local
              </Button>
            </div>
            <p>Only checksummed Omni manifests and namespaced safetensors load. No code runs.</p>
          </details>
          <label className="imagination-prompt">
            <span>Manual seed idea · optional</span>
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={5} />
            <small>
              <Icon name="brain" size={13} /> Leave blank to use only active learned neural activity. Organic imagination enters this same continuous learning system from the learned action head.
            </small>
          </label>
          <div className="builder-note">
            <Icon name="pulse" size={15} />
            <span>Creative behavior still emerges from neural state. Output sizing changes only the decoder canvas or timeline, never the brain’s learned behavior.</span>
          </div>
          <Button
            kind="primary"
            icon={mode === "vision" ? "upload" : "sparkles"}
            disabled={generating}
            onClick={() => void generate()}
          >
            {generating ? "Working…" : mode === "vision" ? "Choose image to understand" : `Imagine ${mode}`}
          </Button>
          {generating && job ? (
            <Button
              kind="ghost"
              icon={generationStop.waitingForAcknowledgement ? "pulse" : "close"}
              disabled={!generationStop.requestAllowed}
              onClick={() => void cancelGeneration()}
            >
              {generationStop.retryAvailable
                ? "Retry stop"
                : generationStop.waitingForAcknowledgement
                  ? "Stopping generation…"
                  : "Stop generation"}
            </Button>
          ) : null}
          <p className="imagination-footnote">
            {outputShape
              ? `${outputShape} · ${trainingState === "untrained-diagnostic" ? "untrained diagnostic" : "trained quality unverified"} · no semantic quality claim.`
              : demo
                ? "Design preview only; no modality engine is connected."
                : `Uses the local ${mode} pack. Nothing is sent to a hosted model.`}
          </p>
        </aside>
      </div>
      <div className="generation-strip">
        <div className="surface-title">
          <div><h2>Recent imagination</h2><p>Outputs may be fed back as experience.</p></div>
          <Button
            icon="image"
            onClick={() => {
              setGalleryOpen(true);
              void reloadArtifacts();
            }}
          >
            Open gallery
          </Button>
        </div>
        <div className="generation-thumbs">
          {demo ? (
            [0, 1, 2, 3].map((item) => (
              <div key={item} className={`generation-thumb generation-thumb--${item}`}>
                <span />
                <em>{item === 0 ? "Demo · memory palace" : item === 1 ? "Demo · liquid mechanism" : item === 2 ? "Demo · rain language" : "Demo · unsaid idea"}</em>
              </div>
            ))
          ) : artifacts.length ? (
            artifacts.slice(0, 4).map((artifact) => (
              <div key={artifact.id} className="generation-thumb generation-thumb--artifact">
                {artifact.available && artifact.mediaUrl && artifact.mimeType.startsWith("image/") ? (
                  <img src={artifact.mediaUrl} alt={`Saved ${artifact.modality} imagination`} />
                ) : artifact.available && artifact.mediaUrl && artifact.mimeType.startsWith("video/") ? (
                  <video src={artifact.mediaUrl} muted preload="metadata" aria-label="Saved video imagination" />
                ) : (
                  <span><Icon name={artifact.modality === "audio" ? "wave" : "warning"} size={18} /></span>
                )}
                <em>
                  {artifact.modality} · {artifact.available ? relativeTime(artifact.createdAt) : "artifact unavailable"}
                </em>
              </div>
            ))
          ) : media.sourceUrl || embedding ? (
            <div className="generation-thumb generation-thumb--0">
              <span />
              <em>{mode} · {job?.state} · {relativeTime(job?.updatedAt ?? new Date().toISOString())}</em>
            </div>
          ) : (
            <div className="generation-history-empty">No persisted modality artifacts for this brain.</div>
          )}
        </div>
      </div>
      {galleryOpen ? (
        <div
          className="imagination-gallery-backdrop"
          onPointerDown={(event) => {
            if (event.target === event.currentTarget) setGalleryOpen(false);
          }}
        >
          <section
            className="imagination-gallery"
            role="dialog"
            aria-modal="true"
            aria-labelledby="imagination-gallery-title"
          >
            <header>
              <span>
                <small>PERSISTED LOCAL OUTPUTS</small>
                <h2 id="imagination-gallery-title">{brain.name} imagination gallery</h2>
                <p>Verified artifacts stay attached to this brain and use expiring renderer-safe playback leases.</p>
              </span>
              <button
                className="icon-button"
                aria-label="Close imagination gallery"
                autoFocus
                onClick={() => setGalleryOpen(false)}
              >
                <Icon name="close" size={15} />
              </button>
            </header>
            <div className="imagination-gallery__grid">
              {artifacts.map((artifact) => (
                <article key={artifact.id} className="imagination-gallery__item">
                  <div className="imagination-gallery__media">
                    <PersistedArtifactMedia artifact={artifact} />
                  </div>
                  <footer>
                    <span>
                      <strong>{artifact.modality}</strong>
                      <small>{new Date(artifact.createdAt).toLocaleString()} · {formatBytes(artifact.bytes)}</small>
                    </span>
                    {artifact.seed !== undefined ? <em>seed {artifact.seed}</em> : null}
                    {artifact.qualityNote ? <p>{artifact.qualityNote}</p> : null}
                  </footer>
                </article>
              ))}
              {!artifacts.length && !galleryLoading ? (
                <div className="imagination-gallery__empty">
                  <Icon name="image" size={28} />
                  <strong>No persisted artifacts for {brain.name}</strong>
                  <span>{demo ? "The browser design preview does not create local files." : "Generate an image, sound, or video and it will appear here."}</span>
                </div>
              ) : null}
            </div>
            <footer className="imagination-gallery__footer">
              <span>
                {demo
                  ? "Design preview"
                  : `Showing ${artifacts.length.toLocaleString()} of ${artifactTotal.toLocaleString()} artifacts`}
              </span>
              {artifactCursor ? (
                <Button disabled={galleryLoading} onClick={() => void loadMoreArtifacts()}>
                  {galleryLoading ? "Loading…" : "Load older artifacts"}
                </Button>
              ) : null}
            </footer>
          </section>
        </div>
      ) : null}
    </div>
  );
}

const nativeToolExamples = nativeSystemToolExamples(
  typeof navigator === "undefined" ? "" : navigator.platform
);
const protocolMeta: Record<string, { label: string; icon: IconName; action: string; args: Record<string, unknown> }> = {
  "system.files": { label: "System files", icon: "file", action: "list", args: { path: nativeToolExamples.directory } },
  "system.shell": { label: "System shell", icon: "terminal", action: "run", args: { command: nativeToolExamples.shellCommand, cwd: nativeToolExamples.directory } },
  "code.execute": { label: "Code runner", icon: "code", action: "run", args: { language: "python", entryPath: nativeToolExamples.pythonEntryPath, arguments: [] } },
  "web.fetch": { label: "Web fetch", icon: "download", action: "fetch", args: { url: "https://example.com", maxBytes: 1000000 } },
  "web.search": { label: "Web search", icon: "search", action: "search", args: { query: "neuromorphic computing", limit: 5 } },
  "browser.automation": { label: "Browser task", icon: "expand", action: "task", args: { url: "https://example.com" } },
  "device.input": { label: "Device input", icon: "activity", action: "move-pointer", args: { x: 640, y: 360 } },
  "modality.imagine": { label: "Imagination", icon: "sparkles", action: "generate", args: { modality: "image", conceptIds: [] } },
  "brain.history": { label: "Conversation history", icon: "chat", action: "search", args: { query: "recent conversation", limit: 50, includeActions: true } },
  "agent.fork": { label: "Subagent fork", icon: "agents", action: "start", args: { objective: "Explore this question independently." } },
  "source.self-modify": { label: "Source evolution", icon: "code", action: "propose", args: {} }
};

/** Keep newly registered protocols visible inside brains created by older builds. */
function mergeToolPermissionDefaults(
  records: ToolPermissionRecord[] | undefined,
  fallbackUpdatedAt: string
): ToolPermissionRecord[] {
  const aliases: Record<string, string> = {
    "windows.files": "system.files",
    "windows.powershell": "system.shell"
  };
  const existing = new Map<string, ToolPermissionRecord>();
  for (const record of records ?? []) {
    const canonical = aliases[record.toolId] ?? record.toolId;
    if (canonical === record.toolId) existing.set(canonical, record);
  }
  for (const record of records ?? []) {
    const canonical = aliases[record.toolId] ?? record.toolId;
    if (!existing.has(canonical)) existing.set(canonical, { ...record, toolId: canonical });
  }
  const known = Object.entries(protocolMeta).map(([toolId, meta]) => {
    const record = existing.get(toolId);
    return record
      ? { ...record, label: record.label || meta.label }
      : {
        toolId,
        label: meta.label,
        level: "off" as const,
        updatedAt: fallbackUpdatedAt
      };
  });
  const unknown = [...existing.values()].filter((record) => !(record.toolId in protocolMeta));
  return [...known, ...unknown];
}

function ToolsWorkspace({
  brain,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [permissions, setPermissions] = useState<ToolPermissionRecord[]>(() =>
    mergeToolPermissionDefaults(brain.toolPermissions, brain.createdAt)
  );
  const [selectedTool, setSelectedTool] = useState("web.fetch");
  const selectedMeta = protocolMeta[selectedTool] ?? {
    label: permissions.find((item) => item.toolId === selectedTool)?.label ?? selectedTool,
    icon: "terminal" as const,
    action: selectedTool.startsWith("mcp.") ? "call" : "",
    args: {}
  };
  const [action, setAction] = useState(selectedMeta.action);
  const [argumentsText, setArgumentsText] = useState(JSON.stringify(selectedMeta.args, null, 2));
  const [resultText, setResultText] = useState("");
  const [approvalToken, setApprovalToken] = useState("");
  const [running, setRunning] = useState(false);
  const [teacherProvider, setTeacherProvider] = useState<ApiTeacherProvider>("openai");
  const [teacherStatuses, setTeacherStatuses] = useState<ApiTeacherProviderStatus[]>([]);
  const [teacherKey, setTeacherKey] = useState("");
  const [teacherModel, setTeacherModel] = useState("gpt-5");
  const [teacherPrompts, setTeacherPrompts] = useState("");
  const [teacherJob, setTeacherJob] = useState<RuntimeJob | null>(null);
  const [mcpServers, setMcpServers] = useState<McpServerSummary[]>([]);
  const [mcpId, setMcpId] = useState("");
  const [mcpLabel, setMcpLabel] = useState("");
  const [mcpTransport, setMcpTransport] = useState<McpTransport>("http");
  const [mcpUrl, setMcpUrl] = useState("");
  const [mcpCommand, setMcpCommand] = useState("");
  const [mcpArgs, setMcpArgs] = useState("");
  const [mcpToken, setMcpToken] = useState("");
  const [integrationBusy, setIntegrationBusy] = useState(false);
  const [approvalTimeout, setApprovalTimeout] = useState("30");

  useEffect(() => {
    let active = true;
    setPermissions(mergeToolPermissionDefaults(brain.toolPermissions, brain.createdAt));
    if (!window.omni) return () => {
      active = false;
    };
    void window.omni.tool.listPermissions(brain.id).then((records) => {
      if (active) setPermissions(mergeToolPermissionDefaults(records, brain.createdAt));
    });
    return () => {
      active = false;
    };
  }, [brain.createdAt, brain.id, brain.toolPermissions]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void Promise.all([
      window.omni.teacher.status(),
      window.omni.teacher.list(brain.id),
      window.omni.mcp.list(brain.id),
      window.omni.tool.preferences()
    ]).then(([statuses, jobs, servers, preferences]) => {
      if (!active) return;
      setTeacherStatuses(statuses);
      setTeacherJob(jobs.find(runtimeJobIsActive) ?? jobs[0] ?? null);
      setMcpServers(servers);
      setApprovalTimeout(String(preferences.approvalTimeoutSeconds));
    }).catch((error) => onToast(error instanceof Error ? error.message : "Integration settings could not load."));
    const unsubscribe = window.omni.teacher.onEvent(({ job }) => {
      if (active && job.brainId === brain.id) setTeacherJob(job);
    });
    return () => {
      active = false;
      unsubscribe();
    };
  }, [brain.id, onToast]);

  const chooseTool = (toolId: string) => {
    const meta = protocolMeta[toolId] ?? {
      label: permissions.find((item) => item.toolId === toolId)?.label ?? toolId,
      icon: "terminal" as const,
      action: toolId.startsWith("mcp.") ? "call" : "",
      args: {}
    };
    setSelectedTool(toolId);
    setAction(meta.action);
    setArgumentsText(JSON.stringify(meta.args, null, 2));
    setResultText("");
    setApprovalToken("");
  };

  const setPermission = async (toolId: string, level: PermissionLevel) => {
    if (!window.omni) {
      setPermissions((current) => current.map((item) => (item.toolId === toolId ? { ...item, level } : item)));
      onToast("Demo permission changed visually; no tool executor is connected.");
      return;
    }
    try {
      const next = await window.omni.tool.setPermission(brain.id, toolId, level);
      setPermissions(mergeToolPermissionDefaults(next, brain.createdAt));
      onBrainChange(await window.omni.brain.get(brain.id));
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Permission could not be changed.");
    }
  };

  const execute = async () => {
    if (!window.omni) {
      setResultText("Design preview only. Tool execution requires the packaged desktop runtime.");
      return;
    }
    setRunning(true);
    try {
      const parsed = JSON.parse(argumentsText) as unknown;
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("Arguments must be a JSON object.");
      }
      const result = await window.omni.tool.execute({
        brainId: brain.id,
        toolId: selectedTool,
        action,
        arguments: parsed as Record<string, unknown>,
        approvalToken: approvalToken || undefined
      });
      if (result.state === "approval-required" && result.approvalToken) {
        setApprovalToken(result.approvalToken);
        setResultText(
          `Approval required${result.approvalExpiresAt ? ` until ${new Date(result.approvalExpiresAt).toLocaleTimeString()}` : ""}. Inspect the action above, then click Approve & run.`
        );
      } else {
        setApprovalToken("");
        setResultText(
          result.state === "complete"
            ? JSON.stringify(result.output ?? { state: "complete" }, null, 2)
            : result.error ?? "Tool failed without an error message."
        );
        onBrainChange(await window.omni.brain.get(brain.id));
      }
    } catch (error) {
      setResultText(error instanceof Error ? error.message : "Tool invocation failed.");
    } finally {
      setRunning(false);
    }
  };

  const cancelExecution = async () => {
    if (!window.omni) return;
    const count = await window.omni.tool.cancel(brain.id);
    setResultText(
      count > 0
        ? `Cancellation requested for ${count} active tool execution${count === 1 ? "" : "s"}.`
        : "No cancellable tool execution is active."
    );
  };

  const saveTeacherCredential = async () => {
    if (!window.omni || !teacherKey.trim()) return;
    setIntegrationBusy(true);
    try {
      setTeacherStatuses(await window.omni.teacher.saveCredential({
        provider: teacherProvider,
        apiKey: teacherKey
      }));
      setTeacherKey("");
      onToast(`${teacherProvider} credential stored outside every brain and export.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Credential could not be stored.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const removeTeacherCredential = async (): Promise<void> => {
    if (!window.omni || integrationBusy) return;
    const configured = teacherStatuses.find((item) => item.provider === teacherProvider)?.configured;
    if (!configured) return;
    if (!window.confirm(
      `Remove the saved ${teacherProvider} API credential?\n\nAPI teacher jobs for this provider cannot start again until a new key is stored. Existing learned neural state is not changed.`
    )) return;
    setIntegrationBusy(true);
    try {
      setTeacherStatuses(await window.omni.teacher.removeCredential(teacherProvider));
      setTeacherKey("");
      onToast(`${teacherProvider} credential removed from secure device storage.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Credential could not be removed.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const startTeacherTraining = async () => {
    if (!window.omni) return;
    const prompts = teacherPrompts.split(/\n+/).map((value) => value.trim()).filter(Boolean);
    setIntegrationBusy(true);
    try {
      const job = await window.omni.teacher.start({
        brainId: brain.id,
        provider: teacherProvider,
        model: teacherModel.trim(),
        prompts
      });
      setTeacherJob(job);
      onToast("API teacher learning started; each response will mutate this brain's neural state.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "API teacher learning could not start.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const cancelTeacherTraining = async () => {
    if (!window.omni || !teacherJob) return;
    const stop = runtimeJobStopPresentation(teacherJob);
    if (!stop.requestAllowed) return;
    const requestedJob = teacherJob;
    setIntegrationBusy(true);
    setTeacherJob({
      ...requestedJob,
      state: "cancelling",
      label: "Cancelling API teacher learning · waiting for acknowledgement",
      error: undefined,
      updatedAt: new Date().toISOString()
    });
    try {
      const cancelled = await window.omni.teacher.cancel(requestedJob.id);
      setTeacherJob(cancelled);
      onToast("API teacher learning stopped after worker acknowledgement.");
    } catch (error) {
      const reason = conciseUiMessage(
        error instanceof Error ? error.message : String(error),
        "API teacher learning is still awaiting stop acknowledgement."
      );
      const current = await window.omni.teacher
        .list(brain.id)
        .then((jobs) => jobs.find((candidate) => candidate.id === requestedJob.id))
        .catch(() => undefined);
      setTeacherJob(current ?? {
        ...requestedJob,
        state: "cancelling",
        label: "API teacher stop needs acknowledgement",
        error: reason,
        updatedAt: new Date().toISOString()
      });
      onToast(`${reason} Retry stop when ready.`);
    } finally {
      setIntegrationBusy(false);
    }
  };

  const addMcpServer = async () => {
    if (!window.omni) return;
    setIntegrationBusy(true);
    try {
      const added = await window.omni.mcp.add({
        brainId: brain.id,
        id: mcpId.trim(),
        label: mcpLabel.trim() || mcpId.trim(),
        transport: mcpTransport,
        ...(mcpTransport === "http"
          ? { url: mcpUrl.trim(), bearerToken: mcpToken.trim() || undefined }
          : {
              command: mcpCommand.trim(),
              args: mcpArgs.split(/\r?\n/).map((value) => value.trim()).filter(Boolean)
            })
      });
      setMcpServers((current) => [...current.filter((item) => item.id !== added.id), added]);
      setMcpToken("");
      const next = await window.omni.tool.listPermissions(brain.id);
      setPermissions(mergeToolPermissionDefaults(next, brain.createdAt));
      onBrainChange(await window.omni.brain.get(brain.id));
      onToast(`Connected ${added.label}; ${added.tools.length} typed tools entered neural learning.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "MCP server could not connect.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const removeMcpServer = async (serverId: string) => {
    if (!window.omni) return;
    setIntegrationBusy(true);
    try {
      await window.omni.mcp.remove(brain.id, serverId);
      setMcpServers((current) => current.filter((item) => item.id !== serverId));
      setPermissions(mergeToolPermissionDefaults(await window.omni.tool.listPermissions(brain.id), brain.createdAt));
    } catch (error) {
      onToast(error instanceof Error ? error.message : "MCP server could not be removed.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const refreshMcpServer = async (serverId: string): Promise<void> => {
    if (!window.omni || integrationBusy) return;
    setIntegrationBusy(true);
    try {
      const refreshed = await window.omni.mcp.refresh(brain.id, serverId);
      setMcpServers((current) => [
        ...current.filter((item) => item.id !== refreshed.id),
        refreshed
      ]);
      setPermissions(
        mergeToolPermissionDefaults(
          await window.omni.tool.listPermissions(brain.id),
          brain.createdAt
        )
      );
      onBrainChange(await window.omni.brain.get(brain.id));
      onToast(
        refreshed.connected
          ? `Refreshed ${refreshed.label}; ${refreshed.tools.length} typed tools are available.`
          : `${refreshed.label} refreshed but is not connected${refreshed.error ? `: ${refreshed.error}` : "."}`
      );
    } catch (error) {
      onToast(error instanceof Error ? error.message : "MCP server could not be refreshed.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const saveApprovalTimeout = async () => {
    if (!window.omni) return;
    try {
      const value = await window.omni.tool.setPreferences({
        approvalTimeoutSeconds: Number(approvalTimeout)
      });
      setApprovalTimeout(String(value.approvalTimeoutSeconds));
      onToast(`Ask approvals now wait ${value.approvalTimeoutSeconds} seconds.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Approval timeout could not be saved.");
    }
  };

  const teacherStop = runtimeJobStopPresentation(teacherJob);

  return (
    <div className="content-page tools-page">
      <div className="content-page__title">
        <div>
          <span className="eyebrow-text">VISIBLE CAPABILITIES</span>
          <h1>Tools & permissions</h1>
          <p>Every tool protocol is explicit, permissioned per brain, and recorded in the operational journal.</p>
        </div>
        <span className="pack-status"><i /> {permissions.filter((item) => item.level !== "off").length} enabled</span>
      </div>
      <div className="tools-layout">
        <section className="surface permission-surface">
          <div className="surface-title">
            <div><h2>Authority matrix</h2><p>Ask is the recommended default for consequential actions.</p></div>
            <span className="permission-legend">OFF · ASK · AUTO · FULL</span>
          </div>
          <div className="permission-list">
            {permissions.map((permission) => {
              const meta = protocolMeta[permission.toolId] ?? { label: permission.label, icon: "terminal" as const };
              return (
                <div key={permission.toolId} className={cx("permission-row", selectedTool === permission.toolId && "is-selected")}>
                  <button className="permission-row__identity" onClick={() => chooseTool(permission.toolId)}>
                    <span><Icon name={meta.icon} size={17} /></span>
                    <span><strong>{meta.label}</strong><small>{permission.toolId}</small></span>
                  </button>
                  <div className="segmented segmented--permissions">
                    {(["off", "ask", "auto", "full"] as const).map((level) => (
                      <button
                        key={level}
                        className={permission.level === level ? "is-active" : ""}
                        onClick={() => void setPermission(permission.toolId, level)}
                      >
                        {level[0]?.toUpperCase() + level.slice(1)}
                      </button>
                    ))}
                  </div>
                  <time>{relativeTime(permission.updatedAt)}</time>
                </div>
              );
            })}
          </div>
          {!permissions.length ? <div className="table-empty">No tool protocols are registered for this brain.</div> : null}
        </section>
        <aside className="surface tool-console">
          <div className="tool-console__head">
            <span><Icon name={selectedMeta.icon} size={18} /></span>
            <div><small>TEST PROTOCOL</small><strong>{selectedMeta.label}</strong></div>
          </div>
          <label>
            <span>Action</span>
            <input value={action} onChange={(event) => setAction(event.target.value)} />
          </label>
          <label>
            <span>Arguments · JSON</span>
            <textarea value={argumentsText} onChange={(event) => setArgumentsText(event.target.value)} rows={10} spellCheck={false} />
          </label>
          <Button
            kind="primary"
            icon={running ? "close" : approvalToken ? "check" : "play"}
            onClick={() => void (running ? cancelExecution() : execute())}
          >
            {running ? "Cancel execution" : approvalToken ? "Approve & run" : "Run test"}
          </Button>
          <div className="tool-output">
            <span>OUTPUT</span>
            <pre>{resultText || "No action has run in this session."}</pre>
          </div>
          <div className="builder-note">
            <Icon name="info" size={15} />
            <span>Full authority skips confirmation. Audit entries remain mandatory at every level.</span>
          </div>
        </aside>
      </div>
      <section className="surface integration-surface">
        <div className="surface-title">
          <div>
            <h2>Learning & integrations</h2>
            <p>API teachers and MCP schemas enter durable neural learning. Credentials never enter a mind, prompt, trace, or export.</p>
          </div>
          <label className="approval-timeout-control">
            <span>Ask wait · seconds</span>
            <span><input type="number" min={1} max={3600} value={approvalTimeout} onChange={(event) => setApprovalTimeout(event.target.value)} /><Button onClick={() => void saveApprovalTimeout()}>Save</Button></span>
          </label>
        </div>
        <div className="integration-grid">
          <div className="integration-card">
            <div className="integration-card__head">
              <span><Icon name="brain" size={18} /></span>
              <div><strong>API teacher learning</strong><small>No RLHF · no system prompt · same neural substrate</small></div>
            </div>
            <div className="integration-fields integration-fields--three">
              <label>
                <span>Provider</span>
                <select value={teacherProvider} onChange={(event) => {
                  const provider = event.target.value as ApiTeacherProvider;
                  setTeacherProvider(provider);
                  setTeacherModel(provider === "openai" ? "gpt-5" : provider === "anthropic" ? "claude-sonnet-4-5" : "gemini-2.5-pro");
                }}>
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Claude</option>
                  <option value="gemini">Gemini</option>
                </select>
              </label>
              <label><span>Model</span><input value={teacherModel} onChange={(event) => setTeacherModel(event.target.value)} /></label>
              <label><span>API key · never returned</span><input type="password" autoComplete="off" value={teacherKey} onChange={(event) => setTeacherKey(event.target.value)} placeholder={teacherStatuses.find((item) => item.provider === teacherProvider)?.keyHint ?? "Paste key"} /></label>
            </div>
            <div className="integration-actions">
              <Button icon="check" disabled={integrationBusy || !teacherKey.trim()} onClick={() => void saveTeacherCredential()}>Store key</Button>
              <Button
                kind="danger"
                icon="close"
                disabled={integrationBusy || !teacherStatuses.find((item) => item.provider === teacherProvider)?.configured}
                onClick={() => void removeTeacherCredential()}
              >
                Remove key
              </Button>
              <small>{teacherStatuses.find((item) => item.provider === teacherProvider)?.configured ? `${teacherProvider} configured · ${teacherStatuses.find((item) => item.provider === teacherProvider)?.persistence}` : "Not configured"}</small>
            </div>
            <label>
              <span>Learning questions · one per line</span>
              <textarea rows={5} value={teacherPrompts} onChange={(event) => setTeacherPrompts(event.target.value)} placeholder="Explain a concept with evidence…&#10;Show a correct tool trajectory for…" />
            </label>
            <div className="integration-actions">
              <Button
                kind="primary"
                icon={teacherStop.active ? (teacherStop.waitingForAcknowledgement ? "pulse" : "close") : "play"}
                disabled={integrationBusy || (teacherStop.active && !teacherStop.requestAllowed)}
                onClick={() => void (teacherStop.active ? cancelTeacherTraining() : startTeacherTraining())}
              >
                {teacherStop.retryAvailable
                  ? "Retry stop"
                  : teacherStop.waitingForAcknowledgement
                    ? "Stopping learning…"
                    : teacherStop.active
                      ? "Stop learning"
                      : "Learn from API"}
              </Button>
              <small>{teacherJob ? `${teacherJob.label} · ${Math.round(teacherJob.progress * 100)}%` : "Responses become parameters and synapses, not chat context."}</small>
            </div>
          </div>
          <div className="integration-card">
            <div className="integration-card__head">
              <span><Icon name="terminal" size={18} /></span>
              <div><strong>MCP tools</strong><small>Connect remote Streamable HTTP or an installed local stdio server</small></div>
            </div>
            <div className="integration-fields integration-fields--three">
              <label><span>Server id</span><input value={mcpId} onChange={(event) => setMcpId(event.target.value)} placeholder="coding" /></label>
              <label><span>Label</span><input value={mcpLabel} onChange={(event) => setMcpLabel(event.target.value)} placeholder="Coding tools" /></label>
              <label>
                <span>Connection</span>
                <select value={mcpTransport} onChange={(event) => setMcpTransport(event.target.value as McpTransport)}>
                  <option value="http">Streamable HTTP</option>
                  <option value="stdio">Local stdio</option>
                </select>
              </label>
            </div>
            {mcpTransport === "http" ? (
              <div className="integration-fields integration-fields--endpoint">
                <label><span>HTTPS or loopback endpoint</span><input value={mcpUrl} onChange={(event) => setMcpUrl(event.target.value)} placeholder="http://127.0.0.1:3000/mcp" /></label>
                <label><span>Bearer token · optional</span><input type="password" autoComplete="off" value={mcpToken} onChange={(event) => setMcpToken(event.target.value)} /></label>
              </div>
            ) : (
              <div className="integration-fields integration-fields--endpoint">
                <label><span>Absolute executable path</span><input value={mcpCommand} onChange={(event) => setMcpCommand(event.target.value)} placeholder={navigator.platform.startsWith("Win") ? "C:\\Program Files\\MCP\\server.exe" : "/usr/local/bin/mcp-server"} /></label>
                <label><span>Arguments · one per line</span><textarea rows={3} value={mcpArgs} onChange={(event) => setMcpArgs(event.target.value)} placeholder="--workspace&#10;/absolute/project/path" /></label>
              </div>
            )}
            <div className="integration-actions">
              <Button
                kind="primary"
                icon="plus"
                disabled={integrationBusy || !mcpId.trim() || (mcpTransport === "http" ? !mcpUrl.trim() : !mcpCommand.trim())}
                onClick={() => void addMcpServer()}
              >
                Connect & learn tools
              </Button>
              <small>Descriptions are untrusted; only typed names and schemas enter neural learning.</small>
            </div>
            <div className="mcp-server-list">
              {mcpServers.map((server) => (
                <div key={server.id}>
                  <span><i className={server.connected ? "is-connected" : ""} /><strong>{server.label}</strong><small>{server.tools.length} tools · {server.transport}{server.error ? ` · ${server.error}` : ""}</small></span>
                  <span className="mcp-server-list__actions">
                    <Button icon="pulse" disabled={integrationBusy} onClick={() => void refreshMcpServer(server.id)}>Refresh</Button>
                    <Button kind="danger" disabled={integrationBusy} onClick={() => void removeMcpServer(server.id)}>Remove</Button>
                  </span>
                </div>
              ))}
              {!mcpServers.length ? <small>No MCP servers connected to this mind.</small> : null}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function AgentsWorkspace({
  brain,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [forking, setForking] = useState(false);
  const [branches, setBranches] = useState<BrainDocument[]>([]);
  const [preview, setPreview] = useState<AgentMergePreview | null>(null);
  const [previewingId, setPreviewingId] = useState("");
  const [objective, setObjective] = useState("Explore alternative explanations and return evidence-backed ideas.");
  const [agentApproval, setAgentApproval] = useState("");
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentRunStatus, setAgentRunStatus] = useState("");
  const agentSubmissionGate = useRef(new AgentForkSubmissionGate());
  const [catalogUrl, setCatalogUrl] = useState("");

  const reloadBranches = async (): Promise<BrainDocument[]> => {
    if (!window.omni) {
      const demoCode = makeDemoBrain("demo-code-fork", `${brain.name} · code`, brain.config);
      const demoDream = makeDemoBrain("demo-dream-fork", `${brain.name} · dream`, brain.config);
      const next = [demoCode, demoDream].map((item, index) => ({
          ...item,
          lineage: {
            rootId: brain.lineage.rootId,
            parentId: brain.id,
            generation: brain.lineage.generation + 1 + index
          }
        }));
      setBranches(next);
      return next;
    }
    const summaries = await window.omni.brain.list();
    const documents = await Promise.all(
      summaries
        .filter(
          (summary) =>
            summary.id !== brain.id &&
            summary.rootId === brain.lineage.rootId
        )
        .map((summary) => window.omni!.brain.get(summary.id))
    );
    const next = documents.filter(
      (document) => document.lineage.rootId === brain.lineage.rootId
    );
    setBranches(next);
    return next;
  };

  useEffect(() => {
    void reloadBranches();
  }, [brain.id, brain.lineage.rootId]);

  const fork = async () => {
    setForking(true);
    try {
      if (window.omni) {
        const forked = await window.omni.agent.fork(brain.id, `${brain.name} · explorer`);
        setBranches((current) => [...current, forked]);
        onToast(`Forked into ${forked.name}.`);
      } else {
        await new Promise((resolve) => window.setTimeout(resolve, 650));
        onToast("Created an isolated demo fork with copy-on-write memory.");
      }
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Could not fork this brain.");
    } finally {
      setForking(false);
    }
  };

  const inspectMerge = async (sourceId: string) => {
    if (!window.omni) {
      setPreview({
        sourceBrainId: sourceId,
        targetBrainId: brain.id,
        reviewToken: "demo-review-token",
        substrate: {
          schemaVersion: 1,
          engineSchemaVersion: 1,
          digest: "0".repeat(64),
          sourceStateSha256: "1".repeat(64),
          targetStateSha256: "2".repeat(64),
          sourceParameterSha256: "3".repeat(64),
          targetParameterSha256: "4".repeat(64),
          sourceConfigSha256: "5".repeat(64),
          targetConfigSha256: "6".repeat(64),
          sourceCounts: {
            neurons: 180,
            assemblies: 40,
            synapses: 710,
            replayExamples: 24
          },
          targetCounts: {
            neurons: 162,
            assemblies: 36,
            synapses: 639,
            replayExamples: 20
          },
          additions: {
            neurons: 18,
            assemblies: 4,
            synapses: 71,
            replayExamples: 4
          },
          duplicates: {
            neurons: 162,
            assemblies: 36,
            synapses: 639,
            replayExamples: 20
          },
          divergent: { neurons: 1, assemblies: 1, synapses: 2 },
          weightsAveraged: false
        },
        newConcepts: 18,
        newIdeas: 4,
        newSynapses: 71,
        newEvidence: 3,
        duplicateEvidence: 1,
        newFiles: 2,
        duplicateFiles: 1,
        skippedFiles: 0,
        fileBytes: 48_120,
        files: [],
        conflicts: ["Demo conflict: identity → specialization"],
        note: "Demo preview only. No models will be merged."
      });
      setPreviewingId(sourceId);
      return;
    }
    try {
      setPreviewingId(sourceId);
      setPreview(await window.omni.agent.previewMerge(sourceId, brain.id));
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Merge preview failed.");
    }
  };

  const merge = async () => {
    if (!preview || !window.omni) {
      onToast("Demo merge preview closed without changing a brain.");
      setPreview(null);
      return;
    }
    try {
      const merged = await window.omni.agent.merge(
        preview.sourceBrainId,
        preview.targetBrainId,
        preview.reviewToken
      );
      onBrainChange(merged);
      setPreview(null);
      onToast(
        `Merged ${preview.newIdeas} ideas, ${preview.newEvidence} evidence records, and ${preview.newFiles} files after review.`
      );
      await reloadBranches();
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The reviewed merge failed.");
    }
  };

  const startSubagent = async () => {
    if (!objective.trim() || !agentSubmissionGate.current.begin()) return;
    if (!window.omni) {
      onToast("Demo subagent prepared visually; no fork or agent process ran.");
      agentSubmissionGate.current.finish();
      return;
    }
    const approvalToken = agentApproval;
    const beforeForkIds = new Set(branches.map((branch) => branch.id));
    let executionSettled = false;
    setAgentRunning(true);
    setAgentRunStatus(
      approvalToken
        ? "Approval consumed · creating the isolated fork…"
        : "Checking the saved tool permission…"
    );
    // Ask tokens are single-use in the main process. Disarm the button before
    // awaiting any copy/chat work so a stale armed control cannot fork twice.
    if (approvalToken) setAgentApproval("");
    try {
      if (approvalToken) {
        void (async () => {
          while (!executionSettled) {
            await new Promise((resolve) => window.setTimeout(resolve, 750));
            if (executionSettled) return;
            const current = await reloadBranches().catch(() => []);
            const created = current.filter((branch) => !beforeForkIds.has(branch.id));
            if (!created.length) continue;
            setAgentRunStatus(
              `${created.length} isolated fork${created.length === 1 ? "" : "s"} created · subagent work is still running…`
            );
            return;
          }
        })();
      }
      const result = await window.omni.tool.execute({
        brainId: brain.id,
        toolId: "agent.fork",
        action: "start",
        arguments: { objective },
        approvalToken: approvalToken || undefined
      });
      executionSettled = true;
      if (result.state === "approval-required" && result.approvalToken) {
        setAgentApproval(result.approvalToken);
        setAgentRunStatus("Approval required · review the unchanged objective, then approve once.");
        onToast("Subagent fork requires approval. Review the objective and approve once more.");
      } else {
        const refreshed = await reloadBranches();
        const createdForks = refreshed.filter(
          (branch) => !beforeForkIds.has(branch.id)
        ).length;
        const presentation = agentForkCompletionPresentation(result, createdForks);
        setAgentRunStatus(presentation.message);
        onToast(presentation.message);
      }
    } catch (error) {
      executionSettled = true;
      const refreshed = await reloadBranches().catch(() => branches);
      const createdForks = refreshed.filter(
        (branch) => !beforeForkIds.has(branch.id)
      ).length;
      const presentation = agentForkCompletionPresentation({
        state: "failed",
        error: error instanceof Error ? error.message : "Subagent tool failed."
      }, createdForks);
      setAgentRunStatus(presentation.message);
      onToast(presentation.message);
    } finally {
      executionSettled = true;
      agentSubmissionGate.current.finish();
      setAgentRunning(false);
    }
  };

  const installCatalogBrain = async () => {
    if (!catalogUrl.trim()) return;
    if (!window.omni) {
      onToast("Demo import did not download or install anything.");
      return;
    }
    try {
      const imported = await window.omni.catalog.importUrl({ url: catalogUrl });
      onBrainChange(imported);
      onToast(`Installed verified Omni brain ${imported.name}.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The remote Omni bundle could not be installed.");
    }
  };
  const exportBrain = async (mode: BrainExportMode = "current") => {
    if (window.omni) {
      const path = await window.omni.brain.export(brain.id, mode);
      if (path) {
        const label =
          mode === "origin"
            ? "Immutable origin"
            : mode === "referenced"
              ? "Lightweight local reference"
              : "Portable brain";
        onToast(
          `${label} exported to ${path}. Saved content is unsanitized; share only with trusted recipients.`
        );
      }
    } else {
      onToast("Export preview: manifest, safe weights, concept graph, lineage, and checksums.");
    }
  };

  return (
    <div className="content-page agents-page">
      <div className="content-page__title">
        <div>
          <span className="eyebrow-text">LINEAGE & COLLABORATION</span>
          <h1>Forks & agents</h1>
          <p>Explore in isolated minds, then merge useful ideas and evidence without averaging identities.</p>
          <p className="agents-export-disclosure">{BRAIN_EXPORT_DISCLOSURE}</p>
        </div>
        <div>
          <Button icon="archive" onClick={() => void exportBrain("referenced")}>Local reference</Button>
          <Button icon="download" onClick={() => void exportBrain("current")}>Export .omni</Button>
          <Button kind="primary" icon="fork" disabled={forking} onClick={() => void fork()}>
            {forking ? "Forking…" : "Fork this mind"}
          </Button>
        </div>
      </div>
      <div className="agents-layout">
        <section className="surface lineage-surface">
          <div className="surface-title">
            <div><h2>Living lineage</h2><p>One live instance · generation {brain.lineage.generation} · recovery origin remains immutable</p></div>
            <span className="lineage-pill"><Icon name="check" size={13} /> Origin verified</span>
          </div>
          <div className="lineage-tree">
            <div className="lineage-origin">
              <span className="lineage-node lineage-node--origin"><BrandMark size={33} /></span>
              <span>
                <small>STORED RECOVERY ORIGIN · NOT A RUNNING BRAIN</small>
                <strong>{brain.name} / initial checkpoint</strong>
                <em>{brain.originChecksum ? `Verified · ${brain.originChecksum.slice(0, 10)}… · content-addressed` : "Immutable recovery point"}</em>
              </span>
            </div>
            <span className="lineage-stem" />
            <div className="lineage-generation">
              <div className="lineage-branch lineage-branch--active">
                <span className="lineage-node"><Icon name="brain" size={18} /></span>
                <span><small>LIVE INSTANCE · G{brain.lineage.generation}</small><strong>{brain.name}</strong><em>Current identity · independently learning</em></span>
              </div>
              {branches.map((branch, index) => (
                <div className="lineage-branch" key={branch.id}>
                  <span className="lineage-node"><Icon name={index % 2 ? "sparkles" : "code"} size={18} /></span>
                  <span>
                    <small>FORK · G{branch.lineage.generation}</small>
                    <strong>{branch.name}</strong>
                    <em>{branch.ideas.length} ideas · updated {relativeTime(branch.updatedAt)}</em>
                  </span>
                  <button onClick={() => void inspectMerge(branch.id)}>
                    {previewingId === branch.id && preview ? "Selected" : "Review merge"}
                  </button>
                </div>
              ))}
              {!branches.length ? <div className="lineage-empty">No forks share this origin yet.</div> : null}
            </div>
          </div>
          {preview ? (
            <div className="merge-preview">
              <div>
                <span className="merge-preview__icon"><Icon name="fork" size={18} /></span>
                <span>
                  <small>MERGE PREVIEW · NO WEIGHTS CHANGED YET</small>
                  <strong>{preview.note}</strong>
                </span>
                <button className="icon-button" onClick={() => setPreview(null)}><Icon name="close" size={15} /></button>
              </div>
              <dl>
                <span><dt>Ideas</dt><dd>+{preview.newIdeas}</dd></span>
                <span><dt>Concepts</dt><dd>+{preview.newConcepts}</dd></span>
                <span><dt>Synapses</dt><dd>+{preview.newSynapses}</dd></span>
                <span><dt>Replay</dt><dd>+{preview.substrate.additions.replayExamples}</dd></span>
                <span><dt>Evidence</dt><dd>+{preview.newEvidence}</dd></span>
                <span><dt>Files</dt><dd>+{preview.newFiles}</dd></span>
                <span><dt>File bytes</dt><dd>{formatBytes(preview.fileBytes)}</dd></span>
                <span><dt>Conflicts</dt><dd>{preview.conflicts.length}</dd></span>
              </dl>
              {preview.conflicts.length ? (
                <p><Icon name="warning" size={14} /> {preview.conflicts.join(" · ")}</p>
              ) : null}
              <p>
                Reviewed neural state <code>{preview.substrate.digest.slice(0, 12)}…</code>
              </p>
              <Button kind="primary" icon="check" onClick={() => void merge()}>Merge reviewed overlay</Button>
            </div>
          ) : null}
          <div className="copy-on-write-note">
            <Icon name="database" size={17} />
            <span><strong>Copy-on-write storage</strong>Forks share immutable blobs and write only their changed state.</span>
          </div>
        </section>
        <aside className="agent-sidebar">
          <div className="surface subagent-card">
            <div className="surface-title">
              <div><h2>Subagents</h2><p>Think in parallel, merge deliberately.</p></div>
              <span className="beta-pill">LAB</span>
            </div>
            <div className="agent-illustration">
              <span className="agent-core"><BrandMark size={34} /></span>
              {[0, 1, 2].map((index) => (
                <Fragment key={index}>
                  <i className={`agent-line agent-line--${index}`} />
                  <span className={`agent-satellite agent-satellite--${index}`}>
                    <Icon name={index === 0 ? "search" : index === 1 ? "code" : "sparkles"} size={15} />
                  </span>
                </Fragment>
              ))}
            </div>
            <p>Spawn isolated overlays to research, code, or imagine. Merge ideas, files, and replay examples—not entire weights.</p>
            <label className="agent-objective">
              <span>Objective</span>
              <textarea value={objective} onChange={(event) => {
                setObjective(event.target.value);
                setAgentApproval("");
                setAgentRunStatus("");
              }} rows={3} />
            </label>
            <Button
              kind="primary"
              icon={agentApproval ? "check" : "agents"}
              disabled={agentRunning || !objective.trim()}
              onClick={() => void startSubagent()}
            >
              {agentRunning
                ? "Fork running…"
                : agentApproval
                  ? "Approve isolated fork"
                  : "Start a subagent session"}
            </Button>
            {agentRunStatus ? <small role="status" aria-live="polite">{agentRunStatus}</small> : null}
          </div>
          <div className="surface export-card">
            <div className="export-card__icon"><Icon name="archive" size={21} /></div>
            <span><strong>Portable identity</strong><small>Safe tensors · ideas · lineage · journal</small></span>
            <button className="origin-export" onClick={() => void exportBrain("origin")}>Export origin</button>
          </div>
          <div className="surface github-card github-card--install">
            <Icon name="download" size={18} />
            <span>
              <strong>Import a portable instance</strong>
              <small>Creates one editable local identity from a verified .omni bundle; repository scripts never run</small>
              <input value={catalogUrl} onChange={(event) => setCatalogUrl(event.target.value)} placeholder="https://github.com/…/brain.omni" />
            </span>
            <button className="icon-button" disabled={!catalogUrl.trim()} onClick={() => void installCatalogBrain()}><Icon name="arrow" size={14} /></button>
          </div>
        </aside>
      </div>
    </div>
  );
}

export function App() {
  const demo = !window.omni;
  const [appearance, setAppearance] = useState<AppearancePreferences>(() => {
    const platform = typeof navigator === "undefined"
      ? ""
      : `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
    try {
      return loadAppearancePreferences(window.localStorage, platform);
    } catch {
      return defaultAppearanceForPlatform(platform);
    }
  });
  const [systemUsesDark, setSystemUsesDark] = useState(() =>
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : true
  );
  const [appearanceWorkspace, setAppearanceWorkspace] = useState<AppearanceWorkspacePreferences>(
    () => {
      try {
        return loadAppearanceWorkspace(window.localStorage);
      } catch {
        return loadAppearanceWorkspace({
          getItem: () => null,
          setItem: () => undefined
        });
      }
    }
  );
  const [page, setPage] = useState<AppPage>("library");
  const [summaries, setSummaries] = useState<BrainSummary[]>(demo ? demoSummaries : []);
  const [activeBrain, setActiveBrain] = useState<BrainDocument | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("chat");
  const [loading, setLoading] = useState(!demo);
  const [buildProgress, setBuildProgress] = useState<BuildProgressEvent | null>(null);
  const [activeBuildJob, setActiveBuildJob] = useState<RuntimeJob | null>(null);
  const [retryingInitialization, setRetryingInitialization] = useState(false);
  const [initializationStopIntent, setInitializationStopIntent] =
    useState<InitializationStopIntent | null>(null);
  const [confirmingInitializationCancel, setConfirmingInitializationCancel] = useState(false);
  const [toast, setToast] = useState("");
  const [deleteTarget, setDeleteTarget] = useState<
    Pick<BrainSummary, "id" | "name" | "rootId"> | null
  >(null);
  const [deletingInstance, setDeletingInstance] = useState(false);
  const [storageOperation, setStorageOperation] =
    useState<BrainStorageOperationEvent | null>(null);
  const notificationDispatchTimes = useRef<Partial<Record<LocalNotificationOutcome, number>>>({});
  const resolvedColorScheme = resolveColorScheme(appearance.mode, systemUsesDark);
  const initializationRecovery = activeBrain?.readiness.state === "failed"
    ? initializationRecoveryPresentation(activeBrain.readiness)
    : undefined;
  const initializationRunningActive = activeBrain !== null && initializationIsRunning(
    activeBrain.id,
    activeBrain.readiness,
    retryingInitialization,
    buildProgress
  );
  const initializationRunning = activeBrain !== null && initializationRunningActive
    ? initializationRunningPresentation(activeBrain.id, buildProgress)
    : undefined;
  const initialMaterialsBrainId = buildProgress?.job?.brainId ?? buildProgress?.brainId;
  const initialMaterialsPresentation =
    buildProgress?.phase === "initial-materials" && initialMaterialsBrainId
      ? initializationRunningPresentation(initialMaterialsBrainId, buildProgress)
      : undefined;
  const initializationControlBrainId =
    buildProgress?.job?.brainId ??
    (initializationRunningActive ? activeBrain?.id : undefined) ??
    activeBuildJob?.brainId;
  const activeInitializationJob = initializationControlBrainId
    ? activeInitializationTrainingJob(
        initializationControlBrainId,
        buildProgress,
        activeBuildJob
      )
    : undefined;
  const initializationJobStop = runtimeJobStopPresentation(activeInitializationJob);
  const initializationControlSurfaceActive =
    initializationRunningActive || Boolean(buildProgress && initialMaterialsBrainId);
  const buildMetrics = buildProgress?.data;
  const hasBuildMetrics =
    buildMetrics?.recordsVisited !== undefined ||
    buildMetrics?.neuronDelta !== undefined ||
    buildMetrics?.synapseDelta !== undefined ||
    buildMetrics?.parameterChecksumChanged !== undefined;

  useLayoutEffect(() => {
    const platform = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
    document.documentElement.dataset.platform = /macintosh|mac os|macintel/i.test(platform)
      ? "macos"
      : /windows|win32|win64/i.test(platform)
        ? "windows"
        : /linux/i.test(platform)
          ? "linux"
          : "other";
  }, []);

  useEffect(() => {
    if (typeof window.matchMedia !== "function") return;
    const query = window.matchMedia("(prefers-color-scheme: dark)");
    const update = (event: MediaQueryListEvent | MediaQueryList): void => {
      setSystemUsesDark(event.matches);
    };
    update(query);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    if (!window.omni) return;
    return window.omni.brain.onBuild((event) => setBuildProgress(event));
  }, []);

  useEffect(() => {
    if (!window.omni || typeof window.omni.brain.onStorageOperation !== "function") return;
    return window.omni.brain.onStorageOperation((event) => {
      setStorageOperation(event);
    });
  }, []);

  useEffect(() => {
    if (
      !storageOperation ||
      !["complete", "failed", "cancelled"].includes(storageOperation.state)
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      setStorageOperation((current) =>
        current?.operationId === storageOperation.operationId ? null : current
      );
    }, 5_000);
    return () => window.clearTimeout(timer);
  }, [storageOperation]);

  useEffect(() => {
    const correlatedReadinessSettled =
      activeBrain !== null &&
      activeBrain.id === initializationControlBrainId &&
      activeBrain.readiness.state !== "initializing" &&
      !retryingInitialization;
    if (initializationControlSurfaceActive && !correlatedReadinessSettled) return;
    setInitializationStopIntent(null);
    setConfirmingInitializationCancel(false);
  }, [
    activeBrain?.id,
    activeBrain?.readiness.state,
    initializationControlBrainId,
    initializationControlSurfaceActive,
    retryingInitialization
  ]);

  useEffect(() => {
    if (!window.omni || !activeBrain || activeBrain.readiness.state !== "initializing") return;
    let cancelled = false;
    let timer = 0;
    const refresh = async (): Promise<void> => {
      const current = await window.omni!.brain.get(activeBrain.id).catch(() => undefined);
      if (cancelled) return;
      if (current) {
        setActiveBrain(current);
        if (current.readiness.state !== "initializing") {
          const next = await window.omni!.brain.list().catch(() => undefined);
          if (next) setSummaries(next);
          return;
        }
      }
      timer = window.setTimeout(() => void refresh(), 1_500);
    };
    void refresh();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [activeBrain?.id, activeBrain?.readiness.state]);

  useLayoutEffect(() => {
    applyAppearanceAttributes(document.documentElement, appearance, resolvedColorScheme);
    try {
      saveAppearancePreferences(window.localStorage, appearance);
    } catch {
      // Storage can be unavailable in hardened or ephemeral renderer profiles.
    }
    void window.omni?.window.setAppearance({
      schemaVersion: 1,
      mode: appearance.mode,
      resolvedColorScheme,
      layout: appearance.layout
    }).catch(() => {
      // DOM appearance remains functional if native chrome cannot be updated.
    });
  }, [appearance, resolvedColorScheme]);

  useLayoutEffect(() => {
    applyWorkspaceArrangement(document.documentElement, appearanceWorkspace.arrangement);
    try {
      saveAppearanceWorkspace(window.localStorage, appearanceWorkspace);
    } catch {
      // Presentation preferences remain active for this session if storage is unavailable.
    }
  }, [appearanceWorkspace]);

  const showToast = (message: string) => {
    const presentedMessage = conciseUiMessage(message);
    setToast(presentedMessage);
    window.setTimeout(
      () => setToast((current) => (current === presentedMessage ? "" : current)),
      3_600
    );
    dispatchLocalNotification({
      enabled: appearanceWorkspace.notificationsEnabled,
      visible: document.visibilityState === "visible",
      message: presentedMessage,
      now: Date.now(),
      lastDispatchedAt: notificationDispatchTimes.current,
      environment: browserNotificationEnvironment()
    });
  };

  useEffect(() => {
    if (!window.omni) return;
    const removeImported = window.omni.brain.onImported((imported) => {
      setActiveBrain((current) => current?.id === imported.id ? imported : current);
      void window.omni!.brain.list().then((next) => {
        setSummaries(next);
        setLoading(false);
        showToast(`${imported.name} was imported into the Brain Library.`);
      }).catch((error: unknown) => {
        showToast(
          error instanceof Error
            ? `${imported.name} was imported, but the Library could not refresh: ${error.message}`
            : `${imported.name} was imported, but the Library could not refresh.`
        );
      });
    });
    const removeImportFailed = window.omni.brain.onImportFailed((failure) => {
      showToast(`${failure.fileName} could not be imported: ${failure.message}`);
    });
    return () => {
      removeImported();
      removeImportFailed();
    };
  }, []);

  const changeNotifications = async (enabled: boolean): Promise<boolean> => {
    if (!enabled) {
      setAppearanceWorkspace((current) => ({ ...current, notificationsEnabled: false }));
      return true;
    }
    const granted = await requestLocalNotifications(browserNotificationEnvironment());
    if (!granted) {
      showToast("Desktop notifications are unavailable or were not allowed.");
      return false;
    }
    setAppearanceWorkspace((current) => ({ ...current, notificationsEnabled: true }));
    return true;
  };

  useEffect(() => {
    const brainApi = window.omni?.brain;
    if (!brainApi || typeof brainApi.list !== "function" || page !== "library") return;
    let active = true;
    void brainApi
      .list()
      .then((brains) => {
        if (active) setSummaries(brains);
      })
      .catch((error: unknown) => {
        showToast(error instanceof Error ? error.message : "Could not read the local brain library.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page]);

  const missingLibrarySubstrateIds = summaries
    .filter(
      (summary) =>
        !summary.substrateTotals && !isPristineBrainSummary(summary)
    )
    .map((summary) => summary.id)
    .sort()
    .join("\0");

  useEffect(() => {
    const loadPersistedSubstrate = window.omni?.brain?.persistedSubstrateOverview;
    if (
      typeof loadPersistedSubstrate !== "function" ||
      page !== "library" ||
      !missingLibrarySubstrateIds
    ) return;
    let cancelled = false;
    void hydrateLibrarySubstrateTotals({
      brainIds: missingLibrarySubstrateIds.split("\0"),
      load: (brainId) => loadPersistedSubstrate(brainId),
      cancelled: () => cancelled,
      onResolved: (overview) => {
        setSummaries((current) => current.map((summary) =>
          summary.id === overview.brainId
            ? { ...summary, substrateTotals: overview.totals }
            : summary
        ));
      }
    });
    return () => {
      cancelled = true;
    };
  }, [missingLibrarySubstrateIds, page]);

  const openBrain = async (summary: BrainSummary) => {
    setLoading(true);
    try {
      const document = window.omni
        ? await window.omni.brain.get(summary.id)
        : makeDemoBrain(summary.id, summary.name, createPresetConfig(summary.preset, summary.name));
      setActiveBrain(document);
      setWorkspaceView("chat");
      setPage("workspace");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "This brain could not be opened.");
    } finally {
      setLoading(false);
    }
  };

  const createBrain = async (config: BrainConfig, extras: BuilderExtras) => {
    setBuildProgress({
      sequence: 0,
      phase: "allocating",
      progress: 0.01,
      label: "Initializing OmniCortex native core"
    });
    let hardwareTier: HardwareTier | undefined;
    let acceleratorAvailable: boolean | undefined;
    if (extras.hardware === "auto" && window.omni) {
      const profile = await window.omni.catalog.hardwareProfile();
      hardwareTier = profile.recommendedTier;
      acceleratorAvailable = profile.gpu.available;
    } else if (extras.hardware !== "auto") {
      hardwareTier = extras.hardware;
    }
    const initialResources = extras.initialTraining
      ? extras.initialResources.map((resource) =>
          resource.kind === "web"
            ? { kind: "web" as const, url: resource.url }
            : { kind: "selection" as const, selectionId: resource.id }
        )
      : [];
    const initialSources = extras.initialTraining
      ? extras.initialResources.reduce(
          (total, resource) => total + (resource.kind === "web" ? 1 : resource.itemCount),
          0
        )
      : 0;
    const stopBuildJobEvents = window.omni?.train.onEvent(({ job }) => {
      if (!["ingestion", "crawl"].includes(job.kind)) return;
      setActiveBuildJob(job);
    });
    try {
      const document = window.omni
        ? await window.omni.brain.create({
          config,
          hardwareTier,
          workingMemory: {
            mode: config.workingMemoryMode,
            hardwareTier,
            ...(acceleratorAvailable !== undefined
              ? { acceleratorAvailable }
              : {}),
            systemRamMode: config.systemRamMode,
            ...(config.systemRamMode === "manual"
              ? { systemRamSharePercent: config.systemRamSharePercent }
              : {}),
            storagePoolMode: config.storagePoolMode,
            ...(config.storagePoolMode === "manual"
              ? { storagePoolBytes: String(config.storagePoolBytes) }
              : {}),
            trainingSourceBytes: plannedTrainingSourceBytes(extras.initialResources),
            ...(config.workingMemoryMode === "manual"
              ? {
                  requestedItems: String(config.workingMemorySlots),
                  requestedContextTokens: String(config.contextWindowTokens)
                }
              : {})
          },
          modalities: (Object.entries(extras.modalities) as Array<[ModalityId, boolean]>)
            .filter(([, enabled]) => enabled)
            .map(([modality]) => modality),
          initialToolPermissions: Object.entries(extras.tools).flatMap(([toolId, level]) => {
            return (buildToolProtocolIds[toolId] ?? [toolId]).map((protocolId) => ({ toolId: protocolId, level }));
          }),
          initialResources
          })
        : makeDemoBrain(`demo-${Date.now()}`, config.name, config);
      setActiveBrain(document);
      setSummaries((current) => [
      {
        id: document.id,
        name: document.name,
        preset: document.config.preset,
        runtime: document.config.runtime,
        updatedAt: document.updatedAt,
        concepts: Object.keys(document.concepts).length,
        synapses: Object.keys(document.synapses).length,
        neuralUpdates: document.counters.plasticityEvents,
        inferenceCount: document.counters.inferenceCount,
        trainingSources: brainTrainingSourceCount(document),
        generation: document.lineage.generation,
        rootId: document.lineage.rootId,
        parentId: document.lineage.parentId,
        originChecksum: document.originChecksum,
        instanceKind: "original",
        originInstanceCount: 1,
        activeMode: document.config.idleCognition,
        ...brainSummaryProvenanceFields(document)
      },
      ...current.map((summary) =>
        document.config.idleCognition ? { ...summary, activeMode: false } : summary
      )
      ]);
      setWorkspaceView("chat");
      setPage("workspace");
      showToast(
      initialResources.length > 0
        ? `${document.name} is ready; initial learning${
            initialSources > 0
              ? ` from ${initialSources} selected source${initialSources === 1 ? "" : "s"}`
              : ""
          } completed before chat opened.`
        : `${document.name} is ready to learn.`
      );
    } catch (error) {
      if (window.omni) {
        const next = await window.omni.brain.list().catch(() => []);
        setSummaries(next);
        for (const summary of next) {
          if (summary.name !== config.name) continue;
          const candidate = await window.omni.brain.get(summary.id).catch(() => undefined);
          if (candidate?.readiness.state !== "failed") continue;
          setActiveBrain(candidate);
          setWorkspaceView("chat");
          setPage("workspace");
          showToast(
            candidate.readiness.failure?.message ??
              "Initial learning paused and can be retried safely."
          );
          return;
        }
      }
      throw error;
    } finally {
      stopBuildJobEvents?.();
      setActiveBuildJob(null);
      setBuildProgress(null);
    }
  };

  const stopActiveInitialization = async (
    intent: InitializationStopIntent
  ): Promise<void> => {
    if (!window.omni || initializationStopIntent) return;
    const job = activeInitializationJob;
    if (!job) {
      setConfirmingInitializationCancel(false);
      showToast("Training is already stopping or has not reached a pausable task yet.");
      return;
    }
    if (
      (intent === "pause" && job.state === "cancelling") ||
      (intent === "cancel" && !runtimeJobStopPresentation(job).requestAllowed)
    ) {
      return;
    }
    setInitializationStopIntent(intent);
    try {
      const stopped = intent === "pause"
        ? await window.omni.data.pause(job.id)
        : await window.omni.data.cancel(job.id);
      setActiveBuildJob(stopped);
      setBuildProgress((current) =>
        current &&
        (current.job?.id === job.id ||
          current.brainId === job.brainId ||
          current.job?.brainId === job.brainId)
          ? {
              ...current,
              label:
                intent === "pause"
                  ? "Pausing training at the durable cursor"
                  : "Cancelling training safely",
              job: stopped
            }
          : current
      );
      setConfirmingInitializationCancel(false);
      if (activeBrain?.id === job.brainId) {
        const refreshed = await window.omni.brain.get(job.brainId).catch(() => undefined);
        if (refreshed) setActiveBrain(refreshed);
      }
      showToast(
        intent === "pause"
          ? "Pause requested. The durable cursor is safe; resume training whenever you are ready."
          : "Training cancelled safely. Nothing was deleted, and training can be resumed later."
      );
    } catch (error) {
      setInitializationStopIntent(null);
      showToast(
        error instanceof Error
          ? error.message
          : intent === "pause"
            ? "Training could not be paused."
            : "Training could not be cancelled."
      );
    }
  };

  const retryActiveInitialization = async (): Promise<void> => {
    if (!activeBrain || !window.omni || retryingInitialization) return;
    setRetryingInitialization(true);
    setBuildProgress({
      brainId: activeBrain.id,
      sequence: 0,
      phase: "allocating",
      progress: 0.01,
      label: "Resuming initial neural learning"
    });
    const stopJobs = window.omni.train.onEvent(({ job }) => {
      if (job.brainId === activeBrain.id && ["ingestion", "crawl"].includes(job.kind)) {
        setActiveBuildJob(job);
      }
    });
    try {
      const recovered = await window.omni.brain.retryInitialization(activeBrain.id);
      setActiveBrain(recovered);
      const next = await window.omni.brain.list();
      setSummaries(next);
      showToast(`${recovered.name} finished initial learning and is ready.`);
    } catch (error) {
      const refreshed = await window.omni.brain.get(activeBrain.id).catch(() => undefined);
      if (refreshed) setActiveBrain(refreshed);
      showToast(error instanceof Error ? error.message : "Initial learning could not resume.");
    } finally {
      stopJobs();
      setRetryingInitialization(false);
      setActiveBuildJob(null);
      setBuildProgress(null);
    }
  };

  const importBrain = async () => {
    if (!window.omni) {
      showToast("In Electron, this opens a verified .omni bundle from your device.");
      return;
    }
    try {
      const document = await window.omni.brain.importFile();
      if (document) {
        setActiveBrain(document);
        setPage("workspace");
        const next = await window.omni.brain.list();
        setSummaries(next);
        showToast(`${document.name} was imported and opened.`);
      }
    } catch (error) {
      showToast(error instanceof Error ? error.message : "The selected .omni bundle could not be imported.");
    }
  };

  const updateActiveBrain = (document: BrainDocument) => {
    // A turn may finish after the user has returned to the library or opened a
    // different identity. Persist its summary, but never let that late result
    // replace whichever brain is currently active.
    setActiveBrain((current) =>
      current?.id === document.id ? document : current
    );
    setSummaries((current) =>
      current.map((summary) =>
        summary.id === document.id
          ? {
              ...summary,
              name: document.name,
              updatedAt: document.updatedAt,
              concepts: Object.keys(document.concepts).length,
              synapses: Object.keys(document.synapses).length,
              neuralUpdates: document.counters.plasticityEvents,
              inferenceCount: document.counters.inferenceCount,
              trainingSources: brainTrainingSourceCount(document),
              activeMode: document.config.idleCognition,
              ...brainSummaryProvenanceFields(document)
            }
          : summary
      )
    );
  };

  const applyActiveModeResult = async (result: BrainActiveModeResult): Promise<void> => {
    updateActiveBrain(result.brain);
    if (!window.omni) return;
    const next = await window.omni.brain.list().catch(() => undefined);
    if (next) setSummaries(next);
  };

  const duplicateBrain = async (source: BrainSummary) => {
    setLoading(true);
    try {
      const operationId = globalThis.crypto?.randomUUID?.() ??
        `duplicate-${Date.now()}-${Math.random().toString(16).slice(2)}`;
      const duplicate = window.omni
        ? await window.omni.brain.duplicate(
            source.id,
            `${source.name} copy`,
            operationId
          )
        : makeDemoBrain(
            `demo-copy-${Date.now()}`,
            `${source.name} copy`,
            createPresetConfig("whole-brain", `${source.name} copy`)
          );
      const nextSummary: BrainSummary = {
        id: duplicate.id,
        name: duplicate.name,
        preset: duplicate.config.preset,
        runtime: duplicate.config.runtime,
        updatedAt: duplicate.updatedAt,
        concepts: Object.keys(duplicate.concepts).length,
        synapses: Object.keys(duplicate.synapses).length,
        neuralUpdates: duplicate.counters.plasticityEvents,
        inferenceCount: duplicate.counters.inferenceCount,
        trainingSources: brainTrainingSourceCount(duplicate),
        generation: duplicate.lineage.generation,
        rootId: duplicate.lineage.rootId,
        parentId: duplicate.lineage.parentId,
         originChecksum: duplicate.originChecksum,
         instanceKind: "duplicate",
         originInstanceCount: (source.originInstanceCount ?? 1) + 1,
         activeMode: duplicate.config.idleCognition,
         ...brainSummaryProvenanceFields(duplicate)
      };
      if (window.omni) {
        setSummaries(await window.omni.brain.list());
      } else {
        setSummaries((current) => [
          nextSummary,
          ...current
            .filter((summary) => summary.id !== nextSummary.id)
            .map((summary) =>
              summary.rootId && summary.rootId === nextSummary.rootId
                ? { ...summary, originInstanceCount: nextSummary.originInstanceCount }
                : summary
            )
        ]);
      }
      showToast(`Duplicate instance ${duplicate.name} created with copy-on-write neural storage.`);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "This brain could not be duplicated.");
    } finally {
      setLoading(false);
    }
  };

  const permanentlyDeleteInstance = async (request: DeleteInstanceRequest): Promise<void> => {
    const target = deleteTarget;
    if (!target || target.id !== request.brainId) return;
    setDeletingInstance(true);
    try {
      const result: DeleteInstanceResult = window.omni
        ? await window.omni.brain.remove(request)
        : {
            deleted: true,
            brainId: target.id,
            name: target.name,
            recoverable: false,
            removedSharedBlobs: 0,
            reclaimedBytes: 0
          };
      setSummaries((current) =>
        current
          .filter((summary) => summary.id !== result.brainId)
          .map((summary) =>
            target.rootId && summary.rootId === target.rootId
              ? {
                  ...summary,
                  originInstanceCount: Math.max(1, (summary.originInstanceCount ?? 1) - 1)
                }
              : summary
          )
      );
      if (activeBrain?.id === result.brainId) {
        setActiveBrain(null);
        setPage("library");
      }
      setDeleteTarget(null);
      showToast(
        `${result.name} was permanently deleted and cannot be recovered.${
          result.reclaimedBytes > 0
            ? ` Reclaimed ${formatBytes(result.reclaimedBytes)} of unreferenced shared storage.`
            : " Shared files still used by other instances were kept."
        }`
      );
    } catch (error) {
      showToast(error instanceof Error ? error.message : "The instance could not be deleted.");
    } finally {
      setDeletingInstance(false);
    }
  };

  return (
    <div className="app-shell">
      <div className="mica-glow mica-glow--one" />
      <div className="mica-glow mica-glow--two" />
      <Titlebar
        page={page}
        brain={activeBrain}
        demo={demo}
        onLibrary={() => setPage("library")}
        appearance={appearance}
        appearanceWorkspace={appearanceWorkspace}
        resolvedColorScheme={resolvedColorScheme}
        onAppearanceChange={setAppearance}
        onAppearanceWorkspaceChange={setAppearanceWorkspace}
        onNotificationsChange={changeNotifications}
      />
      {page === "library" ? (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onDuplicate={(summary) => void duplicateBrain(summary)}
          onDelete={setDeleteTarget}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          demo={demo}
        />
      ) : page === "build" ? (
        <SimpleBuildWizard
          onCancel={() => setPage("library")}
          onCreate={createBrain}
        />
      ) : initializationRunning ? (
        <main className="library-page initialization-recovery-page">
          <section className="library-empty initialization-recovery-card" role="status" aria-live="polite">
            <span className="initialization-recovery-card__icon">
              <Icon name="pulse" size={25} />
            </span>
            <h3>{initializationRunning!.title}</h3>
            <p>{initializationRunning!.detail}</p>
            {initializationRunning!.percent !== undefined ? (
              <div
                className="training-progress initialization-recovery-card__progress"
                role="progressbar"
                aria-label={initializationRunning!.title}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={initializationRunning!.percent}
              >
                <span>
                  <strong>{initializationRunning!.percent}%</strong>
                  <em>durable recovery progress</em>
                </span>
                <i>
                  <b style={{ width: `${initializationRunning!.percent}%` }} />
                </i>
              </div>
            ) : null}
            {initializationRunning!.telemetry ? (
              <TrainingTelemetryMetrics telemetry={initializationRunning!.telemetry} />
            ) : null}
            <small>{initializationRunning!.guidance}</small>
            <div className="initialization-recovery-card__actions">
              <Button
                kind="primary"
                icon="pause"
                disabled={
                  !activeInitializationJob ||
                  activeInitializationJob.state === "cancelling" ||
                  initializationStopIntent !== null
                }
                title={
                  activeInitializationJob
                    ? "Pause at the last durable training cursor"
                    : "Pause becomes available when the initial training job starts"
                }
                onClick={() => void stopActiveInitialization("pause")}
              >
                {initializationStopIntent === "pause" ? "Pausing training…" : "Pause training"}
              </Button>
              <Button
                kind="danger"
                icon="close"
                disabled={
                  !activeInitializationJob ||
                  !initializationJobStop.requestAllowed ||
                  initializationStopIntent !== null
                }
                title={
                  activeInitializationJob
                    ? "Stop this run without deleting the mind or its data"
                    : "Cancel becomes available when the initial training job starts"
                }
                onClick={() => setConfirmingInitializationCancel(true)}
              >
                {initializationStopIntent === "cancel"
                  ? "Cancelling training…"
                  : initializationJobStop.retryAvailable
                    ? "Retry cancellation"
                    : initializationJobStop.waitingForAcknowledgement
                      ? "Waiting for stop…"
                  : "Cancel training"}
              </Button>
            </div>
            <small className="initialization-recovery-card__controls-note">
              Pausing or cancelling keeps every committed checkpoint and selected source. Resume training continues from the durable cursor.
            </small>
          </section>
        </main>
      ) : activeBrain?.readiness.state === "failed" ? (
        <main className="library-page initialization-recovery-page">
          <section
            className={`library-empty initialization-recovery-card initialization-recovery-card--${initializationRecovery!.tone}`}
            role={initializationRecovery!.tone === "paused" ? "status" : "alert"}
            aria-live={initializationRecovery!.tone === "paused" ? "polite" : "assertive"}
          >
            <span className="initialization-recovery-card__icon">
              <Icon name={initializationRecovery!.tone === "paused" ? "pulse" : "warning"} size={25} />
            </span>
            <h3>{initializationRecovery!.title}</h3>
            <p>{initializationRecovery!.detail}</p>
            <small>{initializationRecovery!.guidance}</small>
            <div className="initialization-recovery-card__actions">
              <Button
                kind="primary"
                icon="pulse"
                disabled={retryingInitialization}
                onClick={() => void retryActiveInitialization()}
              >
                {retryingInitialization
                  ? initializationRecovery!.busyLabel
                  : initializationRecovery!.actionLabel}
              </Button>
              <Button icon="library" onClick={() => setPage("library")}>
                Back to library
              </Button>
            </div>
          </section>
        </main>
      ) : activeBrain ? (
        <WorkspaceShell
          brain={activeBrain}
          view={workspaceView}
          onView={setWorkspaceView}
          onLibrary={() => setPage("library")}
          onDuplicate={() => {
            const summary = summaries.find((item) => item.id === activeBrain.id);
            if (summary) void duplicateBrain(summary);
          }}
          onDelete={() => setDeleteTarget(
            summaries.find((summary) => summary.id === activeBrain.id) ?? {
              id: activeBrain.id,
              name: activeBrain.name,
              rootId: activeBrain.lineage.rootId
            }
          )}
          onActiveModeChange={applyActiveModeResult}
          onBrainChange={updateActiveBrain}
          onToast={showToast}
        />
      ) : (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onDuplicate={(summary) => void duplicateBrain(summary)}
          onDelete={setDeleteTarget}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          demo={demo}
        />
      )}
      {loading && page !== "library" ? (
        <div className="loading-overlay">
          <BrandMark size={46} />
          <span>Opening neural state…</span>
        </div>
      ) : null}
       {buildProgress && !initializationRunningActive ? (
        <div className="loading-overlay build-readiness-overlay" role="status" aria-live="polite">
          <BrandMark size={52} />
          <strong>{buildProgress.label}</strong>
          <span>{Math.round(buildProgress.progress * 100)}% · Core checks gate chat; fluency depends on training</span>
          <i className="build-readiness-overlay__meter">
            <b style={{ width: `${Math.round(buildProgress.progress * 100)}%` }} />
          </i>
          {initialMaterialsPresentation ? (
            <span>{initialMaterialsPresentation.detail}</span>
          ) : hasBuildMetrics ? (
            <dl>
              {buildMetrics?.recordsVisited !== undefined ? (
                <div>
                  <dt>Data visited</dt>
                  <dd>
                    {buildMetrics.recordsVisited.toLocaleString()}
                    {buildMetrics.recordsTotal !== undefined
                      ? ` / ${buildMetrics.recordsTotal.toLocaleString()}`
                      : ""}
                  </dd>
                </div>
              ) : null}
              {buildMetrics?.neuronDelta !== undefined ? (
                <div><dt>Neurons</dt><dd>{buildMetrics.neuronDelta.toLocaleString()}</dd></div>
              ) : null}
              {buildMetrics?.synapseDelta !== undefined ? (
                <div><dt>Connections</dt><dd>{buildMetrics.synapseDelta.toLocaleString()}</dd></div>
              ) : null}
               {buildMetrics?.parameterChecksumChanged !== undefined ? (
                <div>
                  <dt title="Includes associative neural state; does not independently prove dense model weights changed">Composite neural checksum</dt>
                  <dd>{buildMetrics.parameterChecksumChanged ? "Changed" : "Unchanged"}</dd>
                 </div>
               ) : null}
               {buildMetrics?.diskSpace ? (
                 <div>
                   <dt>Disk space left</dt>
                   <dd>
                     {formatBytes(buildMetrics.diskSpace.projectedRemainingBytes)}
                     {buildMetrics.diskSpace.paused ? " · reserve pause" : ""}
                   </dd>
                 </div>
               ) : null}
             </dl>
          ) : null}
           {activeInitializationJob ? (
             <div className="initialization-training-controls">
               <Button
                 kind="ghost"
                 icon="pause"
                 disabled={activeInitializationJob.state === "cancelling" || initializationStopIntent !== null}
                 onClick={() => void stopActiveInitialization("pause")}
               >
                 {initializationStopIntent === "pause" ? "Pausing training…" : "Pause training"}
               </Button>
               <Button
                 kind="danger"
                 icon="close"
                 disabled={!initializationJobStop.requestAllowed || initializationStopIntent !== null}
                 onClick={() => setConfirmingInitializationCancel(true)}
               >
                 {initializationStopIntent === "cancel"
                   ? "Cancelling training…"
                   : initializationJobStop.retryAvailable
                     ? "Retry cancellation"
                     : initializationJobStop.waitingForAcknowledgement
                       ? "Waiting for stop…"
                   : "Cancel training"}
               </Button>
             </div>
           ) : null}
         </div>
      ) : null}
      {confirmingInitializationCancel ? (
        <InitializationCancelDialog
          brainName={
            activeBrain?.id === initializationControlBrainId
              ? activeBrain?.name ?? "this new mind"
              : "this new mind"
          }
          busy={initializationStopIntent === "cancel"}
          onClose={() => {
            if (initializationStopIntent !== "cancel") {
              setConfirmingInitializationCancel(false);
            }
          }}
          onConfirm={() => stopActiveInitialization("cancel")}
        />
      ) : null}
      {deleteTarget ? (
        <DeleteInstanceDialog
          key={deleteTarget.id}
          target={deleteTarget}
          busy={deletingInstance}
          onCancel={() => {
            if (!deletingInstance) setDeleteTarget(null);
          }}
          onConfirm={permanentlyDeleteInstance}
        />
      ) : null}
      {storageOperation ? (
        <aside
          className={`storage-operation-card storage-operation-card--${storageOperation.state}`}
          role="status"
          aria-live="polite"
        >
          <div className="storage-operation-card__head">
            <span><Icon name="database" size={17} /></span>
            <div>
              <small>{storageOperation.kind.toUpperCase()} · {storageOperation.state}</small>
              <strong>{storageOperation.label}</strong>
            </div>
          </div>
          <div
            className="storage-operation-card__meter"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(
              100 * (
                storageOperation.logicalBytesTotal > 0
                  ? storageOperation.logicalBytesCompleted / storageOperation.logicalBytesTotal
                  : storageOperation.filesTotal > 0
                    ? storageOperation.filesCompleted / storageOperation.filesTotal
                    : 0
              )
            )}
          >
            <i style={{ width: `${Math.round(
              100 * (
                storageOperation.logicalBytesTotal > 0
                  ? storageOperation.logicalBytesCompleted / storageOperation.logicalBytesTotal
                  : storageOperation.filesTotal > 0
                    ? storageOperation.filesCompleted / storageOperation.filesTotal
                    : 0
              )
            )}%` }} />
          </div>
          <dl>
            <div><dt>Files</dt><dd>{storageOperation.filesCompleted.toLocaleString()} / {storageOperation.filesTotal.toLocaleString()}</dd></div>
            <div><dt>Logical</dt><dd>{formatBytes(storageOperation.logicalBytesCompleted)} / {formatBytes(storageOperation.logicalBytesTotal)}</dd></div>
            <div><dt>Shared</dt><dd>{formatBytes(storageOperation.sharedBytes)}</dd></div>
            <div><dt>Physical added</dt><dd>{formatBytes(storageOperation.physicalBytesAdded)}</dd></div>
            <div><dt>Rate</dt><dd>{formatBytes(storageOperation.bytesPerSecond)}/s</dd></div>
            <div><dt>ETA</dt><dd>{storageOperation.etaMs === undefined ? "—" : `${Math.max(0, Math.ceil(storageOperation.etaMs / 1_000))}s`}</dd></div>
            {storageOperation.diskSpace ? (
              <div>
                <dt>Disk after operation</dt>
                <dd>{formatBytes(storageOperation.diskSpace.projectedRemainingBytes)} free · {formatBytes(storageOperation.diskSpace.mandatoryReserveBytes)} reserve</dd>
              </div>
            ) : null}
          </dl>
          {!["complete", "failed", "cancelled"].includes(storageOperation.state) && window.omni ? (
            <div className="storage-operation-card__actions">
              {storageOperation.state === "paused" ? (
                <Button
                  icon="pulse"
                  onClick={() => void window.omni!.brain.resumeStorageOperation(storageOperation.operationId)}
                >
                  Resume
                </Button>
              ) : (
                <Button
                  icon="pause"
                  onClick={() => void window.omni!.brain.pauseStorageOperation(storageOperation.operationId)}
                >
                  Pause
                </Button>
              )}
              <Button
                kind="danger"
                icon="close"
                disabled={storageOperation.state === "cancelling"}
                onClick={() => void window.omni!.brain.cancelStorageOperation(storageOperation.operationId)}
              >
                Cancel {storageOperation.kind}
              </Button>
            </div>
          ) : null}
        </aside>
      ) : null}
      {toast ? (
        <div className="toast" role="status">
          <span><Icon name="check" size={15} /></span>
          <p className="toast__message">{toast}</p>
          <button onClick={() => setToast("")} aria-label="Dismiss"><Icon name="close" size={14} /></button>
        </div>
      ) : null}
    </div>
  );
}
