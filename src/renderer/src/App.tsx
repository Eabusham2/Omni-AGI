import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode
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
  type HardwareTier,
  type HardwareProfile as SystemHardwareProfile,
  type IngestResult,
  type MobileGatewayStatus,
  type MobilePairingSession,
  type PersistedDatasetCoverageSummary,
  type RuntimeJob,
  type ToolInvocation,
  type ToolExecutionResult,
  type ToolPermissionRecord,
  type ActionEvent,
  type AppearanceLayout,
  type AppearanceMode,
  type AppearancePalette,
  type AppearancePreferences,
  type WorkspaceSnapshot,
  type StoragePoolMode,
  type SystemRamMode,
  type WorkingMemoryMode,
  type WorkingMemoryResourcePlan,
  type McpTransport,
  type McpServerSummary,
  type McpRegistrySummary,
  type McpRegistryUpdateRequest
} from "@shared/types";
import type {
  BrainActiveModeResult,
  BrainSnapshotSummary
} from "@shared/types";
import { chatInputCapacity } from "@shared/chatInput";
import { chatHostSuspensionStatus } from "@shared/chatSuspension";
import { pausedOutputLabel, savedChatSuspension } from "./chatSuspensionPresentation";
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
import { ComposerSubmitControl } from "./ComposerSubmitControl";
import { UniversalSourcePicker } from "./UniversalSourcePicker";
import "./universalSourcePicker.css";
import "./streamlinedBrainControls.css";
import { librarySelection, toggleLibrarySelection, selectVisibleLibraryBrains } from "./librarySelection";
import { ChatQueueStack } from "./ChatQueueStack";
import { UserQuestionPanel } from "./UserQuestionPanel";
import { ChatDirectionStrip } from "./ChatDirectionStrip";
import { useChatDirections } from "./useChatDirections";
import { chatDirectionStripItem, isSteerDirectionMessage } from "./chatDirectionPresentation";
import { chatMessageAttribution } from "./chatMessageAttribution";
import { McpAuthenticationPanel } from "./McpAuthenticationPanel";
import { McpCredentialPanel } from "./McpCredentialPanel";
import { McpPresetPicker } from "./McpPresetPicker";
import { MapConnectionPanel } from "./MapConnectionPanel";
import { MapBasemapPanel } from "./MapBasemapPanel";
import { MapRasterView } from "./MapRasterView";
import { mapRasterFromAction } from "./mapRasterPresentation";
import { MapResultView } from "./MapResultView";
import { mapDisplayFromAction } from "./mapPresentation";
import { McpInteractionPanel } from "./McpInteractionPanel";
import { McpAppView } from "./McpAppView";
import { HtmlArtifactView } from "./HtmlArtifactView";
import { chatHtmlArtifactRequest } from "./htmlArtifactPresentation";
import { PluginManager } from "./PluginManager";
import { PluginApprovalReviewView, PluginToolApprovalReviewView } from "./PluginApprovalReview";
import { LearningWorkspace } from "./LearningWorkspace";
import { ExternalAdvicePanel } from "./ExternalAdvicePanel";
import { ActionReviewDetails } from "./ActionReviewDetails";
import { BuildProgressStatus, StorageOperationTiming } from "./BuildProgressStatus";
import { PluginThemeHost, savePluginThemeSelection, clearPluginThemeSelection } from "./PluginThemes";
import { applyPortablePresentationOnce } from "./portablePresentation";
import { PluginRuntimeContributionsProvider, PluginRuntimeSlot } from "./PluginRuntimeContributions";
import { PluginRuntimeApprovalQueue } from "./PluginRuntimeApprovals";
import { SparseSubstrateJobsProvider, SparseSubstrateJobsPanel, SparseSubstrateJobCard, useSparseSubstrateActionJob } from "./SparseSubstrateJobs";
import { sparseJobLabel } from "./sparseSubstratePresentation";
import { ResponseOutputCounter } from "./ResponseOutputCounter";
import { FullMergePanel } from "./FullMergePanel";
import { DeviceAllocationSelector, synchronizeDeviceRam } from "./DeviceAllocationSelector";
import { RegionalExecutionControls } from "./RegionalExecutionControls";
import { DecoderExecutionControls } from "./DecoderExecutionControls";
import { CachedDecoderExecutionControls } from "./CachedDecoderExecutionControls";
import { NativeOperationAutoControls } from "./NativeOperationAutoControls";
import { PonderPresentationLedger, reconcilePonderPresentation, type PonderPresentation } from "./ponderSelfReportPresentation";
import { groupPonderActions } from "./chatPonderGroups";
import { splitPublicReplyAtActions, verifyReplyActionLocations, type PublicReplySource, type VerifiedReplyLocation } from "./chatReplyProjection";
import { PonderSelfReportView } from "./PonderSelfReportView";
import { McpHostServicesPanel } from "./McpHostServicesPanel";
import { McpTasksPanel } from "./McpTasksPanel";
import { McpTaskActionView } from "./McpTaskActionView";
import { mcpTaskActionLabel } from "./mcpTaskPresentation";
import { WholeBrainMap } from "./WholeBrainMap";
import { InstanceLifecycleControls } from "./InstanceLifecycleControls";
import { InheritedTaskFilesReview } from "./InheritedTaskFilesReview";
import { ImaginationCaptureControls } from "./ImaginationCaptureControls";
import { ProcessMemorySettings } from "./ProcessMemorySettings";
import { ToastNotice } from "./ToastNotice";
import { ApiTeacherPanel } from "./ApiTeacherPanel";
import { libraryRuntimePresentation } from "./libraryRuntimePresentation";
import { ChatImaginationMedia } from "./ChatImaginationMedia";
import { ToolActionPicker, defaultToolAction } from "./ToolActionPicker";
import type { ToolDescription } from "../../shared/toolDescriptions";
import type { InheritedTaskFileReview } from "../../shared/inheritedTaskFiles";
import type { InstanceLifecycleState } from "../../shared/instanceLifecycle";
import type { DeviceResourceSettings } from "@shared/deviceResources";
import { Icon, type IconName } from "./icons";
import {
  createTextFrameBatcher,
  mergeChatActionEvent,
  patchChatActionPreview,
  type TextFrameBatcher
} from "./chatStreaming";
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
  DEFAULT_LIVE_VOICE_PREFERENCES
} from "./liveVoicePreferences";
import { useInstanceVoice } from "./useInstanceVoice";
import { VoiceRoutingControls } from "./VoiceRoutingControls";
import { useTaskAttachmentDraft } from "./useTaskAttachmentDraft";
import { TaskAttachmentStrip } from "./TaskAttachmentStrip";
import { MessageTaskAttachments } from "./MessageTaskAttachments";
import { NativeAudioInputSummary } from "./NativeAudioInputSummary";
import type { ChatTaskAttachmentHandle } from "@shared/taskAttachments";
import type { TouchBarState } from "@shared/touchBar";
import {
  conversationTouchBarState,
  matchTouchBarAction,
  touchBarActionRemainsReady,
  type ConversationTouchBarInput
} from "./touchBarPresentation";
import { studioUiDestination } from "./studioUiActions";
import { LivePerceptionPanel } from "./LivePerceptionPanel";
import { CallLayout } from "./CallLayout";
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
  chatActionMcpAppViewId,
  chatActionStateLabel,
  chatActionTitle,
  chatApprovalStatus,
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
  neuralParameterAccounting,
  neuralResourceRuntimeRows,
  type NeuralParameterAccountingPresentation
} from "./neuralRuntimePresentation";
import { brainProvenancePresentation } from "./brainProvenancePresentation";
import { CognitionProgressView } from "./CognitionProgressView";
import type { CognitionProgress } from "../../shared/cognition";
import {
  connectionCountPresentation,
  conciseUiMessage,
  isPristineBrainSummary,
  recoveryPointCreationErrorMessage,
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
  | "tools"
  | "settings";
type HardwareChoice = "auto" | "micro" | "personal" | "gpu" | "workstation";
type PermissionLevel = "off" | "ask" | "auto" | "full";
type ModalityId = "vision" | "image" | "audio" | "video";

interface BuilderExtras {
  spillPoolSelectionId?: string;
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
  { id: "data", label: "Training & Evolution", icon: "database" },
  { id: "map", label: "Brain map", icon: "brain" },
  { id: "tools", label: "Tools & permissions", icon: "terminal" },
  { id: "settings", label: "Settings", icon: "settings" }
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
  const panel = useRef<HTMLElement>(null);
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

  useLayoutEffect(() => {
    const popup = panel.current;
    if (!open || !popup || typeof popup.showPopover !== "function") return;
    // Only this non-modal panel enters the native top layer. The titlebar and
    // its other controls stay below actual confirmation/permission dialogs.
    popup.showPopover();
    return () => { if (popup.matches(":popover-open")) popup.hidePopover(); };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const closeOnPointer = (event: globalThis.PointerEvent): void => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: globalThis.KeyboardEvent): void => {
      if (event.key === "Escape") {
        setOpen(false);
        // Never pull focus out of another modal that opened while this menu
        // was visible. Ordinary Appearance Escape returns to its own trigger.
        if (!document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]')) {
          container.current?.querySelector<HTMLButtonElement>(".appearance-trigger")?.focus();
        }
      }
    };
    const closeForModalFocus = (event: FocusEvent): void => {
      const modal = event.target instanceof Element
        ? event.target.closest('dialog[open], [role="dialog"][aria-modal="true"]')
        : null;
      if (modal && !panel.current?.contains(modal)) setOpen(false);
    };
    const closeForVisibleModal = (): void => {
      if ([...document.querySelectorAll<HTMLElement>('dialog[open], [role="dialog"][aria-modal="true"]')]
        .some(modal => !panel.current?.contains(modal) && modal.getClientRects().length > 0)) setOpen(false);
    };
    const modalAttributes = new MutationObserver(closeForVisibleModal);
    const modalChildren = new MutationObserver(closeForVisibleModal);
    modalAttributes.observe(document.body, { attributes: true, subtree: true, attributeFilter: ["open", "aria-modal"] });
    const shell = container.current?.closest(".app-shell");
    if (shell) modalChildren.observe(shell, { childList: true });
    closeForVisibleModal();
    document.addEventListener("pointerdown", closeOnPointer);
    document.addEventListener("keydown", closeOnEscape);
    document.addEventListener("focusin", closeForModalFocus);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointer);
      document.removeEventListener("keydown", closeOnEscape);
      document.removeEventListener("focusin", closeForModalFocus);
      modalAttributes.disconnect();
      modalChildren.disconnect();
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
        <section ref={panel} popover="manual" className="appearance-panel" role="dialog" aria-label="Appearance settings">
          <header>
            <span>
              <strong>Appearance</strong>
              <small>Choose how the studio looks. Brain identity and learning never change.</small>
              <em>{activePackName} · {preferences.mode === "system" ? `Auto ${resolvedColorScheme}` : resolvedColorScheme}</em>
            </span>
            <button
              className="icon-button"
              aria-label="Close appearance settings"
              onClick={() => {
                setOpen(false);
                container.current?.querySelector<HTMLButtonElement>(".appearance-trigger")?.focus();
              }}
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
  onExport,
  onMerged,
  demo
}: {
  summaries: BrainSummary[];
  loading: boolean;
  onOpen: (summary: BrainSummary | string) => void;
  onDuplicate: (summaries: BrainSummary[]) => Promise<void>;
  onDelete: (summaries: BrainSummary[]) => void;
  onBuild: () => void;
  onImport: () => void;
  onExport: (summaries: BrainSummary[], mode?: BrainExportMode) => Promise<void>;
  onMerged: () => void;
  demo: boolean;
}) {
  const [query, setQuery] = useState("");
  const [selecting, setSelecting] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [mergeOpen, setMergeOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState<"duplicate" | "export" | null>(null);
  const [actionError, setActionError] = useState("");
  const actionInFlight = useRef(false), libraryAlive = useRef(true);
  const exportMenu = useRef<HTMLDetailsElement>(null);
  const visible = summaries.filter((brain) => brain.name.toLocaleLowerCase().includes(query.toLocaleLowerCase()));
  const selection = librarySelection(summaries, selectedIds, visible);
  const busy = loading || pendingAction !== null;
  const runtimeNotice = libraryRuntimePresentation(demo);
  useEffect(() => { libraryAlive.current = true; return () => { libraryAlive.current = false; }; }, []);
  useEffect(() => {
    const closeOutside = (event: PointerEvent): void => {
      if (exportMenu.current?.open && event.target instanceof Node && !exportMenu.current.contains(event.target)) exportMenu.current.open = false;
    };
    document.addEventListener("pointerdown", closeOutside);
    return () => document.removeEventListener("pointerdown", closeOutside);
  }, []);
  useEffect(() => {
    const available = new Set(summaries.map(brain => brain.id));
    setSelectedIds(current => {
      const next = new Set([...current].filter(id => available.has(id)));
      return next.size === current.size ? current : next;
    });
  }, [summaries]);
  const closeExportMenu = (restoreFocus = false): void => {
    if (!exportMenu.current) return;
    exportMenu.current.open = false;
    if (restoreFocus) exportMenu.current.querySelector("summary")?.focus();
  };
  const toggleBrain = (id: string): void => {
    if (busy) return;
    closeExportMenu();
    setMergeOpen(false); setActionError("");
    setSelectedIds(current => toggleLibrarySelection(current, id));
  };
  const manage = async (action: "duplicate" | "export", mode: BrainExportMode = "current"): Promise<void> => {
    if (actionInFlight.current || busy || !selection.selected.length) return;
    const captured = [...selection.selected];
    closeExportMenu(action === "export");
    actionInFlight.current = true; setPendingAction(action); setActionError(""); setMergeOpen(false);
    try {
      if (action === "duplicate") await onDuplicate(captured);
      else await onExport(captured, mode);
    } catch (error) {
      if (libraryAlive.current) setActionError(error instanceof Error ? error.message : `The selected brains could not be ${action === "duplicate" ? "duplicated" : "exported"}.`);
    } finally { actionInFlight.current = false; if (libraryAlive.current) setPendingAction(null); }
  };

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
            <Button kind="primary" icon="plus" onClick={onBuild} disabled={busy}>
              Build a new brain
            </Button>
            <Button icon="upload" onClick={onImport} disabled={busy} title="Import a complete .omni brain export">
              Import brain
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
            <h2>Your brains</h2>
            <p>{loading ? "Looking for saved brains…" : `${summaries.length} saved ${summaries.length === 1 ? "brain" : "brains"}`}</p>
          </div>
          <div className="library-heading-actions"><label className="search-field">
            <Icon name="search" size={16} />
            <input
              value={query}
              onChange={(event) => { closeExportMenu(); setQuery(event.target.value); }}
              placeholder="Search brains"
              aria-label="Search brains"
            />
          </label>
          {summaries.length || selecting ? <Button disabled={busy} aria-pressed={selecting} onClick={() => {
            closeExportMenu(); setSelecting(current => !current); setSelectedIds(new Set()); setMergeOpen(false); setActionError("");
          }}>{selecting ? "Done" : "Select"}</Button> : null}</div>
        </div>

        {selecting ? <div className="library-selection-toolbar" role="toolbar" aria-label="Manage selected brains">
          <span className="library-selection-count" role="status">{selection.selected.length} selected{selection.hiddenSelected ? ` · ${selection.hiddenSelected} outside this search` : ""}</span>
          <Button disabled={busy || !visible.length || selection.allVisibleSelected} onClick={() => {
            closeExportMenu(); setSelectedIds(current => selectVisibleLibraryBrains(current, visible)); setMergeOpen(false);
          }}>{query ? "Select all results" : "Select all"}</Button>
          <Button disabled={busy || !selection.selected.length} onClick={() => { closeExportMenu(); setSelectedIds(new Set()); setMergeOpen(false); }}>Clear</Button>
          <span className="library-selection-divider" aria-hidden="true" />
          <Button icon="copy" disabled={busy || !selection.selected.length} onClick={() => void manage("duplicate")}>Duplicate</Button>
          <details className="library-export-menu" ref={exportMenu} onKeyDown={(event) => {
            if (event.key === "Escape") { event.preventDefault(); closeExportMenu(true); }
          }}>
            <summary className="button button--secondary" aria-disabled={busy || !selection.selected.length} tabIndex={busy || !selection.selected.length ? -1 : 0} onClick={(event) => {
              if (busy || !selection.selected.length) event.preventDefault();
            }}><Icon name="download" size={16} /><span>Export brain</span><span aria-hidden="true">▾</span></summary>
            <div className="library-export-options" role="group" aria-label="Brain export options">
              <button type="button" className="library-export-option" disabled={busy || !selection.selected.length} onClick={() => void manage("export", "current")}><span>Current brain</span><small>Complete saved state · default</small></button>
              <button type="button" className="library-export-option" disabled={busy || !selection.selected.length} onClick={() => void manage("export", "origin")}><span>Original origin</span><small>Original recovery state</small></button>
              <button type="button" className="library-export-option" disabled={busy || !selection.selected.length} onClick={() => void manage("export", "referenced")}><span>Local reference</span><small>Depends on this device's saved blobs</small></button>
            </div>
          </details>
          <Button disabled={busy || !selection.canMerge || !window.omni?.brain.startFullMerge} title={!window.omni?.brain.startFullMerge ? "Merging requires the desktop native runtime." : selection.mergeReason || "Create a new brain while preserving every selected parent."} onClick={() => { closeExportMenu(); setActionError(""); setMergeOpen(true); }}>Merge</Button>
          <Button kind="danger" disabled={busy || !selection.selected.length} onClick={() => {
            closeExportMenu(); setMergeOpen(false); onDelete([...selection.selected]);
          }}>Delete</Button>
        </div> : null}
        {pendingAction ? <p className="library-management-status" role="status">{pendingAction === "duplicate" ? "Duplicating selected brains…" : "Exporting selected brains…"}</p> : null}
        {actionError ? <p className="library-management-error" role="alert">{actionError}</p> : null}
        <FullMergePanel summaries={summaries} selectedParentIds={selection.selected.map(brain => brain.id)} open={mergeOpen}
          onClose={() => setMergeOpen(false)} api={window.omni} onPublished={onMerged} onOpen={onOpen} />

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
                  className={cx("brain-card", selecting && "brain-card--selectable", selecting && selectedIds.has(brain.id) && "brain-card--selected")}
                  key={brain.id}
                  tabIndex={busy ? -1 : 0}
                  role={selecting ? "checkbox" : "button"}
                  aria-label={`${selecting ? "Select" : "Open"} brain ${brain.name}`}
                  aria-checked={selecting ? selectedIds.has(brain.id) : undefined}
                  aria-disabled={busy}
                  onClick={() => { if (!busy) { if (selecting) toggleBrain(brain.id); else onOpen(brain); } }}
                  onKeyDown={(event) => {
                    if (event.target === event.currentTarget && (event.key === "Enter" || event.key === " ")) {
                      event.preventDefault();
                      if (!busy) { if (selecting) toggleBrain(brain.id); else onOpen(brain); }
                    }
                  }}
                >
                  <div className="brain-card__top">
                    <div className={cx("brain-avatar", `brain-avatar--${meta.color}`)}>
                      <Icon name={meta.icon} size={25} />
                      {brain.activeMode ? <span className="brain-avatar__live" title="Active Mode" /> : null}
                    </div>
                    {selecting ? <span className="library-selection-indicator" aria-hidden="true">{selectedIds.has(brain.id) ? <Icon name="check" size={16} /> : null}</span> : null}
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
                      {selecting ? selectedIds.has(brain.id) ? "Selected" : "Select brain" : <>Open mind <Icon name="arrow" size={14} /></>}
                    </span>
                  </div>
                </article>
              );
            })}
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
              <strong>{runtimeNotice.title}</strong>
              <span>{runtimeNotice.detail}</span>
            </div>
            {runtimeNotice.badge ? <span className="system-card__pill system-card__pill--demo">
              {runtimeNotice.badge}
            </span> : null}
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
  const [deviceResources, setDeviceResources] = useState<DeviceResourceSettings>();
  const effectiveDeviceResources = useMemo(() => synchronizeDeviceRam(deviceResources, systemRamMode, systemRamSharePercent),
    [deviceResources, systemRamMode, systemRamSharePercent]);
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
        ...(effectiveDeviceResources ? { deviceResources: effectiveDeviceResources } : {}),
        ...(extras.spillPoolSelectionId ? { spillPoolSelectionId: extras.spillPoolSelectionId } : {}),
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
    extras.hardware,
    extras.spillPoolSelectionId,
    effectiveDeviceResources
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
      ...(effectiveDeviceResources ? { deviceResources: effectiveDeviceResources } : {}),
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
        modalities: { vision: true, image: true, audio: true, video: true },
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
              ? `Packed ternary signs · ${formatBytes(memoryPlan.architecture.packedTernaryWeightBytes)}; strength and plasticity · ${memoryPlan.architecture.connectionStateBytes === undefined ? "not reported" : formatBytes(memoryPlan.architecture.connectionStateBytes)}. Runtime state is budgeted separately.`
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
          <p>Working memory holds current activity. Learning updates neural state over time.</p>
          <div className="working-memory-planner">
            <div className="working-memory-planner__heading">
              <span>
                <strong>Working memory</strong>
                <small>Sized for this brain and device; checked again before loading.</small>
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
            <details className="brain-settings-disclosure">
              <summary>GPU memory &amp; drives</summary>
            <DeviceAllocationSelector catalog={window.omni?.catalog} settings={effectiveDeviceResources}
              ramMode={systemRamMode} ramPercent={systemRamSharePercent} disabled={building}
              plan={memoryPlan?.deviceResources} planPending={memoryPlanPending}
              onChange={(settings, selectionId) => {
                setDeviceResources(settings); setMemoryPlanPending(Boolean(window.omni));
                const ram = settings.allocations.find((entry) => entry.deviceId === "ram:system");
                if (ram) { setSystemRamMode(ram.mode); setSystemRamSharePercent(ram.percent); }
                if (selectionId) setExtras((current) => ({ ...current, spillPoolSelectionId: selectionId }));
              }} />
            </details>
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
          <h2 tabIndex={-1}>Initial learning</h2>
          <p>Add files, media, folders, or a website. Text, vision, audio, and video are included.</p>
        </div>
      </div>
      <div className="build-resource-actions">
        <UniversalSourcePicker label="Add source" icon="plus" disabled={building}
          onSelect={selection => void selectInitialResource(selection)} />
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
          <div><dt>Working memory</dt><dd>{memoryPlan ? `${memoryPlan.context.selectedTokens.toLocaleString()} tokens` : "Hardware-sized automatically"}</dd></div>
          <div><dt>RAM budget</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.resources.systemRamBudgetBytes)} · ${memoryPlan.resources.systemRamSharePercent}% after system reserve` : "Measuring device"}</dd></div>
          <div><dt>Shared storage</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.resources.sharedStoragePoolBytes)} · ${storagePoolMode === "auto" ? "Auto" : "Custom"}` : "Measuring device"}</dd></div>
          <div><dt>Disk space left</dt><dd>{memoryPlan ? `${formatBytes(memoryPlan.diskSpace.projectedRemainingBytes)} · ${formatBytes(memoryPlan.diskSpace.projectedAboveReserveBytes)} above reserve` : "Measuring device"}</dd></div>
          <div><dt>Native core</dt><dd>{memoryPlan?.architecture ? `${memoryPlan.architecture.exactLogicalParameterCount.toLocaleString()} logical neural parameters · locally initialized` : "Calculating exact architecture"}</dd></div>
          <div><dt>System access</dt><dd>{!systemAccessEnabled ? "Off" : fullAuthorityEnabled ? "Full Authority" : "On · risky actions ask"}</dd></div>
          <div><dt>Self-improvement</dt><dd>Available under chosen access</dd></div>
          <div><dt>First learning</dt><dd>{extras.initialResources.length ? `Tool basics + ${extras.initialResources.length} queued source${extras.initialResources.length === 1 ? "" : "s"} · ${formatBytes(selectedTrainingBytes)} sizing input` : "Tool basics only · language starts primitive"}</dd></div>
          <div><dt>Inputs &amp; imagination</dt><dd>Text, vision, audio, video</dd></div>
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
  onActiveModeChange,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  view: WorkspaceView;
  onView: (view: WorkspaceView) => void;
  onLibrary: () => void;
  onActiveModeChange: (result: BrainActiveModeResult) => Promise<void>;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [health, setHealth] = useState("Checking runtime…");
  const [compactInspectorOpen, setCompactInspectorOpen] = useState(false);
  const [adviceOpen, setAdviceOpen] = useState(false);
  const [adviceVisited, setAdviceVisited] = useState(false);
  const [instanceState, setInstanceState] = useState<InstanceLifecycleState>();
  const [loadReviewRequest, setLoadReviewRequest] = useState(0);
  const runtimeLoaded = !window.omni?.brain.lifecycle || instanceState?.brainId === brain.id && instanceState.phase === "loaded";
  const [chatActivity, setChatActivity] = useState<ChatWorkspaceActivitySnapshot>(
    EMPTY_CHAT_WORKSPACE_ACTIVITY
  );
  const chatActivityStatus = workspaceChatActivityPresentation(chatActivity);
  const offlineSettings = view === "settings" && instanceState && !instanceState.learnedRuntimeResident && ["unloaded", "suspended", "error"].includes(instanceState.phase);
  const viewWaitsForTurn = !offlineSettings && workspaceViewWaitsForForegroundTurn(view, chatActivity);
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
    <PluginRuntimeContributionsProvider brainId={brain.id} onNotice={onToast}>
    <SparseSubstrateJobsProvider brainId={brain.id}>
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
            {view === "chat" ? <PluginRuntimeSlot slot="chat.toolbar" /> : null}
            <button type="button" className="button button--ghost" aria-controls="external-advice-panel" aria-expanded={adviceOpen}
              title="Separate optional assistant and pre-action reviewer" onClick={() => { setAdviceVisited(true); setAdviceOpen(open => !open); }}>Assistant</button>
            <InstanceLifecycleControls key={brain.id} brainId={brain.id} name={brain.name}
              api={window.omni?.brain.lifecycle ? window.omni.brain : undefined} onState={setInstanceState}
              reviewRequest={loadReviewRequest} onEditSettings={() => onView("settings")} />
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
          </div>
        </header>
        <McpInteractionPanel brainId={brain.id} />
        {adviceVisited ? <ExternalAdvicePanel brainId={brain.id} api={window.omni?.advice} hidden={!adviceOpen} onClose={() => setAdviceOpen(false)} /> : null}
        <button
          className="workspace-inspector-backdrop"
          aria-label="Close cortex inspector"
          tabIndex={compactInspectorOpen ? 0 : -1}
          onClick={() => setCompactInspectorOpen(false)}
        />
        <ChatWorkspace
          brain={brain}
          runtimeLoaded={Boolean(runtimeLoaded)}
          resumeCheckpointId={instanceState?.resume?.checkpointId}
          inheritedExecutionId={instanceState?.inheritedExecutions?.entries.find(entry => entry.status === "queued" || entry.status === "suspended")?.id}
          hidden={view !== "chat"}
          onBrainChange={onBrainChange}
          onActiveModeChange={onActiveModeChange}
          onToast={onToast}
          onNavigate={onView}
          onActivityChange={setChatActivity}
        />
        {view !== "chat" ? (
          <div className={cx("workspace-view-host", viewWaitsForTurn && "is-waiting")}>
            <div
              className="workspace-view-host__content"
              inert={viewWaitsForTurn || (!runtimeLoaded && view !== "settings" && view !== "tools") || undefined}
              aria-busy={viewWaitsForTurn || undefined}
            >
              {view === "data" ? (
                <DataWorkspace
                  brain={brain}
                  foregroundTurnActive={chatActivity.turnActive}
                  onBack={() => onView("chat")}
                  onBrainChange={onBrainChange}
                  onToast={onToast}
                  onOpenPermissions={() => onView("tools")}
                />
              ) : view === "map" ? (
                <BrainMapWorkspace brain={brain} onBack={() => onView("chat")} />
              ) : view === "tools" ? (
                <ToolsWorkspace key={brain.id} brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
              ) : view === "settings" ? (
                <DeviceRuntimeWorkspace
                  key={brain.id}
                  brain={brain}
                  foregroundTurnActive={chatActivity.turnActive && chatActivity.phase !== "reply-complete-learning" && chatActivity.phase !== "action-result-learning"}
                  instanceState={instanceState} onReviewLoad={() => setLoadReviewRequest((value) => value + 1)}
                  onBrainChange={onBrainChange}
                  onToast={onToast}
                />
              ) : null}
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
    </SparseSubstrateJobsProvider>
    </PluginRuntimeContributionsProvider>
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
  attachments?: ChatTaskAttachmentHandle[];
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

function ChatActionCard({ event, onCancel, onApprove, onDecline, ponderReports, readablePonder = false, onReadablePonder, flushPonder,
  children, childStatus, inlineMedia = true }: {
  event: ActionEvent;
  onCancel?: () => Promise<void>;
  onApprove?: () => Promise<void>;
  onDecline?: (instruction?: string) => Promise<void>;
  ponderReports?: PonderPresentation[];
  readablePonder?: boolean;
  onReadablePonder?: (value: boolean) => void;
  flushPonder?: boolean;
  children?: ReactNode;
  childStatus?: string;
  inlineMedia?: boolean;
}) {
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
  const title = chatActionTitle(event.action);
  const sparseJob = useSparseSubstrateActionJob(event);
  const stateLabel = sparseJob ? sparseJobLabel(sparseJob.state) : mcpTaskActionLabel(event) ?? (event.cancellationRequested ? "Cancelling" : chatActionStateLabel(event.state));
  const searchActivity = chatActionSearchActivity(event.action, event.state);
  const mcpAppViewId = chatActionMcpAppViewId(event);
  const mapResult = useMemo(() => mapDisplayFromAction(event), [event.id, event.state, event.action, event.execution]);
  const mapRaster = useMemo(() => mapRasterFromAction(event), [event.id, event.state, event.action, event.execution]);
  const visibleArguments = event.action.kind === "ponder"
    ? Object.fromEntries(Object.entries(event.action.arguments).filter(([key]) => key !== "ponderSelfReport")) : event.action.arguments;
  const argumentsText = Object.keys(visibleArguments).length
    ? JSON.stringify(visibleArguments)
    : "";

  return (
    <>
      {inlineMedia ? <ChatImaginationMedia event={event} /> : null}
      <details className={cx("chat-action-card", `chat-action-card--${event.state}`)}>
      <summary aria-label={`${title}: ${stateLabel}${childStatus ? `. ${childStatus}` : ""}. Expand action details`}>
        <span className="chat-action-card__icon"><Icon name={iconByKind[event.action.kind]} size={15} /></span>
        <span className="chat-action-card__summary-copy">
          <strong>{title}</strong>
          <small>{searchActivity ? `${stateLabel} · ${searchActivity}` : stateLabel}{childStatus ? ` · ${childStatus}` : ""}</small>
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
        {event.action.kind === "ponder" && onReadablePonder ? ponderReports?.map(row => <PonderSelfReportView key={row.report.id}
          row={row} readable={readablePonder} onReadableChange={onReadablePonder} flush={flushPonder} />) : null}
        {children}
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
        <PluginApprovalReviewView event={event} />
        <ActionReviewDetails execution={event.execution} onDecline={onDecline} />
        {sparseJob ? <SparseSubstrateJobCard key={`${sparseJob.brainId}/${sparseJob.id}`} job={sparseJob} /> : null}
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
        <ChatImaginationMedia event={event} placement="audit" />
        {studioUiDestination(event) ? (
          <span className="chat-action-card__artifact">
            <Icon name={studioUiDestination(event) === "imagine" ? "sparkles" : "settings"} size={14} /> {studioUiDestination(event) === "imagine" ? "Imagination stays in Conversation" : "Permissions available in Tools"}
          </span>
        ) : null}
        {mcpAppViewId ? <McpAppView brainId={event.brainId} viewId={mcpAppViewId} /> : null}
        <McpTaskActionView event={event} />
        {chatHtmlArtifactRequest(event) ? <HtmlArtifactView event={event} /> : null}
        {mapResult ? <MapResultView result={mapResult} /> : null}
        {mapRaster ? <MapRasterView result={mapRaster} /> : null}
        <span className="chat-action-card__meta">
          {event.action.confidence !== undefined ? `${Math.round(event.action.confidence * 100)}% action confidence · ` : ""}
          {new Date(event.updatedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        </span>
      </div>
      </details>
    </>
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
  runtimeLoaded = true,
  resumeCheckpointId,
  inheritedExecutionId,
  hidden,
  onBrainChange,
  onToast,
  onNavigate,
  onActiveModeChange,
  onActivityChange
}: {
  brain: BrainDocument;
  runtimeLoaded?: boolean;
  resumeCheckpointId?: string;
  inheritedExecutionId?: string;
  hidden?: boolean;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
  onNavigate: (view: WorkspaceView) => void;
  onActiveModeChange: (result: BrainActiveModeResult) => Promise<void>;
  onActivityChange: (activity: ChatWorkspaceActivitySnapshot) => void;
}) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [attachmentDragActive, setAttachmentDragActive] = useState(false);
  const [callOpen, setCallOpen] = useState(false);
  const [captureOpen, setCaptureOpen] = useState(false);
  const [chatPreferenceBusy, setChatPreferenceBusy] = useState<"best-work" | "active" | null>(null);
  const [chatPreferenceStatus, setChatPreferenceStatus] = useState("");
  const chatPreferenceOwner = useRef({ brainId: brain.id, pending: false, active: true });
  if (chatPreferenceOwner.current.brainId !== brain.id) chatPreferenceOwner.current = { brainId: brain.id, pending: false, active: true };
  useEffect(() => {
    const owner = chatPreferenceOwner.current;
    owner.active = true;
    setChatPreferenceBusy(null);
    setChatPreferenceStatus("");
    return () => {
      owner.active = false;
    };
  }, [brain.id]);
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
  const ponderLedger = useMemo(() => new PonderPresentationLedger(brain.id), [brain.id]);
  const [ponderRevision, setPonderRevision] = useState(0);
  const [readablePonder, setReadablePonder] = useState(false);
  const [ponderError, setPonderError] = useState("");
  const finishPonder = useCallback((turnId: string) => {
    if (ponderLedger.finish(turnId)) setPonderRevision(value => value + 1);
  }, [ponderLedger]);
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
  const [outputRateClockMs, setOutputRateClockMs] = useState<number | null>(null);
  const [measuredOutput, setMeasuredOutput] = useState<{ tokens: number; rate?: number } | null>(null);
  const [cognitionProgress, setCognitionProgress] = useState<CognitionProgress | null>(null);
  const cognitionProgressRef = useRef<CognitionProgress | null>(null);
  const measuredOutputRef = useRef<{ tokens: number; rate?: number } | null>(null);
  const outputRateStartRef = useRef<{ at: number; count: number } | null>(null);
  const [chatQueue, setChatQueue] = useState<ChatQueueState | null>(null);
  const [chatDeliveryStatus, setChatDeliveryStatus] = useState("");
  const [inheritedFileReview, setInheritedFileReview] = useState<{
    brainId: string; executionId: string; review: InheritedTaskFileReview; input: string;
    busy: boolean; paused: boolean; error?: string;
  }>();
  const [inheritedReviewAttempt, setInheritedReviewAttempt] = useState(0);
  const waitingInheritedFileReview = Boolean(runtimeLoaded && inheritedFileReview?.brainId === brain.id &&
    inheritedFileReview.executionId === inheritedExecutionId);
  const inheritedFilePolicyKey = JSON.stringify(brain.toolPermissions?.find(permission => permission.toolId === "task.attachments") ?? null);
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
  const [preparingVoiceCall, setPreparingVoiceCall] = useState(false);
  const [savedReplyVoice, setSavedReplyVoice] = useState<{ messageId: string; phase: "preparing" | "playing" } | null>(null);
  const liveVoicePreferences: LiveVoicePreferences = liveVoiceState?.preferences ?? DEFAULT_LIVE_VOICE_PREFERENCES;

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
    if (!activeTurnId) {
      setChatOutputClockMs(null);
      setOutputRateClockMs(null);
      return;
    }
    const tick = (): void => {
      setChatOutputClockMs(Date.now());
      setOutputRateClockMs(performance.now());
      setMeasuredOutput(measuredOutputRef.current);
      setCognitionProgress(cognitionProgressRef.current);
    };
    tick();
    const timer = window.setInterval(tick, 250);
    return () => window.clearInterval(timer);
  }, [activeTurnId]);
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
  const touchBarRevisionRef = useRef(0);
  const touchBarScopeRef = useRef(0);
  const touchBarAcceptedStateRef = useRef<TouchBarState | null>(null);
  const touchBarActionContextRef = useRef<{
    input: ConversationTouchBarInput;
    submit(): void;
    stop(): void;
    call(): void;
  } | null>(null);
  const partialTextRef = useRef("");
  const generationPhasesRef = useRef(new Map<string, ChatGenerationPhase>());
  const committedTurnIdsRef = useRef(new Set<string>());
  const receivedInputTurnIdsRef = useRef(new Set<string>());
  const cancelledTurnIdsRef = useRef(new Set<string>());
  const steeredTurnIdsRef = useRef(new Set<string>());
  const suspendedTurnIdsRef = useRef(new Set<string>());
  const actionTurnIdsRef = useRef(new Map<string, string>());
  const currentBrainIdRef = useRef(brain.id);
  currentBrainIdRef.current = brain.id;
  const inheritedReviewScopeRef = useRef("");
  inheritedReviewScopeRef.current = `${brain.id}:${inheritedExecutionId ?? ""}:${runtimeLoaded}`;
  useLayoutEffect(() => {
    inheritedReviewScopeRef.current = `${brain.id}:${inheritedExecutionId ?? ""}:${runtimeLoaded}`;
    return () => { inheritedReviewScopeRef.current = "unmounted"; };
  }, [brain.id, inheritedExecutionId, runtimeLoaded]);
  const chatQueueRef = useRef<ChatQueueState | null>(null);
  const streamSequenceRef = useRef(new Map<string, number>());
  const tokenBatcherRef = useRef<TextFrameBatcher | null>(null);
  const textTurnGenerationRef = useRef(0);
  const activeHumanTurnsRef = useRef(new Map<
    string,
    { content: string; createdAt: string; attachmentIds?: string[] }
  >());
  const activePreviewIdRef = useRef<string | null>(null);
  const attachmentOperationGateRef = useRef(
    new ChatAttachmentOperationGate()
  );
  const liveVoiceControllerRef = useRef<LiveVoiceController | null>(null);
  const preparingVoiceRef = useRef<LiveVoiceController | null>(null);
  const instanceVoice = useInstanceVoice(brain.id, runtimeLoaded, callOpen, liveVoiceControllerRef, window.omni?.voice);
  const taskAttachmentDraft = useTaskAttachmentDraft(brain.id, window.omni?.chat.attachments);
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
    fallback?: { content: string; createdAt: string; attachmentIds?: string[] }
  ): Promise<void> => {
    const turn = activeHumanTurnsRef.current.get(turnId) ?? fallback;
    const recorder = window.omni?.chat.recordDeliveryReceipt;
    if (!turn || typeof recorder !== "function") return Promise.resolve();
    return recorder(brain.id, {
      schemaVersion: 1,
      turnId,
      content: turn.content,
      createdAt: turn.createdAt,
      ...(turn.attachmentIds?.length ? { attachmentIds: turn.attachmentIds } : {}),
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
      onNavigateRef.current(destination === "imagine" ? "chat" : destination);
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
      preferences: { ...DEFAULT_LIVE_VOICE_PREFERENCES },
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
        measuredOutputRef.current = null;
        setMeasuredOutput(null);
        cognitionProgressRef.current = null;
        setCognitionProgress(null);
        outputRateStartRef.current = null;
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
      if (preparingVoiceRef.current === controller) { preparingVoiceRef.current = null; setPreparingVoiceCall(false); }
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
    suspendedTurnIdsRef.current.clear();
    actionTurnIdsRef.current.clear();
    activeTurnIdRef.current = null;
    setActiveTurnId(null);
    setSending(false);
    partialTextRef.current = "";
    setPartialText("");
    setActiveTurnStartedAtMs(null);
    setChatOutputClockMs(null);
    measuredOutputRef.current = null;
    setMeasuredOutput(null);
    cognitionProgressRef.current = null;
    setCognitionProgress(null);
    outputRateStartRef.current = null;
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
        setMeasuredOutput(measuredOutputRef.current);
      },
      (callback) => window.requestAnimationFrame(callback),
      (handle) => window.cancelAnimationFrame(handle)
    );
    tokenBatcherRef.current = batcher;
    const finishGeneration = (turnId: string, createdAt: string, provisional: boolean): void => {
      finishPonder(turnId);
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
      cognitionProgressRef.current = null;
      setCognitionProgress(null);
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
        ponderLedger.recognize(event.actionEvent, event.turnId);
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
      if (event.type === "chat-ponder-self-report") {
        try { if (ponderLedger.receive(event.turnId, event.report)) setPonderRevision(value => value + 1); }
        catch (error) { setPonderError(error instanceof Error ? error.message : "Ponder publication could not be displayed."); }
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
          ["complete", "steered", "stopped", "no-reply", "failed", "cancelled", "suspended", "detached"].includes(event.state)) {
        const priorQueue = event.turnId === activeTurnIdRef.current ? chatQueueRef.current : null;
        const committed = committedTurnIdsRef.current.has(event.turnId);
        const received = receivedInputTurnIdsRef.current.has(event.turnId);
        finishGeneration(event.turnId, event.createdAt, !committed && event.state !== "complete");
        setPostReplyWork((current) => current.filter((work) => work.turnId !== event.turnId));
        if (event.state === "suspended" || event.state === "detached") {
          const suspendedState = event.state;
          suspendedTurnIdsRef.current.add(event.turnId);
          // Main alone certifies/persists this disposition. The renderer must
          // not turn its RPC unwind into a failed send or a fabricated Stop.
          setOptimisticHumans(current => settleOptimisticChatTurn(current, event.turnId, suspendedState));
          setUncommittedOutputs(current => settleUncommittedChatOutput(current, event.turnId, suspendedState));
          if (!activeTurnIdRef.current) setChatDeliveryStatus(event.suspension &&
            event.suspension.brainId === brain.id && event.suspension.turnId === event.turnId && event.suspension.state === event.state
            ? chatHostSuspensionStatus(event.suspension) : "Host attachment ended; the input is not resubmitted automatically.");
          void loadLatestConversation().catch(() => undefined);
        }
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
      if (event.type === "chat-cognition") {
        const prior = cognitionProgressRef.current;
        cognitionProgressRef.current = event.progress;
        if (!prior || prior.phase !== event.progress.phase || prior.active !== event.progress.active) {
          setCognitionProgress(event.progress);
        }
        return;
      }
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
        measuredOutputRef.current = null;
        setMeasuredOutput(null);
        cognitionProgressRef.current = null;
        setCognitionProgress(null);
        outputRateStartRef.current = null;
      } else if (event.type === "chat-token") {
        chatQueueRef.current = null;
        setChatQueue(null);
        // Keep frequent neural measurements out of React's update queue. Text
        // frames (or the 250ms clock for a non-printing token) publish the latest.
        const current = measuredOutputRef.current;
        if (event.outputTokens !== undefined && (!current || event.outputTokens >= current.tokens)) {
          measuredOutputRef.current = { tokens: event.outputTokens, rate: event.tokensPerSecond };
          if (!outputRateStartRef.current) {
            outputRateStartRef.current = { at: performance.now(), count: event.outputTokens };
          }
        }
        batcher.push(event.delta);
      }
    });
    return () => {
      removeListener();
      batcher.reset();
      if (tokenBatcherRef.current === batcher) tokenBatcherRef.current = null;
      streamSequenceRef.current.clear();
    };
  }, [brain.id, refreshWorkspaceSnapshot, ponderLedger, finishPonder]);

  useEffect(() => {
    if (!runtimeLoaded || (!resumeCheckpointId && !inheritedExecutionId) || !window.omni?.brain.resumePresentation) {
      setInheritedFileReview(undefined);
      return;
    }
    if (activeTurnIdRef.current && generationPhasesRef.current.has(activeTurnIdRef.current)) return;
    let active = true;
    const presentation = inheritedExecutionId
      ? window.omni.brain.inheritedResumePresentation(brain.id, inheritedExecutionId)
      : window.omni.brain.resumePresentation(brain.id);
    void presentation.then(async (attachment) => {
      if (!active || !attachment || (inheritedExecutionId ? attachment.inheritedExecutionId !== inheritedExecutionId : attachment.frontier.checkpointId !== resumeCheckpointId) ||
          generationPhasesRef.current.has(attachment.frontier.turnId)) return;
      if (inheritedExecutionId && !attachment.hostAttached) {
        // Authenticate and review BEFORE claiming active output/turn ownership.
        // An already-attached writer must not be re-entered by a review RPC.
        const review = await window.omni!.brain.prepareInheritedTaskFiles({ brainId: brain.id, executionId: inheritedExecutionId });
        if (!active) return;
        if (review?.state === "review-required") {
          setInheritedFileReview({ brainId: brain.id, executionId: inheritedExecutionId, review,
            input: attachment.input, busy: false, paused: false });
          return;
        }
      }
      setInheritedFileReview(undefined);
      const turnId = attachment.frontier.turnId;
      generationPhasesRef.current.set(turnId, "responding");
      streamSequenceRef.current.set(turnId, attachment.frontier.lastStreamSequence);
      receivedInputTurnIdsRef.current.add(turnId);
      activeHumanTurnsRef.current.set(turnId, { content: attachment.input, createdAt: new Date().toISOString() });
      activeTurnIdRef.current = turnId; setActiveTurnId(turnId);
      partialTextRef.current = attachment.publishedText; setPartialText(attachment.publishedText);
      setSending(true); setCancellingTurn(false); setActiveTurnStartedAtMs(Date.now());
      setChatDeliveryStatus(attachment.priorActions.length
        ? "Continuing the saved turn. Earlier action proposals are retained but are not replayed. External sessions require explicit reconnection."
        : "Continuing the exact saved turn; the accepted input is not submitted again.");
      if (!attachment.hostAttached) {
        const result = inheritedExecutionId
          ? await window.omni!.brain.resumeInheritedTurn({ brainId: brain.id, executionId: inheritedExecutionId })
          : await window.omni!.brain.resumeTurn({ brainId: brain.id, checkpointId: resumeCheckpointId! });
        if (active) onBrainChangeRef.current(result.brain);
      }
    }).catch((reason: unknown) => {
      if (!active) return;
      const error = reason instanceof Error ? reason.message : "Saved-turn reattachment is unavailable; the checkpoint is retained.";
      setChatDeliveryStatus(error);
      setInheritedFileReview(current => current?.brainId === brain.id && current.executionId === inheritedExecutionId
        ? { ...current, busy: false, error } : current);
    });
    return () => { active = false; }; // Navigation does not Cancel the main-owned continuation.
  }, [brain.id, runtimeLoaded, resumeCheckpointId, inheritedExecutionId, inheritedReviewAttempt, inheritedFilePolicyKey]);

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
    if (waitingInheritedFileReview) { onToast("Review the saved task's files first, or leave that task paused. Your draft is retained."); return; }
    if (taskAttachmentDraft.busy && !queuedTurn) { onToast("Wait for attachment preparation, or stop it before sending."); return; }
    if (!runtimeLoaded) { onToast("Load this instance to send or continue neural work. Its saved turn is retained."); return; }
    const text = (queuedTurn?.text ?? input).trim();
    const capacity = chatInputCapacity(text, brain.config.contextWindowTokens);
    if (text && !capacity.fits) {
      if (queuedTurn) setInput((current) => recoverFailedChatDraft(current, text));
      onToast(capacity.reason!);
      return;
    }
    const steering = turnMetadata?.kind === "steer";
    if (!text || (!steering && (sending || activeTurnIdRef.current)) ||
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
    const createdAt = queuedTurn?.createdAt ?? turnMetadata?.createdAt ?? new Date().toISOString();
    const attachments = queuedTurn?.attachments ?? taskAttachmentDraft.take();
    const attachmentIds = attachments.map(value => value.id);
    const metadata: ChatTurnMetadata | undefined = attachmentIds.length
      ? turnMetadata?.kind === "steer" ? { ...turnMetadata, attachmentIds }
        : { kind: "task", source: "human", createdAt, attachmentIds }
      : turnMetadata;
    generationPhasesRef.current.set(turnId, "responding");
    const optimisticHuman: ChatMessage = {
      id: queuedTurn ? `queued-${turnId}` : `pending-${turnId}`,
      role: "human",
      content: text,
      createdAt,
      turnId,
      ...(attachmentIds.length ? { taskAttachmentIds: attachmentIds } : {})
    };
    activeHumanTurnsRef.current.set(turnId, { content: text, createdAt, attachmentIds });
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
        ...(turnMetadata?.kind === "steer"
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
    outputRateStartRef.current = null;
    measuredOutputRef.current = null;
    setMeasuredOutput(null);
    cognitionProgressRef.current = null;
    setCognitionProgress(null);
    setOutputRateClockMs(null);
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
          metadata
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
      if (steeredTurnIdsRef.current.has(turnId) || suspendedTurnIdsRef.current.has(turnId)) return;
      if (!committedTurnIdsRef.current.has(turnId)) {
        const saved = savedChatSuspension(await loadLatestConversation().catch(() => []), brain.id, turnId);
        if (currentBrainIdRef.current !== brain.id) return;
        if (saved) {
          suspendedTurnIdsRef.current.add(turnId);
          setOptimisticHumans(current => settleOptimisticChatTurn(current, turnId, saved.state));
          if (activeTurnIdRef.current === turnId) {
            tokenBatcherRef.current?.flush();
            const publishedPrefix = partialTextRef.current;
            setUncommittedOutputs(current => settleUncommittedChatOutput(retainUncommittedChatOutput(
              current, turnId, publishedPrefix, saved.createdAt), turnId, saved.state));
          } else {
            setUncommittedOutputs(current => settleUncommittedChatOutput(current, turnId, saved.state));
          }
          if (!activeTurnIdRef.current || activeTurnIdRef.current === turnId) setChatDeliveryStatus(chatHostSuspensionStatus(saved));
          return;
        }
      }
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
        void taskAttachmentDraft.restore(attachmentIds).catch(() => undefined);
        onToast(error instanceof Error ? error.message : "The local brain could not respond.");
      }
    } finally {
      if (currentBrainIdRef.current !== brain.id) return;
      finishPonder(turnId);
      setPostReplyWork((current) => current.filter((work) => work.turnId !== turnId));
      // Late frames for retired turns are ignored by the stream's known-turn
      // guard; retain metadata only while its real work/result is outstanding.
      generationPhasesRef.current.delete(turnId);
      committedTurnIdsRef.current.delete(turnId);
      receivedInputTurnIdsRef.current.delete(turnId);
      cancelledTurnIdsRef.current.delete(turnId);
      steeredTurnIdsRef.current.delete(turnId);
      suspendedTurnIdsRef.current.delete(turnId);
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

  const declinePendingTool = async (event: ActionEvent, instruction?: string): Promise<void> => {
    const pending = pendingChatToolFromAction(event);
    if (!pending || !window.omni) return;
    const result = await window.omni.chat.declineAction({ brainId: brain.id,
      actionEventId: event.id, approvalToken: pending.approvalToken });
    setActionEvents(current => mergeChatActionEvent(current, result));
    setPendingTool(current => current?.actionEventId === event.id ? null : current);
    if (instruction) setInput(current => recoverFailedChatDraft(current, instruction));
  };

  const steerCurrentTurn = async (queuedTurn?: QueuedChatTurn): Promise<void> => {
    if (cancellingTurn) return;
    if (!runtimeLoaded) { onToast("Load this instance before steering; the saved task is retained."); return; }
    if (taskAttachmentDraft.busy && !queuedTurn) { onToast("Wait for attachment preparation, or stop it before steering."); return; }
    const text = queuedTurn?.text ?? input;
    if (!text.trim()) return;
    const capacity = chatInputCapacity(text, brain.config.contextWindowTokens);
    if (!capacity.fits) { onToast(capacity.reason!); return; }
    const replacesTurnId = activeTurnIdRef.current;
    if (!replacesTurnId) {
      await send(undefined, queuedTurn);
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
      createdAt: queuedTurn?.createdAt ?? new Date().toISOString()
    }, queuedTurn);
  };

  const queueCurrentTurn = (): void => {
    if (taskAttachmentDraft.busy) { onToast("Wait for attachment preparation, or stop it before queuing."); return; }
    const text = input.trim();
    if (!text) return;
    const capacity = chatInputCapacity(text, brain.config.contextWindowTokens);
    if (!capacity.fits) { onToast(capacity.reason!); return; }
    const id = globalThis.crypto?.randomUUID?.() ??
      `queued-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const createdAt = new Date().toISOString();
    const turn: QueuedChatTurn = { id, text, createdAt, attachments: taskAttachmentDraft.take() };
    setInput((current) => clearSubmittedChatDraft(current, text));
    void persistDeliveryReceipt(id, "queued", {
      content: text,
      createdAt,
      attachmentIds: turn.attachments?.map(value => value.id)
    });
    setQueuedTurns((current) => [...current, turn]);
    setOptimisticHumans((current) => [
      ...current,
      { id: `queued-${id}`, role: "human", content: text, createdAt, turnId: id,
        taskAttachmentIds: turn.attachments?.map(value => value.id) }
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
    void taskAttachmentDraft.restore(queuedTurns.flatMap(turn => turn.attachments?.map(value => value.id) ?? [])).catch(() => undefined);
    queuedTurns.forEach((turn) => {
      void persistDeliveryReceipt(turn.id, "cancelled", {
        content: turn.text,
        createdAt: turn.createdAt,
        attachmentIds: turn.attachments?.map(value => value.id)
      });
    });
    const ids = new Set(queuedTurns.map((turn) => `queued-${turn.id}`));
    setOptimisticHumans((current) =>
      current.filter((message) => !ids.has(message.id))
    );
    setQueuedTurns([]);
  };

  const editQueuedTurn = (id: string, text: string): void => {
    const turn = queuedTurns.find(item => item.id === id);
    if (!turn) return;
    setQueuedTurns(current => current.map(item => item.id === id ? { ...item, text } : item));
    setOptimisticHumans(current => current.map(item => item.id === `queued-${id}` ? { ...item, content: text } : item));
    void persistDeliveryReceipt(id, "queued", { content: text, createdAt: turn.createdAt, attachmentIds: turn.attachments?.map(value => value.id) });
  };
  const removeQueuedTurn = (id: string): void => {
    const turn = queuedTurns.find(item => item.id === id);
    if (!turn) return;
    void taskAttachmentDraft.restore(turn.attachments?.map(value => value.id) ?? []).catch(() => undefined);
    setQueuedTurns(current => current.filter(item => item.id !== id));
    setOptimisticHumans(current => current.filter(item => item.id !== `queued-${id}`));
    void persistDeliveryReceipt(id, "cancelled", { content: turn.text, createdAt: turn.createdAt, attachmentIds: turn.attachments?.map(value => value.id) });
  };
  const moveQueuedTurn = (id: string, direction: -1 | 1): void => {
    setQueuedTurns(current => {
      const index = current.findIndex(item => item.id === id);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      [next[index], next[nextIndex]] = [next[nextIndex]!, next[index]!];
      return next;
    });
  };
  const sendQueuedNow = (id: string): void => {
    const turn = queuedTurns.find(item => item.id === id);
    if (!turn || cancellingTurn) return;
    if (waitingInheritedFileReview) { onToast("The queued message is retained while the saved task's files await review."); return; }
    if (!runtimeLoaded) { onToast("Load this instance to send a queued message. The queue is retained."); return; }
    const capacity = chatInputCapacity(turn.text, brain.config.contextWindowTokens);
    if (!capacity.fits) { onToast(capacity.reason!); return; }
    const active = activeTurnIdRef.current;
    if (sending && !active) return;
    setQueuedTurns(current => current.filter(item => item.id !== id));
    void (active ? steerCurrentTurn(turn) : send(undefined, turn));
  };

  useEffect(() => {
    if (waitingInheritedFileReview || sending || activeTurnIdRef.current || liveVoiceGenerationActive || !queuedTurns.length) return;
    const next = queuedTurns[0]!;
    setQueuedTurns((current) =>
      current[0]?.id === next.id
        ? current.slice(1)
        : current.filter((turn) => turn.id !== next.id)
    );
    void send(undefined, next);
  }, [sending, activeTurnId, liveVoiceGenerationActive, queuedTurns, waitingInheritedFileReview]);

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
      if (preparingVoiceRef.current === controller) return;
      if (!runtimeLoaded) { onToast("Load the instance before starting a new voice session."); return; }
      // Saved-reply waveform output is an independent job, not a voice turn.
      // Stop that selected output before opening a microphone/recognizer.
      savedReplyVoiceGenerationRef.current += 1;
      savedReplySessionRef.current?.cancel();
      savedReplySessionRef.current = null;
      setSavedReplyVoice(null);
      preparingVoiceRef.current = controller; setPreparingVoiceCall(true);
      try {
        await instanceVoice.prepareCall();
        if (liveVoiceControllerRef.current === controller) await controller.start();
      } catch (failure) {
        if (liveVoiceControllerRef.current === controller) onToast(conciseUiMessage(failure, "This voice call could not start."));
      } finally {
        if (preparingVoiceRef.current === controller) { preparingVoiceRef.current = null; setPreparingVoiceCall(false); }
      }
    }
  };

  const bargeInLiveVoice = (): void => {
    liveVoiceControllerRef.current?.interrupt();
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
  useEffect(() => {
    // Drop derived live copies only after exact saved message/action custody
    // exists. Historical reports remain in the paged conversation ledger.
    if (ponderLedger.retirePersisted([...persistedMessages, ...sessionCommittedMessages], persistedActions))
      setPonderRevision(value => value + 1);
  }, [ponderLedger, ponderRevision, persistedMessages, sessionCommittedMessages, persistedActions]);
  const directionTurnIds = useMemo(() => [...new Set(
    [...persistedMessages, ...sessionCommittedMessages, ...optimisticHumans]
      .filter(message => message.role === "human")
      .flatMap(message => message.turnId ?? message.deliveryReceipt?.turnId ?? [])
  )], [persistedMessages, sessionCommittedMessages, optimisticHumans]);
  const directions = useChatDirections(brain.id, directionTurnIds, window.omni?.chat.directions);
  const displayedMessages = useMemo(() => {
    const authoritative = window.omni
      ? mergeCommittedChatMessages(persistedMessages, sessionCommittedMessages)
      : brain.messages;
    const combined = mergeChatMessagesForPresentation(
      authoritative,
      optimisticHumans
    );
    const queuedIds = new Set(queuedTurns.map(turn => `queued-${turn.id}`));
    const visible = [...combined.filter(message => !queuedIds.has(message.id) &&
      !isSteerDirectionMessage(message, directions.records)), ...uncommittedOutputs.map((output) => output.message)];
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
  }, [brain.createdAt, brain.messages, brain.config.runtime, optimisticHumans, persistedMessages, sessionCommittedMessages, uncommittedOutputs, queuedTurns, directions.records]);
  const displayedActions = useMemo(() => {
    const merged = new Map(persistedActions.map((event) => [event.id, event]));
    actionEvents.forEach((event) => merged.set(event.id, event));
    return [...merged.values()];
  }, [actionEvents, persistedActions]);
  const ponderPresentation = useMemo(() => reconcilePonderPresentation(ponderLedger.snapshot(), displayedMessages, displayedActions),
    [ponderLedger, ponderRevision, displayedMessages, displayedActions]);
  const displayedActionIds = useMemo(() => new Set(displayedActions.map(event => event.neuralActionId).filter(Boolean)), [displayedActions]);
  const ponderActionGroups = useMemo(() => groupPonderActions(displayedActions), [displayedActions]);
  const publicReplySources = useMemo<PublicReplySource[]>(() => [
    ...displayedMessages.flatMap(message => message.role === "brain" && message.turnId
      ? [{ key: `message-${message.id}`, turnId: message.turnId, messageId: message.id, content: message.content, live: false }] : []),
    ...(activeTurnId ? [{ key: `live-${activeTurnId}`, turnId: activeTurnId, content: partialText, live: true }] : [])
  ], [displayedMessages, activeTurnId, partialText]);
  const [replyLocationProof, setReplyLocationProof] = useState<{
    brainId: string; sources: PublicReplySource[]; events: ActionEvent[]; locations: Map<string, VerifiedReplyLocation>;
  }>();
  useEffect(() => {
    if (!displayedActions.some(event => event.replyAnchor)) {
      setReplyLocationProof(current => current === undefined ? current : undefined);
      return;
    }
    let current = true;
    void verifyReplyActionLocations(brain.id, publicReplySources, displayedActions).then(locations => {
      if (current) setReplyLocationProof({ brainId: brain.id, sources: publicReplySources, events: displayedActions, locations });
    }).catch(() => {
      // A missing digest never authorizes rearranging history. Older/unbound
      // events remain visible in their ordinary flat timeline instead.
      if (current) setReplyLocationProof(undefined);
    });
    return () => { current = false; };
  }, [brain.id, publicReplySources, displayedActions]);
  const replyActionLocations = useMemo(() => {
    const result = new Map<string, VerifiedReplyLocation>();
    if (replyLocationProof?.brainId !== brain.id) return result;
    const priorEvents = new Map(replyLocationProof.events.map(event => [event.id, event]));
    const priorSources = new Map(replyLocationProof.sources.map(source => [source.key, source]));
    const sources = new Map(publicReplySources.map(source => [source.key, source]));
    for (const event of displayedActions) {
      const location = replyLocationProof.locations.get(event.id), priorEvent = priorEvents.get(event.id);
      if (!location || !priorEvent || JSON.stringify(event.replyAnchor) !== JSON.stringify(priorEvent.replyAnchor) ||
          event.chatTurnId !== priorEvent.chatTurnId || event.brainId !== priorEvent.brainId) continue;
      const source = sources.get(location.sourceKey), prior = priorSources.get(location.sourceKey);
      // New output may append while hashing is in flight. Reuse a verified
      // position only while its exact public prefix and ownership are intact.
      if (source && prior && source.turnId === prior.turnId && source.messageId === prior.messageId && source.live === prior.live &&
          source.content.length >= location.utf16Offset &&
          source.content.slice(0, location.utf16Offset) === prior.content.slice(0, location.utf16Offset)) result.set(event.id, location);
    }
    return result;
  }, [brain.id, replyLocationProof, displayedActions, publicReplySources]);
  const replyActionsBySource = useMemo(() => {
    const result = new Map<string, ActionEvent[]>();
    for (const event of ponderActionGroups.roots) {
      const location = replyActionLocations.get(event.id);
      if (!location) continue;
      const events = result.get(location.sourceKey) ?? [];
      events.push(event); result.set(location.sourceKey, events);
    }
    return result;
  }, [ponderActionGroups, replyActionLocations]);
  const timelineEntries = useMemo(() => [
      ...displayedMessages.map((message, order) => ({
        kind: "message" as const,
        id: `message-${message.id}`,
        createdAt: message.createdAt,
        order,
        message
      })),
      ...ponderActionGroups.roots.filter(event => !replyActionLocations.has(event.id)).map((event, order) => ({
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
      ponderActionGroups,
      replyActionLocations,
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
    const parentByChild = new Map([...ponderActionGroups.childrenByParent].flatMap(([parentId, children]) => children.map(child => [child.id, parentId] as const)));
    const pinnedIds = new Set([
      ...optimisticHumans.slice(-20).map((message) => `message-${message.id}`),
      ...sessionCommittedMessages.slice(-20).map((message) => `message-${message.id}`),
      ...actionEvents.slice(-20).map((event) => {
        const rootId = parentByChild.get(event.id) ?? event.id;
        return replyActionLocations.get(rootId)?.sourceKey ?? `action-${rootId}`;
      }),
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
    ponderActionGroups,
    replyActionLocations,
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
  const touchBarDraftHasContent = Boolean(input.trim());
  const touchBarInput: ConversationTouchBarInput = {
    revision: Math.max(1, touchBarRevisionRef.current),
    brainId: brain.id,
    instanceName: brain.name,
    visible: !hidden,
    runtimeLoaded,
    draftHasContent: touchBarDraftHasContent,
    draftFits: draftCapacity.fits,
    draftBusy: taskAttachmentDraft.busy,
    inheritedReviewPending: waitingInheritedFileReview,
    turnActive,
    activeTurnId: activeTurnIdRef.current ?? activeTurnId,
    cancellingTurn,
    callAvailable: Boolean(liveVoiceControllerRef.current),
    callActive: Boolean(liveVoiceState?.enabled),
    callBusy: preparingVoiceCall
  };
  touchBarActionContextRef.current = {
    input: touchBarInput,
    submit: submitOrdinaryTurn,
    stop: () => { void cancelChatTool(); },
    call: () => { void toggleLiveVoice(); }
  };

  useEffect(() => {
    const touchBar = window.omni?.window.touchBar;
    if (!touchBar) return;
    return touchBar.onAction((event) => {
      const accepted = touchBarAcceptedStateRef.current;
      const current = touchBarActionContextRef.current;
      const matched = matchTouchBarAction(event, accepted);
      if (!accepted || !current || accepted.brainId !== current.input.brainId || !matched) return;
      const liveVoiceController = liveVoiceControllerRef.current;
      const activeTurn = activeTurnIdRef.current;
      const latestInput: ConversationTouchBarInput = {
        ...current.input,
        turnActive: current.input.turnActive || Boolean(activeTurn),
        activeTurnId: activeTurn,
        callAvailable: Boolean(liveVoiceController),
        callActive: Boolean(liveVoiceController?.getState().enabled),
        callBusy: current.input.callBusy || Boolean(preparingVoiceRef.current)
      };
      if (!touchBarActionRemainsReady(matched, latestInput)) return;
      if (matched.action === "stop" && activeTurn !== matched.turnId) return;
      // Consume this exact accepted revision before dispatch so repeated native
      // callbacks cannot operate while React publishes the next semantic state.
      touchBarAcceptedStateRef.current = null;
      if (matched.action === "submit") current.submit();
      else if (matched.action === "stop") current.stop();
      else current.call();
    });
  }, [brain.id]);

  useEffect(() => {
    const touchBar = window.omni?.window.touchBar;
    const scope = ++touchBarScopeRef.current;
    if (!touchBar) {
      touchBarAcceptedStateRef.current = null;
      return;
    }
    const current = touchBarActionContextRef.current;
    if (!current || current.input.brainId !== brain.id) return;
    const revision = ++touchBarRevisionRef.current;
    const proposed = conversationTouchBarState({ ...current.input, revision });
    if (!proposed) {
      touchBarAcceptedStateRef.current = null;
      void touchBar.update(null).catch(() => undefined);
      return;
    }
    void touchBar.update(proposed).then((result) => {
      const latest = touchBarActionContextRef.current;
      if (
        scope !== touchBarScopeRef.current ||
        revision !== touchBarRevisionRef.current ||
        latest?.input.brainId !== proposed.brainId
      ) return;
      if (result.accepted && result.revision === proposed.revision) {
        touchBarAcceptedStateRef.current = proposed;
      }
    }).catch(() => undefined);
  }, [
    activeTurnId,
    brain.id,
    brain.name,
    cancellingTurn,
    draftCapacity.fits,
    hidden,
    liveVoiceState?.enabled,
    preparingVoiceCall,
    runtimeLoaded,
    taskAttachmentDraft.busy,
    touchBarDraftHasContent,
    turnActive,
    waitingInheritedFileReview
  ]);

  useEffect(() => {
    const touchBar = window.omni?.window.touchBar;
    return () => {
      ++touchBarScopeRef.current;
      touchBarAcceptedStateRef.current = null;
      void touchBar?.update(null).catch(() => undefined);
    };
  }, [brain.id]);

  const responseTokenCount = measuredOutput?.tokens ?? 0;
  const responseOutputPresentation = pendingChatOutputPresentation({
    outputTokens: responseTokenCount,
    startedAtMs: activeTurnStartedAtMs,
    nowMs: chatOutputClockMs,
    queue: chatQueue,
    streamStartedAtMs: outputRateStartRef.current?.at,
    streamNowMs: outputRateClockMs,
    initialOutputTokens: outputRateStartRef.current?.count,
    measuredTokensPerSecond: measuredOutput?.rate
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

  const renderChatAction = (event: ActionEvent, inlineMedia = true): ReactNode => {
    const children = ponderActionGroups.childrenByParent.get(event.id) ?? [];
    const waiting = children.filter(child => pendingChatToolFromAction(child)).length;
    const running = children.filter(child => child.state === "running" || child.state === "proposed").length;
    const childStatus = waiting ? `${waiting} awaiting approval` : running ? `${running} active action${running === 1 ? "" : "s"}` : undefined;
    return <Fragment key={event.id}>
      <ChatActionCard event={event} inlineMedia={inlineMedia} childStatus={childStatus}
        ponderReports={ponderPresentation.byAction.get(event.id)} readablePonder={readablePonder}
        onReadablePonder={setReadablePonder} flushPonder={Boolean(hidden)}
        onDecline={pendingChatToolFromAction(event) ? instruction => declinePendingTool(event, instruction) : undefined}
        onCancel={chatActionCancellationTarget(event, actionTurnIdsRef.current.get(event.id)) ? () => cancelActionJob(event) : undefined}
        onApprove={pendingChatToolFromAction(event) ? () => approvePendingTool(pendingChatToolFromAction(event)) : undefined}>
        {children.length ? <div className="chat-ponder-actions" aria-label="Actions selected during this Ponder">
          {children.map(child => renderChatAction(child, false))}
        </div> : null}
      </ChatActionCard>
      {/* Generated output stays visible even while the Ponder audit is closed. */}
      {children.map(child => <ChatImaginationMedia key={`media-${child.id}`} event={child} />)}
    </Fragment>;
  };
  const renderPublicReply = (sourceKey: string, content: string): ReactNode | undefined => {
    const events = replyActionsBySource.get(sourceKey);
    if (!events?.length) return undefined;
    const offsets = new Map(events.flatMap(event => {
      const location = replyActionLocations.get(event.id);
      return location ? [[event.id, location.utf16Offset] as const] : [];
    }));
    const { pieces } = splitPublicReplyAtActions(content, events, offsets);
    return pieces.map(piece => piece.kind === "text"
      ? <span className="message__text-segment" key={`text-${piece.start}`}>{piece.text}</span>
      : <div className="message__reply-action" key={`action-${piece.event.id}`}>{renderChatAction(piece.event)}</div>);
  };
  const liveReplyKey = activeTurnId ? `live-${activeTurnId}` : "";
  const liveReplyHasActions = Boolean(replyActionsBySource.get(liveReplyKey)?.length);

  const changeChatPreference = async (kind: "best-work" | "active", enabled: boolean): Promise<void> => {
    const owner = chatPreferenceOwner.current;
    if (!window.omni || !owner.active || owner.pending || turnActive) return;
    owner.pending = true;
    setChatPreferenceBusy(kind);
    setChatPreferenceStatus("");
    try {
      if (kind === "active") {
        const result = await window.omni.brain.setActiveMode(brain.id, enabled);
        if (chatPreferenceOwner.current !== owner || !owner.active) return;
        await onActiveModeChange(result);
        if (chatPreferenceOwner.current !== owner || !owner.active) return;
        const switched = result.deactivated.map(item => item.name).join(", ");
        setChatPreferenceStatus(enabled && switched ? `Active mode enabled; ${switched} switched off.` : `Active mode ${enabled ? "enabled" : "off"}.`);
      } else {
        const updated = await window.omni.brain.update(brain.id, { ...brain.config, bestWorkEnabled: enabled });
        if (chatPreferenceOwner.current !== owner || !owner.active) return;
        onBrainChange(updated);
        setChatPreferenceStatus(`Best Work ${enabled ? "enabled" : "off"}.`);
      }
    } catch (error) {
      if (chatPreferenceOwner.current === owner && owner.active) {
        const message = error instanceof Error ? error.message : "Chat preference could not be saved.";
        setChatPreferenceStatus(message);
        onToast(message);
      }
    } finally {
      owner.pending = false;
      if (chatPreferenceOwner.current === owner && owner.active) setChatPreferenceBusy(null);
    }
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
          setAttachmentDragActive(false);
          void taskAttachmentDraft.stage("files", Array.from(event.dataTransfer.files));
        }}
      >
        {attachmentDragActive ? (
          <div className="chat-attachment-drop" role="status">
            <span><Icon name="upload" size={24} /></span>
            <strong>Attach files, images, audio, or video</strong>
            <small>Attached for this task. Choose Learn to train from a source.</small>
          </div>
        ) : null}
        <div className="chat-call-toolbar">
          <div className="chat-work-preferences" role="group" aria-label="Chat modes">
            <label title={turnActive ? "Change after the current reply finishes" : "Use extra computation on harder replies"}>
              <input type="checkbox" checked={brain.config.bestWorkEnabled ?? false}
                disabled={!window.omni || Boolean(chatPreferenceBusy) || turnActive}
                onChange={event => void changeChatPreference("best-work", event.target.checked)} />
              <span>Best Work</span>
            </label>
            <label title={turnActive ? "Change after the current reply finishes" : "Allow this brain to work during idle time; existing tool permissions still apply"}>
              <input type="checkbox" checked={brain.config.idleCognition}
                disabled={!window.omni || Boolean(chatPreferenceBusy) || turnActive}
                onChange={event => void changeChatPreference("active", event.target.checked)} />
              <span>Active mode</span>
            </label>
          </div>
          <button type="button" aria-expanded={callOpen} aria-controls="chat-call-panel"
            onClick={() => setCallOpen(current => !current)} title="Voice, camera and screen sharing">
            <Icon name="microphone" size={15} /> Call
            {liveVoiceState?.enabled ? <i aria-label="Call active" /> : null}
          </button>
          <button type="button" aria-expanded={captureOpen} aria-controls="chat-capture-panel"
            onClick={() => setCaptureOpen(current => !current)} title="Record recent and ongoing spontaneous imagination">
            <Icon name="video" size={15} /> Record
          </button>
        </div>
        {chatPreferenceStatus ? <p className="chat-preference-status" role="status" aria-live="polite">{chatPreferenceStatus}</p> : null}
        <div id="chat-capture-panel" className="chat-capture-panel" hidden={!captureOpen}>
          {captureOpen && !hidden ? <ImaginationCaptureControls brainId={brain.id} api={window.omni?.modality.capture} /> : null}
        </div>
        <CallLayout open={callOpen} controls={<section id="chat-call-panel" className="chat-call-panel" hidden={!callOpen} aria-label="Call controls">
          <button type="button" className="chat-call-start" onClick={() => void toggleLiveVoice()}
            disabled={preparingVoiceCall || !runtimeLoaded && !liveVoiceState?.enabled}
            aria-label={liveVoiceState?.enabled ? "Stop live voice" : "Start live voice"}>
            <Icon name="microphone" size={15} /> {preparingVoiceCall ? "Preparing call…" : liveVoiceState?.enabled ? "End voice call" : "Start voice call"}
          </button>
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
          {instanceVoice.error ? <p role="alert">{instanceVoice.error}</p> : null}
          {!instanceVoice.error || instanceVoice.snapshot ? <VoiceRoutingControls key={brain.id} snapshot={instanceVoice.snapshot}
            capabilities={liveVoiceState?.capabilities} callActive={Boolean(liveVoiceState?.enabled) || preparingVoiceCall}
            utteranceActive={Boolean(liveVoiceState?.activeUtterance)} onSave={instanceVoice.save} /> : null}
          <LivePerceptionPanel brain={brain} runtimeLoaded={runtimeLoaded} />
        </section>}>
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
          <PluginRuntimeSlot slot="chat.before" />
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
                Showing {visibleTimelineEntries.length.toLocaleString()} conversation rows · {conversationTotal.toLocaleString()} persisted records
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
              <>
              {entry.message.ponderSelfReports?.filter(report => !displayedActionIds.has(report.actionId)).map(report => <details key={report.id}
                className="chat-action-card"><summary>Ponder</summary><PonderSelfReportView row={{ report, live: false }}
                  readable={readablePonder} onReadableChange={setReadablePonder} flush /></details>)}
              <MessageBubble
                message={entry.message}
                contentProjection={renderPublicReply(entry.id, entry.message.content)}
                brainId={brain.id}
                brainName={brain.name}
                onOwnVoice={() => playSavedReplyWithOwnVoice(entry.message)}
                ownVoicePhase={savedReplyVoice?.messageId === entry.message.id ? savedReplyVoice.phase : undefined}
                ownVoiceUnavailable={!liveVoiceState?.capabilities.neuralVoice || liveVoiceState.enabled}
                ownVoiceQuality={liveVoiceState?.capabilities.neuralVoiceQuality}
                uncommittedOutputState={uncommittedOutputs.find((output) =>
                  output.message.id === entry.message.id)?.state}
              />
              </>
            ) : entry.kind === "action" ? (
              <div className="chat-action-stream" aria-label="Visible chat action">
                {renderChatAction(entry.event)}
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
          {ponderError || ponderPresentation.error ? <p role="alert">{ponderError || ponderPresentation.error}</p> : null}
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
                          : cognitionProgress?.active
                            ? cognitionProgress.phase === "ponder" ? "Pondering" : "Best Work"
                          : pondering
                            ? "Pondering"
                            : actionRunning
                              ? "Running action"
                              : "Responding"
                    }
                  </span>
                  <ResponseOutputCounter presentation={responseOutputPresentation} cognitionWithoutOutput={cognitionProgress?.active} />
                </div>
                {partialText || liveReplyHasActions ? (
                  <div
                    className="message__streaming-text"
                    aria-live="polite"
                  >
                    {renderPublicReply(liveReplyKey, partialText) ?? partialText}
                  </div>
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
                {cognitionProgress ? <CognitionProgressView progress={cognitionProgress} /> : null}
              </div>
            </div>
          ) : null}
            <PluginRuntimeSlot slot="chat.after" />
            <div ref={messagesEnd} />
          </div>
          {!followingOutput || !timelineAtLatest ? (
            <button className="chat-jump-latest" onClick={followLatest}>
              <Icon name="arrow" size={13} /> Latest activity
            </button>
          ) : null}
        </div>
        <div className="composer-wrap">
          {window.omni?.questions ? <UserQuestionPanel brainId={brain.id} api={window.omni.questions} presentation="conversation" /> : null}
          {waitingInheritedFileReview && inheritedFileReview ? <InheritedTaskFilesReview
            review={inheritedFileReview.review} input={inheritedFileReview.input}
            busy={inheritedFileReview.busy} paused={inheritedFileReview.paused} error={inheritedFileReview.error}
            onPause={() => setInheritedFileReview(current => current ? { ...current, paused: true } : current)}
            onReviewAgain={() => {
              setInheritedFileReview(current => current ? { ...current, busy: true, error: undefined } : current);
              setInheritedReviewAttempt(current => current + 1);
            }}
            onApprove={() => {
              const pending = inheritedFileReview;
              const owner = inheritedReviewScopeRef.current;
              if (!window.omni || pending.busy || pending.paused) return;
              setInheritedFileReview(current => current === pending ? { ...current, busy: true, error: undefined } : current);
              void window.omni.brain.approveInheritedTaskFiles({ brainId: brain.id,
                reviewId: pending.review.reviewId, reviewSha256: pending.review.reviewSha256 }).then(receipt => {
                if (inheritedReviewScopeRef.current !== owner) return;
                if (receipt.brainId !== pending.brainId || receipt.executionId !== pending.executionId ||
                    receipt.reviewSha256 !== pending.review.reviewSha256) throw new Error("The file adoption receipt does not match this exact review.");
                setInheritedFileReview(current => current?.review.reviewSha256 === pending.review.reviewSha256 ? undefined : current);
                setInheritedReviewAttempt(current => current + 1);
              }).catch(failure => {
                if (inheritedReviewScopeRef.current !== owner) return;
                setInheritedFileReview(current => current?.review.reviewSha256 === pending.review.reviewSha256
                  ? { ...current, busy: false, error: failure instanceof Error ? failure.message : String(failure) } : current);
              });
            }} /> : null}
          <ChatDirectionStrip key={brain.id} items={directions.records.map(chatDirectionStripItem)} total={directions.total}
            canSteer={runtimeLoaded && composerTurn.steerAvailable && !cancellingTurn}
            busy={directions.busy} hasOlder={directions.hasOlder}
            onOlder={() => void directions.older()} onDismiss={directions.dismiss}
            onRevise={async text => {
              const content = text.trim();
              const capacity = chatInputCapacity(content, brain.config.contextWindowTokens);
              if (!capacity.fits) throw new Error(capacity.reason!);
              if (runtimeLoaded && composerTurn.steerAvailable && !cancellingTurn && activeTurnIdRef.current) {
                // A revision is a fresh direction, never an edit to accepted neural input.
                // Empty attachment selection preserves the current unsent draft's files.
                await steerCurrentTurn({ id: globalThis.crypto.randomUUID(), text: content,
                  createdAt: new Date().toISOString(), attachments: [] });
              } else {
                setInput(current => recoverFailedChatDraft(current, content));
              }
            }} />
          {directions.error ? <p role="alert" className="task-attachment-status">{directions.error}</p> : null}
          <TaskAttachmentStrip entries={taskAttachmentDraft.entries} onRemove={id => void taskAttachmentDraft.remove(id)} />
          {taskAttachmentDraft.status ? <div className="task-attachment-status" role="status" aria-live="polite">
            <span>{taskAttachmentDraft.status}</span>{taskAttachmentDraft.busy ? <button type="button" className="button button--ghost" onClick={taskAttachmentDraft.cancel}>Stop preparing</button> : null}
          </div> : null}
          <ChatQueueStack items={queuedTurns} busy={turnActive}
            onEdit={editQueuedTurn} onRemove={removeQueuedTurn}
            onMove={moveQueuedTurn} onSendNow={sendQueuedNow} />
          {queuedLearningJobCount ? <div className="chat-queue-learning" role="status">
            {queuedLearningJobCount} learning task{queuedLearningJobCount === 1 ? "" : "s"} queued
          </div> : null}
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
              <p>{pendingTool ? chatApprovalStatus(pendingTool.label, pendingTool.approvalExpiresAt, approvalClock) : toolStatus}</p>
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
              readOnly={!runtimeLoaded}
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
              <div className="composer__source-actions">
                <UniversalSourcePicker key={`attach-${brain.id}`} label="Attach" active={!hidden}
                  disabled={taskAttachmentDraft.busy || !window.omni}
                  onSelect={selection => void taskAttachmentDraft.stage(selection)} />
                <UniversalSourcePicker key={`learn-${brain.id}`} label="Learn" icon="brain" active={!hidden}
                  disabled={attaching || !runtimeLoaded || !window.omni}
                  onSelect={selection => void attachExperience(selection)} />
              </div>
              <span className="composer__input-count" title={composerTelemetryAriaLabel}
                aria-label={`${draftTokenCount.toLocaleString()} input tokens. ${composerTelemetryAriaLabel}`}>
                {draftTokenCount ? compactNumber(draftTokenCount) : ""}
              </span>
              <ComposerSubmitControl active={turnActive} stopping={cancellingTurn}
                canSend={Boolean(runtimeLoaded && input.trim() && window.omni && draftCapacity.fits && !taskAttachmentDraft.busy && !waitingInheritedFileReview)}
                canSteer={runtimeLoaded && composerTurn.steerAvailable}
                onSend={submitOrdinaryTurn} onSteer={() => void steerCurrentTurn()}
                onStop={() => void cancelChatTool()} />
            </div>
          </div>
          {!window.omni ? <p className="composer-disclaimer" role="status">
            Preview only — no brain is connected. Sending and learning are unavailable.
          </p> : null}
          {window.omni && !runtimeLoaded ? <p className="composer-disclaimer" role="status">Load this brain to chat.</p> : null}
        </div>
        </CallLayout>
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
            {recentTrace ? <details className="chat-activity-details">
              <summary>Latest activity</summary>
              <ol>{recentTrace.steps.map((step, index) => <li key={`${recentTrace.id}-${index}`}>
                <strong>{step.stage}</strong><span>{step.detail}</span>{step.value ? <small>{step.value}</small> : null}
              </li>)}</ol>
            </details> : null}
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
  contentProjection,
  brainId,
  brainName,
  onOwnVoice,
  ownVoicePhase,
  ownVoiceUnavailable,
  ownVoiceQuality,
  uncommittedOutputState
}: {
  message: ChatMessage;
  /** View-only segments; copying, voice and persistence retain the full message. */
  contentProjection?: ReactNode;
  brainId: string;
  brainName: string;
  onOwnVoice?: () => void;
  ownVoicePhase?: "preparing" | "playing";
  ownVoiceUnavailable?: boolean;
  ownVoiceQuality?: "needs-speech-training" | "unverified";
  uncommittedOutputState?: UncommittedChatOutput["state"];
}) {
  const isBrain = message.role === "brain";
  const attribution = chatMessageAttribution(message, brainName);
  const deliveryState = chatMessageDeliveryState(message);
  const durableReceipt = message.deliveryReceipt?.presentationOnly === true;
  const queued = deliveryState === "queued";
  const steered = deliveryState === "steered";
  const stopped = deliveryState === "stopped";
  const noReply = isBrain && deliveryState === "no-reply";
  const noReplyPresentation = chatNoReplyPresentation(message);
  const failed = deliveryState === "failed";
  const cancelled = deliveryState === "cancelled";
  const paused = pausedOutputLabel(deliveryState);
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
        noReply ? "no-reply" : stopped ? "stopped" : paused ? deliveryState : !isBrain && message.inputReceipt?.committed ? "received" : uncommittedOutputState
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
      data-input-source={attribution.connectedApp ? "mcp-app" : message.role}
    >
      <div className="message__avatar">
        {isBrain ? <BrandMark size={28} /> : <span className="human-avatar">{attribution.connectedApp ? <Icon name="terminal" size={16} /> : "E"}</span>}
      </div>
      <div className="message__body">
        <div className="message__meta">
          <strong><bdi>{attribution.name}</bdi>{attribution.connectedApp ? " · App" : ""}</strong>
          <time>
            {new Date(message.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
          </time>
          {!isBrain && message.inputReceipt?.committed ? <span className="message-queue-label">Received</span> : null}
          {uncommittedOutputState ? (
            <span className="message-queue-label">
              {pausedOutputLabel(uncommittedOutputState) ?? (uncommittedOutputState === "saving"
                ? "Reply complete · save pending, not committed"
                : uncommittedOutputState === "steering"
                  ? "Steering · emitted prefix awaiting safe save"
                : uncommittedOutputState === "cancelled"
                  ? "Stopped output · not committed"
                  : "Save failed · output not committed")}
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
          {paused ? <span className="message-queue-label">{paused}</span> : null}
          {isBrain && message.runtime ? <span className="runtime-label">{message.runtime.replace("-", " ")}</span> : null}
        </div>
        {noReply ? null : <div className="message__content">{contentProjection ?? message.content}</div>}
        {!isBrain && message.nativeAudioInput ? <NativeAudioInputSummary input={message.nativeAudioInput} /> : null}
        {!isBrain && message.turnId && message.taskAttachmentIds?.length ? <MessageTaskAttachments
          brainId={brainId} turnId={message.turnId} attachmentIds={message.taskAttachmentIds} received={message.inputReceipt?.committed === true} /> : null}
        {isBrain ? (
          <div className="message__actions">
            {message.content ? <button aria-label="Copy response" onClick={() => void navigator.clipboard?.writeText(message.content)}>
              <Icon name="copy" size={14} />
            </button> : null}
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
  foregroundTurnActive = false,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  foregroundTurnActive?: boolean;
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
    if (!window.omni || busy || foregroundTurnActive) return;
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
    if (!window.omni || busy || foregroundTurnActive) return;
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
          disabled={Boolean(busy) || !window.omni || foregroundTurnActive}
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
      {foregroundTurnActive ? <p role="status">Recovery points stay readable. Create or restore after the current reply finishes, or use Unload to preserve its exact frontier.</p> : null}
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
                disabled={Boolean(busy) || foregroundTurnActive}
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
    ["Program changes", `${evolutionPermission} permission`],
    ["Tool schemas", enabledTools.length ? `${enabledTools.length} visible` : "None enabled"]
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
  foregroundTurnActive = false,
  instanceState,
  onReviewLoad,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  foregroundTurnActive?: boolean;
  instanceState?: InstanceLifecycleState;
  onReviewLoad?: () => void;
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
  const persistedDeviceResourcesKey = JSON.stringify(brain.config.deviceResources ?? null);
  const [deviceResources, setDeviceResources] = useState<DeviceResourceSettings | undefined>(brain.config.deviceResources);
  const effectiveDeviceResources = useMemo(() => synchronizeDeviceRam(deviceResources, systemRamMode, systemRamSharePercent),
    [deviceResources, systemRamMode, systemRamSharePercent]);
  useEffect(() => {
    setDeviceResources((JSON.parse(persistedDeviceResourcesKey) as DeviceResourceSettings | null) ?? undefined);
  }, [brain.id, persistedDeviceResourcesKey]);
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
        ...(effectiveDeviceResources ? { deviceResources: effectiveDeviceResources } : {}),
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
    workingMemoryMode,
    effectiveDeviceResources
  ]);

  const dirty =
    JSON.stringify(effectiveDeviceResources ?? null) !== persistedDeviceResourcesKey ||
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
    if (!resourcePlan?.allowed || resourcePlanPending || saving || foregroundTurnActive) return;
    setSaving(true);
    setResourceError("");
    try {
      const nextConfig = configWithResourceEnvelope(
        { ...brain.config, ...(effectiveDeviceResources ? { deviceResources: effectiveDeviceResources } : {}) },
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
    setDeviceResources((JSON.parse(persistedDeviceResourcesKey) as DeviceResourceSettings | null) ?? undefined);
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
    if (!window.omni || freshAttentionBusy || foregroundTurnActive) return;
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

  return (
    <div className="content-page resource-settings-page">
      <div className="content-page__title resource-settings-page__title">
        <div>
          <span className="eyebrow-text">BRAIN SETTINGS</span>
          <h1>Memory &amp; storage</h1>
          <p>Adjust working memory and the RAM, GPU memory, and drives this brain can use.</p>
        </div>
        <span
          className={cx("resource-settings-save-state", dirty && "is-dirty")}
          role="status"
          aria-live="polite"
        >
          <Icon name={dirty ? "warning" : "check"} size={14} />
          {dirty ? "Unsaved changes" : "Settings saved"}
        </span>
      </div>

      <div className="resource-settings-layout">
        {onReviewLoad ? <section className="surface resource-settings-residency"><h2>Brain loading</h2>
          <p>{instanceState?.phase === "loaded" ? "Loaded and ready." : "Not loaded. You can edit settings before loading."}
            {instanceState?.resume ? ` The saved turn retains ${instanceState.resume.publishedTokens.toLocaleString()} published tokens.` : ""}</p>
          <Button onClick={onReviewLoad}>Check memory and load</Button>
        </section> : null}

        <section className="surface resource-settings-policy" aria-labelledby="ram-policy-heading">
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="memory" size={19} /></span>
            <span>
              <h2 id="ram-policy-heading">RAM allocation</h2>
              <p>Auto leaves room for the system. A custom limit uses a share of the remaining memory.</p>
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
                <h2>Working memory</h2>
                <p>Recent activity stays in faster memory. Older attention can move to the storage pool.</p>
              </span>
            </div>
            <fieldset className="resource-settings-mode-picker resource-settings-mode-picker--three">
              <legend>Working-memory size</legend>
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
                    <strong>Working-memory tokens</strong>
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
                      aria-label="Working-memory tokens"
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
                  aria-label="Working-memory token limit"
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
                <h2>Storage pool</h2>
                <p>One shared pool covers model overflow, learning, saved state, and growth.</p>
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

          <details className="brain-settings-disclosure">
            <summary>GPU memory &amp; drives</summary>
          <DeviceAllocationSelector catalog={window.omni?.catalog} brainId={brain.id} settings={effectiveDeviceResources}
            ramMode={systemRamMode} ramPercent={systemRamSharePercent} disabled={saving}
            plan={resourcePlan?.deviceResources} planPending={resourcePlanPending}
            onChange={(settings) => {
              setDeviceResources(settings); setResourcePlanPending(Boolean(window.omni));
              const ram = settings.allocations.find((entry) => entry.deviceId === "ram:system");
              if (ram) { setSystemRamMode(ram.mode); setSystemRamSharePercent(ram.percent); }
            }} />
          </details>
          {window.omni?.decoderExecution || window.omni?.cachedDecoderExecution || window.omni?.nativeOperationAuto || window.omni?.regionalExecution ? <details className="brain-settings-disclosure">
            <summary>Advanced execution</summary>
          {window.omni?.decoderExecution ? <DecoderExecutionControls api={window.omni.decoderExecution} brainId={brain.id}
            loaded={instanceState?.phase === "loaded" && instanceState.learnedRuntimeResident === true}
            foregroundTurnActive={foregroundTurnActive} dirty={dirty} disabled={saving} /> : null}
          {window.omni?.cachedDecoderExecution ? <CachedDecoderExecutionControls api={window.omni.cachedDecoderExecution} brainId={brain.id}
            loaded={instanceState?.phase === "loaded" && instanceState.learnedRuntimeResident === true}
            foregroundTurnActive={foregroundTurnActive} dirty={dirty} disabled={saving} /> : null}
          {window.omni?.nativeOperationAuto ? <NativeOperationAutoControls api={window.omni.nativeOperationAuto} brainId={brain.id}
            loaded={instanceState?.phase === "loaded" && instanceState.learnedRuntimeResident === true}
            foregroundTurnActive={foregroundTurnActive} dirty={dirty} disabled={saving} /> : null}
          {window.omni?.regionalExecution ? <RegionalExecutionControls api={window.omni.regionalExecution} brainId={brain.id}
            loaded={instanceState?.phase === "loaded" && instanceState.learnedRuntimeResident === true}
            foregroundTurnActive={foregroundTurnActive} settings={effectiveDeviceResources} dirty={dirty} disabled={saving}
            onChange={settings => {
              // Selecting a regional target must not silently replace the
              // separately selected global RAM policy or raise its ceiling.
              setDeviceResources(synchronizeDeviceRam(settings, systemRamMode, systemRamSharePercent));
              setResourcePlanPending(Boolean(window.omni));
            }} /> : null}
          </details> : null}
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
                    ? "Memory check unavailable"
                    : planBlocked
                      ? "These settings do not fit safely"
                      : resourcePlan
                        ? systemRamMode === "auto"
                          ? `Auto recommends ${resourcePlan.resources.systemRamSharePercent}% after the system reserve`
                          : `${resourcePlan.resources.systemRamSharePercent}% passes the memory check`
                        : "Connect the desktop runtime to measure this device"}
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
              disabled={!dirty || saving || resourcePlanPending || !resourcePlan?.allowed || foregroundTurnActive}
              onClick={() => void saveResourceEnvelope()}
            >
              {saving ? "Saving resources…" : "Save memory & storage"}
            </Button>
            <Button disabled={!dirty || saving} onClick={resetResourceEnvelope}>
              Discard change
            </Button>
          </div>
          {dirty && foregroundTurnActive ? <p className="resource-settings-note" role="status">You can review and edit this allocation now. Apply it after the current reply finishes; the running reply keeps its admitted memory envelope.</p> : null}
        </section>

        <aside className="resource-settings-sidebar">
          <details className="surface resource-settings-device brain-settings-disclosure">
            <summary>Memory use &amp; limits</summary>
            <div className="resource-settings-section-heading">
              <span className="resource-settings-section-icon"><Icon name="activity" size={18} /></span>
              <span>
                <h2 id="safe-pool-heading">Device memory</h2>
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
          </details>
        </aside>

        <details className="surface resource-settings-benchmarks brain-settings-disclosure">
          <summary>Measured speed</summary>
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="pulse" size={18} /></span>
            <span>
              <h2 id="throughput-heading">Device measurements</h2>
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
        </details>

        <details className="surface resource-settings-training brain-settings-disclosure">
          <summary>Training memory</summary>
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="database" size={18} /></span>
            <span>
              <h2 id="adaptive-training-heading">Current training allocation</h2>
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
        </details>

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
              disabled={freshAttentionBusy || saving || foregroundTurnActive || Boolean(window.omni?.brain.lifecycle && (instanceState?.phase !== "loaded" || instanceState.resume))}
              onClick={() => void startFreshAttention()}
            >
              {freshAttentionBusy ? "Starting fresh attention…" : "Start fresh attention"}
            </Button>
          </div>
        </details>

        <BrainSnapshotsPanel
          brain={brain}
          foregroundTurnActive={foregroundTurnActive}
          onBrainChange={onBrainChange}
          onToast={onToast}
        />

        <details className="surface mobile-companion brain-settings-disclosure">
          <summary>Phone companion</summary>
          <div className="resource-settings-section-heading">
            <span className="resource-settings-section-icon"><Icon name="agents" size={18} /></span>
            <span>
              <h2 id="mobile-companion-heading">Connect a phone</h2>
              <p>Use the same brain and conversation from the companion app.</p>
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
                {mobileBusy ? "Updating…" : mobileStatus?.state === "listening" ? "New pairing code" : "Pair a phone"}
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
        </details>
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
  onOpenPermissions,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  foregroundTurnActive: boolean;
  onBack: () => void;
  onOpenPermissions: () => void;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [activeJob, setActiveJob] = useState<RuntimeJob | null>(null);
  const [resumableManifest, setResumableManifest] = useState<DatasetResumeCandidate | null>(null);
  const [persistedCoverage, setPersistedCoverage] = useState<PersistedDatasetCoverageSummary | null>(null);
  const [catalogEntries, setCatalogEntries] = useState<CatalogEntry[]>([]);
  const [controlBusy, setControlBusy] = useState(false);
  const [confirmStop, setConfirmStop] = useState(false);
  const [statusError, setStatusError] = useState("");
  const announcedJobs = useRef(new Set<string>());
  const controlPending = useRef(false);
  const currentOwner = useRef(brain.id);
  currentOwner.current = brain.id;
  const stopState = runtimeJobStopPresentation(activeJob);
  const jobProgress = runtimeJobProgressPresentation(activeJob);
  const output = valueRecord(activeJob?.output);
  const coverageValue = jobProgress.showCoverage ? valueRecord(output?.coverage) : undefined;
  const activeCoverage = coverageValue &&
    typeof coverageValue.discoveredRecords === "number" &&
    typeof coverageValue.processedRecords === "number" &&
    typeof coverageValue.rejectedRecords === "number" &&
    typeof coverageValue.discoveredFiles === "number" &&
    typeof coverageValue.processedFiles === "number" &&
    typeof coverageValue.rejectedFiles === "number"
    ? { discoveredRecords: coverageValue.discoveredRecords, processedRecords: coverageValue.processedRecords,
        rejectedRecords: coverageValue.rejectedRecords, discoveredFiles: coverageValue.discoveredFiles,
        processedFiles: coverageValue.processedFiles, rejectedFiles: coverageValue.rejectedFiles,
        complete: coverageValue.complete === true }
    : null;
  const coverage = runtimeJobScopedDatasetCoverage(activeJob, activeCoverage, !stopState.active ? persistedCoverage : null);
  const progress = activeJob ? jobProgress.percent : persistedCoverage?.complete ? 100 : null;
  const telemetry = activeJob?.telemetry && jobProgress.showTelemetry
    ? presentTrainingTelemetry(activeJob.telemetry) : undefined;

  useEffect(() => {
    const api = window.omni;
    let active = true;
    currentOwner.current = brain.id;
    controlPending.current = false;
    setActiveJob(null); setPersistedCoverage(null); setResumableManifest(null);
    setCatalogEntries([]); setControlBusy(false); setConfirmStop(false); setStatusError("");
    if (!api) return;
    const refreshSaved = async () => {
      await Promise.all([
        api.data.resumable(brain.id).then(resume => { if (active) setResumableManifest(resume); }),
        api.brain.persistedSubstrateOverview(brain.id).then(overview => { if (active) setPersistedCoverage(overview?.latestCompletedCoverage ?? null); })
      ]);
    };
    void api.train.list(brain.id).then(jobs => {
      if (!active) return;
      const latest = jobs.filter(job => job.brainId === brain.id).sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0] ?? null;
      setActiveJob(current => current && (!latest || current.updatedAt >= latest.updatedAt) ? current : latest);
    }).catch(error => { if (active) setStatusError(conciseUiMessage(error, "Saved progress could not load.")); });
    void api.catalog.list().then(entries => { if (active) setCatalogEntries(entries); })
      .catch(error => { if (active) setStatusError(conciseUiMessage(error, "Dataset sources could not load.")); });
    void refreshSaved().catch(error => { if (active) setStatusError(conciseUiMessage(error, "Saved coverage could not load.")); });
    const unsubscribe = api.train.onEvent(({ job }) => {
      if (!active || job.brainId !== brain.id) return;
      setActiveJob(job);
      if (["complete", "cancelled", "failed"].includes(job.state))
        void refreshSaved().catch(error => { if (active) setStatusError(conciseUiMessage(error)); });
      if (job.state === "complete")
        void api.brain.get(brain.id).then(value => { if (active) onBrainChange(value); }).catch(() => undefined);
      if (["complete", "failed"].includes(job.state) && !announcedJobs.current.has(job.id)) {
        announcedJobs.current.add(job.id);
        onToast(job.state === "complete" ? job.label + " completed." : conciseUiMessage(job.error, job.label + " failed."));
      }
    });
    return () => { active = false; if (currentOwner.current === brain.id) currentOwner.current = ""; unsubscribe(); };
  }, [brain.id, onBrainChange, onToast]);

  const controlJob = async (action: "pause" | "stop") => {
    const api = window.omni, job = activeJob, owner = brain.id;
    if (!api || !job || controlPending.current) return;
    if (action === "stop" && !stopState.requestAllowed) return;
    controlPending.current = true; setControlBusy(true); setStatusError("");
    if (action === "stop") setActiveJob({ ...job, state: "cancelling", label: "Stopping " + job.label, updatedAt: new Date().toISOString() });
    try {
      const updated = action === "pause" ? await api.data.pause(job.id)
        : ["ingestion", "crawl"].includes(job.kind) ? await api.data.cancel(job.id) : await api.train.cancel(job.id);
      const resume = await api.data.resumable(owner);
      if (currentOwner.current === owner) { setActiveJob(updated); setResumableManifest(resume); setConfirmStop(false); }
    } catch (error) {
      const jobs = await api.train.list(owner).catch(() => []);
      if (currentOwner.current === owner) {
        setActiveJob(jobs.find(value => value.id === job.id) ?? (action === "stop"
          ? { ...job, state: "cancelling", error: conciseUiMessage(error), updatedAt: new Date().toISOString() } : job));
        setStatusError(conciseUiMessage(error, "The request needs acknowledgement."));
      }
    } finally { if (currentOwner.current === owner) { controlPending.current = false; setControlBusy(false); } }
  };
  const resumeDataset = async () => {
    const api = window.omni, manifest = resumableManifest, owner = brain.id;
    if (!api || !manifest || controlPending.current || stopState.active) return;
    controlPending.current = true; setControlBusy(true); setStatusError("");
    try {
      const job = await api.data.resume({ brainId: owner, manifestId: manifest.manifestId,
        policy: "pretrain", epochs: manifest.requestedEpochs, resume: true });
      if (currentOwner.current === owner) { setActiveJob(job); setResumableManifest(null); }
    } catch (error) { if (currentOwner.current === owner) setStatusError(conciseUiMessage(error, "The dataset could not resume.")); }
    finally { if (currentOwner.current === owner) { controlPending.current = false; setControlBusy(false); } }
  };

  const runtimeStatus = <section className="surface training-card" aria-label="Dataset progress">
    <div className="training-card__head">
      <span className="training-card__icon"><Icon name={stopState.active ? "pulse" : coverage?.complete ? "check" : "info"} size={20} /></span>
      <span><small>{stopState.active ? "ACTIVE JOB" : "SAVED PROGRESS"}</small>
        <strong>{activeJob?.label ?? (persistedCoverage ? "Dataset traversal complete" : "No dataset runs yet")}</strong></span>
      {stopState.active ? <div className="training-card__controls">
        {activeJob && ["ingestion", "crawl"].includes(activeJob.kind) && ["queued", "running"].includes(activeJob.state)
          ? <button className="icon-button" disabled={controlBusy} onClick={() => void controlJob("pause")} aria-label="Pause active job"><Icon name="pause" size={15} /></button> : null}
        <button className="icon-button" disabled={controlBusy || !stopState.requestAllowed}
          onClick={() => setConfirmStop(true)} aria-label={stopState.retryAvailable ? "Retry stopping active job" : "Stop active job"} title={stopState.label}>
          <Icon name={stopState.waitingForAcknowledgement ? "pulse" : "close"} size={15} /></button>
      </div> : null}
    </div>
    {activeJob || persistedCoverage ? <div className={cx("training-progress", progress === null && "is-indeterminate")}>
      <span><strong>{progress === null ? "—" : progress + "%"}</strong><em>{activeJob?.state ?? "complete"}</em></span>
      <i><b style={progress === null ? undefined : { width: progress + "%" }} /></i>
    </div> : null}
    {telemetry ? <TrainingTelemetryMetrics telemetry={telemetry} /> : null}
    {coverage ? <div className="training-coverage" role="status">
      <strong>{(coverage.processedRecords + coverage.rejectedRecords).toLocaleString()} / {coverage.discoveredRecords.toLocaleString()} records visited</strong>
      <small>{coverage.processedRecords.toLocaleString()} learned · {coverage.rejectedRecords.toLocaleString()} rejected · {(coverage.processedFiles + coverage.rejectedFiles).toLocaleString()} / {coverage.discoveredFiles.toLocaleString()} file passes{coverage.complete ? " · complete" : " · resumable"}</small>
    </div> : null}
    {resumableManifest && !stopState.active ? <div className="dataset-resume-note">
      <span><strong>Dataset ready to resume</strong><small>{resumableManifest.processedFiles.toLocaleString()} / {(resumableManifest.discoveredFiles * resumableManifest.requestedEpochs).toLocaleString()} file passes settled</small></span>
      <Button kind="primary" icon="play" disabled={controlBusy} onClick={() => void resumeDataset()}>Resume</Button>
    </div> : null}
    {confirmStop ? <div className="training-stop-confirm" role="group" aria-label="Confirm stopping dataset work">
      <p>Stop this job? Saved learning and its resume cursor stay intact.</p>
      <Button disabled={controlBusy} onClick={() => void controlJob("stop")}>Stop job</Button>
      <Button disabled={controlBusy} onClick={() => setConfirmStop(false)}>Keep running</Button>
    </div> : null}
    {statusError ? <p className="learning-workspace__error" role="alert">{statusError}</p> : null}
    {activeJob?.error ? <p className="learning-workspace__error">{conciseUiMessage(activeJob.error)}</p> : null}
    {catalogEntries.length ? <details className="training-catalog"><summary>Dataset sources</summary>
      {catalogEntries.map(entry => <div key={entry.id}><span><strong>{entry.name}</strong><small>{entry.license}</small></span>
        <Button onClick={() => void window.omni?.window.openExternal(entry.sourceUrl)}>Inspect source</Button></div>)}
    </details> : null}
  </section>;

  return <div className="content-page data-page training-evolution-page" data-training-active={stopState.active}>
    <div className="content-page__title"><div>
      <button className="content-page__back" aria-label="Back to conversation" onClick={onBack}><Icon name="arrow" size={13} className="map-arrow-back" /> Conversation</button>
      <h1>Training &amp; Evolution</h1>
    </div></div>
    {foregroundTurnActive ? <div className="data-foreground-queue-note" role="status"><Icon name="chat" size={15} />
      <span>Current reply continues. New learning joins the shared queue.</span></div> : null}
    <LearningWorkspace brainId={brain.id} embedded api={window.omni?.learning} questions={window.omni?.questions}
      data={window.omni?.data} localTeacher={window.omni?.localTeacher} runtimeStatus={runtimeStatus}
      renderImprovement={objective => <EvolutionWorkspace key={brain.id} brain={brain} objective={objective} embedded
        onOpenPermissions={onOpenPermissions} onToast={onToast} />} />
  </div>;
}
function BrainMapWorkspace({ brain, onBack }: { brain: BrainDocument; onBack: () => void }) {
  return <WholeBrainMap brainId={brain.id} api={window.omni?.wholeBrainInspection} observeActivity={window.omni?.brain.routerActivity} onBack={onBack} />;
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
  const [toolDescriptions, setToolDescriptions] = useState<ToolDescription[]>([]);
  const [toolDescriptionRevision, setToolDescriptionRevision] = useState(0);
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
  const [executionReview, setExecutionReview] = useState<ToolExecutionResult>();
  const [running, setRunning] = useState(false);
  const [mcpServers, setMcpServers] = useState<McpServerSummary[]>([]);
  const [mcpRegistry, setMcpRegistry] = useState<McpRegistrySummary>({ enabled: true, instanceEnabled: true, servers: [] });
  const [mcpForAll, setMcpForAll] = useState(true);
  const [mcpId, setMcpId] = useState("");
  const [mcpLabel, setMcpLabel] = useState("");
  const [mcpTransport, setMcpTransport] = useState<McpTransport>("http");
  const [mcpUrl, setMcpUrl] = useState("");
  const [mcpCommand, setMcpCommand] = useState("");
  const [mcpArgs, setMcpArgs] = useState("");
  const [mcpToken, setMcpToken] = useState("");
  const [mcpEnvironmentText, setMcpEnvironmentText] = useState("");
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
    }).catch((error: unknown) => {
      if (active) onToast(error instanceof Error ? error.message : "Tool permissions could not load.");
    });
    return () => {
      active = false;
    };
  }, [brain.createdAt, brain.id, brain.toolPermissions, onToast]);

  useEffect(() => {
    let active = true;
    if (window.omni) void window.omni.tool.descriptions(brain.id).then(page => {
      if (active && page.brainId === brain.id) setToolDescriptions(page.tools);
    }).catch(reason => { if (active) onToast(reason instanceof Error ? reason.message : "Tool action descriptions could not load."); });
    return () => { active = false; };
  }, [brain.id, brain.toolPermissions, toolDescriptionRevision, onToast]);

  const selectedDescription = toolDescriptions.find(value => value.id === selectedTool);
  useEffect(() => {
    if (!action.trim() && selectedDescription) setAction(defaultToolAction(selectedDescription));
  }, [action, selectedDescription]);

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void Promise.all([
      window.omni.mcp.registry(brain.id),
      window.omni.tool.preferences()
    ]).then(([registry, preferences]) => {
      if (!active) return;
      setMcpRegistry(registry);
      setMcpServers(registry.servers);
      setApprovalTimeout(String(preferences.approvalTimeoutSeconds));
    }).catch((error) => onToast(error instanceof Error ? error.message : "Integration settings could not load."));
    return () => {
      active = false;
    };
  }, [brain.id, onToast]);

  const chooseTool = (toolId: string) => {
    const meta = protocolMeta[toolId] ?? {
      label: permissions.find((item) => item.toolId === toolId)?.label ?? toolId,
      icon: "terminal" as const,
      action: defaultToolAction(toolDescriptions.find(value => value.id === toolId)) || (toolId.startsWith("mcp.") ? "call" : ""),
      args: {}
    };
    setSelectedTool(toolId);
    setAction(meta.action);
    setArgumentsText(JSON.stringify(meta.args, null, 2));
    setResultText("");
    setApprovalToken("");
    setExecutionReview(undefined);
  };

  const setPermission = async (toolId: string, level: PermissionLevel) => {
    if (!window.omni) {
      setPermissions((current) => current.map((item) => (item.toolId === toolId ? { ...item, level } : item)));
      onToast("Demo permission changed visually; no tool executor is connected.");
      return;
    }
    try {
      const next = await window.omni.tool.setPermission(
        brain.id,
        toolId,
        level,
        globalThis.crypto.randomUUID(),
        Math.max(1, ...permissions.map((permission) => permission.revision ?? 1))
      );
      setPermissions(mergeToolPermissionDefaults(next, brain.createdAt));
      if (toolId === selectedTool) { setApprovalToken(""); setExecutionReview(undefined); }
      onBrainChange(await window.omni.brain.get(brain.id));
    } catch (error) {
      // A rejected stale-revision write is not retried as a new grant. Refresh
      // current host authority so the next deliberate click starts from it.
      const current = await window.omni.tool.listPermissions(brain.id).catch(() => undefined);
      if (current) setPermissions(mergeToolPermissionDefaults(current, brain.createdAt));
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
        setExecutionReview(result);
        setResultText(
          `Approval required${result.approvalExpiresAt ? ` until ${new Date(result.approvalExpiresAt).toLocaleTimeString()}` : ""}. Inspect the action above, then click Approve & run.`
        );
      } else {
        setApprovalToken("");
        setExecutionReview(undefined);
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

  const addMcpServer = async () => {
    if (!window.omni) return;
    setIntegrationBusy(true);
    try {
      let environment: Record<string, string> | undefined;
      if (mcpTransport === "stdio" && mcpEnvironmentText.trim()) {
        try { environment = JSON.parse(mcpEnvironmentText) as Record<string, string>; }
        catch { throw new Error("Local server variables must be a valid JSON name/value object."); }
      }
      const added = await window.omni.mcp.add({
        brainId: brain.id,
        id: mcpId.trim(),
        label: mcpLabel.trim() || mcpId.trim(),
        transport: mcpTransport,
        scope: mcpForAll ? "all" : "instance",
        ...(mcpTransport === "http"
          ? { url: mcpUrl.trim(), bearerToken: mcpToken.trim() || undefined }
          : {
              command: mcpCommand.trim(),
              args: mcpArgs.split(/\r?\n/).map((value) => value.trim()).filter(Boolean),
              ...(environment ? { environment } : {})
            })
      });
      setMcpServers((current) => [...current.filter((item) => item.id !== added.id), added]);
      setMcpToken("");
      setMcpEnvironmentText("");
      const next = await window.omni.tool.listPermissions(brain.id);
      setPermissions(mergeToolPermissionDefaults(next, brain.createdAt));
      onBrainChange(await window.omni.brain.get(brain.id));
      onToast(added.connected ? `Connected ${added.label} · ${added.tools.length} tools · ${added.schemaLearning === "complete" ? "capabilities learned" : "capability learning pending"}`
        : `Added ${added.label} · ${added.authentication?.state === "required" ? "sign-in needed" : "connection pending"}`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "MCP server could not connect.");
    } finally {
      setIntegrationBusy(false);
    }
  };

  const configureMcp = async (request: McpRegistryUpdateRequest): Promise<void> => {
    if (!window.omni || integrationBusy) return;
    setIntegrationBusy(true);
    try {
      await window.omni.mcp.configure(request);
      const registry = await window.omni.mcp.registry(brain.id);
      setMcpRegistry(registry);
      setMcpServers(registry.servers);
      setPermissions(mergeToolPermissionDefaults(await window.omni.tool.listPermissions(brain.id), brain.createdAt));
    } catch (error) {
      onToast(error instanceof Error ? error.message : "MCP access could not be changed.");
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

  const onPluginNotice = (message: string): void => {
    onToast(message);
    setToolDescriptionRevision(value => value + 1);
    if (window.omni) void window.omni.tool.listPermissions(brain.id).then(records => {
      setPermissions(mergeToolPermissionDefaults(records, brain.createdAt));
    }).catch(error => onToast(error instanceof Error ? error.message : "Current tool permissions could not refresh."));
  };

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
                  <button className="permission-row__identity" disabled={running} onClick={() => chooseTool(permission.toolId)}>
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
          <ProcessMemorySettings brainId={brain.id} api={window.omni?.processMemory} />
        </section>
        <aside className="surface tool-console">
          <div className="tool-console__head">
            <span><Icon name={selectedMeta.icon} size={18} /></span>
            <div><small>TEST PROTOCOL</small><strong>{selectedMeta.label}</strong></div>
          </div>
          <ToolActionPicker description={selectedDescription} action={action} disabled={running}
            onChange={value => { setAction(value); setApprovalToken(""); setExecutionReview(undefined); }} />
          <label>
            <span>Arguments · JSON</span>
            <textarea value={argumentsText} disabled={running} onChange={(event) => { setArgumentsText(event.target.value); setApprovalToken(""); setExecutionReview(undefined); }} rows={10} spellCheck={false} />
          </label>
          <PluginToolApprovalReviewView execution={executionReview} toolId={selectedTool} action={action} />
          <ActionReviewDetails execution={executionReview} />
          {approvalToken ? <Button kind="ghost" icon="close" onClick={() => void (async () => {
            if (!window.omni) return;
            await window.omni.tool.declineApproval(brain.id, approvalToken);
            setApprovalToken(""); setExecutionReview(undefined); setResultText("This exact approval was discarded; no action dispatched.");
          })().catch(error => onToast(conciseUiMessage(error, "The approval could not be discarded.")))}>Discard approval</Button> : null}
          <Button
            kind="primary"
            icon={running ? "close" : approvalToken ? "check" : "play"}
            disabled={!running && (!action.trim() || selectedDescription?.source === "unregistered")}
            onClick={() => void (running ? cancelExecution() : execute())}
          >
            {running ? "Cancel execution" : approvalToken ? "Approve & run" : "Run action"}
          </Button>
          <div className="tool-output">
            <span>OUTPUT</span>
            <pre>{resultText || "No action has run in this session."}</pre>
          </div>
          <div className="builder-note">
            <Icon name="info" size={15} />
            <span>Full authority permits broad host actions. Uncertain effects and special reviews still ask; every action is recorded.</span>
          </div>
        </aside>
      </div>
      <section className="surface integration-surface">
        <PluginRuntimeSlot slot="tools" showProviderError />
        <SparseSubstrateJobsPanel />
        <PluginManager key={brain.id} brainId={brain.id} onNotice={onPluginNotice}
          onCapabilitiesChanged={pluginId => {
            setToolDescriptionRevision(value => value + 1);
            if (selectedTool.startsWith(`plugin.${pluginId}.`)) {
              if (approvalToken) setResultText("Plugin access or version changed. Submit a new exact action for review.");
              setApprovalToken(""); setExecutionReview(undefined);
            }
          }} />
      </section>
      {window.omni?.maps ? <section className="surface integration-surface"><MapConnectionPanel /></section> : null}
      {window.omni?.maps?.basemaps ? <section className="surface integration-surface"><MapBasemapPanel api={window.omni.maps.basemaps} /></section> : null}
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
          <ApiTeacherPanel brainId={brain.id} api={window.omni?.teacher} onNotice={onToast} />
          <div className="integration-card">
            <div className="integration-card__head">
              <span><Icon name="terminal" size={18} /></span>
              <div><strong>MCP tools</strong><small>Connect remote Streamable HTTP or an installed local stdio server</small></div>
            </div>
            <div className="mcp-access-controls">
              <label><input type="checkbox" checked={mcpRegistry.enabled} disabled={integrationBusy}
                onChange={event => void configureMcp({ enabled: event.target.checked })} /> MCP enabled on this device</label>
              <label><input type="checkbox" checked={mcpRegistry.instanceEnabled} disabled={integrationBusy || !mcpRegistry.enabled}
                onChange={event => void configureMcp({ brainId: brain.id, enabled: event.target.checked })} /> Allow MCP for {brain.name}</label>
            </div>
            <McpPresetPicker disabled={integrationBusy} onChoose={(draft) => {
                setMcpId(draft.id); setMcpLabel(draft.label); setMcpTransport(draft.transport);
                setMcpUrl(draft.url ?? ""); setMcpCommand(draft.command); setMcpArgs(draft.args.join("\n"));
                setMcpToken(""); setMcpEnvironmentText("");
                onToast("Connection draft prepared. No installation, sign-in, connection or access grant has run.");
              }} />
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
                <details className="mcp-environment-editor"><summary>Credential environment · optional</summary>
                  <label><span>Variables · JSON name/value object</span><textarea rows={3} autoComplete="off" spellCheck={false}
                    value={mcpEnvironmentText} onChange={event => setMcpEnvironmentText(event.target.value)} placeholder={'{"SERVICE_API_KEY":"…"}'} /></label>
                </details>
              </div>
            )}
            <div className="integration-actions">
              <label className="mcp-add-scope"><input type="checkbox" checked={mcpForAll} onChange={event => setMcpForAll(event.target.checked)} /> Add for all instances</label>
              <Button
                kind="primary"
                icon="plus"
                disabled={integrationBusy || !mcpId.trim() || (mcpTransport === "http" ? !mcpUrl.trim() : !mcpCommand.trim())}
                onClick={() => void addMcpServer()}
              >
                Connect server
              </Button>
              <small>Each tool follows this instance’s permissions.</small>
            </div>
            <div className="mcp-server-list">
              {mcpServers.map((server) => (
                <div key={server.id} className={server.effectiveEnabled ? "" : "mcp-server--disabled"}>
                  <span><i className={server.connected && server.effectiveEnabled ? "is-connected" : ""} /><strong>{server.label}</strong><small>{server.tools.length} tools · {server.transport}{server.disabledReason ? ` · ${server.disabledReason}` : server.error ? ` · ${server.error}` : ""}</small></span>
                  <span className="mcp-server-list__actions">
                    <label><input type="checkbox" checked={server.globallyEnabled} disabled={integrationBusy || !mcpRegistry.enabled}
                      onChange={event => void configureMcp({ registryId: server.registryId, enabled: event.target.checked })} /> Server enabled</label>
                    <select aria-label={`${server.label} access for ${brain.name}`} value={server.accessOverride}
                      disabled={integrationBusy || !mcpRegistry.enabled || !mcpRegistry.instanceEnabled || !server.globallyEnabled}
                      onChange={event => void configureMcp({ brainId: brain.id, registryId: server.registryId, accessOverride: event.target.value as "inherit" | "on" | "off" })}>
                      <option value="inherit">Default ({server.defaultEnabled ? "on" : "off"})</option>
                      <option value="on">On for this instance</option>
                      <option value="off">Off for this instance</option>
                    </select>
                    <Button icon="pulse" disabled={integrationBusy} onClick={() => void refreshMcpServer(server.id)}>Refresh</Button>
                    <Button kind="danger" disabled={integrationBusy} onClick={() => void removeMcpServer(server.id)}>Remove server</Button>
                  </span>
                  {server.transport === "http" ? <McpAuthenticationPanel key={`${brain.id}:${server.id}:${server.authentication?.state}:${server.credentialConfigured}`} brainId={brain.id} server={server}
                    onChanged={() => { const api = window.omni?.mcp; if (api) void api.registry(brain.id).then(registry => { setMcpRegistry(registry); setMcpServers(registry.servers); }).catch(error => onToast(error instanceof Error ? error.message : "MCP status could not refresh.")); }} /> : null}
                  <McpCredentialPanel key={`${brain.id}:${server.id}:credentials`} brainId={brain.id} server={server} onSaved={updated => {
                    setMcpServers(current => current.map(item => item.id === updated.id ? updated : item));
                    setMcpRegistry(current => ({ ...current, servers: current.servers.map(item => item.id === updated.id ? updated : item) }));
                  }} />
                  <McpHostServicesPanel brainId={brain.id} serverId={server.id} connected={server.connected} enabled={server.effectiveEnabled}
                    roots={window.omni?.mcp.roots} diagnostics={window.omni?.mcp.diagnostics} />
                </div>
              ))}
              {!mcpServers.length ? <small>No MCP servers connected to this mind.</small> : null}
            </div>
            <McpTasksPanel key={brain.id} brainId={brain.id} api={window.omni?.mcp.tasks} tools={window.omni?.tool} servers={mcpServers} />
          </div>
        </div>
      </section>
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
  const selectedWorkspaceView = navItems.some(item => item.id === workspaceView) ? workspaceView : "chat";
  const [loading, setLoading] = useState(!demo);
  const [buildProgress, setBuildProgress] = useState<BuildProgressEvent | null>(null);
  const [activeBuildJob, setActiveBuildJob] = useState<RuntimeJob | null>(null);
  const [retryingInitialization, setRetryingInitialization] = useState(false);
  const [initializationStopIntent, setInitializationStopIntent] =
    useState<InitializationStopIntent | null>(null);
  const [confirmingInitializationCancel, setConfirmingInitializationCancel] = useState(false);
  const [toast, setToast] = useState<{ id: number; text: string } | null>(null);
  const toastSequence = useRef(0);
  const [deleteTarget, setDeleteTarget] = useState<
    Pick<BrainSummary, "id" | "name" | "rootId"> | null
  >(null);
  const pendingDeleteTargets = useRef<Array<Pick<BrainSummary, "id" | "name" | "rootId">>>([]);
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
    buildMetrics?.diskSpace !== undefined;

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

  const showToast = useCallback((message: string) => {
    const presentedMessage = conciseUiMessage(message);
    setToast({ id: ++toastSequence.current, text: presentedMessage });
    dispatchLocalNotification({
      enabled: appearanceWorkspace.notificationsEnabled,
      visible: document.visibilityState === "visible",
      message: presentedMessage,
      now: Date.now(),
      lastDispatchedAt: notificationDispatchTimes.current,
      environment: browserNotificationEnvironment()
    });
  }, [appearanceWorkspace.notificationsEnabled]);

  useEffect(() => {
    const api = window.omni?.portable; if (!api) return;
    let current = true;
    void api.status().then(status => {
      if (!current) return;
      applyPortablePresentationOnce(status, window.localStorage, value => {
        setAppearance(value.appearance); setAppearanceWorkspace(value.workspace);
        if (value.theme) savePluginThemeSelection(window.localStorage, value.theme);
        else clearPluginThemeSelection(window.localStorage);
      });
    }).catch(error => { if (current) showToast(error instanceof Error ? error.message : "Portable appearance could not be restored."); });
    return () => { current = false; };
  }, []);

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

  const openBrain = async (summary: BrainSummary | string, nextView: WorkspaceView = "chat") => {
    setLoading(true);
    try {
      if (typeof summary === "string" && !window.omni) throw new Error("The native instance is unavailable in this design preview.");
      const document = window.omni
        ? await window.omni.brain.get(typeof summary === "string" ? summary : summary.id)
        : typeof summary !== "string" ? makeDemoBrain(summary.id, summary.name, createPresetConfig(summary.preset, summary.name)) : null;
      if (!document) throw new Error("The native instance could not be opened.");
      setActiveBrain(document);
      setWorkspaceView(nextView);
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
          ...(extras.spillPoolSelectionId ? { spillPoolSelectionId: extras.spillPoolSelectionId } : {}),
          hardwareTier,
          workingMemory: {
            mode: config.workingMemoryMode,
            ...(config.deviceResources ? { deviceResources: config.deviceResources } : {}),
            ...(extras.spillPoolSelectionId ? { spillPoolSelectionId: extras.spillPoolSelectionId } : {}),
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

  const updateActiveBrain = useCallback((document: BrainDocument) => {
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
  }, []);

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
      throw error;
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
      setDeleteTarget(pendingDeleteTargets.current.shift() ?? null);
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

  const duplicateSelectedBrains = async (selected: BrainSummary[]): Promise<void> => {
    for (const target of selected) await duplicateBrain(target);
  };
  const requestSelectedDeletion = (selected: BrainSummary[]): void => {
    if (deleteTarget || deletingInstance) return;
    const current = new Map(summaries.map(summary => [summary.id, summary]));
    const targets = [...new Set(selected.map(summary => summary.id))]
      .flatMap(id => current.has(id) ? [current.get(id)!] : [])
      .map(({ id, name, rootId }) => ({ id, name, rootId }));
    pendingDeleteTargets.current = targets.slice(1);
    setDeleteTarget(targets[0] ?? null);
  };
  const exportSelectedBrains = async (selected: BrainSummary[], mode: BrainExportMode = "current"): Promise<void> => {
    if (!window.omni) { showToast("Export requires the connected desktop runtime."); return; }
    let completed = 0;
    for (const target of selected) {
      const destination = await window.omni.brain.export(target.id, mode);
      if (!destination) { showToast(`Export stopped; ${completed} completed.`); return; }
      completed += 1;
    }
    showToast(`${completed} brain${completed === 1 ? "" : "s"} exported.`);
  };

  return (
    <div className="app-shell">
      <PluginThemeHost />
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
          onDuplicate={duplicateSelectedBrains}
          onDelete={requestSelectedDeletion}
          onExport={exportSelectedBrains}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          onMerged={() => { if (window.omni) void window.omni.brain.list().then(setSummaries).catch((error: unknown) => showToast(error instanceof Error ? error.message : "Library refresh failed.")); }}
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
            {initializationControlBrainId ? <PluginRuntimeApprovalQueue key={initializationControlBrainId} brainId={initializationControlBrainId} /> : null}
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
          view={selectedWorkspaceView}
          onView={setWorkspaceView}
          onLibrary={() => setPage("library")}
          onActiveModeChange={applyActiveModeResult}
          onBrainChange={updateActiveBrain}
          onToast={showToast}
        />
      ) : (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onDuplicate={duplicateSelectedBrains}
          onDelete={requestSelectedDeletion}
          onExport={exportSelectedBrains}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          demo={demo}
          onMerged={() => { if (window.omni) void window.omni.brain.list().then(setSummaries).catch((error: unknown) => showToast(error instanceof Error ? error.message : "Library refresh failed.")); }}
        />
      )}
      {loading && page !== "library" ? (
        <div className="loading-overlay">
          <BrandMark size={46} />
          <span>Opening neural state…</span>
        </div>
      ) : null}
       {buildProgress && (!initializationRunningActive || page === "library") ? (
        <div className="loading-overlay build-readiness-overlay" role="status" aria-live="polite">
          <BrandMark size={52} />
          <strong>{buildProgress.label}</strong>
          <BuildProgressStatus progress={buildProgress} />
          {initializationControlBrainId ?? initialMaterialsBrainId ? <PluginRuntimeApprovalQueue
            key={initializationControlBrainId ?? initialMaterialsBrainId}
            brainId={(initializationControlBrainId ?? initialMaterialsBrainId)!} /> : null}
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
            if (!deletingInstance) {
              pendingDeleteTargets.current = [];
              setDeleteTarget(null);
            }
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
            <StorageOperationTiming operation={storageOperation} />
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
        <ToastNotice key={toast.id} text={toast.text} onDismiss={() => setToast(current => current?.id === toast.id ? null : current)} />
      ) : null}
    </div>
  );
}
