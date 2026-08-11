import {
  Fragment,
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
  type BuildResourceSelection,
  type CatalogEntry,
  type ChatMessage,
  type ChatStreamEvent,
  type HardwareTier,
  type HardwareProfile as SystemHardwareProfile,
  type IngestResult,
  type RuntimeJob,
  type InstalledModalityPack,
  type ToolExecutionResult,
  type ToolInvocation,
  type ToolPermissionRecord,
  type AgentMergePreview,
  type ActionEvent,
  type AppearanceLayout,
  type AppearanceMode,
  type AppearancePalette,
  type AppearancePreferences,
  type SubstratePage,
  type WorkspaceSnapshot
} from "@shared/types";
import {
  EXPERIENCE_UPLOADS,
  type ExperienceUploadKind
} from "@shared/uploadSupport";
import { demoSummaries, makeDemoBrain, makeDemoChat } from "./demo";
import { EvolutionWorkspace } from "./EvolutionWorkspace";
import { Icon, type IconName } from "./icons";
import {
  createTextFrameBatcher,
  mergeChatActionEvent,
  patchChatActionPreview,
  type TextFrameBatcher
} from "./chatStreaming";
import {
  APPEARANCE_LAYOUTS,
  APPEARANCE_MODES,
  APPEARANCE_PACKS,
  APPEARANCE_PALETTES,
  DEFAULT_APPEARANCE,
  activeAppearancePack,
  applyAppearanceAttributes,
  loadAppearancePreferences,
  preferencesForPack,
  resolveColorScheme,
  saveAppearancePreferences
} from "./appearance";

type AppPage = "library" | "build" | "workspace";
type WorkspaceView =
  | "chat"
  | "data"
  | "map"
  | "trace"
  | "imagine"
  | "tools"
  | "agents"
  | "evolution";
type HardwareChoice = "auto" | "micro" | "personal" | "gpu" | "workstation";
type PermissionLevel = "off" | "ask" | "auto" | "full";
type ModalityId = "vision" | "image" | "audio" | "video";
type OriginKind = "blank" | "starter";

interface BuilderExtras {
  hardware: HardwareChoice;
  origin: OriginKind;
  starterUrl: string;
  initialTraining: boolean;
  initialResources: BuilderResource[];
  continuousLearning: boolean;
  retainExactSources: boolean;
  extendedWorkingMemory: boolean;
  recursiveImprovement: boolean;
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
    features: ["1.58-bit", "STDP", "CfC", "VSA"]
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
    short: "Leaky integrate-and-fire neurons with local spike-timing plasticity.",
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
    short: "Start with a blank blueprint and choose every cognitive subsystem.",
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
  { id: "agents", label: "Forks & agents", icon: "fork" },
  { id: "evolution", label: "Evolution", icon: "pulse" }
];

const toolRows = [
  ["files", "Local files", "file"],
  ["powershell", "PowerShell", "terminal"],
  ["code", "Code workspace", "code"],
  ["web", "Web access", "search"],
  ["browser", "Browser control", "expand"],
  ["imagination", "Imagination", "sparkles"],
  ["agents", "Subagents", "agents"],
  ["evolution", "Recursive improvement", "pulse"]
] as const;

function cx(...values: Array<string | false | null | undefined>) {
  return values.filter(Boolean).join(" ");
}

function compactNumber(value: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function formatBytes(value: number) {
  if (value < 1_024) return `${value} B`;
  if (value < 1_048_576) return `${(value / 1_024).toFixed(1)} KB`;
  return `${(value / 1_048_576).toFixed(1)} MB`;
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
    <button className={cx("button", `button--${kind}`, className)} {...props}>
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

function AppearanceMenu({
  preferences,
  resolvedColorScheme,
  onChange
}: {
  preferences: AppearancePreferences;
  resolvedColorScheme: "light" | "dark";
  onChange: (preferences: AppearancePreferences) => void;
}) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const activePack = activeAppearancePack(preferences);

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
              <small>Color and layout stay separate from every brain.</small>
            </span>
            <button
              className="icon-button"
              aria-label="Close appearance settings"
              onClick={() => setOpen(false)}
            >
              <Icon name="close" size={14} />
            </button>
          </header>

          <fieldset className="appearance-fieldset">
            <legend>Mode</legend>
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
            <legend>Quick packs</legend>
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
              <legend>Layout</legend>
              <div className="appearance-options">
                {APPEARANCE_LAYOUTS.map((layout) => (
                  <button
                    key={layout}
                    aria-pressed={preferences.layout === layout}
                    className={preferences.layout === layout ? "is-active" : ""}
                    onClick={() => onChange({ ...preferences, layout })}
                  >
                    {appearanceLayoutLabels[layout]}
                  </button>
                ))}
              </div>
            </fieldset>
          </div>
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
  resolvedColorScheme,
  onAppearanceChange
}: {
  page: AppPage;
  brain: BrainDocument | null;
  demo: boolean;
  onLibrary: () => void;
  appearance: AppearancePreferences;
  resolvedColorScheme: "light" | "dark";
  onAppearanceChange: (preferences: AppearancePreferences) => void;
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
          resolvedColorScheme={resolvedColorScheme}
          onChange={onAppearanceChange}
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
  onBuild,
  onImport,
  demo
}: {
  summaries: BrainSummary[];
  loading: boolean;
  onOpen: (summary: BrainSummary) => void;
  onDuplicate: (summary: BrainSummary) => void;
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
            <Button icon="upload" onClick={onImport}>
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
            <h2>Your minds</h2>
            <p>{loading ? "Looking for local brains…" : `${summaries.length} persistent instances on this device`}</p>
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
            {visible.map((brain, index) => {
              const meta = recipeMeta.find((recipe) => recipe.id === brain.preset) ?? recipeMeta[0]!;
              const activity = index === 0 ? [25, 48, 39, 72, 55, 86, 68, 92, 76, 95, 83, 100] : [20, 31, 52, 44, 67, 38, 62, 71, 58, 79, 68, 73];
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
                      {index === 0 ? <span className="brain-avatar__live" /> : null}
                    </div>
                    <button
                      className="brain-card__duplicate"
                      aria-label={`Duplicate ${brain.name}`}
                      title={`Duplicate ${brain.name} with copy-on-write neural storage`}
                      onClick={(event) => {
                        event.stopPropagation();
                        onDuplicate(brain);
                      }}
                    >
                      <Icon name="copy" size={13} /> Duplicate
                    </button>
                  </div>
                  <div className="brain-card__identity">
                    <h3>{brain.name}</h3>
                    <span>{presetLabel(brain.preset)}</span>
                  </div>
                  <p className="brain-card__thought">
                    {index === 0
                      ? "“Identity may be the pattern that survives its own changes.”"
                      : index === 1
                        ? "Exploring rhythm as a form of temporal memory."
                        : "Consolidating patterns from the latest code corpus."}
                  </p>
                  <div className="mini-activity" aria-label="Recent neural activity">
                    {activity.map((height, barIndex) => (
                      <span key={barIndex} style={{ height: `${height}%` }} />
                    ))}
                  </div>
                  <div className="brain-card__stats">
                    <span>
                      <strong>{compactNumber(brain.concepts)}</strong> ideas
                    </span>
                    <span>
                      <strong>{compactNumber(brain.synapses)}</strong> synapses
                    </span>
                    <span>
                      <strong>G{brain.generation}</strong> lineage
                    </span>
                  </div>
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
              <strong>Build another mind</strong>
              <span>Start trained or begin with a blank origin</span>
            </button>
          </div>
        ) : (
          <div className="library-empty">
            <BrandMark size={58} />
            <h3>{query ? "No matching minds" : "This library is waiting for its first mind"}</h3>
            <p>{query ? "Try a different name." : "Choose an origin and grow something new."}</p>
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
              <strong>Tip:</strong> Fork a brain before a bold experiment. Its origin always stays untouched.
            </span>
          </div>
        </div>
      </section>
    </main>
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
  ["Identity", "Name the mind and choose its origin"],
  ["Learning", "Choose memory and continuous growth"],
  ["Senses & data", "Add modalities and first experiences"],
  ["Permissions", "Review tools and create the origin"]
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
  const [starterEntries, setStarterEntries] = useState<CatalogEntry[]>([]);
  const [webDraft, setWebDraft] = useState("");
  const pendingSelections = useRef<BuilderResource[]>([]);
  const [config] = useState<BrainConfig>(() => createPresetConfig("whole-brain", "Nova"));
  const [extras, setExtras] = useState<BuilderExtras>({
    hardware: "auto",
    origin: "starter",
    starterUrl: "",
    initialTraining: false,
    initialResources: [],
    continuousLearning: true,
    retainExactSources: false,
    extendedWorkingMemory: false,
    recursiveImprovement: true,
    modalities: { vision: true, image: true, audio: true, video: true },
    tools: {
      files: "ask",
      powershell: "ask",
      code: "ask",
      web: "ask",
      browser: "ask",
      imagination: "auto",
      agents: "ask",
      evolution: "ask"
    }
  });

  useEffect(() => {
    let active = true;
    if (!window.omni) return () => {
      active = false;
    };
    void window.omni.catalog.hardwareProfile().then((profile) => {
      if (active) setDetectedHardware(profile);
    });
    void window.omni.catalog.list().then((entries) => {
      if (!active) return;
      const starters = entries.filter((entry) => entry.kind === "brain");
      setStarterEntries(starters);
    });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    pendingSelections.current = extras.initialResources;
  }, [extras.initialResources]);

  useEffect(
    () => () => {
      for (const resource of pendingSelections.current) {
        if (resource.kind !== "web") {
          void window.omni?.data.discardBuildResource(resource.id);
        }
      }
    },
    []
  );

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
    if (!/^https:\/\//i.test(url)) return;
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
      preset: "whole-brain" as const,
      runtime: "adaptive-core" as const,
      description: "A persistent adaptive OmniCortex identity with a unified neural substrate.",
      onlineLearning: extras.continuousLearning,
      workingMemorySlots: extras.extendedWorkingMemory
        ? Math.max(config.workingMemorySlots * 2, 64)
        : config.workingMemorySlots,
      retainSourceText: extras.retainExactSources,
      memoryRecipe: extras.retainExactSources ? "total-recall" as const : "human-consolidation" as const,
      extendedWorkingMemory: extras.extendedWorkingMemory,
      recursiveImprovement: extras.recursiveImprovement,
      idleCognition: true
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

  const stageContent = [
    <div className="builder-stage simple-builder-stage" key="identity">
      <div className="builder-stage__intro">
        <span className="stage-number">01</span>
        <div>
          <h2>Who are you creating?</h2>
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
      <div className="origin-selector simple-origin-selector">
        <div className="settings-panel__title">
          <span><Icon name="sparkles" size={17} /> Starting knowledge</span>
          <small>Both choices use the same custom OmniCortex architecture</small>
        </div>
        <div className="origin-selector__choices">
          <button
            className={extras.origin === "starter" ? "is-selected" : ""}
            onClick={() => setExtras((current) => ({ ...current, origin: "starter" }))}
          >
            <span className="choice-card__icon"><Icon name="download" size={19} /></span>
            <span>
              <strong>Omni Starter <em className="recommended-chip">Recommended</em></strong>
              <small>Begin with initially trained language, action, and concept pathways, then keep adapting.</small>
            </span>
            <i><Icon name="check" size={12} /></i>
          </button>
          <button
            className={extras.origin === "blank" ? "is-selected" : ""}
            onClick={() => setExtras((current) => ({ ...current, origin: "blank", starterUrl: "" }))}
          >
            <span className="choice-card__icon"><Icon name="plus" size={19} /></span>
            <span>
              <strong>Blank Brain</strong>
              <small>Advanced: random weights and primitive output until enough training is completed.</small>
            </span>
            <i><Icon name="check" size={12} /></i>
          </button>
        </div>
        {extras.origin === "starter" ? (
          <div className="starter-source-panel">
            {starterEntries.length ? (
              <div className="starter-entry-list">
                {starterEntries.map((entry) => (
                  <button
                    key={entry.id}
                    className={extras.starterUrl === entry.sourceUrl ? "is-selected" : ""}
                    onClick={() => setExtras((current) => ({ ...current, starterUrl: entry.sourceUrl }))}
                  >
                    <span><Icon name="check" size={13} /></span>
                    <strong>{entry.name}</strong>
                    <small>{entry.license}</small>
                  </button>
                ))}
              </div>
            ) : null}
            <label className="starter-url">
              <span>
                {starterEntries.length
                  ? "Or use another verified .omni URL"
                  : "Optional verified .omni starter URL"}
              </span>
              <div>
                <Icon name="download" size={15} />
                <input
                  value={extras.starterUrl}
                  onChange={(event) => setExtras((current) => ({ ...current, starterUrl: event.target.value }))}
                  placeholder="https://…/omni-starter.omni"
                />
              </div>
              <small>
                Leave blank for the bundled, locally materialized Omni Starter. Downloads accept only
                checksummed manifests and safe tensors; repository scripts never execute.
              </small>
            </label>
          </div>
        ) : null}
      </div>
    </div>,

    <div className="builder-stage simple-builder-stage" key="learning">
      <div className="builder-stage__intro">
        <span className="stage-number">02</span>
        <div>
          <h2>How should learning live?</h2>
          <p>These choices control storage and permission—not personality. Curiosity, uncertainty, and pacing emerge from measured neural state.</p>
        </div>
      </div>
      <div className="simple-toggle-grid">
        <Toggle
          checked={extras.continuousLearning}
          onChange={(continuousLearning) => setExtras((current) => ({ ...current, continuousLearning }))}
          label="Learn continuously"
          description="Conversations and completed actions can update fast synapses and queued slow weights."
        />
        <Toggle
          checked={extras.retainExactSources}
          onChange={(retainExactSources) => setExtras((current) => ({ ...current, retainExactSources }))}
          label="Retain exact sources"
          description="Keep a local content-addressed archive in addition to learned neural state."
        />
        <Toggle
          checked={extras.extendedWorkingMemory}
          onChange={(extendedWorkingMemory) => setExtras((current) => ({ ...current, extendedWorkingMemory }))}
          label="Extended working memory"
          description="For new Blank and bundled Starter builds, double both recent-token context and recurrent latent assembly slots. Imported checkpoints retain their recorded shape."
        />
        <Toggle
          checked={extras.recursiveImprovement}
          onChange={(recursiveImprovement) =>
            setExtras((current) => ({
              ...current,
              recursiveImprovement,
              tools: {
                ...current.tools,
                evolution: recursiveImprovement
                  ? (current.tools.evolution ?? "off") === "off" ? "ask" : (current.tools.evolution ?? "ask")
                  : "off"
              }
            }))
          }
          label="Recursive improvement"
          description="Let the brain test isolated improvement candidates; promotion still follows tool permissions."
        />
      </div>
      <div className="working-memory-explainer">
        <span><Icon name="memory" size={22} /></span>
        <div>
          <strong>Working memory is temporary; learning is structural.</strong>
          <p>Recent-token context is separate from recurrent latent assembly slots. Sensory latents, liquid state, and active assemblies use the neural workspace; useful experience consolidates into ternary synapses instead of becoming a hidden prompt.</p>
        </div>
      </div>
      <details className="research-diagnostics">
        <summary>
          <span><Icon name="pulse" size={16} /> Research diagnostics</span>
          <span>Measured internals <Icon name="chevron" size={14} /></span>
        </summary>
        <div className="research-diagnostics__grid">
          <span><small>Forward synapses</small><strong>Exact −1 · 0 · +1</strong></span>
          <span><small>Fast plasticity</small><strong>LIF + STDP</strong></span>
          <span><small>Temporal state</small><strong>CfC liquid control</strong></span>
          <span><small>Whole-input workspace</small><strong>Global latent integration</strong></span>
          <span><small>Growth</small><strong>Resource-governed, no model cap</strong></span>
          <span>
            <small>Hardware profile</small>
            <strong>{detectedHardware ? `${detectedHardware.recommendedTier} · ${detectedHardware.logicalCpus} threads` : "Detected at build time"}</strong>
          </span>
        </div>
      </details>
    </div>,

    <div className="builder-stage simple-builder-stage" key="senses">
      <div className="builder-stage__intro">
        <span className="stage-number">03</span>
        <div>
          <h2>What can it experience first?</h2>
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
          <button
            key={id}
            className={cx("modality-card", extras.modalities[id] && "is-selected")}
            onClick={() =>
              setExtras((current) => ({
                ...current,
                modalities: { ...current.modalities, [id]: !current.modalities[id] }
              }))
            }
          >
            <span className="modality-card__icon"><Icon name={icon} size={20} /></span>
            <span><strong>{title}</strong><p>{copy}</p></span>
            <span className="modality-card__check"><Icon name="check" size={12} /></span>
          </button>
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
            placeholder="https://site.example/resource"
          />
          <button
            disabled={!/^https:\/\//i.test(webDraft.trim())}
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
              aria-label={`Remove ${resource.label} from this build`}
              title="Remove from this build; original files are not deleted"
              onClick={() => removeInitialResource(resource)}
            >
              <Icon name="close" size={13} /> Remove
            </button>
          </div>
        ))}
        {!extras.initialResources.length ? (
          <p>No initial resources selected. The brain can still learn later from chat or Data Studio.</p>
        ) : null}
      </div>
      <small className="build-resource-note">
        Remove only changes this build list; it never deletes or modifies the original files.
        Every selected document, dataset, image, audio clip, video, or folder is queued for
        neural encoding and training. Web sources crawl same-site in parallel, including linked
        image, audio, and video resources.
      </small>
    </div>,

    <div className="builder-stage simple-builder-stage" key="review">
      <div className="builder-stage__intro">
        <span className="stage-number">04</span>
        <div>
          <h2>Choose action permissions.</h2>
          <p>Tools, imagination, agents, and evolution are available directly in conversation. Every external action remains visible.</p>
        </div>
      </div>
      <div className="tool-table simple-tool-table">
        {toolRows.map(([id, title, icon]) => (
          <div className="tool-row" key={id}>
            <span className="tool-row__identity">
              <span><Icon name={icon} size={17} /></span>
              <strong>{title}</strong>
            </span>
            <div className="segmented segmented--permissions">
              {(["off", "ask", "auto", "full"] as const).map((permission) => (
                <button
                  key={permission}
                  className={extras.tools[id] === permission ? "is-active" : ""}
                  onClick={() =>
                    setExtras((current) => ({
                      ...current,
                      tools: { ...current.tools, [id]: permission }
                    }))
                  }
                >
                  {permission === "full" ? "Full" : permission[0]?.toUpperCase() + permission.slice(1)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
      {Object.values(extras.tools).includes("full") ? (
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
          <div><dt>Origin</dt><dd>{extras.origin === "starter" ? "Initially trained Omni Starter" : "Blank random brain"}</dd></div>
          <div><dt>Learning</dt><dd>{extras.continuousLearning ? "Continuous" : "Manual"}</dd></div>
          <div><dt>Memory</dt><dd>{extras.retainExactSources ? "Neural state + exact archive" : "Human consolidation"}</dd></div>
          <div><dt>Workspace</dt><dd>{extras.extendedWorkingMemory ? "Extended" : "Hardware-sized"}</dd></div>
          <div><dt>Recursive improvement</dt><dd>{extras.recursiveImprovement ? "Experiments enabled" : "Off"}</dd></div>
          <div><dt>First learning</dt><dd>{extras.initialResources.length ? `${extras.initialResources.length} queued resource${extras.initialResources.length === 1 ? "" : "s"}` : "From conversation"}</dd></div>
          <div><dt>Modalities</dt><dd>{Object.values(extras.modalities).filter(Boolean).length} enabled</dd></div>
          <div><dt>Behavioral prompt / RLHF</dt><dd>None</dd></div>
        </dl>
      </div>
    </div>
  ];

  const invalidStarter =
    Boolean(window.omni) &&
    extras.origin === "starter" &&
    Boolean(extras.starterUrl.trim()) &&
    !/^https:\/\//i.test(extras.starterUrl.trim());
  return (
    <main className="builder-page simple-builder">
      <aside className="builder-sidebar">
        <button
          className="builder-back"
          onClick={() => {
            for (const resource of extras.initialResources) {
              if (resource.kind !== "web") {
                void window.omni?.data.discardBuildResource(resource.id);
              }
            }
            onCancel();
          }}
        >
          <Icon name="arrow" size={15} /> Brain Library
        </button>
        <div className="builder-sidebar__intro">
          <span>NEW ORIGIN</span>
          <h1>Build a brain</h1>
          <p>Four clear choices. The neural details adapt automatically to this machine.</p>
        </div>
        <ol className="step-list">
          {simpleBuildSteps.map(([title, copy], index) => (
            <li key={title} className={cx(index === step && "is-active", index < step && "is-complete")}>
              <button onClick={() => setStep(index)} aria-current={index === step ? "step" : undefined}>
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
        <div className="builder-main__content">{stageContent[step]}</div>
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
                disabled={step === 0 && (!name.trim() || invalidStarter)}
                onClick={() => setStep((current) => current + 1)}
              >
                Continue <Icon name="arrow" size={14} />
              </Button>
            ) : (
              <Button
                kind="primary"
                icon={building ? "pulse" : "sparkles"}
                disabled={!name.trim() || invalidStarter || building}
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
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  view: WorkspaceView;
  onView: (view: WorkspaceView) => void;
  onLibrary: () => void;
  onDuplicate: () => void;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [health, setHealth] = useState("Adaptive core ready");
  const [compactInspectorOpen, setCompactInspectorOpen] = useState(false);
  const toggleLearning = async () => {
    const nextConfig = { ...brain.config, onlineLearning: !brain.config.onlineLearning };
    try {
      const updated = window.omni
        ? await window.omni.brain.update(brain.id, nextConfig)
        : { ...brain, config: nextConfig, updatedAt: new Date().toISOString() };
      onBrainChange(updated);
      onToast(nextConfig.onlineLearning ? "Online learning resumed." : "Online learning paused.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Learning state could not be changed.");
    }
  };

  useEffect(() => {
    let active = true;
    if (window.omni) {
      void window.omni.brain.health(brain.id).then((result) => {
        if (active) {
          setHealth(
            result.ready
              ? `Python engine${result.pid ? ` · PID ${result.pid}` : ""}`
              : result.detail
          );
        }
      });
    } else {
      setHealth("Demo preview · engine disconnected");
    }
    return () => {
      active = false;
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
        <button className="rail-library" onClick={onLibrary} aria-label="Brain library" title="Brain library">
          <Icon name="library" size={19} />
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
              title={item.label}
            >
              <Icon name={item.icon} size={19} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="workspace-rail__bottom">
          <button
            aria-label="Open local data folder"
            title="Open local data folder"
            onClick={() => void window.omni?.window.revealDataFolder()}
          >
            <Icon name="archive" size={19} />
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
              <i /> {brain.config.onlineLearning ? "Learning" : "Learning paused"}
            </span>
          </div>
          <div className="workspace-header__activity">
            <span className="signal-bars">
              {[35, 70, 45, 88, 62].map((height, index) => (
                <i key={index} style={{ height: `${height}%` }} />
              ))}
            </span>
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
              aria-label={`Duplicate ${brain.name}`}
              title="Duplicate this identity with copy-on-write neural storage"
              onClick={onDuplicate}
            >
              <Icon name="copy" size={14} /> <span>Duplicate</span>
            </button>
            <button
              className="icon-button"
              aria-label={brain.config.onlineLearning ? "Pause learning" : "Resume learning"}
              title={brain.config.onlineLearning ? "Pause learning" : "Resume learning"}
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
        />
        {view === "chat" ? null : view === "data" ? (
          <DataWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        ) : view === "map" ? (
          <BrainMapWorkspace brain={brain} />
        ) : view === "trace" ? (
          <TraceWorkspace brain={brain} />
        ) : view === "imagine" ? (
          <ImaginationWorkspace brain={brain} onToast={onToast} />
        ) : view === "tools" ? (
          <ToolsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        ) : view === "agents" ? (
          <AgentsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        ) : (
          <EvolutionWorkspace
            brain={brain}
            onOpenPermissions={() => onView("tools")}
            onToast={onToast}
          />
        )}
      </section>
    </main>
  );
}

interface ChatToolCommand {
  label: string;
  invocation: Omit<ToolInvocation, "brainId" | "approvalToken">;
  source: "human" | "brain";
  /** Links a one-action approval back to its original visible action card. */
  actionEventId?: string;
}

interface PendingChatTool extends ChatToolCommand {
  approvalToken: string;
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

function toolExperience(command: ChatToolCommand, output: unknown): string {
  let serialized = "";
  try {
    serialized = JSON.stringify(
      output,
      (key, value) =>
        key === "dataUrl" && typeof value === "string"
          ? `[embedded media omitted from text experience; ${value.length} characters]`
          : value,
      2
    );
  } catch {
    serialized = JSON.stringify({ error: "Tool output was not serializable." });
  }
  return [
    "[Visible structured tool result]",
    `tool: ${command.invocation.toolId}`,
    `action: ${command.invocation.action}`,
    `requested-by: ${command.source}`,
    `request: ${command.label.slice(0, 4_000)}`,
    "result:",
    serialized.slice(0, 48_000)
  ].join("\n");
}

function valueRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function ChatActionCard({ event }: { event: ActionEvent }) {
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
  const output = valueRecord(event.execution?.output);
  const preview = event.preview;
  const rawDataUrl =
    typeof output?.dataUrl === "string" ? output.dataUrl : preview?.dataUrl;
  const mimeType =
    typeof output?.mimeType === "string" ? output.mimeType : preview?.mimeType ?? "";
  const dataUrl =
    rawDataUrl &&
    /^(?:image|audio|video)\/[a-z0-9.+-]+$/i.test(mimeType) &&
    rawDataUrl.startsWith(`data:${mimeType};base64,`)
      ? rawDataUrl
      : null;
  const protocol = event.action.toolId
    ? `${event.action.toolId}${event.action.action ? ` · ${event.action.action}` : ""}`
    : event.action.kind;
  const argumentsText = Object.keys(event.action.arguments).length
    ? JSON.stringify(event.action.arguments)
    : "";
  const artifactPath =
    typeof output?.path === "string"
      ? output.path
      : typeof output?.artifactPath === "string"
        ? output.artifactPath
        : preview?.path ?? preview?.artifactPath ?? "";

  return (
    <article className={cx("chat-action-card", `chat-action-card--${event.state}`)}>
      <span className="chat-action-card__icon"><Icon name={iconByKind[event.action.kind]} size={18} /></span>
      <div className="chat-action-card__body">
        <div className="chat-action-card__head">
          <span>
            <small>{event.action.source === "organic" ? "ORGANIC ACTION" : `${event.action.source.toUpperCase()} ACTION`}</small>
            <strong>{protocol}</strong>
          </span>
          <em><i /> {event.state.replace("-", " ")}</em>
        </div>
        {argumentsText ? <code>{argumentsText.length > 360 ? `${argumentsText.slice(0, 357)}…` : argumentsText}</code> : null}
        {event.error ? <p>{event.error}</p> : null}
        {event.state === "running" && event.progress !== undefined ? (
          <span
            className="chat-action-card__progress"
            aria-label={`${Math.round(event.progress * 100)} percent complete`}
          >
            <i style={{ width: `${Math.round(event.progress * 100)}%` }} />
            <small>{event.statusLabel ?? "Generating from active neural assemblies"}</small>
          </span>
        ) : null}
        {preview && event.state === "running" ? (
          <small className="chat-action-card__preview-label">
            Progressive imagination · revision {preview.revision + 1}
          </small>
        ) : null}
        {dataUrl && mimeType.startsWith("image/") ? (
          <img src={dataUrl} alt="Artifact created by the local imagination action" />
        ) : dataUrl && mimeType.startsWith("audio/") ? (
          <audio src={dataUrl} controls />
        ) : dataUrl && mimeType.startsWith("video/") ? (
          <video src={dataUrl} controls />
        ) : artifactPath ? (
          <span className="chat-action-card__artifact"><Icon name="file" size={14} /> {artifactPath}</span>
        ) : null}
        <span className="chat-action-card__meta">
          {event.action.confidence !== undefined ? `${Math.round(event.action.confidence * 100)}% action confidence · ` : ""}
          {new Date(event.updatedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        </span>
      </div>
    </article>
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

function ChatWorkspace({
  brain,
  hidden,
  onBrainChange,
  onToast,
  onNavigate
}: {
  brain: BrainDocument;
  hidden?: boolean;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
  onNavigate: (view: WorkspaceView) => void;
}) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [attachmentDragActive, setAttachmentDragActive] = useState(false);
  const [attachmentReceipts, setAttachmentReceipts] = useState<
    AttachmentLearningReceipt[]
  >([]);
  const [pendingTool, setPendingTool] = useState<PendingChatTool | null>(null);
  const [toolStatus, setToolStatus] = useState("");
  const [toolRunning, setToolRunning] = useState(false);
  const [actionEvents, setActionEvents] = useState<ActionEvent[]>([]);
  const [partialText, setPartialText] = useState("");
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const [optimisticHumans, setOptimisticHumans] = useState<ChatMessage[]>([]);
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState<WorkspaceSnapshot | null>(null);
  const [inspectorTab, setInspectorTab] = useState<"state" | "runtime">("state");
  const messagesEnd = useRef<HTMLDivElement>(null);
  const activeTurnIdRef = useRef<string | null>(null);
  const streamSequenceRef = useRef(new Map<string, number>());
  const tokenBatcherRef = useRef<TextFrameBatcher | null>(null);
  const cancelRequestedRef = useRef(false);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ block: "end" });
  }, [brain.messages, sending, partialText, actionEvents]);

  useEffect(() => {
    if (!window.omni) return;
    return window.omni.chat.onAction((event) => {
      if (event.brainId !== brain.id) return;
      setActionEvents((current) => mergeChatActionEvent(current, event));
      const protocol = `${event.action.toolId ?? event.action.kind}.${event.action.action ?? event.action.kind}`;
      setToolRunning(event.state === "running");
      setToolStatus(
        event.state === "failed"
          ? `${protocol} failed: ${event.error ?? "unknown error"}`
          : `${protocol}: ${event.state}.`
      );
      if (
        event.state === "approval-required" &&
        event.execution?.approvalToken &&
        event.action.toolId &&
        event.action.action
      ) {
        setPendingTool({
          label: `${protocol} proposed in chat`,
          source: event.action.source === "human" ? "human" : "brain",
          actionEventId: event.id,
          invocation: {
            toolId: event.action.toolId,
            action: event.action.action,
            arguments: event.action.arguments
          },
          approvalToken: event.execution.approvalToken
        });
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
      (delta) => setPartialText((current) => current + delta),
      (callback) => window.requestAnimationFrame(callback),
      (handle) => window.cancelAnimationFrame(handle)
    );
    tokenBatcherRef.current = batcher;
    const removeListener = window.omni.chat.onStream((event: ChatStreamEvent) => {
      if (event.brainId !== brain.id) return;
      const lastSequence = streamSequenceRef.current.get(event.turnId) ?? -1;
      if (event.sequence <= lastSequence) return;
      streamSequenceRef.current.set(event.turnId, event.sequence);
      if (event.turnId !== activeTurnIdRef.current) return;
      if (event.type === "chat-token") {
        batcher.push(event.delta);
      } else if (event.type === "chat-action") {
        setActionEvents((current) =>
          mergeChatActionEvent(current, event.actionEvent)
        );
        const protocol = `${event.actionEvent.action.toolId ?? event.actionEvent.action.kind}.${event.actionEvent.action.action ?? event.actionEvent.action.kind}`;
        setToolRunning(event.actionEvent.state === "running");
        setToolStatus(
          event.actionEvent.state === "failed"
            ? `${protocol} failed: ${event.actionEvent.error ?? "unknown error"}`
            : `${protocol}: ${event.actionEvent.state}.`
        );
      } else if (event.type === "modality-preview") {
        setActionEvents((current) =>
          patchChatActionPreview(current, event.actionId, event.preview)
        );
      } else if (event.type === "chat-state" && event.state !== "started") {
        batcher.flush();
        streamSequenceRef.current.delete(event.turnId);
        activeTurnIdRef.current = null;
      }
    });
    return () => {
      removeListener();
      batcher.reset();
      if (tokenBatcherRef.current === batcher) tokenBatcherRef.current = null;
      streamSequenceRef.current.clear();
    };
  }, [brain.id]);

  useEffect(() => {
    const brainApi = window.omni?.brain;
    if (!brainApi?.workspace) {
      setWorkspaceSnapshot(null);
      return;
    }
    let active = true;
    void brainApi.workspace(brain.id).then((snapshot) => {
      if (active) setWorkspaceSnapshot(snapshot);
    }).catch(() => {
      if (active) setWorkspaceSnapshot(null);
    });
    return () => {
      active = false;
    };
  }, [brain.id, brain.updatedAt, sending]);

  const runChatTool = async (
    command: ChatToolCommand,
    approvalToken?: string
  ): Promise<void> => {
    if (!window.omni) {
      setToolStatus("Direct chat tools require the packaged desktop runtime.");
      return;
    }
    setToolRunning(true);
    setToolStatus(`${command.label} is running through the visible tool protocol.`);
    let execution: ToolExecutionResult;
    try {
      execution = await window.omni.tool.execute({
        brainId: brain.id,
        ...command.invocation,
        approvalToken
      });
    } finally {
      setToolRunning(false);
    }
    if (execution.state === "approval-required" && execution.approvalToken) {
      setPendingTool({ ...command, approvalToken: execution.approvalToken });
      setToolStatus(
        `${command.label} is waiting for one-action approval. The exact arguments are locked to this approval.`
      );
      return;
    }
    setPendingTool(null);
    if (command.actionEventId) {
      setActionEvents((current) =>
        current.map((event) =>
          event.id === command.actionEventId
            ? {
                ...event,
                state: execution.state,
                execution,
                error: execution.error,
                updatedAt: execution.finishedAt ?? new Date().toISOString()
              }
            : event
        )
      );
    }
    if (execution.state === "failed") {
      setToolStatus(`${command.invocation.toolId}.${command.invocation.action} failed: ${execution.error ?? "unknown error"}`);
      onBrainChange(await window.omni.brain.get(brain.id));
      return;
    }
    setToolStatus(
      `${command.invocation.toolId}.${command.invocation.action} completed and its visible result entered working experience after the job finished.`
    );
    const result = await window.omni.chat.send(
      brain.id,
      toolExperience(command, execution.output ?? { state: execution.state })
    );
    onBrainChange(result.brain);
    if (result.actionEvents?.length) {
      setActionEvents((current) => {
        const merged = new Map(current.map((event) => [event.id, event]));
        result.actionEvents?.forEach((event) => merged.set(event.id, event));
        return [...merged.values()];
      });
    }
  };

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    cancelRequestedRef.current = false;
    let optimisticHuman: ChatMessage | null = null;
    try {
      if (window.omni) {
        setInput("");
        const turnId = globalThis.crypto?.randomUUID?.() ??
          `turn-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        optimisticHuman = {
          id: `pending-${turnId}`,
          role: "human",
          content: text,
          createdAt: new Date().toISOString()
        };
        // Keep speculative presentation state out of the authoritative brain
        // document. This component stays mounted across workspace navigation,
        // so the pending turn remains visible without becoming durable memory.
        setOptimisticHumans((current) => [...current, optimisticHuman!]);
        tokenBatcherRef.current?.reset();
        streamSequenceRef.current.clear();
        activeTurnIdRef.current = turnId;
        setActiveTurnId(turnId);
        setPartialText("");
        const result = await window.omni.chat.send(brain.id, text, turnId);
        setOptimisticHumans((current) =>
          current.filter((message) => message.id !== optimisticHuman?.id)
        );
        onBrainChange(result.brain);
        if (result.actionEvents?.length) {
          setActionEvents((current) => {
            const merged = new Map(current.map((event) => [event.id, event]));
            result.actionEvents?.forEach((event) => merged.set(event.id, event));
            return [...merged.values()];
          });
          const lastAction = result.actionEvents.at(-1);
          if (lastAction) {
            const protocol = `${lastAction.action.toolId ?? lastAction.action.kind}.${lastAction.action.action ?? lastAction.action.kind}`;
            setToolStatus(
              lastAction.state === "failed"
                ? `${protocol} failed: ${lastAction.error ?? "unknown error"}`
                : `${protocol}: ${lastAction.state}.`
            );
            if (
              lastAction.state === "approval-required" &&
              lastAction.execution?.approvalToken &&
              lastAction.action.toolId &&
              lastAction.action.action
            ) {
              setPendingTool({
                label: `${protocol} proposed in natural chat`,
                source: lastAction.action.source === "human" ? "human" : "brain",
                actionEventId: lastAction.id,
                invocation: {
                  toolId: lastAction.action.toolId,
                  action: lastAction.action.action,
                  arguments: lastAction.action.arguments
                },
                approvalToken: lastAction.execution.approvalToken
              });
            }
          }
        }
      } else {
        setInput("");
        const human: ChatMessage = {
          id: `pending-${Date.now()}`,
          role: "human",
          content: text,
          createdAt: new Date().toISOString()
        };
        onBrainChange({ ...brain, messages: [...brain.messages, human] });
        await new Promise((resolve) => window.setTimeout(resolve, 850));
        onBrainChange(makeDemoChat(brain, text).brain);
      }
    } catch (error) {
      if (window.omni && optimisticHuman) {
        setOptimisticHumans((current) =>
          current.filter((message) => message.id !== optimisticHuman?.id)
        );
        // The main process commits the human and neural messages atomically.
        // Reload after failure so any independently completed neural activity
        // is reflected without ever persisting the optimistic bubble.
        await window.omni.brain.get(brain.id).then(onBrainChange).catch(() => {
          // The existing authoritative document remains valid if refresh fails.
        });
      }
      if (!cancelRequestedRef.current) {
        onToast(error instanceof Error ? error.message : "The local brain could not respond.");
      }
    } finally {
      tokenBatcherRef.current?.reset();
      if (activeTurnIdRef.current) {
        streamSequenceRef.current.delete(activeTurnIdRef.current);
      }
      activeTurnIdRef.current = null;
      setActiveTurnId(null);
      setPartialText("");
      setSending(false);
    }
  };

  const approvePendingTool = async () => {
    if (!pendingTool || sending) return;
    setSending(true);
    try {
      await runChatTool(pendingTool, pendingTool.approvalToken);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The approved tool could not run.");
    } finally {
      setSending(false);
    }
  };

  const cancelChatTool = async () => {
    if (!window.omni) return;
    cancelRequestedRef.current = true;
    const count = sending
      ? await window.omni.chat.cancel(brain.id, activeTurnId ?? undefined)
      : await window.omni.tool.cancel(brain.id);
    setToolStatus(
      count > 0
        ? "Cancellation requested. Partial text and artifacts remain visibly marked in this turn."
        : "No cancellable chat or tool execution is active."
    );
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
    if (attaching || sending) return;
    setAttaching(true);
    const label =
      selection === "folder"
        ? "Folder experience"
        : EXPERIENCE_UPLOADS[selection].shortLabel.replace(
            /^./,
            (character) => character.toLocaleUpperCase()
          );
    setToolStatus(
      `Choose ${selection === "folder" ? "a folder" : EXPERIENCE_UPLOADS[selection].shortLabel}; selected material will be learned into parameters and synapses.`
    );
    try {
      const request = {
        brainId: brain.id,
        policy: "consolidate" as const
      };
      const results =
        selection === "folder"
          ? await window.omni.data.ingestFolder(request)
          : await window.omni.data.ingestFiles({ ...request, selection });
      if (results.length > 0) {
        onBrainChange(await window.omni.brain.get(brain.id));
      }
      recordAttachmentResults(results, label);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The attachment could not be learned.");
      setToolStatus("Attachment learning failed; no completed neural update was reported.");
    } finally {
      setAttaching(false);
    }
  };

  const attachDroppedExperience = async (files: File[]): Promise<void> => {
    setAttachmentDragActive(false);
    if (!files.length || attaching || sending) return;
    if (!window.omni) {
      onNavigate("data");
      return;
    }
    setAttaching(true);
    setToolStatus(
      `Learning ${files.length} dropped item${files.length === 1 ? "" : "s"} into parameters and synapses…`
    );
    try {
      const results = await window.omni.data.ingestDropped(
        { brainId: brain.id, policy: "consolidate" },
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
      setAttaching(false);
    }
  };

  const onComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  };

  const authoritativeAndPendingMessages = [...brain.messages, ...optimisticHumans];
  const displayedMessages =
    authoritativeAndPendingMessages.length > 0
      ? authoritativeAndPendingMessages
      : window.omni
        ? []
        : [
          {
            id: "empty-greeting",
            role: "brain" as const,
            content:
              "I am here, but almost nothing has happened to me yet. What should we explore first?",
            createdAt: brain.createdAt,
            runtime: brain.config.runtime
          }
          ];
  const recentTrace = brain.traces.at(-1);
  const activeConcepts = recentTrace?.activatedConcepts ??
    Object.values(brain.concepts)
      .sort((a, b) => b.activation - a.activation)
      .filter((concept) => concept.activation > 0)
      .map((concept) => ({ id: concept.id, label: concept.label, activation: concept.activation }));
  const measuredActivity = recentTrace
    ? Math.max(
        0,
        Math.min(
          1,
          recentTrace.activatedConcepts.length
            ? recentTrace.activatedConcepts.reduce(
                (sum, concept) => sum + concept.activation,
                0
              ) / recentTrace.activatedConcepts.length
            : (
                recentTrace.driveScores.curiosity +
                recentTrace.driveScores.coherence +
                recentTrace.driveScores.novelty
              ) / 3
        )
      )
    : null;
  const pondering = actionEvents.some(
    (event) =>
      event.action.kind === "ponder" &&
      (event.state === "proposed" || event.state === "running")
  );
  const actionRunning =
    toolRunning ||
    actionEvents.some((event) => event.state === "running");
  const cortexState = pondering
    ? "Pondering"
    : actionRunning
        ? "Running action"
        : sending
          ? "Processing turn"
          : "Idle / ready";

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
        <div className="message-stream">
          {!displayedMessages.length ? (
            <div className="chat-empty">
              <span><BrandMark size={42} /></span>
              <strong>{brain.name} has no conversation yet</strong>
              <p>This is one continuous chat. The first real exchange will begin its conversational history.</p>
            </div>
          ) : null}
          {displayedMessages.map((message) => (
            <MessageBubble
              key={message.id}
              message={message}
              brainName={brain.name}
              onToast={onToast}
              onTrace={() => onNavigate("trace")}
            />
          ))}
          {actionEvents.length ? (
            <div className="chat-action-stream" aria-label="Visible chat actions">
              {actionEvents.map((event) => <ChatActionCard key={event.id} event={event} />)}
            </div>
          ) : null}
          {attachmentReceipts.length ? (
            <div className="chat-action-stream" aria-label="Learned chat attachments">
              {attachmentReceipts.map((receipt) => (
                <AttachmentLearningCard key={receipt.id} receipt={receipt} />
              ))}
            </div>
          ) : null}
          {sending ? (
            <div className="message message--brain">
              <div className="message__avatar">
                <BrandMark size={28} />
              </div>
              <div className="message__body">
                <div className="message__meta">
                  <strong>{brain.name}</strong>
                  <span className="pondering-label">
                    <i /> {pondering ? "pondering" : actionRunning ? "acting" : "processing"}
                  </span>
                </div>
                {partialText ? (
                  <p className="message__streaming-text" aria-live="polite">{partialText}</p>
                ) : (
                  <div className="pondering">
                    <span />
                    <span />
                    <span />
                    <em>
                      {pondering
                        ? "Continuing a recurrent ponder cycle…"
                        : actionRunning
                          ? "Running the visible structured action…"
                          : "Processing the current turn…"}
                    </em>
                  </div>
                )}
              </div>
            </div>
          ) : null}
          <div ref={messagesEnd} />
        </div>
        <div className="composer-wrap">
          {toolStatus ? (
            <div className={cx("chat-tool-status", pendingTool && "chat-tool-status--approval")}>
              <span><Icon name={pendingTool ? "warning" : "terminal"} size={15} /></span>
              <p>{toolStatus}</p>
              {pendingTool ? (
                <Button kind="primary" icon="check" disabled={sending} onClick={() => void approvePendingTool()}>
                  Approve exact action
                </Button>
              ) : toolRunning ? (
                <Button kind="primary" icon="close" onClick={() => void cancelChatTool()}>
                  Cancel
                </Button>
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
              placeholder={`Talk to ${brain.name}…`}
              rows={1}
              aria-label={`Message ${brain.name}`}
            />
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
              </div>
              <span>Enter to send · Shift Enter for a line break</span>
              <button
                className="send-button"
                onClick={() => void (sending ? cancelChatTool() : send())}
                disabled={!sending && !input.trim()}
                aria-label={sending ? "Stop current turn" : "Send message"}
              >
                <Icon name={sending ? "close" : "send"} size={17} />
              </button>
            </div>
            <div className="composer__suggestions" aria-label="Natural chat examples">
              <button onClick={() => setInput("Search the web for ")}><Icon name="search" size={13} /> Search the web</button>
              <button onClick={() => setInput("Imagine an image of ")}><Icon name="image" size={13} /> Imagine</button>
              <button onClick={() => setInput("Fork an agent to ")}><Icon name="agents" size={13} /> Create an agent</button>
              <button onClick={() => setInput("Improve your own implementation by ")}><Icon name="pulse" size={13} /> Evolve</button>
            </div>
          </div>
          <p className="composer-disclaimer">
            Learned parameters, active state, and structured tool schemas drive each turn. Tool actions stay permissioned and visible.
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
            <div className="panel-section">
              <div className="panel-section__head">
                <span>Active assemblies</span>
                <em>{activeConcepts.length} spreading</em>
              </div>
              <div className="concept-list">
                {activeConcepts.map((concept) => (
                  <div key={concept.id}>
                    <span>
                      <i className="concept-spark" />
                      {concept.label}
                    </span>
                    <span className="concept-meter">
                      <i style={{ width: `${concept.activation * 100}%` }} />
                    </span>
                    <em>{Math.round(concept.activation * 100)}</em>
                  </div>
                ))}
              </div>
            </div>
            <div className="workspace-meter">
              <div>
                <span>
                  <Icon name="memory" size={15} />
                  {workspaceSnapshot ? "Recent token context" : "Latent assembly workspace"}
                </span>
                <strong>
                  {workspaceSnapshot
                    ? `${workspaceSnapshot.contextWindow.recentTokenCount ?? 0} / ${workspaceSnapshot.contextWindow.capacityTokens} recent tokens`
                    : `${brain.workingMemory.length} / ${brain.config.workingMemorySlots} latent items`}
                </strong>
              </div>
              <i>
                <b style={{ width: `${Math.min(100, (workspaceSnapshot?.contextWindow.capacityTokens ?? brain.config.workingMemorySlots) ? (workspaceSnapshot?.contextWindow.recentTokenCount ?? brain.workingMemory.length) / (workspaceSnapshot?.contextWindow.capacityTokens ?? brain.config.workingMemorySlots) * 100 : 0)}%` }} />
              </i>
              <small>
                {workspaceSnapshot
                  ? `${workspaceSnapshot.contextWindow.recentTokenCount ?? 0} recent dialogue tokens · ${workspaceSnapshot.contextWindow.evictions ?? 0} token evictions · ${workspaceSnapshot.latentWorkspace.occupancy} / ${workspaceSnapshot.latentWorkspace.capacity} latent slots`
                  : `${brain.liquidState.values.length} recurrent channels · ${brain.counters.consolidationCycles} consolidations`}
              </small>
            </div>
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
          <RuntimeCard brain={brain} workspace={workspaceSnapshot} />
        )}
      </aside>
    </div>
  );
}

function MessageBubble({
  message,
  brainName,
  onToast,
  onTrace
}: {
  message: ChatMessage;
  brainName: string;
  onToast: (message: string) => void;
  onTrace: () => void;
}) {
  const isBrain = message.role === "brain";
  return (
    <article className={cx("message", isBrain ? "message--brain" : "message--human")}>
      <div className="message__avatar">
        {isBrain ? <BrandMark size={28} /> : <span className="human-avatar">E</span>}
      </div>
      <div className="message__body">
        <div className="message__meta">
          <strong>{isBrain ? brainName : "You"}</strong>
          <time>
            {new Date(message.createdAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
          </time>
          {isBrain && message.runtime ? <span className="runtime-label">{message.runtime.replace("-", " ")}</span> : null}
        </div>
        <div className="message__content">{message.content}</div>
        {isBrain ? (
          <div className="message__actions">
            <button aria-label="Copy response" onClick={() => void navigator.clipboard?.writeText(message.content)}>
              <Icon name="copy" size={14} />
            </button>
            <button
              aria-label="Show trace"
              onClick={() => (message.traceId ? onTrace() : onToast("No trace is attached to this message."))}
            >
              <Icon name="trace" size={14} />
            </button>
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
          ? "No measured neural activity trace yet"
          : `${Math.round(measured * 100)} percent neural activity in the latest completed trace`
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

function RuntimeCard({ brain, workspace }: { brain: BrainDocument; workspace?: WorkspaceSnapshot | null }) {
  const stableConfig = brain.config as BrainConfig & {
    extendedWorkingMemory?: boolean;
    recursiveImprovement?: boolean;
  };
  const currentTokens =
    [...brain.traces]
      .reverse()
      .find((trace) => trace.steps.some((step) => step.stage === "encode"))
      ?.steps.find((step) => step.stage === "encode")
      ?.value ?? "No completed turn";
  const enabledTools = (brain.toolPermissions ?? [])
    .filter((permission) => permission.level !== "off")
    .map((permission) => permission.toolId);
  const evolutionPermission = (brain.toolPermissions ?? [])
    .find((permission) => permission.toolId === "source.self-modify")?.level ?? "off";
  const rows = [
    ["Behavioral system prompt", workspace ? (workspace.hiddenBehavioralPrompt ? "Present" : "None") : "None"],
    ["Long-term source injection", workspace ? (workspace.rawLongTermTextInjected ? "Present" : "None") : "None"],
    ["Reward model / RLHF", "None"],
    ["Ternary forward paths", "Mandatory · −1 / 0 / +1"],
    ["Runtime", brain.config.runtime],
    [
      "Current context",
      workspace
        ? `${workspace.contextWindow.tokenCount} / ${workspace.contextWindow.capacityTokens} prompt tokens · ${workspace.contextWindow.recentTokenCount ?? 0} recent · ${workspace.contextWindow.evictions ?? 0} evicted`
        : currentTokens
    ],
    [
      "Recent context digest",
      workspace?.contextWindow.recentTokenHash
        ? workspace.contextWindow.recentTokenHash.slice(0, 16)
        : "Empty"
    ],
    [
      "Baseline response budget",
      workspace?.contextWindow.generationBudgetTokens
        ? `${workspace.contextWindow.generationBudgetTokens} tokens · adjusted per turn from neural state`
        : "Hardware baseline · adjusted per turn from neural state"
    ],
    ["Latent assembly workspace", workspace ? `${workspace.latentWorkspace.occupancy} / ${workspace.latentWorkspace.capacity} slots · ${workspace.latentWorkspace.evictions} evictions` : `${brain.workingMemory.length} / ${brain.config.workingMemorySlots} slots${stableConfig.extendedWorkingMemory ? " · extended" : ""}`],
    ["Liquid recurrent state", workspace ? `${workspace.liquidState.dimensions} dimensions · norm ${workspace.liquidState.norm.toFixed(2)}` : `${brain.liquidState.values.length} channels`],
    ["Consolidation", `${brain.counters.consolidationCycles} completed cycles`],
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
      <p>Key prompt inputs, memory occupancy, tool availability, and runtime policy boundaries are summarized here.</p>
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
        Working context is temporary. Tool schemas describe available actions; they do not prescribe a personality.
      </div>
    </div>
  );
}

function DataWorkspace({
  brain,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [mode, setMode] = useState<"uploads" | "catalog" | "web">("uploads");
  const [policy, setPolicy] = useState<"encode" | "consolidate" | "pretrain" | "archive">("consolidate");
  const [demoTraining, setDemoTraining] = useState(false);
  const [demoProgress, setDemoProgress] = useState(0);
  const [localIngestStatus, setLocalIngestStatus] = useState("");
  const [activeJob, setActiveJob] = useState<RuntimeJob | null>(null);
  const [catalogEntries, setCatalogEntries] = useState<CatalogEntry[]>(
    window.omni
      ? []
      : [
          { id: "demo-1", name: "FineWeb-Edu recipe", description: "Curated language learning manifest", sourceUrl: "https://example.com/demo", license: "ODC-By", kind: "dataset" },
          { id: "demo-2", name: "Audio concepts pack", description: "Demo modality recipe", sourceUrl: "https://example.com/demo", license: "CC BY 4.0", kind: "modality-pack" }
        ]
  );
  const [crawlUrl, setCrawlUrl] = useState("");
  const [respectRobots, setRespectRobots] = useState(true);
  const [followExternalLinks, setFollowExternalLinks] = useState(false);
  const [quarantine, setQuarantine] = useState(true);
  const [dragging, setDragging] = useState(false);

  const busy =
    demoTraining || activeJob?.state === "queued" || activeJob?.state === "running";
  const progress = demoTraining
    ? demoProgress
    : activeJob
      ? Math.round(Math.max(0, Math.min(1, activeJob.progress)) * 100)
      : 0;
  const jobOutput = valueRecord(activeJob?.output);
  const coverageValue = valueRecord(jobOutput?.coverage);
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
  const synapses = Object.values(brain.synapses);
  const averageStability = synapses.length
    ? synapses.reduce((sum, synapse) => sum + synapse.stability, 0) / synapses.length
    : 0;
  const averagePlasticity = synapses.length
    ? synapses.reduce((sum, synapse) => sum + synapse.plasticity, 0) / synapses.length
    : 0;
  const activeRatio = synapses.length
    ? synapses.filter((synapse) => synapse.effectiveWeight !== 0).length / synapses.length
    : 0;
  const consolidationRatio = brain.counters.inferenceCount
    ? Math.min(1, brain.counters.consolidationCycles / brain.counters.inferenceCount)
    : 0;
  const healthScore = synapses.length
    ? Math.round(((averageStability + averagePlasticity + activeRatio + consolidationRatio) / 4) * 100)
    : 0;
  const healthRows: Array<[string, number, string]> = [
    ["Stability", averageStability * 100, synapses.length ? "Measured" : "No synapses"],
    ["Plasticity", averagePlasticity * 100, synapses.length ? "Measured" : "No synapses"],
    ["Active paths", activeRatio * 100, `${synapses.filter((item) => item.effectiveWeight !== 0).length} active`],
    ["Consolidation", consolidationRatio * 100, `${brain.counters.consolidationCycles} cycles`]
  ];

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void Promise.all([window.omni.catalog.list(), window.omni.train.list(brain.id)]).then(
      ([entries, jobs]) => {
        if (!active) return;
        setCatalogEntries(entries);
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
      }
    });
    return () => {
      active = false;
      unsubscribe();
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
    if (busy) return;
    setDemoTraining(true);
    setDemoProgress(4);
    setLocalIngestStatus(
      kind === "folder"
        ? "Choosing a whole folder"
        : `Choosing ${EXPERIENCE_UPLOADS[selection].shortLabel}`
    );
    try {
      if (window.omni) {
        const manifest = await window.omni.data.preview({
          brainId: brain.id,
          policy,
          selection: kind === "folder" ? "folder" : selection
        });
        if (!manifest) {
          setLocalIngestStatus("Selection cancelled");
          return;
        }
        setDemoProgress(10);
        setLocalIngestStatus(
          `Committed ${manifest.discoveredFiles} source${manifest.discoveredFiles === 1 ? "" : "s"} to a resumable traversal manifest`
        );
        const job = await window.omni.data.start({
          brainId: brain.id,
          manifestId: manifest.id,
          policy,
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
      onToast(error instanceof Error ? error.message : "Could not ingest the selected files.");
    } finally {
      setDemoTraining(false);
    }
  };

  const ingestDrop = async (files: File[]) => {
    setDragging(false);
    if (!files.length || busy) return;
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
        { brainId: brain.id, policy },
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
    if (!crawlUrl.trim() || busy) return;
    if (!window.omni) {
      onToast("Web crawling is disabled in the browser design preview.");
      return;
    }
    try {
      const job = await window.omni.data.crawlWeb({
        brainId: brain.id,
        url: crawlUrl,
        policy,
        quarantine,
        respectRobots,
        sameOrigin: !followExternalLinks,
        followExternalLinks
      });
      setActiveJob(job);
      onToast(`${quarantine ? "Quarantined " : ""}continuous crawl started. It will run until stopped or resources pause it.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The web crawl could not start.");
    }
  };

  const ingestPage = async () => {
    if (!crawlUrl.trim() || busy) return;
    if (!window.omni) {
      onToast("Single-page web ingestion is disabled in the browser design preview.");
      return;
    }
    try {
      const result = await window.omni.data.ingestWeb({
        brainId: brain.id,
        url: crawlUrl,
        policy,
        quarantine
      });
      onBrainChange(result.brain);
      onToast(`Encoded ${result.source.name} with retained provenance.`);
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The web page could not be ingested.");
    }
  };

  const startTraining = async () => {
    if (busy || !brain.trainingSources.length) return;
    if (!window.omni) {
      onToast("Demo training queue shown; no model weights were changed.");
      return;
    }
    try {
      const job = await window.omni.train.start({
        brainId: brain.id,
        epochs: 3,
        learningRate: brain.config.learningRate,
        sourceIds: brain.trainingSources.map((source) => source.id)
      });
      setActiveJob(job);
      onToast("Slow-weight candidate training queued.");
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Training could not start.");
    }
  };

  const cancelActiveJob = async () => {
    if (!window.omni || !activeJob) return;
    const cancelled = activeJob.kind === "crawl" || activeJob.kind === "ingestion"
      ? await window.omni.data.cancel(activeJob.id)
      : await window.omni.train.cancel(activeJob.id);
    setActiveJob(cancelled);
    onToast(activeJob.kind === "crawl" ? "The crawler was stopped with its frontier saved." : "Cancellation requested.");
  };

  return (
    <div className="content-page data-page">
      <div className="content-page__title">
        <div>
          <span className="eyebrow-text">EXPERIENCE PIPELINE</span>
          <h1>Data & training</h1>
          <p>Turn documents, code, media, and the open web into durable parameters and connected ideas.</p>
        </div>
        <Button icon="upload" kind="primary" onClick={() => void ingest("files")}>
          Add experience
        </Button>
      </div>
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
                  <Button icon="file" disabled={busy} onClick={() => void ingest("files", "files")}>
                    Files & datasets
                  </Button>
                  <Button icon="image" disabled={busy} onClick={() => void ingest("files", "images")}>
                    Images
                  </Button>
                  <Button icon="volume" disabled={busy} onClick={() => void ingest("files", "audio")}>
                    Audio
                  </Button>
                  <Button icon="video" disabled={busy} onClick={() => void ingest("files", "video")}>
                    Video
                  </Button>
                  <Button icon="archive" disabled={busy} onClick={() => void ingest("folder")}>
                    Whole folder
                  </Button>
                </div>
              </div>
            ) : mode === "catalog" ? (
              <div className="catalog-list">
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
                    <Button onClick={() => void (window.omni?.window.openExternal(entry.sourceUrl))}>Inspect</Button>
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
                    <input value={crawlUrl} onChange={(event) => setCrawlUrl(event.target.value)} placeholder="https://docs.example.com" />
                  </div>
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
                  <Button icon="download" disabled={!crawlUrl.trim() || busy} onClick={() => void ingestPage()}>
                    Learn this page
                  </Button>
                  <Button kind="primary" icon="play" disabled={!crawlUrl.trim() || busy} onClick={() => void startCrawl()}>
                    Crawl until stopped
                  </Button>
                </div>
              </div>
            )}
            <div className="ingest-policy">
              <span>
                <strong>When added</strong>
                <small>Choose what the brain does with this experience</small>
              </span>
              <div className="segmented">
                {(["encode", "consolidate", "pretrain", "archive"] as const).map((item) => (
                  <button key={item} className={policy === item ? "is-active" : ""} onClick={() => setPolicy(item)}>
                    {item[0]?.toUpperCase() + item.slice(1)}
                  </button>
                ))}
              </div>
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
                <span>Representation</span>
                <span>Added</span>
                <span />
              </div>
              {brain.trainingSources.map((source) => (
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
                  <span>
                    <strong>{source.learnedIdeas} assembly deltas</strong>
                    <small>{source.learnedSynapses} synaptic deltas</small>
                  </span>
                  <span>
                    <i className="representation-pill">
                      {source.rawTextRetained ? "Source + parameters" : "Parameters only"}
                    </i>
                  </span>
                  <span>{relativeTime(source.importedAt)}</span>
                  <span />
                </div>
              ))}
              {brain.trainingSources.length === 0 ? (
                <div className="table-empty">No external experiences have been encoded yet.</div>
              ) : null}
            </div>
          </div>
        </section>

        <aside className="training-sidebar">
          <div className="surface training-card">
            <div className="training-card__head">
              <span className="training-card__icon">
                <Icon name={busy ? "pulse" : activeJob?.state === "complete" ? "check" : "info"} size={20} />
              </span>
              <span>
                <small>{busy ? "ACTIVE JOB" : activeJob ? "LATEST JOB" : "TRAINING QUEUE"}</small>
                <strong>
                  {demoTraining
                    ? localIngestStatus || "Encoding experience"
                    : busy
                      ? activeJob?.label ?? "Encoding experience"
                      : activeJob?.label ?? "No jobs yet"}
                </strong>
              </span>
              {busy && activeJob ? (
                <button className="icon-button" onClick={() => void cancelActiveJob()} aria-label="Cancel active job">
                  <Icon name="close" size={15} />
                </button>
              ) : null}
            </div>
            <div className="training-progress">
              <span>
                <strong>{progress}%</strong>
                <em>
                  {demoTraining
                    ? "processing"
                    : busy
                      ? activeJob?.state ?? "processing"
                      : activeJob?.state ?? "idle"}
                </em>
              </span>
              <i>
                <b style={{ width: `${progress}%` }} />
              </i>
            </div>
            <div className="training-metrics">
              <span>
                <small>Manifest coverage</small>
                <strong>
                  {activeCoverage
                    ? `${activeCoverage.processedRecords + activeCoverage.rejectedRecords} / ${activeCoverage.discoveredRecords}`
                    : activeJob && (activeJob.kind === "ingestion" || activeJob.kind === "crawl")
                      ? `${progress}%`
                      : "—"}
                </strong>
                <em>
                  {activeCoverage
                    ? `${activeCoverage.processedRecords} encoded · ${activeCoverage.rejectedRecords} rejected · ${activeCoverage.processedFiles + activeCoverage.rejectedFiles}/${activeCoverage.discoveredFiles} file visits${activeCoverage.complete ? " · complete" : " · incomplete/resumable"}`
                    : activeJob && (activeJob.kind === "ingestion" || activeJob.kind === "crawl")
                      ? "committed traversal in progress"
                      : "no manifest run"}
                </em>
              </span>
              <span>
                <small>Assembly deltas</small>
                <strong>{brain.trainingSources.reduce((sum, source) => sum + source.learnedIdeas, 0) || "—"}</strong>
                <em>{brain.trainingSources.length} encoded sources</em>
              </span>
              <span>
                <small>Synaptic Δ</small>
                <strong>{brain.trainingSources.reduce((sum, source) => sum + source.learnedSynapses, 0) || "—"}</strong>
                <em>recorded changes</em>
              </span>
            </div>
            <div className="training-log">
              <Icon name="terminal" size={15} />
              {demoTraining
                ? localIngestStatus
                : activeJob?.error ?? activeJob?.label ?? "No runtime log entries yet"}
              <span>{activeJob ? relativeTime(activeJob.updatedAt) : ""}</span>
            </div>
            <Button
              kind="primary"
              icon="play"
              disabled={busy || !brain.trainingSources.length}
              onClick={() => void startTraining()}
            >
              Train slow weights
            </Button>
          </div>
          <div className="surface memory-health">
            <div className="surface-title">
              <div>
                <h2>Memory health</h2>
                <p>Last checked just now</p>
              </div>
              <span
                className="health-score"
                style={{ "--health-score": `${healthScore}%` } as React.CSSProperties}
              >
                {healthScore || "—"}
              </span>
            </div>
            {healthRows.map(([label, value, state]) => (
              <div className="health-row" key={String(label)}>
                <span>{label}</span>
                <i>
                  <b style={{ width: `${value}%` }} />
                </i>
                <em>{state}</em>
              </div>
            ))}
            <Button
              icon="pulse"
              onClick={async () => {
                if (window.omni) {
                  onBrainChange(await window.omni.train.consolidate(brain.id));
                }
                onToast("Consolidation cycle queued.");
              }}
            >
              Consolidate now
            </Button>
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
  importance: number;
  uncertainty: number;
  exposures: number;
  members: string[];
  clusterCount?: number;
}

interface SubstrateMapEdge {
  id: string;
  sourceId: string;
  targetId: string;
  effectiveWeight: -1 | 0 | 1;
  latentWeight: number;
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

function BrainMapWorkspace({ brain }: { brain: BrainDocument }) {
  const [selected, setSelected] = useState("");
  const [filter, setFilter] = useState<"all" | "active" | "important">("all");
  const [query, setQuery] = useState("");
  const [zoom, setZoom] = useState(0.72);
  const [offset, setOffset] = useState(0);
  const [region, setRegion] = useState<string | undefined>();
  const [cursor, setCursor] = useState<string | undefined>();
  const [cursorHistory, setCursorHistory] = useState<Array<string | undefined>>([]);
  const [substratePage, setSubstratePage] = useState<SubstratePage | null>(null);
  const [substrateLoading, setSubstrateLoading] = useState(false);
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
          (filter === "important" && concept.importance >= 0.75);
        return matchesQuery && matchesFilter;
      }),
    [concepts, filter, query]
  );
  const clustered = zoom < 1;
  const pageSize = Math.max(48, Math.min(240, Math.round(72 * zoom)));
  const clampedOffset = Math.max(0, Math.min(offset, Math.max(0, filteredConcepts.length - pageSize)));

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
        if (active) setSubstratePage(null);
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
            (filter === "important" && cluster.maxActivation >= 0.75)
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
            (filter === "active" && assembly.confidence >= 0.5) ||
            (filter === "important" && assembly.importance >= 0.75)
          )
          .map((assembly) => ({
            id: assembly.id,
            label: assembly.label,
            region: assembly.region,
            activation: assembly.confidence,
            importance: assembly.importance,
            uncertainty: 1 - assembly.confidence,
            exposures: assembly.rehearsals,
            members: [assembly.id, ...assembly.neuronIds]
          }));
      }
      if (!clustered && substratePage.entity === "neurons") {
        return substratePage.neurons
          .filter((neuron) =>
            filter === "all" ||
            (filter === "active" && neuron.activation >= 0.5) ||
            (filter === "important" && neuron.importance >= 0.75)
          )
          .map((neuron) => ({
            id: neuron.id,
            label: neuron.label,
            region: neuron.region,
            activation: neuron.activation,
            importance: neuron.importance,
            uncertainty: neuron.uncertainty,
            exposures: neuron.exposures,
            members: [neuron.id]
          }));
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
          members: [concept.id]
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
  }, [clampedOffset, clustered, filter, filteredConcepts, pageSize, substratePage]);
  const realToView = useMemo(() => {
    const lookup = new Map<string, string>();
    mapNodes.forEach((node) => {
      lookup.set(node.id, node.id);
      node.members.forEach((member) => lookup.set(member, node.id));
    });
    return lookup;
  }, [mapNodes]);
  const graphEdges = useMemo<SubstrateMapEdge[]>(() => {
    if (substratePage) {
      const directEdges = substratePage.synapses.flatMap((synapse) => {
        const sourceId = realToView.get(synapse.sourceId);
        const targetId = realToView.get(synapse.targetId);
        if (!sourceId || !targetId || sourceId === targetId) return [];
        return [{
          id: synapse.id,
          sourceId,
          targetId,
          effectiveWeight: synapse.effectiveWeight,
          latentWeight: synapse.latentWeight,
          stability: synapse.stability,
          pathways: 1
        }];
      });
      if (directEdges.length) return directEdges;
      const regionNodes = new Map(
        mapNodes.flatMap((node) => node.region ? [[node.region, node.id] as const] : [])
      );
      return substratePage.clusters.flatMap((cluster) => {
        if (cluster.kind !== "pathway" || !cluster.sourceRegion || !cluster.targetRegion) return [];
        const sourceId = regionNodes.get(cluster.sourceRegion);
        const targetId = regionNodes.get(cluster.targetRegion);
        if (!sourceId || !targetId || sourceId === targetId) return [];
        const signed = cluster.effectiveWeights.positive - cluster.effectiveWeights.negative;
        const total = Math.max(1, cluster.effectiveWeights.negative + cluster.effectiveWeights.zero + cluster.effectiveWeights.positive);
        const latentWeight = signed / total;
        return [{
          id: cluster.id,
          sourceId,
          targetId,
          effectiveWeight: latentWeight > 0.15 ? 1 as const : latentWeight < -0.15 ? -1 as const : 0 as const,
          latentWeight,
          stability: cluster.activeCount / Math.max(1, cluster.count),
          pathways: cluster.count
        }];
      });
    }
    const aggregated = new Map<string, {
      sourceId: string;
      targetId: string;
      latent: number;
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
        latent: 0,
        stability: 0,
        pathways: 0
      };
      current.latent += synapse.latentWeight;
      current.stability += synapse.stability;
      current.pathways += 1;
      aggregated.set(key, current);
    });
    return [...aggregated.entries()].map(([id, edge]) => {
      const latentWeight = edge.latent / edge.pathways;
      return {
        id,
        sourceId: edge.sourceId,
        targetId: edge.targetId,
        effectiveWeight: latentWeight > 0.15 ? 1 : latentWeight < -0.15 ? -1 : 0,
        latentWeight,
        stability: edge.stability / edge.pathways,
        pathways: edge.pathways
      };
    });
  }, [brain.synapses, mapNodes, realToView, substratePage]);
  const positions = useMemo(() => {
    const values = new Map<string, { x: number; y: number }>();
    const total = Math.max(1, mapNodes.length);
    mapNodes.forEach((node, index) => {
      const angle = index * 2.399963229728653;
      const radial = Math.sqrt((index + 0.65) / total);
      values.set(node.id, {
        x: 500 + Math.cos(angle) * radial * 420,
        y: 325 + Math.sin(angle) * radial * 265
      });
    });
    return values;
  }, [mapNodes]);
  const selectedConcept =
    brain.concepts[selected] ??
    concepts.find((concept) => concept.id === mapNodes[0]?.members[0]);
  const selectedMapNode = mapNodes.find((node) => node.id === selected || node.members.includes(selected));
  const connectedSynapses = useMemo(
    () =>
      selectedConcept
        ? Object.values(brain.synapses)
            .filter((synapse) => synapse.sourceId === selectedConcept.id || synapse.targetId === selectedConcept.id)
            .sort((a, b) => Math.abs(b.latentWeight) - Math.abs(a.latentWeight))
        : [],
    [brain.synapses, selectedConcept]
  );
  const selectedStability = connectedSynapses.length
    ? connectedSynapses.reduce((sum, synapse) => sum + synapse.stability, 0) / connectedSynapses.length
    : 0;
  const recentSynapse = [...connectedSynapses].sort((a, b) => b.lastUpdatedAt.localeCompare(a.lastUpdatedAt))[0];
  const matchedCount = substratePage?.matched ?? filteredConcepts.length;
  const visibleStart = matchedCount ? (substratePage ? cursorHistory.length * pageSize + 1 : clampedOffset + 1) : 0;
  const visibleEnd = clustered
    ? matchedCount
    : Math.min(matchedCount, visibleStart + mapNodes.length - 1);

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
          <label className="search-field search-field--small">
            <Icon name="search" size={15} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find an assembly" />
          </label>
          <div className="segmented">
            {(["all", "active", "important"] as const).map((item) => (
              <button key={item} className={filter === item ? "is-active" : ""} onClick={() => setFilter(item)}>
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
                ? "Querying substrate…"
                : clustered
                  ? `${mapNodes.length} clusters · ${compactNumber(substratePage?.totals.neurons ?? filteredConcepts.length)} neurons`
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
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke={edge.effectiveWeight < 0 ? "#f183b7" : edge.effectiveWeight === 0 ? "#8c8998" : "#9b87ff"}
                    strokeOpacity={0.1 + Math.min(0.55, Math.abs(edge.latentWeight) * 0.42)}
                    strokeWidth={0.5 + Math.min(3, edge.stability * 1.4 + Math.log2(edge.pathways + 1) * 0.25)}
                    strokeDasharray={edge.effectiveWeight < 0 ? "5 5" : edge.effectiveWeight === 0 ? "2 7" : undefined}
                  />
                );
              })}
              {mapNodes.map((node, index) => {
                const position = positions.get(node.id)!;
                const isSelected = selected === node.id || node.members.includes(selectedConcept?.id ?? "");
                const representedCount = node.clusterCount ?? node.members.length;
                const clusterBoost = clustered ? Math.min(15, Math.log2(representedCount + 1) * 4) : 0;
                const radius = Math.max(7, 10 + node.importance * 10 + clusterBoost - Math.max(0, mapNodes.length - 90) * 0.025);
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
                    <circle
                      r={radius + (isSelected ? 9 : 3)}
                      fill={isSelected ? "#8f74ff" : index % 5 === 0 ? "#55d8cf" : "#8e76ef"}
                      opacity={isSelected ? ".16" : ".065"}
                    />
                    <circle
                      r={radius}
                      fill={isSelected ? "#a790ff" : index % 5 === 0 ? "#5bd8d0" : "#8069d5"}
                      opacity={0.45 + node.activation * 0.45}
                      stroke={isSelected ? "#ede8ff" : "#b7a9ff"}
                      strokeOpacity={isSelected ? ".9" : ".32"}
                      strokeWidth={isSelected ? "2.2" : "1"}
                      filter={isSelected ? "url(#scalableMapGlow)" : undefined}
                    />
                    {clustered && representedCount > 1 ? (
                      <text y="4" textAnchor="middle" fill="#f5f2ff" fontSize="11" fontWeight="700">
                        {compactNumber(representedCount)}
                      </text>
                    ) : <circle r={Math.max(2.5, radius * 0.22)} fill="#f2eeff" opacity=".9" />}
                    {(clustered || mapNodes.length <= 90 || isSelected) ? (
                      <text y={radius + 17} textAnchor="middle" fill="#d9d4eb" fontSize={clustered ? "11" : "10.5"} fontWeight={isSelected ? "650" : "500"}>
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
              {!clustered ? (
                <>
                  <i />
                  <button
                    aria-label="Previous assemblies"
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
                    aria-label="Next assemblies"
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
                </>
              ) : null}
            </div>
          </div>
          <div className="graph-status">
            <span><i /> {compactNumber(brain.counters.plasticityEvents)} plasticity events</span>
            <span>{compactNumber(substratePage?.totals.synapses ?? Object.keys(brain.synapses).length)} substrate synapses</span>
            <span>{substratePage ? "Paged from authoritative substrate" : "Local compatibility view"} · no display ceiling</span>
          </div>
        </section>
        <aside className="map-inspector">
          {selectedConcept ? (
            <>
              <div className="map-inspector__head">
                <span className="map-inspector__node"><i /></span>
                <span><small>SELECTED ASSEMBLY</small><h2>{selectedConcept.label}</h2></span>
              </div>
              <div className="activation-score">
                <div style={{ "--score": `${selectedConcept.activation * 360}deg` } as React.CSSProperties}>
                  <span>{Math.round(selectedConcept.activation * 100)}</span>
                </div>
                <span><strong>Current activation</strong><small>{selectedConcept.activation > 0.8 ? "Highly active" : "Available"}</small></span>
              </div>
              <dl className="inspector-stats">
                <div><dt>Importance</dt><dd>{Math.round(selectedConcept.importance * 100)}%</dd></div>
                <div><dt>Stability</dt><dd>{connectedSynapses.length ? `${Math.round(selectedStability * 100)}%` : "—"}</dd></div>
                <div><dt>Uncertainty</dt><dd>{Math.round(selectedConcept.uncertainty * 100)}%</dd></div>
                <div><dt>Exposures</dt><dd>{compactNumber(selectedConcept.exposures)}</dd></div>
              </dl>
              <div className="panel-section">
                <div className="panel-section__head"><span>Strongest local pathways</span><em>{compactNumber(connectedSynapses.length)} total</em></div>
                <div className="pathway-list">
                  {connectedSynapses.slice(0, 8).map((synapse) => {
                    const otherId = synapse.sourceId === selectedConcept.id ? synapse.targetId : synapse.sourceId;
                    return (
                      <div key={synapse.id}>
                        <span><i /> {brain.concepts[otherId]?.label ?? otherId}</span>
                        <em>{synapse.effectiveWeight > 0 ? "+1" : synapse.effectiveWeight < 0 ? "−1" : "0"}</em>
                      </div>
                    );
                  })}
                  {!connectedSynapses.length ? <span className="pathway-empty">No synapses connect this assembly yet.</span> : null}
                </div>
              </div>
              <div className="panel-section">
                <div className="panel-section__head"><span>Recent change</span><em>STDP</em></div>
                {recentSynapse ? (
                  <div className="change-note">
                    <Icon name="pulse" size={16} />
                    <span>
                      Latest connected synapse changed with <strong>{brain.concepts[recentSynapse.sourceId === selectedConcept.id ? recentSynapse.targetId : recentSynapse.sourceId]?.label ?? "another assembly"}</strong>.
                      <small>{relativeTime(recentSynapse.lastUpdatedAt)} · ternary {recentSynapse.effectiveWeight > 0 ? "+1" : recentSynapse.effectiveWeight < 0 ? "−1" : "0"}</small>
                    </span>
                  </div>
                ) : <span className="pathway-empty">No plasticity event has been recorded for this assembly.</span>}
              </div>
            </>
          ) : selectedMapNode ? (
            <>
              <div className="map-inspector__head">
                <span className="map-inspector__node"><i /></span>
                <span>
                  <small>{selectedMapNode.clusterCount ? "SELECTED CLUSTER" : substratePage?.entity === "neurons" ? "SELECTED NEURON" : "SELECTED ASSEMBLY"}</small>
                  <h2>{selectedMapNode.label}</h2>
                </span>
              </div>
              <div className="activation-score">
                <div style={{ "--score": `${selectedMapNode.activation * 360}deg` } as React.CSSProperties}>
                  <span>{Math.round(selectedMapNode.activation * 100)}</span>
                </div>
                <span>
                  <strong>Measured activation</strong>
                  <small>{selectedMapNode.region ?? "Unified substrate"}</small>
                </span>
              </div>
              <dl className="inspector-stats">
                <div><dt>Importance</dt><dd>{Math.round(selectedMapNode.importance * 100)}%</dd></div>
                <div><dt>Uncertainty</dt><dd>{Math.round(selectedMapNode.uncertainty * 100)}%</dd></div>
                <div><dt>{selectedMapNode.clusterCount ? "Neurons" : "Exposures"}</dt><dd>{compactNumber(selectedMapNode.clusterCount ?? selectedMapNode.exposures)}</dd></div>
                <div><dt>Pathways on page</dt><dd>{graphEdges.filter((edge) => edge.sourceId === selectedMapNode.id || edge.targetId === selectedMapNode.id).length}</dd></div>
              </dl>
              {selectedMapNode.clusterCount ? (
                <Button kind="primary" icon="expand" onClick={() => openNode(selectedMapNode)}>
                  Open this region
                </Button>
              ) : null}
              <div className="panel-section">
                <div className="panel-section__head"><span>Ternary pathways</span><em>current page</em></div>
                <div className="pathway-list">
                  {graphEdges
                    .filter((edge) => edge.sourceId === selectedMapNode.id || edge.targetId === selectedMapNode.id)
                    .map((edge) => {
                      const otherId = edge.sourceId === selectedMapNode.id ? edge.targetId : edge.sourceId;
                      return (
                        <div key={edge.id}>
                          <span><i /> {mapNodes.find((node) => node.id === otherId)?.label ?? otherId}</span>
                          <em>{edge.effectiveWeight > 0 ? "+1" : edge.effectiveWeight < 0 ? "−1" : "0"}</em>
                        </div>
                      );
                    })}
                </div>
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
    return (
      <div className="content-page trace-page">
        <div className="content-page__title content-page__title--compact">
          <div>
            <span className="eyebrow-text">VERIFIABLE ACTIVITY</span>
            <h1>Trace & journal</h1>
            <p>Operational traces will appear after this brain performs inference or learning.</p>
          </div>
        </div>
        <div className="journal-empty surface">
          <span><Icon name="trace" size={24} /></span>
          <h2>No operational traces yet</h2>
          <p>Start a conversation or encode an experience. Real activation and mutation stages will be recorded here.</p>
        </div>
      </div>
    );
  }

  const selected = traces.find((traceItem) => traceItem.id === selectedId) ?? traces.at(-1)!;
  const plasticitySteps = selected.steps.filter((stepItem) =>
    /plastic|synap|hebb|stdp/i.test(`${stepItem.stage} ${stepItem.detail}`)
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
                  <small>{relativeTime(item.createdAt)} · {item.steps.length} stages</small>
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
                      <strong>{index === 1 ? "Consolidation cycle" : index === 2 ? "Document encoded" : "Autonomous reflection"}</strong>
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
              <div><span>Ideas activated</span><strong>{selected.activatedConcepts.length}</strong><small>{selected.recalledIdeas.length} recalled</small></div>
              <div><span>Plasticity stages</span><strong>{plasticitySteps.length}</strong><small>{plasticitySteps[0]?.value ?? "none recorded"}</small></div>
              <div><span>Integration</span><strong>{liquidStep?.value ?? "—"}</strong><small>{liquidStep ? "liquid state" : "not recorded"}</small></div>
            </div>
            <div className="trace-timeline">
              {selected.steps.map((traceStep, index) => (
                <div className="trace-step" key={`${traceStep.stage}-${index}`}>
                  <span className="trace-step__index">{String(index + 1).padStart(2, "0")}</span>
                  <span className="trace-step__line"><i /></span>
                  <div>
                    <span className="trace-step__head">
                      <strong>{traceStep.stage}</strong>
                      {traceStep.value ? <em>{traceStep.value}</em> : null}
                    </span>
                    <p>{traceStep.detail}</p>
                    {index === 1 ? (
                      <div className="trace-concepts">
                        {selected.activatedConcepts.map((concept) => (
                          <span key={concept.id}>{concept.label}<i>{Math.round(concept.activation * 100)}</i></span>
                        ))}
                      </div>
                    ) : null}
                    {index === 3 ? (
                      <div className="branch-row">
                        {Array.from({ length: selected.branches }).map((_, branch) => (
                          <span key={branch} className={branch + 1 === selected.selectedBranch ? "is-selected" : ""}>
                            Branch {branch + 1}<i>{branch + 1 === selected.selectedBranch ? "selected" : "released"}</i>
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
            </div>
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
  const entries = brain.journal ?? [];
  const [selectedId, setSelectedId] = useState(entries.at(-1)?.id ?? entries[0]?.id ?? "");
  const selected = entries.find((entry) => entry.id === selectedId) ?? entries.at(-1);

  if (!selected) {
    return (
      <div className="journal-empty surface">
        <span><Icon name="file" size={24} /></span>
        <h2>No journal entries yet</h2>
        <p>
          Journal entries appear only after the brain records a real learning, consolidation, tool, fork, or system event.
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
            <small>{selected.kind.toUpperCase()} EVENT · {new Date(selected.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</small>
            <h2>{selected.summary}</h2>
          </span>
          <em>{selected.kind}</em>
        </div>
        <div className="journal-prose">
          {(selected.detail ?? selected.summary).split(/\n{2,}/).map((paragraph, index) => <p key={index}>{paragraph}</p>)}
        </div>
        <div className="journal-entry__footer">
          <span><Icon name="brain" size={14} /> Recorded by {brain.name}</span>
          <span><Icon name="pulse" size={14} /> Verifiable {selected.kind} event</span>
          <span>{relativeTime(selected.createdAt)}</span>
        </div>
      </section>
      <aside className="surface journal-sidebar">
        <h3>Recorded entries</h3>
        <div className="journal-entry-list">
          {[...entries].reverse().map((entry) => (
            <button key={entry.id} className={entry.id === selected.id ? "is-active" : ""} onClick={() => setSelectedId(entry.id)}>
              <span>{entry.summary}</span>
              <small>{entry.kind} · {relativeTime(entry.createdAt)}</small>
            </button>
          ))}
        </div>
        <dl className="journal-facts">
          <div><dt>Recorded events</dt><dd>{entries.length}</dd></div>
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
            guaranteed transcript of hidden reasoning.
          </p>
        </div>
      </aside>
    </div>
  );
}

function ImaginationWorkspace({ brain, onToast }: { brain: BrainDocument; onToast: (message: string) => void }) {
  const [mode, setMode] = useState<ModalityId>("image");
  const [prompt, setPrompt] = useState("A memory palace growing new luminous pathways after rain");
  const [generating, setGenerating] = useState(false);
  const [variation, setVariation] = useState(0);
  const [job, setJob] = useState<RuntimeJob | null>(null);
  const [installedPacks, setInstalledPacks] = useState<InstalledModalityPack[]>([]);
  const [packUrl, setPackUrl] = useState("");
  const [packBusy, setPackBusy] = useState(false);
  const settledJobs = useRef(new Set<string>());
  const demo = !window.omni;

  const reloadPacks = async () => {
    if (!window.omni) return;
    setInstalledPacks(await window.omni.catalog.listModalityPacks(brain.id));
  };

  useEffect(() => {
    void reloadPacks();
  }, [brain.id]);

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
    try {
      if (window.omni) {
        const request = {
          brainId: brain.id,
          modality: mode,
          prompt,
          conceptIds: brain.workingMemory.map((item) => item.conceptId)
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
  const dataUrl = typeof output?.dataUrl === "string" ? output.dataUrl : null;
  const embedding = Array.isArray(output?.embedding) ? output.embedding : null;
  const downloadOutput = () => {
    if (!dataUrl) return;
    const mimeType = typeof output?.mimeType === "string" ? output.mimeType : "application/octet-stream";
    const extension = mimeType.includes("mp4")
      ? "mp4"
      : mimeType.includes("png")
        ? "png"
        : mimeType.includes("wav")
          ? "wav"
          : "bin";
    const anchor = document.createElement("a");
    anchor.href = dataUrl;
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
          <i /> 4 built-in baselines · {installedPacks.length} installed
        </span>
      </div>
      <div className="imagine-layout">
        <section className="surface imagination-canvas">
          <div className={cx("generated-art", generating && "is-generating", `generated-art--${variation % 3}`)}>
            {generating ? (
              <span className="generation-state">
                <Icon name="sparkles" size={22} />
                {mode === "vision" ? "Encoding selected image…" : "Imagining through the active neural workspace…"}
              </span>
            ) : dataUrl && mode === "audio" ? (
              <div className="generated-media generated-media--audio">
                <Icon name="wave" size={42} />
                <audio src={dataUrl} controls />
                <span>Generated by the local audio pack</span>
              </div>
            ) : dataUrl && mode === "video" && output?.mimeType === "video/mp4" ? (
              <video
                className="generated-media-video"
                src={dataUrl}
                controls
                autoPlay
                loop
                muted
                aria-label="Locally generated video artifact"
              />
            ) : dataUrl ? (
              <img className="generated-media-image" src={dataUrl} alt={`Locally generated ${mode} artifact`} />
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
            {output?.randomlyInitialized === true ? (
              <span className="baseline-quality">Untrained baseline · experimental quality</span>
            ) : null}
          </div>
          <div className="canvas-footer">
            <span>
              <Icon name="brain" size={15} /> Seeded from 6 active ideas
            </span>
            <div>
              <button className="icon-button" aria-label="Create variation" onClick={() => void generate()}>
                <Icon name="sparkles" size={16} />
              </button>
              <button className="icon-button" aria-label="Download output" disabled={!dataUrl} onClick={downloadOutput}>
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
              <button key={id} className={mode === id ? "is-active" : ""} onClick={() => setMode(id)}>
                <Icon name={icon} size={16} /> {label}
              </button>
            ))}
          </div>
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
              <Icon name="brain" size={13} /> Leave blank to use only the active neural workspace. Organic imagination enters this same pipeline from the learned action head.
            </small>
          </label>
          <div className="builder-note">
            <Icon name="pulse" size={15} />
            <span>Resolution, duration, variation, and creative distance emerge from neural state and the hardware profile; no behavior slider overrides them.</span>
          </div>
          <Button
            kind="primary"
            icon={mode === "vision" ? "upload" : "sparkles"}
            disabled={generating}
            onClick={() => void generate()}
          >
            {generating ? "Working…" : mode === "vision" ? "Choose image to understand" : `Imagine ${mode}`}
          </Button>
          <p className="imagination-footnote">
            {demo ? "Design preview only; no modality engine is connected." : `Uses the local ${mode} pack. Nothing is sent to a hosted model.`}
          </p>
        </aside>
      </div>
      <div className="generation-strip">
        <div className="surface-title">
          <div><h2>Recent imagination</h2><p>Outputs may be fed back as experience.</p></div>
          <Button>Open gallery</Button>
        </div>
        <div className="generation-thumbs">
          {demo ? (
            [0, 1, 2, 3].map((item) => (
              <div key={item} className={`generation-thumb generation-thumb--${item}`}>
                <span />
                <em>{item === 0 ? "Demo · memory palace" : item === 1 ? "Demo · liquid mechanism" : item === 2 ? "Demo · rain language" : "Demo · unsaid idea"}</em>
              </div>
            ))
          ) : dataUrl || embedding ? (
            <div className="generation-thumb generation-thumb--0">
              <span />
              <em>{mode} · {job?.state} · {relativeTime(job?.updatedAt ?? new Date().toISOString())}</em>
            </div>
          ) : (
            <div className="generation-history-empty">No completed modality artifacts in this session.</div>
          )}
        </div>
      </div>
    </div>
  );
}

const protocolMeta: Record<string, { label: string; icon: IconName; action: string; args: Record<string, unknown> }> = {
  "windows.files": { label: "Windows files", icon: "file", action: "list", args: { path: "C:\\Users\\Public" } },
  "windows.powershell": { label: "PowerShell", icon: "terminal", action: "run", args: { command: "Get-Date", cwd: "C:\\Users\\Public" } },
  "code.execute": { label: "Code runner", icon: "code", action: "run", args: { language: "python", entryPath: "C:\\path\\to\\script.py", arguments: [] } },
  "web.fetch": { label: "Web fetch", icon: "download", action: "fetch", args: { url: "https://example.com", maxBytes: 1000000 } },
  "web.search": { label: "Web search", icon: "search", action: "search", args: { query: "neuromorphic computing", limit: 5 } },
  "browser.automation": { label: "Browser task", icon: "expand", action: "task", args: { url: "https://example.com" } },
  "modality.imagine": { label: "Imagination", icon: "sparkles", action: "generate", args: { modality: "image", conceptIds: [] } },
  "agent.fork": { label: "Subagent fork", icon: "agents", action: "start", args: { objective: "Explore this question independently." } },
  "source.self-modify": { label: "Source evolution", icon: "code", action: "propose", args: {} }
};

function ToolsWorkspace({
  brain,
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [permissions, setPermissions] = useState<ToolPermissionRecord[]>(
    brain.toolPermissions ??
      Object.entries(protocolMeta).map(([toolId, meta]) => ({
        toolId,
        label: meta.label,
        level: "off" as const,
        updatedAt: brain.createdAt
      }))
  );
  const [selectedTool, setSelectedTool] = useState("web.fetch");
  const selectedMeta = protocolMeta[selectedTool] ?? protocolMeta["web.fetch"]!;
  const [action, setAction] = useState(selectedMeta.action);
  const [argumentsText, setArgumentsText] = useState(JSON.stringify(selectedMeta.args, null, 2));
  const [resultText, setResultText] = useState("");
  const [approvalToken, setApprovalToken] = useState("");
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (!window.omni) return;
    void window.omni.tool.listPermissions(brain.id).then(setPermissions);
  }, [brain.id]);

  const chooseTool = (toolId: string) => {
    const meta = protocolMeta[toolId] ?? { label: toolId, icon: "terminal" as const, action: "", args: {} };
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
      setPermissions(next);
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
        setResultText("Approval required. Inspect the action above, then click Approve & run.");
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
  const [catalogUrl, setCatalogUrl] = useState("");

  const reloadBranches = async () => {
    if (!window.omni) {
      const demoCode = makeDemoBrain("demo-code-fork", `${brain.name} · code`, brain.config);
      const demoDream = makeDemoBrain("demo-dream-fork", `${brain.name} · dream`, brain.config);
      setBranches(
        [demoCode, demoDream].map((item, index) => ({
          ...item,
          lineage: {
            rootId: brain.lineage.rootId,
            parentId: brain.id,
            generation: brain.lineage.generation + 1 + index
          }
        }))
      );
      return;
    }
    const summaries = await window.omni.brain.list();
    const documents = await Promise.all(
      summaries.filter((summary) => summary.id !== brain.id).map((summary) => window.omni!.brain.get(summary.id))
    );
    setBranches(documents.filter((document) => document.lineage.rootId === brain.lineage.rootId));
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
    if (!objective.trim()) return;
    if (!window.omni) {
      onToast("Demo subagent prepared visually; no fork or agent process ran.");
      return;
    }
    try {
      const result = await window.omni.tool.execute({
        brainId: brain.id,
        toolId: "agent.fork",
        action: "start",
        arguments: { objective },
        approvalToken: agentApproval || undefined
      });
      if (result.state === "approval-required" && result.approvalToken) {
        setAgentApproval(result.approvalToken);
        onToast("Subagent fork requires approval. Review the objective and approve once more.");
      } else if (result.state === "complete") {
        setAgentApproval("");
        onToast("Subagent fork created from an isolated overlay.");
        await reloadBranches();
      } else {
        onToast(result.error ?? "Subagent tool failed.");
      }
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Subagent tool failed.");
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
        onToast(`${label} exported to ${path}`);
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
            <div><h2>Living lineage</h2><p>Generation {brain.lineage.generation} · origin remains immutable</p></div>
            <span className="lineage-pill"><Icon name="check" size={13} /> Origin verified</span>
          </div>
          <div className="lineage-tree">
            <div className="lineage-origin">
              <span className="lineage-node lineage-node--origin"><BrandMark size={33} /></span>
              <span>
                <small>ORIGIN · G0</small>
                <strong>{brain.name} / initial</strong>
                <em>{brain.originChecksum ? `Verified · ${brain.originChecksum.slice(0, 10)}…` : "Immutable recovery point"}</em>
              </span>
            </div>
            <span className="lineage-stem" />
            <div className="lineage-generation">
              <div className="lineage-branch lineage-branch--active">
                <span className="lineage-node"><Icon name="brain" size={18} /></span>
                <span><small>PRIMARY · G{brain.lineage.generation}</small><strong>{brain.name}</strong><em>Current · learning</em></span>
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
              <div><h2>Subagent mode</h2><p>Think in parallel, merge deliberately.</p></div>
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
              <textarea value={objective} onChange={(event) => setObjective(event.target.value)} rows={3} />
            </label>
            <Button kind="primary" icon={agentApproval ? "check" : "agents"} onClick={() => void startSubagent()}>
              {agentApproval ? "Approve isolated fork" : "Start a subagent session"}
            </Button>
          </div>
          <div className="surface export-card">
            <div className="export-card__icon"><Icon name="archive" size={21} /></div>
            <span><strong>Portable identity</strong><small>Safe tensors · ideas · lineage · journal</small></span>
            <button className="origin-export" onClick={() => void exportBrain("origin")}>Export origin</button>
          </div>
          <div className="surface github-card github-card--install">
            <Icon name="download" size={18} />
            <span>
              <strong>Install a premade Omni mind</strong>
              <small>Verified .omni bundle URL; repository scripts never run</small>
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
    try {
      return loadAppearancePreferences(window.localStorage);
    } catch {
      return { ...DEFAULT_APPEARANCE };
    }
  });
  const [systemUsesDark, setSystemUsesDark] = useState(() =>
    typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-color-scheme: dark)").matches
      : true
  );
  const [page, setPage] = useState<AppPage>("library");
  const [summaries, setSummaries] = useState<BrainSummary[]>(demo ? demoSummaries : []);
  const [activeBrain, setActiveBrain] = useState<BrainDocument | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("chat");
  const [loading, setLoading] = useState(!demo);
  const [toast, setToast] = useState("");
  const resolvedColorScheme = resolveColorScheme(appearance.mode, systemUsesDark);

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

  const showToast = (message: string) => {
    setToast(message);
    window.setTimeout(() => setToast((current) => (current === message ? "" : current)), 3_600);
  };

  useEffect(() => {
    if (!window.omni) return;
    let active = true;
    void window.omni.brain
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
  }, []);

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
    let hardwareTier: HardwareTier | undefined;
    if (extras.hardware === "auto" && window.omni) {
      hardwareTier = (await window.omni.catalog.hardwareProfile()).recommendedTier;
    } else if (extras.hardware !== "auto") {
      hardwareTier = extras.hardware;
    }
    let document = window.omni
      ? await window.omni.brain.create({
          config,
          origin: extras.origin,
          starterUrl: extras.origin === "starter" ? extras.starterUrl : undefined,
          hardwareTier,
          modalities: (Object.entries(extras.modalities) as Array<[ModalityId, boolean]>)
            .filter(([, enabled]) => enabled)
            .map(([modality]) => modality),
          initialToolPermissions: Object.entries(extras.tools).flatMap(([toolId, level]) => {
            const protocolIds: Record<string, string[]> = {
              files: ["windows.files"],
              powershell: ["windows.powershell"],
              code: ["code.execute"],
              web: ["web.fetch", "web.search"],
              browser: ["browser.automation"],
              imagination: ["modality.imagine"],
              agents: ["agent.fork"],
              evolution: ["source.self-modify"]
            };
            return (protocolIds[toolId] ?? [toolId]).map((protocolId) => ({ toolId: protocolId, level }));
          })
        })
      : makeDemoBrain(`demo-${Date.now()}`, config.name, config);
    let initialSources = 0;
    let initialLearningQueued = false;
    if (window.omni && extras.initialTraining) {
      for (const resource of extras.initialResources) {
        if (resource.kind === "web") {
          await window.omni.data.crawlWeb({
            brainId: document.id,
            url: resource.url,
            policy: "pretrain",
            sameOrigin: true,
            followExternalLinks: false,
            respectRobots: true,
            quarantine: false
          });
        } else {
          await window.omni.data.startBuildResource({
            brainId: document.id,
            selectionId: resource.id,
            policy: "pretrain"
          });
          initialSources += resource.itemCount;
        }
        initialLearningQueued = true;
      }
      document = await window.omni.brain.get(document.id);
    }
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
        generation: document.lineage.generation
      },
      ...current
    ]);
    setWorkspaceView("chat");
    setPage("workspace");
    showToast(
      initialLearningQueued
        ? `${document.name} is ready; initial learning${
            initialSources > 0
              ? ` from ${initialSources} selected local item${initialSources === 1 ? "" : "s"}`
              : ""
          } is running as resumable work.`
        : `${document.name} has an immutable origin and is ready to learn.`
    );
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
              synapses: Object.keys(document.synapses).length
            }
          : summary
      )
    );
  };

  const duplicateBrain = async (source: Pick<BrainSummary, "id" | "name">) => {
    setLoading(true);
    try {
      const duplicate = window.omni
        ? await window.omni.brain.duplicate(source.id, `${source.name} copy`)
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
        generation: duplicate.lineage.generation
      };
      setSummaries((current) => [
        nextSummary,
        ...current.filter((summary) => summary.id !== nextSummary.id)
      ]);
      showToast(`${duplicate.name} created with copy-on-write neural storage.`);
    } catch (error) {
      showToast(error instanceof Error ? error.message : "This brain could not be duplicated.");
    } finally {
      setLoading(false);
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
        resolvedColorScheme={resolvedColorScheme}
        onAppearanceChange={setAppearance}
      />
      {page === "library" ? (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onDuplicate={(summary) => void duplicateBrain(summary)}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          demo={demo}
        />
      ) : page === "build" ? (
        <SimpleBuildWizard onCancel={() => setPage("library")} onCreate={createBrain} />
      ) : activeBrain ? (
        <WorkspaceShell
          brain={activeBrain}
          view={workspaceView}
          onView={setWorkspaceView}
          onLibrary={() => setPage("library")}
          onDuplicate={() => void duplicateBrain(activeBrain)}
          onBrainChange={updateActiveBrain}
          onToast={showToast}
        />
      ) : (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onDuplicate={(summary) => void duplicateBrain(summary)}
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
      {toast ? (
        <div className="toast" role="status">
          <span><Icon name="check" size={15} /></span>
          {toast}
          <button onClick={() => setToast("")} aria-label="Dismiss"><Icon name="close" size={14} /></button>
        </div>
      ) : null}
    </div>
  );
}
