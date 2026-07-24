import {
  Fragment,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent
} from "react";
import {
  createPresetConfig,
  DEFAULT_CONFIG,
  type ArchitecturePreset,
  type BrainConfig,
  type BrainDocument,
  type BrainExportMode,
  type BrainSummary,
  type BuildRecipe,
  type CatalogEntry,
  type ChatMessage,
  type HardwareTier,
  type HardwareProfile as SystemHardwareProfile,
  type RuntimeJob,
  type InstalledModalityPack,
  type ToolExecutionResult,
  type ToolInvocation,
  type ToolPermissionRecord,
  type AgentMergePreview
} from "@shared/types";
import { demoSummaries, makeDemoBrain, makeDemoChat } from "./demo";
import { Icon, type IconName } from "./icons";

type AppPage = "library" | "build" | "workspace";
type WorkspaceView = "chat" | "data" | "map" | "trace" | "imagine" | "tools" | "agents";
type RecipeId = ArchitecturePreset | "custom";
type HardwareChoice = "auto" | "micro" | "personal" | "gpu" | "workstation";
type MemoryRecipe = "human" | "recall" | "synapses";
type PermissionLevel = "off" | "ask" | "auto" | "full";
type ModalityId = "vision" | "image" | "audio" | "video";
type OriginKind = "blank" | "starter";

interface BuilderExtras {
  recipe: RecipeId;
  hardware: HardwareChoice;
  memoryRecipe: MemoryRecipe;
  origin: OriginKind;
  starterUrl: string;
  initialTraining: boolean;
  modalities: Record<ModalityId, boolean>;
  tools: Record<string, PermissionLevel>;
}

const recipeMeta: Array<{
  id: RecipeId;
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
  { id: "agents", label: "Forks & agents", icon: "fork" }
];

const toolRows = [
  ["files", "Windows files", "file"],
  ["powershell", "PowerShell", "terminal"],
  ["code", "Code workspace", "code"],
  ["web", "Web access", "search"],
  ["browser", "Browser control", "expand"],
  ["agents", "Subagents", "agents"]
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

function Titlebar({
  page,
  brain,
  demo,
  onLibrary
}: {
  page: AppPage;
  brain: BrainDocument | null;
  demo: boolean;
  onLibrary: () => void;
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
  onBuild,
  onImport,
  demo
}: {
  summaries: BrainSummary[];
  loading: boolean;
  onOpen: (summary: BrainSummary) => void;
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
              <span>Start blank or from a recipe</span>
            </button>
          </div>
        ) : (
          <div className="library-empty">
            <BrandMark size={58} />
            <h3>{query ? "No matching minds" : "This library is waiting for its first mind"}</h3>
            <p>{query ? "Try a different name." : "Choose a cognitive recipe and grow something new."}</p>
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
              <span>Windows 11 · Local runtime · Hardware scaling enabled</span>
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

function RangeField({
  label,
  detail,
  value,
  min,
  max,
  step,
  display,
  onChange
}: {
  label: string;
  detail: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <label className="range-field">
      <span className="range-field__head">
        <span>
          <strong>{label}</strong>
          <small>{detail}</small>
        </span>
        <output>{display}</output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        style={{ "--range-progress": `${percent}%` } as React.CSSProperties}
      />
    </label>
  );
}

const buildSteps = [
  ["Architecture", "Choose a cognitive recipe"],
  ["Compute", "Shape it for this machine"],
  ["Memory", "Decide what experience becomes"],
  ["Plasticity", "Tune change and growth"],
  ["Senses & tools", "Give it ways to perceive and act"],
  ["Review", "Create the immutable origin"]
] as const;

function BuildWizard({
  onCancel,
  onCreate
}: {
  onCancel: () => void;
  onCreate: (config: BrainConfig, extras: BuilderExtras) => Promise<void>;
}) {
  const [step, setStep] = useState(0);
  const [name, setName] = useState("Nova");
  const [config, setConfig] = useState<BrainConfig>(createPresetConfig("whole-brain", "Nova"));
  const [extras, setExtras] = useState<BuilderExtras>({
    recipe: "whole-brain",
    hardware: "auto",
    memoryRecipe: "human",
    origin: "blank",
    starterUrl: "",
    initialTraining: false,
    modalities: { vision: true, image: true, audio: false, video: false },
    tools: {
      files: "ask",
      powershell: "ask",
      code: "ask",
      web: "ask",
      browser: "off",
      agents: "ask"
    }
  });
  const [building, setBuilding] = useState(false);
  const [detectedHardware, setDetectedHardware] = useState<SystemHardwareProfile | null>(null);
  const [catalogRecipes, setCatalogRecipes] = useState<CatalogEntry[]>([]);
  const [recipeUrl, setRecipeUrl] = useState("");
  const [recipeLoading, setRecipeLoading] = useState(false);
  const [recipeStatus, setRecipeStatus] = useState("");

  useEffect(() => {
    let active = true;
    if (window.omni) {
      void window.omni.catalog.hardwareProfile().then((profile) => {
        if (active) setDetectedHardware(profile);
      });
      void window.omni.catalog.list().then((entries) => {
        if (active) setCatalogRecipes(entries.filter((entry) => entry.kind === "recipe"));
      });
    }
    return () => {
      active = false;
    };
  }, []);
  const updateConfig = <K extends keyof BrainConfig>(key: K, value: BrainConfig[K]) =>
    setConfig((current) => ({ ...current, [key]: value }));

  const chooseRecipe = (recipe: RecipeId) => {
    const basePreset: ArchitecturePreset = recipe;
    const next =
      recipe === "custom"
        ? {
            ...DEFAULT_CONFIG,
            name,
            preset: "custom" as const,
            description: "A custom cognitive architecture assembled in Omni AGI Studio.",
            spikingDynamics: false,
            stdpPlasticity: false,
            liquidDynamics: false,
            vectorSymbolicMemory: false
          }
        : createPresetConfig(basePreset, name);
    setConfig(next);
    setExtras((current) => ({ ...current, recipe }));
  };

  const updateMemoryRecipe = (recipe: MemoryRecipe) => {
    setExtras((current) => ({ ...current, memoryRecipe: recipe }));
    setConfig((current) => ({
      ...current,
      retainSourceText: recipe === "recall",
      memoryInjection: recipe === "recall" ? "working-memory" : "parameter-only",
      consolidation: recipe !== "recall",
      memoryRecipe:
        recipe === "recall"
          ? "total-recall"
          : recipe === "synapses"
            ? "synapses-only"
            : "human-consolidation"
    }));
  };

  const applyBuildRecipe = (recipe: BuildRecipe) => {
    const permission = (toolId: string): PermissionLevel =>
      recipe.toolPermissions.find((item) => item.toolId === toolId)?.level ?? "ask";
    const memoryRecipe: MemoryRecipe =
      recipe.config.memoryRecipe === "total-recall"
        ? "recall"
        : recipe.config.memoryRecipe === "synapses-only"
          ? "synapses"
          : "human";
    setName(recipe.name);
    setConfig({ ...recipe.config, name: recipe.name });
    setExtras({
      recipe: recipe.config.preset,
      hardware: recipe.hardwareTier,
      memoryRecipe,
      origin: recipe.origin,
      starterUrl: recipe.starterUrl ?? "",
      initialTraining: false,
      modalities: {
        vision: recipe.modalities.includes("vision"),
        image: recipe.modalities.includes("image"),
        audio: recipe.modalities.includes("audio"),
        video: recipe.modalities.includes("video")
      },
      tools: {
        files: permission("windows.files"),
        powershell: permission("windows.powershell"),
        code: permission("code.execute"),
        web: permission("web.fetch"),
        browser: permission("browser.automation"),
        agents: permission("agent.fork")
      }
    });
    setRecipeStatus(
      `Loaded ${recipe.name} · ${recipe.sha256.slice(0, 12)}… · ${recipe.license}`
    );
  };

  const loadBuildRecipe = async (
    loader: () => Promise<BuildRecipe | null>
  ) => {
    if (!window.omni) {
      setRecipeStatus("Recipe validation is available in the packaged desktop app.");
      return;
    }
    setRecipeLoading(true);
    try {
      const recipe = await loader();
      if (recipe) applyBuildRecipe(recipe);
    } catch (error) {
      setRecipeStatus(error instanceof Error ? error.message : "The recipe could not be loaded.");
    } finally {
      setRecipeLoading(false);
    }
  };

  const submit = async () => {
    setBuilding(true);
    try {
      await onCreate({ ...config, name }, extras);
    } finally {
      setBuilding(false);
    }
  };

  const content = [
    <div className="builder-stage" key="architecture">
      <div className="builder-stage__intro">
        <span className="stage-number">01</span>
        <div>
          <h2>What kind of mind are you growing?</h2>
          <p>Recipes are starting anatomies, not permanent limits. Every subsystem stays inspectable.</p>
        </div>
      </div>
      <div className="recipe-grid">
        {recipeMeta.map((recipe) => (
          <button
            key={recipe.id}
            className={cx("recipe-card", extras.recipe === recipe.id && "is-selected")}
            onClick={() => chooseRecipe(recipe.id)}
          >
            <span className={cx("recipe-card__icon", `recipe-card__icon--${recipe.color}`)}>
              <Icon name={recipe.icon} size={24} />
            </span>
            <span className="recipe-card__check">
              <Icon name="check" size={13} />
            </span>
            <strong>{recipe.title}</strong>
            <p>{recipe.short}</p>
            <span className="tag-row">
              {recipe.features.map((feature) => (
                <i key={feature}>{feature}</i>
              ))}
            </span>
          </button>
        ))}
      </div>
      <div className="declarative-recipes">
        <div className="settings-panel__title">
          <span><Icon name="archive" size={16} /> Declarative build recipes</span>
          <small>JSON only · no repository scripts execute</small>
        </div>
        {catalogRecipes.length ? (
          <div className="declarative-recipes__bundled">
            {catalogRecipes.map((entry) => (
              <button
                key={entry.id}
                disabled={recipeLoading}
                onClick={() =>
                  void loadBuildRecipe(() => window.omni!.catalog.loadRecipeEntry(entry.id))
                }
              >
                <strong>{entry.name}</strong>
                <small>{entry.license}</small>
              </button>
            ))}
          </div>
        ) : null}
        <label className="starter-url">
          <span>HTTPS recipe URL</span>
          <div>
            <Icon name="download" size={15} />
            <input
              value={recipeUrl}
              onChange={(event) => setRecipeUrl(event.target.value)}
              placeholder="https://github.com/…/omni-recipe.json"
            />
            <button
              className="inline-link-button"
              disabled={!recipeUrl.trim() || recipeLoading}
              onClick={() =>
                void loadBuildRecipe(() =>
                  window.omni!.catalog.loadRecipeUrl({ url: recipeUrl.trim() })
                )
              }
            >
              Load URL
            </button>
          </div>
        </label>
        <div className="declarative-recipes__actions">
          <Button
            icon="upload"
            disabled={recipeLoading}
            onClick={() => void loadBuildRecipe(() => window.omni!.catalog.loadRecipeFile())}
          >
            Open local JSON
          </Button>
          {recipeStatus ? <small role="status">{recipeStatus}</small> : null}
        </div>
      </div>
      <div className="origin-selector">
        <div className="settings-panel__title">
          <span>
            <Icon name="sparkles" size={17} />
            Starting point
          </span>
          <small>Both run the custom OmniCortex architecture</small>
        </div>
        <div className="origin-selector__choices">
          <button
            className={extras.origin === "blank" ? "is-selected" : ""}
            onClick={() => setExtras((current) => ({ ...current, origin: "blank", starterUrl: "" }))}
          >
            <span className="choice-card__icon"><Icon name="plus" size={19} /></span>
            <span>
              <strong>Blank origin</strong>
              <small>Random weights. Primitive at first; everything is learned here.</small>
            </span>
            <i><Icon name="check" size={12} /></i>
          </button>
          <button
            className={extras.origin === "starter" ? "is-selected" : ""}
            onClick={() => setExtras((current) => ({ ...current, origin: "starter" }))}
          >
            <span className="choice-card__icon"><Icon name="download" size={19} /></span>
            <span>
              <strong>Compatible Omni starter</strong>
              <small>Initialize from a verified safe-tensor OmniCortex checkpoint.</small>
            </span>
            <i><Icon name="check" size={12} /></i>
          </button>
        </div>
        {extras.origin === "starter" ? (
          <label className="starter-url">
            <span>Starter bundle URL</span>
            <div>
              <Icon name="download" size={15} />
              <input
                value={extras.starterUrl}
                onChange={(event) => setExtras((current) => ({ ...current, starterUrl: event.target.value }))}
                placeholder="https://github.com/…/brain.omni"
              />
            </div>
            <small>Only compatible `.omni` manifests and safe weights are accepted; setup scripts never run.</small>
          </label>
        ) : null}
      </div>
      <div className="builder-note">
        <Icon name="info" size={17} />
        <span>
          <strong>Built from random weights.</strong> A blank brain starts primitive and learns through training and experience.
        </span>
      </div>
    </div>,

    <div className="builder-stage" key="compute">
      <div className="builder-stage__intro">
        <span className="stage-number">02</span>
        <div>
          <h2>Fit the brain to your machine.</h2>
          <p>Omni scales width, batching, checkpointing, and media quality around your hardware.</p>
        </div>
      </div>
      <div className="detected-hardware">
        <div className="detected-hardware__glow">
          <Icon name="memory" size={28} />
        </div>
        <div>
          <span className="eyebrow-text">Detected on this device</span>
          <h3>
            {detectedHardware
              ? `${detectedHardware.platform} · ${detectedHardware.architecture}`
              : "Simulated Windows 11 profile"}
          </h3>
          <p>
            {detectedHardware
              ? `${detectedHardware.logicalCpus} logical CPUs · ${(detectedHardware.totalMemoryBytes / 1_073_741_824).toFixed(1)} GB memory · ${
                  detectedHardware.gpu.available
                    ? detectedHardware.gpu.device ?? detectedHardware.gpu.vendor ?? "GPU available"
                    : "CPU execution"
                }`
              : "Design preview only · actual hardware will be detected inside the Windows app"}
          </p>
        </div>
        <span className="healthy-pill">
          <Icon name={detectedHardware ? "check" : "info"} size={13} />{" "}
          {detectedHardware ? `Recommends ${detectedHardware.recommendedTier}` : "Simulated"}
        </span>
      </div>
      <div className="choice-grid choice-grid--four">
        {(
          [
            ["auto", "Automatic", "Recommended", "Continuously adapts batch and memory use.", "sparkles"],
            ["micro", "Micro", "4–8 GB", "Tiny experiments and CPU-only learning.", "memory"],
            ["personal", "Personal", "16 GB", "Balanced local training and generation.", "brain"],
            ["gpu", "GPU", "CUDA / DirectML", "Accelerated local training with safe CPU fallback.", "sparkles"],
            ["workstation", "Workstation", "32 GB+", "Wider models and richer modalities.", "pulse"]
          ] as const
        ).map(([id, title, badge, copy, icon]) => (
          <button
            key={id}
            className={cx("choice-card", extras.hardware === id && "is-selected")}
            onClick={() => setExtras((current) => ({ ...current, hardware: id }))}
          >
            <span className="choice-card__icon">
              <Icon name={icon} size={20} />
            </span>
            <strong>{title}</strong>
            <em>{badge}</em>
            <p>{copy}</p>
          </button>
        ))}
      </div>
      <div className="settings-panel">
        <div className="settings-panel__title">
          <span>
            <Icon name="settings" size={17} />
            Initial capacity
          </span>
          <small>Growth can add structure later</small>
        </div>
        <RangeField
          label="Neuron budget"
          detail="Sparse concept and routing units at origin"
          value={config.initialNeuronBudget}
          min={512}
          max={16384}
          step={512}
          display={compactNumber(config.initialNeuronBudget)}
          onChange={(value) => updateConfig("initialNeuronBudget", value)}
        />
        <RangeField
          label="Maximum concepts"
          detail="Soft storage ceiling before cleanup"
          value={config.maxConcepts}
          min={10_000}
          max={1_000_000}
          step={10_000}
          display={compactNumber(config.maxConcepts)}
          onChange={(value) => updateConfig("maxConcepts", value)}
        />
      </div>
    </div>,

    <div className="builder-stage" key="memory">
      <div className="builder-stage__intro">
        <span className="stage-number">03</span>
        <div>
          <h2>How should experience become memory?</h2>
          <p>Memory recipes govern source retention, consolidation, and how ideas enter generation.</p>
        </div>
      </div>
      <div className="memory-recipes">
        {(
          [
            [
              "human",
              "Human consolidation",
              "Recommended",
              "Experience becomes ideas, associations, fast synapses, and slow weights. Exact wording fades.",
              ["Parameter-only recall", "Semantic consolidation", "Natural forgetting"]
            ],
            [
              "recall",
              "Total recall",
              "Exact archive",
              "Keeps local source passages alongside learned representations for optional exact retrieval.",
              ["Source archive", "Working-memory retrieval", "Higher disk use"]
            ],
            [
              "synapses",
              "Synapses only",
              "Private",
              "Deletes source material after encoding. Recall is reconstructive and may be imperfect.",
              ["No raw source", "Parameter-only", "Irreversible"]
            ]
          ] as const
        ).map(([id, title, badge, copy, bullets]) => (
          <button
            key={id}
            className={cx("memory-card", extras.memoryRecipe === id && "is-selected")}
            onClick={() => updateMemoryRecipe(id)}
          >
            <span className="memory-card__top">
              <span className="memory-card__radio" />
              <em>{badge}</em>
            </span>
            <strong>{title}</strong>
            <p>{copy}</p>
            <span className="memory-card__bullets">
              {bullets.map((bullet) => (
                <i key={bullet}>
                  <Icon name="check" size={12} /> {bullet}
                </i>
              ))}
            </span>
          </button>
        ))}
      </div>
      <div className="settings-panel settings-panel--split">
        <RangeField
          label="Working memory"
          detail="Active idea slots available during a turn"
          value={config.workingMemorySlots}
          min={8}
          max={128}
          step={4}
          display={`${config.workingMemorySlots} slots`}
          onChange={(value) => updateConfig("workingMemorySlots", value)}
        />
        <RangeField
          label="Short-term half-life"
          detail="Time before unused activation is halved"
          value={config.shortTermHalfLifeMinutes}
          min={5}
          max={240}
          step={5}
          display={`${config.shortTermHalfLifeMinutes} min`}
          onChange={(value) => updateConfig("shortTermHalfLifeMinutes", value)}
        />
        <RangeField
          label="Long-term threshold"
          detail="Importance needed for slow consolidation"
          value={config.longTermThreshold}
          min={0.1}
          max={0.95}
          step={0.01}
          display={`${Math.round(config.longTermThreshold * 100)}%`}
          onChange={(value) => updateConfig("longTermThreshold", value)}
        />
      </div>
    </div>,

    <div className="builder-stage" key="plasticity">
      <div className="builder-stage__intro">
        <span className="stage-number">04</span>
        <div>
          <h2>Choose how boldly it can change.</h2>
          <p>Fast local plasticity handles the moment. Consolidation turns repeated patterns into durable structure.</p>
        </div>
      </div>
      <div className="builder-two-column">
        <div className="settings-panel settings-panel--flush">
          <div className="settings-panel__title">
            <span>
              <Icon name="pulse" size={17} /> Learning systems
            </span>
          </div>
          <Toggle
            checked={config.onlineLearning}
            onChange={(value) => updateConfig("onlineLearning", value)}
            label="Online learning"
            description="Every conversation can modify the brain"
          />
          <Toggle
            checked={config.ternaryWeights}
            onChange={(value) => updateConfig("ternaryWeights", value)}
            label="Ternary projections"
            description="Effective −1, 0, +1 synapses with latent master weights"
          />
          <Toggle
            checked={config.spikingDynamics}
            onChange={(value) => updateConfig("spikingDynamics", value)}
            label="Spiking dynamics"
            description="Sparse LIF routing for salience and novelty"
          />
          <Toggle
            checked={config.stdpPlasticity}
            onChange={(value) => updateConfig("stdpPlasticity", value)}
            label="STDP fast synapses"
            description="Timing-sensitive local weight updates"
            disabled={!config.spikingDynamics}
          />
          <Toggle
            checked={config.liquidDynamics}
            onChange={(value) => updateConfig("liquidDynamics", value)}
            label="Liquid dynamics"
            description={`${config.liquidMode === "cfc" ? "CfC" : "LTC"} temporal controller and adaptive time constants`}
          />
          <Toggle
            checked={config.vectorSymbolicMemory}
            onChange={(value) => updateConfig("vectorSymbolicMemory", value)}
            label="VSA idea memory"
            description="Compositional concepts above the token boundary"
          />
          <Toggle
            checked={config.metaplasticity}
            onChange={(value) => updateConfig("metaplasticity", value)}
            label="Metaplastic stability"
            description="Protects frequently reinforced pathways"
          />
          <Toggle
            checked={config.learnFromOwnMessages}
            onChange={(value) => updateConfig("learnFromOwnMessages", value)}
            label="Learn from self"
            description="Its own outputs can become experience"
          />
        </div>
        <div className="settings-panel settings-panel--flush">
          <div className="settings-panel__title">
            <span>
              <Icon name="expand" size={17} /> Structural growth
            </span>
          </div>
          <div className="segmented segmented--wide">
            {(["fixed", "elastic", "unbounded"] as const).map((policy) => (
              <button
                key={policy}
                className={config.growthPolicy === policy ? "is-active" : ""}
                onClick={() => updateConfig("growthPolicy", policy)}
              >
                {policy[0]?.toUpperCase() + policy.slice(1)}
              </button>
            ))}
          </div>
          <p className="panel-copy">
            {config.growthPolicy === "unbounded"
              ? "No model-defined ceiling. Growth pauses before exhausting available memory or disk."
              : config.growthPolicy === "elastic"
                ? "Adds sparse experts after sustained novelty and prunes low-use structures."
                : "The origin architecture remains fixed; only existing weights can change."}
          </p>
          <RangeField
            label="Exploratory noise"
            detail="Variability in activation and selection"
            value={config.noise}
            min={0}
            max={0.3}
            step={0.01}
            display={`${Math.round(config.noise * 100)}%`}
            onChange={(value) => updateConfig("noise", value)}
          />
          <RangeField
            label="Curiosity drive"
            detail="Preference for unresolved, novel pathways"
            value={config.curiosityDrive}
            min={0}
            max={1}
            step={0.01}
            display={`${Math.round(config.curiosityDrive * 100)}%`}
            onChange={(value) => updateConfig("curiosityDrive", value)}
          />
          <RangeField
            label="Parallel thoughts"
            detail="Candidate branches explored before selection"
            value={config.parallelThoughts}
            min={1}
            max={8}
            step={1}
            display={`${config.parallelThoughts}`}
            onChange={(value) => updateConfig("parallelThoughts", value)}
          />
        </div>
      </div>
      <details className="advanced-equations" open={extras.recipe === "custom"}>
        <summary>
          <span>
            <Icon name="code" size={16} />
            Advanced dynamics & equations
          </span>
          <span>Research controls <Icon name="chevron" size={14} /></span>
        </summary>
        <div className="advanced-equations__content">
          <div className="liquid-mode">
            <span>
              <strong>Liquid cell</strong>
              <small>CfC is the stable default; LTC directly integrates a continuous-time ODE.</small>
            </span>
            <div className="segmented">
              <button className={config.liquidMode === "cfc" ? "is-active" : ""} onClick={() => updateConfig("liquidMode", "cfc")}>
                CfC · closed-form
              </button>
              <button className={config.liquidMode === "ltc" ? "is-active" : ""} onClick={() => updateConfig("liquidMode", "ltc")}>
                LTC · ODE
              </button>
            </div>
          </div>
          <div className="equation-strip" aria-label="Active plasticity equations">
            <span>
              <small>LIF membrane</small>
              <code>τₘ dV/dt = −V + RI(t)</code>
            </span>
            <span>
              <small>Timing plasticity</small>
              <code>Δw = A± exp(−|Δt|/τ±)</code>
            </span>
            <span>
              <small>Ternary projection</small>
              <code>W̃ = RoundClip(W / γ)</code>
            </span>
          </div>
          <div className="equation-controls">
            <RangeField
              label="Firing threshold"
              detail="Membrane voltage required to spike"
              value={config.firingThreshold}
              min={0.1}
              max={1}
              step={0.01}
              display={config.firingThreshold.toFixed(2)}
              onChange={(value) => updateConfig("firingThreshold", value)}
            />
            <RangeField
              label="Membrane leak"
              detail="State retained between integration steps"
              value={config.membraneLeak}
              min={0.1}
              max={0.99}
              step={0.01}
              display={config.membraneLeak.toFixed(2)}
              onChange={(value) => updateConfig("membraneLeak", value)}
            />
            <RangeField
              label="STDP window"
              detail="Causal timing horizon"
              value={config.stdpWindow}
              min={1}
              max={32}
              step={1}
              display={`${config.stdpWindow} ticks`}
              onChange={(value) => updateConfig("stdpWindow", value)}
            />
            <RangeField
              label="Consolidation rate"
              detail="Fast-to-slow parameter promotion"
              value={config.consolidationRate}
              min={0.001}
              max={0.25}
              step={0.001}
              display={config.consolidationRate.toFixed(3)}
              onChange={(value) => updateConfig("consolidationRate", value)}
            />
          </div>
        </div>
      </details>
    </div>,

    <div className="builder-stage" key="modalities">
      <div className="builder-stage__intro">
        <span className="stage-number">05</span>
        <div>
          <h2>Give it senses and ways to act.</h2>
          <p>Every enabled modality shares the same idea space. Packs can begin blank and learn later.</p>
        </div>
      </div>
      <h3 className="minor-heading">Perception & imagination</h3>
      <div className="modality-grid">
        {(
          [
            ["vision", "Vision", "Understand images in the shared concept space.", "eye", "Hardware-scaled"],
            ["image", "Image imagination", "Generate images from internal idea vectors.", "image", "Tiny local"],
            ["audio", "Audio", "Hear, encode, and imagine sound or speech.", "volume", "Tiny local"],
            ["video", "Video", "Learn temporal scenes and generate short motion.", "video", "Hardware-scaled"]
          ] as const
        ).map(([id, title, copy, icon, size]) => (
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
            <span className="modality-card__icon">
              <Icon name={icon} size={20} />
            </span>
            <span>
              <strong>{title}</strong>
              <p>{copy}</p>
            </span>
            <em>{size}</em>
            <span className="modality-card__check">
              <Icon name="check" size={12} />
            </span>
          </button>
        ))}
      </div>
      <h3 className="minor-heading minor-heading--tools">Tool permissions</h3>
      <div className="tool-table">
        {toolRows.map(([id, title, icon]) => (
          <div className="tool-row" key={id}>
            <span className="tool-row__identity">
              <span>
                <Icon name={icon} size={17} />
              </span>
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
      <div className="builder-note builder-note--warning">
        <Icon name="warning" size={17} />
        <span>Full authority skips confirmation, but every action remains visible in the operational trace.</span>
      </div>
    </div>,

    <div className="builder-stage" key="review">
      <div className="builder-stage__intro">
        <span className="stage-number">06</span>
        <div>
          <h2>Name the origin.</h2>
          <p>This snapshot never changes. Every future memory, fork, and experiment can trace back here.</p>
        </div>
      </div>
      <div className="review-layout">
        <div>
          <label className="name-field">
            <span>Identity</span>
            <input
              value={name}
              maxLength={40}
              onChange={(event) => {
                setName(event.target.value);
                updateConfig("name", event.target.value);
              }}
              placeholder="Name this mind"
              autoFocus
            />
            <small>This can change later. Its lineage ID cannot.</small>
          </label>
          <label className="description-field">
            <span>Origin note</span>
            <textarea
              value={config.description}
              onChange={(event) => updateConfig("description", event.target.value)}
              rows={3}
            />
          </label>
          <label className="initial-training-card">
            <input
              type="checkbox"
              checked={extras.initialTraining}
              onChange={(event) =>
                setExtras((current) => ({ ...current, initialTraining: event.target.checked }))
              }
            />
            <span className="initial-training-card__icon"><Icon name="database" size={18} /></span>
            <span>
              <strong>Pretrain before the first conversation</strong>
              <small>
                After the immutable origin is created, choose local files to update its initial
                slow weights, ideas, and synapses.
              </small>
            </span>
            <i><Icon name="check" size={12} /></i>
          </label>
          <div className="origin-principles">
            <span>
              <Icon name="check" size={14} /> {extras.origin === "blank" ? "Random-weight origin" : "Verified Omni starter origin"}
            </span>
            <span>
              <Icon name="check" size={14} /> No reward model or RLHF
            </span>
            <span>
              <Icon name="check" size={14} /> No hidden persona prompt
            </span>
            <span>
              <Icon name="check" size={14} /> Immutable recovery point
            </span>
          </div>
        </div>
        <aside className="blueprint-card">
          <div className="blueprint-card__head">
            <span className={cx("recipe-card__icon", "recipe-card__icon--violet")}>
              <Icon name={recipeMeta.find((recipe) => recipe.id === extras.recipe)?.icon ?? "brain"} size={22} />
            </span>
            <div>
              <span>Architecture blueprint</span>
              <strong>{recipeMeta.find((recipe) => recipe.id === extras.recipe)?.title}</strong>
            </div>
          </div>
          <div className="blueprint-flow">
            <span>Language boundary</span>
            <i />
            <span>Ternary cortex</span>
            <i />
            <span>Idea space</span>
          </div>
          <dl className="review-list">
            <div>
              <dt>Origin</dt>
              <dd>{extras.origin === "blank" ? "Blank / random weights" : "Compatible Omni starter"}</dd>
            </div>
            <div>
              <dt>Compute</dt>
              <dd>{extras.hardware === "auto" ? "Automatic scaling" : extras.hardware}</dd>
            </div>
            <div>
              <dt>Memory</dt>
              <dd>{extras.memoryRecipe === "human" ? "Human consolidation" : extras.memoryRecipe}</dd>
            </div>
            <div>
              <dt>Growth</dt>
              <dd>{config.growthPolicy}</dd>
            </div>
            <div>
              <dt>Senses</dt>
              <dd>{Object.values(extras.modalities).filter(Boolean).length} enabled</dd>
            </div>
            <div>
              <dt>Initial neurons</dt>
              <dd>{compactNumber(config.initialNeuronBudget)}</dd>
            </div>
            <div>
              <dt>Initial capability</dt>
              <dd>
                {extras.origin === "starter"
                  ? extras.initialTraining
                    ? "Starter + local pretraining"
                    : "Pretrained starter"
                  : extras.initialTraining
                    ? "Local pretraining"
                    : "Primitive blank brain"}
              </dd>
            </div>
            <div>
              <dt>Trace</dt>
              <dd>{config.traceDetail}</dd>
            </div>
          </dl>
          <div className="blueprint-estimate">
            <Icon name="database" size={17} />
            <span>
              Origin storage
              <strong>Profile-dependent · measured after build</strong>
            </span>
          </div>
        </aside>
      </div>
    </div>
  ];

  return (
    <main className="builder-page">
      <aside className="builder-sidebar">
        <button className="builder-back" onClick={onCancel}>
          <Icon name="arrow" size={15} /> Brain Library
        </button>
        <div className="builder-sidebar__intro">
          <span>NEW ORIGIN</span>
          <h1>Build a brain</h1>
          <p>Shape the starting anatomy. It will decide what to become through experience.</p>
        </div>
        <ol className="step-list">
          {buildSteps.map(([title, copy], index) => (
            <li key={title} className={cx(index === step && "is-active", index < step && "is-complete")}>
              <button onClick={() => setStep(index)} aria-current={index === step ? "step" : undefined}>
                <span>{index < step ? <Icon name="check" size={13} /> : index + 1}</span>
                <span>
                  <strong>{title}</strong>
                  <small>{copy}</small>
                </span>
              </button>
            </li>
          ))}
        </ol>
        <div className="builder-sidebar__privacy">
          <Icon name="memory" size={17} />
          <span>
            <strong>Everything stays local</strong>
            Build state is stored only on this device.
          </span>
        </div>
      </aside>
      <section className="builder-main">
        <div className="builder-main__content">{content[step]}</div>
        <footer className="builder-footer">
          <span>
            Step {step + 1} of {buildSteps.length}
            <i>
              {buildSteps.map((_, index) => (
                <b key={index} className={index <= step ? "is-filled" : ""} />
              ))}
            </i>
          </span>
          <div>
            {step > 0 ? (
              <Button onClick={() => setStep((current) => current - 1)}>Back</Button>
            ) : null}
            {step < buildSteps.length - 1 ? (
              <Button kind="primary" onClick={() => setStep((current) => current + 1)}>
                Continue <Icon name="arrow" size={14} />
              </Button>
            ) : (
          <Button
            kind="primary"
            icon={building ? "pulse" : "sparkles"}
            disabled={!name.trim() || building || (extras.origin === "starter" && !extras.starterUrl.trim())}
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
  onBrainChange,
  onToast
}: {
  brain: BrainDocument;
  view: WorkspaceView;
  onView: (view: WorkspaceView) => void;
  onLibrary: () => void;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
}) {
  const [health, setHealth] = useState("Adaptive core ready");
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
              ? result.worker === "python"
                ? `Python engine${result.pid ? ` · PID ${result.pid}` : ""}`
                : "Built-in fallback engine"
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
      <section className="workspace-body">
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
        {view === "chat" ? (
          <ChatWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} onNavigate={onView} />
        ) : view === "data" ? (
          <DataWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        ) : view === "map" ? (
          <BrainMapWorkspace brain={brain} />
        ) : view === "trace" ? (
          <TraceWorkspace brain={brain} />
        ) : view === "imagine" ? (
          <ImaginationWorkspace brain={brain} onToast={onToast} />
        ) : view === "tools" ? (
          <ToolsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        ) : (
          <AgentsWorkspace brain={brain} onBrainChange={onBrainChange} onToast={onToast} />
        )}
      </section>
    </main>
  );
}

interface ChatToolCommand {
  label: string;
  invocation: Omit<ToolInvocation, "brainId" | "approvalToken">;
  source: "human" | "brain";
}

interface PendingChatTool extends ChatToolCommand {
  approvalToken: string;
}

function objectArguments(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error("Tool arguments must be a JSON object.");
  }
  return value as Record<string, unknown>;
}

function parseChatToolCommand(text: string): ChatToolCommand | null {
  if (text.startsWith("/imagine")) {
    const [, rawModality = "image", ...conceptWords] = text.split(/\s+/);
    const modality = rawModality.toLocaleLowerCase();
    if (!["image", "audio", "video"].includes(modality)) {
      throw new Error("Use /imagine image, /imagine audio, or /imagine video.");
    }
    return {
      label: text,
      source: "human",
      invocation: {
        toolId: "modality.imagine",
        action: "generate",
        arguments: {
          modality,
          conceptIds: conceptWords.length ? [conceptWords.join(" ")] : []
        }
      }
    };
  }
  if (text.startsWith("/agent")) {
    const objective = text.slice("/agent".length).trim();
    if (!objective) throw new Error("Add an objective after /agent.");
    return {
      label: text,
      source: "human",
      invocation: {
        toolId: "agent.fork",
        action: "start",
        arguments: { objective }
      }
    };
  }
  if (!text.startsWith("/tool")) return null;
  const match = /^\/tool\s+([a-z][a-z0-9.-]{1,79})\s+([a-z][a-z0-9_-]{0,79})(?:\s+([\s\S]+))?$/i.exec(
    text
  );
  if (!match) {
    throw new Error('Use /tool <tool.id> <action> {"argument":"value"}.');
  }
  let argumentsValue: Record<string, unknown> = {};
  if (match[3]?.trim()) argumentsValue = objectArguments(JSON.parse(match[3]));
  return {
    label: text,
    source: "human",
    invocation: {
      toolId: match[1]!.toLocaleLowerCase(),
      action: match[2]!.toLocaleLowerCase(),
      arguments: argumentsValue
    }
  };
}

function parseBrainToolCall(text: string): ChatToolCommand | null {
  const match = /<omni-tool>\s*([\s\S]{2,50000}?)\s*<\/omni-tool>/i.exec(text);
  if (!match?.[1]) return null;
  try {
    const value = objectArguments(JSON.parse(match[1]));
    const toolId = typeof value.toolId === "string" ? value.toolId.toLocaleLowerCase() : "";
    const action = typeof value.action === "string" ? value.action.toLocaleLowerCase() : "";
    if (!/^[a-z][a-z0-9.-]{1,79}$/.test(toolId) || !/^[a-z][a-z0-9_-]{0,79}$/.test(action)) {
      return null;
    }
    return {
      label: `${toolId}.${action} proposed by the brain`,
      source: "brain",
      invocation: {
        toolId,
        action,
        arguments: objectArguments(value.arguments ?? {})
      }
    };
  } catch {
    return null;
  }
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

function ChatWorkspace({
  brain,
  onBrainChange,
  onToast,
  onNavigate
}: {
  brain: BrainDocument;
  onBrainChange: (brain: BrainDocument) => void;
  onToast: (message: string) => void;
  onNavigate: (view: WorkspaceView) => void;
}) {
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [attaching, setAttaching] = useState(false);
  const [pendingTool, setPendingTool] = useState<PendingChatTool | null>(null);
  const [toolStatus, setToolStatus] = useState("");
  const [toolRunning, setToolRunning] = useState(false);
  const [inspectorTab, setInspectorTab] = useState<"state" | "runtime">("state");
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ block: "end" });
  }, [brain.messages, sending]);

  const runChatTool = async (
    command: ChatToolCommand,
    approvalToken?: string,
    chainDepth = 0
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
    const nextCall = parseBrainToolCall(result.brainMessage.content);
    if (nextCall && chainDepth < 3) {
      await runChatTool(nextCall, undefined, chainDepth + 1);
    } else if (nextCall) {
      setToolStatus("The brain proposed another tool, but the four-action per-turn limit stopped the chain.");
    }
  };

  const send = async () => {
    const text = input.trim();
    if (!text || sending) return;
    setSending(true);
    try {
      if (text === "/help") {
        setInput("");
        setToolStatus(
          'Chat commands: /tool <tool.id> <action> {"arguments":true}, /imagine image|audio|video, or /agent <objective>. A trained brain may propose the same protocol with <omni-tool> JSON.'
        );
      } else if (window.omni) {
        const command = parseChatToolCommand(text);
        setInput("");
        if (command) {
          await runChatTool(command);
        } else {
          const result = await window.omni.chat.send(brain.id, text);
          onBrainChange(result.brain);
          const proposedTool = parseBrainToolCall(result.brainMessage.content);
          if (proposedTool) await runChatTool(proposedTool);
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
      onToast(error instanceof Error ? error.message : "The local brain could not respond.");
    } finally {
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
    const count = await window.omni.tool.cancel(brain.id);
    setToolStatus(
      count > 0
        ? "Cancellation requested. The interrupted outcome will remain visible in the tool trace."
        : "No cancellable tool execution is active."
    );
  };

  const attachExperience = async () => {
    if (!window.omni) {
      onNavigate("data");
      return;
    }
    if (attaching || sending) return;
    setAttaching(true);
    try {
      const results = await window.omni.data.ingestFiles({
        brainId: brain.id,
        policy: "consolidate"
      });
      if (results.length > 0) {
        onBrainChange(await window.omni.brain.get(brain.id));
        setToolStatus(
          `${results.length} attached source${results.length === 1 ? "" : "s"} encoded into parameters, ideas, and synapses.`
        );
      }
    } catch (error) {
      onToast(error instanceof Error ? error.message : "The attachment could not be learned.");
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

  const displayedMessages =
    brain.messages.length > 0
      ? brain.messages
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
      .slice(0, 5)
      .map((concept) => ({ id: concept.id, label: concept.label, activation: concept.activation }));

  return (
    <div className="chat-layout">
      <section className="conversation">
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
          {sending ? (
            <div className="message message--brain">
              <div className="message__avatar">
                <BrandMark size={28} />
              </div>
              <div className="message__body">
                <div className="message__meta">
                  <strong>{brain.name}</strong>
                  <span className="pondering-label">
                    <i /> pondering
                  </span>
                </div>
                <div className="pondering">
                  <span />
                  <span />
                  <span />
                  <em>Following a quieter association…</em>
                </div>
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
                <button aria-label="Attach experience" title="Attach and learn now" onClick={() => void attachExperience()}>
                  <Icon name={attaching ? "pulse" : "plus"} size={18} />
                </button>
                <button aria-label="Imagine an image" title="Imagine directly in chat" onClick={() => setInput("/imagine image ")}>
                  <Icon name="image" size={18} />
                </button>
                <button aria-label="Attach audio experience" title="Attach audio and learn now" onClick={() => void attachExperience()}>
                  <Icon name="volume" size={18} />
                </button>
              </div>
              <span>Enter to send · Shift Enter for a line break</span>
              <button className="send-button" onClick={() => void send()} disabled={!input.trim() || sending} aria-label="Send message">
                <Icon name="send" size={17} />
              </button>
            </div>
            <div className="composer__commands" aria-label="Chat feature shortcuts">
              <button onClick={() => setInput("/tool web.fetch fetch {\"url\":\"https://example.com\"}")}>/tool</button>
              <button onClick={() => setInput("/imagine image ")}>/imagine</button>
              <button onClick={() => setInput("/agent ")}>/agent</button>
              <button onClick={() => setInput("/help")}>/help</button>
            </div>
          </div>
          <p className="composer-disclaimer">
            Learned parameters, active state, and structured tool schemas drive each turn. Tool actions stay permissioned and visible.
          </p>
        </div>
      </section>

      <aside className="cortex-panel">
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
            <CortexOrb activity={0.82} />
            <div className="cortex-readout">
              <span>
                <i className="dot-violet" />
                <small>Current state</small>
                <strong>{sending ? "Pondering" : "Attentive"}</strong>
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
                <span>Active ideas</span>
                <em>{activeConcepts.length} / {brain.config.workingMemorySlots}</em>
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
            <div className="panel-section">
              <div className="panel-section__head">
                <span>Drives</span>
                <em>adaptive</em>
              </div>
              <div className="drive-grid">
                {[
                  ["Curiosity", recentTrace?.driveScores.curiosity ?? brain.config.curiosityDrive],
                  ["Coherence", recentTrace?.driveScores.coherence ?? brain.config.coherenceDrive],
                  ["Novelty", recentTrace?.driveScores.novelty ?? brain.config.noveltyDrive]
                ].map(([label, value]) => (
                  <div key={String(label)}>
                    <span>{label}</span>
                    <strong>{Math.round(Number(value) * 100)}%</strong>
                    <i>
                      <b style={{ width: `${Number(value) * 100}%` }} />
                    </i>
                  </div>
                ))}
              </div>
            </div>
            <button className="trace-link" onClick={() => onNavigate("trace")}>
              <Icon name="trace" size={15} />
              Open latest operational trace
              <Icon name="arrow" size={13} />
            </button>
          </>
        ) : (
          <RuntimeCard brain={brain} />
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

function CortexOrb({ activity }: { activity: number }) {
  return (
    <div className="cortex-orb" aria-label={`${Math.round(activity * 100)} percent neural activity`}>
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
        {Math.round(activity * 100)}% active
      </span>
    </div>
  );
}

function RuntimeCard({ brain }: { brain: BrainDocument }) {
  const currentTokens =
    [...brain.traces]
      .reverse()
      .find((trace) => trace.steps.some((step) => step.stage === "encode"))
      ?.steps.find((step) => step.stage === "encode")
      ?.value ?? "No completed turn";
  const enabledTools = (brain.toolPermissions ?? [])
    .filter((permission) => permission.level !== "off")
    .map((permission) => permission.toolId);
  const rows = [
    ["Behavioral system prompt", "None"],
    ["Memory injection", brain.config.memoryInjection === "parameter-only" ? "Parameter-only" : "Working memory"],
    ["Reward model / RLHF", "None"],
    ["Runtime", brain.config.runtime],
    ["Current turn tokens", currentTokens],
    ["Working memory", `${brain.workingMemory.length} / ${brain.config.workingMemorySlots} slots`],
    ["Tool schemas", enabledTools.length ? `${enabledTools.length} visible` : "None enabled"],
    ["Trace detail", brain.config.traceDetail]
  ];
  return (
    <div className="runtime-card">
      <div className="runtime-card__seal">
        <Icon name="check" size={20} />
      </div>
      <h3>Transparent runtime</h3>
      <p>Everything outside learned parameters and active neural state is shown here.</p>
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
        Tool schemas describe available actions; they do not prescribe a personality.
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
  const [quarantine, setQuarantine] = useState(true);
  const [dragging, setDragging] = useState(false);

  const busy =
    demoTraining || activeJob?.state === "queued" || activeJob?.state === "running";
  const progress = demoTraining
    ? demoProgress
    : activeJob
      ? Math.round(Math.max(0, Math.min(1, activeJob.progress)) * 100)
      : 0;
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
  const capacityRatio = Math.min(1, Object.keys(brain.concepts).length / Math.max(1, brain.config.maxConcepts));
  const healthScore = synapses.length
    ? Math.round(((averageStability + averagePlasticity + (1 - capacityRatio)) / 3) * 100)
    : 0;
  const healthRows: Array<[string, number, string]> = [
    ["Stability", averageStability * 100, synapses.length ? "Measured" : "No synapses"],
    ["Plasticity", averagePlasticity * 100, synapses.length ? "Measured" : "No synapses"],
    ["Active paths", activeRatio * 100, `${synapses.filter((item) => item.effectiveWeight !== 0).length} active`],
    ["Capacity used", capacityRatio * 100, `${compactNumber(Object.keys(brain.concepts).length)} concepts`]
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

  const ingest = async (kind: "files" | "folder" = "files") => {
    if (busy) return;
    setDemoTraining(true);
    setDemoProgress(8);
    try {
      if (window.omni) {
        setDemoTraining(false);
        const request = { brainId: brain.id, policy };
        const results =
          kind === "folder"
            ? await window.omni.data.ingestFolder(request)
            : await window.omni.data.ingestFiles(request);
        applyResults(results);
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
    try {
      applyResults(await window.omni.data.ingestDropped({ brainId: brain.id, policy }, files));
    } catch (error) {
      onToast(error instanceof Error ? error.message : "Dropped files could not be ingested.");
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
        sameOrigin: true,
        maxPages: 64,
        maxDepth: 3
      });
      setActiveJob(job);
      onToast("Quarantined crawl started. Progress is visible at right.");
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
                <strong>Drop knowledge here</strong>
                <p>PDF, Markdown, text, JSON, source code, images, audio, or video</p>
                <span>Browse files</span>
                <button
                  className="drop-zone__folder"
                  onClick={(event) => {
                    event.stopPropagation();
                    void ingest("folder");
                  }}
                >
                  <Icon name="archive" size={13} /> Choose a folder
                </button>
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
                  <Toggle checked={quarantine} onChange={setQuarantine} label="Quarantine before learning" />
                </div>
                <div className="crawler-actions">
                  <Button icon="download" disabled={!crawlUrl.trim() || busy} onClick={() => void ingestPage()}>
                    Learn this page
                  </Button>
                  <Button kind="primary" icon="play" disabled={!crawlUrl.trim() || busy} onClick={() => void startCrawl()}>
                    Crawl same origin
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
                <h2>Learned sources</h2>
                <p>Provenance stays attached to every derived idea.</p>
              </div>
            </div>
            <div className="source-table">
              <div className="source-table__header">
                <span>Source</span>
                <span>Learned</span>
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
                    <strong>{source.learnedIdeas} ideas</strong>
                    <small>{source.learnedSynapses} synapses</small>
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
                <strong>{busy ? activeJob?.label ?? "Encoding experience" : activeJob?.label ?? "No jobs yet"}</strong>
              </span>
              {busy && activeJob ? (
                <button className="icon-button" onClick={() => void window.omni?.train.cancel(activeJob.id)} aria-label="Cancel active job">
                  <Icon name="close" size={15} />
                </button>
              ) : null}
            </div>
            <div className="training-progress">
              <span>
                <strong>{progress}%</strong>
                <em>{busy ? activeJob?.state ?? "processing" : activeJob?.state ?? "idle"}</em>
              </span>
              <i>
                <b style={{ width: `${progress}%` }} />
              </i>
            </div>
            <div className="training-metrics">
              <span>
                <small>Loss</small>
                <strong>{activeJob ? "live" : "—"}</strong>
                <em>{activeJob ? activeJob.kind : "no run"}</em>
              </span>
              <span>
                <small>New ideas</small>
                <strong>{brain.trainingSources.reduce((sum, source) => sum + source.learnedIdeas, 0) || "—"}</strong>
                <em>{brain.trainingSources.length} sources</em>
              </span>
              <span>
                <small>Synaptic Δ</small>
                <strong>{brain.trainingSources.reduce((sum, source) => sum + source.learnedSynapses, 0) || "—"}</strong>
                <em>recorded changes</em>
              </span>
            </div>
            <div className="training-log">
              <Icon name="terminal" size={15} />
              {activeJob?.error ?? activeJob?.label ?? "No runtime log entries yet"}
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

function BrainMapWorkspace({ brain }: { brain: BrainDocument }) {
  const [selected, setSelected] = useState("memory");
  const [filter, setFilter] = useState<"all" | "active" | "important">("all");
  const [query, setQuery] = useState("");
  const concepts = Object.values(brain.concepts);
  const selectedConcept = brain.concepts[selected] ?? concepts[0];
  const fallback = concepts.length ? concepts : window.omni ? [] : Object.values(makeDemoBrain().concepts);
  const nodes = fallback
    .filter((concept) => {
      const matchesQuery = concept.label.toLocaleLowerCase().includes(query.toLocaleLowerCase());
      const matchesFilter =
        filter === "all" ||
        (filter === "active" && concept.activation >= 0.5) ||
        (filter === "important" && concept.importance >= 0.75);
      return matchesQuery && matchesFilter;
    })
    .slice(0, 12);
  const positions: Array<readonly [number, number]> = [
    [47, 48], [25, 30], [70, 27], [72, 62], [30, 69], [51, 18],
    [51, 78], [14, 52], [86, 44], [36, 45], [61, 46], [54, 63]
  ];
  const nodeIndex = new Map(nodes.map((concept, index) => [concept.id, index]));
  const graphEdges = Object.values(brain.synapses)
    .filter((synapse) => nodeIndex.has(synapse.sourceId) && nodeIndex.has(synapse.targetId))
    .sort((a, b) => Math.abs(b.latentWeight) - Math.abs(a.latentWeight))
    .slice(0, 36);
  const connectedSynapses = selectedConcept
    ? Object.values(brain.synapses)
        .filter((synapse) => synapse.sourceId === selectedConcept.id || synapse.targetId === selectedConcept.id)
        .sort((a, b) => Math.abs(b.latentWeight) - Math.abs(a.latentWeight))
    : [];
  const selectedStability = connectedSynapses.length
    ? connectedSynapses.reduce((sum, synapse) => sum + synapse.stability, 0) / connectedSynapses.length
    : 0;
  const recentSynapse = [...connectedSynapses].sort((a, b) => b.lastUpdatedAt.localeCompare(a.lastUpdatedAt))[0];

  return (
    <div className="content-page map-page">
      <div className="content-page__title content-page__title--compact">
        <div>
          <span className="eyebrow-text">LIVE CONNECTOME</span>
          <h1>Brain map</h1>
          <p>Inspect ideas, pathways, stability, and activity as the mind changes.</p>
        </div>
        <div className="map-toolbar">
          <label className="search-field search-field--small">
            <Icon name="search" size={15} />
            <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Find an idea" />
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
      <div className="map-layout">
        <section className="surface graph-surface">
          <div className="graph-legend">
            <span><i className="legend-dot legend-dot--active" /> Active now</span>
            <span><i className="legend-dot legend-dot--stable" /> Stable</span>
            <span><i className="legend-line" /> Excitatory</span>
            <span><i className="legend-line legend-line--negative" /> Inhibitory</span>
          </div>
          <div className="brain-graph">
            {!nodes.length ? (
              <div className="graph-empty">
                <Icon name="brain" size={28} />
                <strong>No concepts have formed yet</strong>
                <span>Conversation and encoded experience will grow the first connected ideas.</span>
              </div>
            ) : null}
            <svg viewBox="0 0 1000 650" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Interactive concept and synapse graph">
              <defs>
                <radialGradient id="mapBackground">
                  <stop offset="0" stopColor="#7259d2" stopOpacity=".13" />
                  <stop offset="1" stopColor="#0e0d16" stopOpacity="0" />
                </radialGradient>
                <filter id="mapGlow">
                  <feGaussianBlur stdDeviation="5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <ellipse cx="500" cy="325" rx="380" ry="285" fill="url(#mapBackground)" />
              {graphEdges.map((synapse) => {
                const startIndex = nodeIndex.get(synapse.sourceId) ?? 0;
                const endIndex = nodeIndex.get(synapse.targetId) ?? 0;
                const start = positions[startIndex] ?? positions[0]!;
                const end = positions[endIndex] ?? positions[0]!;
                return (
                  <Fragment key={synapse.id}>
                    <line
                      x1={start[0] * 10}
                      y1={start[1] * 6.5}
                      x2={end[0] * 10}
                      y2={end[1] * 6.5}
                      stroke={synapse.effectiveWeight < 0 ? "#f183b7" : "#9b87ff"}
                      strokeOpacity={0.16 + Math.min(0.48, Math.abs(synapse.latentWeight) * 0.45)}
                      strokeWidth={0.75 + synapse.stability * 1.5}
                      strokeDasharray={synapse.effectiveWeight < 0 ? "5 5" : undefined}
                    />
                  </Fragment>
                );
              })}
              {nodes.map((concept, index) => {
                const position = positions[index] ?? positions[0]!;
                const isSelected = selected === concept.id;
                const radius = 15 + concept.importance * 15;
                return (
                  <g
                    key={concept.id}
                    className="graph-node"
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelected(concept.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") setSelected(concept.id);
                    }}
                    transform={`translate(${position[0] * 10} ${position[1] * 6.5})`}
                  >
                    <circle
                      r={radius + (isSelected ? 10 : 3)}
                      fill={isSelected ? "#8f74ff" : index % 4 === 0 ? "#55d8cf" : "#8e76ef"}
                      opacity={isSelected ? ".15" : ".07"}
                    />
                    <circle
                      r={radius}
                      fill={isSelected ? "#a790ff" : index % 4 === 0 ? "#5bd8d0" : "#8069d5"}
                      opacity={0.55 + concept.activation * 0.35}
                      stroke={isSelected ? "#ede8ff" : "#b7a9ff"}
                      strokeOpacity={isSelected ? ".9" : ".34"}
                      strokeWidth={isSelected ? "2.5" : "1"}
                      filter={isSelected ? "url(#mapGlow)" : undefined}
                    />
                    <circle r={Math.max(4, radius * 0.25)} fill="#f2eeff" opacity=".92" />
                    <text y={radius + 21} textAnchor="middle" fill="#d9d4eb" fontSize="13" fontWeight={isSelected ? "650" : "500"}>
                      {concept.label.length > 18 ? `${concept.label.slice(0, 17)}…` : concept.label}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="graph-status">
            <span><i /> {compactNumber(brain.counters.plasticityEvents)} plasticity events</span>
            <span>{compactNumber(Object.keys(brain.synapses).length)} visible synapses</span>
            <span>Updated live</span>
          </div>
        </section>
        <aside className="map-inspector">
          {selectedConcept ? (
            <>
              <div className="map-inspector__head">
                <span className="map-inspector__node"><i /></span>
                <span>
                  <small>SELECTED IDEA</small>
                  <h2>{selectedConcept.label}</h2>
                </span>
              </div>
              <div className="activation-score">
                <div style={{ "--score": `${selectedConcept.activation * 360}deg` } as React.CSSProperties}>
                  <span>{Math.round(selectedConcept.activation * 100)}</span>
                </div>
                <span>
                  <strong>Current activation</strong>
                  <small>{selectedConcept.activation > 0.8 ? "Highly active" : "Available"}</small>
                </span>
              </div>
              <dl className="inspector-stats">
                <div><dt>Importance</dt><dd>{Math.round(selectedConcept.importance * 100)}%</dd></div>
                <div><dt>Stability</dt><dd>{connectedSynapses.length ? `${Math.round(selectedStability * 100)}%` : "—"}</dd></div>
                <div><dt>Uncertainty</dt><dd>{Math.round(selectedConcept.uncertainty * 100)}%</dd></div>
                <div><dt>Exposures</dt><dd>{selectedConcept.exposures}</dd></div>
              </dl>
              <div className="panel-section">
                <div className="panel-section__head"><span>Strongest pathways</span><em>effective</em></div>
                <div className="pathway-list">
                  {connectedSynapses.slice(0, 4).map((synapse) => {
                    const otherId = synapse.sourceId === selectedConcept.id ? synapse.targetId : synapse.sourceId;
                    const concept = brain.concepts[otherId];
                    return (
                    <div key={synapse.id}>
                      <span><i /> {concept?.label ?? otherId}</span>
                      <em>{synapse.latentWeight >= 0 ? "+" : ""}{synapse.latentWeight.toFixed(2)}</em>
                    </div>
                  )})}
                  {!connectedSynapses.length ? <span className="pathway-empty">No synapses connect this idea yet.</span> : null}
                </div>
              </div>
              <div className="panel-section">
                <div className="panel-section__head"><span>Recent change</span><em>STDP</em></div>
                {recentSynapse ? (
                  <div className="change-note">
                    <Icon name="pulse" size={16} />
                    <span>
                      Latest connected synapse changed with <strong>{brain.concepts[recentSynapse.sourceId === selectedConcept.id ? recentSynapse.targetId : recentSynapse.sourceId]?.label ?? "another idea"}</strong>.
                      <small>{relativeTime(recentSynapse.lastUpdatedAt)} · latent {recentSynapse.latentWeight.toFixed(3)}</small>
                    </span>
                  </div>
                ) : <span className="pathway-empty">No plasticity event has been recorded for this idea.</span>}
              </div>
            </>
          ) : null}
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
          <div><dt>Novelty drive</dt><dd>{Math.round(brain.config.noveltyDrive * 100)}%</dd></div>
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
  const [freedom, setFreedom] = useState(0.68);
  const [resolution, setResolution] = useState(768);
  const [selectedSeedLabels, setSelectedSeedLabels] = useState(["memory", "rain", "growth"]);
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
          conceptIds: brain.workingMemory.map((item) => item.conceptId),
          settings: { resolution, freedom, seedLabels: selectedSeedLabels.join(",") }
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
                {mode === "vision" ? "Encoding selected image…" : `Imagining across ${brain.config.parallelThoughts} branches…`}
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
            <span>Seed idea</span>
            <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={5} />
            <small>
              <Icon name="brain" size={13} /> Text describes the starting idea; active memory can reshape it.
            </small>
          </label>
          <div className="idea-seeds">
            <span>Active memory influence</span>
            <div>
              {["memory", "rain", "growth", "identity", "violet light"].map((idea) => (
                <button
                  key={idea}
                  className={selectedSeedLabels.includes(idea) ? "is-active" : ""}
                  onClick={() =>
                    setSelectedSeedLabels((current) =>
                      current.includes(idea) ? current.filter((item) => item !== idea) : [...current, idea]
                    )
                  }
                >
                  {idea}
                </button>
              ))}
            </div>
          </div>
          <RangeField
            label="Internal freedom"
            detail="Distance from the seed idea"
            value={freedom}
            min={0}
            max={1}
            step={0.01}
            display={`${Math.round(freedom * 100)}%`}
            onChange={setFreedom}
          />
          <RangeField
            label="Resolution"
            detail="Scaled to available hardware"
            value={resolution}
            min={256}
            max={1024}
            step={128}
            display={`${resolution} px`}
            onChange={setResolution}
          />
          <Button
            kind="primary"
            icon={mode === "vision" ? "upload" : "sparkles"}
            disabled={generating || (mode !== "vision" && !prompt.trim())}
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
  "browser.automation": { label: "Browser snapshot", icon: "expand", action: "task", args: { url: "https://example.com" } },
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
      setResultText("Design preview only. Tool execution requires the Windows app runtime.");
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
                <span><dt>Evidence</dt><dd>+{preview.newEvidence}</dd></span>
                <span><dt>Files</dt><dd>+{preview.newFiles}</dd></span>
                <span><dt>File bytes</dt><dd>{formatBytes(preview.fileBytes)}</dd></span>
                <span><dt>Conflicts</dt><dd>{preview.conflicts.length}</dd></span>
              </dl>
              {preview.conflicts.length ? (
                <p><Icon name="warning" size={14} /> {preview.conflicts.join(" · ")}</p>
              ) : null}
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
  const [page, setPage] = useState<AppPage>("library");
  const [summaries, setSummaries] = useState<BrainSummary[]>(demo ? demoSummaries : []);
  const [activeBrain, setActiveBrain] = useState<BrainDocument | null>(null);
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>("chat");
  const [loading, setLoading] = useState(!demo);
  const [toast, setToast] = useState("");

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
              agents: ["agent.fork"]
            };
            return (protocolIds[toolId] ?? [toolId]).map((protocolId) => ({ toolId: protocolId, level }));
          })
        })
      : makeDemoBrain(`demo-${Date.now()}`, config.name, config);
    let initialSources = 0;
    if (window.omni && extras.initialTraining) {
      const results = await window.omni.data.ingestFiles({
        brainId: document.id,
        policy: "pretrain"
      });
      initialSources = results.length;
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
      initialSources > 0
        ? `${document.name} pretrained on ${initialSources} local source${initialSources === 1 ? "" : "s"} and is ready.`
        : `${document.name} has an immutable origin and is ready to learn.`
    );
  };

  const importBrain = async () => {
    if (!window.omni) {
      showToast("In Electron, this opens a verified .omni bundle from your Windows device.");
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
    setActiveBrain(document);
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

  return (
    <div className="app-shell">
      <div className="mica-glow mica-glow--one" />
      <div className="mica-glow mica-glow--two" />
      <Titlebar page={page} brain={activeBrain} demo={demo} onLibrary={() => setPage("library")} />
      {page === "library" ? (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
          onBuild={() => setPage("build")}
          onImport={() => void importBrain()}
          demo={demo}
        />
      ) : page === "build" ? (
        <BuildWizard onCancel={() => setPage("library")} onCreate={createBrain} />
      ) : activeBrain ? (
        <WorkspaceShell
          brain={activeBrain}
          view={workspaceView}
          onView={setWorkspaceView}
          onLibrary={() => setPage("library")}
          onBrainChange={updateActiveBrain}
          onToast={showToast}
        />
      ) : (
        <LibraryPage
          summaries={summaries}
          loading={loading}
          onOpen={(summary) => void openBrain(summary)}
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
