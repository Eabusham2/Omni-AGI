import { randomUUID } from "node:crypto";
import { chmod, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type {
  IdleCognitionScheduler,
  IdleCognitionSchedulerStatus
} from "./idleCognitionScheduler";

export interface BackgroundRuntimePreferences {
  schemaVersion: 1;
  keepActiveAfterWindowClose: boolean;
}

const DEFAULT_BACKGROUND_RUNTIME_PREFERENCES: BackgroundRuntimePreferences = {
  schemaVersion: 1,
  keepActiveAfterWindowClose: false
};

function normalizePreferences(value: unknown): BackgroundRuntimePreferences {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return { ...DEFAULT_BACKGROUND_RUNTIME_PREFERENCES };
  }
  const record = value as Record<string, unknown>;
  return {
    schemaVersion: 1,
    keepActiveAfterWindowClose:
      typeof record.keepActiveAfterWindowClose === "boolean"
        ? record.keepActiveAfterWindowClose
        : false
  };
}

/** App-owned operational preference; it never enters neural state or prompts. */
export class BackgroundRuntimePreferenceStore {
  private value = { ...DEFAULT_BACKGROUND_RUNTIME_PREFERENCES };

  constructor(private readonly path: string) {}

  async initialize(): Promise<BackgroundRuntimePreferences> {
    try {
      this.value = normalizePreferences(JSON.parse(await readFile(this.path, "utf8")));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
    return this.get();
  }

  get(): BackgroundRuntimePreferences {
    return { ...this.value };
  }

  async setKeepActiveAfterWindowClose(
    enabled: boolean
  ): Promise<BackgroundRuntimePreferences> {
    if (typeof enabled !== "boolean") {
      throw new Error("Background runtime preference must be a boolean.");
    }
    this.value = { schemaVersion: 1, keepActiveAfterWindowClose: enabled };
    await mkdir(dirname(this.path), { recursive: true });
    const temporary = `${this.path}.${randomUUID()}.tmp`;
    try {
      await writeFile(temporary, JSON.stringify(this.value), {
        encoding: "utf8",
        mode: 0o600
      });
      await chmod(temporary, 0o600).catch(() => undefined);
      await rename(temporary, this.path);
    } finally {
      await rm(temporary, { force: true }).catch(() => undefined);
    }
    return this.get();
  }
}

export interface BackgroundRuntimeView {
  mode: "window-open" | "background-resident" | "quitting";
  keepActiveAfterWindowClose: boolean;
  cognition: IdleCognitionSchedulerStatus;
  title: string;
  resourceStatus: string;
  permissionStatus: string;
  cancellable: boolean;
}

export interface BackgroundRuntimeCommands {
  openStudio(): void;
  setKeepActive(enabled: boolean): void;
  cancelBackground(): void;
  quitStudio(): void;
}

export interface BackgroundRuntimeTray {
  render(view: BackgroundRuntimeView, commands: BackgroundRuntimeCommands): void;
  destroy(): void;
}

export interface BackgroundRuntimeScheduler {
  start(): void;
  stop(): void;
  status(): IdleCognitionSchedulerStatus;
  onStatus(listener: (status: IdleCognitionSchedulerStatus) => void): () => void;
  cancelActive(): Promise<boolean>;
}

export interface BackgroundRuntimeControllerOptions {
  store: BackgroundRuntimePreferenceStore;
  scheduler: BackgroundRuntimeScheduler | IdleCognitionScheduler;
  tray: BackgroundRuntimeTray;
  hasVisibleWindows(): boolean;
  openStudio(): Promise<void> | void;
  requestAppQuit(): void;
  resourceStatus?(): string;
  permissionStatus?(): string;
}

function statusTitle(status: IdleCognitionSchedulerStatus): string {
  switch (status.phase) {
    case "running":
      return "Background cognition is running";
    case "cancelling":
      return "Background cognition is cancelling";
    case "paused-training":
      return "Paused for training";
    case "paused-foreground":
      return "Paused for foreground work";
    case "paused-initialization":
      return "Paused for initial learning";
    case "paused-inactive":
      return "Active Mode is off";
    case "backoff":
      return "Neural worker is recovering";
    case "startup-delay":
      return "Waiting for startup recovery";
    case "stopped":
      return "Background cognition is stopped";
    default:
      return "Background cognition is ready";
  }
}

/**
 * Owns desktop lifetime policy after the last Studio window closes.
 *
 * The neural scheduler remains responsible for duty limits and choosing an
 * organic cycle. This controller only keeps the process resident, exposes its
 * real state through the tray, and provides explicit cancellation and Quit.
 */
export class BackgroundRuntimeController {
  private preferences = { ...DEFAULT_BACKGROUND_RUNTIME_PREFERENCES };
  private initialized = false;
  private quitting = false;
  private disposeStatus?: () => void;

  constructor(private readonly options: BackgroundRuntimeControllerOptions) {}

  async initialize(): Promise<BackgroundRuntimeView> {
    if (!this.initialized) {
      this.preferences = await this.options.store.initialize();
      this.disposeStatus = this.options.scheduler.onStatus(() => this.render());
      this.options.scheduler.start();
      this.initialized = true;
    }
    return this.render();
  }

  /** Return true when the app must remain resident instead of calling quit. */
  handleLastWindowClosed(): boolean {
    if (this.quitting || !this.preferences.keepActiveAfterWindowClose) {
      return false;
    }
    this.render();
    return true;
  }

  windowOpened(): void {
    this.render();
  }

  snapshot(): BackgroundRuntimeView {
    const cognition = this.options.scheduler.status();
    return {
      mode: this.quitting
        ? "quitting"
        : this.options.hasVisibleWindows()
          ? "window-open"
          : "background-resident",
      keepActiveAfterWindowClose: this.preferences.keepActiveAfterWindowClose,
      cognition,
      title: statusTitle(cognition),
      resourceStatus:
        this.options.resourceStatus?.() ??
        `Idle neural work is capped at ${Math.round(cognition.maxDutyCycle * 100)}% duty; foreground and training preempt it.`,
      permissionStatus:
        this.options.permissionStatus?.() ??
        "External actions use each mind's saved Off, Ask, Auto, or Full Authority permissions.",
      cancellable: cognition.cancellable
    };
  }

  async setKeepActive(enabled: boolean): Promise<void> {
    this.preferences =
      await this.options.store.setKeepActiveAfterWindowClose(enabled);
    this.render();
    if (!enabled && !this.options.hasVisibleWindows()) {
      await this.quitExplicitly();
    }
  }

  async cancelBackground(): Promise<boolean> {
    const cancelled = await this.options.scheduler.cancelActive();
    this.render();
    return cancelled;
  }

  async quitExplicitly(): Promise<void> {
    if (this.quitting) return;
    this.quitting = true;
    this.options.scheduler.stop();
    await this.options.scheduler.cancelActive().catch(() => false);
    this.render();
    this.options.tray.destroy();
    this.options.requestAppQuit();
  }

  /** Cleanup for OS shutdown or an app-level fatal startup path. */
  dispose(): void {
    this.quitting = true;
    this.options.scheduler.stop();
    this.disposeStatus?.();
    this.disposeStatus = undefined;
    this.options.tray.destroy();
  }

  private render(): BackgroundRuntimeView {
    const view = this.snapshot();
    if (!this.quitting) {
      this.options.tray.render(view, {
        openStudio: () => {
          void Promise.resolve(this.options.openStudio()).then(() => this.windowOpened());
        },
        setKeepActive: (enabled) => {
          void this.setKeepActive(enabled);
        },
        cancelBackground: () => {
          void this.cancelBackground();
        },
        quitStudio: () => {
          void this.quitExplicitly();
        }
      });
    }
    return view;
  }
}
