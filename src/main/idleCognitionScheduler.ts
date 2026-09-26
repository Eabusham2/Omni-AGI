import type {
  BrainDocument,
  BrainSummary,
  IdleCycleResult
} from "../shared/types";

export interface IdleBrainRepository {
  list(): Promise<BrainSummary[]>;
  get(id: string): Promise<BrainDocument>;
}

export interface IdleActionController {
  idle(brainId: string, minimumIdleSeconds?: number): Promise<IdleCycleResult>;
  isBusy?(brainId: string): boolean;
}

export interface IdleCognitionSchedulerOptions {
  intervalMs?: number;
  /**
   * Delay before the first prompt-free cycle after desktop startup.
   *
   * Starting immediately can make a cold, multi-gigabyte checkpoint load win
   * the shared serial worker before initialization recovery or a user-started
   * Build has had a chance to claim it.
   */
  startupDelayMs?: number;
  minimumIdleSeconds?: number;
  /**
   * Maximum fraction of wall time that prompt-free cognition may occupy.
   * This is hardware policy, not a personality or curiosity control.
   */
  maxDutyCycle?: number;
  /** Quiet window in which an immediate user retry outranks idle reload. */
  postCancelCooldownMs?: number;
  now?: () => number;
  /** True while neural/data/modality learning owns this instance. */
  isLearning?: (brainId: string) => boolean;
  /** True while foreground initialization owns the shared neural worker. */
  isInitializationBusy?: () => boolean;
  /**
   * Interrupt the optional neural request without cancelling foreground chat,
   * tools, or durable learning. EngineSupervisor.claimForeground() supplies
   * this in the desktop host.
   */
  preemptBackground?: () => Promise<void> | void;
  onError?: (error: Error) => void;
}

export type IdleCognitionSchedulerPhase =
  | "stopped"
  | "startup-delay"
  | "waiting"
  | "running"
  | "cancelling"
  | "paused-training"
  | "paused-foreground"
  | "paused-initialization"
  | "paused-inactive"
  | "backoff";

export interface IdleCognitionSchedulerStatus {
  phase: IdleCognitionSchedulerPhase;
  detail: string;
  activeBrainId?: string;
  /** Persisted identity allowed to acquire optional prompt-free work. */
  activeModeBrainId?: string;
  activeSince?: number;
  lastCycleAt?: number;
  nextEligibleAt?: number;
  cancellable: boolean;
  /** Resource policy, not a neural/personality setting. */
  maxDutyCycle: number;
  updatedAt: number;
}

export type IdleCognitionStatusListener = (
  status: IdleCognitionSchedulerStatus
) => void;

/**
 * Single-lease desktop scheduler for prompt-free cognition.
 *
 * It processes only the persisted Active Mode owner, never overlaps ticks, and delegates
 * every proposed external action to ChatActionController's normal permission
 * and audit path.
 */
export class IdleCognitionScheduler {
  private timer?: NodeJS.Timeout;
  private startupTimer?: NodeJS.Timeout;
  private ticking = false;
  private readonly intervalMs: number;
  private readonly startupDelayMs: number;
  private readonly minimumIdleSeconds: number;
  private readonly maxDutyCycle: number;
  private readonly postCancelCooldownMs: number;
  private readonly now: () => number;
  private readonly isLearning?: (brainId: string) => boolean;
  private readonly isInitializationBusy?: () => boolean;
  private readonly preemptBackground?: () => Promise<void> | void;
  private readonly nextEligibleAt = new Map<string, number>();
  private readonly statusListeners = new Set<IdleCognitionStatusListener>();
  private consecutiveFailures = 0;
  private failureBackoffUntil = 0;
  private activeBrainId?: string;
  private activeModeBrainId?: string;
  private activeSince?: number;
  private lastCycleAt?: number;
  private currentStatus: IdleCognitionSchedulerStatus;

  constructor(
    private readonly repository: IdleBrainRepository,
    private readonly actions: IdleActionController,
    options: IdleCognitionSchedulerOptions = {}
  ) {
    this.intervalMs = Math.max(1_000, Math.round(options.intervalMs ?? 12_000));
    this.startupDelayMs = Math.max(
      0,
      Math.round(options.startupDelayMs ?? Math.max(30_000, this.intervalMs))
    );
    this.minimumIdleSeconds = Math.max(
      0,
      Math.min(86_400, options.minimumIdleSeconds ?? 6)
    );
    this.maxDutyCycle = Math.max(
      0.01,
      Math.min(0.5, options.maxDutyCycle ?? 0.12)
    );
    this.postCancelCooldownMs = Math.max(
      1_000,
      Math.round(
        options.postCancelCooldownMs ?? Math.max(60_000, this.intervalMs * 3)
      )
    );
    this.now = options.now ?? Date.now;
    this.isLearning = options.isLearning;
    this.isInitializationBusy = options.isInitializationBusy;
    this.preemptBackground = options.preemptBackground;
    this.onError = options.onError;
    this.currentStatus = {
      phase: "stopped",
      detail: "Background cognition is stopped.",
      cancellable: false,
      maxDutyCycle: this.maxDutyCycle,
      updatedAt: this.now()
    };
  }

  private readonly onError?: (error: Error) => void;

  start(): void {
    if (this.timer || this.startupTimer) return;
    // Startup/recovery and a user-started Build have priority over optional
    // prompt-free work. Once the grace period has elapsed, the normal
    // duty-capped cadence keeps the mind active without monopolizing the host.
    this.updateStatus("startup-delay", "Waiting for startup and recovery work.");
    this.startupTimer = setTimeout(() => {
      this.startupTimer = undefined;
      this.updateStatus("waiting", "Ready for an organic idle cycle.");
      void this.tick();
      this.timer = setInterval(() => void this.tick(), this.intervalMs);
      this.timer.unref();
    }, this.startupDelayMs);
    this.startupTimer.unref();
  }

  stop(): void {
    if (this.startupTimer) {
      clearTimeout(this.startupTimer);
      this.startupTimer = undefined;
    }
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
    this.updateStatus("stopped", "Background cognition is stopped.");
  }

  status(): IdleCognitionSchedulerStatus {
    return { ...this.currentStatus };
  }

  onStatus(listener: IdleCognitionStatusListener): () => void {
    this.statusListeners.add(listener);
    listener(this.status());
    return () => this.statusListeners.delete(listener);
  }

  /**
   * Cancel only the currently running optional idle request. A cooldown keeps
   * it from immediately reacquiring the worker after the user cancels it.
   */
  async cancelActive(): Promise<boolean> {
    const brainId = this.activeBrainId;
    if (!brainId || !this.ticking) return false;
    this.reserveForegroundAfterCancel(brainId);
    this.updateStatus(
      "cancelling",
      "Cancelling optional background cognition.",
      brainId,
      true
    );
    if (!this.preemptBackground) return false;
    await this.preemptBackground();
    return true;
  }

  private updateStatus(
    phase: IdleCognitionSchedulerPhase,
    detail: string,
    activeBrainId?: string,
    cancellable = false,
    nextEligibleAt?: number
  ): void {
    const next: IdleCognitionSchedulerStatus = {
      phase,
      detail,
      ...(activeBrainId ? { activeBrainId } : {}),
      ...(this.activeModeBrainId
        ? { activeModeBrainId: this.activeModeBrainId }
        : {}),
      ...(activeBrainId && this.activeSince !== undefined
        ? { activeSince: this.activeSince }
        : {}),
      ...(this.lastCycleAt !== undefined ? { lastCycleAt: this.lastCycleAt } : {}),
      ...(nextEligibleAt !== undefined ? { nextEligibleAt } : {}),
      cancellable,
      maxDutyCycle: this.maxDutyCycle,
      updatedAt: this.now()
    };
    this.currentStatus = next;
    for (const listener of this.statusListeners) listener({ ...next });
  }

  reserveForegroundAfterCancel(
    brainId: string,
    cooldownMs = this.postCancelCooldownMs
  ): void {
    const id = brainId.trim();
    if (!id) return;
    const eligibleAt = this.now() + Math.max(1_000, Math.round(cooldownMs));
    this.nextEligibleAt.set(
      id,
      Math.max(this.nextEligibleAt.get(id) ?? 0, eligibleAt)
    );
  }

  async tick(): Promise<IdleCycleResult | undefined> {
    if (this.ticking) return undefined;
    if (this.failureBackoffUntil > this.now()) {
      this.updateStatus(
        "backoff",
        "Neural worker recovery is cooling down.",
        undefined,
        false,
        this.failureBackoffUntil
      );
      return undefined;
    }
    if (this.isInitializationBusy?.() === true) {
      this.updateStatus(
        "paused-initialization",
        "Initial learning has priority over optional cognition."
      );
      return undefined;
    }
    this.ticking = true;
    try {
      const brains = await this.repository.list();
      if (brains.length === 0) {
        this.activeModeBrainId = undefined;
        this.updateStatus("waiting", "No ready minds are available.");
        return undefined;
      }
      const documents = await Promise.all(
        brains.map((summary) => this.repository.get(summary.id))
      );
      // The persisted API normally leaves exactly one enabled identity. This
      // selection is a defensive process-level lease: even a legacy/corrupt
      // set containing several true flags can never round-robin and mutate
      // more than the newest listed owner.
      const leasedIndex = documents.findIndex(
        (brain) => brain.config.idleCognition === true
      );
      if (leasedIndex < 0) {
        this.activeModeBrainId = undefined;
        this.updateStatus(
          "paused-inactive",
          "Active Mode is off for every mind."
        );
        return undefined;
      }
      const leasedBrain = documents[leasedIndex]!;
      this.activeModeBrainId = leasedBrain.id;
      let deferredPhase: IdleCognitionSchedulerPhase = "waiting";
      let deferredDetail = "Waiting for the next duty-capped idle cycle.";
      let earliestEligibleAt: number | undefined;
      for (const brain of [leasedBrain]) {
        // This persisted marker is authoritative. Unlike RuntimeJobManager's
        // live job set, it survives a renderer reload or full app restart and
        // is present while the foundation itself is still being constructed.
        if (brain.readiness && brain.readiness.state !== "ready") {
          const eligibleAt =
            this.now() + this.minimumIdleSeconds * 1_000;
          this.nextEligibleAt.set(
            brain.id,
            eligibleAt
          );
          earliestEligibleAt = Math.min(earliestEligibleAt ?? eligibleAt, eligibleAt);
          deferredPhase = "paused-initialization";
          deferredDetail = "Initial learning has priority over optional cognition.";
          continue;
        }
        if (this.isLearning?.(brain.id) === true) {
          // Training is neural activity, but it is not a voluntary decision
          // to Ponder or act. Keep a post-training quiet interval so a cycle
          // that was skipped during learning cannot immediately surface when
          // the job releases the worker.
          const eligibleAt =
            this.now() + this.minimumIdleSeconds * 1_000;
          this.nextEligibleAt.set(
            brain.id,
            eligibleAt
          );
          earliestEligibleAt = Math.min(earliestEligibleAt ?? eligibleAt, eligibleAt);
          deferredPhase = "paused-training";
          deferredDetail = "Training and media learning have priority.";
          continue;
        }
        if (this.actions.isBusy?.(brain.id) === true) {
          deferredPhase = "paused-foreground";
          deferredDetail = "Chat, tools, or approvals have priority.";
          continue;
        }
        const eligibleAt = this.nextEligibleAt.get(brain.id) ?? 0;
        if (eligibleAt > this.now()) {
          earliestEligibleAt = Math.min(earliestEligibleAt ?? eligibleAt, eligibleAt);
          continue;
        }
        const startedAt = this.now();
        this.activeBrainId = brain.id;
        this.activeSince = startedAt;
        this.updateStatus(
          "running",
          "An organic prompt-free neural cycle is running.",
          brain.id,
          true
        );
        const result = await this.actions.idle(
          brain.id,
          this.minimumIdleSeconds
        );
        this.consecutiveFailures = 0;
        this.failureBackoffUntil = 0;
        const completedAt = this.now();
        const occupiedMs = Math.max(1, completedAt - startedAt);
        const dutyCooldownMs = Math.max(
          this.intervalMs,
          occupiedMs * ((1 - this.maxDutyCycle) / this.maxDutyCycle)
        );
        const workerCooldownMs = Math.max(
          0,
          Number(result.retryAfterSeconds ?? 0) * 1_000
        );
        this.nextEligibleAt.set(
          brain.id,
          completedAt + Math.max(dutyCooldownMs, workerCooldownMs)
        );
        this.lastCycleAt = completedAt;
        this.activeBrainId = undefined;
        this.activeSince = undefined;
        const nextAt = this.nextEligibleAt.get(brain.id);
        this.updateStatus(
          result.reason === "foreground-work"
            ? "paused-foreground"
            : "waiting",
          result.reason === "foreground-work"
            ? "Foreground work preempted optional cognition."
            : "Waiting for the next duty-capped idle cycle.",
          undefined,
          false,
          nextAt
        );
        return result;
      }
      this.updateStatus(
        deferredPhase,
        deferredDetail,
        undefined,
        false,
        earliestEligibleAt
      );
      return undefined;
    } catch (error) {
      this.consecutiveFailures += 1;
      const failureCooldownMs = Math.min(
        5 * 60_000,
        this.intervalMs * 2 ** Math.min(8, this.consecutiveFailures - 1)
      );
      this.failureBackoffUntil = this.now() + failureCooldownMs;
      this.activeBrainId = undefined;
      this.activeSince = undefined;
      this.updateStatus(
        "backoff",
        "Neural worker recovery is cooling down.",
        undefined,
        false,
        this.failureBackoffUntil
      );
      // Report the first failure and then only exponentially spaced recovery
      // attempts. A stopped/restarting worker must not create a 12-second log
      // storm or monopolize startup while the same persistent state recovers.
      this.onError?.(error instanceof Error ? error : new Error(String(error)));
      return undefined;
    } finally {
      this.activeBrainId = undefined;
      this.activeSince = undefined;
      this.ticking = false;
    }
  }
}
