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
  minimumIdleSeconds?: number;
  /**
   * Maximum fraction of wall time that prompt-free cognition may occupy.
   * This is hardware policy, not a personality or curiosity control.
   */
  maxDutyCycle?: number;
  now?: () => number;
  onError?: (error: Error) => void;
}

/**
 * Round-robin desktop scheduler for prompt-free cognition.
 *
 * It processes at most one brain per tick, never overlaps ticks, and delegates
 * every proposed external action to ChatActionController's normal permission
 * and audit path.
 */
export class IdleCognitionScheduler {
  private timer?: NodeJS.Timeout;
  private ticking = false;
  private cursor = 0;
  private readonly intervalMs: number;
  private readonly minimumIdleSeconds: number;
  private readonly maxDutyCycle: number;
  private readonly now: () => number;
  private readonly nextEligibleAt = new Map<string, number>();

  constructor(
    private readonly repository: IdleBrainRepository,
    private readonly actions: IdleActionController,
    options: IdleCognitionSchedulerOptions = {}
  ) {
    this.intervalMs = Math.max(1_000, Math.round(options.intervalMs ?? 12_000));
    this.minimumIdleSeconds = Math.max(
      0,
      Math.min(86_400, options.minimumIdleSeconds ?? 6)
    );
    this.maxDutyCycle = Math.max(
      0.01,
      Math.min(0.5, options.maxDutyCycle ?? 0.12)
    );
    this.now = options.now ?? Date.now;
    this.onError = options.onError;
  }

  private readonly onError?: (error: Error) => void;

  start(): void {
    if (this.timer) return;
    // The mind is live as soon as the desktop starts. Subsequent cycles are
    // duty-capped below, so prompt-free activity cannot monopolize the host.
    void this.tick();
    this.timer = setInterval(() => void this.tick(), this.intervalMs);
    this.timer.unref();
  }

  stop(): void {
    if (!this.timer) return;
    clearInterval(this.timer);
    this.timer = undefined;
  }

  async tick(): Promise<IdleCycleResult | undefined> {
    if (this.ticking) return undefined;
    this.ticking = true;
    try {
      const brains = await this.repository.list();
      if (brains.length === 0) return undefined;
      for (let attempts = 0; attempts < brains.length; attempts += 1) {
        const index = (this.cursor + attempts) % brains.length;
        const summary = brains[index]!;
        const brain = await this.repository.get(summary.id);
        if (!brain.config.idleCognition) continue;
        if (this.actions.isBusy?.(brain.id) === true) continue;
        if ((this.nextEligibleAt.get(brain.id) ?? 0) > this.now()) continue;
        this.cursor = (index + 1) % brains.length;
        const startedAt = this.now();
        const result = await this.actions.idle(
          brain.id,
          this.minimumIdleSeconds
        );
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
        return result;
      }
      this.cursor = (this.cursor + 1) % brains.length;
      return undefined;
    } catch (error) {
      this.onError?.(error instanceof Error ? error : new Error(String(error)));
      return undefined;
    } finally {
      this.ticking = false;
    }
  }
}
