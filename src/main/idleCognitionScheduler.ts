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
}

export interface IdleCognitionSchedulerOptions {
  intervalMs?: number;
  minimumIdleSeconds?: number;
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

  constructor(
    private readonly repository: IdleBrainRepository,
    private readonly actions: IdleActionController,
    options: IdleCognitionSchedulerOptions = {}
  ) {
    this.intervalMs = Math.max(10_000, Math.round(options.intervalMs ?? 60_000));
    this.minimumIdleSeconds = Math.max(
      0,
      Math.min(86_400, options.minimumIdleSeconds ?? 45)
    );
    this.onError = options.onError;
  }

  private readonly onError?: (error: Error) => void;

  start(): void {
    if (this.timer) return;
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
        this.cursor = (index + 1) % brains.length;
        return await this.actions.idle(
          brain.id,
          this.minimumIdleSeconds
        );
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
