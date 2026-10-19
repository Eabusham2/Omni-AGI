import type { BrainStorageOperationEvent } from "../../src/shared/types";

/** Test-only inert transport; never a production preload capability. */
export interface WholeAppCompositionControl {
  status(): { format: "synthetic-whole-app-composition"; id: string; pendingTurns: string[];
    streamSubscribers: number; buildSubscribers: number;
    receipts: Array<{ operation: string; turnId?: string; detail?: string }>;
    denied: string[]; models: 0; hostEffects: 0 };
  start(): void;
  openingText(): void;
  ponder(): Promise<void>;
  toolDuringPonder(): Promise<void>;
  phaseOnlyUnboundAction(): void;
  requestToolApproval(): void;
  resumeText(): void;
  finish(turnId?: string): Promise<void>;
  fail(): void;
  training(phase: "extracting" | "learning" | "saving" | "finalizing" | "complete"): void;
  storage(state: BrainStorageOperationEvent["state"]): void;
  notification(): void;
}

declare global { interface Window { omniWholeAppComposition?: WholeAppCompositionControl } }
