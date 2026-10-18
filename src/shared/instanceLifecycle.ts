/** Explicit learned-runtime lifecycle. Suspension is not chat cancellation. */
export type InstanceLifecyclePhase =
  | "unloaded" | "checking" | "loading" | "loaded"
  | "suspending" | "suspended" | "resuming" | "error";

export interface InstanceResumeFrontier {
  checkpointId: string;
  turnId: string;
  inputSha256: string;
  lastStreamSequence: number;
  sampledTokens: number;
  publishedTokens: number;
  phase: "native-text-token-boundary";
  exactSupportedResume: true;
}

export interface InstanceReconnectRequirement {
  id: string;
  kind: "microphone" | "camera" | "screen" | "browser" | "remote-session" | "tool";
  label: string;
  reason: string;
  permissionRequired: boolean;
}

/** Exact ancestral frontiers execute through their registered regional owners. */
export interface InstanceInheritedExecutions {
  format: "omni-connected-execution-queue";
  pending: number;
  rootMutationHeld: boolean;
  activeExecutionId: string | null;
  minimumContextTokens: number;
  entries: Array<{ id: string; turnId: string; parentId: string;
    status: "queued" | "suspended" | "complete" | "stopped";
    backend: string; publishedText: string }>;
}

export interface InstanceLifecycleState {
  brainId: string;
  phase: InstanceLifecyclePhase;
  /** Off until the user explicitly opts in. */
  autoLoad: boolean;
  revision: number;
  updatedAt: string;
  learnedRuntimeResident: boolean;
  resume?: InstanceResumeFrontier;
  resumeCompatibility?: { compatible: boolean; reason?: string };
  inheritedExecutions?: InstanceInheritedExecutions;
  pausedJobIds: string[];
  reconnect: InstanceReconnectRequirement[];
  /** Waiting for a named safe boundary is not an unloaded success. */
  waitingFor?: { kind: "native-boundary" | "external-effect" | "checkpoint" | "resources"; label: string };
  error?: { code: string; message: string; recoverable: boolean };
}

export interface InstanceLoadRequest { brainId: string; expectedCheckpointId?: string; }
export interface InstanceUnloadRequest { brainId: string; }
export interface InstanceAutoLoadRequest { brainId: string; enabled: boolean; }
