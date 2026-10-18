import type { InstanceAutoLoadRequest, InstanceInheritedExecutions, InstanceLifecycleState, InstanceLoadRequest, InstanceReconnectRequirement,
  InstanceResumeFrontier, InstanceUnloadRequest } from "./instanceLifecycle";
import type { WorkingMemoryResourcePlan } from "./types";
import type { ChatResult, StructuredAction } from "./types";
import type { InheritedTaskFilesApi } from "./inheritedTaskFiles";

/** A no-load native observation. The native checkpoint remains authoritative. */
export interface NativeInstanceStatus {
  brainId: string;
  learnedRuntimeResident: boolean;
  minimumContextTokens: number;
  resume?: InstanceResumeFrontier;
  resumeCompatibility?: { compatible: boolean; reason?: string };
  inheritedExecutions?: InstanceInheritedExecutions;
  waitingFor?: InstanceLifecycleState["waitingFor"];
  reconnect: InstanceReconnectRequirement[];
  unloaded?: boolean;
  presentation?: { input: string; publishedText: string; actions: unknown[]; inputReceipt?: unknown };
}

export interface InstanceLoadFit {
  brainId: string;
  checkedAt: string;
  allowed: boolean;
  minimumContextTokens: number;
  blockers: string[];
  plan?: WorkingMemoryResourcePlan;
}

export interface InstanceResumeAttachment {
  brainId: string;
  frontier: InstanceResumeFrontier;
  input: string;
  publishedText: string;
  hostAttached: boolean;
  /** Historical proposals are context/provenance, never new effect authority. */
  priorActions: StructuredAction[];
  /** Only the authenticated connected queue can supply this routing identity. */
  inheritedExecutionId?: string;
  nativeAudioInput?: import("./nativeAudioChat").NativeAudioMessageInfo;
}

export interface NativeInheritedPresentation {
  brainId: string; executionId: string; parentId: string; originalTurnId: string; rootHumanMessageId: string;
  frontier: InstanceResumeFrontier;
  presentation: { input: string; publishedText: string; actions: unknown[]; inputReceipt?: unknown };
}

export interface InstanceLifecycleApi extends InheritedTaskFilesApi {
  lifecycle(brainId: string): Promise<InstanceLifecycleState>;
  loadFit(brainId: string): Promise<InstanceLoadFit>;
  load(request: InstanceLoadRequest): Promise<InstanceLifecycleState>;
  unload(request: InstanceUnloadRequest): Promise<InstanceLifecycleState>;
  setAutoLoad(request: InstanceAutoLoadRequest): Promise<InstanceLifecycleState>;
  resumePresentation(brainId: string): Promise<InstanceResumeAttachment | null>;
  resumeTurn(request: { brainId: string; checkpointId: string }): Promise<ChatResult>;
  inheritedResumePresentation(brainId: string, executionId: string): Promise<InstanceResumeAttachment>;
  resumeInheritedTurn(request: { brainId: string; executionId: string }): Promise<ChatResult>;
  stopSavedTurn(request: { brainId: string; turnId: string; checkpointId?: string; executionId?: string }): Promise<InstanceLifecycleState>;
  onLifecycle(listener: (state: InstanceLifecycleState) => void): () => void;
}
