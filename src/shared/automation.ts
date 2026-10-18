import type { ToolExecutionResult, ToolInvocation } from "./types";

export type AutomationJson = null | boolean | number | string | AutomationJson[] | { [key: string]: AutomationJson };
export interface AutomationReferenceValue {
  source: "input" | "event" | "step";
  path: Array<string | number>;
  stepId?: string;
}
/** References are interpreted only in the authored file, never inside incoming data. */
export type AutomationOperand = AutomationJson;
export interface AutomationCondition {
  reference: AutomationReferenceValue;
  operator: "exists" | "equals" | "not-equals" | "greater" | "at-least" | "less" | "at-most";
  value?: AutomationJson;
}
export type AutomationStep = {
  id: string;
  when?: AutomationCondition;
} & ({ kind: "wait"; delayMs: number } | {
  kind: "tool";
  /** A script uses the existing shell/code/plugin adapter and its permissions. */
  style?: "tool" | "script-delegate";
  toolId: string;
  action: string;
  arguments: Record<string, AutomationOperand>;
});
/** Portable, inert action file. It contains no owner, grant or activation token. */
export interface AutomationProgram {
  format: "omni-host-action-file";
  version: 1;
  name: string;
  description?: string;
  steps: AutomationStep[];
  /** Inert presentation only. Execution order is always the authored steps array. */
  canvas?: AutomationCanvas;
}
export interface AutomationCanvas {
  version: 1;
  mode: "grid" | "freeform";
  positions: Array<{ stepId: string; x: number; y: number }>;
}
export interface AutomationActivation {
  enabled: boolean;
  allowTriggers: boolean;
  ownerBrainId: string;
  ownerEpoch: string;
  catalogRevision: number;
  policyRevision: string;
  revisionSha256: string;
  changedAt: string;
}
export interface AutomationDefinition {
  id: string;
  brainId: string;
  revision: number;
  revisionSha256: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  activation?: AutomationActivation;
  inheritedFrom?: { brainId: string; definitionId: string };
}
/** An explicit Clock delivery, not a disguised prompt or notification. */
export interface ScheduledAutomationReference {
  definitionId: string;
  revisionSha256: string;
  inputs: Record<string, AutomationJson>;
}
export type AutomationRunState = "queued" | "running" | "waiting" | "approval-required" | "paused" | "review-required" | "stopping" | "stopped" | "complete" | "failed";
export interface AutomationRun {
  id: string;
  brainId: string;
  definitionId: string;
  revisionSha256: string;
  inputSha256: string;
  inputs: Record<string, AutomationJson>;
  state: AutomationRunState;
  nextStep: number;
  createdAt: string;
  updatedAt: string;
  activation: AutomationActivation;
  trigger?: { occurrenceId: string; eventId?: string; eventData?: AutomationJson };
  detail?: string;
  pausedFor?: "user" | "shutdown" | "copied-instance" | "recovery" | "resources";
  inheritedFrom?: { brainId: string; runId: string };
}
export interface AutomationStepReceipt {
  runId: string;
  stepId: string;
  index: number;
  attempt: number;
  requestId: string;
  state: "intent" | "waiting" | "approval-required" | "complete" | "failed" | "skipped" | "unknown" | "reviewed-skip";
  invocation?: ToolInvocation;
  invocationSha256?: string;
  startedAt: string;
  updatedAt: string;
  outcome?: ToolExecutionResult;
  wait?: { remainingMs: number; dueAt: string; sessionId: string; monotonicDeadline: number };
  review?: { at: string; reason: string; resolution: "skip" | "retry-confirmed"; acknowledgedPossibleEffects: boolean };
}
export interface AutomationPageQuery { after?: string; limit?: number }
export interface AutomationPage<T> { entries: T[]; total: number; nextAfter?: string }
export interface AutomationWorkspaceSnapshot {
  brainId: string;
  catalogRevision: number;
  counts: { definitions: number; activeDefinitions: number; runs: number; activeRuns: number; reviewRequired: number; approvals: number };
  hostExecution: true;
  processResidentRequired: true;
}
export interface AutomationWorkspaceEvent { brainId: string; type: "changed"; at: string }
export interface AutomationWorkspaceApi {
  snapshot(brainId: string): Promise<AutomationWorkspaceSnapshot>;
  definitions(brainId: string, query?: AutomationPageQuery): Promise<AutomationPage<AutomationDefinition>>;
  program(brainId: string, definitionId: string, revisionSha256?: string): Promise<{ definition: AutomationDefinition; program: AutomationProgram }>;
  runs(brainId: string, query?: AutomationPageQuery): Promise<AutomationPage<AutomationRun>>;
  steps(brainId: string, runId: string, query?: AutomationPageQuery): Promise<AutomationPage<AutomationStepReceipt>>;
  currentStep(brainId: string, runId: string): Promise<AutomationStepReceipt | null>;
  onEvent(listener: (event: AutomationWorkspaceEvent) => void): () => void;
}
