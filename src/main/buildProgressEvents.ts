import type { BuildProgressEvent } from "../shared/types";
import type { EngineEvent } from "./engineSupervisor";

const ENGINE_BUILD_PHASES = new Set<BuildProgressEvent["phase"]>([
  "allocating",
  "language-foundation",
  "capability-curriculum",
  "action-policy",
  "modalities",
  "packing",
  "readiness",
  "complete"
]);

export interface BuildProgressTimingSample {
  startedAt: string;
  updatedAt: string;
  elapsedMs: number;
}

function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function enginePhase(event: EngineEvent): BuildProgressEvent["phase"] {
  const value = record(event.data) ? event.data.phase : undefined;
  return typeof value === "string" && ENGINE_BUILD_PHASES.has(value as BuildProgressEvent["phase"])
    ? value as BuildProgressEvent["phase"]
    : "allocating";
}

function boundedProgress(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.max(0, Math.min(1, value))
    : 0;
}

function engineMetrics(event: EngineEvent): BuildProgressEvent["data"] | undefined {
  const value = record(event.data) ? event.data.metrics : undefined;
  return record(value) ? value as BuildProgressEvent["data"] : undefined;
}

/**
 * Projects an engine-owned core-build event into host initialization progress.
 * The engine's `complete` phase means only that the native core is complete;
 * documentation/capability learning and the ready-document commit still follow.
 */
export function engineBuildProgressEvent(input: {
  event: EngineEvent;
  brainId?: string;
  sequence: number;
  fallbackLabel: string;
  timing: BuildProgressTimingSample;
}): BuildProgressEvent {
  const rawPhase = enginePhase(input.event);
  const coreComplete = rawPhase === "complete";
  const metrics = engineMetrics(input.event);
  const brainId = input.brainId ?? input.event.brainId;
  return {
    ...(brainId ? { brainId } : {}),
    ...(input.event.streamId ? { streamId: input.event.streamId } : {}),
    sequence: input.sequence,
    phase: coreComplete ? "readiness" : rawPhase,
    // No engine event can publish the host's durable terminal state.
    progress: coreComplete ? 0.99 : Math.min(0.99, boundedProgress(input.event.progress)),
    label: coreComplete
      ? "Native core initialized · finalizing documentation, capabilities, and durable readiness"
      : input.event.message ?? input.fallbackLabel,
    ...input.timing,
    ...(metrics ? { data: metrics } : {})
  };
}

/** Emitted only after BuildInitializationCoordinator returns a ready document. */
export function hostCommittedBuildProgressEvent(input: {
  brainId: string;
  sequence: number;
  timing: BuildProgressTimingSample;
}): BuildProgressEvent {
  return {
    brainId: input.brainId,
    sequence: input.sequence,
    phase: "complete",
    progress: 1,
    label: "OmniCortex ready · initialization durably committed",
    ...input.timing
  };
}
