import type {
  BrainReadiness,
  BuildProgressEvent,
  RuntimeJob
} from "../../shared/types";
import {
  presentTrainingTelemetry,
  type TrainingTelemetryPresentation
} from "../../shared/trainingTelemetry";
import { conciseUiMessage } from "./uiPresentation";

export interface InitializationRecoveryPresentation {
  tone: "paused" | "error";
  title: string;
  detail: string;
  guidance: string;
  actionLabel: string;
  busyLabel: string;
}

export interface InitializationRunningPresentation {
  title: string;
  detail: string;
  guidance: string;
  progress?: number;
  percent?: number;
  telemetry?: TrainingTelemetryPresentation;
}

const RESUMABLE_LEARNING_PAUSE = /^(?:paused before\b|dataset learning (?:paused|ended before)\b|initial learning (?:was )?cancelled\b)/i;

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function count(value: unknown): number | undefined {
  return Number.isSafeInteger(value) && Number(value) >= 0
    ? Number(value)
    : undefined;
}

/**
 * A retry becomes visibly active before its awaited IPC call returns the
 * repository's new `initializing` document. Correlate that short renderer-only
 * interval to the retry's build stream so an unrelated mind can never inherit
 * its running surface.
 */
export function initializationIsRunning(
  brainId: string,
  readiness: BrainReadiness,
  retrying: boolean,
  event: BuildProgressEvent | null | undefined
): boolean {
  if (readiness.state === "initializing") return true;
  if (!retrying || !event) return false;
  const eventMatches = event.brainId === brainId || event.job?.brainId === brainId;
  const eventConflicts =
    (event.brainId !== undefined && event.brainId !== brainId) ||
    (event.job !== undefined && event.job.brainId !== brainId);
  return eventMatches && !eventConflicts;
}

/**
 * Resolve only the live ingestion/crawl job that belongs to this mind. Build
 * events are main-authoritative during cold recovery; the fallback covers the
 * short renderer-owned create/retry interval before the first build event.
 */
export function activeInitializationTrainingJob(
  brainId: string,
  event: BuildProgressEvent | null | undefined,
  fallback: RuntimeJob | null | undefined
): RuntimeJob | undefined {
  const eventMatches =
    event !== null &&
    event !== undefined &&
    (event.brainId === undefined || event.brainId === brainId);
  const candidates = [eventMatches ? event?.job : undefined, fallback];
  return candidates.find(
    (job): job is RuntimeJob =>
      job !== null &&
      job !== undefined &&
      job.brainId === brainId &&
      ["ingestion", "crawl"].includes(job.kind) &&
      ["queued", "running", "cancelling"].includes(job.state)
  );
}

/**
 * Presents only progress correlated to the active recovering brain. Startup
 * recovery runs without an invoking renderer request, so its RuntimeJob
 * snapshot arrives through the same build-event bridge used by an explicit
 * retry. Record/checkpoint counts come from the job output, not inferred from
 * the outer readiness polling loop.
 */
export function initializationRunningPresentation(
  brainId: string,
  event: BuildProgressEvent | null | undefined
): InitializationRunningPresentation {
  const fallback: InitializationRunningPresentation = {
    title: "Resuming initial learning",
    detail:
      "The main process is recovering the committed OmniCortex native core, dataset cursor, or web frontier.",
    guidance:
      "Chat and organic activity stay locked until every selected first-learning task completes."
  };
  if (
    !event ||
    (event.brainId !== undefined && event.brainId !== brainId) ||
    (event.job !== undefined && event.job.brainId !== brainId)
  ) {
    return fallback;
  }

  const jobOutput = record(event.job?.output);
  const jobProgress = record(jobOutput?.progress);
  const currentRecord = count(jobProgress?.currentRecord);
  const committedRecords = count(jobProgress?.committedRecords);
  const expectedRecords = count(jobProgress?.expectedRecords);
  const recordParts: string[] = [];
  if (currentRecord !== undefined && currentRecord > 0) {
    recordParts.push(
      expectedRecords !== undefined && expectedRecords > 0
        ? `Record ${currentRecord.toLocaleString()} of ${expectedRecords.toLocaleString()}`
        : `Record ${currentRecord.toLocaleString()}`
    );
  }
  if (committedRecords !== undefined && committedRecords > 0) {
    recordParts.push(`${committedRecords.toLocaleString()} committed`);
  }
  const progress = Math.max(
    0,
    Math.min(
      1,
      typeof event.job?.progress === "number" && Number.isFinite(event.job.progress)
        ? event.job.progress
        : event.progress
    )
  );
  const telemetry = event.job?.telemetry
    ? presentTrainingTelemetry(event.job.telemetry)
    : undefined;
  const cancellationAwaiting = event.job?.state === "cancelling";
  const cancellationNeedsRetry = cancellationAwaiting && Boolean(event.job?.error?.trim());
  return {
    title: cancellationNeedsRetry
      ? "Initial-learning stop needs retry"
      : cancellationAwaiting
        ? "Stopping initial learning"
        : event.job?.label || event.label || fallback.title,
    detail: cancellationNeedsRetry
      ? conciseUiMessage(
          event.job?.error,
          "The worker has not acknowledged this stop yet."
        )
      : cancellationAwaiting
        ? "Waiting for the exact training worker and its durable ledger to acknowledge termination."
        : recordParts.length > 0
          ? recordParts.join(" · ")
          : event.phase === "initial-materials"
            ? "Continuing from the last committed initial-learning cursor."
            : fallback.detail,
    guidance: cancellationAwaiting
      ? "No checkpoint or source is deleted. Keep this surface open; retry Stop only if an acknowledgement error appears."
      : fallback.guidance,
    progress,
    percent: Math.round(progress * 100),
    ...(telemetry ? { telemetry } : {})
  };
}

/**
 * Keep a recoverable dataset interruption distinct from a genuine neural-core,
 * checkpoint, selection, or configuration failure. The durable readiness
 * gate remains authoritative in both cases; this helper changes only truthful
 * user-facing copy and never infers that an unknown failure is safe.
 */
export function initializationRecoveryPresentation(
  readiness: BrainReadiness
): InitializationRecoveryPresentation {
  const failure = readiness.failure;
  const rawMessage = failure?.message ?? "";
  const learningPaused =
    failure?.phase === "initial-learning" &&
    RESUMABLE_LEARNING_PAUSE.test(rawMessage.trim());

  if (learningPaused) {
    return {
      tone: "paused",
      title: "Learning paused",
      detail: conciseUiMessage(
        rawMessage,
        "Learning paused before the current source could be committed."
      ),
      guidance:
        "Chat stays locked until learning finishes. Resume continues from the committed dataset cursor or web frontier without repeating learned records.",
      actionLabel: "Resume training",
      busyLabel: "Resuming training…"
    };
  }

  const foundationFailure = failure?.phase !== "initial-learning";
  return {
    tone: "error",
    title: foundationFailure ? "Setup failed" : "Initial learning failed",
    detail: conciseUiMessage(
      rawMessage,
      foundationFailure
        ? "The OmniCortex native core could not be completed."
        : "A selected first-learning source could not be completed."
    ),
    guidance: foundationFailure
      ? "Chat remains locked. Retry rebuilds or verifies the OmniCortex native core; use the operational trace if the same error returns."
      : "Chat remains locked. Retry uses the saved recovery plan; a permanent source problem may need to be fixed before retrying.",
    actionLabel: foundationFailure ? "Retry setup" : "Retry initial learning",
    busyLabel: "Retrying…"
  };
}
