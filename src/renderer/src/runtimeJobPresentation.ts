import type { RuntimeJob } from "../../shared/types";

export interface RuntimeJobStopPresentation {
  active: boolean;
  requestAllowed: boolean;
  waitingForAcknowledgement: boolean;
  retryAvailable: boolean;
  label: "Cancel safely" | "Waiting for stop…" | "Retry stop";
}

export interface RuntimeJobProgressPresentation {
  determinate: boolean;
  fraction: number | null;
  percent: number | null;
  label: string;
  showCoverage: boolean;
  showTelemetry: boolean;
  statusText: string;
}

/**
 * A running worker is not necessarily traversing data yet. Cold substrate
 * validation can take meaningful time, but it has no record denominator and
 * therefore must not manufacture a percentage, coverage, speed, or ETA.
 */
export function runtimeJobProgressPresentation(
  job: RuntimeJob | null | undefined
): RuntimeJobProgressPresentation {
  const preparing = Boolean(
    job &&
    job.kind === "ingestion" &&
    job.phase === "preparing-neural-substrate" &&
    !["complete"].includes(job.state)
  );
  if (preparing) {
    const statusText = job?.state === "cancelled"
      ? "No dataset traversal was measured before cancellation"
      : job?.state === "failed"
        ? "No dataset traversal was measured before the job failed"
        : "Waiting for the first measured dataset traversal";
    return {
      determinate: false,
      fraction: null,
      percent: null,
      label: job?.label ?? "Preparing neural substrate",
      showCoverage: false,
      showTelemetry: false,
      statusText
    };
  }
  const fraction = Math.max(0, Math.min(1, job?.progress ?? 0));
  return {
    determinate: true,
    fraction,
    percent: Math.round(fraction * 100),
    label: job?.label ?? "Waiting",
    showCoverage: true,
    showTelemetry: true,
    statusText: `${Math.round(fraction * 100)}% complete`
  };
}

/**
 * Coverage shown beside a dataset job must belong to that job. Falling back
 * to the latest completed manifest while a newer ingestion or crawl is
 * selected can make a cancelled 0/1 traversal look like an older 1/1 success.
 * A job with no measured traversal therefore presents no coverage instead of
 * borrowing another manifest's durable summary.
 */
export function runtimeJobScopedDatasetCoverage<T>(
  job: RuntimeJob | null | undefined,
  jobCoverage: T | null | undefined,
  latestCompletedCoverage: T | null | undefined
): T | null {
  if (job && (job.kind === "ingestion" || job.kind === "crawl")) {
    return jobCoverage ?? null;
  }
  return latestCompletedCoverage ?? null;
}

/**
 * Cancellation is an acknowledged transition, not an instantaneous terminal
 * state. A failed acknowledgement leaves the job in `cancelling` with an
 * error, where an explicit retry is safe; an in-flight acknowledgement must
 * disable duplicate Stop requests.
 */
export function runtimeJobStopPresentation(
  job: RuntimeJob | null | undefined
): RuntimeJobStopPresentation {
  if (!job || !["queued", "running", "cancelling"].includes(job.state)) {
    return {
      active: false,
      requestAllowed: false,
      waitingForAcknowledgement: false,
      retryAvailable: false,
      label: "Cancel safely"
    };
  }
  if (job.state !== "cancelling") {
    return {
      active: true,
      requestAllowed: true,
      waitingForAcknowledgement: false,
      retryAvailable: false,
      label: "Cancel safely"
    };
  }
  const retryAvailable = Boolean(job.error?.trim());
  return {
    active: true,
    requestAllowed: retryAvailable,
    waitingForAcknowledgement: !retryAvailable,
    retryAvailable,
    label: retryAvailable ? "Retry stop" : "Waiting for stop…"
  };
}

export function runtimeJobIsActive(
  job: RuntimeJob | null | undefined
): job is RuntimeJob {
  return runtimeJobStopPresentation(job).active;
}
