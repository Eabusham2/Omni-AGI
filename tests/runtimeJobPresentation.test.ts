import { describe, expect, it } from "vitest";
import {
  runtimeJobIsActive,
  runtimeJobProgressPresentation,
  runtimeJobScopedDatasetCoverage,
  runtimeJobStopPresentation
} from "../src/renderer/src/runtimeJobPresentation";
import type { RuntimeJob } from "../src/shared/types";

function job(
  state: RuntimeJob["state"],
  error?: string
): RuntimeJob {
  return {
    id: "job-1",
    brainId: "brain-1",
    kind: "training",
    state,
    progress: 0.4,
    label: "Training",
    createdAt: "2026-09-12T00:00:00.000Z",
    updatedAt: "2026-09-12T00:00:01.000Z",
    ...(error ? { error } : {})
  };
}

describe("runtime job cancellation presentation", () => {
  it("keeps cold substrate preparation indeterminate until traversal begins", () => {
    const preparing = {
      ...job("running"),
      kind: "ingestion" as const,
      progress: 0,
      label: "Preparing neural substrate",
      phase: "preparing-neural-substrate" as const
    };
    expect(runtimeJobProgressPresentation(preparing)).toEqual({
      determinate: false,
      fraction: null,
      percent: null,
      label: "Preparing neural substrate",
      showCoverage: false,
      showTelemetry: false,
      statusText: "Waiting for the first measured dataset traversal"
    });

    expect(runtimeJobProgressPresentation({
      ...preparing,
      progress: 0.25,
      label: "Epoch 1/1 · record 1/4",
      phase: "traversing-dataset"
    })).toMatchObject({
      determinate: true,
      fraction: 0.25,
      percent: 25,
      showCoverage: true,
      showTelemetry: true
    });
  });

  it("keeps an acknowledged cancellation visibly active and blocks duplicate Stop", () => {
    expect(runtimeJobStopPresentation(job("cancelling"))).toEqual({
      active: true,
      requestAllowed: false,
      waitingForAcknowledgement: true,
      retryAvailable: false,
      label: "Waiting for stop…"
    });
    expect(runtimeJobIsActive(job("cancelling"))).toBe(true);
  });

  it("offers an explicit retry only after acknowledgement failed", () => {
    expect(runtimeJobStopPresentation(
      job("cancelling", "Worker acknowledgement timed out.")
    )).toMatchObject({
      active: true,
      requestAllowed: true,
      waitingForAcknowledgement: false,
      retryAvailable: true,
      label: "Retry stop"
    });
  });

  it("does not present settled jobs as active or stoppable", () => {
    expect(runtimeJobIsActive(job("cancelled"))).toBe(false);
    expect(runtimeJobStopPresentation(job("complete"))).toMatchObject({
      active: false,
      requestAllowed: false
    });
  });

  it("never borrows stale completed coverage for a newer cancelled dataset job", () => {
    const cancelled = {
      ...job("cancelled"),
      kind: "ingestion" as const,
      progress: 0,
      label: "ingestion cancelled",
      phase: "preparing-neural-substrate" as const
    };
    const staleCompletedCoverage = {
      manifestId: "older-complete-manifest",
      discoveredRecords: 1,
      processedRecords: 1,
      complete: true
    };

    expect(runtimeJobScopedDatasetCoverage(
      cancelled,
      null,
      staleCompletedCoverage
    )).toBeNull();
    expect(runtimeJobProgressPresentation(cancelled)).toMatchObject({
      determinate: false,
      percent: null,
      showCoverage: false,
      statusText: "No dataset traversal was measured before cancellation"
    });
  });

  it("shows a dataset job's own incomplete coverage and retains completed coverage outside dataset jobs", () => {
    const currentCoverage = {
      manifestId: "current-manifest",
      discoveredRecords: 1,
      processedRecords: 0,
      complete: false
    };
    const staleCompletedCoverage = {
      manifestId: "older-complete-manifest",
      discoveredRecords: 1,
      processedRecords: 1,
      complete: true
    };

    expect(runtimeJobScopedDatasetCoverage(
      { ...job("cancelled"), kind: "crawl" },
      currentCoverage,
      staleCompletedCoverage
    )).toBe(currentCoverage);
    expect(runtimeJobScopedDatasetCoverage(
      { ...job("complete"), kind: "audio" },
      null,
      staleCompletedCoverage
    )).toBe(staleCompletedCoverage);
    expect(runtimeJobScopedDatasetCoverage(
      null,
      null,
      staleCompletedCoverage
    )).toBe(staleCompletedCoverage);
  });
});
