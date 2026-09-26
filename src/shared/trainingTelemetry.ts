import type { DiskSpaceTelemetry } from "./types";

export const TRAINING_TELEMETRY_SCHEMA_VERSION = 1 as const;
export const TRAINING_CHECKPOINT_INTERVAL_RECORDS = 512;

const MINIMUM_RATE_SAMPLE_MS = 250;
const RATE_SMOOTHING_WEIGHT = 0.3;

export type TrainingTelemetryStage = "learning" | "foundation" | "checkpoint";

export interface FoundationTelemetrySample {
  batchIndex: number;
  batchCount: number;
  targetWindowIndex?: number;
  targetWindowCount?: number;
  completedTokens: number;
  targetTokens: number;
  completedChunks: number;
  totalChunks: number;
}

export interface TrainingResourceReadings {
  processMemoryBytes?: number;
  processPeakMemoryBytes?: number;
  availableMemoryBytes?: number;
  totalMemoryBytes?: number;
  acceleratorFreeMemoryBytes?: number;
  acceleratorTotalMemoryBytes?: number;
  diskFreeBytes?: number;
  diskTotalBytes?: number;
  diskReserveBytes?: number;
  mandatoryFreeDiskBytes?: number;
  projectedDiskFreeBytes?: number;
  estimatedWriteBytes?: number;
  diskPressure?: boolean;
}

export interface TrainingTelemetrySample {
  atMs: number;
  /** Hash-bound manifest entry/worker transaction; never a path or source id. */
  scopeId?: string;
  recordsCompleted: number;
  bytesCompleted: number;
  recordsTotal?: number;
  bytesTotal?: number;
  currentRecord?: number;
  committedRecords?: number;
  expectedRecords?: number;
  checkpointCommitted?: boolean;
  foundation?: FoundationTelemetrySample;
  resourceReadings?: TrainingResourceReadings;
}

export interface TrainingTelemetry {
  schemaVersion: typeof TRAINING_TELEMETRY_SCHEMA_VERSION;
  stage: TrainingTelemetryStage;
  startedAt: string;
  updatedAt: string;
  elapsedMs: number;
  recordsCompleted: number;
  bytesCompleted: number;
  recordsTotal?: number;
  bytesTotal?: number;
  recordsPerSecond?: number;
  bytesPerSecond?: number;
  totalEtaMs?: number;
  nextCheckpoint: {
    intervalRecords: typeof TRAINING_CHECKPOINT_INTERVAL_RECORDS;
    scopeId?: string;
    currentRecord: number;
    committedRecords: number;
    expectedRecords?: number;
    targetRecord: number;
    etaMs?: number;
  };
  foundation?: {
    batchIndex: number;
    batchCount: number;
    targetWindowIndex?: number;
    targetWindowCount?: number;
    completedTokens: number;
    targetTokens: number;
    completedChunks: number;
    totalChunks: number;
    tokensPerSecond?: number;
  };
  memory?: {
    source: "worker-process";
    currentPhysicalBytes: number;
    peakPhysicalBytes: number;
    availablePhysicalBytes?: number;
    totalPhysicalBytes?: number;
    acceleratorFreeBytes?: number;
    acceleratorTotalBytes?: number;
  };
  diskSpace?: DiskSpaceTelemetry;
  sampling: {
    sampleCount: number;
    lastEventAtMs: number;
    recordSampleAtMs: number;
    recordSampleCount: number;
    byteSampleAtMs: number;
    byteSampleCount: number;
    foundationBatchIndex?: number;
    foundationSampleAtMs?: number;
    foundationSampleTokens?: number;
  };
}

export interface TrainingTelemetryPresentation {
  stageLabel: "Learning" | "Answering cortex" | "Checkpoint commit";
  throughput: string[];
  timing: string[];
  memory: string;
  storage: string;
}

function nonnegative(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : undefined;
}

function count(value: unknown): number | undefined {
  const resolved = nonnegative(value);
  return resolved !== undefined && Number.isSafeInteger(resolved)
    ? resolved
    : undefined;
}

function positive(value: unknown): number | undefined {
  const resolved = nonnegative(value);
  return resolved !== undefined && resolved > 0 ? resolved : undefined;
}

function smooth(previous: number | undefined, instantaneous: number): number {
  return previous === undefined
    ? instantaneous
    : previous * (1 - RATE_SMOOTHING_WEIGHT) +
        instantaneous * RATE_SMOOTHING_WEIGHT;
}

function measuredRate(
  previous: number | undefined,
  delta: number,
  elapsedMs: number
): number | undefined {
  if (delta <= 0 || elapsedMs < MINIMUM_RATE_SAMPLE_MS) {
    return undefined;
  }
  const instantaneous = delta / (elapsedMs / 1_000);
  return Number.isFinite(instantaneous) && instantaneous > 0
    ? smooth(previous, instantaneous)
    : undefined;
}

function etaMs(remaining: number, rate: number | undefined): number | undefined {
  if (remaining <= 0 || rate === undefined || !Number.isFinite(rate) || rate <= 0) {
    return undefined;
  }
  return Math.ceil((remaining / rate) * 1_000);
}

function iso(atMs: number): string {
  return new Date(atMs).toISOString();
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

function scopeId(value: unknown): string | undefined {
  return typeof value === "string" && /^[a-f0-9]{64}$/iu.test(value)
    ? value.toLowerCase()
    : undefined;
}

export function foundationTelemetrySample(
  value: unknown
): FoundationTelemetrySample | undefined {
  const input = record(value);
  if (!input) return undefined;
  return normalizedFoundation({
    batchIndex: Number(input.batchIndex),
    batchCount: Number(input.batchCount),
    ...(input.targetWindowIndex !== undefined
      ? { targetWindowIndex: Number(input.targetWindowIndex) }
      : {}),
    ...(input.targetWindowCount !== undefined
      ? { targetWindowCount: Number(input.targetWindowCount) }
      : {}),
    completedTokens: Number(input.completedTargetTokens),
    targetTokens: Number(input.targetTokens),
    completedChunks: Number(input.lossChunksCompleted),
    totalChunks: Number(input.lossChunksTotal)
  });
}

export function trainingResourceReadings(
  value: unknown
): TrainingResourceReadings | undefined {
  const input = record(value);
  if (!input) return undefined;
  const result: TrainingResourceReadings = {};
  for (const field of [
    "processMemoryBytes",
    "processPeakMemoryBytes",
    "availableMemoryBytes",
    "totalMemoryBytes",
    "acceleratorFreeMemoryBytes",
    "acceleratorTotalMemoryBytes",
    "diskFreeBytes",
    "diskTotalBytes",
    "diskReserveBytes",
    "mandatoryFreeDiskBytes",
    "projectedDiskFreeBytes",
    "estimatedWriteBytes"
  ] as const) {
    const resolved = nonnegative(input[field]);
    if (resolved !== undefined) result[field] = resolved;
  }
  if (typeof input.diskPressure === "boolean") {
    result.diskPressure = input.diskPressure;
  }
  return Object.keys(result).length > 0 ? result : undefined;
}

function normalizedFoundation(
  value: FoundationTelemetrySample | undefined
): FoundationTelemetrySample | undefined {
  if (!value) return undefined;
  const batchIndex = count(value.batchIndex);
  const batchCount = count(value.batchCount);
  const completedTokens = count(value.completedTokens);
  const targetTokens = count(value.targetTokens);
  const completedChunks = count(value.completedChunks);
  const totalChunks = count(value.totalChunks);
  if (
    batchIndex === undefined ||
    batchCount === undefined ||
    completedTokens === undefined ||
    targetTokens === undefined ||
    completedChunks === undefined ||
    totalChunks === undefined
  ) {
    return undefined;
  }
  if (batchIndex < 1 || batchCount < 1 || batchIndex > batchCount) {
    return undefined;
  }
  const targetWindowIndex = count(value.targetWindowIndex);
  const targetWindowCount = count(value.targetWindowCount);
  if (
    (targetWindowIndex === undefined) !== (targetWindowCount === undefined) ||
    (targetWindowIndex !== undefined &&
      targetWindowCount !== undefined &&
      (targetWindowIndex < 1 ||
        targetWindowCount < 1 ||
        targetWindowIndex > targetWindowCount))
  ) {
    return undefined;
  }
  return {
    batchIndex,
    batchCount,
    ...(targetWindowIndex !== undefined && targetWindowCount !== undefined
      ? { targetWindowIndex, targetWindowCount }
      : {}),
    completedTokens: Math.min(completedTokens, targetTokens),
    targetTokens,
    completedChunks: Math.min(completedChunks, totalChunks),
    totalChunks
  };
}

function normalizedMemory(
  previous: TrainingTelemetry | undefined,
  readings: TrainingResourceReadings | undefined
): TrainingTelemetry["memory"] {
  const current = nonnegative(readings?.processMemoryBytes);
  const reportedPeak = nonnegative(readings?.processPeakMemoryBytes);
  const previousMemory = previous?.memory;
  // A process peak is not a current resident reading. Without either a real
  // current sample or a prior current sample, do not manufacture one from it.
  if (current === undefined && previousMemory === undefined) return undefined;
  if (!readings) return previousMemory;
  const currentPhysicalBytes = current ?? previousMemory!.currentPhysicalBytes;
  const peakPhysicalBytes = Math.max(
    currentPhysicalBytes,
    reportedPeak ?? 0,
    previousMemory?.peakPhysicalBytes ?? 0
  );
  const availablePhysicalBytes =
    nonnegative(readings.availableMemoryBytes) ?? previousMemory?.availablePhysicalBytes;
  const totalPhysicalBytes =
    nonnegative(readings.totalMemoryBytes) ?? previousMemory?.totalPhysicalBytes;
  const acceleratorFreeBytes =
    nonnegative(readings.acceleratorFreeMemoryBytes) ?? previousMemory?.acceleratorFreeBytes;
  const acceleratorTotalBytes =
    nonnegative(readings.acceleratorTotalMemoryBytes) ?? previousMemory?.acceleratorTotalBytes;
  return {
    source: "worker-process",
    currentPhysicalBytes,
    peakPhysicalBytes,
    ...(availablePhysicalBytes !== undefined ? { availablePhysicalBytes } : {}),
    ...(totalPhysicalBytes !== undefined ? { totalPhysicalBytes } : {}),
    ...(acceleratorFreeBytes !== undefined ? { acceleratorFreeBytes } : {}),
    ...(acceleratorTotalBytes !== undefined ? { acceleratorTotalBytes } : {})
  };
}

function normalizedDiskSpaceValue(value: unknown): DiskSpaceTelemetry | undefined {
  const input = record(value);
  if (
    input?.schemaVersion !== 1 ||
    typeof input.measuredAt !== "string" ||
    !Number.isFinite(Date.parse(input.measuredAt)) ||
    typeof input.paused !== "boolean"
  ) {
    return undefined;
  }
  const fields = [
    "diskTotalBytes",
    "diskFreeBytes",
    "mandatoryReserveBytes",
    "selectedDatasetBytes",
    "modelBytes",
    "checkpointBytes",
    "maximumWorkingMemorySpillBytes",
    "futureGrowthBytes",
    "operationWriteBytes",
    "projectedRemainingBytes",
    "projectedAboveReserveBytes"
  ] as const;
  const values = Object.fromEntries(
    fields.map((field) => [field, count(input[field])])
  ) as Record<(typeof fields)[number], number | undefined>;
  if (fields.some((field) => values[field] === undefined)) return undefined;
  const diskTotalBytes = values.diskTotalBytes!;
  const diskFreeBytes = values.diskFreeBytes!;
  const mandatoryReserveBytes = values.mandatoryReserveBytes!;
  const projectedRemainingBytes = values.projectedRemainingBytes!;
  const projectedAboveReserveBytes = values.projectedAboveReserveBytes!;
  if (
    diskFreeBytes > diskTotalBytes ||
    projectedRemainingBytes > diskFreeBytes ||
    projectedAboveReserveBytes !==
      Math.max(0, projectedRemainingBytes - mandatoryReserveBytes) ||
    input.paused !== (projectedRemainingBytes <= mandatoryReserveBytes)
  ) {
    return undefined;
  }
  return {
    schemaVersion: 1,
    measuredAt: new Date(input.measuredAt).toISOString(),
    diskTotalBytes,
    diskFreeBytes,
    mandatoryReserveBytes,
    selectedDatasetBytes: values.selectedDatasetBytes!,
    modelBytes: values.modelBytes!,
    checkpointBytes: values.checkpointBytes!,
    maximumWorkingMemorySpillBytes:
      values.maximumWorkingMemorySpillBytes!,
    futureGrowthBytes: values.futureGrowthBytes!,
    operationWriteBytes: values.operationWriteBytes!,
    projectedRemainingBytes,
    projectedAboveReserveBytes,
    paused: input.paused
  };
}

function normalizedDiskSpace(
  previous: TrainingTelemetry | undefined,
  readings: TrainingResourceReadings | undefined,
  atMs: number
): DiskSpaceTelemetry | undefined {
  const diskTotalBytes = count(readings?.diskTotalBytes);
  const diskFreeBytes = count(readings?.diskFreeBytes);
  const mandatoryReserveBytes = count(
    readings?.mandatoryFreeDiskBytes ?? readings?.diskReserveBytes
  );
  if (
    diskTotalBytes === undefined ||
    diskFreeBytes === undefined ||
    mandatoryReserveBytes === undefined ||
    diskFreeBytes > diskTotalBytes
  ) {
    return previous?.diskSpace;
  }
  const operationWriteBytes = count(readings?.estimatedWriteBytes) ?? 0;
  const projectedRemainingBytes = Math.min(
    diskFreeBytes,
    count(readings?.projectedDiskFreeBytes) ??
      Math.max(0, diskFreeBytes - operationWriteBytes)
  );
  return {
    schemaVersion: 1,
    measuredAt: iso(atMs),
    diskTotalBytes,
    diskFreeBytes,
    mandatoryReserveBytes,
    selectedDatasetBytes: previous?.diskSpace?.selectedDatasetBytes ?? 0,
    modelBytes: previous?.diskSpace?.modelBytes ?? 0,
    checkpointBytes: previous?.diskSpace?.checkpointBytes ?? 0,
    maximumWorkingMemorySpillBytes:
      previous?.diskSpace?.maximumWorkingMemorySpillBytes ?? 0,
    futureGrowthBytes: previous?.diskSpace?.futureGrowthBytes ?? 0,
    operationWriteBytes,
    projectedRemainingBytes,
    projectedAboveReserveBytes: Math.max(
      0,
      projectedRemainingBytes - mandatoryReserveBytes
    ),
    paused: projectedRemainingBytes <= mandatoryReserveBytes
  };
}

export function updateTrainingTelemetry(
  previous: TrainingTelemetry | undefined,
  sample: TrainingTelemetrySample
): TrainingTelemetry {
  const atMs = nonnegative(sample.atMs);
  if (atMs === undefined || !Number.isSafeInteger(atMs)) {
    throw new Error("Training telemetry requires a finite event timestamp.");
  }
  if (previous && atMs < previous.sampling.lastEventAtMs) {
    throw new Error("Training telemetry timestamps must be monotonic.");
  }
  const sampleScopeId = scopeId(sample.scopeId);
  if (sample.scopeId !== undefined && sampleScopeId === undefined) {
    throw new Error("Training telemetry checkpoint scope is invalid.");
  }
  const sameCheckpointScope =
    previous === undefined || previous.nextCheckpoint.scopeId === sampleScopeId;
  const recordsCompleted = Math.max(
    previous?.recordsCompleted ?? 0,
    count(sample.recordsCompleted) ?? 0
  );
  const bytesCompleted = Math.max(
    previous?.bytesCompleted ?? 0,
    count(sample.bytesCompleted) ?? 0
  );
  const recordsTotalCandidate = count(sample.recordsTotal);
  const bytesTotalCandidate = count(sample.bytesTotal);
  const recordsTotal = Math.max(
    recordsCompleted,
    recordsTotalCandidate ?? 0,
    previous?.recordsTotal ?? 0
  );
  const bytesTotal = Math.max(
    bytesCompleted,
    bytesTotalCandidate ?? 0,
    previous?.bytesTotal ?? 0
  );
  const recordsTotalKnown =
    recordsTotalCandidate !== undefined || previous?.recordsTotal !== undefined;
  const bytesTotalKnown =
    bytesTotalCandidate !== undefined || previous?.bytesTotal !== undefined;
  const recordDelta = previous
    ? recordsCompleted - previous.sampling.recordSampleCount
    : 0;
  const byteDelta = previous
    ? bytesCompleted - previous.sampling.byteSampleCount
    : 0;
  const recordRateSampleReady = Boolean(
    previous &&
    recordDelta > 0 &&
    atMs - previous.sampling.recordSampleAtMs >= MINIMUM_RATE_SAMPLE_MS
  );
  const byteRateSampleReady = Boolean(
    previous &&
    byteDelta > 0 &&
    atMs - previous.sampling.byteSampleAtMs >= MINIMUM_RATE_SAMPLE_MS
  );
  const nextRecordsRate = previous
    ? measuredRate(
        previous.recordsPerSecond,
        recordDelta,
        atMs - previous.sampling.recordSampleAtMs
      )
    : undefined;
  const nextBytesRate = previous
    ? measuredRate(
        previous.bytesPerSecond,
        byteDelta,
        atMs - previous.sampling.byteSampleAtMs
      )
    : undefined;
  const recordsPerSecond = nextRecordsRate ?? previous?.recordsPerSecond;
  const bytesPerSecond = nextBytesRate ?? previous?.bytesPerSecond;
  const foundationSample = normalizedFoundation(sample.foundation);
  let foundation: TrainingTelemetry["foundation"];
  let sameFoundationBatch = false;
  let foundationRateSampleReady = false;
  if (foundationSample) {
    const previousFoundation = previous?.stage === "foundation"
      ? previous.foundation
      : undefined;
    const previousSampling = previous?.sampling;
    sameFoundationBatch =
      previousFoundation?.batchIndex === foundationSample.batchIndex &&
      previousFoundation.batchCount === foundationSample.batchCount &&
      previousFoundation.targetWindowIndex === foundationSample.targetWindowIndex &&
      previousFoundation.targetWindowCount === foundationSample.targetWindowCount &&
      previousFoundation.targetTokens === foundationSample.targetTokens &&
      sameCheckpointScope &&
      previousSampling?.foundationBatchIndex === foundationSample.batchIndex;
    foundationRateSampleReady = Boolean(
      sameFoundationBatch &&
      foundationSample.completedTokens -
        (previousSampling?.foundationSampleTokens ?? 0) > 0 &&
      atMs - (previousSampling?.foundationSampleAtMs ?? atMs) >=
        MINIMUM_RATE_SAMPLE_MS
    );
    const tokensPerSecond = sameFoundationBatch
      ? measuredRate(
          previousFoundation?.tokensPerSecond,
          foundationSample.completedTokens -
            (previousSampling?.foundationSampleTokens ?? 0),
          atMs - (previousSampling?.foundationSampleAtMs ?? atMs)
        )
      : undefined;
    foundation = {
      ...foundationSample,
      ...(tokensPerSecond !== undefined ? { tokensPerSecond } : {})
    };
  } else {
    foundation = sameCheckpointScope ? previous?.foundation : undefined;
  }
  const committedRecords = Math.max(
    sameCheckpointScope ? previous?.nextCheckpoint.committedRecords ?? 0 : 0,
    count(sample.committedRecords) ?? 0
  );
  const currentRecord = Math.max(
    sameCheckpointScope ? previous?.nextCheckpoint.currentRecord ?? 0 : 0,
    count(sample.currentRecord) ?? recordsCompleted,
    committedRecords
  );
  const expectedRecordsCandidate = count(sample.expectedRecords);
  const previousExpectedRecords = sameCheckpointScope
    ? previous?.nextCheckpoint.expectedRecords
    : undefined;
  const expectedRecords =
    expectedRecordsCandidate !== undefined || previousExpectedRecords !== undefined
      ? Math.max(
          currentRecord,
          expectedRecordsCandidate ?? 0,
          previousExpectedRecords ?? 0
        )
      : undefined;
  const nextBoundary =
    (Math.floor(committedRecords /
      TRAINING_CHECKPOINT_INTERVAL_RECORDS) + 1) *
    TRAINING_CHECKPOINT_INTERVAL_RECORDS;
  // The generic 512-record checkpoint is not a promise that a finite source
  // contains 512 records. Cap it by the exact whole-job remainder so a
  // one-record manifest never extrapolates 511 nonexistent records. A
  // bytes-only finite stream without an explicit per-entry record total has no
  // trustworthy record denominator and cannot support a record-checkpoint ETA.
  const manifestRecordTarget = recordsTotalKnown
    ? Math.max(
        currentRecord,
        recordsTotal - Math.max(0, recordsCompleted - currentRecord)
      )
    : undefined;
  const targetRecord = Math.min(
    nextBoundary,
    expectedRecords ?? Number.POSITIVE_INFINITY,
    manifestRecordTarget ?? Number.POSITIVE_INFINITY
  );
  const checkpointEta =
    bytesTotalKnown && !recordsTotalKnown && expectedRecords === undefined
    ? undefined
    : etaMs(targetRecord - currentRecord, recordsPerSecond);
  const totalEta = etaMs(bytesTotal - bytesCompleted, bytesPerSecond) ??
    etaMs(recordsTotal - recordsCompleted, recordsPerSecond);
  const stage: TrainingTelemetryStage = sample.checkpointCommitted
    ? "checkpoint"
    : foundationSample
      ? "foundation"
      : "learning";
  const startedAt = previous?.startedAt ?? iso(atMs);
  const memory = normalizedMemory(previous, sample.resourceReadings);
  const diskSpace = normalizedDiskSpace(
    previous,
    sample.resourceReadings,
    atMs
  );
  return {
    schemaVersion: TRAINING_TELEMETRY_SCHEMA_VERSION,
    stage,
    startedAt,
    updatedAt: iso(atMs),
    elapsedMs: Math.max(0, atMs - Date.parse(startedAt)),
    recordsCompleted,
    bytesCompleted,
    ...(recordsTotalCandidate !== undefined || previous?.recordsTotal !== undefined
      ? { recordsTotal }
      : {}),
    ...(bytesTotalCandidate !== undefined || previous?.bytesTotal !== undefined
      ? { bytesTotal }
      : {}),
    ...(recordsPerSecond !== undefined ? { recordsPerSecond } : {}),
    ...(bytesPerSecond !== undefined ? { bytesPerSecond } : {}),
    ...(totalEta !== undefined ? { totalEtaMs: totalEta } : {}),
    nextCheckpoint: {
      intervalRecords: TRAINING_CHECKPOINT_INTERVAL_RECORDS,
      ...(sampleScopeId ? { scopeId: sampleScopeId } : {}),
      currentRecord,
      committedRecords,
      ...(expectedRecords !== undefined ? { expectedRecords } : {}),
      targetRecord,
      ...(checkpointEta !== undefined ? { etaMs: checkpointEta } : {})
    },
    ...(foundation ? { foundation } : {}),
    ...(memory ? { memory } : {}),
    ...(diskSpace ? { diskSpace } : {}),
    sampling: {
      sampleCount: (previous?.sampling.sampleCount ?? 0) + 1,
      lastEventAtMs: atMs,
      recordSampleAtMs: recordRateSampleReady || !previous
        ? atMs
        : previous.sampling.recordSampleAtMs,
      recordSampleCount: recordRateSampleReady || !previous
        ? recordsCompleted
        : previous.sampling.recordSampleCount,
      byteSampleAtMs: byteRateSampleReady || !previous
        ? atMs
        : previous.sampling.byteSampleAtMs,
      byteSampleCount: byteRateSampleReady || !previous
        ? bytesCompleted
        : previous.sampling.byteSampleCount,
      ...(foundationSample
        ? {
            foundationBatchIndex: foundationSample.batchIndex,
            foundationSampleAtMs:
              !previous ||
              !sameFoundationBatch ||
              foundationRateSampleReady
                ? atMs
                : previous.sampling.foundationSampleAtMs,
            foundationSampleTokens:
              !previous ||
              !sameFoundationBatch ||
              foundationRateSampleReady
                ? foundationSample.completedTokens
                : previous.sampling.foundationSampleTokens
          }
        : sameCheckpointScope &&
            previous?.sampling.foundationBatchIndex !== undefined
          ? {
              foundationBatchIndex: previous.sampling.foundationBatchIndex,
              foundationSampleAtMs: previous.sampling.foundationSampleAtMs,
              foundationSampleTokens: previous.sampling.foundationSampleTokens
            }
          : {})
    }
  };
}

/**
 * Telemetry is operational recovery state, never neural authority. Invalid or
 * legacy snapshots are discarded so they cannot block a dataset resume or
 * manufacture a rate from malformed counters.
 */
export function normalizePersistedTrainingTelemetry(
  value: unknown
): TrainingTelemetry | undefined {
  const input = record(value);
  const checkpoint = record(input?.nextCheckpoint);
  const sampling = record(input?.sampling);
  if (
    input?.schemaVersion !== TRAINING_TELEMETRY_SCHEMA_VERSION ||
    !["learning", "foundation", "checkpoint"].includes(String(input.stage)) ||
    typeof input.startedAt !== "string" ||
    !Number.isFinite(Date.parse(input.startedAt)) ||
    typeof input.updatedAt !== "string" ||
    !Number.isFinite(Date.parse(input.updatedAt)) ||
    count(input.elapsedMs) === undefined ||
    count(input.recordsCompleted) === undefined ||
    count(input.bytesCompleted) === undefined ||
    !checkpoint ||
    checkpoint.intervalRecords !== TRAINING_CHECKPOINT_INTERVAL_RECORDS ||
    count(checkpoint.currentRecord) === undefined ||
    count(checkpoint.committedRecords) === undefined ||
    count(checkpoint.targetRecord) === undefined ||
    !sampling ||
    count(sampling.sampleCount) === undefined ||
    nonnegative(sampling.lastEventAtMs) === undefined ||
    nonnegative(sampling.recordSampleAtMs) === undefined ||
    count(sampling.recordSampleCount) === undefined ||
    nonnegative(sampling.byteSampleAtMs) === undefined ||
    count(sampling.byteSampleCount) === undefined
  ) {
    return undefined;
  }
  for (const optional of [
    input.recordsTotal,
    input.bytesTotal,
    input.totalEtaMs,
    checkpoint.etaMs,
    checkpoint.expectedRecords,
    sampling.foundationBatchIndex,
    sampling.foundationSampleTokens
  ]) {
    if (optional !== undefined && count(optional) === undefined) return undefined;
  }
  if (
    checkpoint.scopeId !== undefined &&
    scopeId(checkpoint.scopeId) === undefined
  ) {
    return undefined;
  }
  for (const optional of [input.recordsPerSecond, input.bytesPerSecond]) {
    if (optional !== undefined && positive(optional) === undefined) return undefined;
  }
  if (
    sampling.foundationSampleAtMs !== undefined &&
    nonnegative(sampling.foundationSampleAtMs) === undefined
  ) {
    return undefined;
  }
  const foundation = input.foundation === undefined
    ? undefined
    : record(input.foundation);
  if (
    input.foundation !== undefined &&
    (!foundation ||
      count(foundation.batchIndex) === undefined ||
      count(foundation.batchCount) === undefined ||
      count(foundation.completedTokens) === undefined ||
      count(foundation.targetTokens) === undefined ||
      count(foundation.completedChunks) === undefined ||
      count(foundation.totalChunks) === undefined ||
      (foundation.targetWindowIndex !== undefined &&
        count(foundation.targetWindowIndex) === undefined) ||
      (foundation.targetWindowCount !== undefined &&
        count(foundation.targetWindowCount) === undefined) ||
      (foundation.tokensPerSecond !== undefined &&
        positive(foundation.tokensPerSecond) === undefined))
  ) {
    return undefined;
  }
  const memory = input.memory === undefined ? undefined : record(input.memory);
  if (
    input.memory !== undefined &&
    (!memory ||
      memory.source !== "worker-process" ||
      nonnegative(memory.currentPhysicalBytes) === undefined ||
      nonnegative(memory.peakPhysicalBytes) === undefined)
  ) {
    return undefined;
  }
  for (const optional of [
    memory?.availablePhysicalBytes,
    memory?.totalPhysicalBytes,
    memory?.acceleratorFreeBytes,
    memory?.acceleratorTotalBytes
  ]) {
    if (optional !== undefined && nonnegative(optional) === undefined) return undefined;
  }
  const diskSpace = input.diskSpace === undefined
    ? undefined
    : normalizedDiskSpaceValue(input.diskSpace);
  if (input.diskSpace !== undefined && !diskSpace) return undefined;
  const startedAtMs = Date.parse(input.startedAt);
  const updatedAtMs = Date.parse(input.updatedAt);
  const elapsedMs = count(input.elapsedMs)!;
  const recordsCompleted = count(input.recordsCompleted)!;
  const bytesCompleted = count(input.bytesCompleted)!;
  const currentRecord = count(checkpoint.currentRecord)!;
  const committedRecords = count(checkpoint.committedRecords)!;
  const targetRecord = count(checkpoint.targetRecord)!;
  const lastEventAtMs = nonnegative(sampling.lastEventAtMs)!;
  const recordSampleAtMs = nonnegative(sampling.recordSampleAtMs)!;
  const byteSampleAtMs = nonnegative(sampling.byteSampleAtMs)!;
  const recordSampleCount = count(sampling.recordSampleCount)!;
  const byteSampleCount = count(sampling.byteSampleCount)!;
  const expectedCheckpointRecords = count(checkpoint.expectedRecords);
  const derivedCheckpointBoundary =
    (Math.floor(committedRecords / TRAINING_CHECKPOINT_INTERVAL_RECORDS) + 1) *
    TRAINING_CHECKPOINT_INTERVAL_RECORDS;
  const persistedRecordsRate = positive(input.recordsPerSecond);
  const persistedBytesRate = positive(input.bytesPerSecond);
  const persistedRecordsTotal = count(input.recordsTotal);
  const persistedBytesTotal = count(input.bytesTotal);
  const manifestRecordTarget = persistedRecordsTotal !== undefined
    ? Math.max(
        currentRecord,
        persistedRecordsTotal - Math.max(0, recordsCompleted - currentRecord)
      )
    : undefined;
  const derivedCheckpointTarget = Math.min(
    derivedCheckpointBoundary,
    expectedCheckpointRecords ?? Number.POSITIVE_INFINITY,
    manifestRecordTarget ?? Number.POSITIVE_INFINITY
  );
  const derivedCheckpointEta =
    persistedBytesTotal !== undefined &&
    persistedRecordsTotal === undefined &&
    expectedCheckpointRecords === undefined
      ? undefined
      : etaMs(
          derivedCheckpointTarget - currentRecord,
          persistedRecordsRate
        );
  const derivedTotalEta = (
    persistedBytesTotal !== undefined
      ? etaMs(persistedBytesTotal - bytesCompleted, persistedBytesRate)
      : undefined
  ) ?? (
    persistedRecordsTotal !== undefined
      ? etaMs(persistedRecordsTotal - recordsCompleted, persistedRecordsRate)
      : undefined
  );
  if (
    updatedAtMs < startedAtMs ||
    elapsedMs !== updatedAtMs - startedAtMs ||
    lastEventAtMs !== updatedAtMs ||
    recordSampleAtMs < startedAtMs ||
    byteSampleAtMs < startedAtMs ||
    recordSampleAtMs > lastEventAtMs ||
    byteSampleAtMs > lastEventAtMs ||
    recordSampleCount > recordsCompleted ||
    byteSampleCount > bytesCompleted ||
    committedRecords > currentRecord ||
    targetRecord !== derivedCheckpointTarget ||
    (checkpoint.etaMs === undefined) !== (derivedCheckpointEta === undefined) ||
    (checkpoint.etaMs !== undefined &&
      count(checkpoint.etaMs) !== derivedCheckpointEta) ||
    (input.totalEtaMs === undefined) !== (derivedTotalEta === undefined) ||
    (input.totalEtaMs !== undefined && count(input.totalEtaMs) !== derivedTotalEta) ||
    (checkpoint.expectedRecords !== undefined &&
      count(checkpoint.expectedRecords)! < currentRecord) ||
    (input.recordsTotal !== undefined &&
      count(input.recordsTotal)! < recordsCompleted) ||
    (input.bytesTotal !== undefined && count(input.bytesTotal)! < bytesCompleted)
  ) {
    return undefined;
  }
  if (foundation) {
    const batchIndex = count(foundation.batchIndex)!;
    const batchCount = count(foundation.batchCount)!;
    const completedTokens = count(foundation.completedTokens)!;
    const targetTokens = count(foundation.targetTokens)!;
    const completedChunks = count(foundation.completedChunks)!;
    const totalChunks = count(foundation.totalChunks)!;
    const targetWindowIndex = count(foundation.targetWindowIndex);
    const targetWindowCount = count(foundation.targetWindowCount);
    const foundationSampleAtMs = nonnegative(sampling.foundationSampleAtMs);
    const foundationSampleTokens = count(sampling.foundationSampleTokens);
    if (
      batchIndex < 1 ||
      batchCount < 1 ||
      batchIndex > batchCount ||
      completedTokens > targetTokens ||
      completedChunks > totalChunks ||
      (targetWindowIndex === undefined) !== (targetWindowCount === undefined) ||
      (targetWindowIndex !== undefined &&
        targetWindowCount !== undefined &&
        (targetWindowIndex < 1 ||
          targetWindowCount < 1 ||
          targetWindowIndex > targetWindowCount)) ||
      count(sampling.foundationBatchIndex) !== batchIndex ||
      foundationSampleAtMs === undefined ||
      foundationSampleAtMs < startedAtMs ||
      foundationSampleAtMs > lastEventAtMs ||
      foundationSampleTokens === undefined ||
      foundationSampleTokens > targetTokens
    ) {
      return undefined;
    }
  } else if (
    sampling.foundationBatchIndex !== undefined ||
    sampling.foundationSampleAtMs !== undefined ||
    sampling.foundationSampleTokens !== undefined
  ) {
    return undefined;
  }
  if (memory) {
    const currentPhysicalBytes = nonnegative(memory.currentPhysicalBytes)!;
    const peakPhysicalBytes = nonnegative(memory.peakPhysicalBytes)!;
    const availablePhysicalBytes = nonnegative(memory.availablePhysicalBytes);
    const totalPhysicalBytes = nonnegative(memory.totalPhysicalBytes);
    const acceleratorFreeBytes = nonnegative(memory.acceleratorFreeBytes);
    const acceleratorTotalBytes = nonnegative(memory.acceleratorTotalBytes);
    if (
      peakPhysicalBytes < currentPhysicalBytes ||
      (availablePhysicalBytes !== undefined &&
        totalPhysicalBytes !== undefined &&
        availablePhysicalBytes > totalPhysicalBytes) ||
      (acceleratorFreeBytes !== undefined &&
        acceleratorTotalBytes !== undefined &&
        acceleratorFreeBytes > acceleratorTotalBytes)
    ) {
      return undefined;
    }
  }
  const normalizedFoundationState = foundation
    ? {
        batchIndex: count(foundation.batchIndex)!,
        batchCount: count(foundation.batchCount)!,
        ...(foundation.targetWindowIndex !== undefined &&
        foundation.targetWindowCount !== undefined
          ? {
              targetWindowIndex: count(foundation.targetWindowIndex)!,
              targetWindowCount: count(foundation.targetWindowCount)!
            }
          : {}),
        completedTokens: count(foundation.completedTokens)!,
        targetTokens: count(foundation.targetTokens)!,
        completedChunks: count(foundation.completedChunks)!,
        totalChunks: count(foundation.totalChunks)!,
        ...(foundation.tokensPerSecond !== undefined
          ? { tokensPerSecond: positive(foundation.tokensPerSecond)! }
          : {})
      }
    : undefined;
  const normalizedMemoryState = memory
    ? {
        source: "worker-process" as const,
        currentPhysicalBytes: nonnegative(memory.currentPhysicalBytes)!,
        peakPhysicalBytes: nonnegative(memory.peakPhysicalBytes)!,
        ...(memory.availablePhysicalBytes !== undefined
          ? { availablePhysicalBytes: nonnegative(memory.availablePhysicalBytes)! }
          : {}),
        ...(memory.totalPhysicalBytes !== undefined
          ? { totalPhysicalBytes: nonnegative(memory.totalPhysicalBytes)! }
          : {}),
        ...(memory.acceleratorFreeBytes !== undefined
          ? { acceleratorFreeBytes: nonnegative(memory.acceleratorFreeBytes)! }
          : {}),
        ...(memory.acceleratorTotalBytes !== undefined
          ? { acceleratorTotalBytes: nonnegative(memory.acceleratorTotalBytes)! }
          : {})
      }
    : undefined;
  return {
    schemaVersion: TRAINING_TELEMETRY_SCHEMA_VERSION,
    stage: input.stage as TrainingTelemetryStage,
    startedAt: new Date(input.startedAt).toISOString(),
    updatedAt: new Date(input.updatedAt).toISOString(),
    elapsedMs,
    recordsCompleted,
    bytesCompleted,
    ...(input.recordsTotal !== undefined
      ? { recordsTotal: count(input.recordsTotal)! }
      : {}),
    ...(input.bytesTotal !== undefined
      ? { bytesTotal: count(input.bytesTotal)! }
      : {}),
    ...(input.recordsPerSecond !== undefined
      ? { recordsPerSecond: positive(input.recordsPerSecond)! }
      : {}),
    ...(input.bytesPerSecond !== undefined
      ? { bytesPerSecond: positive(input.bytesPerSecond)! }
      : {}),
    ...(input.totalEtaMs !== undefined
      ? { totalEtaMs: nonnegative(input.totalEtaMs)! }
      : {}),
    nextCheckpoint: {
      intervalRecords: TRAINING_CHECKPOINT_INTERVAL_RECORDS,
      ...(checkpoint.scopeId !== undefined
        ? { scopeId: scopeId(checkpoint.scopeId)! }
        : {}),
      currentRecord,
      committedRecords,
      ...(checkpoint.expectedRecords !== undefined
        ? { expectedRecords: count(checkpoint.expectedRecords)! }
        : {}),
      targetRecord,
      ...(checkpoint.etaMs !== undefined
        ? { etaMs: nonnegative(checkpoint.etaMs)! }
        : {})
    },
    ...(normalizedFoundationState
      ? { foundation: normalizedFoundationState }
      : {}),
    ...(normalizedMemoryState ? { memory: normalizedMemoryState } : {}),
    ...(diskSpace ? { diskSpace } : {}),
    sampling: {
      sampleCount: count(sampling.sampleCount)!,
      lastEventAtMs: nonnegative(sampling.lastEventAtMs)!,
      recordSampleAtMs: nonnegative(sampling.recordSampleAtMs)!,
      recordSampleCount: count(sampling.recordSampleCount)!,
      byteSampleAtMs: nonnegative(sampling.byteSampleAtMs)!,
      byteSampleCount: count(sampling.byteSampleCount)!,
      ...(sampling.foundationBatchIndex !== undefined
        ? { foundationBatchIndex: count(sampling.foundationBatchIndex)! }
        : {}),
      ...(sampling.foundationSampleAtMs !== undefined
        ? { foundationSampleAtMs: nonnegative(sampling.foundationSampleAtMs)! }
        : {}),
      ...(sampling.foundationSampleTokens !== undefined
        ? { foundationSampleTokens: count(sampling.foundationSampleTokens)! }
        : {})
    }
  };
}

export function resumeTrainingTelemetry(
  previous: TrainingTelemetry,
  atMs: number
): TrainingTelemetry {
  const resumedAt = nonnegative(atMs);
  if (
    resumedAt === undefined ||
    !Number.isSafeInteger(resumedAt) ||
    resumedAt < previous.sampling.lastEventAtMs
  ) {
    throw new Error("Training telemetry resume time is invalid.");
  }
  const {
    foundationBatchIndex: _foundationBatchIndex,
    foundationSampleAtMs: _foundationSampleAtMs,
    foundationSampleTokens: _foundationSampleTokens,
    ...samplingWithoutFoundation
  } = previous.sampling;
  return {
    ...previous,
    stage: "learning",
    startedAt: iso(Math.max(0, resumedAt - previous.elapsedMs)),
    updatedAt: iso(resumedAt),
    recordsPerSecond: undefined,
    bytesPerSecond: undefined,
    totalEtaMs: undefined,
    nextCheckpoint: {
      ...previous.nextCheckpoint,
      etaMs: undefined
    },
    ...(previous.foundation
      ? { foundation: { ...previous.foundation, tokensPerSecond: undefined } }
      : {}),
    sampling: {
      ...samplingWithoutFoundation,
      lastEventAtMs: resumedAt,
      recordSampleAtMs: resumedAt,
      recordSampleCount: previous.recordsCompleted,
      byteSampleAtMs: resumedAt,
      byteSampleCount: previous.bytesCompleted,
      ...(previous.foundation
        ? {
            foundationBatchIndex: previous.foundation.batchIndex,
            foundationSampleAtMs: resumedAt,
            foundationSampleTokens: previous.foundation.completedTokens
          }
        : {})
    }
  };
}

function formatRate(value: number): string {
  return value >= 100 ? value.toFixed(0) : value >= 10 ? value.toFixed(1) : value.toFixed(2);
}

function formatBytes(value: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let amount = value;
  let unit = 0;
  while (amount >= 1_024 && unit < units.length - 1) {
    amount /= 1_024;
    unit += 1;
  }
  return `${amount >= 10 || unit === 0 ? amount.toFixed(0) : amount.toFixed(1)} ${units[unit]}`;
}

function formatDuration(value: number): string {
  const seconds = Math.max(0, Math.round(value / 1_000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainder}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatFinishTime(value: number): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(new Date(value));
}

export function presentTrainingTelemetry(
  telemetry: TrainingTelemetry
): TrainingTelemetryPresentation {
  const knownRemaining = [
    telemetry.recordsTotal === undefined
      ? undefined
      : Math.max(0, telemetry.recordsTotal - telemetry.recordsCompleted),
    telemetry.bytesTotal === undefined
      ? undefined
      : Math.max(0, telemetry.bytesTotal - telemetry.bytesCompleted)
  ].filter((value): value is number => value !== undefined);
  const finiteManifest = knownRemaining.length > 0;
  const finiteTraversalAtEnd =
    finiteManifest && knownRemaining.every((value) => value === 0);
  const remainingRecords = telemetry.recordsTotal === undefined
    ? undefined
    : Math.max(0, telemetry.recordsTotal - telemetry.recordsCompleted);
  const finalRecordInFlight =
    telemetry.stage !== "checkpoint" &&
    (telemetry.recordsTotal ?? 0) > 0 &&
    telemetry.nextCheckpoint.currentRecord > 0 &&
    telemetry.nextCheckpoint.targetRecord <= telemetry.nextCheckpoint.currentRecord;
  const throughput: string[] = [];
  if (telemetry.stage === "foundation" && telemetry.foundation) {
    if (telemetry.foundation.tokensPerSecond !== undefined) {
      throughput.push(
        `Smoothed avg ${formatRate(
          telemetry.foundation.tokensPerSecond
        )} foundation tokens/s`
      );
    }
    if (
      telemetry.foundation.targetWindowIndex !== undefined &&
      telemetry.foundation.targetWindowCount !== undefined
    ) {
      throughput.push(
        `window ${telemetry.foundation.targetWindowIndex.toLocaleString()}/${
          telemetry.foundation.targetWindowCount.toLocaleString()
        }`
      );
    }
    throughput.push(
      `${telemetry.foundation.completedChunks.toLocaleString()} / ${
        telemetry.foundation.totalChunks.toLocaleString()
      } chunks`
    );
  } else if (finiteTraversalAtEnd || finalRecordInFlight) {
    throughput.push(
      telemetry.stage === "checkpoint"
        ? "Full manifest traversal measured"
        : "Final record neural work in progress"
    );
  } else {
    if (telemetry.recordsPerSecond !== undefined) {
      const recordsPerMinute = telemetry.recordsPerSecond * 60;
      throughput.push(
        telemetry.recordsPerSecond < 1
          ? `Smoothed avg ${formatRate(telemetry.recordsPerSecond)} records/s · ${
              recordsPerMinute >= 10
                ? recordsPerMinute.toFixed(0)
                : recordsPerMinute >= 1
                  ? recordsPerMinute.toFixed(1)
                  : recordsPerMinute.toFixed(2)
            }/min`
          : `Smoothed avg ${formatRate(telemetry.recordsPerSecond)} records/s`
      );
    }
    if (telemetry.bytesPerSecond !== undefined) {
      throughput.push(
        `Smoothed avg ${formatBytes(telemetry.bytesPerSecond)}/s`
      );
    }
  }
  const timing = [`${formatDuration(telemetry.elapsedMs)} elapsed`];
  if (telemetry.totalEtaMs !== undefined) {
    timing.push(`${formatDuration(telemetry.totalEtaMs)} remaining`);
    timing.push(
      `${formatDuration(
        telemetry.elapsedMs + telemetry.totalEtaMs
      )} projected total`
    );
    timing.push(
      `Finish around ${formatFinishTime(
        Date.parse(telemetry.updatedAt) + telemetry.totalEtaMs
      )}`
    );
  } else if (finiteManifest) {
    timing.push(
      finiteTraversalAtEnd || finalRecordInFlight
        ? telemetry.stage === "checkpoint"
          ? "Manifest traversal complete"
          : telemetry.recordsTotal !== undefined && telemetry.recordsTotal > 0
            ? "Finishing current record…"
            : "Finalizing manifest…"
        : "Remaining time measuring…"
    );
  }
  const checkpointDistance = Math.max(
    0,
    telemetry.nextCheckpoint.targetRecord - telemetry.nextCheckpoint.currentRecord
  );
  const checkpointPrecedesKnownFinish =
    remainingRecords === undefined || checkpointDistance < remainingRecords;
  if (
    telemetry.nextCheckpoint.etaMs !== undefined &&
    checkpointPrecedesKnownFinish
  ) {
    timing.push(
      `${formatDuration(telemetry.nextCheckpoint.etaMs)} to record ${
        telemetry.nextCheckpoint.targetRecord.toLocaleString()
      } checkpoint`
    );
  }
  const stageLabel = telemetry.stage === "foundation"
    ? "Answering cortex"
    : telemetry.stage === "checkpoint"
      ? "Checkpoint commit"
      : "Learning";
  const memory = telemetry.memory
    ? `${formatBytes(telemetry.memory.currentPhysicalBytes)} current · ${
        formatBytes(telemetry.memory.peakPhysicalBytes)
      } peak worker memory`
    : "Worker memory unavailable";
  const storage = telemetry.diskSpace
    ? `${formatBytes(telemetry.diskSpace.diskFreeBytes)} free · ${
        formatBytes(telemetry.diskSpace.projectedRemainingBytes)
      } projected · ${
        formatBytes(telemetry.diskSpace.mandatoryReserveBytes)
      } reserve${telemetry.diskSpace.paused ? " · paused" : ""}`
    : "Worker disk reading unavailable";
  return { stageLabel, throughput, timing, memory, storage };
}
