// Failed package health diagnostics deliberately exclude raw error strings,
// arbitrary fields, credentials, paths, environment and worker stderr.
export function packagedWorkerHealthDiagnostic(health, expectedPlatform) {
  const kernels = health?.capabilities?.nativePackedKernels;
  const known = new Map([
    ["Native packed kernel integrity verification failed", "native-library-integrity-mismatch"],
    ["Native packed CPU kernel must be rebuilt for the current source", "native-source-hash-mismatch"],
    ["Native packed kernel ABI is incompatible", "native-library-ABI-mismatch"]
  ]);
  const nativeError = typeof kernels?.error === "string" && kernels.error
    ? known.get(kernels.error) || "native-library-load-or-unclassified-error"
    : kernels?.cpu === false ? "native-library-missing-or-unavailable" : null;
  return {
    ready: health?.ready === true,
    worker: health?.worker === "python" || health?.worker === "python-inspection" ? health.worker : "invalid-worker-kind",
    protocolVersionValid: health?.protocolVersion === 1,
    operatingSystemMatches: health?.operatingSystem === expectedPlatform,
    loadedBrains: typeof health?.loadedBrains === "number" && Number.isSafeInteger(health.loadedBrains)
      && health.loadedBrains >= 0 ? health.loadedBrains : "invalid-count",
    versionFieldsPresent: ["engineVersion", "pythonVersion", "torchVersion", "platform"].every(
      key => typeof health?.[key] === "string" && health[key].length > 0),
    nativePackedCpu: kernels?.cpu === true,
    nativeKernelFailure: nativeError
  };
}

export function packagedWorkerRpcDiagnostic(error) {
  const data = error?.data, status = data?.status;
  const causes = new Set(["RequestMemoryPressure", "NeuralStateResourcePause", "SharedQuotaPause",
    "MemoryError", "OverflowError", "OperationalError", "DatabaseError", "IntegrityError",
    "OSError", "PermissionError", "TimeoutError", "RuntimeError", "ValueError"]);
  const result = { code: Number.isSafeInteger(error?.code) ? error.code : null,
    causeClass: causes.has(data?.causeClass) ? data.causeClass : "unclassified" };
  for (const key of ["resourcePaused", "recoverable", "frameRejected", "inputTruncated", "workerRestartRequired"]) {
    if (typeof data?.[key] === "boolean") result[key] = data[key];
  }
  const numeric = {};
  for (const key of ["systemRamBudgetBytes", "admissionResidentMemoryBytes", "availableMemoryBytes",
    "ramReserveBytes", "managedMemorySampleDurationMs", "managedMemorySampleAgeSeconds"]) {
    if (typeof status?.[key] === "number" && Number.isFinite(status[key]) && status[key] >= 0) numeric[key] = status[key];
  }
  for (const key of ["memoryPressure", "ramAdmissionVerified", "managedMemorySampleCached"]) {
    if (typeof status?.[key] === "boolean") numeric[key] = status[key];
  }
  for (const key of ["ramEscrowBytes", "globalRamCeilingBytes"]) {
    const value = status?.sharedQuota?.[key];
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) numeric[key] = value;
  }
  return { ...result, status: numeric };
}
