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
