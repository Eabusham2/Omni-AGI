/** Shared stable-version policy; a new major release must not require bypassing validation. */
export function stableReleaseVersion(version) {
  if (typeof version !== "string" || !/^(?:[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)$/u.test(version)) {
    throw new Error(`Release version must be a stable semantic version with major >= 1, received ${String(version)}.`);
  }
  const [major, minor, patch] = version.split(".").map(Number);
  if (![major, minor, patch].every(Number.isSafeInteger) || minor >= 100 || patch >= 100) {
    throw new Error("Android versionCode mapping requires safe integers and minor and patch versions below 100.");
  }
  const androidVersionCode = major * 10_000 + minor * 100 + patch;
  // Google Play's documented ceiling: https://developer.android.com/studio/publish/versioning
  if (!Number.isSafeInteger(androidVersionCode) || androidVersionCode > 2_100_000_000) {
    throw new Error("Release version exceeds the Android versionCode range.");
  }
  return { major, minor, patch, tag: `v${version}`, androidVersionCode };
}
