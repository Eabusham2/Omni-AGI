import { createHash } from "node:crypto";
import bundledManifest from "../../licenses/ffmpeg-runtime-manifest.json";
import bundledPolicy from "../../licenses/ffmpeg-runtime-policy.json";

export const VIDEO_RUNTIME_TARGETS = [
  "win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "linux-x64", "linux-arm64"
] as const;
export type VideoRuntimeTarget = typeof VIDEO_RUNTIME_TARGETS[number];
export type VideoRuntimeLicense =
  | "LGPL-2.1-or-later" | "LGPL-3.0-or-later" | "GPL-2.0-or-later" | "GPL-3.0-or-later";

/** Pins are compiled into the app, never obtained from an imported recipe or remote manifest. */
export interface VideoRuntimeFilePin {
  fileName: string;
  url: string;
  sizeBytes: number;
  sha256: string;
}

export interface VideoRuntimeArtifact {
  id: string;
  target: VideoRuntimeTarget;
  version: string;
  license: VideoRuntimeLicense;
  binary: VideoRuntimeFilePin;
  correspondingSource: VideoRuntimeFilePin;
  buildAndInstallMaterial: VideoRuntimeFilePin;
  licenseNotices: VideoRuntimeFilePin;
  provenance: {
    upstreamReleaseUrl: string;
    buildRepositoryUrl: string;
    buildCommit: string;
    configuration: string;
    linkage: "standalone-cli-static-except-system";
    reviewId: string;
    reviewedAt: string;
    completeCorrespondingSourceReviewed: true;
    allLinkedNonSystemLibrariesReviewed: true;
    distributionLicenseReviewed: true;
  };
  linkedLibraries: Array<{
    configureFlag: string;
    name: string;
    license: string;
    source: VideoRuntimeFilePin;
  }>;
}

export interface VideoRuntimeManifest {
  schemaVersion: 1;
  component: "FFmpeg";
  artifacts: VideoRuntimeArtifact[];
  unavailableTargets: Array<{ target: VideoRuntimeTarget; reason: string }>;
}

const SHA256 = /^[a-f0-9]{64}$/;
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
const SAFE_FILE = /^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$/;
const MAX_PIN_BYTES = 16 * 1024 ** 3;
const RELEASE_PREFIX = "/Eabusham2/Omni-AGI/releases/download/";
const LICENSES = new Set([
  "LGPL-2.1-or-later", "LGPL-3.0-or-later", "GPL-2.0-or-later", "GPL-3.0-or-later"
]);

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
  return value as Record<string, unknown>;
}

function onlyKeys(value: Record<string, unknown>, keys: string[], label: string): void {
  if (Object.keys(value).some((key) => !keys.includes(key))) {
    throw new Error(`${label} contains an unsupported field; runtime manifests cannot supply commands.`);
  }
}

function text(value: unknown, label: string, maximum = 8_192): string {
  if (typeof value !== "string" || !value.trim() || value.length > maximum || /[\0\r\n]/.test(value)) {
    throw new Error(`${label} must be non-empty, bounded single-line text.`);
  }
  return value;
}

function target(value: unknown): VideoRuntimeTarget {
  if (!VIDEO_RUNTIME_TARGETS.includes(value as VideoRuntimeTarget)) {
    throw new Error("Unsupported video runtime target.");
  }
  return value as VideoRuntimeTarget;
}

function httpsUrl(value: unknown, label: string): URL {
  const url = new URL(text(value, label));
  if (url.protocol !== "https:" || url.username || url.password || url.port || url.hash) {
    throw new Error(`${label} must use credential-free HTTPS on the default port.`);
  }
  return url;
}

/**
 * Automatic payloads must be published together in an explicitly versioned Omni
 * release after provenance review. Vendor "latest" endpoints and arbitrary
 * repository install scripts are not an automatic dependency source.
 */
export function validateVideoRuntimeDownloadUrl(value: string): URL {
  const url = httpsUrl(value, "Video runtime artifact URL");
  const suffix = url.pathname.startsWith(RELEASE_PREFIX)
    ? url.pathname.slice(RELEASE_PREFIX.length).split("/") : [];
  if (
    url.hostname !== "github.com" || url.search || suffix.length !== 2 ||
    !SAFE_ID.test(suffix[0] ?? "") || !SAFE_FILE.test(suffix[1] ?? "") ||
    /^(?:latest|main|master|head)$/i.test(suffix[0] ?? "")
  ) {
    throw new Error("Automatic video runtime files must use pinned Omni release assets, not a mutable or arbitrary URL.");
  }
  return url;
}

function pin(value: unknown, label: string): VideoRuntimeFilePin {
  const input = record(value, label);
  onlyKeys(input, ["fileName", "url", "sizeBytes", "sha256"], label);
  const fileName = text(input.fileName, `${label} fileName`, 200);
  if (!SAFE_FILE.test(fileName)) throw new Error(`${label} has an unsafe file name.`);
  const url = text(input.url, `${label} URL`);
  validateVideoRuntimeDownloadUrl(url);
  if (!Number.isSafeInteger(input.sizeBytes) || Number(input.sizeBytes) < 1 || Number(input.sizeBytes) > MAX_PIN_BYTES) {
    throw new Error(`${label} requires an exact bounded byte size.`);
  }
  const hash = text(input.sha256, `${label} SHA-256`, 64);
  if (!SHA256.test(hash)) throw new Error(`${label} requires an exact SHA-256 pin.`);
  return { fileName, url, sizeBytes: Number(input.sizeBytes), sha256: hash };
}

function artifact(value: unknown): VideoRuntimeArtifact {
  const input = record(value, "Video runtime artifact");
  onlyKeys(input, ["id", "target", "version", "license", "binary", "correspondingSource", "buildAndInstallMaterial", "licenseNotices", "provenance", "linkedLibraries"], "Video runtime artifact");
  const id = text(input.id, "Video runtime id", 128);
  const version = text(input.version, "FFmpeg version", 128);
  if (!SAFE_ID.test(id) || !SAFE_ID.test(version)) throw new Error("Video runtime id/version is unsafe.");
  const runtimeTarget = target(input.target);
  const license = text(input.license, "FFmpeg license", 64) as VideoRuntimeLicense;
  if (!LICENSES.has(license)) throw new Error("Unreviewed/nonfree FFmpeg license is not supported.");
  const binary = pin(input.binary, "FFmpeg binary");
  if (binary.fileName !== (runtimeTarget.startsWith("win32-") ? "ffmpeg.exe" : "ffmpeg") || binary.sizeBytes > 1024 ** 3) {
    throw new Error("Video runtime binary must be a bounded standalone native FFmpeg executable.");
  }
  const provenance = record(input.provenance, "FFmpeg provenance");
  onlyKeys(provenance, ["upstreamReleaseUrl", "buildRepositoryUrl", "buildCommit", "configuration", "linkage", "reviewId", "reviewedAt", "completeCorrespondingSourceReviewed", "allLinkedNonSystemLibrariesReviewed", "distributionLicenseReviewed"], "FFmpeg provenance");
  const upstreamReleaseUrl = httpsUrl(provenance.upstreamReleaseUrl, "FFmpeg upstream release URL");
  if (upstreamReleaseUrl.hostname !== "ffmpeg.org" || !/^\/releases\/ffmpeg-[0-9.]+\.tar\.(?:xz|gz|bz2)$/.test(upstreamReleaseUrl.pathname) || upstreamReleaseUrl.search) {
    throw new Error("FFmpeg provenance must identify an exact official upstream source release.");
  }
  const buildRepositoryUrl = httpsUrl(provenance.buildRepositoryUrl, "FFmpeg build repository");
  if (!["https://github.com/Eabusham2/Omni-AGI", "https://github.com/BtbN/FFmpeg-Builds"].includes(buildRepositoryUrl.toString())) {
    throw new Error("FFmpeg build repository is not a reviewed official build source.");
  }
  const buildCommit = text(provenance.buildCommit, "FFmpeg build commit", 40);
  if (!/^[a-f0-9]{40}$/.test(buildCommit)) throw new Error("FFmpeg build provenance needs an exact source commit.");
  const configuration = text(provenance.configuration, "FFmpeg build configuration", 32_768);
  if (/(?:^|\s)--enable-nonfree(?:\s|$)/.test(configuration)) throw new Error("Nonfree FFmpeg configurations cannot be provisioned.");
  const gpl = /(?:^|\s)--enable-gpl(?:\s|$)/.test(configuration);
  const version3 = /(?:^|\s)--enable-version3(?:\s|$)/.test(configuration);
  if (gpl !== license.startsWith("GPL-") || (version3 && !license.includes("3.0"))) {
    throw new Error("FFmpeg license does not match its pinned build configuration.");
  }
  if (provenance.linkage !== "standalone-cli-static-except-system") {
    throw new Error("Only a separately executed, self-contained native FFmpeg CLI is supported.");
  }
  for (const field of ["completeCorrespondingSourceReviewed", "allLinkedNonSystemLibrariesReviewed", "distributionLicenseReviewed"] as const) {
    if (provenance[field] !== true) throw new Error(`FFmpeg ${field} must be reviewed before automatic distribution.`);
  }
  const reviewId = text(provenance.reviewId, "FFmpeg distribution review", 128);
  const reviewedAt = text(provenance.reviewedAt, "FFmpeg review timestamp", 64);
  if (!SAFE_ID.test(reviewId) || !Number.isFinite(Date.parse(reviewedAt))) throw new Error("FFmpeg provenance review is missing or invalid.");
  if (!Array.isArray(input.linkedLibraries) || input.linkedLibraries.length > 128) throw new Error("FFmpeg linked-library sources must be a bounded enumeration.");
  const linkedLibraries = input.linkedLibraries.map((value) => {
    const library = record(value, "FFmpeg linked library");
    onlyKeys(library, ["configureFlag", "name", "license", "source"], "FFmpeg linked library");
    const configureFlag = text(library.configureFlag, "FFmpeg library configure flag", 128);
    if (!/^--enable-lib[A-Za-z0-9_-]+$/.test(configureFlag)) throw new Error("FFmpeg library configuration flag is invalid.");
    return {
      configureFlag,
      name: text(library.name, "FFmpeg linked library name", 128),
      license: text(library.license, "FFmpeg linked library license", 128),
      source: pin(library.source, "FFmpeg linked-library source")
    };
  });
  const enabledLibraries = configuration.match(/--enable-lib[A-Za-z0-9_-]+/g) ?? [];
  if (new Set(linkedLibraries.map((library) => library.configureFlag)).size !== linkedLibraries.length ||
      enabledLibraries.length !== linkedLibraries.length ||
      enabledLibraries.some((flag) => !linkedLibraries.some((library) => library.configureFlag === flag))) {
    throw new Error("Every enabled non-system library must have its reviewed exact corresponding-source pin.");
  }
  const result: VideoRuntimeArtifact = {
    id, target: runtimeTarget, version, license, binary,
    correspondingSource: pin(input.correspondingSource, "FFmpeg complete corresponding source"),
    buildAndInstallMaterial: pin(input.buildAndInstallMaterial, "FFmpeg build/install material"),
    licenseNotices: pin(input.licenseNotices, "FFmpeg license notices"),
    provenance: {
      upstreamReleaseUrl: upstreamReleaseUrl.toString(), buildRepositoryUrl: buildRepositoryUrl.toString(),
      buildCommit, configuration, linkage: "standalone-cli-static-except-system", reviewId, reviewedAt,
      completeCorrespondingSourceReviewed: true, allLinkedNonSystemLibrariesReviewed: true, distributionLicenseReviewed: true
    },
    linkedLibraries
  };
  const files = videoRuntimeFiles(result);
  if (new Set(files.map((file) => file.fileName.toLowerCase())).size !== files.length ||
      files.some((file) => file.fileName.toLowerCase() === "receipt.json")) {
    throw new Error("FFmpeg payload file names must be unique and cannot replace the install receipt.");
  }
  if (Buffer.byteLength(JSON.stringify(result)) > 384 * 1024) {
    throw new Error("FFmpeg reviewed provenance metadata exceeds the safe receipt bound.");
  }
  return result;
}

export function videoRuntimeFiles(value: VideoRuntimeArtifact): VideoRuntimeFilePin[] {
  return [value.correspondingSource, value.buildAndInstallMaterial, value.licenseNotices,
    ...value.linkedLibraries.map((library) => library.source), value.binary];
}

export function validateVideoRuntimeManifest(value: unknown): VideoRuntimeManifest {
  const input = record(value, "Video runtime manifest");
  onlyKeys(input, ["schemaVersion", "component", "artifacts", "unavailableTargets"], "Video runtime manifest");
  if (input.schemaVersion !== 1 || input.component !== "FFmpeg" || !Array.isArray(input.artifacts) || !Array.isArray(input.unavailableTargets)) {
    throw new Error("Unsupported video runtime manifest.");
  }
  const artifacts = input.artifacts.map(artifact);
  const unavailableTargets = input.unavailableTargets.map((value) => {
    const entry = record(value, "Unavailable video runtime target");
    onlyKeys(entry, ["target", "reason"], "Unavailable video runtime target");
    return { target: target(entry.target), reason: text(entry.reason, "Video runtime unavailable reason", 4_000) };
  });
  const coverage = [...artifacts.map((entry) => entry.target), ...unavailableTargets.map((entry) => entry.target)];
  if (coverage.length !== VIDEO_RUNTIME_TARGETS.length || new Set(coverage).size !== VIDEO_RUNTIME_TARGETS.length ||
      new Set(artifacts.map((entry) => entry.id)).size !== artifacts.length) {
    throw new Error("Video runtime manifest must account for all six desktop targets exactly once.");
  }
  return { schemaVersion: 1, component: "FFmpeg", artifacts, unavailableTargets };
}

export function bundledVideoRuntimeManifest(): VideoRuntimeManifest {
  const provisioning = bundledPolicy.executable.automaticProvisioning;
  const contentHash = createHash("sha256").update(JSON.stringify(bundledManifest)).digest("hex");
  if (provisioning.manifestFile !== "ffmpeg-runtime-manifest.json" ||
      provisioning.manifestContentSha256 !== contentHash ||
      provisioning.executeDownloadedSetupScripts !== false) {
    throw new Error("The bundled video runtime catalog does not match its reviewed distribution policy.");
  }
  const manifest = validateVideoRuntimeManifest(bundledManifest);
  if (provisioning.status === "blocked-awaiting-vetted-builds") {
    if (manifest.artifacts.length) throw new Error("Blocked video runtime policy cannot activate unreviewed builds.");
  } else if (provisioning.status !== "ready-pinned" || manifest.artifacts.length !== VIDEO_RUNTIME_TARGETS.length || manifest.unavailableTargets.length) {
    throw new Error("Ready video runtime policy requires the complete reviewed native target matrix.");
  }
  return manifest;
}

export function videoRuntimeArtifactHash(value: VideoRuntimeArtifact): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}
