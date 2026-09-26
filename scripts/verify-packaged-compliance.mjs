import { createHash } from "node:crypto";
import { opendir, readFile, stat, writeFile } from "node:fs/promises";
import { basename, join, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { unzipSync } from "fflate";

const REQUIRED_NOTICE =
  "Required Notice: Copyright 2026 Omni-AGI contributors.";
const ROOT_LEGAL_FILES = [
  "LICENSE.md",
  "COMMERCIAL_LICENSE.md",
  "THIRD_PARTY_NOTICES.md"
];
const POLICY_FILE = "licenses/ffmpeg-runtime-policy.json";

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function option(name, argv = process.argv.slice(2)) {
  const inline = argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = argv.indexOf(name);
  return index >= 0 ? argv[index + 1] : undefined;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function normalizedArchivePath(value) {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

function isBundledFfmpegExecutable(value) {
  const path = normalizedArchivePath(value).toLowerCase();
  const name = basename(path);
  if (
    /^ffmpeg(?:[-_.].*)?(?:\.exe)?$/.test(name) &&
    !/\.(?:json|md|py|pyc|pyo|rst|txt)$/.test(name)
  ) {
    return true;
  }
  return (
    path.includes("/imageio_ffmpeg/binaries/") &&
    /^ffmpeg(?:[-.]|$)/.test(name) &&
    !/\.(?:md|py|pyc|pyo|txt)$/.test(name)
  );
}

async function regularFiles(root) {
  const files = [];
  async function walk(directory) {
    const iterator = await opendir(directory);
    for await (const entry of iterator) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await walk(path);
      else if (entry.isFile()) files.push(path);
    }
  }
  await walk(root);
  return files;
}

export async function legalSourcePayload(repoRoot = resolve(".")) {
  const root = resolve(repoRoot);
  const licenseRoot = join(root, "licenses");
  const sources = new Map();
  for (const relativePath of ROOT_LEGAL_FILES) {
    sources.set(relativePath, await readFile(join(root, relativePath)));
  }
  for (const path of await regularFiles(licenseRoot)) {
    const relativePath = normalizedArchivePath(relative(root, path));
    sources.set(relativePath, await readFile(path));
  }
  const primaryLicense = sources.get("LICENSE.md")?.toString("utf8") ?? "";
  const notice = sources.get("licenses/REQUIRED_NOTICE.txt")?.toString("utf8").trim();
  requireValue(
    primaryLicense.includes(REQUIRED_NOTICE),
    "LICENSE.md is missing the Required Notice."
  );
  requireValue(
    notice === REQUIRED_NOTICE,
    "licenses/REQUIRED_NOTICE.txt does not exactly reproduce the Required Notice."
  );
  const policyBytes = sources.get(POLICY_FILE);
  requireValue(policyBytes, `${POLICY_FILE} is missing.`);
  let policy;
  try {
    policy = JSON.parse(policyBytes.toString("utf8"));
  } catch (error) {
    throw new Error(
      `${POLICY_FILE} is invalid JSON: ${error instanceof Error ? error.message : error}`
    );
  }
  requireValue(
    policy?.schemaVersion === 1 &&
      policy?.wrapper?.name === "imageio-ffmpeg" &&
      policy?.wrapper?.bundled === true &&
      policy?.executable?.bundled === false &&
      policy?.executable?.distributionMode === "external-runtime-only" &&
      policy?.verification?.correspondingSourceRequiredForThisDistribution === false &&
      policy?.policyChange?.requireCompleteCorrespondingSource === true &&
      policy?.policyChange?.requireBuildAndInstallScripts === true &&
      policy?.policyChange?.requireLinkedNonSystemLibrarySources === true,
    `${POLICY_FILE} does not describe the enforced external-runtime distribution.`
  );
  const notices = sources.get("THIRD_PARTY_NOTICES.md")?.toString("utf8") ?? "";
  requireValue(
    notices.includes("ffmpeg-runtime-policy.json") &&
      notices.includes("wheel-provided FFmpeg"),
    "THIRD_PARTY_NOTICES.md does not document the FFmpeg distribution boundary."
  );
  return { root, sources, policy, policySha256: sha256(policyBytes) };
}

export async function verifyEngineCompliance(engineDirectory, repoRoot = resolve(".")) {
  const legal = await legalSourcePayload(repoRoot);
  const engineRoot = resolve(engineDirectory);
  const metadata = await stat(engineRoot);
  requireValue(metadata.isDirectory(), `${engineRoot} is not an engine directory.`);
  const files = await regularFiles(engineRoot);
  const prohibited = files
    .map((path) => normalizedArchivePath(relative(engineRoot, path)))
    .filter(isBundledFfmpegExecutable)
    .sort();
  requireValue(
    prohibited.length === 0,
    `Packaged engine contains a prohibited FFmpeg executable: ${prohibited.join(", ")}`
  );
  return {
    schemaVersion: 1,
    kind: "engine-runtime-compliance",
    ffmpegExecutableBundled: false,
    ffmpegPolicySha256: legal.policySha256,
    inspectedFiles: files.length
  };
}

export async function verifyDesktopCompliance(appDirectory, repoRoot = resolve(".")) {
  const legal = await legalSourcePayload(repoRoot);
  const appRoot = resolve(appDirectory);
  const metadata = await stat(appRoot);
  requireValue(metadata.isDirectory(), `${appRoot} is not an extracted desktop package.`);
  const files = await regularFiles(appRoot);
  const relativeFiles = files.map((path) => normalizedArchivePath(relative(appRoot, path)));
  const licenseRoots = relativeFiles
    .filter((path) => {
      const lower = path.toLowerCase();
      return lower.endsWith("/resources/licenses/license.md") || lower === "resources/licenses/license.md";
    })
    .map((path) => path.slice(0, -"LICENSE.md".length));
  requireValue(
    licenseRoots.length === 1,
    `Desktop package must contain exactly one resources/licenses/LICENSE.md; found ${licenseRoots.length}.`
  );
  const licenseRoot = licenseRoots[0];
  const byRelativePath = new Map(
    files.map((path) => [normalizedArchivePath(relative(appRoot, path)), path])
  );
  for (const [sourcePath, expected] of legal.sources) {
    const name = sourcePath.startsWith("licenses/")
      ? sourcePath.slice("licenses/".length)
      : sourcePath;
    const packagedPath = `${licenseRoot}${name}`;
    const localPath = byRelativePath.get(packagedPath);
    requireValue(localPath, `Desktop package is missing ${packagedPath}.`);
    const actual = await readFile(localPath);
    requireValue(
      actual.equals(expected),
      `Desktop package contains a stale or modified ${packagedPath}.`
    );
  }
  const prohibited = relativeFiles.filter(isBundledFfmpegExecutable).sort();
  requireValue(
    prohibited.length === 0,
    `Desktop package contains a prohibited FFmpeg executable: ${prohibited.join(", ")}`
  );
  return {
    schemaVersion: 1,
    kind: "desktop-artifact-compliance",
    legalFilesVerified: legal.sources.size,
    ffmpegExecutableBundled: false,
    ffmpegPolicySha256: legal.policySha256,
    inspectedFiles: files.length
  };
}

function appRootFromIosEntries(paths) {
  const candidates = paths
    .filter((path) => /(?:^|\/)[^/]+\.app\/LICENSE\.md$/.test(path))
    .map((path) => path.slice(0, -"LICENSE.md".length));
  requireValue(
    candidates.length === 1,
    `iOS archive must contain exactly one app-level LICENSE.md; found ${candidates.length}.`
  );
  return candidates[0];
}

export async function verifyMobileCompliance(
  artifactPath,
  platform,
  repoRoot = resolve(".")
) {
  requireValue(platform === "android" || platform === "ios", "platform must be android or ios.");
  const legal = await legalSourcePayload(repoRoot);
  const archive = unzipSync(new Uint8Array(await readFile(resolve(artifactPath))));
  const entries = new Map(
    Object.entries(archive).map(([path, value]) => [
      normalizedArchivePath(path),
      Buffer.from(value)
    ])
  );
  const paths = [...entries.keys()];
  const appRoot = platform === "android" ? "assets/legal/" : appRootFromIosEntries(paths);
  for (const [relativePath, expected] of legal.sources) {
    const archivePath = `${appRoot}${relativePath}`;
    const actual = entries.get(archivePath);
    requireValue(actual, `${platform} artifact is missing ${archivePath}.`);
    requireValue(
      actual.equals(expected),
      `${platform} artifact contains a stale or modified ${archivePath}.`
    );
  }
  const prohibited = paths.filter(isBundledFfmpegExecutable).sort();
  requireValue(
    prohibited.length === 0,
    `${platform} artifact contains a prohibited FFmpeg executable: ${prohibited.join(", ")}`
  );
  return {
    schemaVersion: 1,
    kind: "mobile-artifact-compliance",
    platform,
    artifact: basename(artifactPath),
    requiredNotice: REQUIRED_NOTICE,
    legalFilesVerified: legal.sources.size,
    ffmpegExecutableBundled: false,
    ffmpegPolicySha256: legal.policySha256
  };
}

async function main() {
  const argv = process.argv.slice(2);
  const repoRoot = resolve(option("--repo-root", argv) ?? ".");
  const engineDirectory = option("--engine-dir", argv);
  const desktopDirectory = option("--desktop-dir", argv);
  const artifact = option("--artifact", argv);
  const platform = option("--platform", argv);
  const output = option("--output", argv);
  requireValue(
    [engineDirectory, desktopDirectory, artifact].filter(Boolean).length === 1,
    "Provide exactly one of --engine-dir, --desktop-dir, or --artifact."
  );
  const report = engineDirectory
    ? await verifyEngineCompliance(engineDirectory, repoRoot)
    : desktopDirectory
      ? await verifyDesktopCompliance(desktopDirectory, repoRoot)
      : await verifyMobileCompliance(artifact, platform, repoRoot);
  const serialized = `${JSON.stringify(report, null, 2)}\n`;
  if (output) await writeFile(resolve(output), serialized, "utf8");
  process.stdout.write(serialized);
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(resolve(process.argv[1])).href
) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : error}\n`);
    process.exitCode = 1;
  });
}
