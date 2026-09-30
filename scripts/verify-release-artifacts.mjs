import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir, rename, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import {
  publicReleaseAssetName,
  releaseArtifactName
} from "./release-artifact-names.mjs";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function validateSourceCommit(record, expectedCommit, label) {
  requireValue(
    record?.sourceCommit === expectedCommit,
    `${label} is not bound to verified source commit ${expectedCommit}.`
  );
}

const complianceReports = [];
function validatePackagedCompliance(report, expectedKind, label) {
  requireValue(
    report?.schemaVersion === 1 &&
      report?.kind === expectedKind &&
      Number.isInteger(report?.legalFilesVerified) &&
      report.legalFilesVerified >= 10 &&
      report?.ffmpegExecutableBundled === false &&
      typeof report?.ffmpegPolicySha256 === "string" &&
      /^[a-f0-9]{64}$/u.test(report.ffmpegPolicySha256),
    `${label} lacks valid packaged legal and FFmpeg compliance evidence.`
  );
  complianceReports.push({ label, report });
}

async function sha256(path) {
  const digest = createHash("sha256");
  await new Promise((resolveHash, rejectHash) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.once("error", rejectHash);
    stream.once("end", resolveHash);
  });
  return digest.digest("hex");
}

async function readJson(path, label) {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`${label} is not valid JSON: ${error instanceof Error ? error.message : error}`);
  }
}

function requiredNames(product, version) {
  const names = [];
  for (const arch of ["x64", "arm64"]) {
    for (const extension of ["exe", "zip"]) {
      names.push(`${product}-${version}-Windows-${arch}.${extension}`);
    }
    for (const extension of ["dmg", "zip"]) {
      names.push(`${product}-${version}-macOS-${arch}.${extension}`);
    }
    for (const extension of ["AppImage", "deb", "tar.gz"]) {
      names.push(
        releaseArtifactName({
          product,
          version,
          platform: "linux",
          architecture: arch,
          extension
        })
      );
    }
    names.push(`windows-package-smoke-${arch}.json`);
    names.push(`mac-package-smoke-${arch}.json`);
    names.push(`linux-package-smoke-${arch}.json`);
  }
  names.push(
    `Omni-AGI-Companion-${version}-Android-debug-signed.apk`,
    `Omni-AGI-Companion-${version}-Android-release-unsigned.apk`,
    "android-package-smoke.json",
    `Omni-AGI-Companion-${version}-iOS-unsigned.ipa`,
    "ios-package-smoke.json"
  );
  return names;
}

async function validateRecordedArtifact(record, expectedName, files, label) {
  requireValue(record && typeof record === "object", `${label} is missing.`);
  requireValue(record.name === expectedName, `${label} names ${String(record.name)}, not ${expectedName}.`);
  const publicName = publicReleaseAssetName(expectedName);
  const path = files.get(publicName);
  requireValue(path, `${publicName} is absent from the release set.`);
  const metadata = await stat(path);
  const actualHash = await sha256(path);
  requireValue(
    typeof record.sha256 === "string" && record.sha256.toLowerCase() === actualHash,
    `${label} hash does not match ${publicName}.`
  );
  if (record.bytes !== undefined) {
    requireValue(record.bytes === metadata.size, `${label} size does not match ${publicName}.`);
  }
  return { name: publicName, bytes: metadata.size, sha256: actualHash };
}

async function canonicalizeReleaseAssetNames(directory, packagedNames, entries) {
  const existing = new Set(entries.map((entry) => entry.name));
  const renames = packagedNames
    .map((packagedName) => ({
      packagedName,
      publicName: publicReleaseAssetName(packagedName)
    }))
    .filter(({ packagedName, publicName }) => packagedName !== publicName);

  for (const { packagedName, publicName } of renames) {
    requireValue(
      !(existing.has(packagedName) && existing.has(publicName)),
      `Release directory contains both packaged and public names: ${packagedName}, ${publicName}.`
    );
  }
  for (const { packagedName, publicName } of renames) {
    if (!existing.has(packagedName)) continue;
    await rename(join(directory, packagedName), join(directory, publicName));
  }
}

function validateWorkerSmoke(smoke, label, expectedVersion) {
  requireValue(smoke?.healthOnly === true, `${label} did not pass health-only smoke.`);
  requireValue(
    smoke?.persistedBrain === false &&
      smoke?.safeTensorCheckpoint === false &&
      smoke?.sqliteEventLog === false,
    `${label} incorrectly claims neural acceptance evidence from CI.`
  );
  requireValue(smoke?.protocolVersion === 1, `${label} used an incompatible protocol.`);
  requireValue(
    smoke?.engineVersion === expectedVersion,
    `${label} worker version ${String(smoke?.engineVersion)} does not match release ${expectedVersion}.`
  );
}

const directory = resolve(option("--directory") ?? "artifacts");
const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const product = packageDocument.build.productName;
const version = packageDocument.version;
const sourceCommit = String(
  option("--source-commit") ?? process.env.OMNI_RELEASE_COMMIT ?? ""
).trim().toLowerCase();
requireValue(
  /^[a-f0-9]{40}$/u.test(sourceCommit),
  "--source-commit must be the verified full 40-hex Git commit SHA."
);
const packagedExpected = requiredNames(product, version);
const expected = packagedExpected.map((name) => publicReleaseAssetName(name));
const generatedMetadata = new Set(["SHA256SUMS.txt", "RELEASE-MANIFEST.json"]);
const allowed = new Set([...expected, ...generatedMetadata]);

const packagedEntries = await readdir(directory, { withFileTypes: true });
const nonFiles = packagedEntries.filter((entry) => !entry.isFile()).map((entry) => entry.name);
requireValue(
  nonFiles.length === 0,
  `Release directory contains non-file entries: ${nonFiles.join(", ")}.`
);
await canonicalizeReleaseAssetNames(directory, packagedExpected, packagedEntries);
const entries = await readdir(directory, { withFileTypes: true });
const unexpected = entries
  .filter((entry) => !allowed.has(entry.name))
  .map((entry) => entry.name)
  .sort();
requireValue(
  unexpected.length === 0,
  `Release directory contains unverified files: ${unexpected.join(", ")}.`
);
const files = new Map(
  entries
    .filter((entry) => expected.includes(entry.name))
    .map((entry) => [entry.name, join(directory, entry.name)])
);
for (const name of expected) {
  requireValue(files.has(name), `Release set requires exactly one ${name}; found 0.`);
  requireValue((await stat(files.get(name))).size > 0, `${name} is empty.`);
}
requireValue(files.size === expected.length, "Release set contains duplicate or missing required names.");

const platformStatus = {
  windows: {},
  macOS: {},
  linux: {},
  android: {},
  iOS: {}
};
for (const arch of ["x64", "arm64"]) {
  const windowsName = `windows-package-smoke-${arch}.json`;
  const macName = `mac-package-smoke-${arch}.json`;
  const linuxName = `linux-package-smoke-${arch}.json`;
  const windows = await readJson(files.get(windowsName), windowsName);
  const mac = await readJson(files.get(macName), macName);
  const linux = await readJson(files.get(linuxName), linuxName);
  validateSourceCommit(windows, sourceCommit, `Windows ${arch} evidence`);
  validateSourceCommit(mac, sourceCommit, `macOS ${arch} evidence`);
  validateSourceCommit(linux, sourceCommit, `Linux ${arch} evidence`);
  validatePackagedCompliance(
    windows.zip?.compliance,
    "desktop-artifact-compliance",
    `Windows ${arch} ZIP`
  );
  validatePackagedCompliance(
    windows.nsis?.compliance,
    "desktop-artifact-compliance",
    `Windows ${arch} NSIS`
  );

  requireValue(windows.architecture === arch, `Windows evidence reports the wrong ${arch} architecture.`);
  requireValue(
    windows.zip?.packagedWorkerArchitecture === "x64",
    `Windows ${arch} ZIP reports an unsupported worker architecture.`
  );
  requireValue(
    windows.zip?.desktopArchitecture === arch &&
      windows.nsis?.desktopArchitecture === arch,
    `Windows ${arch} desktop architecture evidence is incomplete.`
  );
  requireValue(
    windows.nsis?.packagedWorkerArchitecture === "x64",
    `Windows ${arch} NSIS reports an unsupported worker architecture.`
  );
  requireValue(windows.nsis?.silentInstall === true, `Windows ${arch} NSIS was not installed.`);
  requireValue(
    windows.nsis?.desktopShellLaunched === true &&
      windows.nsis?.neuralAcceptanceTested === false,
    `Windows ${arch} package lacks truthful installed-shell evidence.`
  );
  validateWorkerSmoke(windows.zip?.rpcSmoke, `Windows ${arch} ZIP worker`, version);
  validateWorkerSmoke(windows.nsis?.rpcSmoke, `Windows ${arch} installed worker`, version);
  const windowsSigning = windows.signing;
  requireValue(
    typeof windowsSigning?.expectedSigned === "boolean" &&
      typeof windowsSigning?.fullySigned === "boolean",
    `Windows ${arch} signing evidence is missing.`
  );
  if (windowsSigning.expectedSigned) {
    requireValue(windowsSigning.fullySigned, `Windows ${arch} was expected to be signed.`);
  }
  if (windowsSigning.fullySigned) {
    requireValue(
      windows.zip?.desktopSignature?.valid === true &&
        windows.nsis?.installerSignature?.valid === true &&
        windows.nsis?.desktopSignature?.valid === true,
      `Windows ${arch} reports full signing without three valid signatures.`
    );
  }
  requireValue(
    windowsSigning.state ===
      (windowsSigning.fullySigned ? "signed" : "unsigned-signed-ready"),
    `Windows ${arch} signing label disagrees with its signatures.`
  );
  const windowsExe = `${product}-${version}-Windows-${arch}.exe`;
  const windowsZip = `${product}-${version}-Windows-${arch}.zip`;
  await validateRecordedArtifact(windows.nsis, windowsExe, files, `Windows ${arch} NSIS evidence`);
  await validateRecordedArtifact(windows.zip, windowsZip, files, `Windows ${arch} ZIP evidence`);
  platformStatus.windows[arch] = {
    shellArchitecture: arch,
    workerArchitecture: windows.nsis.packagedWorkerArchitecture,
    signing: windowsSigning.state
  };

  requireValue(
    mac.architecture === arch &&
      mac.platform === "mac" &&
      mac.desktopShellLaunched === true &&
      mac.neuralAcceptanceTested === false,
    `macOS ${arch} package lacks truthful desktop-shell evidence.`
  );
  requireValue(
    mac.formatValidation?.dmg?.verified === true &&
      mac.formatValidation?.zip?.extracted === true,
    `macOS ${arch} format validation is incomplete.`
  );
  validateWorkerSmoke(mac.workerSmoke, `macOS ${arch} worker`, version);
  requireValue(
    typeof mac.signing?.expectedSigned === "boolean" &&
      typeof mac.signing?.expectedNotarized === "boolean" &&
      typeof mac.signing?.certificateSigned === "boolean" &&
      typeof mac.signing?.notarized === "boolean",
    `macOS ${arch} signing evidence is missing.`
  );
  if (mac.signing.expectedSigned) {
    requireValue(mac.signing.certificateSigned, `macOS ${arch} was expected to be signed.`);
  }
  if (mac.signing.expectedNotarized) {
    requireValue(mac.signing.notarized, `macOS ${arch} was expected to be notarized.`);
  }
  if (mac.signing.certificateSigned) {
    requireValue(mac.signing.signatureValid, `macOS ${arch} certificate signature is invalid.`);
  }
  if (mac.signing.notarized) {
    requireValue(mac.signing.certificateSigned, `macOS ${arch} notarization lacks signing.`);
  }
  const expectedMacSigningState = mac.signing.notarized
    ? "signed-and-notarized"
    : mac.signing.certificateSigned
      ? "signed-not-notarized"
      : "unsigned-signed-ready";
  requireValue(
    mac.signing.state === expectedMacSigningState,
    `macOS ${arch} signing label disagrees with its signature evidence.`
  );
  const expectedMacArtifacts = [
    `${product}-${version}-macOS-${arch}.dmg`,
    `${product}-${version}-macOS-${arch}.zip`
  ];
  requireValue(
    Array.isArray(mac.artifacts) && mac.artifacts.length === expectedMacArtifacts.length,
    `macOS ${arch} evidence has an unexpected artifact count.`
  );
  for (const name of expectedMacArtifacts) {
    validatePackagedCompliance(
      mac.artifactCompliance?.[name],
      "desktop-artifact-compliance",
      `macOS ${arch} ${name}`
    );
    await validateRecordedArtifact(
      mac.artifacts.find((artifact) => artifact.name === name),
      name,
      files,
      `macOS ${arch} evidence`
    );
  }
  platformStatus.macOS[arch] = {
    workerArchitecture: arch,
    signing: mac.signing.state
  };

  requireValue(
    linux.architecture === arch &&
      linux.platform === "linux" &&
      linux.desktopShellLaunched === true &&
      linux.neuralAcceptanceTested === false,
    `Linux ${arch} package lacks truthful desktop-shell evidence.`
  );
  requireValue(
    linux.formatValidation?.["tar.gz"]?.extracted === true &&
      linux.formatValidation?.deb?.extracted === true &&
      linux.formatValidation?.AppImage?.extracted === true,
    `Linux ${arch} format validation is incomplete.`
  );
  requireValue(
    linux.formatValidation.deb.architecture === (arch === "arm64" ? "arm64" : "amd64"),
    `Linux ${arch} DEB architecture evidence is incorrect.`
  );
  validateWorkerSmoke(linux.workerSmoke, `Linux ${arch} worker`, version);
  requireValue(
    linux.signing?.state === "not-applicable",
    `Linux ${arch} must label code signing as not applicable.`
  );
  const expectedLinuxArtifacts = [
    ...["AppImage", "deb", "tar.gz"].map((extension) =>
      releaseArtifactName({
        product,
        version,
        platform: "linux",
        architecture: arch,
        extension
      })
    )
  ];
  requireValue(
    Array.isArray(linux.artifacts) && linux.artifacts.length === expectedLinuxArtifacts.length,
    `Linux ${arch} evidence has an unexpected artifact count.`
  );
  for (const name of expectedLinuxArtifacts) {
    validatePackagedCompliance(
      linux.artifactCompliance?.[name],
      "desktop-artifact-compliance",
      `Linux ${arch} ${name}`
    );
    await validateRecordedArtifact(
      linux.artifacts.find((artifact) => artifact.name === name),
      name,
      files,
      `Linux ${arch} evidence`
    );
  }
  platformStatus.linux[arch] = {
    workerArchitecture: arch,
    signing: "not-applicable"
  };
}

const androidEvidenceName = "android-package-smoke.json";
const android = await readJson(files.get(androidEvidenceName), androidEvidenceName);
validateSourceCommit(android, sourceCommit, "Android package evidence");
validatePackagedCompliance(
  android.artifacts?.debug?.compliance,
  "mobile-artifact-compliance",
  "Android debug APK"
);
validatePackagedCompliance(
  android.artifacts?.release?.compliance,
  "mobile-artifact-compliance",
  "Android release APK"
);
requireValue(
  android?.schemaVersion === 1 &&
    android.platform === "android" &&
    android.version === version,
  "Android package evidence has an incompatible schema, platform, or version."
);
requireValue(
  android.sameBrainGateway?.protocolVersion === 1 &&
    android.sameBrainGateway?.emulatorTested === true &&
    android.sameBrainGateway?.streamedChat === true &&
    android.sameBrainGateway?.attachmentStreaming === true,
  "Android package lacks passing same-brain emulator, chat, or attachment evidence."
);
const expectedAndroidDebug = `Omni-AGI-Companion-${version}-Android-debug-signed.apk`;
const expectedAndroidRelease = `Omni-AGI-Companion-${version}-Android-release-unsigned.apk`;
requireValue(
  android.artifacts?.debug?.variant === "debug" &&
    android.artifacts.debug.installable === true &&
    android.artifacts.debug.zipAligned === true &&
    android.artifacts.debug.signing?.state === "debug-signed" &&
    android.artifacts.debug.signing?.verified === true,
  "Android debug APK lacks verified installable debug-signing evidence."
);
requireValue(
  android.artifacts?.release?.variant === "release" &&
    android.artifacts.release.installable === false &&
    android.artifacts.release.zipAligned === true &&
    android.artifacts.release.signing?.state === "unsigned-signed-ready" &&
    android.artifacts.release.signing?.verified === true,
  "Android release APK must be verified and explicitly labeled unsigned-signed-ready."
);
await validateRecordedArtifact(
  android.artifacts.debug,
  expectedAndroidDebug,
  files,
  "Android debug APK evidence"
);
await validateRecordedArtifact(
  android.artifacts.release,
  expectedAndroidRelease,
  files,
  "Android release APK evidence"
);
platformStatus.android = {
  emulatorTested: true,
  debugSigning: "debug-signed",
  releaseSigning: "unsigned-signed-ready"
};

const iosEvidenceName = "ios-package-smoke.json";
const ios = await readJson(files.get(iosEvidenceName), iosEvidenceName);
validateSourceCommit(ios, sourceCommit, "iOS package evidence");
validatePackagedCompliance(
  ios.artifact?.compliance,
  "mobile-artifact-compliance",
  "iOS IPA"
);
requireValue(
  ios?.schemaVersion === 1 && ios.platform === "ios" && ios.version === version,
  "iOS package evidence has an incompatible schema, platform, or version."
);
requireValue(
  ios.sameBrainGateway?.protocolVersion === 1 &&
    ios.sameBrainGateway?.emulatorTested === true &&
    ios.sameBrainGateway?.streamedChat === true &&
    ios.sameBrainGateway?.attachmentStreaming === true,
  "iOS package lacks passing same-brain simulator, chat, or attachment evidence."
);
requireValue(
  ios.artifact?.variant === "release" &&
    ios.artifact.installable === false &&
    ios.artifact.signing?.state === "unsigned-signed-ready" &&
    ios.artifact.signing?.verified === true,
  "iOS IPA must be verified and explicitly labeled unsigned-signed-ready."
);
const expectedIos = `Omni-AGI-Companion-${version}-iOS-unsigned.ipa`;
await validateRecordedArtifact(ios.artifact, expectedIos, files, "iOS IPA evidence");
platformStatus.iOS = {
  simulatorTested: true,
  signing: "unsigned-signed-ready"
};

requireValue(
  complianceReports.length === 17,
  `Release set requires 17 packaged compliance reports; found ${complianceReports.length}.`
);
const ffmpegPolicyHashes = new Set(
  complianceReports.map(({ report }) => report.ffmpegPolicySha256)
);
requireValue(
  ffmpegPolicyHashes.size === 1,
  "Every release artifact must use one identical nonempty FFmpeg policy hash."
);
const ffmpegPolicySha256 = [...ffmpegPolicyHashes][0];
const minimumLegalFilesVerified = Math.min(
  ...complianceReports.map(({ report }) => report.legalFilesVerified)
);
const checkedInFfmpegPolicySha256 = createHash("sha256")
  .update(await readFile(resolve("licenses/ffmpeg-runtime-policy.json")))
  .digest("hex");
requireValue(
  ffmpegPolicySha256 === checkedInFfmpegPolicySha256,
  "Packaged FFmpeg policy hash does not match the verified release source."
);

const deliverables = [...files.values()].sort((left, right) =>
  basename(left).localeCompare(basename(right))
);
const artifacts = [];
for (const path of deliverables) {
  artifacts.push({
    name: basename(path),
    bytes: (await stat(path)).size,
    sha256: await sha256(path)
  });
}
await writeFile(
  join(directory, "SHA256SUMS.txt"),
  `${artifacts.map((artifact) => `${artifact.sha256}  ${artifact.name}`).join("\n")}\n`,
  "utf8"
);
await writeFile(
  join(directory, "RELEASE-MANIFEST.json"),
  `${JSON.stringify(
    {
      schemaVersion: 1,
      product,
      version,
      tag: `v${version}`,
      sourceCommit,
      ffmpegPolicySha256,
      packagedCompliance: {
        schemaVersion: 1,
        artifactReports: complianceReports.length,
        minimumLegalFilesVerified,
        ffmpegExecutableBundled: false,
        ffmpegPolicySha256
      },
      artifactCount: artifacts.length,
      artifacts,
      platformStatus
    },
    null,
    2
  )}\n`,
  "utf8"
);
process.stdout.write(`Verified ${artifacts.length} release files for v${version}.\n`);
