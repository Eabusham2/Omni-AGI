import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, readdir, stat, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { releaseArtifactName } from "./release-artifact-names.mjs";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
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
  return names;
}

async function validateRecordedArtifact(record, expectedName, files, label) {
  requireValue(record && typeof record === "object", `${label} is missing.`);
  requireValue(record.name === expectedName, `${label} names ${String(record.name)}, not ${expectedName}.`);
  const path = files.get(expectedName);
  requireValue(path, `${expectedName} is absent from the release set.`);
  const metadata = await stat(path);
  const actualHash = await sha256(path);
  requireValue(
    typeof record.sha256 === "string" && record.sha256.toLowerCase() === actualHash,
    `${label} hash does not match ${expectedName}.`
  );
  if (record.bytes !== undefined) {
    requireValue(record.bytes === metadata.size, `${label} size does not match ${expectedName}.`);
  }
  return { name: expectedName, bytes: metadata.size, sha256: actualHash };
}

function validateWorkerSmoke(smoke, label) {
  requireValue(smoke?.persistedBrain === true, `${label} did not persist a brain.`);
  requireValue(smoke?.safeTensorCheckpoint === true, `${label} did not verify safe tensors.`);
  requireValue(smoke?.sqliteEventLog === true, `${label} did not verify SQLite events.`);
  requireValue(smoke?.protocolVersion === 1, `${label} used an incompatible protocol.`);
  requireValue(
    typeof smoke?.engineVersion === "string" && smoke.engineVersion.length > 0,
    `${label} omitted the engine version.`
  );
}

const directory = resolve(option("--directory") ?? "artifacts");
const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const product = packageDocument.build.productName;
const version = packageDocument.version;
const expected = requiredNames(product, version);
const generatedMetadata = new Set(["SHA256SUMS.txt", "RELEASE-MANIFEST.json"]);
const allowed = new Set([...expected, ...generatedMetadata]);

const entries = await readdir(directory, { withFileTypes: true });
const nonFiles = entries.filter((entry) => !entry.isFile()).map((entry) => entry.name);
requireValue(
  nonFiles.length === 0,
  `Release directory contains non-file entries: ${nonFiles.join(", ")}.`
);
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
  linux: {}
};
for (const arch of ["x64", "arm64"]) {
  const windowsName = `windows-package-smoke-${arch}.json`;
  const macName = `mac-package-smoke-${arch}.json`;
  const linuxName = `linux-package-smoke-${arch}.json`;
  const windows = await readJson(files.get(windowsName), windowsName);
  const mac = await readJson(files.get(macName), macName);
  const linux = await readJson(files.get(linuxName), linuxName);

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
    windows.nsis?.desktopEndToEnd === true &&
      windows.nsis?.desktopRestart === true &&
      windows.nsis?.accessibilityNavigation === true &&
      windows.nsis?.modalityGeneration === true,
    `Windows ${arch} package lacks passing installed-desktop evidence.`
  );
  validateWorkerSmoke(windows.zip?.rpcSmoke, `Windows ${arch} ZIP worker`);
  validateWorkerSmoke(windows.nsis?.rpcSmoke, `Windows ${arch} installed worker`);
  requireValue(
    windows.nsis.rpcSmoke?.comprehensive === true &&
      windows.nsis.rpcSmoke?.trainingLossDecreased === true &&
      windows.nsis.rpcSmoke?.pdfIngested === true &&
      windows.nsis.rpcSmoke?.chatParameterMutation === true &&
      ["image", "audio", "video"].every((kind) =>
        windows.nsis.rpcSmoke?.generatedModalities?.includes(kind)
      ),
    `Windows ${arch} installed worker lacks comprehensive neural evidence.`
  );
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
      mac.desktopEndToEnd === true,
    `macOS ${arch} package lacks passing desktop evidence.`
  );
  requireValue(
    mac.formatValidation?.dmg?.verified === true &&
      mac.formatValidation?.zip?.extracted === true,
    `macOS ${arch} format validation is incomplete.`
  );
  validateWorkerSmoke(mac.workerSmoke, `macOS ${arch} worker`);
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
      linux.desktopEndToEnd === true,
    `Linux ${arch} package lacks passing desktop evidence.`
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
  validateWorkerSmoke(linux.workerSmoke, `Linux ${arch} worker`);
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
