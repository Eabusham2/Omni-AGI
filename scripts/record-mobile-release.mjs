import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import { createReadStream } from "node:fs";
import { open, readFile, stat, writeFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { verifyMobileCompliance } from "./verify-packaged-compliance.mjs";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

function checkedOutCommit() {
  const result = spawnSync(
    "git",
    ["rev-parse", "--verify", "HEAD^{commit}"],
    { cwd: resolve("."), encoding: "utf8", shell: false }
  );
  requireValue(
    result.status === 0,
    `Could not resolve the mobile package source commit: ${result.stderr || result.stdout}`
  );
  const commit = result.stdout.trim().toLowerCase();
  requireValue(
    /^[a-f0-9]{40}$/u.test(commit),
    "The checked-out mobile package source commit is invalid."
  );
  return commit;
}

function validatedCompliance(report, platform, label) {
  requireValue(
    report?.schemaVersion === 1 &&
      report?.kind === "mobile-artifact-compliance" &&
      report?.platform === platform &&
      Number.isInteger(report?.legalFilesVerified) &&
      report.legalFilesVerified >= 10 &&
      report?.ffmpegExecutableBundled === false &&
      typeof report?.ffmpegPolicySha256 === "string" &&
      /^[a-f0-9]{64}$/u.test(report.ffmpegPolicySha256),
    `${label} did not pass packaged mobile legal and FFmpeg compliance.`
  );
  return report;
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

async function inspectArchive(path, expectedName) {
  requireValue(typeof path === "string" && path.length > 0, `Missing artifact ${expectedName}.`);
  const resolved = resolve(path);
  requireValue(basename(resolved) === expectedName, `${resolved} must be named ${expectedName}.`);
  const metadata = await stat(resolved);
  requireValue(metadata.isFile() && metadata.size > 4, `${resolved} is not a non-empty file.`);
  const header = Buffer.alloc(4);
  const handle = await open(resolved, "r");
  try {
    await handle.read(header, 0, header.length, 0);
  } finally {
    await handle.close();
  }
  const signature = header.toString("hex");
  requireValue(
    ["504b0304", "504b0506", "504b0708"].includes(signature),
    `${resolved} is not a ZIP-based APK/IPA container.`
  );
  return {
    name: expectedName,
    bytes: metadata.size,
    sha256: await sha256(resolved)
  };
}

const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const version = packageDocument.version;
const platform = option("--platform");
const output = resolve(option("--output") ?? `${platform}-package-smoke.json`);
const actualSourceCommit = checkedOutCommit();
const sourceCommit = String(
  option("--source-commit") ??
    process.env.OMNI_RELEASE_COMMIT ??
    process.env.GITHUB_SHA ??
    actualSourceCommit
).trim().toLowerCase();

requireValue(platform === "android" || platform === "ios", "--platform must be android or ios.");
requireValue(
  /^[a-f0-9]{40}$/u.test(sourceCommit),
  "--source-commit must be a full 40-hex Git commit SHA."
);
requireValue(
  sourceCommit === actualSourceCommit,
  `Mobile package source commit ${actualSourceCommit} does not match verified commit ${sourceCommit}.`
);
requireValue(option("--emulator-tested") === "1", "A passing emulator test is required.");
requireValue(option("--same-brain-chat-tested") === "1", "Same-brain streamed chat evidence is required.");
requireValue(
  option("--attachment-stream-tested") === "1",
  "Same-brain attachment streaming evidence is required."
);

let record;
if (platform === "android") {
  requireValue(option("--debug-signature-verified") === "1", "The debug APK signature was not verified.");
  requireValue(
    option("--release-unsigned-verified") === "1",
    "The release APK unsigned state was not verified."
  );
  requireValue(option("--zipalign-verified") === "1", "Android zip alignment was not verified.");
  const debugName = `Omni-AGI-Companion-${version}-Android-debug-signed.apk`;
  const releaseName = `Omni-AGI-Companion-${version}-Android-release-unsigned.apk`;
  const debug = await inspectArchive(option("--debug-artifact"), debugName);
  const release = await inspectArchive(option("--release-artifact"), releaseName);
  const debugCompliance = validatedCompliance(
    await verifyMobileCompliance(option("--debug-artifact"), "android"),
    "android",
    "Android debug APK"
  );
  const releaseCompliance = validatedCompliance(
    await verifyMobileCompliance(option("--release-artifact"), "android"),
    "android",
    "Android release APK"
  );
  requireValue(
    debugCompliance.ffmpegPolicySha256 === releaseCompliance.ffmpegPolicySha256,
    "Android APK variants were verified against different FFmpeg policies."
  );
  record = {
    schemaVersion: 1,
    sourceCommit,
    platform: "android",
    version,
    sameBrainGateway: {
      protocolVersion: 1,
      emulatorTested: true,
      streamedChat: true,
      attachmentStreaming: true
    },
    artifacts: {
      debug: {
        ...debug,
        compliance: debugCompliance,
        variant: "debug",
        installable: true,
        zipAligned: true,
        signing: { state: "debug-signed", verified: true }
      },
      release: {
        ...release,
        compliance: releaseCompliance,
        variant: "release",
        installable: false,
        zipAligned: true,
        signing: { state: "unsigned-signed-ready", verified: true }
      }
    }
  };
} else {
  requireValue(option("--unsigned-verified") === "1", "The IPA unsigned state was not verified.");
  const ipaName = `Omni-AGI-Companion-${version}-iOS-unsigned.ipa`;
  const ipa = await inspectArchive(option("--artifact"), ipaName);
  const compliance = validatedCompliance(
    await verifyMobileCompliance(option("--artifact"), "ios"),
    "ios",
    "iOS IPA"
  );
  record = {
    schemaVersion: 1,
    sourceCommit,
    platform: "ios",
    version,
    sameBrainGateway: {
      protocolVersion: 1,
      emulatorTested: true,
      streamedChat: true,
      attachmentStreaming: true
    },
    artifact: {
      ...ipa,
      compliance,
      variant: "release",
      installable: false,
      signing: { state: "unsigned-signed-ready", verified: true }
    }
  };
}

await writeFile(output, `${JSON.stringify(record, null, 2)}\n`, "utf8");
process.stdout.write(`Recorded ${platform} mobile release evidence at ${output}.\n`);
