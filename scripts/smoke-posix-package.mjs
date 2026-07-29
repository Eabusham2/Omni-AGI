import { createHash } from "node:crypto";
import { spawn } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, join, relative, resolve, sep } from "node:path";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function run(command, args, options = {}) {
  return new Promise((resolveRun, rejectRun) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: options.capture ? ["ignore", "pipe", "pipe"] : "inherit",
      shell: false
    });
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr?.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.once("error", rejectRun);
    child.once("exit", (code, signal) => {
      if (code === 0) resolveRun({ stdout, stderr });
      else {
        rejectRun(
          new Error(
            `${command} exited with code ${String(code)} and signal ${String(signal)}.\n${stderr}`
          )
        );
      }
    });
  });
}

function probe(command, args, options = {}) {
  return new Promise((resolveProbe, rejectProbe) => {
    const child = spawn(command, args, {
      cwd: options.cwd,
      env: options.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      shell: false
    });
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.once("error", rejectProbe);
    child.once("exit", (code, signal) => {
      resolveProbe({ code, signal, stdout, stderr });
    });
  });
}

async function walk(directory) {
  const result = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) result.push(...(await walk(path)));
    else if (entry.isFile()) result.push(path);
  }
  return result;
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

const platform = option("--platform");
const arch = option("--arch");
const releaseRoot = resolve(option("--release-root") ?? "release");
const desktopE2e = process.argv.includes("--desktop-e2e");
if (!["mac", "linux"].includes(platform) || !["x64", "arm64"].includes(arch)) {
  throw new Error(
    "Usage: node scripts/smoke-posix-package.mjs --platform <mac|linux> --arch <x64|arm64> [--desktop-e2e]"
  );
}

const expectedPlatform = platform === "mac" ? "darwin" : "linux";
if (process.platform !== expectedPlatform) {
  throw new Error(`A ${platform} package must be smoked on ${expectedPlatform}.`);
}
function expectation(name) {
  const value = (process.env[name] ?? "0").trim();
  if (!["0", "1"].includes(value)) throw new Error(`${name} must be 0 or 1.`);
  return value === "1";
}
const expectedSigned = expectation("OMNI_EXPECT_SIGNED");
const expectedNotarized = expectation("OMNI_EXPECT_NOTARIZED");
if (expectedNotarized && !expectedSigned) {
  throw new Error("A notarized package must also be expected to be certificate-signed.");
}
if (platform === "linux" && (expectedSigned || expectedNotarized)) {
  throw new Error("Linux packages do not use the macOS signing expectations.");
}

const packageDocument = JSON.parse(
  await import("node:fs/promises").then(({ readFile }) =>
    readFile(resolve("package.json"), "utf8")
  )
);
const platformLabel = platform === "mac" ? "macOS" : "Linux";
const extensions = platform === "mac" ? ["dmg", "zip"] : ["AppImage", "deb", "tar.gz"];
const files = await walk(releaseRoot);
const artifacts = [];
for (const extension of extensions) {
  const expectedName = `${packageDocument.build.productName}-${packageDocument.version}-${platformLabel}-${arch}.${extension}`;
  const matches = files.filter((path) => basename(path) === expectedName);
  if (matches.length !== 1) {
    throw new Error(`Expected exactly one ${expectedName}; found ${matches.length}.`);
  }
  if ((await stat(matches[0])).size === 0) {
    throw new Error(`${expectedName} is empty.`);
  }
  artifacts.push(matches[0]);
}

const scratch = await mkdtemp(join(tmpdir(), `omni-${platform}-${arch}-smoke-`));
const payloadRoot = join(scratch, "payload");
await mkdir(payloadRoot, { recursive: true });
const sourceArchive =
  platform === "mac"
    ? artifacts.find((path) => path.endsWith(".zip"))
    : artifacts.find((path) => path.endsWith(".tar.gz"));
if (!sourceArchive) throw new Error(`No portable ${platform} archive was produced.`);
if (platform === "mac") {
  const dmg = artifacts.find((path) => path.endsWith(".dmg"));
  if (!dmg) throw new Error("No macOS DMG was produced.");
  await run("hdiutil", ["verify", dmg], { capture: true });
  await run("ditto", ["-x", "-k", sourceArchive, payloadRoot]);
} else {
  await run("tar", ["-xzf", sourceArchive, "-C", payloadRoot]);
}
const payloadFiles = await walk(payloadRoot);

const workers = payloadFiles.filter(
  (path) =>
    basename(path) === "omni-engine" &&
    path.split(sep).includes("engine-runtime")
);
if (workers.length !== 1) {
  throw new Error(`Expected one archived ${platform}-${arch} neural worker; found ${workers.length}.`);
}
const worker = workers[0];

const apps =
  platform === "mac"
    ? payloadFiles.filter(
        (path) =>
          basename(path) === packageDocument.build.productName &&
          path.includes(`${packageDocument.build.productName}.app${sep}Contents${sep}MacOS`)
      )
    : payloadFiles.filter(
        (path) =>
          basename(path) === packageDocument.build.linux.executableName
      );
if (apps.length !== 1) {
  throw new Error(`Expected one archived ${platform}-${arch} desktop executable; found ${apps.length}.`);
}
const desktopExecutable = apps[0];

for (const executable of [worker, desktopExecutable]) {
  if (!existsSync(executable)) throw new Error(`Missing packaged executable ${executable}`);
  const inspection = await run("file", [executable], { capture: true });
  const pattern = arch === "arm64" ? /(arm64|aarch64)/i : /(x86[_-]64|x86-64)/i;
  if (!pattern.test(inspection.stdout)) {
    throw new Error(
      `${relative(payloadRoot, executable)} does not report ${arch}: ${inspection.stdout.trim()}`
    );
  }
}

const formatValidation = {};
let signing = {
  expectedSigned: false,
  expectedNotarized: false,
  signatureValid: false,
  certificateSigned: false,
  notarized: false,
  state: "not-applicable"
};
if (platform === "mac") {
  const signatureCheck = await probe(
    "codesign",
    ["--verify", "--deep", "--strict", "--verbose=2", desktopExecutable]
  );
  const signatureDetails = await probe(
    "codesign",
    ["--display", "--verbose=4", desktopExecutable]
  );
  const signatureText = `${signatureDetails.stdout}\n${signatureDetails.stderr}`;
  const authority = signatureText.match(/^Authority=(.+)$/m)?.[1]?.trim() ?? "";
  const teamIdentifier =
    signatureText.match(/^TeamIdentifier=(.+)$/m)?.[1]?.trim() ?? "";
  const adHoc = /^Signature=adhoc$/m.test(signatureText);
  const signatureValid = signatureCheck.code === 0;
  const certificateSigned =
    signatureValid &&
    !adHoc &&
    authority.startsWith("Developer ID Application:") &&
    teamIdentifier.length > 0;
  const staplerCheck = await probe(
    "xcrun",
    ["stapler", "validate", desktopExecutable]
  );
  const notarized = staplerCheck.code === 0;
  if (expectedSigned && !certificateSigned) {
    throw new Error(
      "macOS signing credentials were configured, but the archived app lacks a valid certificate signature."
    );
  }
  if (expectedNotarized && !notarized) {
    throw new Error(
      "macOS notarization credentials were configured, but the archived app lacks a stapled ticket."
    );
  }
  if (notarized && !certificateSigned) {
    throw new Error("The archived macOS app reports notarization without a certificate signature.");
  }
  signing = {
    expectedSigned,
    expectedNotarized,
    signatureValid,
    certificateSigned,
    notarized,
    authority,
    teamIdentifier,
    state: notarized
      ? "signed-and-notarized"
      : certificateSigned
        ? "signed-not-notarized"
        : "unsigned-signed-ready"
  };
  formatValidation.dmg = { verified: true };
  formatValidation.zip = { extracted: true };
} else {
  const deb = artifacts.find((path) => path.endsWith(".deb"));
  const appImage = artifacts.find((path) => path.endsWith(".AppImage"));
  if (!deb || !appImage) {
    throw new Error("Linux package validation requires DEB and AppImage artifacts.");
  }
  const expectedDebArchitecture = arch === "arm64" ? "arm64" : "amd64";
  const debArchitecture = (
    await run("dpkg-deb", ["--field", deb, "Architecture"], { capture: true })
  ).stdout.trim();
  if (debArchitecture !== expectedDebArchitecture) {
    throw new Error(
      `DEB reports ${debArchitecture} instead of ${expectedDebArchitecture}.`
    );
  }
  const debRoot = join(scratch, "deb");
  await mkdir(debRoot, { recursive: true });
  await run("dpkg-deb", ["--extract", deb, debRoot], { capture: true });
  const debFiles = await walk(debRoot);
  const debWorkers = debFiles.filter(
    (path) =>
      basename(path) === "omni-engine" &&
      path.split(sep).includes("engine-runtime")
  );
  const debApps = debFiles.filter(
    (path) => basename(path) === packageDocument.build.linux.executableName
  );
  if (debWorkers.length !== 1 || debApps.length !== 1) {
    throw new Error("DEB must contain exactly one desktop and one neural worker executable.");
  }
  for (const executable of [debWorkers[0], debApps[0]]) {
    const inspection = await run("file", [executable], { capture: true });
    const pattern = arch === "arm64" ? /(arm64|aarch64)/i : /(x86[_-]64|x86-64)/i;
    if (!pattern.test(inspection.stdout)) {
      throw new Error(`DEB executable has the wrong architecture: ${inspection.stdout.trim()}`);
    }
  }

  const appImageRoot = join(scratch, "appimage");
  await mkdir(appImageRoot, { recursive: true });
  await run(appImage, ["--appimage-extract"], {
    capture: true,
    cwd: appImageRoot
  });
  const appImageFiles = await walk(join(appImageRoot, "squashfs-root"));
  const appImageWorkers = appImageFiles.filter(
    (path) =>
      basename(path) === "omni-engine" &&
      path.split(sep).includes("engine-runtime")
  );
  const appImageApps = appImageFiles.filter(
    (path) => basename(path) === packageDocument.build.linux.executableName
  );
  if (appImageWorkers.length !== 1 || appImageApps.length !== 1) {
    throw new Error("AppImage must contain exactly one desktop and one neural worker executable.");
  }
  for (const executable of [appImageWorkers[0], appImageApps[0]]) {
    const inspection = await run("file", [executable], { capture: true });
    const pattern = arch === "arm64" ? /(arm64|aarch64)/i : /(x86[_-]64|x86-64)/i;
    if (!pattern.test(inspection.stdout)) {
      throw new Error(
        `AppImage executable has the wrong architecture: ${inspection.stdout.trim()}`
      );
    }
  }
  formatValidation["tar.gz"] = { extracted: true };
  formatValidation.deb = {
    extracted: true,
    architecture: debArchitecture
  };
  formatValidation.AppImage = { extracted: true };
}

const workerSmoke = await run(
  process.execPath,
  [
    resolve("scripts/smoke-engine.mjs"),
    "--executable",
    worker,
    "--brain-root",
    join(scratch, "brain")
  ],
  { capture: true, cwd: resolve(".") }
);
const workerEvidence = JSON.parse(workerSmoke.stdout);
if (workerEvidence.operatingSystem !== expectedPlatform) {
  throw new Error(
    `Archived worker reported ${String(workerEvidence.operatingSystem)} instead of ${expectedPlatform}.`
  );
}

if (desktopE2e) {
  const environment = {
    ...process.env,
    CI: process.env.CI ?? "1",
    OMNI_E2E_EXECUTABLE: desktopExecutable
  };
  if (platform === "linux") {
    await run("xvfb-run", ["-a", "npm", "run", "test:ui:built"], {
      cwd: resolve("."),
      env: environment
    });
  } else {
    await run("npm", ["run", "test:ui:built"], {
      cwd: resolve("."),
      env: environment
    });
  }
}

const artifactEvidence = [];
for (const artifact of artifacts) {
  artifactEvidence.push({
    name: basename(artifact),
    bytes: (await stat(artifact)).size,
    sha256: await sha256(artifact)
  });
}
const evidence = {
  platform,
  architecture: arch,
  artifacts: artifactEvidence,
  verifiedArchive: basename(sourceArchive),
  packagedWorker: relative(payloadRoot, worker),
  desktopExecutable: relative(payloadRoot, desktopExecutable),
  workerSmoke: workerEvidence,
  desktopEndToEnd: desktopE2e,
  formatValidation,
  signing
};
await mkdir(releaseRoot, { recursive: true });
const evidencePath = join(releaseRoot, `${platform}-package-smoke-${arch}.json`);
await writeFile(evidencePath, `${JSON.stringify(evidence, null, 2)}\n`, "utf8");
await rm(scratch, { recursive: true, force: true });
process.stdout.write(`Package smoke evidence: ${evidencePath}\n`);
