import { execFileSync } from "node:child_process";
import { access, readFile } from "node:fs/promises";
import { verifyReleaseTagBinding } from "./release-tag-state.mjs";

function option(name) {
  const inline = process.argv.find((entry) => entry.startsWith(`${name}=`));
  if (inline) return inline.slice(name.length + 1);
  const index = process.argv.indexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function requireValue(condition, message) {
  if (!condition) throw new Error(message);
}

const packageDocument = JSON.parse(await readFile("package.json", "utf8"));
const packageLock = JSON.parse(await readFile("package-lock.json", "utf8"));
const expectedTag = `v${packageDocument.version}`;
const requestedTag = option("--tag") ?? process.env.OMNI_RELEASE_TAG;

requireValue(
  /^\d+\.\d+\.\d+$/.test(packageDocument.version),
  `Release version must be a stable semantic version, received ${packageDocument.version}.`
);
requireValue(
  packageDocument.version.split(".")[0] === "1",
  `Stable Omni AGI Studio v1 releases must use major version 1, received ${packageDocument.version}.`
);
requireValue(
  packageLock.version === packageDocument.version &&
    packageLock.packages?.[""]?.version === packageDocument.version,
  "package-lock.json root versions must match package.json."
);
if (requestedTag) {
  requireValue(
    requestedTag === expectedTag,
    `Release tag ${requestedTag} must exactly match package version ${expectedTag}.`
  );
  verifyReleaseTagBinding(requestedTag);
}
requireValue(packageDocument.build?.asar === true, "Release packages must enable ASAR.");
requireValue(
  packageDocument.build?.appId === "ai.omniagi.studio",
  "Release appId must remain ai.omniagi.studio."
);
requireValue(
  packageDocument.build?.mac?.hardenedRuntime === true &&
    packageDocument.build?.mac?.notarize === false &&
    packageDocument.build?.mac?.entitlements === "build/entitlements.mac.plist" &&
    packageDocument.build?.mac?.entitlementsInherit === "build/entitlements.mac.plist",
  "macOS must be hardened and use the checked-in entitlements; wrappers opt into notarization."
);
requireValue(
  packageDocument.build?.dmg?.sign === false,
  "The DMG container must remain unsigned; the app bundle is the signed/notarized payload."
);

for (const [platform, expectedTargets] of Object.entries({
  win: ["nsis", "zip"],
  mac: ["dmg", "zip"],
  linux: ["AppImage", "deb", "tar.gz"]
})) {
  const targets = packageDocument.build?.[platform]?.target;
  requireValue(
    Array.isArray(targets) && expectedTargets.every((target) => targets.includes(target)),
    `${platform} packaging must include ${expectedTargets.join(", ")}.`
  );
}
for (const script of [
  "package:win",
  "package:win:arm64",
  "package:mac:x64",
  "package:mac:arm64",
  "package:linux:x64",
  "package:linux:arm64",
  "package:android:debug",
  "test:android",
  "package:ios:unsigned",
  "package:ios:signed",
  "test:ios"
]) {
  requireValue(typeof packageDocument.scripts?.[script] === "string", `Missing ${script}.`);
}
requireValue(
  packageDocument.build?.extraResources?.some(
    (resource) =>
      resource.from === "engine-dist/omni-engine" &&
      resource.to === "engine-runtime"
  ),
  "Every package must include the self-contained engine runtime."
);
for (const resource of packageDocument.build?.extraResources ?? []) {
  requireValue(typeof resource.from === "string", "extraResources entries need a source path.");
  if (resource.from === "engine-dist/omni-engine") continue;
  try {
    await access(resource.from);
  } catch {
    throw new Error(`Release resource ${resource.from} does not exist.`);
  }
}

const androidBuild = await readFile("mobile/android/app/build.gradle.kts", "utf8");
const [major, minor, patch] = packageDocument.version.split(".").map(Number);
requireValue(
  minor < 100 && patch < 100,
  "Android versionCode mapping requires minor and patch versions below 100."
);
const androidVersionCode = major * 10_000 + minor * 100 + patch;
requireValue(
  androidBuild.includes(`versionCode = ${androidVersionCode}`) &&
    androidBuild.includes(`versionName = "${packageDocument.version}"`),
  `Android versionCode/versionName must match ${packageDocument.version}.`
);

const iosProject = await readFile(
  "mobile/ios/OmniCompanion.xcodeproj/project.pbxproj",
  "utf8"
);
const iosMarketingVersions = [
  ...iosProject.matchAll(/\bMARKETING_VERSION = ([^;]+);/gu)
].map((match) => match[1]);
const iosBuildNumbers = [
  ...iosProject.matchAll(/\bCURRENT_PROJECT_VERSION = ([^;]+);/gu)
].map((match) => match[1]);
requireValue(
  iosMarketingVersions.length === 2 &&
    iosMarketingVersions.every((version) => version === packageDocument.version),
  `Every iOS app configuration must use MARKETING_VERSION ${packageDocument.version}.`
);
requireValue(
  iosBuildNumbers.length === 2 &&
    iosBuildNumbers.every((build) => build === String(androidVersionCode)),
  `Every iOS app configuration must use build number ${androidVersionCode} for ${packageDocument.version}.`
);
const iosInfo = await readFile("mobile/ios/OmniCompanion/Info.plist", "utf8");
requireValue(
  iosInfo.includes("<string>$(MARKETING_VERSION)</string>") &&
    iosInfo.includes("<string>$(CURRENT_PROJECT_VERSION)</string>"),
  "The iOS bundle must source its marketing and build versions from the verified project settings."
);

const enginePackage = await readFile("engine/omni_core/__init__.py", "utf8");
requireValue(
  enginePackage.includes(`__version__ = "${packageDocument.version}"`),
  `The packaged engine version must match ${packageDocument.version}.`
);
const mcpClient = await readFile("src/main/mcpClient.ts", "utf8");
requireValue(
  mcpClient.includes(
    `clientInfo: { name: "Omni AGI Studio", version: "${packageDocument.version}" }`
  ),
  `The MCP client version must match ${packageDocument.version}.`
);
const engineRequirements = await readFile("engine/requirements.txt", "utf8");
for (const constraint of [
  'torch==2.2.2; platform_system == "Darwin" and platform_machine == "x86_64"',
  'torch==2.10.0; platform_system == "Darwin" and platform_machine == "arm64"',
  'torch==2.10.0; platform_system != "Darwin"',
  'numpy==1.26.4; platform_system == "Darwin" and platform_machine == "x86_64"',
  'numpy==2.2.6; platform_system != "Darwin" or platform_machine != "x86_64"',
  'pyarrow==17.0.0; platform_system == "Darwin" and platform_machine == "x86_64"',
  'pyarrow==22.0.0; platform_system != "Darwin" or platform_machine != "x86_64"'
]) {
  requireValue(
    engineRequirements.includes(constraint),
    `Engine requirements are missing the native runtime constraint ${constraint}.`
  );
}
const lockPython =
  process.env.OMNI_BUILD_PYTHON ?? (process.platform === "win32" ? "python" : "python3");
try {
  execFileSync(lockPython, ["scripts/install-engine-lock.py", "--verify-locks"], {
    cwd: process.cwd(),
    stdio: "pipe"
  });
} catch (error) {
  const detail = error instanceof Error ? error.message : String(error);
  throw new Error(`Engine release dependency locks failed verification: ${detail}`);
}

const packagedComplianceVerifier = await readFile(
  "scripts/verify-packaged-compliance.mjs",
  "utf8"
);
const releaseArtifactVerifier = await readFile(
  "scripts/verify-release-artifacts.mjs",
  "utf8"
);
const posixPackageSmoke = await readFile(
  "scripts/smoke-posix-package.mjs",
  "utf8"
);
const windowsPackageSmoke = await readFile(
  "scripts/smoke-windows-package.ps1",
  "utf8"
);
const mobileReleaseRecorder = await readFile(
  "scripts/record-mobile-release.mjs",
  "utf8"
);
for (const fragment of [
  "export async function verifyDesktopCompliance",
  "export async function verifyMobileCompliance",
  'kind: "desktop-artifact-compliance"',
  'kind: "mobile-artifact-compliance"',
  "ffmpegExecutableBundled: false",
  "ffmpegPolicySha256"
]) {
  requireValue(
    packagedComplianceVerifier.includes(fragment),
    `Packaged compliance verifier is missing ${fragment}.`
  );
}
requireValue(
  posixPackageSmoke.includes("verifyDesktopCompliance") &&
    windowsPackageSmoke.includes("verify-packaged-compliance.mjs") &&
    mobileReleaseRecorder.includes("verifyMobileCompliance") &&
    releaseArtifactVerifier.includes("complianceReports.length === 17") &&
    releaseArtifactVerifier.includes("legalFilesVerified >= 10") &&
    releaseArtifactVerifier.includes("ffmpegExecutableBundled === false") &&
    releaseArtifactVerifier.includes("ffmpegPolicySha256"),
  "Every final desktop/mobile artifact must carry verified legal and FFmpeg compliance evidence."
);

const workflowPaths = [
  ".github/workflows/windows.yml",
  ".github/workflows/macos.yml",
  ".github/workflows/linux.yml",
  ".github/workflows/android.yml",
  ".github/workflows/ios.yml",
  ".github/workflows/release.yml"
];
const workflows = new Map();
for (const path of workflowPaths) {
  workflows.set(path, await readFile(path, "utf8"));
}

function workflowJob(path, jobName) {
  const source = workflows.get(path);
  const lines = source.split(/\r?\n/u);
  const start = lines.findIndex((line) => line === `  ${jobName}:`);
  requireValue(start >= 0, `${path} is missing the ${jobName} job.`);
  let end = start + 1;
  while (end < lines.length && !/^  [a-zA-Z0-9_-]+:\s*$/u.test(lines[end])) end += 1;
  return lines.slice(start, end).join("\n");
}

function workflowStep(path, jobName, stepName) {
  const job = workflowJob(path, jobName);
  const lines = job.split(/\r?\n/u);
  const marker = `      - name: ${stepName}`;
  const start = lines.findIndex((line) => line === marker);
  requireValue(start >= 0, `${path} ${jobName} is missing the ${stepName} step.`);
  let end = start + 1;
  while (end < lines.length && !lines[end].startsWith("      - ")) end += 1;
  return lines.slice(start, end).join("\n");
}

function architectureEntry(path, jobName, architecture) {
  const job = workflowJob(path, jobName);
  const lines = job.split(/\r?\n/u);
  const marker = `          - arch: ${architecture}`;
  const start = lines.findIndex((line) => line === marker);
  requireValue(start >= 0, `${path} ${jobName} is missing its ${architecture} matrix entry.`);
  let end = start + 1;
  while (
    end < lines.length &&
    lines[end] !== "    steps:" &&
    !lines[end].startsWith("          - arch: ")
  ) {
    end += 1;
  }
  return lines.slice(start, end).join("\n");
}

function requireArchitecture(path, jobName, architecture, fields) {
  const entry = architectureEntry(path, jobName, architecture);
  for (const [field, value] of Object.entries(fields)) {
    requireValue(
      entry.includes(`            ${field}: ${value}`),
      `${path} ${jobName} ${architecture} must set ${field}: ${value}.`
    );
  }
}

for (const [path, source] of workflows) {
  const remoteActions = [
    ...source.matchAll(/uses:\s+[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+@([^\s#]+)/g)
  ];
  requireValue(remoteActions.length > 0, `${path} does not use the expected remote actions.`);
  for (const match of remoteActions) {
    requireValue(
      /^[a-f0-9]{40}$/.test(match[1]),
      `${path} must pin ${match[0]} to a full commit SHA.`
    );
  }
  const uploadCount = (source.match(/uses:\s+actions\/upload-artifact@/g) ?? []).length;
  const noRecompressionCount = (source.match(/compression-level:\s+0/g) ?? []).length;
  requireValue(
    uploadCount === noRecompressionCount,
    `${path} must disable redundant compression for every pre-compressed package upload.`
  );
}
for (const [path, runner] of [
  [".github/workflows/windows.yml", "windows-latest"],
  [".github/workflows/windows.yml", "windows-11-arm"],
  [".github/workflows/macos.yml", "macos-15-intel"],
  [".github/workflows/macos.yml", "macos-15"],
  [".github/workflows/linux.yml", "ubuntu-24.04"],
  [".github/workflows/linux.yml", "ubuntu-24.04-arm"]
]) {
  requireValue(workflows.get(path).includes(`runner: ${runner}`), `${path} is missing ${runner}.`);
}
for (const [path, runner] of [
  [".github/workflows/android.yml", "ubuntu-24.04"],
  [".github/workflows/ios.yml", "macos-15"]
]) {
  requireValue(
    workflows.get(path).includes(`runs-on: ${runner}`),
    `${path} is missing ${runner}.`
  );
}

for (const [path, jobName] of [
  [".github/workflows/android.yml", "apk-and-emulator"],
  [".github/workflows/release.yml", "android"]
]) {
  const job = workflowJob(path, jobName);
  for (const requiredFragment of [
    "actions/setup-node@",
    "node-version: 22",
    "actions/setup-java@",
    'java-version: "17"',
    "verify-signature: true",
    "mobile/android/gradle.properties",
    "android-actions/setup-android@",
    'cmdline-tools-version: "14742923"',
    'packages: ""',
    '"platforms;android-35"',
    '"build-tools;35.0.0"',
    '"system-images;android-35;google_apis;x86_64"',
    ":app:testDebugUnitTest",
    ":app:assembleDebug",
    ":app:assembleRelease",
    ":app:connectedDebugAndroidTest",
    "record-mobile-release.mjs"
  ]) {
    requireValue(
      job.includes(requiredFragment),
      `${path} ${jobName} is missing reproducible Android gate ${requiredFragment}.`
    );
  }
}

for (const [path, jobName] of [
  [".github/workflows/ios.yml", "ipa-and-simulator"],
  [".github/workflows/release.yml", "ios"]
]) {
  const job = workflowJob(path, jobName);
  for (const requiredFragment of [
    "DEVELOPER_DIR: /Applications/Xcode_16.4.app/Contents/Developer",
    'test "$(xcodebuild -version | head -n 1)" = "Xcode 16.4"',
    "com.apple.CoreSimulator.SimRuntime.iOS-18-5",
    "xcrun simctl bootstatus",
    "npm run test:ios",
    "npm run package:ios:unsigned",
    "codesign --verify --deep --strict",
    "record-mobile-release.mjs"
  ]) {
    requireValue(
      job.includes(requiredFragment),
      `${path} ${jobName} is missing reproducible iOS gate ${requiredFragment}.`
    );
  }
}

for (const [path, jobName] of [
  [".github/workflows/windows.yml", "test-and-package"],
  [".github/workflows/release.yml", "windows"]
]) {
  requireArchitecture(path, jobName, "x64", {
    runner: "windows-latest",
    node_arch: "x64",
    python_arch: "x64",
    worker_arch: "x64"
  });
  requireArchitecture(path, jobName, "arm64", {
    runner: "windows-11-arm",
    node_arch: "arm64",
    python_arch: "x64",
    worker_arch: "x64"
  });
  requireValue(
    workflowJob(path, jobName).includes('-ExpectedWorkerArch "${{ matrix.worker_arch }}"'),
    `${path} ${jobName} must smoke the declared Windows worker architecture.`
  );
}

for (const [path, jobName, entries] of [
  [
    ".github/workflows/macos.yml",
    "test-and-package",
    [
      ["x64", { runner: "macos-15-intel", node_arch: "x64", python_arch: "x64" }],
      ["arm64", { runner: "macos-15", node_arch: "arm64", python_arch: "arm64" }]
    ]
  ],
  [
    ".github/workflows/release.yml",
    "macos",
    [
      ["x64", { runner: "macos-15-intel", node_arch: "x64", python_arch: "x64" }],
      ["arm64", { runner: "macos-15", node_arch: "arm64", python_arch: "arm64" }]
    ]
  ],
  [
    ".github/workflows/linux.yml",
    "test-and-package",
    [
      ["x64", { runner: "ubuntu-24.04", node_arch: "x64", python_arch: "x64" }],
      ["arm64", { runner: "ubuntu-24.04-arm", node_arch: "arm64", python_arch: "arm64" }]
    ]
  ],
  [
    ".github/workflows/release.yml",
    "linux",
    [
      ["x64", { runner: "ubuntu-24.04", node_arch: "x64", python_arch: "x64" }],
      ["arm64", { runner: "ubuntu-24.04-arm", node_arch: "arm64", python_arch: "arm64" }]
    ]
  ]
]) {
  for (const [architecture, fields] of entries) {
    requireArchitecture(path, jobName, architecture, fields);
  }
}

const desktopPythonContracts = [
  {
    path: ".github/workflows/windows.yml",
    jobName: "test-and-package",
    pythonVersion: "3.11.9",
    lockedSteps: [
      "Build self-contained brain worker",
      "Package unsigned signed-ready Windows installer and archive"
    ]
  },
  {
    path: ".github/workflows/macos.yml",
    jobName: "test-and-package",
    pythonVersion: "3.11.9",
    lockedSteps: ["Package native macOS application"]
  },
  {
    path: ".github/workflows/linux.yml",
    jobName: "test-and-package",
    pythonVersion: "3.11.16",
    lockedSteps: ["Package native Linux application"]
  },
  {
    path: ".github/workflows/release.yml",
    jobName: "windows",
    pythonVersion: "3.11.9",
    lockedSteps: ["Package Windows"]
  },
  {
    path: ".github/workflows/release.yml",
    jobName: "macos",
    pythonVersion: "3.11.9",
    lockedSteps: ["Package, sign, and notarize macOS when credentials exist"]
  },
  {
    path: ".github/workflows/release.yml",
    jobName: "linux",
    pythonVersion: "3.11.16",
    lockedSteps: ["Package Linux"]
  }
];
for (const { path, jobName, pythonVersion, lockedSteps } of desktopPythonContracts) {
  const job = workflowJob(path, jobName);
  requireValue(
    job.includes(`python-version: "${pythonVersion}"`) &&
      (job.match(/python-version:/gu) ?? []).length === 1,
    `${path} ${jobName} must use exact CPython ${pythonVersion}.`
  );
  for (const dependencyPath of [
    "engine/requirements.txt",
    "engine/requirements-build.txt",
    "engine/locks/*.lock"
  ]) {
    requireValue(
      job.includes(dependencyPath),
      `${path} ${jobName} pip cache must include ${dependencyPath}.`
    );
  }
  requireValue(
    job.includes("cache-dependency-path: |") &&
      (job.match(/python scripts\/install-engine-lock\.py/gu) ?? []).length === 1,
    `${path} ${jobName} must install exactly one target-selected hash lock.`
  );
  requireValue(
    !job.includes("python -m pip install -r engine/requirements.txt") &&
      !job.includes("--index-url https://download.pytorch.org") &&
      !job.includes("matrix.torch_version") &&
      !job.includes("matrix.numpy_constraint"),
    `${path} ${jobName} must not retain the superseded ad-hoc Torch requirements install.`
  );
  for (const stepName of lockedSteps) {
    requireValue(
      workflowStep(path, jobName, stepName).includes(
        'OMNI_SKIP_BUILD_DEPENDENCY_INSTALL: "1"'
      ),
      `${path} ${jobName} ${stepName} must verify, not mutate, the installed lock.`
    );
  }
}
for (const [path, jobName] of [
  [".github/workflows/windows.yml", "test-and-package"],
  [".github/workflows/macos.yml", "test-and-package"],
  [".github/workflows/linux.yml", "test-and-package"],
  [".github/workflows/release.yml", "windows"],
  [".github/workflows/release.yml", "macos"],
  [".github/workflows/release.yml", "linux"]
]) {
  const job = workflowJob(path, jobName);
  requireValue(
    job.includes("python -m pip check") &&
      job.includes("torch.empty(1)"),
    `${path} ${jobName} must validate the installed native Torch runtime.`
  );
  requireValue(
    job.includes("python -m compileall -q engine") &&
      !job.includes("test:python:portable") &&
      !job.includes("test_distributed_torchrun") &&
      !job.includes("--desktop-e2e"),
    `${path} ${jobName} must use code-only Python checks, not train a brain in CI.`
  );
}

const stableRelease = workflows.get(".github/workflows/release.yml");
const continuousWindows = workflows.get(".github/workflows/windows.yml");
for (const fragment of [
  "OMNI_RELEASE_APPLE_API_KEY_P8",
  "omni-notary-api-key.p8",
  "base64.b64decode",
  'export APPLE_API_KEY="$OMNI_NOTARY_KEY_FILE"',
  "destination.chmod(0o600)",
  "unset OMNI_RELEASE_APPLE_API_KEY_P8"
]) {
  requireValue(
    stableRelease.includes(fragment),
    `Stable macOS release is missing private API-key materialization gate ${fragment}.`
  );
}
requireValue(
  !stableRelease.includes(
    'export APPLE_API_KEY="$OMNI_RELEASE_APPLE_API_KEY"'
  ),
  "Stable macOS release must not pass secret contents as the notarytool key path."
);
requireValue(
  !continuousWindows.includes("secrets.") &&
    continuousWindows.includes('OMNI_EXPECT_SIGNED: "0"'),
  "Continuous Windows CI must remain unsigned and must not receive release signing secrets."
);
requireValue(
  workflowJob(".github/workflows/release.yml", "windows").includes(
    "secrets.WINDOWS_CSC_LINK"
  ),
  "Only the verified stable-release Windows job may consume signing credentials."
);
for (const path of workflowPaths.filter((path) => path !== ".github/workflows/release.yml")) {
  requireValue(
    workflows.get(path).includes('branches: ["main"'),
    `${path} must validate pushes to main.`
  );
  requireValue(
    !workflows.get(path).includes("secrets."),
    `${path} is continuous CI and must not receive release credentials.`
  );
}
for (const fragment of [
  'tags: ["v*.*.*"]',
  "git fetch --no-tags origin +refs/heads/main:refs/remotes/origin/main",
  "commit: ${{ steps.release.outputs.commit }}",
  'VERIFIED_COMMIT="$(git rev-parse --verify "HEAD^{commit}")"',
  'git rev-parse --verify "$REQUESTED_TAG^{commit}"',
  'git merge-base --is-ancestor "$VERIFIED_COMMIT" origin/main',
  'test "$(git rev-parse --verify "$RELEASE_TAG^{commit}")" = "$RELEASE_COMMIT"'
]) {
  requireValue(
    stableRelease.includes(fragment),
    `Stable release is missing tag/main gate ${fragment}.`
  );
}
const commitBoundCheckouts = (
  stableRelease.match(/ref: \$\{\{ needs\.verify\.outputs\.commit \}\}/gu) ?? []
).length;
requireValue(
  commitBoundCheckouts === 6 &&
    !stableRelease.includes("ref: ${{ needs.verify.outputs.tag }}") &&
    stableRelease.includes("--source-commit") &&
    stableRelease.includes("OMNI_RELEASE_COMMIT: ${{ needs.verify.outputs.commit }}"),
  "Every release build, evidence record, and publish step must bind to the verified commit SHA."
);
for (const jobName of ["android", "ios"]) {
  const job = workflowJob(".github/workflows/release.yml", jobName);
  requireValue(
    job.includes("needs: verify") && job.includes("record-mobile-release.mjs"),
    `Stable release ${jobName} must consume the verified tag and produce hash-bound evidence.`
  );
}
requireValue(
  workflowJob(".github/workflows/release.yml", "android").includes(
    ":app:connectedDebugAndroidTest"
  ) &&
    workflowJob(".github/workflows/release.yml", "android").includes(
      ":app:assembleRelease"
    ),
  "Stable Android release must emulator-test the client and build a release APK."
);
requireValue(
  workflowJob(".github/workflows/release.yml", "ios").includes("npm run test:ios") &&
    workflowJob(".github/workflows/release.yml", "ios").includes(
      "npm run package:ios:unsigned"
    ),
  "Stable iOS release must simulator-test the client and build an unsigned signing-ready IPA."
);
requireValue(
  workflowJob(".github/workflows/release.yml", "publish").includes(
    "needs: [verify, windows, macos, linux, android, ios]"
  ),
  "Stable publishing must wait for every desktop and mobile release gate."
);

requireValue(
  stableRelease.includes("contents: write") && stableRelease.includes("verify-release-artifacts.mjs"),
  "Stable publishing must have a scoped write permission and use the artifact verifier."
);

process.stdout.write(
  `${JSON.stringify(
    {
      name: packageDocument.build.productName,
      version: packageDocument.version,
      tag: expectedTag,
      formats: {
        windows: packageDocument.build.win.target,
        macOS: packageDocument.build.mac.target,
        linux: packageDocument.build.linux.target,
        android: ["debug-signed.apk", "release-unsigned.apk"],
        iOS: ["unsigned.ipa"]
      },
      signedReady: {
        windows: true,
        macOS: true,
        macOSNotarization: true,
        androidRelease: true,
        iOS: true
      },
      pinnedActions: true,
      verified: true
    },
    null,
    2
  )}\n`
);
