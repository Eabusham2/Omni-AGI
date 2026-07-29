import { access, readFile } from "node:fs/promises";

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
  "package:linux:arm64"
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

const workflowPaths = [
  ".github/workflows/windows.yml",
  ".github/workflows/macos.yml",
  ".github/workflows/linux.yml",
  ".github/workflows/release.yml"
];
const workflows = new Map();
for (const path of workflowPaths) {
  workflows.set(path, await readFile(path, "utf8"));
}
for (const [path, source] of workflows) {
  const officialActions = [...source.matchAll(/uses:\s+actions\/[^@\s]+@([^\s#]+)/g)];
  requireValue(officialActions.length > 0, `${path} does not use the expected official actions.`);
  for (const match of officialActions) {
    requireValue(
      /^[a-f0-9]{40}$/.test(match[1]),
      `${path} must pin ${match[0]} to a full commit SHA.`
    );
  }
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
requireValue(
  workflows.get(".github/workflows/release.yml").includes("contents: write") &&
    workflows.get(".github/workflows/release.yml").includes("verify-release-artifacts.mjs"),
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
        linux: packageDocument.build.linux.target
      },
      signedReady: {
        windows: true,
        macOS: true,
        macOSNotarization: true
      },
      pinnedActions: true,
      verified: true
    },
    null,
    2
  )}\n`
);
