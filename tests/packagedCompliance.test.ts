import { spawnSync } from "node:child_process";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { zipSync } from "fflate";
import { afterEach, describe, expect, it } from "vitest";

const repository = resolve(process.cwd());
const verifier = join(repository, "scripts", "verify-packaged-compliance.mjs");
const temporaryRoots: string[] = [];
const legalFileCount = 3 + (await readdir(join(repository, "licenses"))).length;

async function legalEntries(prefix: string): Promise<Record<string, Uint8Array>> {
  const entries: Record<string, Uint8Array> = {};
  for (const name of ["LICENSE.md", "COMMERCIAL_LICENSE.md", "THIRD_PARTY_NOTICES.md"]) {
    entries[`${prefix}${name}`] = await readFile(join(repository, name));
  }
  for (const name of await readdir(join(repository, "licenses"))) {
    entries[`${prefix}licenses/${name}`] = await readFile(
      join(repository, "licenses", name),
    );
  }
  return entries;
}

function runVerifier(args: string[]) {
  return spawnSync(process.execPath, [verifier, "--repo-root", repository, ...args], {
    cwd: repository,
    encoding: "utf8",
  });
}

describe("packaged license and FFmpeg compliance", () => {
  afterEach(async () => {
    await Promise.all(
      temporaryRoots.splice(0).map((path) => rm(path, { recursive: true, force: true })),
    );
  });

  it("accepts canonical notices in Android and iOS application containers", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mobile-compliance-"));
    temporaryRoots.push(root);
    const android = join(root, "fixture.apk");
    const ios = join(root, "fixture.ipa");
    await writeFile(android, Buffer.from(zipSync(await legalEntries("assets/legal/"))));
    await writeFile(
      ios,
      Buffer.from(zipSync(await legalEntries("Payload/Omni AGI Companion.app/"))),
    );

    const androidResult = runVerifier([
      "--platform",
      "android",
      "--artifact",
      android,
    ]);
    const iosResult = runVerifier(["--platform", "ios", "--artifact", ios]);

    expect(androidResult.status, androidResult.stderr).toBe(0);
    expect(iosResult.status, iosResult.stderr).toBe(0);
    expect(JSON.parse(androidResult.stdout)).toMatchObject({
      platform: "android",
      legalFilesVerified: legalFileCount,
      ffmpegExecutableBundled: false,
    });
    expect(JSON.parse(iosResult.stdout)).toMatchObject({
      platform: "ios",
      legalFilesVerified: legalFileCount,
      ffmpegExecutableBundled: false,
    });
  });

  it("rejects a stale Required Notice and any bundled FFmpeg executable", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-mobile-compliance-reject-"));
    temporaryRoots.push(root);
    const missingNotice = await legalEntries("assets/legal/");
    missingNotice["assets/legal/licenses/REQUIRED_NOTICE.txt"] = Buffer.from(
      "Required Notice: stale\n",
    );
    const missingNoticePath = join(root, "missing-notice.apk");
    await writeFile(missingNoticePath, Buffer.from(zipSync(missingNotice)));
    const noticeResult = runVerifier([
      "--platform",
      "android",
      "--artifact",
      missingNoticePath,
    ]);
    expect(noticeResult.status).toBe(1);
    expect(noticeResult.stderr).toMatch(/stale or modified.*REQUIRED_NOTICE/i);

    const bundled = await legalEntries("assets/legal/");
    bundled["assets/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.1"] =
      Buffer.from("ELF fixture");
    const bundledPath = join(root, "bundled-ffmpeg.apk");
    await writeFile(bundledPath, Buffer.from(zipSync(bundled)));
    const bundledResult = runVerifier([
      "--platform",
      "android",
      "--artifact",
      bundledPath,
    ]);
    expect(bundledResult.status).toBe(1);
    expect(bundledResult.stderr).toMatch(/prohibited FFmpeg executable/i);
  });

  it("fails a packaged worker when an imageio wheel binary survives", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-engine-compliance-"));
    temporaryRoots.push(root);
    const binaries = join(root, "_internal", "imageio_ffmpeg", "binaries");
    await mkdir(binaries, { recursive: true });
    await writeFile(join(binaries, "__init__.py"), "");
    const clean = runVerifier(["--engine-dir", root]);
    expect(clean.status, clean.stderr).toBe(0);

    await writeFile(join(binaries, "ffmpeg-win-x86_64-v7.1.exe"), "MZ fixture");
    const rejected = runVerifier(["--engine-dir", root]);
    expect(rejected.status).toBe(1);
    expect(rejected.stderr).toMatch(/prohibited FFmpeg executable/i);
  });

  it("verifies canonical notices and the no-binary policy in extracted desktop apps", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-desktop-compliance-"));
    temporaryRoots.push(root);
    const resources = join(root, "Omni AGI Studio.app", "Contents", "Resources", "licenses");
    await mkdir(resources, { recursive: true });
    const sourceEntries = await legalEntries("");
    for (const [sourcePath, contents] of Object.entries(sourceEntries)) {
      const name = sourcePath.startsWith("licenses/")
        ? sourcePath.slice("licenses/".length)
        : sourcePath;
      await writeFile(join(resources, name), contents);
    }
    const accepted = runVerifier(["--desktop-dir", root]);
    expect(accepted.status, accepted.stderr).toBe(0);
    expect(JSON.parse(accepted.stdout)).toMatchObject({
      kind: "desktop-artifact-compliance",
      legalFilesVerified: legalFileCount,
      ffmpegExecutableBundled: false,
    });

    const wheel = join(
      root,
      "Omni AGI Studio.app",
      "Contents",
      "Resources",
      "engine-runtime",
      "imageio_ffmpeg",
      "binaries",
    );
    await mkdir(wheel, { recursive: true });
    await writeFile(join(wheel, "ffmpeg-macos-aarch64-v7.1"), "Mach-O fixture");
    const rejected = runVerifier(["--desktop-dir", root]);
    expect(rejected.status).toBe(1);
    expect(rejected.stderr).toMatch(/prohibited FFmpeg executable/i);
  });

  it("wires canonical assets and post-package verification into both mobile builds", async () => {
    const [android, iosProject, iosPackage] = await Promise.all([
      readFile(join(repository, "mobile/android/app/build.gradle.kts"), "utf8"),
      readFile(
        join(repository, "mobile/ios/OmniCompanion.xcodeproj/project.pbxproj"),
        "utf8",
      ),
      readFile(join(repository, "scripts/package-ios.sh"), "utf8"),
    ]);
    expect(android).toContain("prepareOmniComplianceAssets");
    expect(android).toContain('into("licenses")');
    expect(android).not.toContain('"META-INF/LICENSE*"');
    expect(android).toContain("verify-packaged-compliance.mjs");
    expect(android).toContain("finalizedBy(verifyDebugOmniCompliance)");
    expect(android).toContain("finalizedBy(verifyReleaseOmniCompliance)");
    expect(iosProject).toContain("LICENSE.md in Resources");
    expect(iosProject).toContain("THIRD_PARTY_NOTICES.md in Resources");
    expect(iosProject).toContain("licenses in Resources");
    expect(iosPackage).toContain("verify-packaged-compliance.mjs");
  });

  it("strips only wheel FFmpeg payloads and packages canonical desktop notices", async () => {
    const [posix, windows, packageDocument] = await Promise.all([
      readFile(join(repository, "scripts/build-engine-posix.sh"), "utf8"),
      readFile(join(repository, "scripts/build-engine.ps1"), "utf8"),
      readFile(join(repository, "package.json"), "utf8"),
    ]);
    for (const source of [posix, windows]) {
      expect(source).toContain("imageio_ffmpeg");
      expect(source).toContain("verify-packaged-compliance.mjs");
      expect(source).toContain("--engine-dir");
    }
    expect(posix).toContain("/imageio_ffmpeg/binaries/ffmpeg*");
    expect(windows).toMatch(/imageio_ffmpeg.*binaries/);
    const packaged = JSON.parse(packageDocument) as {
      build: { extraResources: Array<{ from: string; to: string }> };
    };
    expect(packaged.build.extraResources).toEqual(
      expect.arrayContaining([
        { from: "LICENSE.md", to: "licenses/LICENSE.md" },
        { from: "COMMERCIAL_LICENSE.md", to: "licenses/COMMERCIAL_LICENSE.md" },
        {
          from: "licenses/REQUIRED_NOTICE.txt",
          to: "licenses/REQUIRED_NOTICE.txt",
        },
        {
          from: "licenses/ffmpeg-runtime-policy.json",
          to: "licenses/ffmpeg-runtime-policy.json",
        },
      ]),
    );
  });
});
