import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { zipSync } from "fflate";
import { afterEach, describe, expect, it } from "vitest";

const repository = resolve(".");
const packageDocument = JSON.parse(
  readFileSync(join(repository, "package.json"), "utf8")
) as {
  version: string;
  build: { productName: string };
};
const temporaryDirectories: string[] = [];
const commitProbe = spawnSync(
  "git",
  ["rev-parse", "--verify", "HEAD^{commit}"],
  { cwd: repository, encoding: "utf8" }
);
if (commitProbe.status !== 0 || !/^[a-f0-9]{40}$/u.test(commitProbe.stdout.trim())) {
  throw new Error(commitProbe.stderr || "Could not resolve the release fixture commit.");
}
const sourceCommit = commitProbe.stdout.trim();

function publicName(name: string): string {
  return name.replace(/\s+/gu, ".");
}

function linuxArtifactArchitecture(
  architecture: "x64" | "arm64",
  extension: "AppImage" | "deb" | "tar.gz"
): string {
  if (architecture !== "x64") return architecture;
  if (extension === "AppImage") return "x86_64";
  if (extension === "deb") return "amd64";
  return architecture;
}

function hash(bytes: Buffer | string): string {
  return createHash("sha256").update(bytes).digest("hex");
}

const ffmpegPolicySha256 = hash(
  readFileSync(join(repository, "licenses/ffmpeg-runtime-policy.json"))
);
const legalFileCount = 3 + readdirSync(join(repository, "licenses")).length;

function packagedCompliance(
  kind: "desktop-artifact-compliance" | "mobile-artifact-compliance",
  platform?: "android" | "ios"
): Record<string, unknown> {
  return {
    schemaVersion: 1,
    kind,
    ...(platform ? { platform } : {}),
    legalFilesVerified: legalFileCount,
    ffmpegExecutableBundled: false,
    ffmpegPolicySha256
  };
}

function legalArchive(prefix: string): Record<string, Uint8Array> {
  const entries: Record<string, Uint8Array> = {};
  for (const name of ["LICENSE.md", "COMMERCIAL_LICENSE.md", "THIRD_PARTY_NOTICES.md"]) {
    entries[`${prefix}${name}`] = readFileSync(join(repository, name));
  }
  for (const name of readdirSync(join(repository, "licenses"))) {
    entries[`${prefix}licenses/${name}`] = readFileSync(
      join(repository, "licenses", name)
    );
  }
  return entries;
}

function workerSmoke(comprehensive = false): Record<string, unknown> {
  return {
    engineVersion: "1.1.0",
    protocolVersion: 1,
    persistedBrain: true,
    safeTensorCheckpoint: true,
    sqliteEventLog: true,
    comprehensive,
    ...(comprehensive
      ? {
          trainingLossDecreased: true,
          pdfIngested: true,
          chatParameterMutation: true,
          generatedModalities: ["image", "audio", "video"]
        }
      : {})
  };
}

function writeReleaseFixture(): string {
  const directory = mkdtempSync(join(tmpdir(), "omni-release-verifier-"));
  temporaryDirectories.push(directory);
  const product = packageDocument.build.productName;
  const version = packageDocument.version;
  const artifact = (name: string): { name: string; bytes: number; sha256: string } => {
    const bytes = Buffer.from(`fixture:${name}`);
    writeFileSync(join(directory, name), bytes);
    return { name, bytes: bytes.length, sha256: hash(bytes) };
  };

  for (const arch of ["x64", "arm64"] as const) {
    const windowsExe = artifact(`${product}-${version}-Windows-${arch}.exe`);
    const windowsZip = artifact(`${product}-${version}-Windows-${arch}.zip`);
    const macDmg = artifact(`${product}-${version}-macOS-${arch}.dmg`);
    const macZip = artifact(`${product}-${version}-macOS-${arch}.zip`);
    const linuxAppImage = artifact(
      `${product}-${version}-Linux-${linuxArtifactArchitecture(arch, "AppImage")}.AppImage`
    );
    const linuxDeb = artifact(
      `${product}-${version}-Linux-${linuxArtifactArchitecture(arch, "deb")}.deb`
    );
    const linuxTar = artifact(
      `${product}-${version}-Linux-${linuxArtifactArchitecture(arch, "tar.gz")}.tar.gz`
    );

    writeFileSync(
      join(directory, `windows-package-smoke-${arch}.json`),
      JSON.stringify({
        sourceCommit,
        architecture: arch,
        zip: {
          ...windowsZip,
          compliance: packagedCompliance("desktop-artifact-compliance"),
          packagedWorkerArchitecture: "x64",
          desktopArchitecture: arch,
          desktopSignature: { valid: false },
          rpcSmoke: workerSmoke()
        },
        nsis: {
          ...windowsExe,
          compliance: packagedCompliance("desktop-artifact-compliance"),
          packagedWorkerArchitecture: "x64",
          desktopArchitecture: arch,
          silentInstall: true,
          desktopEndToEnd: true,
          desktopRestart: true,
          accessibilityNavigation: true,
          modalityGeneration: true,
          installerSignature: { valid: false },
          desktopSignature: { valid: false },
          rpcSmoke: workerSmoke(true)
        },
        signing: {
          expectedSigned: false,
          fullySigned: false,
          state: "unsigned-signed-ready"
        }
      })
    );
    writeFileSync(
      join(directory, `mac-package-smoke-${arch}.json`),
      JSON.stringify({
        sourceCommit,
        architecture: arch,
        platform: "mac",
        artifacts: [macDmg, macZip],
        artifactCompliance: {
          [macDmg.name]: packagedCompliance("desktop-artifact-compliance"),
          [macZip.name]: packagedCompliance("desktop-artifact-compliance")
        },
        desktopEndToEnd: true,
        formatValidation: {
          dmg: { verified: true },
          zip: { extracted: true }
        },
        workerSmoke: workerSmoke(),
        signing: {
          expectedSigned: false,
          expectedNotarized: false,
          signatureValid: false,
          certificateSigned: false,
          notarized: false,
          state: "unsigned-signed-ready"
        }
      })
    );
    writeFileSync(
      join(directory, `linux-package-smoke-${arch}.json`),
      JSON.stringify({
        sourceCommit,
        architecture: arch,
        platform: "linux",
        artifacts: [linuxAppImage, linuxDeb, linuxTar],
        artifactCompliance: {
          [linuxAppImage.name]: packagedCompliance("desktop-artifact-compliance"),
          [linuxDeb.name]: packagedCompliance("desktop-artifact-compliance"),
          [linuxTar.name]: packagedCompliance("desktop-artifact-compliance")
        },
        desktopEndToEnd: true,
        formatValidation: {
          "tar.gz": { extracted: true },
          deb: {
            extracted: true,
            architecture: arch === "arm64" ? "arm64" : "amd64"
          },
          AppImage: { extracted: true }
        },
        workerSmoke: workerSmoke(),
        signing: {
          expectedSigned: false,
          expectedNotarized: false,
          state: "not-applicable"
        }
      })
    );
  }
  const androidDebug = artifact(
    `Omni-AGI-Companion-${version}-Android-debug-signed.apk`
  );
  const androidRelease = artifact(
    `Omni-AGI-Companion-${version}-Android-release-unsigned.apk`
  );
  writeFileSync(
    join(directory, "android-package-smoke.json"),
    JSON.stringify({
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
          ...androidDebug,
          compliance: packagedCompliance("mobile-artifact-compliance", "android"),
          variant: "debug",
          installable: true,
          zipAligned: true,
          signing: { state: "debug-signed", verified: true }
        },
        release: {
          ...androidRelease,
          compliance: packagedCompliance("mobile-artifact-compliance", "android"),
          variant: "release",
          installable: false,
          zipAligned: true,
          signing: { state: "unsigned-signed-ready", verified: true }
        }
      }
    })
  );
  const ios = artifact(`Omni-AGI-Companion-${version}-iOS-unsigned.ipa`);
  writeFileSync(
    join(directory, "ios-package-smoke.json"),
    JSON.stringify({
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
        ...ios,
        compliance: packagedCompliance("mobile-artifact-compliance", "ios"),
        variant: "release",
        installable: false,
        signing: { state: "unsigned-signed-ready", verified: true }
      }
    })
  );
  return directory;
}

function verify(directory: string): string {
  const result = spawnSync(
    process.execPath,
    [
      join(repository, "scripts/verify-release-artifacts.mjs"),
      "--directory",
      directory,
      "--source-commit",
      sourceCommit
    ],
    { cwd: repository, encoding: "utf8" }
  );
  if (result.status !== 0) {
    throw new Error(result.stderr || result.stdout || `Verifier exited ${String(result.status)}.`);
  }
  return result.stdout;
}

afterEach(() => {
  while (temporaryDirectories.length > 0) {
    rmSync(temporaryDirectories.pop() as string, { recursive: true, force: true });
  }
});

describe("stable release artifact gate", () => {
  it("records hash-bound mobile packages only for exact ZIP signatures", () => {
    const directory = mkdtempSync(join(tmpdir(), "omni-mobile-recorder-"));
    temporaryDirectories.push(directory);
    const version = packageDocument.version;
    const debugName = `Omni-AGI-Companion-${version}-Android-debug-signed.apk`;
    const releaseName = `Omni-AGI-Companion-${version}-Android-release-unsigned.apk`;
    const debugPath = join(directory, debugName);
    const releasePath = join(directory, releaseName);
    const outputPath = join(directory, "android-package-smoke.json");
    writeFileSync(
      debugPath,
      Buffer.from(zipSync(legalArchive("assets/legal/")))
    );
    writeFileSync(
      releasePath,
      Buffer.from(zipSync(legalArchive("assets/legal/")))
    );

    const recorded = spawnSync(
      process.execPath,
      [
        join(repository, "scripts/record-mobile-release.mjs"),
        "--platform",
        "android",
        "--output",
        outputPath,
        "--source-commit",
        sourceCommit,
        "--debug-artifact",
        debugPath,
        "--release-artifact",
        releasePath,
        "--emulator-tested",
        "1",
        "--same-brain-chat-tested",
        "1",
        "--attachment-stream-tested",
        "1",
        "--debug-signature-verified",
        "1",
        "--release-unsigned-verified",
        "1",
        "--zipalign-verified",
        "1"
      ],
      { cwd: repository, encoding: "utf8" }
    );
    expect(recorded.status, recorded.stderr).toBe(0);
    const evidence = JSON.parse(readFileSync(outputPath, "utf8")) as {
      sourceCommit: string;
      artifacts: {
        debug: {
          sha256: string;
          bytes: number;
          compliance: { legalFilesVerified: number; ffmpegPolicySha256: string };
        };
        release: {
          sha256: string;
          bytes: number;
          compliance: { legalFilesVerified: number; ffmpegPolicySha256: string };
        };
      };
    };
    expect(evidence.sourceCommit).toBe(sourceCommit);
    expect(evidence.artifacts.debug).toMatchObject({
      bytes: readFileSync(debugPath).byteLength,
      sha256: hash(readFileSync(debugPath))
    });
    expect(evidence.artifacts.release).toMatchObject({
      bytes: readFileSync(releasePath).byteLength,
      sha256: hash(readFileSync(releasePath))
    });
    expect(evidence.artifacts.debug.compliance).toMatchObject({
      legalFilesVerified: legalFileCount,
      ffmpegPolicySha256
    });
    expect(evidence.artifacts.release.compliance).toMatchObject({
      legalFilesVerified: legalFileCount,
      ffmpegPolicySha256
    });

    const mismatchedCommit = spawnSync(
      process.execPath,
      [
        join(repository, "scripts/record-mobile-release.mjs"),
        "--platform",
        "android",
        "--output",
        join(directory, "wrong-commit-smoke.json"),
        "--source-commit",
        "b".repeat(40),
        "--debug-artifact",
        debugPath,
        "--release-artifact",
        releasePath,
        "--emulator-tested",
        "1",
        "--same-brain-chat-tested",
        "1",
        "--attachment-stream-tested",
        "1",
        "--debug-signature-verified",
        "1",
        "--release-unsigned-verified",
        "1",
        "--zipalign-verified",
        "1"
      ],
      { cwd: repository, encoding: "utf8" }
    );
    expect(mismatchedCommit.status).not.toBe(0);
    expect(mismatchedCommit.stderr).toContain(
      "does not match verified commit"
    );

    const malformedName = `Omni-AGI-Companion-${version}-iOS-unsigned.ipa`;
    const malformedPath = join(directory, malformedName);
    writeFileSync(malformedPath, Buffer.from([0x50, 0x4b, 0x03, 0x58, 1, 2, 3, 4]));
    const rejected = spawnSync(
      process.execPath,
      [
        join(repository, "scripts/record-mobile-release.mjs"),
        "--platform",
        "ios",
        "--output",
        join(directory, "ios-package-smoke.json"),
        "--source-commit",
        sourceCommit,
        "--artifact",
        malformedPath,
        "--emulator-tested",
        "1",
        "--same-brain-chat-tested",
        "1",
        "--attachment-stream-tested",
        "1",
        "--unsigned-verified",
        "1"
      ],
      { cwd: repository, encoding: "utf8" }
    );
    expect(rejected.status).not.toBe(0);
    expect(rejected.stderr).toContain("is not a ZIP-based APK/IPA container");
  });

  it("accepts one complete hash-bound cross-platform artifact set", () => {
    const directory = writeReleaseFixture();
    expect(verify(directory)).toContain("Verified 25 release files");
    const manifest = JSON.parse(
      readFileSync(join(directory, "RELEASE-MANIFEST.json"), "utf8")
    ) as {
      sourceCommit: string;
      ffmpegPolicySha256: string;
      packagedCompliance: {
        artifactReports: number;
        minimumLegalFilesVerified: number;
        ffmpegExecutableBundled: boolean;
        ffmpegPolicySha256: string;
      };
      artifactCount: number;
      artifacts: Array<{ name: string }>;
      platformStatus: Record<string, unknown>;
    };
    expect(manifest.sourceCommit).toBe(sourceCommit);
    expect(manifest.ffmpegPolicySha256).toBe(ffmpegPolicySha256);
    expect(manifest.packagedCompliance).toEqual({
      schemaVersion: 1,
      artifactReports: 17,
      minimumLegalFilesVerified: legalFileCount,
      ffmpegExecutableBundled: false,
      ffmpegPolicySha256
    });
    expect(manifest.artifactCount).toBe(25);
    expect(manifest.platformStatus).toHaveProperty("windows.arm64.workerArchitecture", "x64");
    expect(manifest.platformStatus).toHaveProperty(
      "android.releaseSigning",
      "unsigned-signed-ready"
    );
    expect(manifest.platformStatus).toHaveProperty("iOS.signing", "unsigned-signed-ready");
    expect(manifest.artifacts.map((artifact) => artifact.name)).not.toEqual(
      expect.arrayContaining([expect.stringContaining(" ")])
    );
    const checksums = readFileSync(join(directory, "SHA256SUMS.txt"), "utf8")
      .trim()
      .split("\n");
    expect(checksums).toHaveLength(25);
    const checksumNames = checksums.map((line) => line.replace(/^[a-f0-9]{64}  /u, ""));
    expect(checksumNames).toEqual(manifest.artifacts.map((artifact) => artifact.name));
    expect(checksumNames.every((name) => !name.includes(" "))).toBe(true);
    for (const [index, name] of checksumNames.entries()) {
      expect(hash(readFileSync(join(directory, name)))).toBe(checksums[index]?.slice(0, 64));
    }
    expect(verify(directory)).toContain("Verified 25 release files");
    expect(
      existsSync(
        join(
          directory,
          publicName(
            `${packageDocument.build.productName}-${packageDocument.version}-Windows-x64.exe`
          )
        )
      )
    ).toBe(true);
    expect(
      existsSync(
        join(
          directory,
          `${packageDocument.build.productName}-${packageDocument.version}-Windows-x64.exe`
        )
      )
    ).toBe(false);
  });

  it("rejects a packaged/public filename collision before writing metadata", () => {
    const directory = writeReleaseFixture();
    const packaged = `${packageDocument.build.productName}-${packageDocument.version}-Windows-x64.exe`;
    copyFileSync(join(directory, packaged), join(directory, publicName(packaged)));
    expect(() => verify(directory)).toThrow(/both packaged and public names/);
  });

  it("rejects an extra file that the publisher would otherwise upload", () => {
    const directory = writeReleaseFixture();
    writeFileSync(join(directory, "unverified.bin"), "not reviewed");
    expect(() => verify(directory)).toThrow(/unverified files: unverified\.bin/);
  });

  it("rejects a package changed after its smoke record was produced", () => {
    const directory = writeReleaseFixture();
    const target = join(
      directory,
      `${packageDocument.build.productName}-${packageDocument.version}-Linux-arm64.AppImage`
    );
    writeFileSync(target, "tampered");
    expect(() => verify(directory)).toThrow(/hash does not match/);
  });

  it("rejects mobile artifacts without verified emulator and signing evidence", () => {
    const directory = writeReleaseFixture();
    const evidencePath = join(directory, "android-package-smoke.json");
    const evidence = JSON.parse(readFileSync(evidencePath, "utf8")) as {
      sameBrainGateway: { emulatorTested: boolean };
      artifacts: { release: { signing: { verified: boolean } } };
    };
    evidence.sameBrainGateway.emulatorTested = false;
    evidence.artifacts.release.signing.verified = false;
    writeFileSync(evidencePath, JSON.stringify(evidence));
    expect(() => verify(directory)).toThrow(/same-brain emulator/);
  });

  it("rejects incomplete legal evidence or a bundled FFmpeg executable", () => {
    const incompleteDirectory = writeReleaseFixture();
    const incompletePath = join(
      incompleteDirectory,
      "windows-package-smoke-x64.json"
    );
    const incomplete = JSON.parse(readFileSync(incompletePath, "utf8")) as {
      zip: { compliance: { legalFilesVerified: number } };
    };
    incomplete.zip.compliance.legalFilesVerified = 9;
    writeFileSync(incompletePath, JSON.stringify(incomplete));
    expect(() => verify(incompleteDirectory)).toThrow(
      /lacks valid packaged legal and FFmpeg compliance evidence/
    );

    const bundledDirectory = writeReleaseFixture();
    const bundledPath = join(bundledDirectory, "ios-package-smoke.json");
    const bundled = JSON.parse(readFileSync(bundledPath, "utf8")) as {
      artifact: { compliance: { ffmpegExecutableBundled: boolean } };
    };
    bundled.artifact.compliance.ffmpegExecutableBundled = true;
    writeFileSync(bundledPath, JSON.stringify(bundled));
    expect(() => verify(bundledDirectory)).toThrow(
      /lacks valid packaged legal and FFmpeg compliance evidence/
    );
  });

  it("rejects artifacts verified against different FFmpeg policies", () => {
    const directory = writeReleaseFixture();
    const evidencePath = join(directory, "linux-package-smoke-arm64.json");
    const evidence = JSON.parse(readFileSync(evidencePath, "utf8")) as {
      artifactCompliance: Record<string, { ffmpegPolicySha256: string }>;
    };
    const first = Object.values(evidence.artifactCompliance)[0];
    if (!first) throw new Error("Missing fixture compliance record.");
    first.ffmpegPolicySha256 = "b".repeat(64);
    writeFileSync(evidencePath, JSON.stringify(evidence));
    expect(() => verify(directory)).toThrow(/one identical nonempty FFmpeg policy hash/);
  });

  it("rejects a smoke record produced from any commit except the verified release commit", () => {
    const directory = writeReleaseFixture();
    const evidencePath = join(directory, "mac-package-smoke-arm64.json");
    const evidence = JSON.parse(readFileSync(evidencePath, "utf8")) as {
      sourceCommit: string;
    };
    evidence.sourceCommit = "b".repeat(40);
    writeFileSync(evidencePath, JSON.stringify(evidence));
    expect(() => verify(directory)).toThrow(/not bound to verified source commit/);
  });

  it("binds every stable release checkout and evidence command to the verified commit", () => {
    const workflow = readFileSync(
      join(repository, ".github/workflows/release.yml"),
      "utf8"
    );
    expect(workflow).toContain("commit: ${{ steps.release.outputs.commit }}");
    expect(workflow).toContain('echo "commit=$VERIFIED_COMMIT" >> "$GITHUB_OUTPUT"');
    expect(workflow.match(/ref: \$\{\{ needs\.verify\.outputs\.commit \}\}/gu)).toHaveLength(6);
    expect(workflow).not.toContain("ref: ${{ needs.verify.outputs.tag }}");
    expect(workflow.match(/(?:SourceCommit|source-commit)/gu)?.length ?? 0).toBeGreaterThanOrEqual(5);
    expect(workflow).toContain("OMNI_RELEASE_COMMIT: ${{ needs.verify.outputs.commit }}");
    expect(workflow).toContain("RELEASE_COMMIT: ${{ needs.verify.outputs.commit }}");
    expect(workflow).toContain(
      'test "$(git rev-parse --verify "$RELEASE_TAG^{commit}")" = "$RELEASE_COMMIT"'
    );
  });
});
