import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync
} from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

const repository = resolve(".");
const packageDocument = JSON.parse(
  readFileSync(join(repository, "package.json"), "utf8")
) as {
  version: string;
  build: { productName: string };
};
const temporaryDirectories: string[] = [];

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

function workerSmoke(comprehensive = false): Record<string, unknown> {
  return {
    engineVersion: "1.0.0",
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
        architecture: arch,
        zip: {
          ...windowsZip,
          packagedWorkerArchitecture: "x64",
          desktopArchitecture: arch,
          desktopSignature: { valid: false },
          rpcSmoke: workerSmoke()
        },
        nsis: {
          ...windowsExe,
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
        architecture: arch,
        platform: "mac",
        artifacts: [macDmg, macZip],
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
        architecture: arch,
        platform: "linux",
        artifacts: [linuxAppImage, linuxDeb, linuxTar],
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
  return directory;
}

function verify(directory: string): string {
  const result = spawnSync(
    process.execPath,
    [join(repository, "scripts/verify-release-artifacts.mjs"), "--directory", directory],
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
  it("accepts one complete hash-bound cross-platform artifact set", () => {
    const directory = writeReleaseFixture();
    expect(verify(directory)).toContain("Verified 20 release files");
    const manifest = JSON.parse(
      readFileSync(join(directory, "RELEASE-MANIFEST.json"), "utf8")
    ) as {
      artifactCount: number;
      artifacts: Array<{ name: string }>;
      platformStatus: Record<string, unknown>;
    };
    expect(manifest.artifactCount).toBe(20);
    expect(manifest.platformStatus).toHaveProperty("windows.arm64.workerArchitecture", "x64");
    expect(manifest.artifacts.map((artifact) => artifact.name)).not.toEqual(
      expect.arrayContaining([expect.stringContaining(" ")])
    );
    const checksums = readFileSync(join(directory, "SHA256SUMS.txt"), "utf8")
      .trim()
      .split("\n");
    expect(checksums).toHaveLength(20);
    const checksumNames = checksums.map((line) => line.replace(/^[a-f0-9]{64}  /u, ""));
    expect(checksumNames).toEqual(manifest.artifacts.map((artifact) => artifact.name));
    expect(checksumNames.every((name) => !name.includes(" "))).toBe(true);
    for (const [index, name] of checksumNames.entries()) {
      expect(hash(readFileSync(join(directory, name)))).toBe(checksums[index]?.slice(0, 64));
    }
    expect(verify(directory)).toContain("Verified 20 release files");
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
});
