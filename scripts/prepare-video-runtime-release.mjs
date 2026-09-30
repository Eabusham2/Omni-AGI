// SPDX-License-Identifier: MIT
// Mechanical staging from reviewed CI artifacts; does not execute a codec.
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { copyFile, lstat, mkdir, readFile, writeFile } from "node:fs/promises";
import { join, basename } from "node:path";
import { execFileSync } from "node:child_process";

const [root, tag, output] = process.argv.slice(2);
if (!root || !output || !/^omni-video-[0-9.]+-r[0-9]+$/.test(tag ?? "")) throw Error("Explicit review root, versioned codec tag and output required");
const targets = ["win32-x64", "win32-arm64", "darwin-x64", "darwin-arm64", "linux-x64", "linux-arm64"];
const version = "9.0.2", x264 = "0480cb05fa188d37ae87e8f4fd8f1aea3711f7ee";
const reviewedAt = new Date().toISOString();
const releasePrefix = `https://github.com/Eabusham2/Omni-AGI/releases/download/${tag}/`;
await mkdir(output, { recursive: true });
async function hashFile(path) {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw Error(`Unsafe review payload: ${path}`);
  const digest = createHash("sha256");
  for await (const part of createReadStream(path, { highWaterMark: 64 * 1024 })) digest.update(part);
  return { sizeBytes: info.size, sha256: digest.digest("hex") };
}
const staged = new Map(), artifacts = [];
for (const target of targets) {
  const directory = join(root, target);
  const receipt = JSON.parse(await readFile(join(directory, "build-receipt.json"), "utf8"));
  if (receipt.format !== "omni-codec-build" || receipt.formatVersion !== 1 || receipt.target !== target ||
      receipt.version !== version || receipt.x264Commit !== x264 || !/^[a-f0-9]{40}$/.test(receipt.commit) ||
      receipt.sourceSignatureFingerprint !== "FCF986EA15E6E293A5644F10B4322F04D67658D8" ||
      receipt.codecSmoke?.h264VideoDecoded !== true || receipt.codecSmoke?.aacAudioDecoded !== true ||
      !receipt.configuration.includes("--disable-autodetect") || !receipt.configuration.includes("--enable-libx264") ||
      !receipt.configuration.includes("--enable-gpl") || /--enable-(?:nonfree|version3|lib(?!x264\b))/.test(receipt.configuration)) throw Error(`Unreviewed codec recipe: ${target}`);
  if (!Array.isArray(receipt.files)) throw Error("Missing exact CI payload inventory");
  const checked = new Map();
  for (const file of receipt.files) {
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]+$/.test(file.fileName) || !/^[a-f0-9]{64}$/.test(file.sha256)) throw Error("Unsafe payload inventory");
    const measured = await hashFile(join(directory, file.fileName));
    if (measured.sizeBytes !== file.sizeBytes || measured.sha256 !== file.sha256) throw Error(`CI payload mismatch: ${file.fileName}`);
    checked.set(file.fileName, measured);
  }
  const dependencies = await readFile(join(directory, "binary-dependencies.txt"), "utf8");
  if (target.startsWith("linux-")) {
    for (const line of dependencies.split(/\r?\n/).filter(line => line.trim())) {
      if (!/linux-vdso|lib[cm]\.so|ld-linux/.test(line) || /not found/.test(line)) throw Error(`Non-system/missing Linux dependency: ${line}`);
    }
  } else if (target.startsWith("darwin-")) {
    for (const line of dependencies.split(/\r?\n/).slice(1).filter(line => line.trim())) {
      if (!/^\s+(?:\/usr\/lib\/|\/System\/Library\/Frameworks\/)/.test(line)) throw Error(`Non-system macOS dependency: ${line}`);
    }
  } else {
    for (const line of dependencies.split(/\r?\n/).filter(line => line.trim())) {
      const match = /DLL Name:\s+([^\s]+)/i.exec(line);
      if (!match || !/^(?:api-ms-win-.*|kernel32|bcrypt|shell32|advapi32|user32|ws2_32|ole32|secur32|ntdll|msvcrt|ucrtbase)\.dll$/i.test(match[1])) throw Error(`Non-system Windows dependency: ${line}`);
    }
  }
  const signature = await readFile(join(directory, "source-signature-status.txt"), "utf8");
  if (!signature.includes("VALIDSIG FCF986EA15E6E293A5644F10B4322F04D67658D8 ")) throw Error("Missing pinned official FFmpeg signature evidence");
  const sourceEntries = execFileSync("tar", ["-tf", join(directory, `ffmpeg-${version}-source.tar.xz`)], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  if (!["configure", "LICENSE.md", "COPYING.GPLv2", "libavcodec/libx264.c"].every(path => sourceEntries.split("\n").includes(`ffmpeg-${version}/${path}`))) throw Error("Incomplete FFmpeg corresponding sources");
  const linkedEntries = execFileSync("tar", ["-tf", join(directory, `x264-${x264}-source.tar.gz`)], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  if (!["COPYING", "configure", "encoder/api.c", "version.sh", ".git/HEAD"].every(path => linkedEntries.split("\n").includes(`x264/${path}`))) throw Error("Incomplete pinned x264 corresponding sources");
  async function pin(sourceName, localName, assetName = sourceName) {
    const measured = checked.get(sourceName);
    if (!measured) throw Error(`Required matching payload absent: ${sourceName}`);
    const prior = staged.get(assetName);
    if (prior && prior.sha256 !== measured.sha256) throw Error(`Conflicting release asset ${assetName}`);
    if (!prior) { await copyFile(join(directory, sourceName), join(output, assetName)); staged.set(assetName, measured); }
    return { fileName: localName, url: releasePrefix + assetName, ...measured };
  }
  const executable = `ffmpeg-${target}${target.startsWith("win32-") ? ".exe" : ""}`;
  const bytes = await readFile(join(directory, executable));
  if (target.startsWith("win32-")) {
    const offset = bytes.readUInt32LE(60);
    if (bytes.toString("ascii", 0, 2) !== "MZ" || bytes.toString("ascii", offset, offset + 2) !== "PE" ||
        bytes.readUInt16LE(offset + 4) !== (target.endsWith("arm64") ? 0xaa64 : 0x8664)) throw Error("Wrong native PE target");
  } else if (target.startsWith("linux-")) {
    if (bytes.toString("hex", 0, 6) !== "7f454c460201" || bytes.readUInt16LE(18) !== (target.endsWith("arm64") ? 183 : 62)) throw Error("Wrong native ELF target");
  } else if (bytes.readUInt32LE(0) !== 0xfeedfacf || bytes.readUInt32LE(4) !== (target.endsWith("arm64") ? 0x0100000c : 0x01000007)) throw Error("Wrong native Mach-O target");
  const artifact = {
    id: `ffmpeg-${version}-${target}-r1`, target, version, license: "GPL-2.0-or-later",
    binary: await pin(executable, target.startsWith("win32-") ? "ffmpeg.exe" : "ffmpeg"),
    correspondingSource: await pin(`ffmpeg-${version}-source.tar.xz`, "ffmpeg-source.tar.xz"),
    buildAndInstallMaterial: await pin(`build-material-${target}.tar.gz`, "build-material.tar.gz"),
    licenseNotices: await pin(`LICENSE-NOTICES-${target}.txt`, "LICENSE-NOTICES.txt"),
    provenance: { upstreamReleaseUrl: `https://ffmpeg.org/releases/ffmpeg-${version}.tar.xz`,
      buildRepositoryUrl: "https://github.com/Eabusham2/Omni-AGI", buildCommit: receipt.commit,
      configuration: receipt.configuration, linkage: "standalone-cli-static-except-system",
      reviewId: "source-distribution-2026-09-30", reviewedAt,
      completeCorrespondingSourceReviewed: true, allLinkedNonSystemLibrariesReviewed: true, distributionLicenseReviewed: true },
    linkedLibraries: [{ configureFlag: "--enable-libx264", name: "x264", license: "GPL-2.0-or-later",
      source: await pin(`x264-${x264}-source.tar.gz`, "x264-source.tar.gz", `x264-${target}-source.tar.gz`) }]
  };
  // Publish immutable build and dependency evidence too, outside installer inputs.
  await copyFile(join(directory, "build-receipt.json"), join(output, `build-receipt-${target}.json`));
  await copyFile(join(directory, "binary-dependencies.txt"), join(output, `binary-dependencies-${target}.txt`));
  for (const name of [`ffmpeg-${version}-source.tar.xz.asc`, "ffmpeg-signing-key.asc", "GPL-2.0.txt", "x264-COPYING.txt"]) await pin(name, name);
  artifacts.push(artifact);
}
const manifest = { schemaVersion: 1, component: "FFmpeg", artifacts, unavailableTargets: [] };
await writeFile(join(output, "ffmpeg-runtime-manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
const checksums = [];
for (const name of [...staged.keys(), "ffmpeg-runtime-manifest.json", ...targets.flatMap(target => [`build-receipt-${target}.json`, `binary-dependencies-${target}.txt`])].sort()) {
  const measured = await hashFile(join(output, name));
  checksums.push(`${measured.sha256}  ${basename(name)}`);
}
await writeFile(join(output, "SHA256SUMS.txt"), checksums.join("\n") + "\n");
console.log(JSON.stringify({ targets: artifacts.map(artifact => artifact.target), manifestContentSha256: createHash("sha256").update(JSON.stringify(manifest)).digest("hex"), output, files: checksums.length, executedCodec: false }));
