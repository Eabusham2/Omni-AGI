import { createHash } from "node:crypto";
import { constants, createReadStream } from "node:fs";
import { lstat, open, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
function verifyCompiledPortableProtocol(path) {
  // Inspect actual packaged bytes, never execute their JavaScript. This is a
  // stale-build guard, not behavioral or signed attestation.
  const { extractFile } = require("@electron/asar");
  const packaged = JSON.parse(extractFile(path, "package.json", false).toString("utf8"));
  if (packaged.main !== "out/main/index.js") throw new Error("Portable runtime has an unexpected main entry.");
  const main = extractFile(path, packaged.main, false).toString("utf8");
  for (const fragment of ["omni-portable-launcher", "omni-portable-workspace", "portable-destination.json", "portable-adoption"]) {
    if (!main.includes(fragment)) throw new Error("The packaged desktop main is stale and lacks the portable workspace protocol. Rebuild before packaging.");
  }
  return packaged.version;
}

async function peArchitecture(path) {
  const info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Portable runtime PE must be a regular file.");
  const handle = await open(path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0));
  try {
    const dos = Buffer.alloc(64); if ((await handle.read(dos, 0, 64, 0)).bytesRead !== 64 || dos.readUInt16LE() !== 0x5a4d) throw new Error("Invalid portable runtime DOS header.");
    const offset = dos.readUInt32LE(0x3c); if (offset > info.size - 6) throw new Error("Invalid portable runtime PE offset.");
    const header = Buffer.alloc(6); if ((await handle.read(header, 0, 6, offset)).bytesRead !== 6 || header.readUInt32LE() !== 0x4550) throw new Error("Invalid portable runtime PE signature.");
    const value = header.readUInt16LE(4); if (value === 0x8664) return "x64"; if (value === 0xaa64) return "arm64"; throw new Error("Unsupported portable Windows architecture.");
  } finally { await handle.close(); }
}
export async function writeWindowsPortableRuntimeReceipt({ directory, architecture, version, sourceCommit = null }) {
  if (!["x64", "arm64"].includes(architecture) || !/^\d+\.\d+\.\d+$/.test(version) || sourceCommit !== null && !/^[a-f0-9]{40}$/.test(sourceCommit)) throw new Error("Invalid portable runtime build identity.");
  const desktop = await peArchitecture(join(directory, "Omni AGI Studio.exe"));
  if (desktop !== architecture) throw new Error("Portable runtime desktop architecture differs from its package.");
  const workerArchitecture = await peArchitecture(join(directory, "resources", "engine-runtime", "omni-engine.exe"));
  const path = join(directory, "resources", "app.asar"), info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink()) throw new Error("Portable runtime requires a regular packaged app.asar.");
  if (verifyCompiledPortableProtocol(path) !== version) throw new Error("Packaged portable main version differs from the package build.");
  const digest = createHash("sha256"); for await (const chunk of createReadStream(path)) digest.update(chunk);
  const receipt = { format: "omni-portable-runtime", version: 1, portableProtocol: 1, appVersion: version, sourceCommit,
    platform: "win32", architecture, workerArchitecture, executable: "Omni AGI Studio.exe",
    appAsar: { path: "resources/app.asar", bytes: info.size, sha256: digest.digest("hex") } };
  await writeFile(join(directory, "resources", "portable-runtime.json"), `${JSON.stringify(receipt)}\n`, { flag: "wx", mode: 0o600 });
  return receipt;
}
export async function verifyWindowsPortableRuntimeReceipt({ directory, version, sourceCommit }) {
  const path = join(directory, "resources", "portable-runtime.json"), info = await lstat(path);
  if (!info.isFile() || info.isSymbolicLink() || info.size > 4096) throw new Error("Portable runtime receipt must be a small regular file.");
  const receipt = JSON.parse(await readFile(path, "utf8"));
  if (receipt.format !== "omni-portable-runtime" || receipt.version !== 1 || receipt.portableProtocol !== 1 || receipt.platform !== "win32" || receipt.executable !== "Omni AGI Studio.exe" || !["x64", "arm64"].includes(receipt.architecture) || !["x64", "arm64"].includes(receipt.workerArchitecture) || version !== undefined && receipt.appVersion !== version || receipt.appAsar?.path !== "resources/app.asar") throw new Error("Unsupported portable runtime receipt.");
  if (receipt.sourceCommit !== null && (typeof receipt.sourceCommit !== "string" || !/^[a-f0-9]{40}$/.test(receipt.sourceCommit)) || sourceCommit !== undefined && (!/^[a-f0-9]{40}$/.test(sourceCommit) || receipt.sourceCommit !== sourceCommit)) throw new Error("Portable runtime source commit differs from the verified package commit.");
  if (await peArchitecture(join(directory, receipt.executable)) !== receipt.architecture || await peArchitecture(join(directory, "resources", "engine-runtime", "omni-engine.exe")) !== receipt.workerArchitecture) throw new Error("Portable runtime PE architecture differs from its receipt.");
  const asar = join(directory, "resources", "app.asar"), asarInfo = await lstat(asar), digest = createHash("sha256");
  if (!asarInfo.isFile() || asarInfo.isSymbolicLink() || asarInfo.size !== receipt.appAsar.bytes) throw new Error("Portable runtime ASAR size differs from its receipt.");
  for await (const chunk of createReadStream(asar)) digest.update(chunk);
  if (digest.digest("hex") !== receipt.appAsar.sha256 || verifyCompiledPortableProtocol(asar) !== receipt.appVersion) throw new Error("Portable runtime ASAR differs from its sealed receipt.");
  return { portableProtocol: 1, appVersion: receipt.appVersion, sourceCommit: receipt.sourceCommit, desktopArchitecture: receipt.architecture, workerArchitecture: receipt.workerArchitecture,
    runtimeReceiptVerified: true, selectedInstanceTested: false, destinationReservationTested: false, neuralAcceptanceTested: false };
}
