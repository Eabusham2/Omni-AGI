#!/usr/bin/env node
/** Preserve the pinned builder; extend only its own ordinary detach attempts. */
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const CHECKSUMS = Object.freeze({
  "dmgbuild-bundle-arm64-75c8a6c.tar.gz": "793404d0c96687e27d5ee40a668d498c92e36a64d6c2906df511031adb33cbeb",
  "dmgbuild-bundle-x86_64-75c8a6c.tar.gz": "1664972f9cc2d6e8fce3b63e42cd30078aff602669c5856939c4519921200433"
});

export function pinnedDmgbuildToolset(architecture) {
  if (architecture !== "x64" && architecture !== "arm64") throw new Error("Unsupported native DMG builder architecture.");
  const arch = architecture === "arm64" ? "arm64" : "x86_64";
  return { releaseName: "dmg-builder@1.2.5", filenameWithExt: `dmgbuild-bundle-${arch}-75c8a6c.tar.gz`,
    checksums: { ...CHECKSUMS } };
}

export async function runDmgbuild(args, options = {}) {
  const { platform = process.platform, architecture = process.arch, execute = spawnSync,
    download = value => require("app-builder-lib/out/util/electronGet").downloadBuilderToolset(value) } = options;
  if (platform !== "darwin") throw new Error("Native DMG construction requires macOS.");
  // This is the exact invocation issued by the pinned electron-builder, not a
  // generic launcher or an image/device selected from ambient mount state.
  if (args.length !== 4 || args[0] !== "-s" || args.some(value => typeof value !== "string" || !value || value.includes("\0")) ||
      !isAbsolute(args[1]) || !isAbsolute(args[3]) || !args[3].endsWith(".dmg"))
    throw new Error("Unexpected pinned DMG builder invocation.");
  const directory = await download(pinnedDmgbuildToolset(architecture));
  if (typeof directory !== "string" || !isAbsolute(directory)) throw new Error("Pinned DMG toolset path is unavailable.");
  const vendor = join(directory, "dmgbuild");
  const help = execute(vendor, ["--help"], { encoding: "utf8", shell: false, timeout: 10_000, maxBuffer: 65_536 });
  if (help.error) throw help.error;
  if (help.status !== 0 || !String(help.stdout).includes("--detach-retries"))
    throw new Error("Pinned DMG builder does not advertise bounded detach retries.");
  // The vendor retains custody of its just-created temporary image and mount.
  // Never detach a guessed disk, force unmount, or retry an entire image build.
  const result = execute(vendor, ["--detach-retries", "30", ...args], { stdio: "inherit", shell: false });
  if (result.error) throw result.error;
  if (result.signal || !Number.isInteger(result.status)) throw new Error("Pinned DMG builder was interrupted before successful completion.");
  return result.status; // Nonzero remains fatal to electron-builder and its smoke gates.
}

if (process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1])) {
  runDmgbuild(process.argv.slice(2)).then(code => { process.exitCode = code; }, error => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
