// electron-builder afterPack: generated before signing and archive creation.
// It records actual packaged bytes; it never invokes the native worker.
module.exports = async function portableRuntimeAfterPack(context) {
  if (context.electronPlatformName === "darwin") {
    const { join } = require("node:path");
    const { finalizeAppNativeKernels } = require("./mac-native-kernel-sign.cjs");
    await finalizeAppNativeKernels(join(context.appOutDir, `${context.packager.appInfo.productFilename}.app`));
    return;
  }
  if (context.electronPlatformName !== "win32") return;
  const { Arch } = require("builder-util");
  const { pathToFileURL } = require("node:url");
  const { join } = require("node:path");
  const { writeWindowsPortableRuntimeReceipt } = await import(pathToFileURL(join(__dirname, "portable-runtime-receipt.mjs")).href);
  await writeWindowsPortableRuntimeReceipt({ directory: context.appOutDir, architecture: Arch[context.arch],
    version: context.packager.appInfo.version, sourceCommit: process.env.OMNI_RELEASE_COMMIT || null });
};
