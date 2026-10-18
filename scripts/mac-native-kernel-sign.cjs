// Keep the standard electron-builder signer, including its identity, retry,
// entitlements, hardened-runtime and subsequent notarization behavior.
const { realpath } = require("node:fs/promises");
const { realpathSync } = require("node:fs");
const { join, resolve, relative, isAbsolute } = require("node:path");
const { execFile, execFileSync } = require("node:child_process");
const { promisify } = require("node:util");

const executeFile = promisify(execFile);

function contained(root, path) {
  const value = relative(root, path);
  return value === "" || (!value.startsWith("..") && !isAbsolute(value));
}

function checkedFinalization(stdout) {
  const report = JSON.parse(String(stdout));
  if (!Array.isArray(report.nativeKernels) || report.nativeKernels.length !== 2 ||
      report.nativeKernels.some(value => value.compiledPayloadUnchanged !== true ||
        !/^[a-f0-9]{64}$/.test(value.sha256 || ""))) {
    throw new Error("macOS native kernel finalization did not verify both compiled payloads");
  }
  return report;
}

async function finalizeAppNativeKernels(app, execute = executeFile) {
  const result = await execute(process.env.OMNI_BUILD_PYTHON || "python3", ["-I",
    join(__dirname, "finalize-native-kernel-manifest.py"), "--app-root", resolve(app)],
  { timeout: 30_000, maxBuffer: 64 * 1024, windowsHide: true });
  return checkedFinalization(result.stdout);
}

function finalizeAppNativeKernelsSync(app, execute = execFileSync) {
  const output = execute(process.env.OMNI_BUILD_PYTHON || "python3", ["-I",
    join(__dirname, "finalize-native-kernel-manifest.py"), "--app-root", resolve(app)],
  { timeout: 30_000, maxBuffer: 64 * 1024, windowsHide: true, encoding: "utf8" });
  return checkedFinalization(output);
}

async function createNativeKernelSignOptions(options, { resolveReal = realpath, resolveFile = realpathSync, finalize = finalizeAppNativeKernelsSync } = {}) {
  if (options.platform !== "darwin" || typeof options.app !== "string" || !options.app.endsWith(".app")) {
    throw new Error("Native kernel signing requires its exact macOS app root");
  }
  const root = await resolveReal(resolve(options.app));
  const resources = join(root, "Contents", "Resources");
  const libraries = await Promise.all([
    join(resources, "engine", "omni_core", "native", "omni_packed_kernel.dylib"),
    join(resources, "engine-runtime", "_internal", "omni_core", "native", "omni_packed_kernel.dylib")
  ].map(path => resolveReal(path)));
  if (libraries.some(path => !contained(root, path))) throw new Error("Native kernel signer encountered a foreign library root");
  const signedChildren = new Set();
  let rootSeen = false;
  const original = options.optionsForFile;
  return {
    options: { ...options, optionsForFile: file => {
      // The pinned osx-sign API consumes this return synchronously. A Promise
      // would discard per-file entitlements and race the outer codesign seal.
      const path = resolveFile(resolve(file));
      if (!contained(root, path) || rootSeen) throw new Error("Unexpected native kernel signing root or callback order");
      const perFile = original ? original(file) : null;
      if (perFile && typeof perFile.then === "function") throw new Error("Native signing needs synchronous per-file options");
      if (path === root) {
        // The pinned standard signer invokes this callback after successfully
        // signing every child, and before codesign seals the outer Resources.
        if (libraries.some(library => !signedChildren.has(library))) {
          throw new Error("Native kernel manifests cannot precede nested library signing");
        }
        const result = finalize(root);
        if (result && typeof result.then === "function") throw new Error("Native kernel finalization must finish synchronously before the outer seal");
        rootSeen = true;
      } else if (libraries.includes(path)) {
        signedChildren.add(path);
      }
      return perFile;
    } },
    assertCompleted() {
      if (!rootSeen) throw new Error("The standard signer did not seal the expected native app root");
    }
  };
}

module.exports = async function signMacApp(options) {
  const { signAsync } = await import("@electron/osx-sign");
  const { retry } = require("builder-util");
  // Same pinned retry policy as electron-builder, with fresh callback custody
  // on each attempt rather than mistaking a retry for post-root child signing.
  await retry(async () => {
    const wrapped = await createNativeKernelSignOptions(options);
    await signAsync(wrapped.options);
    wrapped.assertCompleted();
  }, { retries: 3, interval: 5000, backoff: 5000 });
};
module.exports.createNativeKernelSignOptions = createNativeKernelSignOptions;
module.exports.finalizeAppNativeKernels = finalizeAppNativeKernels;
module.exports.finalizeAppNativeKernelsSync = finalizeAppNativeKernelsSync;
