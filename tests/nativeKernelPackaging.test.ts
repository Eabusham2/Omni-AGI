import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { runInNewContext } from "node:vm";
import { describe, expect, it } from "vitest";

const require = createRequire(import.meta.url);
const signer = require("../scripts/mac-native-kernel-sign.cjs") as {
  createNativeKernelSignOptions(options: Record<string, unknown>, dependencies: Record<string, unknown>): Promise<{
    options: Record<string, unknown> & { optionsForFile(file: string): unknown }; assertCompleted(): void;
  }>;
  finalizeAppNativeKernels(app: string, execute: (...args: unknown[]) => Promise<{ stdout: string }>): Promise<unknown>;
};
const health = await import(pathToFileURL(resolve("scripts/packaged-worker-health.mjs")).href) as {
  packagedWorkerHealthDiagnostic(value: unknown, platform: string): Record<string, unknown>;
};
const app = resolve("work/Kernel fixture.app");
const nativeLibraries = [join(app, "Contents/Resources/engine/omni_core/native/omni_packed_kernel.dylib"),
  join(app, "Contents/Resources/engine-runtime/_internal/omni_core/native/omni_packed_kernel.dylib")];

describe("native macOS kernel package custody", () => {
  it("delegates all original signing options and reseals only after both children before the outer app", async () => {
    const events: string[] = [];
    const perFile = { entitlements: "original.entitlements", hardenedRuntime: true };
    const original = { app, platform: "darwin", identity: "original-identity-hash", keychain: "original-keychain",
      hardenedRuntime: true, optionsForFile: () => perFile };
    const wrapped = await signer.createNativeKernelSignOptions(original, { resolveReal: async (path: string) => path,
      resolveFile: (path: string) => path, finalize: () => { events.push("manifest-seal"); } });
    for (const key of ["identity", "keychain", "hardenedRuntime"]) expect(wrapped.options[key]).toBe(original[key as keyof typeof original]);
    for (const path of nativeLibraries) {
      expect(await wrapped.options.optionsForFile(path)).toBe(perFile);
      events.push("child-signed");
    }
    expect(await wrapped.options.optionsForFile(app)).toBe(perFile);
    events.push("outer-signed"); wrapped.assertCompleted();
    expect(events).toEqual(["child-signed", "child-signed", "manifest-seal", "outer-signed"]);
    const vendor = readFileSync(require.resolve("@electron/osx-sign/dist/cjs/sign.js"), "utf8");
    expect(vendor).toContain("for (const filePath of [...children, opts.app])");
    const signingLoop = vendor.slice(vendor.indexOf("for (const filePath of [...children, opts.app])"));
    expect(signingLoop.indexOf("opts.optionsForFile(filePath)")).toBeLessThan(signingLoop.indexOf("await (0, util_1.execFileAsync)('codesign'"));
  });

  it("rejects missing children, foreign roots, after-root callbacks and incomplete signers", async () => {
    let seals = 0;
    const setup = () => signer.createNativeKernelSignOptions({ app, platform: "darwin" }, {
      resolveReal: async (path: string) => path, resolveFile: (path: string) => path, finalize: () => { seals += 1; } });
    const first = await setup();
    expect(() => first.options.optionsForFile(app)).toThrow("cannot precede");
    expect(() => first.assertCompleted()).toThrow("expected native app root");
    expect(() => first.options.optionsForFile(resolve("elsewhere/foreign.dylib"))).toThrow("Unexpected");
    const second = await setup();
    for (const path of nativeLibraries) await second.options.optionsForFile(path);
    await second.options.optionsForFile(app);
    expect(() => second.options.optionsForFile(nativeLibraries[0]!)).toThrow("callback order");
    expect(seals).toBe(1);
  });

  it("matches the installed signer's synchronous options API and finishes sealing before its real outer signing call", async () => {
    const sourcePath = require.resolve("@electron/osx-sign/dist/cjs/sign.js"), dependency = createRequire(sourcePath);
    const events: string[] = [], signingArguments: string[][] = [];
    const owned = { exports: {} as { signApp(options: unknown): Promise<void> } };
    const inertUtil = { debugLog: () => undefined, debugWarn: () => undefined,
      validateOptsApp: async () => undefined, validateOptsPlatform: async () => "darwin",
      getAppContentsPath: () => join(app, "Contents"), walkAsync: async () => [...nativeLibraries],
      execFileAsync: async (command: string, args: string[]) => {
        expect(command).toBe("codesign");
        if (args[0] === "--sign") { signingArguments.push(args); events.push(args.at(-1) === app ? "outer-signed" : "child-signed"); }
        return "";
      } };
    runInNewContext(readFileSync(sourcePath, "utf8"), { module: owned, exports: owned.exports, __dirname: dirname(sourcePath),
      require: (id: string) => id === "./util" ? inertUtil : id === "os" ? { release: () => "24.6.0" } : dependency(id) });
    const original = { app, platform: "darwin", identity: "owned-identity", identityValidation: false,
      keychain: "owned-keychain", preEmbedProvisioningProfile: false, preAutoEntitlements: false,
      optionsForFile: () => ({ entitlements: "owned-original.entitlements", hardenedRuntime: true,
        additionalArguments: ["--identifier", "owned.native.fixture"] }) };
    const wrapped = await signer.createNativeKernelSignOptions(original, { resolveReal: async (path: string) => path,
      resolveFile: (path: string) => path, finalize: () => { events.push("manifest-seal"); } });
    // Execute the installed signing loop with inert process callbacks only.
    // An async optionsForFile loses entitlements and leaves rootSeen false here.
    await owned.exports.signApp(wrapped.options); wrapped.assertCompleted();
    expect(events).toEqual(["child-signed", "child-signed", "manifest-seal", "outer-signed"]);
    expect(signingArguments).toHaveLength(3);
    for (const args of signingArguments) {
      expect(args).toEqual(expect.arrayContaining(["--keychain", "owned-keychain", "--entitlements", "owned-original.entitlements",
        "--identifier", "owned.native.fixture", "--options", "runtime"]));
    }
  });

  it("runs only the contained stdlib finalizer with bounded output and requires two verified payloads", async () => {
    const calls: unknown[][] = [];
    const report = { nativeKernels: Array.from({ length: 2 }, () => ({ compiledPayloadUnchanged: true, sha256: "a".repeat(64) })) };
    expect(await signer.finalizeAppNativeKernels(app, async (...args) => { calls.push(args); return { stdout: JSON.stringify(report) }; })).toEqual(report);
    expect(calls).toHaveLength(1);
    expect(calls[0]![1]).toEqual(["-I", resolve("scripts/finalize-native-kernel-manifest.py"), "--app-root", app]);
    expect(calls[0]![2]).toMatchObject({ timeout: 30_000, maxBuffer: 65_536, windowsHide: true });
    await expect(signer.finalizeAppNativeKernels(app, async () => ({ stdout: JSON.stringify({ nativeKernels: [{}] }) }))).rejects.toThrow("both compiled payloads");
  });

  it("reports the exact known integrity reason without emitting raw errors paths keys or arbitrary fields", () => {
    const result = health.packagedWorkerHealthDiagnostic({ ready: false, worker: "python", protocolVersion: 1,
      operatingSystem: process.platform, loadedBrains: 0, engineVersion: "2", pythonVersion: "3", torchVersion: "2", platform: "native",
      capabilities: { nativePackedKernels: { cpu: false, error: "Native packed kernel integrity verification failed" } },
      apiKey: "do-not-output" }, process.platform);
    expect(result).toMatchObject({ ready: false, nativePackedCpu: false, loadedBrains: 0, nativeKernelFailure: "native-library-integrity-mismatch" });
    const hidden = health.packagedWorkerHealthDiagnostic({ ready: false, worker: "private-user-path", loadedBrains: "private-secret",
      capabilities: { nativePackedKernels: { error: "/private/user/credential=do-not-output" } } }, process.platform);
    expect(JSON.stringify([result, hidden])).not.toMatch(/do-not-output|private-user-path|private-secret|credential=/u);
    expect(JSON.stringify(hidden).length).toBeLessThan(1024);
  });

  it("keeps full hash and compiled payload checks across unsigned and signed package paths", () => {
    const pkg = JSON.parse(readFileSync(resolve("package.json"), "utf8")) as { build: { mac: { sign: string }; afterPack: string } };
    expect(pkg.build.mac.sign).toBe("./scripts/mac-native-kernel-sign.cjs");
    const afterPack = readFileSync(resolve(pkg.build.afterPack), "utf8");
    expect(afterPack).toContain('context.electronPlatformName === "darwin"');
    expect(afterPack).toContain("writeWindowsPortableRuntimeReceipt");
    const source = readFileSync(resolve("scripts/mac-native-kernel-sign.cjs"), "utf8");
    expect(source).toContain('await import("@electron/osx-sign")');
    expect(source).toContain("{ retries: 3, interval: 5000, backoff: 5000 }");
    expect(source).not.toContain("signIgnore");
    const smoke = readFileSync(resolve("scripts/smoke-engine.mjs"), "utf8");
    expect(smoke).toContain("healthOnly && health?.loadedBrains !== 0");
    expect(smoke).toContain("packagedWorkerHealthDiagnostic(health, process.platform)");
  });
});
