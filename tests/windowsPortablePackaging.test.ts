import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, win32 } from "node:path";
import { pathToFileURL } from "node:url";
import { createRequire } from "node:module";
import { runInNewContext } from "node:vm";
import type { Writable } from "node:stream";
import { finished } from "node:stream/promises";
import { createPackage } from "@electron/asar";
import { afterEach, describe, expect, it } from "vitest";
const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
const require = createRequire(import.meta.url);
function pe(machine: number): Buffer { const bytes = Buffer.alloc(70); bytes.writeUInt16LE(0x5a4d, 0); bytes.writeUInt32LE(64, 60); bytes.writeUInt32LE(0x4550, 64); bytes.writeUInt16LE(machine, 68); return bytes; }
async function fixture(portable: boolean) {
  const root = await mkdtemp(join(tmpdir(), "omni-portable-packaging-")); roots.push(root);
  await mkdir(join(root, "resources", "engine-runtime"), { recursive: true }); await writeFile(join(root, "Omni AGI Studio.exe"), pe(0xaa64)); await writeFile(join(root, "resources", "engine-runtime", "omni-engine.exe"), pe(0x8664));
  const source = join(root, "owned-inert-asar"); await mkdir(join(source, "out", "main"), { recursive: true });
  await writeFile(join(source, "package.json"), JSON.stringify({ version: "1.1.1", main: "out/main/index.js" }));
  await writeFile(join(source, "out", "main", "index.js"), portable ? "/* Never executed fixture: omni-portable-launcher omni-portable-workspace portable-destination.json portable-adoption */" : "/* Deliberately stale inert main */");
  // This pinned ASAR version resolves to out.end()'s Writable before its OS
  // handle is closed. A receipt must inspect completed actual archive bytes.
  const archive = await createPackage(source, join(root, "resources", "app.asar")) as unknown as Writable;
  await finished(archive, { cleanup: true }); expect(archive.closed).toBe(true);
  return root;
}
describe("portable Windows build receipts (owned inert files only)", () => {
  it("afterPack uses the installed builder's real Arch enum, verifies compiled source presence and records only actual package-byte evidence", async () => {
    const root = await fixture(true), hook = require(join(process.cwd(), "scripts", "portable-runtime-after-pack.cjs")) as (context: unknown) => Promise<void>;
    await hook({ electronPlatformName: "linux" }); // No file lookup or receipt for another platform.
    await hook({ electronPlatformName: "win32", arch: 3, appOutDir: root, packager: { appInfo: { version: "1.1.1" } } });
    const url = pathToFileURL(join(process.cwd(), "scripts", "portable-runtime-receipt.mjs")).href;
    const reader = await import(url) as { verifyWindowsPortableRuntimeReceipt(value: { directory: string; version: string }): Promise<Record<string, unknown>> };
    expect(await reader.verifyWindowsPortableRuntimeReceipt({ directory: root, version: "1.1.1" })).toMatchObject({ desktopArchitecture: "arm64", workerArchitecture: "x64", runtimeReceiptVerified: true, selectedInstanceTested: false, destinationReservationTested: false, neuralAcceptanceTested: false });
    await writeFile(join(root, "resources", "app.asar"), "tampered"); await expect(reader.verifyWindowsPortableRuntimeReceipt({ directory: root, version: "1.1.1" })).rejects.toThrow("ASAR");
  });
  it("rejects a stale desktop build and includes the truthful portable receipt in both ZIP/installed Windows smoke records", async () => {
    const root = await fixture(false), hook = require(join(process.cwd(), "scripts", "portable-runtime-after-pack.cjs")) as (context: unknown) => Promise<void>;
    await expect(hook({ electronPlatformName: "win32", arch: 3, appOutDir: root, packager: { appInfo: { version: "1.1.1" } } })).rejects.toThrow("stale");
    const source = await readFile(join(process.cwd(), "scripts", "smoke-windows-package.ps1"), "utf8");
    expect(source).toContain("scripts\\verify-portable-runtime.mjs"); expect(source).toContain("portableRuntime = $ZipPortable"); expect(source).toContain("portableRuntime = $InstalledPortable"); expect(source).toContain("$Record.selectedInstanceTested -ne $false");
  });
  it("requires the release commit inside the shipped portable receipt and rejects missing or mismatched provenance", async () => {
    const root = await fixture(true);
    const url = pathToFileURL(join(process.cwd(), "scripts", "portable-runtime-receipt.mjs")).href;
    const receipts = await import(url) as {
      writeWindowsPortableRuntimeReceipt(value: { directory: string; architecture: string; version: string; sourceCommit: string }): Promise<unknown>;
      verifyWindowsPortableRuntimeReceipt(value: { directory: string; version: string; sourceCommit: string }): Promise<Record<string, unknown>>;
    };
    const sourceCommit = "a".repeat(40);
    await receipts.writeWindowsPortableRuntimeReceipt({ directory: root, architecture: "arm64", version: "1.1.1", sourceCommit });
    expect(await receipts.verifyWindowsPortableRuntimeReceipt({ directory: root, version: "1.1.1", sourceCommit })).toMatchObject({ sourceCommit, desktopArchitecture: "arm64", workerArchitecture: "x64" });
    await expect(receipts.verifyWindowsPortableRuntimeReceipt({ directory: root, version: "1.1.1", sourceCommit: "b".repeat(40) })).rejects.toThrow("source commit");
    const receiptPath = join(root, "resources", "portable-runtime.json");
    const stored = JSON.parse(await readFile(receiptPath, "utf8"));
    await writeFile(receiptPath, JSON.stringify({ ...stored, sourceCommit: null }));
    await expect(receipts.verifyWindowsPortableRuntimeReceipt({ directory: root, version: "1.1.1", sourceCommit })).rejects.toThrow("source commit");
  });
  it("uses the host ASAR path separator without changing the exact packaged entry-point contract", async () => {
    // Exercise the installed dependency's actual header traversal with Win32
    // path operations, not a newly invented mock lookup or a running EXE.
    const dependencyPath = join(dirname(require.resolve("@electron/asar")), "filesystem.js");
    const dependency = createRequire(dependencyPath), owned = { exports: {} as { Filesystem?: new (root: string) => {
      setHeader(header: unknown, size: number): void; getFile(path: string, follow: boolean): unknown;
    } } };
    runInNewContext(await readFile(dependencyPath, "utf8"), { module: owned, exports: owned.exports,
      require: (id: string) => id === "path" ? win32 : dependency(id), Buffer, BigInt });
    const filesystem = new owned.exports.Filesystem!("C:\\owned-inert-app");
    filesystem.setHeader({ files: { out: { files: { main: { files: { "index.js": { size: 4, offset: "0" } } } } } } }, 0);
    expect(() => filesystem.getFile("out/main/index.js", false)).toThrow(/not found/);
    expect(filesystem.getFile(win32.normalize("out/main/index.js"), false)).toMatchObject({ size: 4, offset: "0" });
    const source = await readFile(join(process.cwd(), "scripts", "portable-runtime-receipt.mjs"), "utf8");
    expect(source).toContain('packaged.main !== "out/main/index.js"');
    expect(source).toContain("extractFile(path, normalize(packaged.main), false)");
    expect(source).not.toContain("extractFile(path, packaged.main, false)");
  });
});
