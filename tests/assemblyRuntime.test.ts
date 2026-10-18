/** Constructor-free adapter fixtures. Mock compiler output is explicitly not native execution. */
import { mkdir, mkdtemp, readFile, rm, writeFile, symlink } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { executeAssembly, AssemblyExecutionError, type AssemblyDependencies } from "../src/main/assemblyRuntime";
const roots: string[] = [];
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-assembly-adapter-")); roots.push(root);
  let failure = false, refuse = false, wait = false;
  const calls: Array<{ command: string; args: string[]; cwd: string }> = [];
  const run: AssemblyDependencies["run"] = async (command, args, cwd, _timeout, _env, signal) => {
    calls.push({ command, args: [...args], cwd }); signal?.throwIfAborted();
    if (args[0] === "--version") return { exitCode: 0, stdout: "Fixture compiler, not a real toolchain", stderr: "", truncated: false };
    if (args[0] === "-dumpmachine") return { exitCode: 0, stdout: "arm64-apple-darwin\n", stderr: "", truncated: false };
    const output = args.indexOf("-o");
    if (output >= 0) {
      await writeFile(args[output + 1]!, failure || wait ? "mock partial output" : "mock native executable bytes");
      if (wait) await new Promise<void>((_resolve, reject) => { signal!.addEventListener("abort", () => reject(new Error("Fixture process cancelled")), { once: true }); });
      return { exitCode: failure ? 1 : 0, stdout: "", stderr: failure ? "fixture syntax failure" : "", truncated: false };
    }
    return { exitCode: 0, stdout: "fixture mock output\n", stderr: "", truncated: false };
  };
  const dependencies: AssemblyDependencies = { root, platform: "darwin", arch: "arm64", run,
    findExecutable: async name => name === "clang" ? "/fixture/clang" : undefined,
    checkResources: async () => { if (refuse) throw new Error("Fixture resource refusal"); }, reserveWrite: async () => undefined };
  const execute = (action: string, args: Record<string, unknown>, signal = new AbortController().signal) => executeAssembly(dependencies, action, args, signal);
  return { root, calls, execute, dependencies, fail() { failure = true; }, refuse() { refuse = true; }, wait() { wait = true; } };
}
describe("explicit assembly adapter", () => {
  it("inspects actual selected command availability without guessing unsupported installed tools", async () => {
    const f = await fixture(), result = await f.execute("inspect", {}) as any;
    expect(result.platform).toBe("darwin"); expect(result.architecture).toBe("arm64");
    expect(result.toolchains.find((row: any) => row.id === "clang")).toMatchObject({ available: true, path: "/fixture/clang", target: "arm64-apple-darwin" });
    expect(result.toolchains.find((row: any) => row.id === "nasm").available).toBe(false);
  });
  it("retains exact source and hashed managed output, then runs exact argv without shell interpolation or truncation", async () => {
    const f = await fixture(), source = ".text\n// exact source\n";
    const compiled = await f.execute("compile", { source, kind: "executable", toolchain: "clang" }) as any;
    expect(compiled.state).toBe("compiled"); expect(await readFile(join(compiled.directory, "source.s"), "utf8")).toBe(source);
    expect(compiled.artifact.sha256).toMatch(/^[a-f0-9]{64}$/);
    const args = ["one argument", "$(must stay literal)", ...Array.from({ length: 70 }, (_, index) => `arg-${index}`)];
    const result = await f.execute("run", { artifactId: compiled.artifactId, artifactSha256: compiled.artifact.sha256, arguments: args }) as any;
    expect(result.state).toBe("exited"); expect(f.calls.at(-1)!.args).toEqual(args); expect(result.stdoutSha256).toMatch(/^[a-f0-9]{64}$/);
    expect(result.outputHashScope).toBe("captured-bounded-output");
  });
  it("preserves partial compiler outputs and exact error receipt without calling a failed artifact runnable", async () => {
    const f = await fixture(); f.fail(); const result = await f.execute("compile", { source: "bad syntax" }) as any;
    expect(result.state).toBe("compile-failed"); expect(result.processes[0]).toMatchObject({ exitCode: 1, stderr: "fixture syntax failure" });
    expect(result.partialFiles.some((row: any) => row.name === "program" && /^[a-f0-9]{64}$/.test(row.sha256))).toBe(true);
    await expect(f.execute("run", { artifactId: result.artifactId, artifactSha256: "a".repeat(64) })).rejects.toThrow(/compiled executable/);
  });
  it("cancels through the supplied real runner boundary and durably retains partial output", async () => {
    const f = await fixture(); f.wait(); const controller = new AbortController();
    const pending = f.execute("compile", { source: "fixture wait" }, controller.signal);
    await vi.waitFor(() => expect(f.calls.some(row => row.args.includes("-o"))).toBe(true)); controller.abort();
    let caught: unknown; try { await pending; } catch (error) { caught = error; }
    expect(caught).toBeInstanceOf(AssemblyExecutionError); const receipt = (caught as AssemblyExecutionError).receipt;
    expect(receipt.state).toBe("cancelled"); expect(receipt.partialFiles.length).toBeGreaterThan(0);
    expect(JSON.parse(await readFile(receipt.receiptPath, "utf8"))).toMatchObject({ state: "cancelled" });
  });
  it("refuses budget pressure, malformed operands and unsupported formats rather than rewriting or truncating requests", async () => {
    const f = await fixture(); f.refuse(); await expect(f.execute("compile", { source: "fixture" })).rejects.toThrow(/resource refusal/); expect(f.calls).toHaveLength(0);
    const g = await fixture(); for (const args of [{ source: "", timeoutMs: 0 }, { source: "fixture", flags: ["-o", "/foreign"] }, { source: "fixture", syntax: "unknown" }, { source: "fixture", timeoutMs: 999999999 }])
      await expect(g.execute("compile", args)).rejects.toThrow(); expect(g.calls).toHaveLength(0);
  });
  it("refuses stale binary hashes and unsafe linked artifact directories before native execution", async () => {
    const f = await fixture(), compiled = await f.execute("compile", { source: "fixture" }) as any, count = f.calls.length;
    await writeFile(join(compiled.directory, "program"), "changed binary");
    await expect(f.execute("run", { artifactId: compiled.artifactId, artifactSha256: compiled.artifact.sha256 })).rejects.toThrow(/hash|changed/); expect(f.calls).toHaveLength(count);
    const outside = join(f.root, "outside"); await mkdir(outside); await symlink(outside, join(f.root, "assembly", "b".repeat(32)));
    await expect(f.execute("run", { artifactId: "b".repeat(32), artifactSha256: "a".repeat(64) })).rejects.toThrow(/directory|link/);
  });
});
