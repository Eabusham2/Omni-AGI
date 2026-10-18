import { access, mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { constants } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { inspectLimaVm, prepareSandboxEnvironment, runSandboxEnvironment } from "../src/main/environmentAdapters";
import { runProcess } from "../src/main/toolExecutor";
import { ENVIRONMENT_RECORD_FORMAT, type ManagedEnvironmentRecord } from "../src/shared/environments";

const roots: string[] = [];
afterAll(async () => { await Promise.all(roots.map(path => rm(path, { recursive: true, force: true }))); });
const available = process.platform === "darwin" && await access("/usr/bin/sandbox-exec", constants.X_OK).then(() => true, () => false);

describe.skipIf(!available)("actual bounded Darwin sandbox adapter", () => {
  it("runs a harmless system fixture, writes only its workspace, and denies an outside source read", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-sandbox-host-")), directory = join(root, "environment"), workspace = join(directory, "workspace"); roots.push(root);
    await mkdir(join(workspace, "tmp"), { recursive: true }); const outside = join(root, "outside.txt"); await writeFile(outside, "outside-secret");
    const at = new Date().toISOString(), record: ManagedEnvironmentRecord = { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: "f".repeat(32),
      brainId: "55555555-5555-4555-8555-555555555555", kind: "sandbox", name: "Host fixture", adapter: "darwin-sandbox-exec", state: "loaded",
      revision: 2, network: "none", createdAt: at, updatedAt: at };
    await prepareSandboxEnvironment(record, directory);
    const deps = { platform: "darwin" as const, run: runProcess, findExecutable: async (name: string) => name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : undefined };
    const harmless = await runSandboxEnvironment(deps, record, directory, "/usr/bin/true", [], 5_000, new AbortController().signal);
    expect(harmless.exitCode, JSON.stringify(harmless)).toBe(0);
    process.env.OMNI_SANDBOX_SYNTHETIC_SECRET = "must-not-cross";
    try {
      const environment = await runSandboxEnvironment(deps, record, directory, "/usr/bin/env", [], 5_000, new AbortController().signal);
      expect(environment.exitCode).toBe(0); expect(environment.stdout).not.toContain("OMNI_SANDBOX_SYNTHETIC_SECRET");
    } finally { delete process.env.OMNI_SANDBOX_SYNTHETIC_SECRET; }
    const write = await runSandboxEnvironment(deps, record, directory, "/usr/bin/touch", [join(workspace, "owned.txt")], 5_000, new AbortController().signal);
    expect(write.exitCode, JSON.stringify(write)).toBe(0); await expect(access(join(workspace, "owned.txt"))).resolves.toBeUndefined();
    const denied = await runSandboxEnvironment(deps, record, directory, "/bin/cat", [outside], 5_000, new AbortController().signal);
    expect(denied.exitCode).not.toBe(0); expect(denied.stdout).not.toContain("outside-secret");
  });
});

describe("managed VM adapter state normalization", () => {
  it("does not misreport transitional or broken Lima states as stopped", async () => {
    const at = new Date().toISOString(), record: ManagedEnvironmentRecord = { format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: "e".repeat(32),
      brainId: "55555555-5555-4555-8555-555555555555", kind: "vm", name: "VM fixture", adapter: "lima", state: "review-required", revision: 2,
      network: "guest-outbound", createdAt: at, updatedAt: at, externalId: "omni-555555555555-eeeeeeeeeeee" };
    const state = await inspectLimaVm({ run: async (_command, args) => ({ exitCode: 0, stdout: JSON.stringify({ name: args.at(-1), status: "Broken" }), stderr: "", truncated: false }),
      findExecutable: async name => name === "limactl" ? "/fixture/limactl" : undefined }, record, "/fixture", "/fixture/lima-home", new AbortController().signal);
    expect(state).toBe("unknown");
  });
});
