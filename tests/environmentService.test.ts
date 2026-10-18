import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EnvironmentOperationError, EnvironmentService } from "../src/main/environmentService";
import type { EnvironmentProcessRunner } from "../src/main/environmentAdapters";
import { EnvironmentStore } from "../src/main/environmentStore";
import { ENVIRONMENT_OPERATION_FORMAT, ENVIRONMENT_RECORD_FORMAT } from "../src/shared/environments";

const roots: string[] = [];
const brainId = "22222222-2222-4222-8222-222222222222";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });
async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-environment-service-")); roots.push(root);
  const calls: Array<{ command: string; args: string[]; environment?: Record<string, string> }> = [];
  const run: EnvironmentProcessRunner = vi.fn(async (command, args, _cwd, _timeout, environment, signal, onSpawn) => {
    calls.push({ command, args: [...args], environment }); onSpawn?.(4321);
    if (args.includes("__hold")) await new Promise<void>((resolve, reject) => {
      const abort = () => reject(signal?.reason ?? new Error("cancelled")); signal?.addEventListener("abort", abort, { once: true });
      if (signal?.aborted) abort();
    });
    if (args.includes("__bad")) throw new Error(`fixture\0${"x".repeat(9000)}`);
    return { exitCode: 0, stdout: "fixture-output", stderr: "", truncated: false };
  });
  const service = new EnvironmentService({ brainDirectory: async () => root, run, platform: "darwin",
    findExecutable: async name => name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : name === "limactl" ? "/fixture/limactl" : undefined,
    checkResources: async () => undefined, reserveWrite: async () => undefined });
  return { root, calls, service };
}

describe("managed sandbox and VM tool service", () => {
  it("creates, loads and uses one actual owned sandbox identity with lost-ACK idempotence", async () => {
    const { calls, service } = await fixture(), signal = new AbortController().signal, createArgs = { name: "Research box" };
    const create = service.execute(brainId, "runtime.sandbox", "create", createArgs, "create-request", signal); createArgs.name = "Mutated after call";
    const created = await create as any;
    expect(created.environment).toMatchObject({ kind: "sandbox", name: "Research box", state: "ready", network: "none" });
    const repeated = await service.execute(brainId, "runtime.sandbox", "create", { name: "Research box" }, "create-request", signal) as any;
    expect(repeated.environment.id).toBe(created.environment.id);
    const loaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "load-request", signal) as any;
    expect(loaded.environment.state).toBe("loaded");
    await expect(service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "load-again", signal)).rejects.toThrow(/not at a loadable boundary/i);
    expect(((await service.execute(brainId, "runtime.sandbox", "status", { environmentId: created.environment.id }, "status-after-load", signal)) as any).environment.state).toBe("loaded");
    const used = await service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id, program: "/usr/bin/true", arguments: ["ok"] }, "run-request", signal) as any;
    expect(used).toMatchObject({ environment: { state: "loaded" }, operation: { state: "complete", process: { stdout: "fixture-output" } } });
    const dispatch = calls.at(-1)!;
    expect(dispatch.command).toBe("/usr/bin/env"); expect(dispatch.args.slice(0, 2)).toEqual(["-i", expect.stringMatching(/^HOME=/)]);
    expect(dispatch.args).toContain("/usr/bin/sandbox-exec"); expect(dispatch.environment).toEqual({});
    try { await service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id, program: "/usr/bin/true", arguments: ["__bad"] }, "run-bad", signal); throw new Error("Expected fixture failure"); }
    catch (error) { expect(error).toBeInstanceOf(EnvironmentOperationError); const retained = (error as EnvironmentOperationError).result;
      expect(retained.operation.error?.length).toBeLessThanOrEqual(8000); expect(retained.operation.error).not.toContain("\0"); expect(retained.environment.warning?.length).toBeLessThanOrEqual(2000); }
    await service.shutdown();
  });

  it("cancels a live exact operation through Stop and never substitutes a VM host runner", async () => {
    const { service } = await fixture(), signal = new AbortController().signal;
    const created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Stop box" }, "create-stop", signal) as any;
    await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "load-stop", signal);
    const running = service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id, program: "/usr/bin/true", arguments: ["__hold"] }, "run-stop", signal);
    const runningFailure = expect(running).rejects.toBeInstanceOf(EnvironmentOperationError);
    await expect(service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id, program: "/usr/bin/true", arguments: ["different"] }, "run-stop", signal)).rejects.toThrow(/active with different operands/i);
    await vi.waitFor(async () => { const status = await service.execute(brainId, "runtime.sandbox", "status", { environmentId: created.environment.id }, "status-ignored", signal) as any; expect(status.environment.state).toBe("busy"); });
    const stopped = await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: created.environment.id }, "stop-request", signal) as any;
    await runningFailure;
    expect(stopped).toMatchObject({ environment: { state: "stopped" }, operation: { state: "complete" } });

    const capabilities = await service.execute(brainId, "runtime.vm", "capabilities", {}, "capability-ignored", signal) as any;
    expect(capabilities.adapters.find((row: any) => row.id === "lima")).toMatchObject({ available: true, supportsCreate: false, requiresLocalImage: true });
    await expect(service.execute(brainId, "runtime.vm", "create", { name: "No fake VM", imageAttachmentId: "x", imageSha256: "a".repeat(64), architecture: "aarch64", network: "guest-outbound" }, "vm-create", signal)).rejects.toThrow(/hash-verified local-image resolver/i);
  });

  it("binds VM creation to exact local image bytes and generates no download, mount, setup-script or credential operand", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-environment-vm-")); roots.push(root);
    const image = join(root, "reviewed-image.qcow2"), bytes = Buffer.from("tiny fixture image bytes"), sha256 = createHash("sha256").update(bytes).digest("hex"); await writeFile(image, bytes);
    const calls: Array<{ args: string[]; config?: string }> = []; let failStop = false;
    const run: EnvironmentProcessRunner = async (_command, args, _cwd, _timeout, _environment, _signal, onSpawn) => {
      onSpawn?.(8765); const configPath = args.includes("--tty=false") && args.includes("create") ? args.at(-1) : undefined;
      calls.push({ args: [...args], ...(configPath ? { config: await readFile(configPath, "utf8") } : {}) });
      return { exitCode: failStop && args.includes("stop") ? 3 : 0, stdout: args.includes("--version") ? "limactl version fixture" : args.includes("--json") ? JSON.stringify({ name: args.at(-1), status: "Running" }) : "",
        stderr: failStop && args.includes("stop") ? "fixture stop refusal" : "", truncated: false };
    };
    const deleteVmRuntime = vi.fn(async () => undefined);
    const service = new EnvironmentService({ brainDirectory: async () => root, vmRuntimeDirectory: async (_brain, environmentId) => join(root, "private-vm-runtime", environmentId), deleteVmRuntime, run, platform: "darwin", findExecutable: async name => name === "limactl" ? "/fixture/limactl" : name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : undefined,
      checkResources: async () => undefined, reserveWrite: async () => undefined,
      resolveVmImage: async (_brain, turnId, attachmentId, expected, architecture) => {
        expect(turnId).toBe("turn-vm"); expect(attachmentId).toBe("image-attachment"); expect(expected).toBe(sha256);
        return { attachmentId, path: image, sha256, bytes: bytes.byteLength, architecture };
      } });
    const signal = new AbortController().signal, created = await service.execute(brainId, "runtime.vm", "create", { name: "Local image VM", imageAttachmentId: "image-attachment", imageSha256: sha256,
      architecture: "aarch64", cpus: 2, memoryGiB: 2, diskGiB: 20, network: "guest-outbound" }, "vm-local-create", signal, { chatTurnId: "turn-vm" }) as any;
    expect(created.environment).toMatchObject({ kind: "vm", adapter: "lima", state: "ready", imageSha256: sha256, network: "guest-outbound" });
    const creation = calls.find(call => call.args.includes("create"))!;
    expect(creation.args).toEqual(expect.arrayContaining(["--plain", "--mount-none", "--containerd=none"]));
    expect(creation.args.join(" ")).not.toMatch(/https?:|template:|--mount=/i);
    expect(creation.args.slice(0, 2)).toEqual(["-i", expect.stringMatching(/^HOME=/)]); expect(creation.args.some(value => value.startsWith("LIMA_HOME="))).toBe(true);
    expect(creation.args).toContain("PATH=/usr/bin:/bin:/usr/sbin:/sbin"); expect(creation.args).toContain("SSH=/usr/bin/ssh");
    expect(creation.config).toContain(`digest: \"sha256:${sha256}\"`); expect(creation.config).toContain("mounts: []"); expect(creation.config).toContain("provision: []"); expect(creation.config).not.toMatch(/https?:/i);
    const loaded = await service.execute(brainId, "runtime.vm", "load", { environmentId: created.environment.id }, "vm-local-load", signal) as any; expect(loaded.environment.state).toBe("loaded");
    const used = await service.execute(brainId, "runtime.vm", "run", { environmentId: created.environment.id, program: "/usr/bin/true", arguments: [] }, "vm-local-run", signal) as any; expect(used.operation.state).toBe("complete");
    expect(calls.find(call => call.args.includes("shell"))?.args).not.toContain("--preserve-env");
    failStop = true;
    try { await service.execute(brainId, "runtime.vm", "stop", { environmentId: created.environment.id }, "vm-local-stop-fails", signal); throw new Error("Expected stop failure"); }
    catch (error) { expect(error).toBeInstanceOf(EnvironmentOperationError); expect((error as EnvironmentOperationError).result).toMatchObject({ environment: { state: "review-required" }, operation: { state: "failed", process: { exitCode: 3 } } }); }
    const reconciled = await service.execute(brainId, "runtime.vm", "status", { environmentId: created.environment.id }, "vm-status-after-stop", signal) as any; expect(reconciled.environment.state).toBe("loaded");
    failStop = false; const stopped = await service.execute(brainId, "runtime.vm", "stop", { environmentId: created.environment.id }, "vm-local-stop", signal) as any; expect(stopped.environment.state).toBe("stopped");
    await service.deleteInstallationRuntime(brainId); expect(deleteVmRuntime).toHaveBeenCalledWith(brainId); await service.shutdown();
  });

  it("does not let a delayed VM status probe overwrite a newer busy owner", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-environment-status-race-")); roots.push(root);
    const image = join(root, "image.qcow2"), bytes = Buffer.from("status fixture image"), sha256 = createHash("sha256").update(bytes).digest("hex"); await writeFile(image, bytes);
    let enterStatus!: () => void, releaseStatus!: () => void, enterRun!: () => void, releaseRun!: () => void;
    const statusEntered = new Promise<void>(resolve => { enterStatus = resolve; }), statusReleased = new Promise<void>(resolve => { releaseStatus = resolve; });
    const runEntered = new Promise<void>(resolve => { enterRun = resolve; }), runReleased = new Promise<void>(resolve => { releaseRun = resolve; });
    const run: EnvironmentProcessRunner = async (_command, args) => {
      if (args.includes("--version")) return { exitCode: 0, stdout: "fixture lima", stderr: "", truncated: false };
      if (args.includes("--json")) { enterStatus(); await statusReleased; return { exitCode: 0, stdout: JSON.stringify({ name: args.at(-1), status: "Running" }), stderr: "", truncated: false }; }
      if (args.includes("shell")) { enterRun(); await runReleased; }
      return { exitCode: 0, stdout: "", stderr: "", truncated: false };
    };
    const service = new EnvironmentService({ brainDirectory: async () => root, vmRuntimeDirectory: async (_brain, environmentId) => join(root, "private-runtime", environmentId), run,
      platform: "darwin", findExecutable: async name => name === "limactl" ? "/fixture/limactl" : name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : undefined,
      checkResources: async () => undefined, reserveWrite: async () => undefined,
      resolveVmImage: async (_brain, _turn, attachmentId, _expected, architecture) => ({ attachmentId, path: image, sha256, bytes: bytes.byteLength, architecture }) });
    const signal = new AbortController().signal, created = await service.execute(brainId, "runtime.vm", "create", { name: "Race VM", imageAttachmentId: "race-image", imageSha256: sha256,
      architecture: "aarch64", network: "guest-outbound" }, "race-create", signal, { chatTurnId: "race-turn" }) as any;
    await service.execute(brainId, "runtime.vm", "load", { environmentId: created.environment.id }, "race-load", signal);
    const status = service.execute(brainId, "runtime.vm", "status", { environmentId: created.environment.id }, "race-status", signal); await statusEntered;
    const use = service.execute(brainId, "runtime.vm", "run", { environmentId: created.environment.id, program: "/usr/bin/true" }, "race-run", signal); await runEntered;
    releaseStatus(); expect((await status as any).environment.state).toBe("busy");
    releaseRun(); expect((await use as any).operation.state).toBe("complete"); await service.shutdown();
  });

  it("keeps an unreconciled crashed sandbox sticky across repeated Stop and Load attempts", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-environment-orphan-")); roots.push(root); const environmentId = "9".repeat(32), operationId = "8".repeat(32), at = new Date().toISOString();
    const store = await EnvironmentStore.open(root, brainId); await import("node:fs/promises").then(fs => fs.mkdir(store.instanceDirectory(environmentId), { recursive: true }));
    store.putEnvironment({ format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: environmentId, brainId, kind: "sandbox", name: "Interrupted",
      adapter: "darwin-sandbox-exec", state: "busy", revision: 2, network: "none", createdAt: at, updatedAt: at, lastOperationId: operationId });
    store.putOperation({ format: ENVIRONMENT_OPERATION_FORMAT, version: 1, id: operationId, brainId, environmentId, requestId: "orphan-run",
      action: "run", state: "running", argumentsSha256: "7".repeat(64), dispatchStarted: true, createdAt: at, updatedAt: at }); store.close();
    const service = new EnvironmentService({ brainDirectory: async () => root, run: async () => ({ exitCode: 0, stdout: "", stderr: "", truncated: false }), platform: "darwin",
      findExecutable: async name => name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : undefined, checkResources: async () => undefined, reserveWrite: async () => undefined });
    const signal = new AbortController().signal, first = await service.execute(brainId, "runtime.sandbox", "status", { environmentId }, "status-orphan", signal) as any;
    expect(first.environment).toMatchObject({ state: "review-required", orphanedProcessRisk: true });
    for (const [request, action] of [["stop-orphan-one", "stop"], ["stop-orphan-two", "stop"], ["load-orphan", "load"]] as const) {
      await expect(service.execute(brainId, "runtime.sandbox", action, { environmentId }, request, signal)).rejects.toThrow(/crashed process|safely stopped|unreconciled/i);
      const current = await service.execute(brainId, "runtime.sandbox", "status", { environmentId }, `status-${request}`, signal) as any;
      expect(current.environment).toMatchObject({ state: "review-required", orphanedProcessRisk: true });
    }
  });

  it("keeps a pre-dispatch cancelled Create non-loadable instead of inventing a workspace", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-environment-create-cancel-")); roots.push(root); const controller = new AbortController();
    const service = new EnvironmentService({ brainDirectory: async () => root, run: async () => ({ exitCode: 0, stdout: "", stderr: "", truncated: false }), platform: "darwin",
      findExecutable: async name => name === "sandbox-exec" ? "/usr/bin/sandbox-exec" : undefined, checkResources: async () => undefined,
      reserveWrite: async () => { controller.abort(new Error("fixture cancellation before dispatch")); } });
    let environmentId = "";
    try { await service.execute(brainId, "runtime.sandbox", "create", { name: "Cancelled" }, "cancel-create", controller.signal); throw new Error("Expected Create cancellation"); }
    catch (error) { expect(error).toBeInstanceOf(EnvironmentOperationError); environmentId = (error as EnvironmentOperationError).result.environment.id;
      expect((error as EnvironmentOperationError).result).toMatchObject({ environment: { state: "error" }, operation: { state: "cancelled", dispatchStarted: false } }); }
    await expect(service.execute(brainId, "runtime.sandbox", "load", { environmentId }, "cancelled-load", new AbortController().signal)).rejects.toThrow(/not at a loadable boundary/i);
  });
});
