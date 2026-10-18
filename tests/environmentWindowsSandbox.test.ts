import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  environmentAdapterCapabilities,
  startWindowsSandbox,
  windowsSandboxCommand,
  windowsSandboxConfiguration,
  windowsSandboxListState,
  type EnvironmentProcessRunner
} from "../src/main/environmentAdapters";
import { EnvironmentOperationError, EnvironmentService } from "../src/main/environmentService";
import { EnvironmentStore } from "../src/main/environmentStore";
import { ENVIRONMENT_OPERATION_FORMAT } from "../src/shared/environments";
import { ENVIRONMENT_RECORD_FORMAT, type ManagedEnvironmentRecord } from "../src/shared/environments";

const roots: string[] = [];
const brainId = "66666666-6666-4666-8666-666666666666";
afterEach(async () => { await Promise.all(roots.splice(0).map(path => rm(path, { recursive: true, force: true }))); });

async function fixture(initialSessions: readonly string[] = []) {
  const root = await mkdtemp(join(tmpdir(), "omni-windows-sandbox-")); roots.push(root);
  const sessions = new Set(initialSessions.map(id => id.toLocaleLowerCase()));
  const calls: Array<{ args: string[] }> = [], resources: number[] = []; let stopExitCode = 0;
  const run: EnvironmentProcessRunner = vi.fn(async (_command, args, _cwd, _timeout, _environment, signal) => {
    calls.push({ args: [...args] });
    if (args[0] === "list") return { exitCode: 0,
      stdout: JSON.stringify({ sandboxes: [...sessions].map(id => ({ id, status: "Running", uptime: 1 })) }), stderr: "", truncated: false };
    if (args[0] === "start") {
      const id = args[args.indexOf("--id") + 1].toLocaleLowerCase(); sessions.add(id);
      return { exitCode: 0, stdout: JSON.stringify({ id }), stderr: "", truncated: false };
    }
    if (args[0] === "stop") {
      if (stopExitCode === 0) sessions.delete(args[args.indexOf("--id") + 1].toLocaleLowerCase());
      return { exitCode: stopExitCode, stdout: stopExitCode === 0 ? "stopped" : "", stderr: stopExitCode === 0 ? "" : "fixture stop failed", truncated: false };
    }
    if (args[0] === "exec" && args.join(" ").includes("__hold")) await new Promise<void>((_resolve, reject) => {
      const abort = (): void => reject(signal?.reason ?? new Error("cancelled"));
      signal?.addEventListener("abort", abort, { once: true }); if (signal?.aborted) abort();
    });
    return { exitCode: 0, stdout: "wsb-host-diagnostic", stderr: "", truncated: false };
  });
  const service = new EnvironmentService({ brainDirectory: async () => root, run, platform: "win32",
    windowsSandboxExecutable: async () => ({
      path: "C:\\Program Files\\WindowsApps\\MicrosoftWindows.WindowsSandbox_fixture\\wsb.exe",
      installationLocation: "C:\\Program Files\\WindowsApps\\MicrosoftWindows.WindowsSandbox_fixture",
      packageFamilyName: "MicrosoftWindows.WindowsSandbox_cw5n1h2txyewy",
      signatureStatus: "Valid"
    }),
    checkResources: async (_brain, bytes) => { resources.push(bytes); }, reserveWrite: async () => undefined,
    deleteVmRuntime: async () => undefined });
  return { calls, resources, root, service, sessions, setStopExitCode(value: number) { stopExitCode = value; } };
}

describe("Windows Sandbox CLI managed adapter", () => {
  it("constructs one hardened disposable configuration and parses only an exact owned UUID", () => {
    const id = "4dc275bc-f021-4d96-858e-d81a4f474947";
    const config = windowsSandboxConfiguration();
    expect(config).toContain("<Networking>Disable</Networking>");
    expect(config).toContain("<ProtectedClient>Enable</ProtectedClient>");
    expect(config).toContain("<ClipboardRedirection>Disable</ClipboardRedirection>");
    expect(config).not.toMatch(/MappedFolder|LogonCommand|Enable<\/Networking>/);
    expect(windowsSandboxListState(JSON.stringify({ sandboxes: [{ id, status: "Running" }] }), id)).toBe("loaded");
    expect(windowsSandboxListState(JSON.stringify({ sandboxes: [{ id, status: "Running" },
      { id: "30c5aa9c-26f5-4e92-9d1b-2a84909e2f31", status: "Stopped" }] }), id)).toBe("loaded");
    expect(windowsSandboxListState(JSON.stringify({ sandboxes: [] }), id)).toBe("missing");
    expect(windowsSandboxListState('{"sandboxes":', id)).toBe("unknown");
    expect(windowsSandboxListState('{"unrelated":[]}', id)).toBe("unknown");
    expect(windowsSandboxListState(JSON.stringify({ id, status: "Stopped" }), id)).toBe("unknown");
    expect(windowsSandboxListState(JSON.stringify([{ correlationId: id, status: "Stopped" }]), id)).toBe("unknown");
    expect(windowsSandboxListState(JSON.stringify({ sandboxes: [] }), id, true)).toBe("unknown");
    expect(windowsSandboxCommand("C:\\Windows\\System32\\cmd.exe", ["/c", "echo a b", "quoted\"value"]))
      .toBe('C:\\Windows\\System32\\cmd.exe /c "echo a b" "quoted\\"value"');
    expect(() => windowsSandboxCommand("cmd.exe", [])).toThrow(/absolute guest executable/i);
  });

  it("ignores the generic executable finder and serializes cross-brain Start ownership", async () => {
    const invalid = await environmentAdapterCapabilities({ platform: "win32", run: async () => ({ exitCode: 0, stdout: '{"sandboxes":[]}', stderr: "", truncated: false }),
      findExecutable: async () => "C:\\Users\\owner\\wsb.exe",
      windowsSandboxExecutable: async () => ({ path: "C:\\Users\\owner\\wsb.exe", installationLocation: "C:\\Users\\owner",
        packageFamilyName: "MicrosoftWindows.WindowsSandbox_cw5n1h2txyewy", signatureStatus: "Valid" }) });
    expect(invalid.find(row => row.id === "windows-sandbox-cli")?.available).toBe(false);

    const root = await mkdtemp(join(tmpdir(), "omni-wsb-global-")); roots.push(root);
    const sessions = new Set<string>(), starts: string[] = [];
    const evidence = async () => ({ path: "C:\\Program Files\\WindowsApps\\MicrosoftWindows.WindowsSandbox_fixture\\wsb.exe",
      installationLocation: "C:\\Program Files\\WindowsApps\\MicrosoftWindows.WindowsSandbox_fixture",
      packageFamilyName: "MicrosoftWindows.WindowsSandbox_cw5n1h2txyewy" as const, signatureStatus: "Valid" as const });
    const run: EnvironmentProcessRunner = async (_command, args) => {
      if (args[0] === "list") return { exitCode: 0, stdout: JSON.stringify({ sandboxes: [...sessions].map(id => ({ id, status: "Running" })) }), stderr: "", truncated: false };
      if (args[0] === "start") { const id = String(args[args.indexOf("--id") + 1] ?? "").toLocaleLowerCase(); starts.push(id); sessions.add(id); }
      return { exitCode: 0, stdout: "", stderr: "", truncated: false };
    };
    const at = new Date().toISOString(), records = ["4dc275bc-f021-4d96-858e-d81a4f474947", "30c5aa9c-26f5-4e92-9d1b-2a84909e2f31"].map((externalId, index): ManagedEnvironmentRecord => ({
      format: ENVIRONMENT_RECORD_FORMAT, version: 1, id: String(index + 1).repeat(32), brainId: `${index + 1}`.repeat(8) + "-6666-4666-8666-666666666666",
      kind: "sandbox", name: `Concurrent ${index}`, adapter: "windows-sandbox-cli", state: "loading", revision: 2, network: "none", createdAt: at, updatedAt: at, externalId
    }));
    const settled = await Promise.allSettled(records.map(record => startWindowsSandbox({ platform: "win32", run, windowsSandboxExecutable: evidence }, record, root, new AbortController().signal)));
    expect(settled.filter(result => result.status === "fulfilled")).toHaveLength(1);
    expect(settled.filter(result => result.status === "rejected")).toHaveLength(1);
    expect(starts).toHaveLength(1);
  });

  it("loads, runs, stops and reloads an exact disposable UUID without claiming guest output", async () => {
    const { calls, resources, root, service, sessions } = await fixture(), signal = new AbortController().signal;
    const created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Windows research" }, "wsb-create", signal) as any;
    expect(created.environment).toMatchObject({ adapter: "windows-sandbox-cli", state: "ready", network: "none" });
    const loaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-load", signal) as any;
    const firstId = loaded.environment.externalId; expect(firstId).toMatch(/^[a-f0-9-]{36}$/); expect(sessions.has(firstId)).toBe(true);
    const start = calls.find(call => call.args[0] === "start")!;
    expect(start.args).toEqual(["start", "--raw", "--id", firstId, "--config", windowsSandboxConfiguration()]);
    expect(start.args.join(" ")).not.toMatch(/share|MappedFolder|LogonCommand/i);
    expect(resources).toContain(2 * 1024 ** 3);
    const used = await service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id,
      program: "C:\\Windows\\System32\\whoami.exe", arguments: [] }, "wsb-run", signal) as any;
    expect(used).toMatchObject({ environment: { state: "loaded" }, operation: { state: "complete",
      process: { exitCode: 0, guestOutput: "guest-output-unavailable" } } });
    expect(used.operation.process.stdout).toBe("wsb-host-diagnostic");
    const store = await EnvironmentStore.open(root, brainId);
    try {
      const missingEvidence = structuredClone(used.operation); delete missingEvidence.process.guestOutput;
      expect(() => store.putOperation(missingEvidence)).toThrow(/guest-output evidence/i);
      expect(() => store.putOperation({ ...created.operation, process: { exitCode: 0, stdout: "", stderr: "", truncated: false,
        guestOutput: "guest-output-unavailable" } })).toThrow(/only an environment Run/i);
    } finally { store.close(); }
    const exec = calls.find(call => call.args[0] === "exec")!;
    expect(exec.args).toEqual(expect.arrayContaining(["--id", firstId, "--run-as", "System"]));
    expect(exec.args).not.toContain("ExistingLogin");
    const stopped = await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: created.environment.id }, "wsb-stop", signal) as any;
    expect(stopped).toMatchObject({ environment: { state: "stopped", externalId: undefined },
      operation: { process: { sandboxDisposition: "stopped" } } }); expect(sessions.size).toBe(0);
    const reloaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-reload", signal) as any;
    expect(reloaded.environment.externalId).not.toBe(firstId);
    await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: created.environment.id }, "wsb-restop", signal);
  });

  it("never adopts or stops an unrelated current-user sandbox", async () => {
    const unrelated = "30c5aa9c-26f5-4e92-9d1b-2a84909e2f31", { calls, service, sessions } = await fixture([unrelated]);
    const signal = new AbortController().signal, created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Blocked" }, "wsb-blocked-create", signal) as any;
    await expect(service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-blocked-load", signal)).rejects.toBeInstanceOf(EnvironmentOperationError);
    expect(sessions).toEqual(new Set([unrelated]));
    expect(calls.filter(call => call.args[0] === "stop").every(call => call.args.includes(unrelated) === false)).toBe(true);
  });

  it("cancels a guest command by stopping only the exact owned disposable sandbox", async () => {
    const { calls, service, sessions } = await fixture(), signal = new AbortController().signal;
    const created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Cancel guest" }, "wsb-cancel-create", signal) as any;
    const loaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-cancel-load", signal) as any;
    const controller = new AbortController(), running = service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id,
      program: "C:\\Windows\\System32\\cmd.exe", arguments: ["/c", "__hold"] }, "wsb-cancel-run", controller.signal);
    await vi.waitFor(() => expect(calls.some(call => call.args[0] === "exec")).toBe(true)); controller.abort(new Error("user cancelled exact run"));
    try { await running; throw new Error("Expected cancellation"); }
    catch (error) { expect(error).toBeInstanceOf(EnvironmentOperationError); expect((error as EnvironmentOperationError).result).toMatchObject({
      environment: { state: "stopped", externalId: undefined }, operation: { state: "cancelled" } });
      expect((error as EnvironmentOperationError).result.environment.warning).toMatch(/stopped.*guest state was deleted/i); }
    expect(sessions.size).toBe(0); expect(calls.some(call => call.args[0] === "stop" && call.args.includes(loaded.environment.externalId))).toBe(true);
  });

  it("requires exact-ID disposal when a guest command may have survived host recovery", async () => {
    const { root, service, sessions } = await fixture(), signal = new AbortController().signal;
    const created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Recovered command" }, "wsb-recovery-create", signal) as any;
    const loaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-recovery-load", signal) as any;
    const store = await EnvironmentStore.open(root, brainId), operationId = "7".repeat(32), at = new Date().toISOString();
    try {
      store.putEnvironment({ ...loaded.environment, state: "busy", revision: loaded.environment.revision + 1, updatedAt: at, lastOperationId: operationId });
      store.putOperation({ format: ENVIRONMENT_OPERATION_FORMAT, version: 1, id: operationId, brainId, environmentId: loaded.environment.id,
        requestId: "wsb-recovered-run", action: "run", state: "running", argumentsSha256: "8".repeat(64), dispatchStarted: true,
        createdAt: at, updatedAt: at });
    } finally { store.close(); }
    await service.forget(brainId);
    const status = await service.execute(brainId, "runtime.sandbox", "status", { environmentId: loaded.environment.id }, "wsb-recovered-status", signal) as any;
    expect(status.environment).toMatchObject({ state: "review-required", orphanedProcessRisk: true, externalId: loaded.environment.externalId });
    await expect(service.execute(brainId, "runtime.sandbox", "run", { environmentId: loaded.environment.id,
      program: "C:\\Windows\\System32\\whoami.exe" }, "wsb-recovered-rerun", signal)).rejects.toThrow(/Load this environment/i);
    const stopped = await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: loaded.environment.id }, "wsb-recovered-stop", signal) as any;
    expect(stopped.environment).toMatchObject({ state: "stopped", externalId: undefined, orphanedProcessRisk: undefined }); expect(sessions.size).toBe(0);
  });

  it("retains disposal-required ownership when explicit Stop of an active Run is unconfirmed", async () => {
    const { calls, service, sessions, setStopExitCode } = await fixture(), signal = new AbortController().signal;
    const created = await service.execute(brainId, "runtime.sandbox", "create", { name: "Failed stop" }, "wsb-failed-stop-create", signal) as any;
    const loaded = await service.execute(brainId, "runtime.sandbox", "load", { environmentId: created.environment.id }, "wsb-failed-stop-load", signal) as any;
    const running = service.execute(brainId, "runtime.sandbox", "run", { environmentId: created.environment.id,
      program: "C:\\Windows\\System32\\cmd.exe", arguments: ["/c", "__hold"] }, "wsb-failed-stop-run", signal);
    await vi.waitFor(() => expect(calls.some(call => call.args[0] === "exec")).toBe(true)); setStopExitCode(9);
    const runningFailure = expect(running).rejects.toBeInstanceOf(EnvironmentOperationError);
    let failed!: EnvironmentOperationError;
    try { await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: created.environment.id }, "wsb-failed-stop", signal); }
    catch (error) { failed = error as EnvironmentOperationError; }
    await runningFailure;
    expect(failed).toBeInstanceOf(EnvironmentOperationError);
    expect(failed.result).toMatchObject({ environment: { state: "review-required", externalId: loaded.environment.externalId, orphanedProcessRisk: true },
      operation: { state: "failed", process: { exitCode: 9, sandboxDisposition: "unconfirmed" } } });
    const status = await service.execute(brainId, "runtime.sandbox", "status", { environmentId: created.environment.id }, "wsb-failed-stop-status", signal) as any;
    expect(status.environment).toMatchObject({ state: "review-required", orphanedProcessRisk: true }); expect(sessions.size).toBe(1);
    setStopExitCode(0); const stopped = await service.execute(brainId, "runtime.sandbox", "stop", { environmentId: created.environment.id }, "wsb-failed-stop-retry", signal) as any;
    expect(stopped.environment).toMatchObject({ state: "stopped", externalId: undefined, orphanedProcessRisk: undefined });
  });
});
