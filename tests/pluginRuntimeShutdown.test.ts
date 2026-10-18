/** Tiny owned bootstrap + fake host; no Electron, neural worker or real effect. */
import { randomUUID } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, expect, it, vi } from "vitest";
import { pluginRuntimeScript } from "../src/main/pluginRuntimeBootstrap";
import { PluginRuntimeService } from "../src/main/pluginRuntimeService";
import { PluginRuntimeStore } from "../src/main/pluginRuntimeStore";
import type { PluginService } from "../src/main/pluginService";
import type { PluginRuntimeDelivery, PluginRuntimeLaunch, PluginRuntimeModuleDefinition, PluginRuntimeProcessFactory, PluginRuntimeShutdownDiagnostic } from "../src/shared/pluginRuntime";
import type { ToolExecutionResult } from "../src/shared/types";

const roots: string[] = [], services: PluginRuntimeService[] = [];
afterEach(async () => {
  vi.restoreAllMocks();
  if (vi.isFakeTimers()) { await vi.runOnlyPendingTimersAsync(); vi.useRealTimers(); }
  for (const service of services.splice(0)) await service.dispose();
  for (const path of roots.splice(0)) await rm(path, { recursive: true, force: true });
});
function deferred() { let resolve!: () => void; return { promise: new Promise<void>(done => { resolve = done; }), resolve }; }
async function fixture(script: string, shutdownMs = 30, pendingEffect = false) {
  const directory = await mkdtemp(join(tmpdir(), "omni-shutdown-owned-")); roots.push(directory);
  const hash = "a".repeat(64), definition: PluginRuntimeModuleDefinition = { id: "cleanup", title: "Owned cleanup", description: "Test-owned local callbacks only", script,
    events: [], ui: { kinds: [], slots: [] }, dynamicTools: false };
  const plugin = { list: async () => [{ pluginId: "owned.cleanup", activeContentSha256: hash, revision: 1, enabled: true }],
    instance: async () => ({ contentSha256: hash, revision: 2, enabled: true, capabilities: pendingEffect ? [{ toolId: "system.files", actions: ["read"], enabled: true }] : [] }),
    package: async () => ({ manifest: { runtimeModules: [definition] } }), onEvent: () => () => undefined } as unknown as PluginService;
  let launch!: PluginRuntimeLaunch, callbacks!: Parameters<PluginRuntimeProcessFactory["launch"]>[1], delivery!: (message: PluginRuntimeDelivery) => void;
  const events: string[] = [], sent: PluginRuntimeDelivery[] = [], cleanup = deferred(), entered = deferred();
  const admitted = deferred(); let endEffect!: (value: ToolExecutionResult) => void;
  const effect = new Promise<ToolExecutionResult>(resolve => { endEffect = resolve; });
  const execute = vi.fn(async () => { if (!pendingEffect) throw new Error("No host effect is authorized by this fixture"); admitted.resolve(); return effect; });
  const cancel = vi.fn(async () => { events.push("cancel-started"); endEffect({ id: randomUUID(), toolId: "system.files", action: "read", state: "failed", startedAt: new Date().toISOString(), error: "Test-owned pending effect cancelled; no OS action" }); return 0; });
  const terminate = vi.fn(async () => { events.push("terminate"); callbacks.gone("Test-owned renderer stopped"); }), release = vi.fn(async () => { events.push("release"); });
  const retire = vi.fn(async () => undefined);
  const service = new PluginRuntimeService(plugin, {
    brainRoot: () => directory, moduleReadyTimeoutMs: 100, moduleShutdownTimeoutMs: shutdownMs,
    admit: async () => ({ maxMessageBytes: 8192, check: async () => undefined, release }), admitRead: async () => ({ release: () => undefined }),
    tools: { execute, cancelAndWait: cancel, invalidateApproval: () => true }, publishTools: async () => undefined, retireTools: retire,
    processFactory: { launch: async (input, observed) => {
      launch = input; callbacks = observed; observed.signal.addEventListener("abort", () => events.push("abort"));
      setImmediate(() => {
        const transport = { subscribe: (listener: typeof delivery) => { delivery = listener; }, request: (message: unknown) => observed.receive(23, input.url, message) };
        // This executes only the fixed test-owned local callback body; no package
        // or isolated renderer is launched and this is not a security boundary.
        new Function("globalThis", "crypto", "setTimeout", pluginRuntimeScript(input, definition))({ __omniRuntimeTransport: transport,
          cleanup: cleanup.promise, entered: () => entered.resolve(), observe: (event: string) => events.push(event) }, { randomUUID }, setTimeout);
      });
      return { senderId: 23, processId: 4321, terminate, send: message => {
        sent.push(message); if (message.kind === "deactivate") { events.push(`deactivate-before-abort:${!observed.signal.aborted}`); }
        delivery(message);
      } };
    } }
  }); services.push(service);
  const activate = () => service.activate({ brainId: "owner", pluginId: "owned.cleanup", contentSha256: hash, moduleId: "cleanup", expectedCatalogRevision: 1, expectedInstanceRevision: 2, requestId: randomUUID() });
  const request = (operation: string, args: Record<string, unknown>, id: string = randomUUID(), sender = 23) => callbacks.receive(sender, launch.url,
    { format: "omni-plugin-runtime-rpc", version: 1, sessionId: launch.session.sessionId, nonce: launch.nonce, id, operation, arguments: args });
  const diagnostics = async () => { const store = await PluginRuntimeStore.open(directory, "owner", { readOnly: true });
    try { return store!.history(undefined, 65536).events.filter(row => row.kind === "runtime-shutdown").map(row => row.value as PluginRuntimeShutdownDiagnostic); } finally { store!.close(); } };
  return { service, activate, launch: () => launch, callbacks: () => callbacks, request, events, sent, execute, cancel, retire, terminate, release, cleanup, entered, admitted, diagnostics };
}

it("accepts only the matching bootstrap cleanup ACK before abort while refusing all new effects", async () => {
  const f = await fixture(`const sdk=globalThis.omniRuntime;
sdk.onDeactivate(async()=>{globalThis.observe("cleanup-enter");globalThis.entered();try{await sdk.call("system.files","read",{});}catch(error){globalThis.observe(error.message);}await globalThis.cleanup;globalThis.observe("cleanup-finished");});
sdk.onDeactivate(()=>{throw new Error("owned local cleanup error");});`, 100, true);
  const session = await f.activate(), prior = f.request("call", { toolId: "system.files", action: "read", arguments: {} }, "prior-pending").catch(() => undefined);
  await f.admitted.promise;
  // This test inspects a held cleanup before the real 100ms deadline. Disk
  // reads on a Windows runner are allowed to take longer than that deadline;
  // control only its clock instead of raising the product timeout.
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const stopping = f.service.stop("owner", session.sessionId); await f.entered.promise;
  expect(f.callbacks().signal.aborted).toBe(false); expect((await f.service.list("owner")).sessions[0]!.state).toBe("stopping");
  const shutdown = f.sent.find(message => message.kind === "deactivate")!;
  expect(shutdown.shutdown).toMatchObject({ format: "omni-plugin-runtime-shutdown", version: 1, sessionId: session.sessionId });
  await expect(f.request("shutdown-ack", { shutdownNonce: "wrong", callbacks: 0, completed: 0, failed: 0, errors: [] }, shutdown.id)).rejects.toThrow(/main-owned challenge/);
  await expect(f.request("shutdown-ack", { shutdownNonce: shutdown.shutdown!.nonce, callbacks: 0, completed: 0, failed: 0, errors: [] }, shutdown.id, 99)).rejects.toThrow(/foreign renderer/);
  expect((await f.request("call", { toolId: "system.files", action: "read", arguments: {} })).state).toBe("failed"); expect(f.execute).toHaveBeenCalledOnce();
  expect(f.cancel).toHaveBeenCalledOnce(); expect(f.events.indexOf("cancel-started")).toBeLessThan(f.events.indexOf("cleanup-enter"));
  f.cleanup.resolve(); expect((await stopping).state).toBe("stopped");
  await prior;
  const diagnostic = (await f.diagnostics())[0]!;
  expect(diagnostic).toMatchObject({ state: "acknowledged", acknowledgement: { callbacks: 2, completed: 1, failed: 1, errors: [{ callback: 1, message: "owned local cleanup error" }] } });
  expect(JSON.stringify(diagnostic)).not.toContain(shutdown.shutdown!.nonce); expect(JSON.stringify(diagnostic)).not.toContain(f.launch().nonce);
  expect(f.events).toContain("Runtime is closed"); expect(f.events.indexOf("cleanup-finished")).toBeLessThan(f.events.indexOf("abort"));
  expect(f.events.indexOf("deactivate-before-abort:true")).toBeLessThan(f.events.indexOf("cleanup-enter"));
  expect(f.terminate).toHaveBeenCalledOnce(); expect(f.release).toHaveBeenCalledOnce(); expect(f.callbacks().signal.aborted).toBe(true);
});

it("keeps revoked closing-session diagnostics and owner disposal pending until durable custody finishes", async () => {
  const f = await fixture(`globalThis.omniRuntime.onDeactivate(()=>{});`), session = await f.activate();
  const entered = deferred(), flush = deferred(), original = PluginRuntimeStore.prototype.record;
  const write = vi.spyOn(PluginRuntimeStore.prototype, "record").mockImplementation(async function(this: PluginRuntimeStore, kind, value) {
    if (kind === "runtime-shutdown") { entered.resolve(); await flush.promise; }
    return original.call(this, kind, value);
  });
  const stopping = f.service.stop("owner", session.sessionId);
  try {
    await entered.promise;
    expect(f.callbacks().signal.aborted).toBe(true);
    expect((await f.service.list("owner")).sessions[0]!.state).toBe("stopping");
    await expect(f.request("call", { toolId: "system.files", action: "read", arguments: {} })).rejects.toThrow(/stale or foreign/);
    let ownerDone = false, disposed = false;
    const owner = f.service.stopOwner("owner").then(() => { ownerDone = true; });
    const disposal = f.service.dispose().then(() => { disposed = true; });
    await new Promise(resolve => setImmediate(resolve));
    expect(ownerDone).toBe(false); expect(disposed).toBe(false);
    flush.resolve();
    expect((await stopping).state).toBe("stopped"); await owner; await disposal;
    expect(ownerDone).toBe(true); expect(disposed).toBe(true);
    expect(f.terminate).toHaveBeenCalledOnce(); expect(f.release).toHaveBeenCalledOnce();
  } finally { flush.resolve(); await stopping.catch(() => undefined); write.mockRestore(); }
});

it("bounds missing cleanup ACK, persists truthful timeout and still terminates/releases on diagnostic write refusal", async () => {
  const script = `globalThis.omniRuntime.onDeactivate(()=>new Promise(()=>{}));`;
  const timed = await fixture(script, 15), session = await timed.activate();
  const started = performance.now(); expect((await timed.service.stop("owner", session.sessionId)).state).toBe("stopped");
  const diagnostic = (await timed.diagnostics())[0]!;
  expect(diagnostic).toMatchObject({ state: "timeout", deadlineMs: 15 }); expect(diagnostic).not.toHaveProperty("acknowledgement");
  expect(performance.now() - started).toBeLessThan(1000); expect(timed.terminate).toHaveBeenCalledOnce(); expect(timed.release).toHaveBeenCalledOnce();
  // Separate tiny failure branch: the attempted timeout diagnostic is NOT
  // claimed committed when disk admission refuses its write.
  const refused = await fixture(script, 15), second = await refused.activate();
  const write = vi.spyOn(PluginRuntimeStore.prototype, "record").mockRejectedValueOnce(new Error("owned diagnostic disk reserve refusal"));
  await expect(refused.service.stop("owner", second.sessionId)).rejects.toThrow("owned diagnostic disk reserve refusal");
  expect(write.mock.calls[0]![0]).toBe("runtime-shutdown"); expect(write.mock.calls[0]![1]).toMatchObject({ state: "timeout" }); write.mockRestore();
  expect(await refused.diagnostics()).toEqual([]); expect(refused.callbacks().signal.aborted).toBe(true);
  expect(refused.terminate).toHaveBeenCalledOnce(); expect(refused.release).toHaveBeenCalledOnce();
});

it("continues cancellation retirement and diagnostic cleanup when a stopping observer throws", async () => {
  const f = await fixture(`globalThis.omniRuntime.onDeactivate(()=>{});`, 100, true), session = await f.activate();
  const prior = f.request("call", { toolId: "system.files", action: "read", arguments: {} }, "prior-observer-test").catch(() => undefined);
  await f.admitted.promise;
  const unsubscribe = f.service.onEvent(event => { if (event.session?.state === "stopping") throw new Error("owned stopping observer failure"); });
  try { await expect(f.service.stop("owner", session.sessionId)).rejects.toThrow("owned stopping observer failure"); }
  finally { unsubscribe(); }
  await prior;
  expect(f.cancel).toHaveBeenCalledOnce(); expect(f.retire).toHaveBeenCalledOnce(); expect(f.terminate).toHaveBeenCalledOnce(); expect(f.release).toHaveBeenCalledOnce();
  expect((await f.diagnostics())[0]).toMatchObject({ state: "acknowledged", acknowledgement: { callbacks: 1, completed: 1, failed: 0 } });
  expect(f.callbacks().signal.aborted).toBe(true);
});
