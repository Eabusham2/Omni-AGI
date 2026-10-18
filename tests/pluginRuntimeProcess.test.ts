/** Injected Electron lifecycle/security fixtures only; no renderer or module executes. */
import { randomUUID } from "node:crypto";
import { EventEmitter } from "node:events";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ElectronPluginRuntimeProcessFactory } from "../src/main/pluginRuntimeProcess";
import type { ElectronPluginRuntimeProcessOptions } from "../src/main/pluginRuntimeProcess";
import type { PluginRuntimeLaunch, PluginRuntimeProcess, PluginRuntimeRpcReply } from "../src/shared/pluginRuntime";
import { PLUGIN_RUNTIME_DELIVERY_CHANNEL, PLUGIN_RUNTIME_SCHEME } from "../src/shared/pluginRuntime";

type RequestEvent = Parameters<ElectronPluginRuntimeProcessFactory["request"]>[0];
type Rpc = { id: string; [key: string]: unknown };
const deferred = <T>() => { let resolve!: (value: T) => void, reject!: (error: Error) => void;
  return { promise: new Promise<T>((yes, no) => { resolve = yes; reject = no; }), resolve, reject }; };
const reply = (id: string): PluginRuntimeRpcReply => ({ format: "omni-plugin-runtime-reply", version: 1, id, state: "complete" });

class FakeSession extends EventEmitter {
  document?: (request: Request) => Promise<Response>;
  network?: (details: { url: string; resourceType: string; method: string; webContentsId?: number }, callback: (result: { cancel?: boolean }) => void) => void;
  protocol = { handle: vi.fn((_scheme: string, handler: (request: Request) => Promise<Response>) => { this.document = handler; }),
    unhandle: vi.fn(() => { this.document = undefined; }) };
  webRequest = { onBeforeRequest: vi.fn((_filter: unknown, listener: NonNullable<FakeSession["network"]>) => { this.network = listener; }) };
  requestPermission?: (...args: any[]) => void;
  checkPermission?: (...args: any[]) => boolean;
  devicePermission?: (...args: any[]) => boolean;
  displayPermission?: (...args: any[]) => void;
  setPermissionRequestHandler = vi.fn((handler: (...args: any[]) => void) => { this.requestPermission = handler; });
  setPermissionCheckHandler = vi.fn((handler: (...args: any[]) => boolean) => { this.checkPermission = handler; });
  setDevicePermissionHandler = vi.fn((handler: (...args: any[]) => boolean) => { this.devicePermission = handler; });
  setDisplayMediaRequestHandler = vi.fn((handler: (...args: any[]) => void) => { this.displayPermission = handler; });
}
class FakeContents extends EventEmitter {
  url = ""; mainFrame = { url: "", parent: null }; destroyed = false;
  openWindow?: () => { action: string };
  send = vi.fn(); stop = vi.fn(); setAudioMuted = vi.fn();
  setupThrows = false;
  constructor(readonly id: number, public pid: number, private readonly crash: () => void = () => undefined) { super(); }
  getURL(): string { return this.url; }
  isDestroyed(): boolean { return this.destroyed; }
  getOSProcessId(): number { return this.pid; }
  setWindowOpenHandler = vi.fn((handler: () => { action: string }) => { if (this.setupThrows) throw new Error("injected setup failure"); this.openWindow = handler; });
  forcefullyCrashRenderer = vi.fn(() => { this.crash(); this.emit("render-process-gone", {}, { reason: "killed" }); });
}
class FakeWindow extends EventEmitter {
  destroyed = false;
  constructor(readonly webContents: FakeContents, private readonly onClose: () => void, private readonly onLoad?: (url: string) => Promise<void>) { super(); }
  loadURL(url: string): Promise<void> { this.webContents.url = url; this.webContents.mainFrame.url = url; return this.onLoad?.(url) ?? Promise.resolve(); }
  isDestroyed(): boolean { return this.destroyed; }
  destroy = vi.fn(() => { if (this.destroyed) return; this.destroyed = true; this.webContents.destroyed = true; this.onClose(); this.webContents.emit("destroyed"); this.emit("closed"); });
}

const processes: PluginRuntimeProcess[] = [];
afterEach(async () => { for (const process of processes.splice(0)) await process.terminate().catch(() => undefined); vi.useRealTimers(); });
function fixture(config: { load?: (window: FakeWindow, url: string) => Promise<void>; setupThrows?: boolean; startupMs?: number; stopMs?: number; keepMetrics?: boolean; pid?: number } = {}) {
  const windows: FakeWindow[] = [], sessions: FakeSession[] = [], partitions: string[] = [], createOptions: unknown[] = [];
  const studio = new FakeContents(1, 400); studio.url = studio.mainFrame.url = "file:///studio/index.html";
  const peers: FakeContents[] = [studio], metrics = new Map<number, { pid: number; type: "Tab" | "Browser" | "Utility"; creationTime: number; sandboxed: boolean }>([[400, { pid: 400, type: "Tab", creationTime: 100, sandboxed: true }]]);
  const document = vi.fn(async (_url: string) => new Response("<!doctype html><script>/* inert fake document */</script>", {
    headers: { "Content-Type": "text/html; charset=utf-8", "Content-Security-Policy": "default-src 'none'; script-src 'sha256-fixture-only'; connect-src 'none'" } }));
  const options: ElectronPluginRuntimeProcessOptions = {
    runtimePreloadPath: "/owned/out/preload/pluginRuntime.js", mainProcessId: 123,
    startupTimeoutMs: config.startupMs ?? 100, stopTimeoutMs: config.stopMs ?? 100,
    createWindow: input => {
      createOptions.push(input); const index = windows.length, pid = config.pid ?? 700 + index;
      const closeMetric = () => { if (!config.keepMetrics && !peers.some(other => !other.destroyed && other.pid === pid && other !== contents)) metrics.delete(pid); };
      const contents = new FakeContents(10 + index, pid, closeMetric); contents.setupThrows = !!config.setupThrows;
      let window: FakeWindow;
      window = new FakeWindow(contents, closeMetric, (url): Promise<void> => config.load?.(window, url) ?? Promise.resolve());
      windows.push(window); peers.push(contents); if (!metrics.has(pid)) metrics.set(pid, { pid, type: "Tab", creationTime: 1000 + index, sandboxed: true });
      return window as unknown as ReturnType<ElectronPluginRuntimeProcessOptions["createWindow"]>;
    },
    sessionFromPartition: (partition, options) => { expect(options).toEqual({ cache: false }); partitions.push(partition); const session = new FakeSession(); sessions.push(session);
      return session as unknown as ReturnType<ElectronPluginRuntimeProcessOptions["sessionFromPartition"]>; },
    allWebContents: () => peers, appMetrics: () => [...metrics.values()], document
  };
  const factory = new ElectronPluginRuntimeProcessFactory(options), controller = new AbortController();
  const input: PluginRuntimeLaunch = { session: { format: "omni-plugin-runtime-session", version: 1, brainId: "owned-brain", pluginId: "owned-plugin", contentSha256: "a".repeat(64),
    moduleId: "module", sessionId: randomUUID(), catalogRevision: 1, instanceRevision: 1, state: "starting", startedAt: "2026-10-14T03:00:00.000Z" },
    url: "", nonce: "b".repeat(64), maxMessageBytes: 4096 };
  input.url = `${PLUGIN_RUNTIME_SCHEME}://${input.session.sessionId}/index.html`;
  const receive = vi.fn(async (_sender: number, _url: string, message: unknown) => reply((message as Rpc).id)), gone = vi.fn();
  const callbacks = { signal: controller.signal, receive, gone };
  const launch = async (value = input) => { const process = await factory.launch(value, callbacks); processes.push(process); return process; };
  const event = (contents = windows[0]!.webContents, frame: unknown = contents.mainFrame): RequestEvent => ({ sender: contents, senderFrame: frame }) as unknown as RequestEvent;
  const rpc = (values: Record<string, unknown> = {}) => ({ format: "omni-plugin-runtime-rpc", version: 1, sessionId: input.session.sessionId, nonce: input.nonce, id: randomUUID(), operation: "ready", arguments: {}, ...values });
  return { factory, options, launch, input, controller, callbacks, receive, gone, windows, sessions, partitions, createOptions, studio, peers, metrics, document, event, rpc };
}

describe("dedicated Electron runtime process factory (no real Electron)", () => {
  it("requires exact typed launch bindings and a current signal before creating a window", async () => {
    const f = fixture(); f.controller.abort(new Error("revoked")); await expect(f.launch()).rejects.toThrow("revoked"); expect(f.windows).toHaveLength(0);
    for (const change of [{ nonce: "b".repeat(64) + "\n" }, { maxMessageBytes: 4095 }, { url: "https://foreign.invalid" }, { session: { ...f.input.session, sessionId: "foreign\n" } }]) {
      const value = fixture(); await expect(value.launch({ ...value.input, ...change })).rejects.toThrow(/launch binding/); expect(value.windows).toHaveLength(0);
    }
    expect(() => new ElectronPluginRuntimeProcessFactory({ ...f.options, runtimePreloadPath: "relative.js" })).toThrow(/absolute/);
  });
  it("creates hidden sandbox/context-isolated renderers in distinct nonpersistent private sessions", async () => {
    const f = fixture(); const first = await f.launch();
    const next = { ...f.input, session: { ...f.input.session, sessionId: randomUUID() } }; next.url = `${PLUGIN_RUNTIME_SCHEME}://${next.session.sessionId}/index.html`;
    const second = await f.launch(next);
    expect(first.processId).not.toBe(second.processId); expect(new Set(f.partitions).size).toBe(2); expect(f.partitions.every(value => !value.startsWith("persist:"))).toBe(true);
    expect(f.createOptions[0]).toMatchObject({ show: false, skipTaskbar: true, focusable: false, webPreferences: { preload: "/owned/out/preload/pluginRuntime.js", sandbox: true,
      contextIsolation: true, nodeIntegration: false, nodeIntegrationInWorker: false, nodeIntegrationInSubFrames: false, webviewTag: false, devTools: false, webSecurity: true,
      additionalArguments: ["--omni-plugin-runtime-max-frame=4096"] } });
    expect(f.sessions[0]!.protocol.handle).toHaveBeenCalledWith(PLUGIN_RUNTIME_SCHEME, expect.any(Function));
    await expect(f.launch()).rejects.toThrow(/already-owned/);
  });
  it("serves only the exact private GET document with current service security headers", async () => {
    const f = fixture(); await f.launch(); const serve = f.sessions[0]!.document!;
    expect((await serve(new Request(f.input.url))).status).toBe(200); expect(f.document).toHaveBeenCalledWith(f.input.url);
    for (const url of [f.input.url + "?other=1", f.input.url + "#other", f.input.url.replace("index.html", "other.html"), "https://foreign.invalid/"])
      expect((await serve(new Request(url))).status).toBe(410);
    expect((await serve(new Request(f.input.url, { method: "POST" }))).status).toBe(410); expect(f.document).toHaveBeenCalledTimes(1);
    f.document.mockResolvedValueOnce(new Response("missing CSP", { headers: { "Content-Type": "text/html" } }));
    await expect(serve(new Request(f.input.url))).rejects.toThrow(/security headers/);
  });
  it("denies network/subframes/navigation/new windows/permissions/devices/downloads", async () => {
    const f = fixture(); await f.launch(); const session = f.sessions[0]!, contents = f.windows[0]!.webContents;
    const allowed = { url: f.input.url, resourceType: "mainFrame", method: "GET", webContentsId: contents.id };
    const check = (change: object) => { let result: unknown; session.network!({ ...allowed, ...change }, value => { result = value; }); return result; };
    expect(check({})).toEqual({ cancel: false });
    for (const change of [{ url: "https://network.invalid" }, { url: "file:///private.txt" }, { url: "ws://network.invalid" }, { resourceType: "subFrame" }, { resourceType: "xhr" }, { method: "POST" }, { webContentsId: f.studio.id }, { webContentsId: undefined }]) expect(check(change)).toEqual({ cancel: true });
    const permission = vi.fn(), display = vi.fn(); session.requestPermission!({}, "media", permission); session.displayPermission!({}, display);
    expect(permission).toHaveBeenCalledWith(false); expect(display).toHaveBeenCalledWith({}); expect(session.checkPermission!()).toBe(false); expect(session.devicePermission!()).toBe(false);
    const preventDefault = vi.fn(), cancel = vi.fn(); session.emit("will-download", { preventDefault }, { cancel }); expect(preventDefault).toHaveBeenCalledOnce(); expect(cancel).toHaveBeenCalledOnce();
    for (const name of ["will-navigate", "will-frame-navigate", "will-redirect", "will-attach-webview"]) contents.emit(name, { preventDefault });
    expect(preventDefault).toHaveBeenCalledTimes(5); expect(contents.openWindow!()).toEqual({ action: "deny" });
  });
  it("accepts only actual owned sender/main-frame/URL/nonce/size and returns a matching bounded reply", async () => {
    const f = fixture(); const process = await f.launch(); const message = f.rpc();
    expect(await f.factory.request(f.event(), message)).toEqual(reply(message.id)); expect(f.receive).toHaveBeenCalledWith(process.senderId, f.input.url, message);
    const contents = f.windows[0]!.webContents, before = f.receive.mock.calls.length;
    await expect(f.factory.request(f.event(f.studio), message)).rejects.toThrow(/Foreign/);
    const spoof = { ...contents, id: contents.id } as unknown as FakeContents;
    await expect(f.factory.request(f.event(spoof), message)).rejects.toThrow(/Foreign/);
    await expect(f.factory.request(f.event(contents, { url: f.input.url }), message)).rejects.toThrow(/non-main frame/);
    await expect(f.factory.request(f.event(), f.rpc({ nonce: "c".repeat(64) }))).rejects.toThrow(/binding/);
    await expect(f.factory.request(f.event(), f.rpc({ arguments: { large: "x".repeat(4096) } }))).rejects.toThrow(/admitted frame/);
    contents.mainFrame.url += "#changed"; await expect(f.factory.request(f.event(), message)).rejects.toThrow(/foreign document/);
    expect(f.receive).toHaveBeenCalledTimes(before);
  });
  it("uses only the fixed delivery channel and rejects stale/oversized deliveries and replies", async () => {
    const f = fixture(); const process = await f.launch(), message = { kind: "event" as const, id: randomUUID(), topic: "owned", data: { value: 1 } };
    process.send(message); expect(f.windows[0]!.webContents.send).toHaveBeenCalledWith(PLUGIN_RUNTIME_DELIVERY_CHANNEL, message);
    expect(() => process.send({ ...message, data: "x".repeat(4096) })).toThrow(/admitted frame/);
    f.receive.mockImplementationOnce(async (_sender, _url, request) => ({ ...reply((request as Rpc).id), value: "x".repeat(4096) }));
    await expect(f.factory.request(f.event(), f.rpc())).rejects.toThrow(/admitted frame/);
    f.receive.mockResolvedValueOnce(reply("another-request")); await expect(f.factory.request(f.event(), f.rpc())).rejects.toThrow(/current request/);
    await process.terminate(); expect(() => process.send(message)).toThrow(/retired/); await expect(f.factory.request(f.event(), f.rpc())).rejects.toThrow(/Foreign/);
  });
  it("does not deliver an async reply after its actual document owner retires", async () => {
    const f = fixture(); const process = await f.launch(), gate = deferred<PluginRuntimeRpcReply>(), message = f.rpc();
    f.receive.mockImplementationOnce(() => gate.promise); const pending = f.factory.request(f.event(), message); await process.terminate(); gate.resolve(reply(message.id));
    await expect(pending).rejects.toThrow(/retired/); expect(f.gone).toHaveBeenCalledOnce();
  });
  it("verifies exclusive PID plus creation time before forceful stop and removes all authority once", async () => {
    const f = fixture(); const process = await f.launch(), contents = f.windows[0]!.webContents, session = f.sessions[0]!;
    expect(f.factory.owns(process.senderId)).toBe(true); const first = process.terminate(), second = process.terminate(); expect(second).toBe(first); await first;
    expect(contents.forcefullyCrashRenderer).toHaveBeenCalledOnce(); expect(f.windows[0]!.destroy).toHaveBeenCalledOnce(); expect(f.studio.destroyed).toBe(false);
    expect(f.factory.owns(process.senderId)).toBe(false); expect(session.protocol.unhandle).toHaveBeenCalledWith(PLUGIN_RUNTIME_SCHEME); expect(f.gone).toHaveBeenCalledOnce();
    expect(contents.listenerCount("will-navigate")).toBe(0); let network: unknown; session.network!({ url: f.input.url, resourceType: "mainFrame", method: "GET", webContentsId: contents.id }, value => { network = value; });
    expect(network).toEqual({ cancel: true });
  });
  it("refuses shared Studio/main/nonrenderer identities without crashing any foreign process", async () => {
    for (const pid of [400, 123]) {
      const f = fixture({ pid }); await expect(f.launch()).rejects.toThrow(/exclusive|shared/);
      expect(f.windows[0]!.webContents.forcefullyCrashRenderer).not.toHaveBeenCalled(); expect(f.windows[0]!.destroyed).toBe(true); expect(f.studio.destroyed).toBe(false);
    }
    const f = fixture(); f.options.appMetrics = () => [{ pid: 700, type: "Utility", creationTime: 1000 }]; await expect(f.launch()).rejects.toThrow(/protected or unverifiable/);
    expect(f.windows[0]!.webContents.forcefullyCrashRenderer).not.toHaveBeenCalled();
  });
  it("does not crash a renderer shared or PID-reused after activation", async () => {
    for (const changed of ["shared", "reused"] as const) {
      const f = fixture(); const process = await f.launch(), contents = f.windows[0]!.webContents;
      if (changed === "shared") f.peers.push(new FakeContents(99, process.processId)); else f.metrics.get(process.processId)!.creationTime += 1;
      await process.terminate(); expect(contents.forcefullyCrashRenderer).not.toHaveBeenCalled(); expect(f.windows[0]!.destroyed).toBe(true); expect(f.studio.destroyed).toBe(false);
    }
  });
  it("aborts a pending startup without waiting for loadURL or leaving a child", async () => {
    const gate = deferred<void>(), f = fixture({ load: () => gate.promise }); const pending = f.launch();
    f.controller.abort(new Error("resource lease revoked")); await expect(pending).rejects.toThrow(/aborted|revoked/);
    expect(f.windows[0]!.destroyed).toBe(true); expect(f.windows[0]!.webContents.forcefullyCrashRenderer).toHaveBeenCalledOnce(); expect(f.gone).toHaveBeenCalledOnce();
    expect(f.factory.owns(f.windows[0]!.webContents.id)).toBe(false); gate.resolve();
  });
  it("bounds hung startup and destroys a window if security setup throws", async () => {
    vi.useFakeTimers(); const f = fixture({ startupMs: 20, load: () => new Promise(() => undefined) });
    const failed = expect(f.launch()).rejects.toThrow(/bounded deadline/); await vi.advanceTimersByTimeAsync(25); await failed;
    expect(f.windows[0]!.destroyed).toBe(true); expect(f.gone).toHaveBeenCalledOnce(); vi.useRealTimers();
    const setup = fixture({ setupThrows: true }); await expect(setup.launch()).rejects.toThrow(/injected setup failure/); expect(setup.windows[0]!.destroyed).toBe(true);
  });
  it("reports unverified OS release at the bounded deadline instead of silently releasing a child", async () => {
    vi.useFakeTimers(); const f = fixture({ keepMetrics: true, stopMs: 20 }); const process = await f.launch();
    const failed = expect(process.terminate()).rejects.toThrow(/release could not be verified/); await vi.advanceTimersByTimeAsync(25); await failed;
    expect(f.gone).toHaveBeenCalledOnce(); expect(f.gone.mock.calls[0]![0]).toMatch(/release unverified/); expect(f.studio.destroyed).toBe(false);
  });
  it("reports the actual rejected startup reason only after its owned window is destroyed", async () => {
    const f = fixture({ load: async () => { throw new Error("owned document rejected"); } });
    const observed: Array<{ destroyed: boolean; reason: unknown }> = [];
    f.gone.mockImplementation(reason => { observed.push({ destroyed: f.windows[0]!.destroyed, reason }); });
    await expect(f.launch()).rejects.toThrow("owned document rejected"); expect(f.gone).toHaveBeenCalledOnce();
    expect(observed).toEqual([{ destroyed: true, reason: "Runtime startup failed: owned document rejected" }]);
  });
  it("stops safely when BrowserWindow webContents getter throws after destroy", async () => {
    const f = fixture(), create = f.options.createWindow;
    let contents!: FakeContents, lateGetterReads = 0;
    f.options.createWindow = options => {
      const window = create(options) as unknown as FakeWindow;
      contents = window.webContents;
      Object.defineProperty(window, "webContents", { configurable: true, get: () => {
        if (window.destroyed) { lateGetterReads++; throw new TypeError("Object has been destroyed"); }
        return contents;
      } });
      // Model Electron destruction independently of its now-invalid getter.
      window.destroy = vi.fn(() => {
        if (window.destroyed) return;
        window.destroyed = true; contents.destroyed = true; f.metrics.delete(contents.pid);
        contents.emit("destroyed"); window.emit("closed");
      });
      return window as unknown as ReturnType<ElectronPluginRuntimeProcessOptions["createWindow"]>;
    };
    const process = await f.launch();
    await expect(process.terminate()).resolves.toBeUndefined();
    await expect(process.terminate()).resolves.toBeUndefined();
    expect(lateGetterReads).toBe(0); expect(contents.isDestroyed()).toBe(true);
    expect(contents.forcefullyCrashRenderer).toHaveBeenCalledOnce();
    expect(f.factory.owns(process.senderId)).toBe(false); expect(f.metrics.has(process.processId)).toBe(false);
    expect(f.gone).toHaveBeenCalledOnce(); expect(f.studio.destroyed).toBe(false);
    expect(f.sessions[0]!.protocol.unhandle).toHaveBeenCalledWith(PLUGIN_RUNTIME_SCHEME);
  });
  it("discards a late private document after current abort and cleans foreign in-page navigation", async () => {
    const f = fixture(); await f.launch(); const gate = deferred<Response>(); f.document.mockImplementationOnce(() => gate.promise);
    const serve = f.sessions[0]!.document!, pending = serve(new Request(f.input.url)); f.controller.abort(); gate.resolve(new Response("late", { headers: { "Content-Type": "text/html", "Content-Security-Policy": "default-src 'none'" } }));
    expect((await pending).status).toBe(410);
    const other = fixture(); const process = await other.launch(); other.windows[0]!.webContents.emit("did-navigate-in-page", {}, other.input.url + "#foreign");
    await process.terminate(); expect(other.factory.owns(process.senderId)).toBe(false); expect(other.gone).toHaveBeenCalledOnce();
  });
});
