import { randomUUID } from "node:crypto";
import { isAbsolute } from "node:path";
import type { EventEmitter } from "node:events";
import type { BrowserWindow, BrowserWindowConstructorOptions, IpcMainInvokeEvent, ProcessMetric, Session, WebContents } from "electron";
import type { PluginRuntimeDelivery, PluginRuntimeLaunch, PluginRuntimeProcess, PluginRuntimeProcessFactory, PluginRuntimeRpcReply } from "../shared/pluginRuntime";
import { PLUGIN_RUNTIME_DELIVERY_CHANNEL, PLUGIN_RUNTIME_SCHEME } from "../shared/pluginRuntime";

type RuntimeWindow = Pick<BrowserWindow, "webContents" | "loadURL" | "isDestroyed" | "destroy" | "on" | "removeListener">;
type RuntimeSession = Pick<Session, "protocol" | "webRequest" | "setPermissionRequestHandler" | "setPermissionCheckHandler" | "setDevicePermissionHandler" | "setDisplayMediaRequestHandler" | "on" | "removeListener">;
type RuntimeContents = Pick<WebContents, "id" | "isDestroyed" | "getOSProcessId">;
type RuntimeProcessMetric = Pick<ProcessMetric, "pid" | "type" | "creationTime" | "sandboxed">;
type RuntimeCallbacks = Parameters<PluginRuntimeProcessFactory["launch"]>[1];
type RuntimeRequestEvent = Pick<IpcMainInvokeEvent, "sender" | "senderFrame">;

export interface ElectronPluginRuntimeProcessOptions {
  /** The separately bundled restricted runtime preload, never Studio's preload. */
  runtimePreloadPath: string;
  createWindow(options: BrowserWindowConstructorOptions): RuntimeWindow;
  sessionFromPartition(partition: string, options: { cache: false }): RuntimeSession;
  allWebContents(): readonly RuntimeContents[];
  appMetrics(): readonly RuntimeProcessMetric[];
  /** Main-only current runtime service document callback; never a remote fetch. */
  document(url: string): Response | undefined | Promise<Response | undefined>;
  mainProcessId?: number;
  startupTimeoutMs?: number;
  stopTimeoutMs?: number;
}

interface Owned {
  input: PluginRuntimeLaunch;
  callbacks: RuntimeCallbacks;
  session: RuntimeSession;
  window: RuntimeWindow;
  /** BrowserWindow.webContents is an invalid getter after window.destroy(). */
  contents: WebContents;
  senderId: number;
  accepting: boolean;
  goneSent: boolean;
  terminalReason?: string;
  identity?: { pid: number; creationTime: number };
  termination?: Promise<void>;
  rejectStartup?: (reason: Error) => void;
  removeListeners: Array<() => void>;
}

const errorOf = (value: unknown): Error => value instanceof Error ? value : new Error(String(value));
const jsonFrame = <T>(value: T, limit: number): T => {
  let json: string | undefined;
  try { json = JSON.stringify(value); } catch { throw new Error("Runtime transport requires structured JSON."); }
  if (json === undefined || Buffer.byteLength(json, "utf8") > limit) throw new Error("Runtime transport exceeds its admitted frame.");
  return JSON.parse(json) as T;
};
const boundedTime = (value: number | undefined, fallback: number, maximum: number): number => {
  const time = value ?? fallback;
  if (!Number.isSafeInteger(time) || time < 1 || time > maximum) throw new Error("Invalid bounded runtime lifecycle timeout.");
  return time;
};

/** No Electron import side effects: root supplies the real host bindings.
 * Root routes ONLY PLUGIN_RUNTIME_REQUEST_CHANNEL directly to request(). */
export class ElectronPluginRuntimeProcessFactory implements PluginRuntimeProcessFactory {
  private readonly owned = new Map<number, Owned>();
  private readonly sessions = new Set<string>();
  private readonly mainPid: number;
  private readonly startupMs: number;
  private readonly stopMs: number;
  constructor(private readonly options: ElectronPluginRuntimeProcessOptions) {
    if (!isAbsolute(options.runtimePreloadPath)) throw new Error("Runtime requires its exact absolute dedicated preload path.");
    this.mainPid = options.mainProcessId ?? process.pid;
    if (!Number.isSafeInteger(this.mainPid) || this.mainPid < 1) throw new Error("Runtime main-process identity is unavailable.");
    this.startupMs = boundedTime(options.startupTimeoutMs, 15_000, 60_000);
    this.stopMs = boundedTime(options.stopTimeoutMs, 2_000, 10_000);
  }

  owns(senderId: number): boolean {
    const value = this.owned.get(senderId);
    return !!value?.accepting && !value.callbacks.signal.aborted && !value.contents.isDestroyed();
  }

  /** Identity comes only from Electron's actual event, never message arguments. */
  async request(event: RuntimeRequestEvent, message: unknown): Promise<PluginRuntimeRpcReply> {
    const value = this.owned.get(event.sender.id);
    if (!value || event.sender !== value.contents) throw new Error("Foreign runtime renderer.");
    this.assertFrame(value, event);
    const frame = jsonFrame(message, value.input.maxMessageBytes);
    if (!frame || typeof frame !== "object" || Array.isArray(frame)) throw new Error("Invalid runtime request frame.");
    const row = frame as Record<string, unknown>;
    if (row.format !== "omni-plugin-runtime-rpc" || row.version !== 1 || row.sessionId !== value.input.session.sessionId || row.nonce !== value.input.nonce ||
        typeof row.id !== "string") throw new Error("Stale or foreign runtime transport binding.");
    const response = await value.callbacks.receive(event.sender.id, event.senderFrame!.url, frame);
    this.assertFrame(value, event);
    const result = jsonFrame(response, value.input.maxMessageBytes);
    if (result.format !== "omni-plugin-runtime-reply" || result.version !== 1 || result.id !== row.id) throw new Error("Runtime reply does not match its current request.");
    return result;
  }

  async launch(input: PluginRuntimeLaunch, callbacks: RuntimeCallbacks): Promise<PluginRuntimeProcess> {
    callbacks.signal.throwIfAborted();
    if (typeof input.session.sessionId !== "string" || input.session.sessionId.length !== 36 || !/^[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}$/.test(input.session.sessionId) ||
        input.url !== `${PLUGIN_RUNTIME_SCHEME}://${input.session.sessionId}/index.html` || typeof input.nonce !== "string" || input.nonce.length !== 64 || !/^[a-f0-9]+$/.test(input.nonce) ||
        !Number.isSafeInteger(input.maxMessageBytes) || input.maxMessageBytes < 4096 || this.sessions.has(input.session.sessionId)) throw new Error("Invalid or already-owned runtime launch binding.");
    const launch = structuredClone(input);
    const partition = `omni-plugin-runtime-${launch.session.sessionId}-${randomUUID()}`;
    const session = this.options.sessionFromPartition(partition, { cache: false });
    session.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
    session.setPermissionCheckHandler(() => false);
    session.setDevicePermissionHandler(() => false);
    session.setDisplayMediaRequestHandler((_request, callback) => callback({}));
    let value: Owned | undefined;
    const denyDownload = (event: { preventDefault(): void }, item: { cancel(): void }): void => { event.preventDefault(); item.cancel(); };
    session.on("will-download", denyDownload);
    // Keep a deny-all interceptor installed after retirement as well. It must
    // never become a network-enabled session while an OS child is settling.
    session.webRequest.onBeforeRequest({ urls: ["<all_urls>"] }, (details, callback) => {
      callback({ cancel: !value?.accepting || callbacks.signal.aborted || details.url !== launch.url || details.resourceType !== "mainFrame" ||
        details.method !== "GET" || details.webContentsId !== value.senderId });
    });
    session.protocol.handle(PLUGIN_RUNTIME_SCHEME, async request => {
      if (!value?.accepting || callbacks.signal.aborted || request.url !== launch.url || request.method !== "GET") return new Response("Runtime unavailable", { status: 410 });
      const response = await this.options.document(launch.url);
      if (!value.accepting || callbacks.signal.aborted || !response) return new Response("Runtime retired", { status: 410 });
      // Source identity/hash-CSP belongs to the current runtime service, not a
      // second script builder or permissive fall-back document here.
      if (response.ok && (!response.headers.get("content-security-policy") || !response.headers.get("content-type")?.startsWith("text/html"))) throw new Error("Runtime document is missing its service security headers.");
      return response;
    });
    let window: RuntimeWindow;
    try {
      callbacks.signal.throwIfAborted();
      window = this.options.createWindow({ show: false, width: 1, height: 1, skipTaskbar: true, focusable: false,
        webPreferences: { session: session as Session, preload: this.options.runtimePreloadPath,
          sandbox: true, contextIsolation: true, nodeIntegration: false, nodeIntegrationInWorker: false, nodeIntegrationInSubFrames: false,
          webviewTag: false, devTools: false, webSecurity: true, allowRunningInsecureContent: false, plugins: false,
          spellcheck: false, safeDialogs: true, disableDialogs: true, navigateOnDragDrop: false,
          additionalArguments: [`--omni-plugin-runtime-max-frame=${launch.maxMessageBytes}`] } });
    } catch (error) {
      session.protocol.unhandle(PLUGIN_RUNTIME_SCHEME); session.removeListener("will-download", denyDownload);
      session.webRequest.onBeforeRequest({ urls: ["<all_urls>"] }, (_details, callback) => callback({ cancel: true }));
      throw error;
    }
    const contents = window.webContents, senderId = contents.id;
    value = { input: launch, callbacks, session, window, contents, senderId, accepting: true, goneSent: false, removeListeners: [] };
    const owned = value;
    if (this.owned.has(senderId)) { window.destroy(); session.protocol.unhandle(PLUGIN_RUNTIME_SCHEME); throw new Error("Runtime sender identity is already owned."); }
    this.owned.set(senderId, owned); this.sessions.add(launch.session.sessionId);
    try {
      const listen = (emitter: Pick<WebContents, "on" | "removeListener"> | RuntimeWindow, name: string, listener: (...args: any[]) => void): void => {
        // Both Electron host objects are documented Node EventEmitters; their
        // per-class literal overloads otherwise intersect on a union here.
        const events = emitter as Pick<EventEmitter, "on" | "removeListener">;
        events.on(name, listener); owned.removeListeners.push(() => { events.removeListener(name, listener); });
      };
      const fail = (reason: string): void => { owned.terminalReason ??= reason; owned.rejectStartup?.(new Error(reason)); void this.terminateOwned(owned).catch(() => undefined); };
      contents.setWindowOpenHandler(() => ({ action: "deny" }));
      contents.setAudioMuted(true);
      for (const event of ["will-navigate", "will-frame-navigate", "will-redirect", "will-attach-webview"])
        listen(contents, event, (details: { preventDefault(): void }) => { details.preventDefault(); });
      listen(contents, "did-navigate-in-page", (_event, url: string) => { if (url !== launch.url) fail("Runtime attempted an unowned in-page navigation."); });
      listen(contents, "render-process-gone", (_event, details: { reason?: string }) => fail(`Runtime renderer exited: ${details.reason ?? "unknown"}.`));
      listen(contents, "destroyed", () => { if (!owned.termination) fail("Runtime renderer was destroyed."); });
      listen(window, "closed", () => { if (!owned.termination) fail("Runtime window was closed."); });
      const abort = (): void => fail("Runtime activation was aborted.");
      callbacks.signal.addEventListener("abort", abort, { once: true });
      owned.removeListeners.push(() => callbacks.signal.removeEventListener("abort", abort), () => session.removeListener("will-download", denyDownload));
      let timer: ReturnType<typeof setTimeout> | undefined;
      const interrupted = new Promise<never>((_resolve, reject) => { owned.rejectStartup = reject; timer = setTimeout(() => fail("Runtime startup exceeded its bounded deadline."), this.startupMs); });
      try { callbacks.signal.throwIfAborted(); await Promise.race([window.loadURL(launch.url), interrupted]); }
      finally { if (timer) clearTimeout(timer); owned.rejectStartup = undefined; }
      this.assertCurrent(owned);
      if (contents.getURL() !== launch.url || contents.mainFrame.url !== launch.url) throw new Error("Runtime loaded an unowned document.");
      owned.identity = this.exclusiveIdentity(owned);
      return Object.freeze({ senderId, processId: owned.identity.pid,
        send: (message: PluginRuntimeDelivery): void => {
          this.assertCurrent(owned);
          if (contents.getURL() !== launch.url || contents.mainFrame.url !== launch.url || contents.getOSProcessId() !== owned.identity?.pid) throw new Error("Runtime delivery owner changed.");
          contents.send(PLUGIN_RUNTIME_DELIVERY_CHANNEL, jsonFrame(message, launch.maxMessageBytes));
        }, terminate: (): Promise<void> => this.terminateOwned(owned) });
    } catch (error) { owned.terminalReason ??= `Runtime startup failed: ${errorOf(error).message}`; await this.terminateOwned(owned); throw error; }
  }

  private assertCurrent(value: Owned): void {
    value.callbacks.signal.throwIfAborted();
    if (!value.accepting || this.owned.get(value.senderId) !== value || value.window.isDestroyed() || value.contents.isDestroyed()) throw new Error("Runtime process is retired.");
  }
  private assertFrame(value: Owned, event: RuntimeRequestEvent): void {
    this.assertCurrent(value);
    const contents = value.contents;
    if (!event.senderFrame || event.senderFrame !== contents.mainFrame || event.senderFrame.url !== value.input.url || contents.getURL() !== value.input.url ||
        value.identity && contents.getOSProcessId() !== value.identity.pid) throw new Error("Runtime request came from a foreign document or non-main frame.");
  }
  private notifyGone(value: Owned, reason: string): void {
    if (value.goneSent) return;
    value.goneSent = true;
    try { value.callbacks.gone(reason); } catch { /* A notification cannot prevent owned child cleanup. */ }
  }
  private exclusiveIdentity(value: Owned): { pid: number; creationTime: number } {
    const contents = value.contents;
    if (contents.isDestroyed()) throw new Error("Runtime has no exclusive renderer OS identity.");
    const pid = contents.getOSProcessId();
    if (!Number.isSafeInteger(pid) || pid < 1 || pid === this.mainPid) throw new Error("Runtime has no exclusive renderer OS identity.");
    const peers = this.options.allWebContents().filter(other => !other.isDestroyed() && other.getOSProcessId() === pid);
    const metrics = this.options.appMetrics().filter(row => row.pid === pid);
    if (peers.length !== 1 || peers[0] !== contents || metrics.length !== 1 || metrics[0]!.type !== "Tab" ||
        !Number.isFinite(metrics[0]!.creationTime) || metrics[0]!.creationTime <= 0 || metrics[0]!.sandboxed === false ||
        value.identity && (value.identity.pid !== pid || value.identity.creationTime !== metrics[0]!.creationTime) || contents.getOSProcessId() !== pid)
      throw new Error("Runtime OS process is shared, reused, protected or unverifiable; forceful stop is forbidden.");
    return { pid, creationTime: metrics[0]!.creationTime };
  }
  private terminateOwned(value: Owned): Promise<void> {
    if (value.termination) return value.termination;
    value.accepting = false; this.owned.delete(value.senderId); this.sessions.delete(value.input.session.sessionId);
    const contents = value.contents;
    let releaseError: Error | undefined;
    value.termination = Promise.resolve().then(async () => {
      let exclusive: { pid: number; creationTime: number } | undefined;
      try {
        if (!contents.isDestroyed()) {
          contents.stop();
          try { exclusive = this.exclusiveIdentity(value); } catch { /* Never kill a shared/foreign/unknown process. */ }
          if (exclusive) {
            const current = this.exclusiveIdentity(value); // no await between actual ownership check and crash
            if (current.pid !== exclusive.pid || current.creationTime !== exclusive.creationTime) throw new Error("Runtime process changed before stop.");
            contents.forcefullyCrashRenderer();
          }
        }
      } finally {
        if (!value.window.isDestroyed()) value.window.destroy();
      }
      const deadline = performance.now() + this.stopMs;
      while (!value.window.isDestroyed() || !contents.isDestroyed() || exclusive && this.options.appMetrics().some(row => row.pid === exclusive!.pid && row.creationTime === exclusive!.creationTime)) {
        if (performance.now() >= deadline) throw new Error("Runtime child release could not be verified within its bounded stop; no foreign process was killed.");
        await new Promise(resolve => setTimeout(resolve, Math.min(20, this.stopMs)));
      }
    }).catch(error => { releaseError = errorOf(error); throw error; }).finally(() => {
      for (const remove of value.removeListeners.splice(0)) { try { remove(); } catch { /* Already destroyed. */ } }
      try { value.session.protocol.unhandle(PLUGIN_RUNTIME_SCHEME); } catch { /* Already unhandled. */ }
      try { value.session.webRequest.onBeforeRequest({ urls: ["<all_urls>"] }, (_details, callback) => callback({ cancel: true })); } catch { /* Closed session remains inaccessible. */ }
      this.notifyGone(value, releaseError ? `Runtime release unverified: ${releaseError.message}` : value.terminalReason ?? "Runtime child stopped.");
    });
    return value.termination;
  }
}
