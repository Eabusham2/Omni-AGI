import { randomUUID } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginRuntimeService } from "../src/main/pluginRuntimeService";
import { PluginRuntimeStore, PLUGIN_RUNTIME_WORKSPACE } from "../src/main/pluginRuntimeStore";
import { canonicalPluginPackage } from "../src/main/pluginCatalog";
import { pluginRuntimeDocument, pluginRuntimeScript } from "../src/main/pluginRuntimeBootstrap";
import { runtimeCss, runtimeHash, runtimeJson } from "../src/main/pluginRuntimeValidation";
import type { PluginService } from "../src/main/pluginService";
import type { PluginRuntimeActivationRequest, PluginRuntimeDelivery, PluginRuntimeLaunch, PluginRuntimeModuleDefinition, PluginRuntimeProcessFactory, PluginRuntimePublishedTool, PluginRuntimeRpcReply, PluginRuntimeSchemaLearning } from "../src/shared/pluginRuntime";
import type { ToolExecutionResult, ToolInvocation } from "../src/shared/types";
import { OMNI_PLUGIN_FORMAT, OMNI_PLUGIN_FORMAT_VERSION } from "../src/shared/plugins";

const roots: string[] = [], services: PluginRuntimeService[] = [];
afterEach(async () => { for (const service of services.splice(0)) await service.dispose(); for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); vi.useRealTimers(); });
async function root(): Promise<string> { const value = await mkdtemp(join(tmpdir(), "omni-runtime-source-")); roots.push(value); return value; }
function deferred<T>() { let resolve!: (value: T) => void, reject!: (error: Error) => void; const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail; }); return { promise, resolve, reject }; }
const moduleDefinition = (script = "globalThis.omniRuntime.onActivate(()=>{});"): PluginRuntimeModuleDefinition => ({ id: "feature", title: "Owned feature", description: "Tiny source fixture", script, events: ["task.changed"], ui: { kinds: ["style", "button", "panel"], slots: ["studio", "run.sidebar", "run.toolbar"] }, dynamicTools: true });
const complete = (output?: unknown): ToolExecutionResult => ({ id: randomUUID(), toolId: "system.files", action: "read", state: "complete", startedAt: new Date().toISOString(), ...(output === undefined ? {} : { output }) });
async function fixture(options: { frame?: number; module?: PluginRuntimeModuleDefinition; execute?: (invocation: ToolInvocation) => Promise<ToolExecutionResult>; delayedLaunch?: boolean; loaded?: boolean; readiness?: "ok" | "failed" | "none" } = {}) {
  const directory = await root(), hash = "a".repeat(64), definition = options.module ?? moduleDefinition(), sent: PluginRuntimeDelivery[] = [], published: PluginRuntimePublishedTool[] = [], schemas: PluginRuntimeSchemaLearning[] = [];
  let launch!: PluginRuntimeLaunch, callbacks!: Parameters<PluginRuntimeProcessFactory["launch"]>[1], listener: ((event: any) => void) | undefined;
  const entry = { pluginId: "fixture.feature", enabled: true, activeContentSha256: hash, revision: 1 }, access = { enabled: true, contentSha256: hash, revision: 2, capabilities: [{ toolId: "system.files", enabled: true, actions: ["read"] }] };
  const plugin = { list: async () => [entry], instance: async () => access, package: async () => ({ manifest: { runtimeModules: [definition] } }), onEvent: (fn: typeof listener) => { listener = fn; return () => { listener = undefined; }; } } as unknown as PluginService;
  const terminate = vi.fn(async () => undefined), check = vi.fn(async (_pid?: number) => undefined), release = vi.fn(async () => undefined), ready = deferred<void>(), launched = deferred<void>();
  const execute = vi.fn(options.execute ?? (async () => complete({ exact: "actual result" }))), cancel = vi.fn(async () => 1), invalidated: string[] = [], retire = vi.fn(async (_brain: string, _ids: string[]) => undefined);
  const service = new PluginRuntimeService(plugin, { brainRoot: () => directory, admit: async () => ({ maxMessageBytes: options.frame ?? 8192, check, release }), admitRead: async () => ({ release: () => undefined }),
    tools: { execute, cancelAndWait: cancel, invalidateApproval: (_brain, token) => { invalidated.push(token); return true; } },
    moduleReadyTimeoutMs: 50,
    processFactory: { launch: async (input, observed) => { launch = input; callbacks = observed; launched.resolve(); if (options.delayedLaunch) await ready.promise;
      if (options.readiness !== "none") setImmediate(() => { void observed.receive(23, input.url, { format: "omni-plugin-runtime-rpc", version: 1, sessionId: input.session.sessionId, nonce: input.nonce, id: "owned-bootstrap-ready", operation: "ready", arguments: options.readiness === "failed" ? { ok: false, error: "Owned source initialization failed" } : { ok: true } }).catch(() => undefined); });
      return { senderId: 23, processId: 4321, send: message => sent.push(message), terminate }; } },
    publishTools: async (_brain, tools) => { published.push(...tools); }, retireTools: retire, isBrainLoaded: () => options.loaded ?? false,
    learnSchemas: async job => { schemas.push(job); return { sourceSha256: job.sourceSha256, receipt: { actual: "fake source-only callback" } }; } }); services.push(service);
  const request: PluginRuntimeActivationRequest = { brainId: "owner", pluginId: entry.pluginId, contentSha256: hash, moduleId: definition.id, expectedCatalogRevision: 1, expectedInstanceRevision: 2, requestId: "human-activation" };
  const activate = () => service.activate(request), call = (operation: string, args: Record<string, unknown> = {}, id: string = randomUUID(), sender = 23, url?: string) => service.receive(sender, url ?? launch.url, { format: "omni-plugin-runtime-rpc", version: 1, sessionId: launch.session.sessionId, nonce: launch.nonce, id, operation, arguments: args });
  return { service, directory, hash, definition, entry, access, sent, published, schemas, request, activate, call, execute, cancel, terminate, check, release, ready, launchEntered: launched.promise, launch: () => launch, callbacks: () => callbacks, invalidate: invalidated, retire, mutate: () => listener?.({ type: "instance", brainId: "owner", pluginId: entry.pluginId }) };
}
async function settle(test: () => boolean): Promise<void> { for (let i = 0; i < 200; i++) { if (test()) return; await new Promise(resolve => setImmediate(resolve)); } throw new Error("Owned async fixture did not settle."); }

describe("optional lifecycle modules - new source/fake-host edges only", () => {
  it("preserves old manifest hashes when runtimeModules is absent, while reviewed scripts alter the hash", () => {
    const base = { format: OMNI_PLUGIN_FORMAT, formatVersion: OMNI_PLUGIN_FORMAT_VERSION, manifest: { id: "fixture.feature", version: "1.0.0", name: "Feature", description: "Existing fixture", author: "Fixture", license: "MIT", documentation: "Fixture docs", capabilities: [], tools: [], panels: [], themes: [], trainingFormats: [] } };
    const old = canonicalPluginPackage(base); expect(old.json).toBe(runtimeJson(base)); expect(old.package.manifest).not.toHaveProperty("runtimeModules");
    const next = canonicalPluginPackage({ ...base, manifest: { ...base.manifest, runtimeModules: [moduleDefinition()] } }); expect(next.sha256).not.toBe(old.sha256);
  });

  it("does not launch on construction/list and rejects stale hash, sender and subframe identity", async () => {
    const f = await fixture(); expect(await f.service.list("owner")).toEqual({ sessions: [], total: 0 }); expect(f.launch()).toBeUndefined();
    await expect(f.service.activate({ ...f.request, contentSha256: "b".repeat(64) })).rejects.toThrow(/exact enabled/); const session = await f.activate();
    await expect(f.call("ready", {}, "foreign", 24)).rejects.toThrow(/foreign renderer/); await expect(f.call("ready", {}, "frame", 23, `${f.launch().url}?other`)).rejects.toThrow(/foreign renderer/);
    expect((await f.service.list("owner")).sessions[0]?.sessionId).toBe(session.sessionId); expect((await f.service.document(`${f.launch().url}?nonce`)).status).toBe(404);
  });

  it("coalesces exact concurrent requests before awaits and refuses changed actual operands", async () => {
    const gate = deferred<ToolExecutionResult>(), f = await fixture({ execute: async () => gate.promise }); await f.activate(); const args = { toolId: "system.files", action: "read", arguments: { path: "owned-fixture" } };
    const first = f.call("call", args, "same-rpc"); await settle(() => f.execute.mock.calls.length === 1); const retry = f.call("call", args, "same-rpc");
    await expect(f.call("call", { ...args, arguments: { path: "changed" } }, "same-rpc")).rejects.toThrow(/changed actual/); gate.resolve(complete({ exact: 7 })); expect(await first).toEqual(await retry); expect(f.execute).toHaveBeenCalledTimes(1);
  });

  it("retains uncertain dispatched effects and never reruns after a lost ACK", async () => {
    let effects = 0; const f = await fixture({ execute: async () => { effects++; throw new Error("Transport lost after attempted effect"); } }); await f.activate();
    const args = { toolId: "system.files", action: "read", arguments: {} }; expect((await f.call("call", args, "lost-ack")).state).toBe("uncertain"); expect((await f.call("call", args, "lost-ack")).state).toBe("uncertain"); expect(effects).toBe(1);
  });

  it("requires enabled declared nested capabilities and keeps Ask tokens outside portable/module state", async () => {
    const f = await fixture({ execute: async invocation => invocation.approvalToken ? complete({ approved: true }) : { ...complete(), state: "approval-required", approvalToken: "DEVICE_ONLY_SECRET_TOKEN" } }); await f.activate();
    expect((await f.call("call", { toolId: "system.shell", action: "run", arguments: {} })).state).toBe("failed");
    expect((await f.call("call", { toolId: "system.files", action: "read", arguments: {} }, "ask")).state).toBe("approval-required");
    const pending = (await f.service.approvals("owner"))[0]!; expect(JSON.stringify(pending)).not.toContain("DEVICE_ONLY_SECRET_TOKEN");
    const result = await f.service.review({ brainId: "owner", sessionId: pending.sessionId, rpcId: pending.rpcId, reviewId: pending.reviewId, decision: "approve" }); expect(result.value).toEqual({ approved: true });
    expect((await readFile(join(f.directory, PLUGIN_RUNTIME_WORKSPACE))).includes(Buffer.from("DEVICE_ONLY_SECRET_TOKEN"))).toBe(false); expect(JSON.stringify(f.sent)).not.toContain("DEVICE_ONLY_SECRET_TOKEN");
  });

  it("supports beyond 256 operations in one session with full paged durable history", async () => {
    const f = await fixture(); await f.activate(); for (let i = 0; i < 270; i++) expect((await f.call("state.put", { key: "counter", value: i, expectedRevision: i }, `op-${i}`)).state).toBe("complete");
    expect((await f.call("state.get", { key: "counter" })).value).toEqual({ revision: 270, value: 269 }); expect(f.terminate).not.toHaveBeenCalled();
    const first = await f.service.history("owner", { maxBytes: 4096 }); expect(first.nextSequence).toBeDefined(); expect(first.total).toBeGreaterThan(270); expect(first.events.length).toBeLessThan(first.total);
  });

  it("registers maximal local id with full 74-character integration identity and actual schema-source receipt", async () => {
    const f = await fixture({ loaded: true }); await f.activate(); const definition = { id: "a".repeat(64), title: "Max local id", description: "Actual typed tool", actions: [{ id: "compute", description: "Compute", inputSchema: { type: "object", properties: { n: { type: "number" } }, required: ["n"] } }] };
    expect((await f.call("register-tool", { definition })).state).toBe("complete"); await f.service.flushSchemaLearning("owner"); const tool = f.published[0]!;
    expect(tool.toolId).toMatch(/^extension\.[a-f0-9]{64}$/); expect(tool.toolId.length).toBe(74); expect(tool.definition.id).toBe(definition.id); expect(f.schemas).toHaveLength(1); expect(runtimeHash(f.schemas[0]!.text)).toBe(f.schemas[0]!.sourceSha256); expect(f.schemas[0]!.sourceSha256).toBe(tool.sourceSha256);
    const output = f.service.executeTool("owner", tool.toolId, "compute", { n: 6 }, "actual-host-id", new AbortController().signal); await settle(() => f.sent.some(row => row.kind === "invoke-tool")); const invocation = f.sent.find(row => row.kind === "invoke-tool")!;
    await f.call("tool-result", { id: invocation.id, value: 720 }, "result"); expect(await output).toBe(720);
  });

  it("pages a large retained result without effect repetition or body truncation", async () => {
    const text = "complete unsanitized🙂".repeat(700), f = await fixture({ execute: async () => complete(text) }); await f.activate(); const result = await f.call("call", { toolId: "system.files", action: "read", arguments: {} }, "large");
    expect(result.receipt?.bytes).toBeGreaterThan(8192); let offset = 0; const parts: Buffer[] = [];
    do { const page = (await f.call("receipt.read", { id: "large", offset, length: 3000 })).value as { text: string; nextOffset?: number }; parts.push(Buffer.from(page.text, "base64")); if (page.nextOffset === undefined) break; offset = page.nextOffset; } while (true);
    const body = Buffer.concat(parts).toString("utf8"); expect(runtimeHash(body)).toBe(result.receipt?.sha256); expect(JSON.parse(body).value).toBe(text); expect(f.execute).toHaveBeenCalledTimes(1);
  });

  it("cancels a still-starting process and releases its admitted lease", async () => {
    const f = await fixture({ delayedLaunch: true }), activating = f.activate(); await settle(() => Boolean(f.launch())); const sessionId = f.launch().session.sessionId;
    await f.service.stop("owner", sessionId); expect(f.callbacks().signal.aborted).toBe(true); f.ready.resolve(); await expect(activating).rejects.toThrow(/stopped while/); expect(f.terminate).toHaveBeenCalledTimes(1); expect(f.release).toHaveBeenCalledTimes(1);
  });

  it("checks whole-family resource pressure while the process factory is still starting", async () => {
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] }); const f = await fixture({ delayedLaunch: true }), activating = f.activate();
    // Custody I/O must reach the actual starting-factory boundary before the
    // fake pressure clock advances; event-loop spin counts are not I/O fences.
    await f.launchEntered;
    const signal = f.callbacks().signal, aborted = new Promise<void>(resolve => signal.addEventListener("abort", () => resolve(), { once: true }));
    f.check.mockRejectedValue(new Error("Actual family pressure")); await vi.advanceTimersByTimeAsync(1001); await aborted;
    expect(signal.aborted).toBe(true); f.ready.resolve();
    await expect(activating).rejects.toThrow(/stopped while/); expect(f.release).toHaveBeenCalledTimes(1);
  });

  it("keeps loadURL success unaccepted until a real module ACK and records source errors/missing ACK truthfully", async () => {
    const failed = await fixture({ readiness: "failed" }); await expect(failed.activate()).rejects.toThrow(/Owned source initialization failed/); expect(failed.sent.some(row => row.kind === "activate")).toBe(false); expect(failed.terminate).toHaveBeenCalledTimes(1);
    const absent = await fixture({ readiness: "none" }); await expect(absent.activate()).rejects.toThrow(/did not acknowledge/); expect(absent.sent.some(row => row.kind === "activate")).toBe(false); expect(absent.release).toHaveBeenCalledTimes(1);
  });

  it("validates actual dynamic arguments locally without coercion/defaults or remote refs before module dispatch", async () => {
    const f = await fixture(); await f.activate(); const schema = { type: "object", properties: { n: { type: "integer", minimum: 1 }, label: { type: "string", default: "must-not-be-added" } }, required: ["n"], additionalProperties: false };
    expect((await f.call("register-tool", { definition: { id: "typed", title: "Typed", description: "Actual input schema", actions: [{ id: "compute", description: "Compute", inputSchema: schema }] } })).state).toBe("complete");
    const tool = f.published[0]!; await expect(f.service.executeTool("owner", tool.toolId, "compute", { n: "6" }, "invalid-args", new AbortController().signal)).rejects.toThrow(/actual|Actual/); expect(f.sent.filter(row => row.kind === "invoke-tool")).toHaveLength(0);
    const args = { n: 6 }, output = f.service.executeTool("owner", tool.toolId, "compute", args, "typed-call", new AbortController().signal); await settle(() => f.sent.some(row => row.kind === "invoke-tool")); const invoked = f.sent.find(row => row.kind === "invoke-tool")!; expect(invoked.data).toEqual({ n: 6 }); expect(args).toEqual({ n: 6 }); await f.call("tool-result", { id: invoked.id, value: 720 }, "typed-result"); expect(await output).toBe(720);
    expect((await f.call("register-tool", { definition: { id: "remote", title: "Remote ref", description: "Must fail locally", actions: [{ id: "compute", description: "Compute", inputSchema: { type: "object", $ref: "https://example.invalid/schema" } }] } })).state).toBe("failed"); expect(f.published).toHaveLength(1);
    expect((await f.call("register-tool", { definition: { id: "async", title: "Async schema", description: "Must not bypass synchronous dispatch", actions: [{ id: "compute", description: "Compute", inputSchema: { type: "object", $async: true } }] } })).state).toBe("failed"); expect(f.published).toHaveLength(1);
  });

  it("waits for the same genuine ACK on concurrent activation and does not parse oversized retry replies", async () => {
    const f = await fixture({ delayedLaunch: true }), first = f.activate();
    // Await the actual factory boundary after admitted filesystem/custody work,
    // not an arbitrary number of event-loop turns that may finish before I/O.
    // The real bootstrap ACK and its production deadline remain unchanged.
    await f.launchEntered; let resolved = false; const second = f.activate().then(value => { resolved = true; return value; });
    await new Promise(resolve => setImmediate(resolve)); expect(resolved).toBe(false); f.ready.resolve(); expect((await first).sessionId).toBe((await second).sessionId);
    const store = (await PluginRuntimeStore.open(f.directory, "owner"))!, id = "huge-retry", payload = "retained🙂".repeat(4000); try {
      await store.beginRpc(f.launch().session.sessionId, id, "call", { real: 1 }); await store.markRpc(f.launch().session.sessionId, id, "complete", { format: "omni-plugin-runtime-reply", version: 1, id, state: "complete", value: payload });
      const parse = vi.spyOn(JSON, "parse"), retained = store.rpc(f.launch().session.sessionId, id, 4096); expect(retained?.reply?.receipt?.bytes).toBeGreaterThan(4096); expect(retained?.reply?.state).toBe("uncertain"); expect(parse.mock.calls.every(call => !String(call[0]).includes(payload))).toBe(true); parse.mockRestore();
    } finally { store.close(); }
  });

  it("keeps metadata/contribution callbacks current and never fabricates event completion", async () => {
    const f = await fixture(); await f.activate(); await f.call("ui.put", { contribution: { id: "calculate", kind: "button", slot: "run.toolbar", title: "Compute", event: "calculate" } });
    const item = (await f.service.contributions("owner"))[0]!; await f.service.invokeContribution("owner", item.sessionId, item.id, "click-once", { n: 4 }); await f.service.invokeContribution("owner", item.sessionId, item.id, "click-once", { n: 4 }); expect(f.sent.filter(row => row.kind === "event")).toHaveLength(1);
    await f.service.stop("owner", item.sessionId); await expect(f.service.invokeContribution("owner", item.sessionId, item.id, "stale")).rejects.toThrow(/stale/); expect(await f.service.contributions("owner")).toEqual([]);
  });
});

describe("owned program bootstrap and CSS subset", () => {
  it("runs generic owned JS handlers through fixed fake transport and installs tool handlers before registration ACK", async () => {
    const requests: any[] = [], reports: unknown[] = []; let delivery!: (message: PluginRuntimeDelivery) => void; const done = deferred<void>();
    const module = moduleDefinition(`const sdk=globalThis.omniRuntime;sdk.onActivate(async()=>{await sdk.registerTool({id:"math",title:"Math",description:"Compute",actions:[{id:"factorial",description:"Product",inputSchema:{type:"object"}}]},{factorial:({n})=>{let x=1;for(let i=2;i<=n;i++)x*=i;return x;}});globalThis.report("active");});sdk.on("task.changed",data=>globalThis.report(data));`);
    const launch: PluginRuntimeLaunch = { session: { format: "omni-plugin-runtime-session", version: 1, brainId: "owner", pluginId: "fixture.feature", contentSha256: "a".repeat(64), moduleId: "feature", sessionId: randomUUID(), catalogRevision: 1, instanceRevision: 2, state: "starting", startedAt: new Date().toISOString() }, nonce: "b".repeat(64), maxMessageBytes: 8192, url: "unused" };
    const transport = { subscribe: (fn: typeof delivery) => { delivery = fn; }, request: async (request: any): Promise<PluginRuntimeRpcReply> => { requests.push(request);
      if (request.operation === "register-tool") delivery({ kind: "invoke-tool", id: "racing-call", toolId: "math", action: "factorial", data: { n: 6 } }); if (request.operation === "tool-result") done.resolve();
      return { format: "omni-plugin-runtime-reply", version: 1, id: request.id, state: "complete" }; } };
    // new Function executes ONLY this test-owned body; it is not the production security boundary.
    new Function("globalThis", "crypto", "setTimeout", pluginRuntimeScript(launch, module))({ __omniRuntimeTransport: transport, report: (value: unknown) => reports.push(value) }, { randomUUID }, setTimeout);
    delivery({ kind: "activate", id: "activate" }); await done.promise; expect(requests.find(row => row.operation === "tool-result")?.arguments.value).toBe(720); await settle(() => reports.includes("active")); delivery({ kind: "event", id: "event", topic: "task.changed", data: { exact: 1 } }); await settle(() => reports.length === 2); expect(reports[1]).toEqual({ exact: 1 });
    const document = pluginRuntimeDocument(launch, module); expect(document.headers.get("Content-Security-Policy")).toContain("connect-src 'none'"); expect(document.headers.get("Content-Security-Policy")).not.toContain("unsafe-eval");
  });

  it("keeps responsive CSS while rejecting escaped/commented/resource-bearing forms", () => {
    expect(runtimeCss("@media(max-width:600px){.omni{color:var(--text)}} @supports(display:grid){.omni{display:grid}} @keyframes fade{to{opacity:1}}")).toContain("@media");
    for (const css of [".a{background:u\\72l('https://example.test')}", ".a{background:image-set('https://example.test' 1x)}", "@import 'https://example.test'", "@font-face{font-family:x;src:'https://example.test'}", ".a{background:u/**/rl(x)}", ".a{background:image('x')}"]) expect(() => runtimeCss(css)).toThrow(/resources/);
  });
});
