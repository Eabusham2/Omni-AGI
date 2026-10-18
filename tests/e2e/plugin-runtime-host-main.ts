/** Actual Electron process boundary with owned JS/synthetic catalog, never a neural fixture. */
import { app, BrowserWindow, ipcMain, protocol, session, webContents } from "electron";
import { mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
import { freemem } from "node:os";
import { getHeapStatistics } from "node:v8";
import assert from "node:assert/strict";
import { PluginRuntimeService } from "../../src/main/pluginRuntimeService";
import { ElectronPluginRuntimeProcessFactory } from "../../src/main/pluginRuntimeProcess";
import { PluginRuntimeResources } from "../../src/main/pluginRuntimeResources";
import { automationResidentBytes } from "../../src/main/automationResources";
import { PluginRuntimeStore } from "../../src/main/pluginRuntimeStore";
import type { PluginService } from "../../src/main/pluginService";
import type { PluginRuntimePublishedTool, PluginRuntimeShutdownDiagnostic } from "../../src/shared/pluginRuntime";
import { PLUGIN_RUNTIME_SCHEME, PLUGIN_RUNTIME_REQUEST_CHANNEL } from "../../src/shared/pluginRuntime";

protocol.registerSchemesAsPrivileged([{ scheme: PLUGIN_RUNTIME_SCHEME, privileges: { standard: true, secure: true, supportFetchAPI: false } }]);
if (!process.env.OMNI_PLUGIN_HOST_REPORT || !process.env.OMNI_PLUGIN_HOST_PRELOAD) throw new Error("Exact isolated report and compiled runtime preload paths are required.");
app.setName("Omni owned plugin boundary check");
let runtime: PluginRuntimeService | undefined, shutdownDiagnostic: PluginRuntimeShutdownDiagnostic | undefined, peak = 0, passed = false;
const phases: Array<{ phase: string; elapsedMs: number }> = [], start = performance.now();
const mark = (phase: string) => { phases.push({ phase, elapsedMs: performance.now() - start }); process.stdout.write(`${phase}\n`); };
let watchdog: ReturnType<typeof setTimeout>;
app.whenReady().then(async () => {
  const directory = await mkdtemp(join(tmpdir(), "omni-owned-module-"));
  await mkdir(join(directory, "electron-profile"));
  app.setPath("userData", join(directory, "electron-profile"));
  const hash = "a".repeat(64), brainId = "owned-plugin-fixture", published: PluginRuntimePublishedTool[] = [];
  const module = { id: "feature", title: "Owned math feature", description: "No brain or host effects", events: [],
    ui: { kinds: ["button" as const], slots: ["tools"] }, dynamicTools: true, script: `
omniRuntime.onDeactivate(() => {});
omniRuntime.onActivate(async () => {
  await omniRuntime.state.put("isolation", {node:typeof require,process:typeof process,studio:typeof globalThis.omni},0);
  await omniRuntime.registerTool({id:"sum",title:"Owned sum",description:"Pure fixture math",actions:[{id:"add",description:"Add",inputSchema:{type:"object",properties:{a:{type:"number"},b:{type:"number"}},required:["a","b"],additionalProperties:false}}]}, {add: ({a,b})=>({sum:a+b})});
  await omniRuntime.ui.put({id:"button",kind:"button",slot:"tools",title:"Owned fixture",event:"owned"});
});` };
  const plugins = { list: async () => [{ pluginId: "fixture.owned", enabled: true, activeContentSha256: hash, revision: 1 }],
    instance: async () => ({ enabled: true, contentSha256: hash, revision: 1, capabilities: [] }),
    package: async () => ({ manifest: { runtimeModules: [module] } }), onEvent: () => () => {} } as unknown as PluginService;
  const resources = new PluginRuntimeResources(async () => {
    const residentBytes = await automationResidentBytes([...app.getAppMetrics().map(row => row.pid), process.pid]); peak = Math.max(peak, residentBytes);
    return { residentBytes, ceilingBytes: 768 * 1024 * 1024, freeBytes: Math.floor(freemem()), heapAvailableBytes: getHeapStatistics().heap_size_limit - process.memoryUsage().heapUsed };
  });
  const processes = new ElectronPluginRuntimeProcessFactory({ runtimePreloadPath: resolve(process.env.OMNI_PLUGIN_HOST_PRELOAD!),
    createWindow: options => new BrowserWindow(options), sessionFromPartition: (partition, options) => session.fromPartition(partition, options),
    allWebContents: () => webContents.getAllWebContents(), appMetrics: () => app.getAppMetrics(), document: url => runtime?.document(url), startupTimeoutMs: 5000 });
  runtime = new PluginRuntimeService(plugins, { brainRoot: () => directory, processFactory: processes,
    admit: input => resources.admit(input.brainId, input.scriptBytes), admitRead: (id, bytes) => resources.admitTransfer(id, bytes),
    tools: { execute: async () => { throw new Error("No real host action is allowed in this fixture."); }, cancelAndWait: async () => 0, invalidateApproval: () => false },
    checkDisk: async (_id, bytes) => { if (bytes > 8 * 1024 * 1024) throw new Error("Unexpected fixture write volume."); },
    publishTools: async (_id, tools) => { published.push(...tools); }, retireTools: async () => {}, isBrainLoaded: () => false, moduleReadyTimeoutMs: 5000 });
  ipcMain.handle(PLUGIN_RUNTIME_REQUEST_CHANNEL, (event, message) => processes.request(event, message));
  watchdog = setTimeout(() => { process.stderr.write("Owned plugin fixture deadline exceeded\n"); void runtime?.dispose().finally(() => app.exit(2)); }, 20000);
  try {
    mark("activate");
    const active = await runtime.activate({ brainId, pluginId: "fixture.owned", contentSha256: hash, moduleId: module.id, expectedCatalogRevision: 1, expectedInstanceRevision: 1, requestId: "owned-human-activation" });
    assert.equal(active.state, "active"); assert.ok(active.processId && active.processId !== process.pid); mark("ready-ack");
    const deadline = performance.now() + 5000;
    while (!published.length || !(await runtime.contributions(brainId)).length) {
      if (performance.now() > deadline) throw new Error("Owned activation handlers did not finish.");
      await new Promise(resolve => setTimeout(resolve, 20));
    }
    const store = await PluginRuntimeStore.open(directory, brainId, { readOnly: true });
    assert.ok(store); try { assert.deepEqual(store.value(hash, module.id, "isolation").value, { node: "undefined", process: "undefined", studio: "undefined" }); } finally { store.close(); }
    mark("isolated-state-and-contribution");
    const tool = published[0]!;
    assert.deepEqual(await runtime.executeTool(brainId, tool.toolId, "add", { a: 2, b: 3 }, "owned-valid-request", new AbortController().signal), { sum: 5 });
    await assert.rejects(runtime.executeTool(brainId, tool.toolId, "add", { a: "2", b: 3 }, "owned-invalid-request", new AbortController().signal), /schema|argument|input/i);
    mark("typed-pure-tool");
    await runtime.stop(brainId, active.sessionId); assert.equal((await runtime.contributions(brainId)).length, 0);
    shutdownDiagnostic = (await runtime.history(brainId, { maxBytes: 65536 })).events.find(event => event.kind === "runtime-shutdown")?.value as PluginRuntimeShutdownDiagnostic | undefined;
    assert.ok(shutdownDiagnostic); assert.equal(shutdownDiagnostic.state, "acknowledged");
    assert.deepEqual(shutdownDiagnostic.acknowledgement, { callbacks: 1, completed: 1, failed: 0, errors: [] });
    mark("local-cleanup-acknowledged");
    assert.equal(webContents.getAllWebContents().filter(item => !item.isDestroyed()).length, 0); mark("owned-process-released"); passed = true;
  } catch (error) { phases.push({ phase: `FAILED: ${error instanceof Error ? error.message : String(error)}`, elapsedMs: performance.now() - start }); throw error; }
  finally {
    clearTimeout(watchdog); await runtime.dispose(); ipcMain.removeHandler(PLUGIN_RUNTIME_REQUEST_CHANNEL);
    await writeFile(resolve(process.env.OMNI_PLUGIN_HOST_REPORT!), JSON.stringify({ passed, neuralExecution: false, realHostActions: false, syntheticCatalog: true,
      peakSampledFamilyRssBytes: peak, directory, phases, shutdownDiagnostic, elapsedMs: performance.now() - start }, null, 2));
    app.exit(passed ? 0 : 1);
  }
}).catch(error => { console.error(error); app.exit(1); });
app.on("window-all-closed", () => {});
