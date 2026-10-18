import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BrainRepository } from "../src/main/brainRepository";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import type { EngineSupervisor } from "../src/main/engineSupervisor";
import { PluginCatalog, canonicalPluginPackage } from "../src/main/pluginCatalog";
import {
  PluginInteractivePanelService,
  allowPluginInteractivePanelNavigation,
  allowPluginInteractivePanelRequest,
  pluginInteractivePanelRequestHeaders,
  pluginInteractivePanelResponseHeaders,
  type PluginPanelToolExecutor
} from "../src/main/pluginInteractivePanels";
import { pluginSemanticLearningExperience } from "../src/main/pluginSemanticLearning";
import { PluginService } from "../src/main/pluginService";
import { ToolExecutor } from "../src/main/toolExecutor";
import {
  OMNI_PLUGIN_FORMAT,
  OMNI_PLUGIN_FORMAT_VERSION,
  pluginToolId,
  type OmniPluginPackage
} from "../src/shared/plugins";
import {
  PLUGIN_INTERACTIVE_MESSAGE_FORMAT,
  PLUGIN_INTERACTIVE_MESSAGE_VERSION,
  parsePluginInteractivePanelUrl,
  type PluginInteractiveToolRequest
} from "../src/shared/pluginInteractivePanels";
import { DEFAULT_CONFIG, type ToolExecutionResult } from "../src/shared/types";

function deferred<T = void>(): { promise: Promise<T>; resolve(value: T): void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((settle) => { resolve = settle; });
  return { promise, resolve };
}

function packageValue(): OmniPluginPackage {
  return {
    format: OMNI_PLUGIN_FORMAT,
    formatVersion: OMNI_PLUGIN_FORMAT_VERSION,
    manifest: {
      id: "example.canvas",
      version: "1.0.0",
      name: "Canvas helper",
      description: "An isolated interactive panel fixture.",
      author: "Fixture",
      license: "MIT",
      documentation: "# Canvas helper\n\nUses one declared file reader.",
      capabilities: [{ toolId: "system.files", actions: ["read"], reason: "Read a reviewed file." }],
      tools: [{ id: "reader", title: "Reader", description: "Read a file", actions: [{
        id: "read",
        inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
        delegate: { toolId: "system.files", action: "read", arguments: { forward: ["path"] } }
      }] }],
      panels: [{ id: "about", title: "Static", html: "<p>Static stays inert.</p>" }],
      interactivePanels: [{
        id: "canvas",
        title: "Canvas </title>",
        html: "<main><button id=\"run\">Run</button><canvas id=\"view\"></canvas></main>",
        css: "main { padding: 1rem; }",
        script: "document.querySelector('#run').addEventListener('click',()=>void omniPlugin.call('reader','read',{path:'note.txt'}));",
        tools: [{ toolId: "reader", actions: ["read"] }]
      }],
      themes: [],
      trainingFormats: []
    }
  };
}

async function enabledFixture(root: string, brainId = "brain-one") {
  const catalog = new PluginCatalog(join(root, "plugins"));
  await catalog.initialize();
  const installed = await catalog.install(packageValue());
  const semantic = pluginSemanticLearningExperience(installed.package, installed.contentSha256).sha256;
  const entry = (await catalog.list())[0]!;
  await catalog.setEnabled(entry.pluginId, true, entry.activeContentSha256, entry.revision, semantic);
  const outerToolId = pluginToolId(entry.pluginId, "reader");
  await catalog.setInstanceAccess({
    brainId,
    pluginId: entry.pluginId,
    contentSha256: installed.contentSha256,
    enabled: true,
    capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
    toolPermissions: [{ toolId: outerToolId, level: "full" }],
    requestId: "fixture-access"
  }, semantic);
  const brain = {
    registerExternalToolSchemas: vi.fn(),
    unregisterExternalToolSchemas: vi.fn(),
    listToolPermissions: vi.fn(async () => [{ toolId: outerToolId, label: "Reader", level: "full" as const,
      revision: 1, updatedAt: new Date(0).toISOString() }]),
    setToolPermission: vi.fn()
  } as unknown as BrainService;
  const plugins = new PluginService(catalog, brain);
  await plugins.initialize();
  return { catalog, installed, plugins, outerToolId };
}

function request(launch: Awaited<ReturnType<PluginInteractivePanelService["open"]>>, overrides: Partial<PluginInteractiveToolRequest> = {}): PluginInteractiveToolRequest {
  return {
    format: PLUGIN_INTERACTIVE_MESSAGE_FORMAT,
    version: PLUGIN_INTERACTIVE_MESSAGE_VERSION,
    viewId: launch.viewId,
    channelNonce: launch.channelNonce,
    requestId: "panel-request-one",
    toolId: "reader",
    action: "read",
    arguments: { path: "note.txt" },
    ...overrides
  };
}
function panelReceiptOptions(directory: string) { return { brainRoot: () => directory, checkDisk: async () => undefined, admitTransfer: async () => ({ release: () => undefined }) }; }

describe("interactive plugin panel host", () => {
  const roots: string[] = [];
  afterEach(async () => {
    await Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
  });

  async function root(): Promise<string> {
    const value = await mkdtemp(join(tmpdir(), "omni-plugin-panel-app-"));
    roots.push(value);
    return value;
  }

  it("keeps static panels unchanged and rejects script/body breakouts before hashing", async () => {
    const directory = await root();
    const catalog = new PluginCatalog(join(directory, "catalog"));
    await catalog.initialize();
    const good = await catalog.install(packageValue());
    expect(good.package.manifest.panels[0]?.html).toContain("Static stays inert");
    expect(good.package.manifest.interactivePanels?.[0]?.html).toContain("<canvas");
    const earlierStaticOnly = packageValue();
    delete earlierStaticOnly.manifest.interactivePanels;
    const canonicalStaticOnly = canonicalPluginPackage(earlierStaticOnly);
    expect(canonicalStaticOnly.package.manifest).not.toHaveProperty("interactivePanels");
    expect(canonicalStaticOnly.json).not.toContain("interactivePanels");
    expect(pluginSemanticLearningExperience(canonicalStaticOnly.package, canonicalStaticOnly.sha256).content)
      .not.toContain("interactivePanels");

    const scriptBreakout = packageValue();
    scriptBreakout.manifest.interactivePanels![0]!.script = "document.body.textContent='x';</script><script>parent.postMessage('escape','*')";
    await expect(catalog.install(scriptBreakout)).rejects.toThrow(/closing script/i);
    const malformedClose = packageValue();
    malformedClose.manifest.interactivePanels![0]!.script = "const x = '</script data-x>';";
    await expect(catalog.install(malformedClose)).rejects.toThrow(/closing script/i);
    for (const html of ["<svg/onload=alert(1)>", "<img/src=https://example.test/leak>",
      "</script><script>bad()</script>", "<iframe srcdoc='<script>bad()</script>'></iframe>"]) {
      const bodyBreakout = packageValue();
      bodyBreakout.manifest.interactivePanels![0]!.html = html;
      await expect(catalog.install(bodyBreakout)).rejects.toThrow(/local controls and canvases/i);
    }
    const undeclared = packageValue();
    undeclared.manifest.interactivePanels![0]!.tools[0]!.actions = ["delete"];
    await expect(catalog.install(undeclared)).rejects.toThrow(/undeclared plugin tool action/i);
  });

  it("serves one CSP-hashed opaque document and blocks descendant resources/navigation", async () => {
    const directory = await root();
    const { installed, plugins } = await enabledFixture(directory);
    const executor = {
      execute: vi.fn(async (): Promise<ToolExecutionResult> => ({ id: "execution", toolId: "plugin.example.canvas.reader",
        action: "read", state: "complete", startedAt: new Date(0).toISOString(), finishedAt: new Date(1).toISOString(), output: { ok: true } })),
      cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    expect(parsePluginInteractivePanelUrl(launch.url)).toBe(launch.viewId);
    const response = await service.document(launch.url);
    expect(response.status).toBe(200);
    const csp = response.headers.get("content-security-policy")!;
    expect(csp).toContain("script-src 'sha256-");
    expect(csp).toContain("sandbox allow-scripts");
    expect(csp).not.toContain("unsafe-eval");
    expect(csp).not.toContain("allow-same-origin");
    const html = await response.text();
    expect(html).toContain("&lt;/title&gt;");
    expect(html).toContain("Object.defineProperty(window,\"omniPlugin\"");
    expect(html).not.toContain("</title><script>bad");

    const studio = "file:///studio/index.html";
    const parent = { url: studio };
    expect(allowPluginInteractivePanelNavigation({ url: launch.url, isMainFrame: false,
      frame: { url: "about:blank", parent }, initiator: parent }, studio)).toBe(true);
    expect(allowPluginInteractivePanelNavigation({ url: launch.url, isMainFrame: false,
      frame: { url: launch.url, parent }, initiator: parent }, studio)).toBe(true);
    expect(allowPluginInteractivePanelNavigation({ url: "https://example.test", isMainFrame: false,
      frame: { url: launch.url, parent }, initiator: { url: launch.url, parent } }, studio)).toBe(false);
    expect(allowPluginInteractivePanelRequest({ url: launch.url, resourceType: "subFrame",
      frame: { url: "about:blank", parent } })).toBe(true);
    expect(allowPluginInteractivePanelRequest({ url: launch.url, resourceType: "subFrame",
      frame: { url: launch.url, parent } })).toBe(true);
    expect(allowPluginInteractivePanelRequest({ url: launch.url, resourceType: "subFrame",
      frame: { url: "about:blank", parent: { url: launch.url, parent } } })).toBe(false);
    expect(allowPluginInteractivePanelRequest({ url: "https://example.test/leak", resourceType: "xhr",
      frame: { url: launch.url, parent } })).toBe(false);
    expect(pluginInteractivePanelRequestHeaders({ Cookie: "secret", Authorization: "secret", Origin: studio, Accept: "text/html" }))
      .toEqual({ Accept: "text/html" });
    expect(pluginInteractivePanelResponseHeaders({ "content-security-policy": [csp], "set-cookie": ["secret=1"] })["Content-Security-Policy"])
      .toEqual([csp]);
    await service.dispose();
  });

  it("enforces main-side request replay identity and never exposes an Ask token to the child", async () => {
    const directory = await root();
    const { installed, plugins, outerToolId } = await enabledFixture(directory);
    let calls = 0;
    const executor = {
      execute: vi.fn(async (invocation): Promise<ToolExecutionResult> => {
        calls += 1;
        if (calls === 1) return {
          id: "ask-execution", toolId: invocation.toolId, action: invocation.action,
          state: "approval-required", startedAt: new Date(0).toISOString(),
          approvalToken: "main-only-approval-token", output: {
            kind: "plugin-nested-operation", pluginId: "example.canvas", pluginName: "Canvas helper",
            contentSha256: installed.contentSha256,
            outer: { toolId: outerToolId, action: "read" },
            resolved: { toolId: "system.files", action: "read", arguments: { path: "note.txt" } },
            permissionSources: [{ toolId: outerToolId, level: "full", inherited: false },
              { toolId: "system.files", level: "ask", inherited: true }]
          }
        };
        expect(invocation.approvalToken).toBe("main-only-approval-token");
        return { id: "complete-execution", toolId: invocation.toolId, action: invocation.action,
          state: "complete", dispatchStarted: true, startedAt: new Date(0).toISOString(), finishedAt: new Date(1).toISOString(), output: { content: "real result" } };
      }),
      cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    const first = await service.request(request(launch));
    expect(first).toMatchObject({ state: "approval-required", review: { resolved: { toolId: "system.files", action: "read" } } });
    expect(JSON.stringify(first)).not.toContain("main-only-approval-token");
    await expect(service.request({ ...request(launch), approvalToken: "main-only-approval-token" } as unknown as PluginInteractiveToolRequest))
      .rejects.toThrow(/unsupported fields/i);
    await expect(service.request(request(launch, { arguments: { path: "other.txt" } })))
      .rejects.toThrow(/replayed with different operands/i);
    await expect(service.review({ viewId: launch.viewId, channelNonce: launch.channelNonce,
      requestId: first.requestId, reviewId: "wrong-review", decision: "approve" }))
      .rejects.toThrow(/stale|another request/i);
    const exactReview = { viewId: launch.viewId, channelNonce: launch.channelNonce,
      requestId: first.requestId, reviewId: first.reviewId!, decision: "approve" as const };
    const completed = await service.review(exactReview);
    expect(completed).toMatchObject({ state: "complete", dispatchStarted: true, output: { content: "real result" } });
    await expect(service.review(exactReview)).resolves.toEqual(completed);
    expect(JSON.stringify(completed)).not.toContain("main-only-approval-token");
    expect((await service.request(request(launch))).state).toBe("complete");
    expect(executor.execute).toHaveBeenCalledTimes(2);
    await service.dispose();
  });

  it("atomically reserves one main request identity after concurrent authority reads", async () => {
    const directory = await root();
    const { installed, plugins } = await enabledFixture(directory);
    const executor = {
      execute: vi.fn(async (invocation): Promise<ToolExecutionResult> => ({
        id: "one-execution", toolId: invocation.toolId, action: invocation.action,
        state: "complete", dispatchStarted: true, startedAt: new Date(0).toISOString(),
        finishedAt: new Date(1).toISOString(), output: { path: invocation.arguments.path }
      })),
      cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    const originalInstance = plugins.instance.bind(plugins);
    const entered = deferred(), release = deferred();
    let waiting = 0;
    vi.spyOn(plugins, "instance").mockImplementation(async (...args) => {
      waiting += 1;
      if (waiting === 3) entered.resolve();
      await release.promise;
      return originalInstance(...args);
    });
    const first = service.request(request(launch));
    const exactReplay = service.request(request(launch));
    const changedReplay = service.request(request(launch, { arguments: { path: "changed.txt" } }))
      .then((value) => value as unknown, (error) => error as unknown);
    await entered.promise;
    release.resolve();
    const [firstReceipt, replayReceipt, changed] = await Promise.all([first, exactReplay, changedReplay]);
    expect(replayReceipt).toEqual(firstReceipt);
    expect(changed).toBeInstanceOf(Error);
    expect(String(changed)).toMatch(/replayed with different operands/i);
    expect(executor.execute).toHaveBeenCalledTimes(1);
    await service.dispose();
  });

  it("rejects authority that finishes after its view was closed", async () => {
    const directory = await root();
    const { installed, plugins } = await enabledFixture(directory);
    const executor = {
      execute: vi.fn(async (): Promise<ToolExecutionResult> => ({ id: "must-not-run", toolId: "plugin.example.canvas.reader",
        action: "read", state: "complete", startedAt: new Date(0).toISOString() })),
      cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    const originalInstance = plugins.instance.bind(plugins);
    const entered = deferred(), release = deferred();
    vi.spyOn(plugins, "instance").mockImplementationOnce(async (...args) => {
      entered.resolve();
      await release.promise;
      return originalInstance(...args);
    });
    const pending = service.request(request(launch)).then((value) => value as unknown, (error) => error as unknown);
    await entered.promise;
    await service.close("brain-one", launch.viewId, launch.channelNonce);
    release.resolve();
    const result = await pending;
    expect(result).toBeInstanceOf(Error);
    expect(String(result)).toMatch(/closed/i);
    expect(executor.execute).not.toHaveBeenCalled();
    await service.dispose();
  });

  it("closes fresh handles and retained approvals when package access is revoked", async () => {
    const directory = await root();
    const { catalog, installed, plugins } = await enabledFixture(directory);
    const executor = {
      execute: vi.fn(async (invocation): Promise<ToolExecutionResult> => ({
        id: "approval", toolId: invocation.toolId, action: invocation.action, state: "approval-required",
        startedAt: new Date(0).toISOString(), approvalToken: "private-token", output: {
          kind: "plugin-nested-operation", pluginId: "example.canvas", pluginName: "Canvas helper",
          contentSha256: installed.contentSha256, outer: { toolId: invocation.toolId, action: invocation.action },
          resolved: { toolId: "system.files", action: "read", arguments: invocation.arguments },
          permissionSources: [{ toolId: invocation.toolId, level: "full", inherited: false },
            { toolId: "system.files", level: "ask", inherited: true }]
        }
      })),
      cancelAndWait: vi.fn(async () => 1), invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    await service.request(request(launch));
    const entry = (await catalog.list())[0]!;
    await plugins.setEnabled(entry.pluginId, false, entry.activeContentSha256, entry.revision);
    await vi.waitFor(() => expect(executor.invalidateApproval).toHaveBeenCalledWith("brain-one", "private-token"));
    expect((await service.document(launch.url)).status).toBe(410);
    await expect(service.request(request(launch, { requestId: "after-revoke" }))).rejects.toThrow(/expired|foreign|closed/i);
    await service.dispose();
  });

  it("cancels an in-flight panel request when its per-open handle closes", async () => {
    const directory = await root();
    const { installed, plugins } = await enabledFixture(directory);
    let finish!: (value: ToolExecutionResult) => void;
    const running = new Promise<ToolExecutionResult>((resolve) => { finish = resolve; });
    const executor = {
      execute: vi.fn(() => running),
      cancelAndWait: vi.fn(async () => {
        finish({ id: "cancelled", toolId: "plugin.example.canvas.reader", action: "read", state: "failed",
          startedAt: new Date(0).toISOString(), finishedAt: new Date(1).toISOString(), error: "cancelled before effect" });
        return 1;
      }),
      invalidateApproval: vi.fn(() => true)
    } satisfies PluginPanelToolExecutor;
    const service = new PluginInteractivePanelService(plugins, executor, panelReceiptOptions(directory));
    const launch = await service.open("brain-one", "example.canvas", installed.contentSha256, "canvas");
    const pending = service.request(request(launch));
    await vi.waitFor(() => expect(executor.execute).toHaveBeenCalledTimes(1));
    await service.close("brain-one", launch.viewId, launch.channelNonce);
    await expect(pending).resolves.toMatchObject({ state: "failed", error: "cancelled before effect" });
    expect(executor.cancelAndWait).toHaveBeenCalledWith("brain-one", expect.stringMatching(/^plugin-panel-/));
    await service.dispose();
  });

  it("routes panel calls through real nested ToolExecutor policy instead of direct dispatch", async () => {
    const directory = await root();
    const repository = new BrainRepository(join(directory, "brains"));
    await repository.initialize();
    const brain = await repository.create({ ...DEFAULT_CONFIG, name: "Interactive policy", idleCognition: false });
    const service = new BrainService(repository, {
      request: vi.fn(async () => ({})), requestStream: vi.fn(async () => ({ text: "", trace: { id: "unused" } }))
    } as unknown as EngineSupervisor);
    const plugins = new PluginService(new PluginCatalog(join(directory, "real-plugins")), service);
    await plugins.initialize();
    const installed = await plugins.installPackage(packageValue());
    let entry = (await plugins.list())[0]!;
    entry = await plugins.setEnabled(entry.pluginId, true, entry.activeContentSha256, entry.revision);
    const outerToolId = pluginToolId(entry.pluginId, "reader");
    await plugins.setInstanceAccess({ brainId: brain.id, pluginId: entry.pluginId, contentSha256: installed.contentSha256,
      enabled: true, capabilities: [{ toolId: "system.files", actions: ["read"], enabled: true }],
      toolPermissions: [{ toolId: outerToolId, level: "full" }], requestId: "interactive-real-policy" });
    await service.setToolPermission(brain.id, "system.files", "off", "interactive-nested-off");
    const executor = new ToolExecutor(service, { generate: vi.fn() } as unknown as RuntimeJobManager,
      undefined, undefined, undefined, undefined, plugins);
    vi.spyOn(executor as unknown as { audit(...args: unknown[]): Promise<void> }, "audit").mockResolvedValue(undefined);
    const panels = new PluginInteractivePanelService(plugins, executor, { ...panelReceiptOptions(directory), brainRoot: id => repository.brainDirectory(id) });
    const launch = await panels.open(brain.id, entry.pluginId, installed.contentSha256, "canvas");
    const target = join(repository.brainDirectory(brain.id), "note.txt");
    await writeFile(target, "must not be read while nested Off");
    const result = await panels.request(request(launch, { arguments: { path: target } }));
    expect(result).toMatchObject({ state: "failed", error: expect.stringMatching(/nested .* disabled/i) });
    expect(result.dispatchStarted).toBeUndefined();
    await panels.dispose();
  });
});
