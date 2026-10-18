import { cp, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  HtmlArtifactService,
  allowHtmlArtifactNavigation,
  allowHtmlArtifactRequest,
  htmlArtifactRequestHeaders,
  htmlArtifactResponseHeaders,
  parseHtmlArtifactUrl,
  type HtmlArtifactToolExecutor
} from "../src/main/htmlArtifactService";
import { rekeyHtmlArtifactWorkspace } from "../src/main/htmlArtifactPortability";
import { HtmlArtifactStore } from "../src/main/htmlArtifactStore";
import { HTML_ARTIFACT_MESSAGE_FORMAT, type HtmlArtifactToolRequest } from "../src/shared/htmlArtifacts";
import type { ToolExecutionResult } from "../src/shared/types";

function content() {
  return { title: "Chart", html: "<button id=\"run\">Run</button><canvas id=\"plot\"></canvas>",
    css: "canvas{width:100%}", script: "document.querySelector('#run').addEventListener('click',()=>void omniArtifact.call('system.files','read',{path:'note.txt'}));",
    tools: [{ toolId: "system.files", actions: ["read"] }], height: 360 };
}

function openRequest(artifact: Awaited<ReturnType<HtmlArtifactStore["create"]>>["artifact"]) {
  return { brainId: artifact.brainId, originalBrainId: artifact.originalBrainId,
    artifactId: artifact.artifactId, sourceActionId: artifact.sourceActionId,
    contentSha256: artifact.contentSha256 };
}

describe("chat HTML artifact host", () => {
  const roots: string[] = [];
  afterEach(async () => Promise.all(roots.splice(0).map((root) => rm(root, { recursive: true, force: true }))));
  async function root(name: string) { const value = await mkdtemp(join(tmpdir(), `omni-${name}-`)); roots.push(value); return value; }
  const repository = (directory: string) => ({ brainDirectory: (brainId: string) => join(directory, brainId) });

  it("serves only a fresh opaque hash-CSP document and denies descendant network/navigation", async () => {
    const directory = await root("html-host"), brainRoot = join(directory, "brain-one");
    const created = await new HtmlArtifactStore(brainRoot, "brain-one").create("action-one", content());
    const executor = { execute: vi.fn(), cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true) } as unknown as HtmlArtifactToolExecutor;
    const service = new HtmlArtifactService(repository(directory), executor);
    const launch = await service.open(openRequest(created.artifact));
    expect(parseHtmlArtifactUrl(launch.url)).toBe(launch.viewId);
    expect(launch).toMatchObject({ scriptEnabled: true, height: 360 });
    const response = await service.document(launch.url), csp = response.headers.get("content-security-policy")!;
    expect(csp).toContain("script-src 'sha256-"); expect(csp).toContain("sandbox allow-scripts");
    expect(csp).not.toContain("unsafe-eval"); expect(csp).not.toContain("allow-same-origin");
    expect(await response.text()).toContain("Object.defineProperty(window,\"omniArtifact\"");
    const studio = "file:///studio/index.html", parent = { url: studio };
    expect(allowHtmlArtifactNavigation({ url: launch.url, isMainFrame: false, frame: { url: "about:blank", parent }, initiator: parent }, studio)).toBe(true);
    expect(allowHtmlArtifactNavigation({ url: "https://leak.invalid", isMainFrame: false, frame: { url: launch.url, parent }, initiator: { url: launch.url, parent } }, studio)).toBe(false);
    expect(allowHtmlArtifactRequest({ url: launch.url, resourceType: "subFrame", frame: { url: "about:blank", parent } })).toBe(true);
    expect(allowHtmlArtifactRequest({ url: "https://leak.invalid", resourceType: "xhr", frame: { url: launch.url, parent } })).toBe(false);
    expect(htmlArtifactRequestHeaders({ Cookie: "x", Authorization: "y", Origin: studio, Accept: "text/html" })).toEqual({ Accept: "text/html" });
    expect(htmlArtifactResponseHeaders({ "content-security-policy": [csp], "set-cookie": ["x=1"] })["Content-Security-Policy"]).toEqual([csp]);
    await service.dispose();
  });

  it("routes only declared calls through ToolExecutor and withholds Ask tokens from the view", async () => {
    const directory = await root("html-host-call"), brainRoot = join(directory, "brain-one");
    const created = await new HtmlArtifactStore(brainRoot, "brain-one").create("action-one", content());
    let calls = 0;
    const executor = {
      execute: vi.fn(async (invocation): Promise<ToolExecutionResult> => {
        calls += 1;
        if (calls === 1) return { id: "ask", toolId: invocation.toolId, action: invocation.action,
          state: "approval-required", startedAt: new Date(0).toISOString(), approvalToken: "main-only" };
        expect(invocation.approvalToken).toBe("main-only");
        return { id: "done", toolId: invocation.toolId, action: invocation.action, state: "complete",
          dispatchStarted: true, startedAt: new Date(0).toISOString(), finishedAt: new Date(1).toISOString(), output: { content: "actual" } };
      }), cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true)
    } satisfies HtmlArtifactToolExecutor;
    const service = new HtmlArtifactService(repository(directory), executor), launch = await service.open(openRequest(created.artifact));
    const request: HtmlArtifactToolRequest = { format: HTML_ARTIFACT_MESSAGE_FORMAT, version: 1,
      viewId: launch.viewId, channelNonce: launch.channelNonce, requestId: "request-one",
      toolId: "system.files", action: "read", arguments: { path: "note.txt" } };
    const ask = await service.request(request);
    expect(ask).toMatchObject({ state: "approval-required", review: { resolved: { toolId: "system.files", action: "read" } } });
    expect(JSON.stringify(ask)).not.toContain("main-only");
    await expect(service.request({ ...request, requestId: "undeclared", action: "write" })).rejects.toThrow(/did not declare/i);
    const completed = await service.review({ viewId: launch.viewId, channelNonce: launch.channelNonce,
      requestId: request.requestId, reviewId: ask.reviewId!, decision: "approve" });
    expect(completed).toMatchObject({ state: "complete", dispatchStarted: true, output: { content: "actual" } });
    await expect(service.review({ viewId: launch.viewId, channelNonce: launch.channelNonce,
      requestId: request.requestId, reviewId: ask.reviewId!, decision: "approve" })).resolves.toEqual(completed);
    expect(executor.execute).toHaveBeenCalledTimes(2);
    await service.dispose();
  });

  it("opens copied artifacts presentation-only until exact current-owner activation", async () => {
    const directory = await root("html-host-dormant"), source = join(directory, "source"), target = join(directory, "target");
    const created = await new HtmlArtifactStore(source, "source-brain").create("action-one", content());
    await cp(join(source, "html-artifacts"), join(target, "html-artifacts"), { recursive: true });
    await rekeyHtmlArtifactWorkspace(target, "source-brain", "target-brain");
    const executor = { execute: vi.fn(), cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true) } as unknown as HtmlArtifactToolExecutor;
    const service = new HtmlArtifactService({ brainDirectory: () => target }, executor);
    const request = { brainId: "target-brain", originalBrainId: created.artifact.originalBrainId,
      artifactId: created.artifact.artifactId, sourceActionId: created.artifact.sourceActionId,
      contentSha256: created.artifact.contentSha256 }, dormant = await service.open(request);
    expect(dormant).toMatchObject({ scriptEnabled: false, artifact: { operationalState: "dormant" } });
    const html = await (await service.document(dormant.url)).text();
    expect(html).not.toContain("omniArtifact"); expect(html).not.toContain("<script>");
    const enabled = await service.activate({ ...request, expectedRevision: dormant.artifact.revision, acknowledged: true });
    expect(enabled).toMatchObject({ operationalState: "enabled" });
    expect((await service.document(dormant.url)).status).toBe(410);
    expect(await service.open(request)).toMatchObject({ scriptEnabled: true, artifact: { revision: enabled.revision } });
    await service.dispose();
  });
  it("makes expired cleanup idempotent but rejects foreign live ownership", async () => {
    const directory = await root("html-host-close-race"), brainRoot = join(directory, "brain-one");
    const created = await new HtmlArtifactStore(brainRoot, "brain-one").create("action-one", content());
    const executor = { execute: vi.fn(), cancelAndWait: vi.fn(async () => 0), invalidateApproval: vi.fn(() => true) } satisfies HtmlArtifactToolExecutor;
    const service = new HtmlArtifactService(repository(directory), executor), launch = await service.open(openRequest(created.artifact));
    await expect(service.close("other-brain", launch.viewId, launch.channelNonce)).rejects.toThrow(/foreign live/);
    await expect(service.close("brain-one", launch.viewId, "0".repeat(64))).rejects.toThrow(/foreign live/);
    expect((await service.document(launch.url)).status).toBe(200);
    await service.close("brain-one", launch.viewId, launch.channelNonce);
    await expect(service.close("brain-one", launch.viewId, launch.channelNonce)).resolves.toBeUndefined();
    await service.dispose();
    await expect(service.close("brain-one", launch.viewId, launch.channelNonce)).resolves.toBeUndefined();
    expect((await service.document(launch.url)).status).toBe(410);
    expect(executor.execute).not.toHaveBeenCalled();
  });
});
