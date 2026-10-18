import { createHash } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ToolExecutor } from "../src/main/toolExecutor";
import type { BrainService, RuntimeJobManager } from "../src/main/brainService";
import type { LearningCoordinator } from "../src/main/learningCoordinator";
import type { ToolExecutionContext } from "../src/main/learningExecutionContext";
import { makeWebLearningPolicy } from "../src/main/webLearningPolicy";
import type { ToolPermissionLevel } from "../src/shared/types";

vi.mock("node:dns/promises", () => ({ lookup: vi.fn(async () => [{ address: "93.184.216.34", family: 4 }]) }));

describe("host-bound learning tools without a neural model", () => {
  let root: string, executor: ToolExecutor, level: ToolPermissionLevel, context: ToolExecutionContext;
  let fetcher: ReturnType<typeof vi.fn>;
  beforeEach(async () => {
    root = await mkdtemp(join(tmpdir(), "omni-learning-tools-")); await mkdir(join(root, "brain"));
    level = "auto";
    const service = { repository: { brainDirectory: () => join(root, "brain") },
      listToolPermissions: async () => ["web.fetch", "web.search", "data.learn", "data.fact-check"].map(toolId => ({ toolId, level, updatedAt: "2026-10-10T00:00:00.000Z" })),
      authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined) } as unknown as BrainService;
    executor = new ToolExecutor(service, {} as RuntimeJobManager);
    vi.spyOn(executor as never, "audit").mockImplementation(async () => undefined);
    context = { actionEventId: "host-event", learning: { brainId: "brain", taskId: "task", claimId: "claim", turnId: "owned-turn",
      policy: async () => makeWebLearningPolicy({ allowedSites: [{ domain: "example.org" }] }, "Use only this site", 1),
      observe: vi.fn(async () => undefined) } };
    fetcher = vi.fn(); vi.stubGlobal("fetch", fetcher);
  });
  afterEach(async () => { vi.unstubAllGlobals(); vi.restoreAllMocks(); await rm(root, { recursive: true, force: true }); });
  const response = (url: string, body: string, status = 200, headers: Record<string, string> = {}) => {
    const value = new Response(body, { status, headers }); Object.defineProperty(value, "url", { value: url }); return value;
  };
  it("checks Off/Ask before dispatch and retains a stable controller action identity", async () => {
    const start = vi.fn(async () => ({ format: "omni-learning-task", id: "task", state: "queued", traversalComplete: false }));
    executor.setLearningCoordinator({ start } as unknown as LearningCoordinator);
    const invocation = { brainId: "brain", toolId: "data.learn", action: "start", arguments: { inputs: [{ kind: "text", text: "The full selected passage." }] } };
    level = "off"; expect((await executor.execute(invocation, undefined, "turn", undefined, context)).state).toBe("failed"); expect(start).not.toHaveBeenCalled();
    level = "ask"; const pending = await executor.execute(invocation, undefined, "turn", undefined, context);
    expect(pending.state).toBe("approval-required"); expect(start).not.toHaveBeenCalled();
    const done = await executor.execute({ ...invocation, approvalToken: pending.approvalToken }, undefined, "approved-event", undefined, context);
    expect(done.state).toBe("complete"); expect(start).toHaveBeenCalledWith(expect.objectContaining({ requestId: "host-event" }), "task");
  });
  it("checks the live scope on a delayed approval and blocks an out-of-scope redirect before its request", async () => {
    const invocation = { brainId: "brain", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org/page" } };
    level = "ask"; const pending = await executor.execute(invocation, undefined, "turn", undefined, context);
    context.learning!.policy = async () => makeWebLearningPolicy({ allowedSites: [{ domain: "elsewhere.org" }] }, "Changed scope", 2);
    const denied = await executor.execute({ ...invocation, approvalToken: pending.approvalToken }, undefined, "event", undefined, context);
    expect(denied.error).toMatch(/outside/); expect(fetcher).not.toHaveBeenCalled();
    level = "auto"; context.learning!.policy = async () => makeWebLearningPolicy({ allowedSites: [{ domain: "example.org" }] }, "", 3);
    fetcher.mockImplementation(async (url: URL) => response(url.href, "", 302, { location: "https://elsewhere.org/escape" }));
    const redirected = await executor.execute(invocation, undefined, "turn", undefined, context);
    expect(redirected.error).toMatch(/outside/); expect(fetcher).toHaveBeenCalledTimes(1);
    expect(context.learning!.observe).not.toHaveBeenCalled();
  });
  it("rejects unknown publication dates and captures only a successfully admitted actual fetch", async () => {
    context.learning!.policy = async () => makeWebLearningPolicy({ dates: { from: "2025-01-01", through: "2025-12-31" } }, "", 1);
    let body = "<p>An undated page.</p>";
    fetcher.mockImplementation(async (url: URL) => response(url.href, body, 200, { "content-type": "text/html" }));
    const invocation = { brainId: "brain", toolId: "web.fetch", action: "fetch", arguments: { url: "https://example.org/page" } };
    expect((await executor.execute(invocation, undefined, "turn", undefined, context)).error).toMatch(/unknown-date/);
    expect(context.learning!.observe).not.toHaveBeenCalled();
    body = '<meta property="article:published_time" content="2025-05-01"><p>A dated page.</p>';
    const actual = await executor.execute(invocation, undefined, "turn", undefined, context);
    expect(actual.state).toBe("complete"); expect(actual.output).toMatchObject({ sha256: createHash("sha256").update(body).digest("hex"), content: body });
    expect(context.learning!.observe).toHaveBeenCalledWith(invocation, expect.objectContaining({ id: actual.id, state: "complete" }));
  });
  it("keeps date-unverified search hits as navigation only and excludes forbidden sites", async () => {
    context.learning!.policy = async () => makeWebLearningPolicy({ allowedSites: [{ domain: "example.org" }], dates: { from: "2025-01-01" } }, "", 1);
    fetcher.mockImplementation(async (url: URL) => response(url.href, '<rss><channel><item><title>Must not be learned as dated</title><link>https://example.org/a</link><description>Undated search snippet.</description></item><item><title>Excluded</title><link>https://elsewhere.org/b</link></item></channel></rss>'));
    const actual = await executor.execute({ brainId: "brain", toolId: "web.search", action: "search", arguments: { query: "selected claim" } }, undefined, "turn", undefined, context);
    expect(actual.output).toMatchObject({ excludedResults: 1, results: [{ url: "https://example.org/a", contentWithheld: "publication-date-unverified" }] });
    expect(JSON.stringify(actual.output)).not.toContain("Undated search snippet");
  });
  it("rejects model-written claim substitution rather than recording another task's finding", async () => {
    const review = vi.fn(); executor.setLearningCoordinator({ review } as unknown as LearningCoordinator);
    const actual = await executor.execute({ brainId: "brain", toolId: "data.fact-check", action: "record", arguments: {
      taskId: "another-task", claimId: "claim", conclusion: "undetermined", explanation: "No evidence", evidenceIds: []
    } }, undefined, "turn", undefined, context);
    expect(actual.error).toMatch(/controller-owned/); expect(review).not.toHaveBeenCalled();
  });
});
