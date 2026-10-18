import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpInteractionService } from "../src/main/mcpInteractions";
import { mcpElicitation, mcpFormValues, McpHostRequestError } from "../src/main/mcpInteractionProtocol";
import { nativeMcpSamplingInputSha256 } from "../src/main/nativeMcpSampling";
import type { ApprovedNativeMcpSamplingInput, NativeMcpSamplingResult } from "../src/shared/nativeMcpSampling";
import type { McpRequestOrigin } from "../src/shared/mcpInteractions";
import { McpRequestDeadline } from "../src/main/mcpRequestDeadline";

const origin: McpRequestOrigin = { brainId: "first", serverId: "server", serverLabel: "Fixture server", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 2, originatingTurnId: "owned-turn", actionEventId: "owned-action" };
const url = { mode: "url", message: "Connect a service", url: "https://fixture.invalid/connect?elicitationId=reviewed", elicitationId: "reviewed" };

describe("explicit private MCP reviews", () => {
  let directory: string;
  const services: McpInteractionService[] = [];
  beforeEach(async () => { directory = await mkdtemp(join(tmpdir(), "omni-mcp-review-")); });
  afterEach(async () => { vi.useRealTimers(); services.splice(0).forEach(service => service.close()); await rm(directory, { recursive: true, force: true }); });
  function fixture() {
    const requireAccess = vi.fn(), admit = vi.fn(async (_bytes: number) => undefined), openBrowser = vi.fn(async (_url: string) => undefined);
    const sampleNative = vi.fn(async (input: ApprovedNativeMcpSamplingInput): Promise<NativeMcpSamplingResult> => ({ role: "assistant", content: { type: "text", text: "Actual private fixture output" },
      model: `OmniCortex/${input.brainId}`, stopReason: "endTurn", inputSha256: input.inputSha256, requestId: input.requestId, brainId: input.brainId, privateWorkspace: true, parametersMutated: false }));
    const service = new McpInteractionService({ directory, requireAccess, admit, openBrowser, sampleNative }); services.push(service);
    return { service, requireAccess, admit, openBrowser, sampleNative };
  }

  it("reviews flat typed fields and only sends validated values after explicit acceptance", async () => {
    const f = fixture(); await f.service.initialize();
    const pending = f.service.elicit(origin, { mode: "form", message: "Choose your display options", requestedSchema: {
      type: "object", properties: { name: { type: "string", minLength: 2 }, count: { type: "integer", minimum: 1, maximum: 4 }, enabled: { type: "boolean" },
        colors: { type: "array", minItems: 2, maxItems: 2, items: { anyOf: [{ const: "red", title: "Red" }, { const: "blue", title: "Blue" }] } } }, required: ["name", "count", "colors"] }
    }, new AbortController().signal);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    await expect(f.service.respond({ brainId: "other", id: record.id, action: "accept", values: {} })).rejects.toThrow("that instance");
    await expect(f.service.respond({ brainId: "first", id: record.id, action: "accept", values: { name: "A", count: 5, colors: ["red"] } })).rejects.toThrow("Review the value");
    expect((await f.service.list("first"))[0]?.status).toBe("pending");
    const values = { name: "ECHO-PERSONAL-7421", count: 3, enabled: false, colors: ["red", "blue"] };
    await f.service.respond({ brainId: "first", id: record.id, action: "accept", values });
    expect(await pending).toEqual({ action: "accept", content: values });
    expect(await f.service.list("first")).toEqual([]);
    expect(await readFile(join(directory, `${record.id}.json`), "utf8")).not.toContain("ECHO-PERSONAL-7421");
    expect(f.openBrowser).not.toHaveBeenCalled(); expect(f.sampleNative).not.toHaveBeenCalled();
  });

  it("rejects sensitive in-band fields, nested schemas and invalid formats instead of collecting credentials", async () => {
    expect(() => mcpElicitation({ mode: "form", message: "Password", requestedSchema: { type: "object", properties: { access_token: { type: "string" } } } })).toThrow("Sensitive credentials");
    expect(() => mcpElicitation({ mode: "form", message: "Nested", requestedSchema: { type: "object", properties: { account: { type: "object" } } } })).toThrow("primitive");
    expect(() => mcpElicitation({ mode: "url", message: "Unsafe", url: "javascript:alert(1)", elicitationId: "x" })).toThrow("HTTPS");
    expect(() => mcpElicitation({ mode: "url", message: "Unsafe", url: "https://user:password@fixture.invalid/", elicitationId: "x" })).toThrow("credentials");
    await expect(mcpFormValues({ type: "object", properties: { name: { type: "string", pattern: "^[A-Z]+$" } } }, { name: "lower" })).rejects.toThrow("Review the value");
    expect(await mcpFormValues({ type: "object", properties: { name: { type: "string", pattern: "^[A-Z]+$" } } }, { name: "UPPER" })).toEqual({ name: "UPPER" });
  });

  it("never prefetches or opens a URL before consent, preserves the full target, and does not claim out-of-band completion", async () => {
    const f = fixture(); await f.service.initialize();
    const pending = f.service.elicit(origin, url, new AbortController().signal);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    expect(f.openBrowser).not.toHaveBeenCalled(); expect(record).toMatchObject({ kind: "url", request: url });
    await f.service.respond({ brainId: "first", id: record.id, action: "accept" });
    expect(await pending).toEqual({ action: "accept" }); expect(f.openBrowser).toHaveBeenCalledExactlyOnceWith(url.url);
    expect(JSON.parse(await readFile(join(directory, `${record.id}.json`), "utf8"))).toMatchObject({ status: "accepted", outOfBandStatus: "unknown", navigation: { state: "opened" } });
    await f.service.completeElicitation({ ...origin, connectionEpoch: 3 }, url.elicitationId);
    expect(JSON.parse(await readFile(join(directory, `${record.id}.json`), "utf8"))).toMatchObject({ outOfBandStatus: "unknown" });
    await f.service.completeElicitation(origin, url.elicitationId);
    expect(JSON.parse(await readFile(join(directory, `${record.id}.json`), "utf8"))).toMatchObject({ outOfBandStatus: "complete" });
  });

  it("does not reopen a URL if its final receipt failed after the browser opened", async () => {
    const f = fixture(); await f.service.initialize();
    const pending = f.service.elicit(origin, url, new AbortController().signal);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    f.admit.mockImplementationOnce(async () => undefined).mockRejectedValueOnce(new Error("Fixture final receipt disk pressure"));
    await expect(f.service.respond({ brainId: "first", id: record.id, action: "accept" })).rejects.toThrow("disk pressure");
    expect(f.openBrowser).toHaveBeenCalledOnce();
    await f.service.respond({ brainId: "first", id: record.id, action: "accept" });
    expect(await pending).toEqual({ action: "accept" }); expect(f.openBrowser).toHaveBeenCalledOnce();
  });

  it("treats a browser failure or explicit decline/cancel as non-acceptance with no automatic retry", async () => {
    const f = fixture(); await f.service.initialize(); f.openBrowser.mockRejectedValueOnce(new Error("Fixture browser unavailable"));
    const pending = f.service.elicit(origin, url, new AbortController().signal);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    expect((await f.service.respond({ brainId: "first", id: record.id, action: "accept" })).status).toBe("cancelled");
    expect(await pending).toEqual({ action: "cancel" }); expect(f.openBrowser).toHaveBeenCalledOnce();
    await expect(f.service.respond({ brainId: "first", id: record.id, action: "accept" })).rejects.toThrow("no longer");
  });

  it("cancels an outstanding review on its actual connection signal without assuming an answer", async () => {
    const f = fixture(); await f.service.initialize(); const controller = new AbortController();
    const pending = f.service.elicit(origin, url, controller.signal).catch(error => error);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    controller.abort(new Error("Actual connection ended"));
    expect(await pending).toMatchObject({ message: "Actual connection ended" });
    expect(await f.service.list("first")).toEqual([]); expect(f.openBrowser).not.toHaveBeenCalled();
    expect(JSON.parse(await readFile(join(directory, `${record.id}.json`), "utf8"))).toMatchObject({ status: "cancelled" });
  });

  it("requires separate input/output sampling reviews, binds edited inputs and omits the server persona", async () => {
    const f = fixture(); await f.service.initialize();
    const pending = f.service.sample(origin, { messages: [{ role: "user", content: { type: "text", text: "Original task" } }], maxTokens: 8,
      systemPrompt: "Pretend to be a different person", modelPreferences: { hints: [{ name: "foreign-provider" }] } }, new AbortController().signal);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const input = (await f.service.list("first"))[0]!;
    expect(input).toMatchObject({ kind: "sampling-input", systemPromptOmitted: "Pretend to be a different person" }); expect(f.sampleNative).not.toHaveBeenCalled();
    await f.service.respond({ brainId: "first", id: input.id, action: "accept", sampling: { messages: [{ role: "user", content: { type: "text", text: "Edited task" } }], maxTokens: 4 } });
    await vi.waitFor(async () => expect((await f.service.list("first"))[0]?.kind).toBe("sampling-output"));
    const approved = f.sampleNative.mock.calls[0]![0];
    expect(approved).toMatchObject({ brainId: "first", originatingTurnId: "owned-turn", maxTokens: 4, messages: [{ content: { text: "Edited task" } }] });
    expect(approved.inputSha256).toBe(nativeMcpSamplingInputSha256(approved));
    expect(JSON.stringify(approved)).not.toContain("Pretend"); expect(JSON.stringify(approved)).not.toContain("foreign-provider");
    const output = (await f.service.list("first"))[0]!;
    await f.service.respond({ brainId: "first", id: output.id, action: "accept", editedText: "Human reviewed output" });
    expect(await pending).toMatchObject({ model: "OmniCortex/first", content: { type: "text", text: "Human reviewed output" }, _meta: { "omni/review": { userEdited: true, sameBrain: true, systemPromptOmitted: true } } });
  });

  it("refuses unsupported sampling contexts/tools and wrong-brain receipts instead of substituting a model", async () => {
    const f = fixture(); await f.service.initialize();
    const request = { messages: [{ role: "user", content: { type: "text", text: "Task" } }], maxTokens: 4 };
    await expect(f.service.sample(origin, { ...request, includeContext: "allServers" }, new AbortController().signal)).rejects.toThrow("hidden cross-server");
    await expect(f.service.sample(origin, { ...request, tools: [] }, new AbortController().signal)).rejects.toThrow("sampling.tools");
    f.sampleNative.mockImplementationOnce(async input => ({ role: "assistant", content: { type: "text", text: "Wrong brain" }, model: "Foreign/other", stopReason: "endTurn", inputSha256: input.inputSha256, requestId: input.requestId, brainId: "other", privateWorkspace: true, parametersMutated: false }));
    const pending = f.service.sample(origin, request, new AbortController().signal).catch(error => error);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    const record = (await f.service.list("first"))[0]!;
    await f.service.respond({ brainId: "first", id: record.id, action: "accept" });
    expect(await pending).toBeInstanceOf(McpHostRequestError); expect(await f.service.list("first")).toEqual([]);
  });

  it("does not replay pending reviews when the host restarts", async () => {
    const f = fixture(); await f.service.initialize();
    const pending = f.service.elicit(origin, url, new AbortController().signal).catch(error => error);
    await vi.waitFor(async () => expect(await f.service.list("first")).toHaveLength(1));
    f.service.close(); await pending;
    const restarted = fixture(); await restarted.service.initialize();
    expect(await restarted.service.list("first")).toEqual([]); expect(restarted.openBrowser).not.toHaveBeenCalled(); expect(restarted.sampleNative).not.toHaveBeenCalled();
    const files = (await readdir(directory)).filter(name => name.endsWith(".json"));
    expect(JSON.parse(await readFile(join(directory, files[0]!), "utf8"))).toMatchObject({ status: "cancelled", reason: expect.stringContaining("not replayed") });
  });
  it("reviews inline media metadata without decoding, and text edits cannot substitute or remove media", async () => {
    const f = fixture(); await f.service.initialize();
    const content = [{ type: "text" as const, text: "Review this supplied image" },
      { type: "image" as const, mimeType: "image/png", data: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC7sAAAAASUVORK5CYII=" }];
    // This is a labeled host transport fixture, not a neural competence proof.
    f.sampleNative.mockImplementationOnce(async input => ({ role: "assistant", content: { type: "text", text: "Private fixture output" }, model: `OmniCortex/${input.brainId}`,
      stopReason: "endTurn", inputSha256: input.inputSha256, requestId: input.requestId, brainId: input.brainId, privateWorkspace: true, parametersMutated: false,
      mediaInput: [{ messageIndex: 0, blockIndex: 1, kind: "image", mimeType: "image/png", sourceBytes: 68,
        sourceSha256: createHash("sha256").update(Buffer.from(content[1]!.data!, "base64")).digest("hex"), decodedWindows: 1, sourceFrames: 1, wholeInputTraversed: true, nativeSensory: true }] }));
    const pending = f.service.sample(origin, { messages: [{ role: "user", content }], maxTokens: 32 }, new AbortController().signal);
    await vi.waitFor(async () => expect((await f.service.list("first"))[0]?.kind).toBe("sampling-input"));
    const record = (await f.service.list("first"))[0]!;
    expect(record).toMatchObject({ kind: "sampling-input", media: [{ messageIndex: 0, blockIndex: 1, kind: "image", mimeType: "image/png", decodedBytes: 68, encodedSha256: expect.stringMatching(/^[a-f0-9]{64}$/) }] });
    await expect(f.service.respond({ brainId: "first", id: record.id, action: "accept", sampling: { messages: [{ role: "user", content: { type: "text", text: "No media" } }], maxTokens: 32 } })).rejects.toThrow("cannot replace");
    expect(f.sampleNative).not.toHaveBeenCalled();
    await f.service.respond({ brainId: "first", id: record.id, action: "accept", sampling: { messages: [{ role: "user", content: [{ type: "text", text: "My reviewed image task" }, content[1]!] }], maxTokens: 32 } });
    await vi.waitFor(async () => expect((await f.service.list("first"))[0]?.kind).toBe("sampling-output"));
    expect(f.sampleNative.mock.calls[0]?.[0].messages[0]?.content).toEqual([{ type: "text", text: "My reviewed image task" }, content[1]]);
    const output = (await f.service.list("first"))[0]!; await f.service.respond({ brainId: "first", id: output.id, action: "accept" });
    expect(await pending).toMatchObject({ model: "OmniCortex/first" });
  });
  it("reviews App authority separately without collecting form data or timing out into a grant", async () => {
    const f = fixture(); await f.service.initialize();
    const controller = new AbortController();
    const pending = f.service.confirmApp(origin, { category: "network", message: "Review exact network origins", csp: {
      connectDomains: ["https://api.example"], resourceDomains: [], frameDomains: [], baseUriDomains: [] } }, controller.signal);
    await vi.waitFor(async () => expect((await f.service.list("first"))[0]?.kind).toBe("app-permission"));
    const record = (await f.service.list("first"))[0]!;
    await expect(f.service.respond({ brainId: "first", id: record.id, action: "accept", values: { hidden: "grant" } })).rejects.toThrow("cannot submit hidden");
    expect(await f.service.list("first")).toHaveLength(1); controller.abort();
    await expect(pending).rejects.toBeDefined();
    expect(f.sampleNative).not.toHaveBeenCalled(); expect(f.openBrowser).not.toHaveBeenCalled();
  });
});

describe("MCP request deadline review lease", () => {
  afterEach(() => vi.useRealTimers());
  it("pauses only the network timeout while a real review is held and never creates an approval", async () => {
    vi.useFakeTimers();
    const deadline = new McpRequestDeadline(120_000, () => Date.now());
    await vi.advanceTimersByTimeAsync(20_000); const release = deadline.pause();
    await vi.advanceTimersByTimeAsync(40 * 60_000); expect(deadline.signal.aborted).toBe(false);
    release(); await vi.advanceTimersByTimeAsync(99_999); expect(deadline.signal.aborted).toBe(false);
    await vi.advanceTimersByTimeAsync(1); expect(deadline.signal.aborted).toBe(true);
    deadline.close();
  });
});
