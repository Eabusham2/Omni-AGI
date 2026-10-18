import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginInteractivePanelService } from "../src/main/pluginInteractivePanels";
import { PANEL_RECEIPT_PATH, PluginPanelReceiptStore } from "../src/main/pluginPanelReceiptStore";
import { mergePanelReceiptWorkspace, validatePanelReceiptWorkspace } from "../src/main/pluginPanelReceiptPortability";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import type { PluginService } from "../src/main/pluginService";
import type { PluginInteractiveToolRequest } from "../src/shared/pluginInteractivePanels";
import type { ToolExecutionResult, ToolInvocation } from "../src/shared/types";

const roots: string[] = [], services: PluginInteractivePanelService[] = [];
afterEach(async () => { for (const service of services.splice(0)) await service.dispose(); for (const path of roots.splice(0)) await rm(path, { recursive: true, force: true }); });
async function root() { const path = await mkdtemp(join(tmpdir(), "omni-panel-replay-")); roots.push(path); return path; }
function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>(done => { resolve = done; }); return { promise, resolve }; }
const result = (output: unknown = { exact: "inert fake result" }): ToolExecutionResult => ({ id: "actual-fake-execution", toolId: "plugin.fixture.panel.reader", action: "read", state: "complete", startedAt: "2026-10-15T12:00:00.000Z", finishedAt: "2026-10-15T12:00:01.000Z", output });
async function fixture(behavior?: (invocation: ToolInvocation) => Promise<ToolExecutionResult>) {
  const directory = await root(), hash = "a".repeat(64); let diskPressure = false, memoryPressure = false, diskBlock: Promise<void> | undefined; const leases = new Set<object>(), writeChecks: number[] = [], invalidated: string[] = [];
  const plugin = { list: async () => [{ pluginId: "fixture.panel", enabled: true, activeContentSha256: hash, revision: 1 }], instance: async () => ({ enabled: true, contentSha256: hash, revision: 2 }), package: async () => ({ manifest: { interactivePanels: [{ id: "canvas", title: "Fixture panel", html: "<p>Owned fixture</p>", script: "void 0;", tools: [{ toolId: "reader", actions: ["read"] }] }] } }), onEvent: () => () => undefined } as unknown as PluginService;
  const execute = vi.fn(behavior ?? (async invocation => result({ n: invocation.arguments.n }))), cancel = vi.fn(async () => 0);
  const service = new PluginInteractivePanelService(plugin, { execute, cancelAndWait: cancel, invalidateApproval: (_brain, token) => { invalidated.push(token); return true; } }, {
    brainRoot: () => directory, checkDisk: async (_brain, bytes) => { writeChecks.push(bytes); if (diskBlock) await diskBlock; if (diskPressure) throw new Error("Actual configured disk reserve pressure"); },
    admitTransfer: async () => { if (memoryPressure) throw new Error("Actual family RAM pressure"); const lease = {}; leases.add(lease); return { release: () => { leases.delete(lease); } }; } }); services.push(service);
  const launch = await service.open("owner", "fixture.panel", hash, "canvas"), request = (id: string, n = 1): PluginInteractiveToolRequest => ({ format: "omni-plugin-panel-message", version: 1, viewId: launch.viewId, channelNonce: launch.channelNonce, requestId: id, toolId: "reader", action: "read", arguments: { n } });
  return { service, directory, launch, request, execute, cancel, leases, writeChecks, invalidated, disk: (value: boolean) => { diskPressure = value; }, memory: (value: boolean) => { memoryPressure = value; }, block: (value?: Promise<void>) => { diskBlock = value; } };
}
async function settle(predicate: () => boolean) { for (let i = 0; i < 500; i++) { if (predicate()) return; await new Promise(resolve => setImmediate(resolve)); } throw new Error("Owned replay fixture did not settle."); }

describe("scripted panel durable replay without a lifetime request quota", () => {
  it("crosses 256 calls in one unchanged view, evicts live terminal records, and replays exact old IDs only", async () => {
    const f = await fixture(); let first: unknown; for (let i = 0; i < 270; i++) { const receipt = await f.service.request(f.request(`request-${i}`, i)); if (i === 0) first = receipt; expect(receipt.output).toEqual({ n: i }); }
    await new Promise(resolve => setImmediate(resolve)); expect(f.execute).toHaveBeenCalledTimes(270); const views = (f.service as unknown as { views: Map<string, { requests: Map<string, unknown>; hostRequestIds: Set<string> }> }).views, live = views.get(f.launch.viewId)!;
    expect(live.requests.size).toBe(0); expect(live.hostRequestIds.size).toBe(0); expect(f.leases.size).toBe(0); expect(await f.service.request(f.request("request-0", 0))).toEqual(first); expect(f.execute).toHaveBeenCalledTimes(270);
    await expect(f.service.request(f.request("request-0", 999))).rejects.toThrow(/different operands/); expect(f.execute).toHaveBeenCalledTimes(270); const page = await f.service.history("owner", { maxBytes: 4096 }); expect(page.total).toBeGreaterThan(1000); expect(page.nextSequence).toBeDefined(); expect(page.records.length).toBeLessThan(page.total); expect(f.writeChecks.every(bytes => bytes >= 65536)).toBe(true);
  });

  it("distinguishes pressure before any effect from a lost-ACK effect and never repeats the latter", async () => {
    const f = await fixture(); f.memory(true); await expect(f.service.request(f.request("ram"))).rejects.toThrow(/RAM pressure/); expect(f.execute).not.toHaveBeenCalled(); f.memory(false); f.disk(true); await expect(f.service.request(f.request("before-write"))).rejects.toThrow(/disk reserve/); expect(f.execute).not.toHaveBeenCalled();
    f.disk(false); f.execute.mockImplementation(async () => { f.disk(true); return result({ actual: "effect attempted once" }); }); await expect(f.service.request(f.request("lost-ack"))).rejects.toThrow(/disk reserve/); await new Promise(resolve => setImmediate(resolve)); f.disk(false);
    const first = await f.service.request(f.request("lost-ack")), second = await f.service.request(f.request("lost-ack")); expect(first.state).toBe("uncertain"); expect(second).toEqual(first); expect(f.execute).toHaveBeenCalledTimes(1); expect(f.leases.size).toBe(0);
  });

  it("cancels while reserve admission is pending without later dispatch or an in-memory receipt leak", async () => {
    const f = await fixture(), gate = deferred<void>(); f.block(gate.promise); const pending = f.service.request(f.request("cancel-before-write")); const rejected = expect(pending).rejects.toThrow();
    await settle(() => f.writeChecks.length >= 3); f.block(); await f.service.close("owner", f.launch.viewId, f.launch.channelNonce); gate.resolve(); await rejected; expect(f.execute).not.toHaveBeenCalled(); expect(f.leases.size).toBe(0); await expect(f.service.request(f.request("late"))).rejects.toThrow(/expired/);
  });

  it("keeps Ask authority private, releases it on terminal review, and replays durable reviewed receipts", async () => {
    const f = await fixture(async invocation => invocation.approvalToken ? result({ reviewed: true }) : { ...result(), state: "approval-required", approvalToken: "HOST_PRIVATE_PANEL_TOKEN", output: { kind: "plugin-nested-operation", pluginId: "fixture.panel", contentSha256: "a".repeat(64), outer: { toolId: invocation.toolId, action: invocation.action }, resolved: { toolId: "system.files", action: "read", arguments: invocation.arguments }, permissionSources: [] } });
    const initial = await f.service.request(f.request("ask")); expect(initial.state).toBe("approval-required"); expect(f.leases.size).toBe(1); const review = { viewId: f.launch.viewId, channelNonce: f.launch.channelNonce, requestId: "ask", reviewId: initial.reviewId!, decision: "approve" as const }, completed = await f.service.review(review);
    expect(completed.output).toEqual({ reviewed: true }); expect(await f.service.review(review)).toEqual(completed); expect(await f.service.request(f.request("ask"))).toEqual(completed); expect(f.execute).toHaveBeenCalledTimes(2); expect(f.leases.size).toBe(0);
    const bytes = await readFile(join(f.directory, PANEL_RECEIPT_PATH)); expect(bytes.includes(Buffer.from("HOST_PRIVATE_PANEL_TOKEN"))).toBe(false); expect(bytes.includes(Buffer.from(f.launch.channelNonce))).toBe(false);
  });

  it("preserves full over-frame actual output in paged history rather than cropping it", async () => {
    const output = "owned raw output🙂".repeat(140000), f = await fixture(async () => result(output)), receipt = await f.service.request(f.request("large")); expect(receipt.output).toBeUndefined(); expect(receipt.error).toContain("bounded panel");
    let cursor: number | undefined, descriptor: { sequence: number; sha256: string; bytes?: number } | undefined; do { const page = await f.service.history("owner", { maxBytes: 4096, afterSequence: cursor }); descriptor = page.records.find(row => row.kind === "actual-host-result"); cursor = page.nextSequence; } while (!descriptor && cursor !== undefined); expect(descriptor?.bytes).toBeGreaterThan(2 * 1024 * 1024);
    const parts: Buffer[] = []; let offset = 0; do { const page = await f.service.historyBytes("owner", descriptor!.sequence, offset, 65536); parts.push(Buffer.from(page.text, "base64")); if (page.nextOffset === undefined) break; offset = page.nextOffset; } while (true);
    const text = Buffer.concat(parts).toString("utf8"); expect(createHash("sha256").update(text).digest("hex")).toBe(descriptor?.sha256); expect(JSON.parse(text).output).toBe(output); expect(f.execute).toHaveBeenCalledTimes(1);
  });

  it("exports/copies/merges all history dormant while leaving original bytes and grants untouched", async () => {
    const f = await fixture(); await f.service.request(f.request("source-fact", 42)); await f.service.close("owner", f.launch.viewId, f.launch.channelNonce); const sourcePath = join(f.directory, PANEL_RECEIPT_PATH), before = createHash("sha256").update(await readFile(sourcePath)).digest("hex"), packaged = await root(), target = await root(), merged = await root();
    const snapshot = await snapshotLearningWorkspace(f.directory, packaged, "owner"); expect(snapshot).toBeDefined(); await adoptLearningWorkspace(snapshot!, target, "copy"); await validatePanelReceiptWorkspace(target, "copy"); const copied = (await PluginPanelReceiptStore.open(target, "copy"))!;
    try { expect(copied.db.prepare("SELECT state FROM views WHERE id=?").get(f.launch.viewId)?.state).toBe("dormant"); expect(copied.get(f.launch.viewId, "source-fact")?.receipt?.output).toEqual({ n: 42 }); } finally { copied.close(); }
    await mergePanelReceiptWorkspace(target, merged, "copy"); await validatePanelReceiptWorkspace(merged, "copy"); const store = (await PluginPanelReceiptStore.open(merged, "copy"))!; try { expect(Number(store.db.prepare("SELECT COUNT(*) AS n FROM archives").get()?.n)).toBeGreaterThan(0); expect(store.db.prepare("SELECT COUNT(*) AS n FROM views").get()?.n).toBe(0); } finally { store.close(); }
    expect(createHash("sha256").update(await readFile(sourcePath)).digest("hex")).toBe(before); await expect(f.service.request(f.request("source-fact", 42))).rejects.toThrow(/expired/);
  });

  it("keeps an already-closing writer in the trusted owner boundary until its terminal custody flush finishes", async () => {
    const f = await fixture(), gate = deferred<void>(); f.block(gate.promise); let finished = false; const closing = f.service.close("owner", f.launch.viewId, f.launch.channelNonce), owner = f.service.closeOwner("owner").then(() => { finished = true; });
    await new Promise(resolve => setImmediate(resolve)); expect(finished).toBe(false); f.block(); gate.resolve(); await closing; await owner; expect(finished).toBe(true);
  });
});
