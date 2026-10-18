import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { McpAppArtifactStore, mergeMcpAppArtifacts, rebindMcpAppArtifactOwner, validateMcpAppArtifactStore } from "../src/main/mcpAppArtifacts";
import { projectMcpAppAction } from "../src/main/mcpAppPresentation";
import { adoptFullMergeMcpApps } from "../src/main/fullMergeMcpApps";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import type { McpCompletedAppCall } from "../src/main/mcpApps";
import type { ActionEvent } from "../src/shared/types";

const csp = { connectDomains: [], resourceDomains: [], frameDomains: [], baseUriDomains: [] };
const artifactId = "12345678-1234-4567-8123-123456789012";
const call: McpCompletedAppCall = { origin: { brainId: "source", serverId: "fixture", serverLabel: "Original source", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 27 },
  tool: { id: "mcp.source.fixture.open-12345678", remoteName: "open", actions: ["call"], inputSchema: { type: "object" }, metadata: { ui: { resourceUri: "ui://fixture/view.html" } } },
  arguments: { title: "Original input", userId: "opaque-user" }, result: { content: [{ type: "text", text: "Original output" }], nested: { userId: "opaque-user", fileId: "opaque-file" } } };
describe("inert brain-owned MCP App artifact closure", () => {
  let root: string, source: string;
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-mcp-artifacts-")); source = join(root, "source"); await mkdir(source); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  async function seeded() {
    const store = new McpAppArtifactStore(source, "source"); const saved = await store.capture(artifactId, call);
    const resource = await store.resource(artifactId, saved.sha256, saved.capture.source, saved.capture.resourceUri, "<html><script>untrusted()</script></html>", {}, csp);
    return { store, saved, resource };
  }
  it("persists exact untrusted call/resources/provenance without a runtime token, session, grant or executable file", async () => {
    const f = await seeded(), loaded = await f.store.load(artifactId);
    expect(loaded.input).toEqual(call.arguments); expect(loaded.result).toEqual(call.result);
    expect(await f.store.summary(artifactId)).toMatchObject({ selectedCapture: f.saved.sha256, selectedResource: f.resource, needsMappingReview: false });
    await validateMcpAppArtifactStore(source, "source");
    const db = new DatabaseSync(join(source, "mcp-apps", "index.sqlite3"), { readOnly: true });
    try { const rows = db.prepare("SELECT payload_json FROM records").all().map(row => String(row.payload_json)).join("\n");
      expect(rows).toContain('"untrusted":true'); expect(rows).not.toContain("connectionEpoch"); expect(rows).not.toContain("viewToken"); expect(rows).not.toContain("approvalToken");
    } finally { db.close(); }
  });
  it("copies/rebinds only the owner while preserving original immutable content and lineage", async () => {
    const f = await seeded(), target = join(root, "copy"); await mkdir(target); await cp(join(source, "mcp-apps"), join(target, "mcp-apps"), { recursive: true });
    await rebindMcpAppArtifactOwner(target, "source", "copy"); const copied = new McpAppArtifactStore(target, "copy");
    expect((await copied.load(artifactId)).sha256).toBe(f.saved.sha256);
    expect((await copied.load(artifactId)).capture.source.brainId).toBe("source");
    expect(await copied.projection(artifactId)).toMatchObject({ brainId: "copy", savedArtifact: true, originalSource: { brainId: "source" } });
    await expect(new McpAppArtifactStore(target, "wrong").summary(artifactId)).rejects.toThrow("another owner");
  });
  it("is included in the existing full workspace archive, adopts after source removal, and forks again without live handles", async () => {
    const f = await seeded(); const archive = await snapshotLearningWorkspace(source, join(root, "archive"), "source");
    const target = join(root, "imported"); await mkdir(target); await rm(source, { recursive: true, force: true });
    await adoptLearningWorkspace(archive!, target, "imported");
    const imported = new McpAppArtifactStore(target, "imported"); expect((await imported.load(artifactId)).sha256).toBe(f.saved.sha256);
    const again = await snapshotLearningWorkspace(target, join(root, "again"), "imported"), fork = join(root, "fork"); await mkdir(fork);
    await adoptLearningWorkspace(again!, fork, "fork"); expect((await new McpAppArtifactStore(fork, "fork").load(artifactId)).result).toEqual(call.result);
  });
  it("deduplicates shared ancestry but preserves conflicting resources/captures for explicit mapping review", async () => {
    const f = await seeded(), branch = join(root, "branch"); await mkdir(branch); await cp(join(source, "mcp-apps"), join(branch, "mcp-apps"), { recursive: true });
    await rebindMcpAppArtifactOwner(branch, "source", "branch"); const b = new McpAppArtifactStore(branch, "branch");
    await b.resource(artifactId, f.saved.sha256, { ...f.saved.capture.source, brainId: "branch" }, f.saved.capture.resourceUri, "<html>branch version</html>", {}, csp);
    const merged = join(root, "merged"); await mkdir(merged);
    await mergeMcpAppArtifacts([{ root: source, brainId: "source" }, { root: branch, brainId: "branch" }], merged, "merged");
    const store = new McpAppArtifactStore(merged, "merged"), summary = await store.summary(artifactId);
    expect(summary?.captures).toHaveLength(1); expect(summary?.resources).toHaveLength(2); expect(summary?.needsMappingReview).toBe(true);
    expect((await store.load(artifactId)).result).toEqual(call.result);
    const other = join(root, "other"); await mkdir(other); const o = new McpAppArtifactStore(other, "other");
    await o.capture(artifactId, { ...call, origin: { ...call.origin, brainId: "other" }, result: { content: [{ type: "text", text: "Conflicting real outcome" }] } });
    await mergeMcpAppArtifacts([{ root: other, brainId: "other" }], merged, "merged");
    expect((await store.summary(artifactId))?.captures).toHaveLength(2); expect((await store.summary(artifactId))?.selectedCapture).toBeUndefined();
    await expect(store.load(artifactId)).rejects.toThrow("record identity");
    await store.select(artifactId, f.saved.sha256); expect((await store.load(artifactId)).result).toEqual(call.result);
  });
  it("adds only a verified owner projection and leaves execution.output byte-for-byte unchanged", async () => {
    const f = await seeded(), target = join(root, "copy"); await mkdir(target); await cp(join(source, "mcp-apps"), join(target, "mcp-apps"), { recursive: true }); await rebindMcpAppArtifactOwner(target, "source", "copy");
    const output = { ...call.result, _meta: { "omni/mcpApp": { id: artifactId, brainId: "source", toolId: call.tool.id, untrusted: true } } };
    const event: ActionEvent = { id: randomUUID(), brainId: "copy", createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), state: "complete",
      action: { kind: "tool", source: "brain", toolId: call.tool.id, action: "call", arguments: call.arguments },
      execution: { id: randomUUID(), state: "complete", toolId: call.tool.id, action: "call", startedAt: new Date().toISOString(), finishedAt: new Date().toISOString(), output } };
    const before = JSON.stringify(event.execution!.output), projected = await projectMcpAppAction(target, "copy", event);
    expect(JSON.stringify(projected.execution!.output)).toBe(before); expect(projected.mcpAppView).toMatchObject({ brainId: "copy", artifactSha256: f.saved.sha256, originalSource: { brainId: "source" } });
    const forged = await projectMcpAppAction(target, "copy", { ...event, execution: { ...event.execution!, output: { ...output, nested: { userId: "changed" } } } });
    expect(forged.mcpAppView).toBeUndefined();
  });
  it("full-merges the sealed App closure rather than a racy raw parent-tree copy", async () => {
    const f = await seeded(); await snapshotLearningWorkspace(source, join(source, "portable-learning-workspace"), "source");
    await rm(join(source, "mcp-apps"), { recursive: true, force: true });
    const target = join(root, "merged"); await mkdir(target);
    await adoptFullMergeMcpApps({ targetBrainId: "merged", targetDirectory: target,
      parents: [{ brainId: "source", name: "Source", generationSha256: "a".repeat(64), ancestry: [] }],
      request: {} as never, files: () => [], blobPath: () => { throw new Error("not used"); },
      materializeParent: async () => source, reserve: async () => undefined, checkpoint: async () => undefined });
    const merged = new McpAppArtifactStore(target, "merged");
    expect((await merged.load(artifactId)).sha256).toBe(f.saved.sha256);
    expect((await merged.load(artifactId)).capture.source.brainId).toBe("source");
    expect((await merged.summary(artifactId))?.resources).toHaveLength(1);
  });
  it("rejects corrupt blobs and executable ledger schema instead of claiming a restored artifact", async () => {
    const f = await seeded(); await writeFile(join(source, "mcp-apps", "blobs", `${f.saved.capture.result.sha256}.data`), "corrupt");
    await expect(f.store.load(artifactId)).rejects.toThrow("checksum");
    const db = new DatabaseSync(join(source, "mcp-apps", "index.sqlite3")); db.exec("CREATE TRIGGER unexpected AFTER INSERT ON records BEGIN SELECT 1; END"); db.close();
    await expect(validateMcpAppArtifactStore(source, "source")).rejects.toThrow("executable schema");
  });
  it("retains a reviewed hash-chain queue, supersedes old current context, dedups lost ACKs and backpressures without fake learning", async () => {
    const f = await seeded(), message = { method: "ui/message" as const, params: { role: "user" as const, content: { type: "text" as const, text: "App-proposed current task" } } };
    const first = await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, randomUUID(), message, 4096);
    const context = { method: "ui/update-model-context" as const, params: { structuredContent: { selected: "old", userId: "opaque-user" } } };
    const second = await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, randomUUID(), context, 4096);
    const review = randomUUID(), latest = { ...context, params: { structuredContent: { selected: "new", userId: "opaque-user" } } };
    const third = await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, review, latest, 4096);
    expect([first.ordinal, second.ordinal, third.ordinal]).toEqual([1, 2, 3]); expect(third.streamId).toBe(first.streamId);
    expect(third).toMatchObject({ state: "pending-native-admission", neuralAdmission: "not-yet-admitted" });
    expect(await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, review, latest, 4096)).toEqual(third);
    await expect(f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, review, context, 4096)).rejects.toThrow("changed task");
    await expect(f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, randomUUID(), message, 0)).rejects.toThrow("Nothing was silently dropped");
    expect((await f.store.summary(artifactId))?.observationQueue).toMatchObject({ pending: 2, superseded: 1, latest: third });
    const db = new DatabaseSync(join(source, "mcp-apps", "index.sqlite3"), { readOnly: true });
    try { const record = JSON.parse(String(db.prepare("SELECT payload_json FROM records WHERE record_sha=?").get(third.recordSha256)?.payload_json));
      expect(record.previousSha256).toBe(second.recordSha256); expect(JSON.parse(await f.store.readBlob(record.content))).toEqual(latest);
    } finally { db.close(); }
    await validateMcpAppArtifactStore(source, "source");
  });
  it("makes copied and merged pending App observations dormant instead of transferring approval or replaying them", async () => {
    const f = await seeded(), data = { method: "ui/message" as const, params: { role: "user" as const, content: { type: "text" as const, text: "Reviewed only on source" } } };
    const queued = await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, randomUUID(), data, 4096);
    const archive = await snapshotLearningWorkspace(source, join(root, "archive-queue"), "source"), copy = join(root, "copy-queue"); await mkdir(copy);
    await adoptLearningWorkspace(archive!, copy, "copy"); const copied = new McpAppArtifactStore(copy, "copy");
    expect((await copied.summary(artifactId))?.observationQueue).toMatchObject({ pending: 0, dormant: 1, latest: { ...queued, state: "dormant" } });
    const fresh = await copied.queueObservation(artifactId, f.saved.sha256, { ...f.saved.capture.source, brainId: "copy" }, randomUUID(), data, 4096);
    expect(fresh.ordinal).toBe(1); expect(fresh.streamId).not.toBe(queued.streamId);
    const merged = join(root, "merge-queue"); await mkdir(merged);
    await mergeMcpAppArtifacts([{ root: source, brainId: "source" }, { root: copy, brainId: "copy" }], merged, "merged");
    const result = await new McpAppArtifactStore(merged, "merged").summary(artifactId);
    expect(result?.observationQueue).toMatchObject({ pending: 0, dormant: 2 }); await validateMcpAppArtifactStore(merged, "merged");
  });
});
