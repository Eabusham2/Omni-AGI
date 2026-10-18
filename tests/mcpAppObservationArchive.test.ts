import { randomUUID } from "node:crypto";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { McpAppArtifactStore, mergeMcpAppArtifacts, validateMcpAppArtifactStore } from "../src/main/mcpAppArtifacts";
import { adoptLearningWorkspace, snapshotLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { nativeAppObservationFrontierSha256, nativeAppObservationRawSha256 } from "../src/main/nativeAppObservation";
import type { NativeAppObservationResult } from "../src/shared/nativeAppObservation";
import { APP_AUDIT_PNG, syntheticAppAudit } from "./fixtures/mcpAppAudit";

describe("brain-owned App observation audit acknowledgement (synthetic native receipt)", () => {
  let root: string, directory: string;
  const artifactId = "12345678-1234-4567-8123-123456789012";
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-app-audit-archive-")); directory = join(root, "source"); await mkdir(directory); });
  afterEach(async () => { await rm(root, { recursive: true, force: true }); });
  async function fixture() {
    const store = new McpAppArtifactStore(directory, "source"), saved = await store.capture(artifactId, {
      origin: { brainId: "source", serverId: "fixture", serverLabel: "Synthetic App", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 },
      tool: { id: "mcp.source.fixture.open", remoteName: "open", actions: ["call"], inputSchema: { type: "object" }, metadata: { ui: { resourceUri: "ui://fixture/view" } } },
      arguments: {}, result: { content: [{ type: "text", text: "original" }] } });
    const data = { method: "ui/update-model-context" as const, params: { content: [{ type: "image" as const, mimeType: "image/png", data: APP_AUDIT_PNG }] } };
    await store.queueObservation(artifactId, saved.sha256, saved.capture.source, randomUUID(), data, 4096);
    const claim = (await store.claimObservation())!; await store.bindObservationFrontier(claim.recordSha256, "0".repeat(64));
    const f = syntheticAppAudit(claim.dataJson, claim.recordSha256, "source"), engine = join(directory, "engine");
    await mkdir(join(engine, "app-observation-audits"), { recursive: true }); await writeFile(join(engine, f.content.audit.path), f.raw);
    const head: NativeAppObservationResult["head"] = { format: "omni-native-app-observation-head", version: 1,
      streamId: claim.record.streamId, ordinal: claim.record.ordinal, recordSha256: claim.recordSha256, dataSha256: claim.record.content.sha256,
      previousSha256: claim.record.previousSha256, baseFrontierSha256: "0".repeat(64), disposition: "observe", state: "observed",
      committedAt: new Date().toISOString(), learning: { kind: "experience", source: "mcp-app", sourceLabel: "Synthetic App",
        steps: 0, experienceCountBefore: 0, experienceCountAfter: 1, assemblyId: "synthetic-only", afterimageId: "",
        textSha256: nativeAppObservationRawSha256(""), textBytes: 0 }, content: f.content };
    const result: NativeAppObservationResult = { format: "omni-native-app-observation-frontier", version: 1, brainId: "source",
      head, frontierSha256: nativeAppObservationFrontierSha256(head), duplicate: false };
    return { store, saved, data, claim, f, engine, result };
  }
  it("requires exact raw archive before ACK and retains the prior archive proof before advancing", async () => {
    const f = await fixture();
    await expect(f.store.acknowledgeObservation(f.claim.recordSha256, f.result)).rejects.toThrow(/durably verified/);
    await expect(f.store.bindObservationArchive(f.claim.recordSha256, f.f.content.audit.sha256)).rejects.toThrow(/durable archive/);
    expect(await f.store.archiveObservationAudit(f.result.head, f.engine)).toBe(f.f.content.audit.sha256);
    await expect(f.store.acknowledgeObservation(f.claim.recordSha256, { ...f.result, reloadRequired: true } as NativeAppObservationResult)).rejects.toThrow(/exact reviewed queued observation/);
    await expect(f.store.acknowledgeObservation(f.claim.recordSha256, { ...f.result, permissionGranted: true } as NativeAppObservationResult)).rejects.toThrow(/exact reviewed queued observation/);
    expect(await f.store.acknowledgeObservation(f.claim.recordSha256, f.result)).toMatchObject({ neuralAdmission: "admitted" });
    await rm(join(f.engine, f.f.content.audit.path));
    // Lost native ACK can recover from the already durable raw archive, not
    // by decoding or learning again; native duplicate is separately tested.
    expect(await f.store.archiveObservationAudit(f.result.head, f.engine)).toBe(f.f.content.audit.sha256);
    await f.store.queueObservation(artifactId, f.saved.sha256, f.saved.capture.source, randomUUID(), f.data, 4096);
    const next = (await f.store.claimObservation())!;
    expect(await f.store.bindObservationArchive(next.recordSha256, f.f.content.audit.sha256)).toBe(f.f.content.audit.sha256);
    expect(await f.store.bindObservationArchive(next.recordSha256, undefined)).toBe(f.f.content.audit.sha256);
    await validateMcpAppArtifactStore(directory, "source");
    await writeFile(join(directory, "mcp-apps", "blobs", `${f.f.content.audit.sha256}.data`), "corrupted raw audit");
    await expect(f.store.archiveObservationAudit(f.result.head, f.engine)).rejects.toThrow(/exact owned regular|checksum/);
  });
  it("copies/merges exact original audits while copied progress remains dormant and no authority is restored", async () => {
    const f = await fixture(); await f.store.archiveObservationAudit(f.result.head, f.engine);
    await f.store.acknowledgeObservation(f.claim.recordSha256, f.result);
    const archive = (await snapshotLearningWorkspace(directory, join(root, "archive"), "source"))!, copy = join(root, "copy"); await mkdir(copy);
    await adoptLearningWorkspace(archive, copy, "copy");
    const copied = new McpAppArtifactStore(copy, "copy"); expect(await copied.claimObservation()).toBeUndefined();
    expect((await copied.summary(artifactId))?.observationQueue).toMatchObject({ pending: 0, dormant: 1, admitted: 0 });
    const merged = join(root, "merged"); await mkdir(merged);
    await mergeMcpAppArtifacts([{ root: directory, brainId: "source" }, { root: copy, brainId: "copy" }], merged, "merged");
    await validateMcpAppArtifactStore(merged, "merged");
    expect(await readFile(join(merged, "mcp-apps", "blobs", `${f.f.content.audit.sha256}.data`), "utf8")).toBe(f.f.raw);
    const db = new DatabaseSync(join(merged, "mcp-apps", "index.sqlite3"), { readOnly: true });
    try { expect(db.prepare("SELECT COUNT(*) AS count FROM observation_audits").get()?.count).toBe(1);
      expect(String(db.prepare("SELECT descriptor_json FROM observation_audits").get()?.descriptor_json)).toContain('"sourceBrainId":"source"');
      expect(db.prepare("SELECT value FROM meta WHERE key='activeObservationStream'").get()).toBeUndefined();
    } finally { db.close(); }
  });
});
