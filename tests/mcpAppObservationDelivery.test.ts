import { createHash, randomUUID } from "node:crypto";
import { mkdir, mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { McpAppArtifactStore } from "../src/main/mcpAppArtifacts";
import { McpAppObservationDelivery } from "../src/main/mcpAppObservationDelivery";
import { nativeAppObservationFrontierSha256 } from "../src/main/nativeAppObservation";
import type { NativeAppObservationHead, NativeAppObservationInput, NativeAppObservationResult } from "../src/shared/nativeAppObservation";
import type { McpAppTaskData } from "../src/shared/mcpAppTaskData";
import type { ChatResult } from "../src/shared/types";
import { cleanChatInput } from "../src/shared/chatInput";
import type { NativeAppChatInput } from "../src/shared/chatInputProvenance";
import { EngineRequestError } from "../src/main/engineSupervisor";

const hash = (text: string) => createHash("sha256").update(text).digest("hex");
const artifactId = "12345678-1234-4567-8123-123456789012";
describe("App observation host delivery (synthetic native gateway, not model competence)", () => {
  let root: string, directory: string;
  const deliveries: McpAppObservationDelivery[] = [];
  beforeEach(async () => { root = await mkdtemp(join(tmpdir(), "omni-app-observation-delivery-")); directory = join(root, "brain"); await mkdir(directory); });
  afterEach(async () => { deliveries.splice(0).forEach(value => value.close()); await rm(root, { recursive: true, force: true }); });
  async function fixture() {
    const directory = join(root, randomUUID()); await mkdir(directory);
    const store = new McpAppArtifactStore(directory, "brain");
    const saved = await store.capture(artifactId, { origin: { brainId: "brain", serverId: "fixture", serverLabel: "Synthetic App", endpoint: "https://fixture.invalid/mcp", connectionEpoch: 1 },
      tool: { id: "mcp.brain.fixture.open", remoteName: "open", actions: ["call"], inputSchema: { type: "object" }, metadata: { ui: { resourceUri: "ui://fixture/view" } } }, arguments: {}, result: { content: [{ type: "text", text: "original fixture output" }] } });
    let loaded = true, sourceEnabled = true, head: NativeAppObservationHead | null = null, learns = 0, loseAck = false, releaseAfterCommit = false;
    const events: unknown[] = [];
    const frontier = vi.fn(async () => ({ format: "omni-native-app-observation-frontier" as const, version: 1 as const, brainId: "brain", frontierSha256: nativeAppObservationFrontierSha256(head), head }));
    const admit = vi.fn(async (input: NativeAppObservationInput): Promise<NativeAppObservationResult> => {
      const record = JSON.parse(input.recordJson), data = JSON.parse(input.dataJson);
      if (head?.recordSha256 === input.recordSha256) return { ...await frontier(), head, duplicate: true };
      expect(input.expectedFrontierSha256).toBe(nativeAppObservationFrontierSha256(head));
      const text = data.method === "ui/message" ? cleanChatInput((Array.isArray(data.params.content) ? data.params.content : [data.params.content]).map((block: { text: string }) => block.text).join("\n")) : JSON.stringify(data.params);
      const observed = input.disposition === "observe";
      head = { format: "omni-native-app-observation-head", version: 1, streamId: record.streamId, ordinal: record.ordinal,
        recordSha256: input.recordSha256, dataSha256: input.dataSha256, previousSha256: record.previousSha256,
        baseFrontierSha256: input.expectedFrontierSha256, disposition: input.disposition,
        state: observed ? "observed" : "advanced-without-observation", committedAt: new Date().toISOString(),
        learning: observed ? { kind: "experience", source: "mcp-app", sourceLabel: record.source.serverLabel, steps: 0,
          experienceCountBefore: learns, experienceCountAfter: ++learns, assemblyId: `synthetic-assembly-${learns}`, afterimageId: "", textSha256: hash(text), textBytes: Buffer.byteLength(text) } : null };
      if (loseAck) { loseAck = false; throw new Error("Synthetic lost native ACK after commit"); }
      return { ...await frontier(), head, duplicate: false, ...(releaseAfterCommit ? { reloadRequired: true as const, nativeOwnerReleased: true as const } : {}) };
    });
    const followup = vi.fn(async (_id: string, input: string, turnId: string, proof: NativeAppChatInput): Promise<ChatResult> => {
      const humanMessage = { id: randomUUID(), role: "human" as const, content: input, turnId, createdAt: new Date().toISOString(), inputProvenance: proof.provenance };
      // Real controller streams can reconcile before send() resolves.
      await store.reconcileFollowup(humanMessage);
      return { humanMessage } as ChatResult;
    });
    const ownerReleased = vi.fn(async () => { loaded = false; });
    const delivery = new McpAppObservationDelivery({ store: () => store, directory: () => directory,
      requireLoaded: () => { if (!loaded) throw new Error("Explicit Load required"); }, requireSource: () => { if (!sourceEnabled) throw new Error("MCP source disabled"); },
      withWriter: async (_id, work) => work(), frontier, admit, followup, ownerReleased, publish: event => { events.push(event); } });
    deliveries.push(delivery);
    const queue = (data: McpAppTaskData) => store.queueObservation(artifactId, saved.sha256, saved.capture.source, randomUUID(), data, 4096);
    return { store, delivery, queue, admit, frontier, followup, events, ownerReleased, releaseAfterCommit: () => { releaseAfterCommit = true; }, loaded: (value: boolean) => { loaded = value; }, source: (value: boolean) => { sourceEnabled = value; }, loseAck: () => { loseAck = true; }, learns: () => learns };
  }
  const context = (selected: string): McpAppTaskData => ({ method: "ui/update-model-context", params: { structuredContent: { selected } } });
  const message: McpAppTaskData = { method: "ui/message", params: { role: "user", content: { type: "text", text: "A reviewed current App task" } } };
  it("does not autoload or call native before explicit Load, then observes ordinary data exactly once", async () => {
    const f = await fixture(); await f.queue(context("current")); f.loaded(false); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.frontier).not.toHaveBeenCalled(); expect(f.admit).not.toHaveBeenCalled();
    f.loaded(true); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.learns()).toBe(1); expect(f.followup).not.toHaveBeenCalled();
    expect((await f.store.summary(artifactId))?.observationQueue).toMatchObject({ pending: 0, admitted: 1, latest: { neuralAdmission: "admitted" } });
  });
  it("advances superseded context explicitly without pretending it was learned", async () => {
    const f = await fixture(); await f.queue(context("old")); await f.queue(context("latest"));
    f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.admit.mock.calls.map(args => args[0].disposition)).toEqual(["superseded", "observe"]); expect(f.learns()).toBe(1);
    expect((await f.store.summary(artifactId))?.observationQueue).toMatchObject({ pending: 0, admitted: 1, superseded: 1 });
  });
  it("binds a genuine App-attributed follow-up to the exact reviewed body with one durable dispatch", async () => {
    const f = await fixture(); await f.queue(message); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.followup).toHaveBeenCalledOnce();
    const args = f.followup.mock.calls[0]!; expect(args[1]).toBe("A reviewed current App task");
    expect(args[3].provenance).toMatchObject({ kind: "mcp-app", sourceBrainId: "brain", serverLabel: "Synthetic App", resourceUri: "ui://fixture/view", inputSha256: hash(args[1]) });
    f.delivery.notify("brain"); await f.delivery.settle("brain"); expect(f.followup).toHaveBeenCalledOnce(); expect(f.learns()).toBe(1);
  });
  it("retries an exact lost-ACK native head without learning twice", async () => {
    const f = await fixture(); await f.queue(context("current")); f.loseAck(); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.learns()).toBe(1); expect((await f.store.summary(artifactId))?.observationQueue?.pending).toBe(1);
    f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.learns()).toBe(1); expect(f.admit).toHaveBeenCalledTimes(2); expect((await f.store.summary(artifactId))?.observationQueue?.admitted).toBe(1);
  });
  it("keeps a safely preempted read claim pending and retries only on a later readiness event", async () => {
    const f = await fixture(); await f.queue(context("read phase pending"));
    f.admit.mockRejectedValueOnce(new EngineRequestError("Synthetic preemption", -32800, { brainId: "brain", nativeAppObservation: true,
      cancelled: true, safeBoundary: true, learningApplied: false, nativeOwnerReleased: false }));
    f.delivery.notify("brain"); await f.delivery.settle("brain"); expect(f.learns()).toBe(0); expect(f.events).toEqual([]);
    expect((await f.store.summary(artifactId))?.observationQueue?.pending).toBe(1);
    f.delivery.retryWaiting(); await f.delivery.settle("brain"); expect(f.learns()).toBe(1); expect(f.ownerReleased).not.toHaveBeenCalled();
  });
  it("does not replay an uncertain follow-up or let source revocation admit data", async () => {
    const f = await fixture(); await f.queue(message); f.followup.mockRejectedValueOnce(new Error("Synthetic uncertain generation outcome"));
    f.delivery.notify("brain"); await f.delivery.settle("brain"); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.followup).toHaveBeenCalledOnce(); expect(f.learns()).toBe(1);
    const other = await fixture(); await other.queue(context("disabled")); other.source(false); other.delivery.notify("brain"); await other.delivery.settle("brain");
    expect(other.admit).not.toHaveBeenCalled(); expect(other.learns()).toBe(0);
  });
  it("records a committed native admission but requires explicit Load after owner cleanup releases RAM", async () => {
    const f = await fixture(); await f.queue(message); f.releaseAfterCommit(); f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.ownerReleased).toHaveBeenCalledOnce(); expect(f.followup).not.toHaveBeenCalled();
    expect((await f.store.summary(artifactId))?.observationQueue).toMatchObject({ admitted: 1, latest: { neuralAdmission: "admitted" } });
  });
  it("invalidates a confirmed released owner even when durable audit archival fails after commit", async () => {
    const f = await fixture(); await f.queue(context("committed before disk failure")); f.releaseAfterCommit();
    vi.spyOn(f.store, "archiveObservationAudit").mockResolvedValueOnce(undefined).mockRejectedValueOnce(new Error("Synthetic archive disk failure"));
    f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.learns()).toBe(1); expect(f.ownerReleased).toHaveBeenCalledOnce(); expect(f.followup).not.toHaveBeenCalled();
    expect((await f.store.summary(artifactId))?.observationQueue?.admitted).toBe(0);
    expect(f.events.at(-1)).toMatchObject({ error: "Synthetic archive disk failure" });
  });
  it("invalidates only the exact worker-confirmed failed owner, never a foreign release claim", async () => {
    const f = await fixture(); await f.queue(context("not committed")); f.admit.mockRejectedValueOnce(new EngineRequestError("Reload required", -32031, {
      nativeAppObservation: true, reloadRequired: true, nativeOwnerReleased: true, committed: false, brainId: "brain" }));
    f.delivery.notify("brain"); await f.delivery.settle("brain"); expect(f.ownerReleased).toHaveBeenCalledOnce(); expect(f.learns()).toBe(0);
    const other = await fixture(); await other.queue(context("foreign")); other.admit.mockRejectedValueOnce(new EngineRequestError("Forged other owner", -32031, {
      nativeAppObservation: true, reloadRequired: true, nativeOwnerReleased: true, committed: false, brainId: "unrelated" }));
    other.delivery.notify("brain"); await other.delivery.settle("brain"); expect(other.ownerReleased).not.toHaveBeenCalled();
  });
  it("preserves an unknown save-outcome claim and requires explicit Load instead of retrying or claiming nothing applied", async () => {
    const f = await fixture(); await f.queue(context("unknown publication outcome"));
    f.admit.mockRejectedValueOnce(new EngineRequestError("Synthetic publication outcome unknown; explicit Load required", -32031, {
      nativeAppObservation: true, reloadRequired: true, nativeOwnerReleased: true, committed: null, learningApplied: null, brainId: "brain" }));
    f.delivery.notify("brain"); await f.delivery.settle("brain");
    expect(f.ownerReleased).toHaveBeenCalledOnce(); expect((await f.store.summary(artifactId))?.observationQueue?.pending).toBe(1);
    f.delivery.retryWaiting(); await f.delivery.settle("brain"); expect(f.admit).toHaveBeenCalledOnce();
    expect(f.events.at(-1)).toMatchObject({ error: "Synthetic publication outcome unknown; explicit Load required" });
    expect(JSON.stringify(f.events)).not.toContain("nothing applied");
  });
});
