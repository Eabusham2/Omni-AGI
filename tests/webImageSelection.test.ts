/** New web-image source edges: tiny private SQLite/files + fake HTTPS transport. */
import { createHash } from "node:crypto";
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, describe, expect, it } from "vitest";
import type { BrainRepository } from "../src/main/brainRepository";
import type { LearningExecutionContext } from "../src/main/learningExecutionContext";
import { TaskAttachmentService } from "../src/main/taskAttachmentService";
import { TaskAttachmentStore } from "../src/main/taskAttachmentStore";
import { rekeyTaskAttachmentWorkspace, validateTaskAttachmentWorkspace } from "../src/main/taskAttachmentPortability";
import { WebImageService, readWebImageActionEvidence, type WebImageServiceDependencies } from "../src/main/webImageService";
import { htmlImageCandidates, searchImageCandidates, WebImageReferenceStore, webImageHash } from "../src/main/webImageReferences";
import { declaredWebImageCapabilityPack } from "../src/main/webImageCapabilityPack";
import { makeWebLearningPolicy } from "../src/main/webLearningPolicy";
import { CompletedActionEvidenceStore } from "../src/main/completedActionEvidence";

const roots: string[] = [], owner = "image-owner", turn = "chat-turn", event = "action-event";
const hash = (value: string | Buffer) => createHash("sha256").update(value).digest("hex");
const bytes = Buffer.alloc(150_001, 20); Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(bytes); // Container-signature fixture, not decoded imagery.
afterEach(async () => { for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true }); });
async function directory(): Promise<string> { const path = await mkdtemp(join(tmpdir(), "omni-web-image-")); roots.push(path); return path; }
async function fixture(options: {
  mime?: string; body?: Buffer; declared?: number; redirected?: string;
  afterFetch?(): void; checkResources?(bytes: number): void; learning?: LearningExecutionContext;
} = {}) {
  const root = await directory(), repository = { get: async (id: string) => ({ id }), brainDirectory: () => root } as unknown as BrainRepository;
  const attachments = new TaskAttachmentService(repository), taskStore = new TaskAttachmentStore(root, owner);
  const network = { calls: 0, cancelled: 0, urls: [] as string[] }, permission = { level: "auto", revision: "1".repeat(64) };
  const payload = options.body ?? bytes;
  const deps: WebImageServiceDependencies = {
    root: async () => root, initializeAttachments: () => taskStore.initialize(), attachments, requireLoaded: () => undefined,
    permission: async () => ({ ...permission }), learningContext: async () => options.learning,
    checkResources: async (_brainId, estimate) => { options.checkResources?.(estimate); }, reserveWrite: async () => undefined,
    fetchPublic: async (url, init, redirects, hooks) => {
      expect(redirects).toBe(5); network.calls++;
      const route = [url, ...(options.redirected ? [new URL(options.redirected)] : [])];
      for (const current of route) {
        init.signal?.throwIfAborted();
        // Injected guard stands in for production safeFetch; no DNS/network is performed here.
        if (current.protocol !== "https:" || current.hostname === "127.0.0.1") throw new Error("fake transport refused private/unsafe destination");
        await hooks.beforeRequest(current); network.urls.push(current.toString());
        hooks.afterResponse(current, new Response(null, { status: current === route.at(-1) ? 200 : 302 }));
      }
      let offset = 0;
      const body = new ReadableStream<Uint8Array>({
        pull(controller) { if (offset >= payload.length) { controller.close(); return; } const end = Math.min(offset + 8192, payload.length); controller.enqueue(payload.subarray(offset, end)); offset = end; },
        cancel() { network.cancelled++; }
      });
      options.afterFetch?.();
      return new Response(body, { headers: { "content-type": options.mime ?? "image/png", "content-length": String(options.declared ?? payload.length) } });
    }
  };
  const service = new WebImageService(deps);
  const refs = await service.register(owner, turn, { kind: "search-response", url: "https://search.example/query", sha256: hash("actual provider response"), fetchedAt: "2026-10-13T15:00:00Z" },
    searchImageCandidates([{ url: "https://publisher.example/page", img_src: "https://images.example/photo.png", thumbnail: "https://images.example/photo.png", title: "Untrusted provider label" }]), new AbortController().signal);
  return { root, service, refs, attachments, taskStore, network, permission, deps };
}

describe("explicit web image reference → exact same-brain sensory source", () => {
  it("discovers only literal attributable references without fetching pixels or trusting executable URLs", async () => {
    const f = await fixture(); expect(f.refs).toHaveLength(1); expect(f.network.calls).toBe(0);
    expect(f.refs[0]?.fetched).toBe(false); expect(f.refs[0]?.source.sha256).toBe(hash("actual provider response"));
    const html = '<img src="/first.png" alt="not instructions &amp; data"><meta property="og:image" content="https://cdn.example/second.jpg"><img src="data:image/png;base64,aGVsbG8="><img src="file:///secret"><link rel="image_src" href="/third.webp">';
    expect([...htmlImageCandidates(html, "https://publisher.example/page")].map(row => row.url)).toEqual([
      "https://publisher.example/first.png", "https://cdn.example/second.jpg", "https://publisher.example/third.webp"
    ]);
    const store = await WebImageReferenceStore.open(f.root, owner, true);
    try { expect(() => store.resolve("other-turn", f.refs[0]!.id)).toThrow(/another task/); expect(() => store.resolve(turn, "f".repeat(64))).toThrow(/actually returned/); }
    finally { store.close(); }
    const pack = declaredWebImageCapabilityPack(); expect(pack.executedHostActions).toBe(false);
    expect(pack.records.some(row => row.action === "inspect-image")).toBe(true);
  });

  it("streams selected bytes into exact CAS custody without changing the original turn's sealed attachment set or pretending perception", async () => {
    const f = await fixture({ redirected: "https://cdn.example/final.png" });
    const original = await f.taskStore.stageBytes("original", { data: Buffer.from("task file"), displayName: "task.txt", mimeType: "text/plain", kind: "file", source: { kind: "local-file", selectedAt: new Date().toISOString() } });
    await f.taskStore.bindToTurn(turn, [original.id]);
    const selected = await f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined);
    expect(selected.download.sha256).toBe(hash(bytes)); expect(selected.download.bytes).toBe(bytes.length);
    expect(selected.download.redirects).toEqual(["https://images.example/photo.png", "https://cdn.example/final.png"]);
    expect(selected.observation).toEqual({ state: "pending-native-perception", pixelsRead: false, parametersChanged: false, rawSourceTextInjected: false });
    expect(JSON.stringify(selected)).not.toContain("dataBase64"); expect(JSON.stringify(selected)).not.toContain(bytes.toString("base64"));
    expect(await f.attachments.handlesForTurn(owner, turn)).toHaveLength(1);
    await expect(f.attachments.resolveForTurn(owner, turn, selected.attachment.attachmentId)).rejects.toThrow(/exact chat task/);
    expect(await readFile(join(f.root, selected.attachment.relativePath))).toEqual(bytes);
    expect(await f.service.validateObservation(owner, turn, selected)).toEqual(selected);
    const retry = await f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined);
    expect(retry.selectionSha256).toBe(selected.selectionSha256); expect(f.network.calls).toBe(1);
    expect((await readdir(f.root)).filter(name => name.startsWith(".web-image-"))).toHaveLength(0);
    await validateTaskAttachmentWorkspace(f.root, owner);
  });

  it("refuses unsupported MIME, false signatures, truncated responses, explicit stop-size overflow and unsafe redirects before pixel observation", async () => {
    for (const options of [
      { mime: "image/svg+xml" }, { body: Buffer.from("<html>not pixels</html>") },
      { declared: bytes.length + 1 }, { redirected: "https://127.0.0.1/private.png" }
    ]) {
      const f = await fixture(options);
      await expect(f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined)).rejects.toThrow();
      expect(await f.attachments.list(owner)).toHaveLength(0);
      expect((await readdir(f.root)).filter(name => name.startsWith(".web-image-"))).toHaveLength(0);
    }
    const f = await fixture();
    await expect(f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined, undefined, 50)).rejects.toThrow(/stop size/);
  });

  it("rechecks exact permission after response and refuses changed learning scope or resource admission without staged learned bytes", async () => {
    let revoke: () => void = () => undefined;
    const f = await fixture({ afterFetch: () => revoke() }); revoke = () => { f.permission.level = "off"; f.permission.revision = "2".repeat(64); };
    await expect(f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined)).rejects.toThrow(/authority changed/);
    expect(await f.attachments.list(owner)).toHaveLength(0); expect(f.network.cancelled).toBeGreaterThan(0);
    const policy = makeWebLearningPolicy({ allowedSites: [{ domain: "publisher.example" }] }, "", 1);
    const context: LearningExecutionContext = { brainId: owner, turnId: turn, taskId: "task", policy: async () => policy, observe: async () => undefined };
    const scoped = await fixture({ learning: context });
    await expect(scoped.service.inspect(owner, turn, event, scoped.refs[0]!.id, new AbortController().signal, async () => undefined, context)).rejects.toThrow(/source scope/);
    let pressure = false;
    const memory = await fixture({ checkResources: () => { if (pressure) throw new Error("current RAM pressure"); } }); pressure = true;
    await expect(memory.service.inspect(owner, turn, event, memory.refs[0]!.id, new AbortController().signal, async () => undefined)).rejects.toThrow(/RAM pressure/);
    expect(memory.network.calls).toBe(0);
  });

  it("honors cancellation and guards the actual evidence/action/selection source rather than a fabricated learned receipt", async () => {
    const f = await fixture(), controller = new AbortController(); controller.abort(new Error("explicit web image Stop"));
    await expect(f.service.inspect(owner, turn, event, f.refs[0]!.id, controller.signal, async () => undefined)).rejects.toThrow(/explicit web image Stop/);
    expect(f.network.calls).toBe(0);
    const selected = await f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined);
    const evidenceId = "e".repeat(64), executionId = "12345678-1234-4234-8234-123456789abc", path = join(f.root, "source-evidence.jsonl");
    const body = { format: "omni-completed-action-result-v1", originBrainId: owner, evidenceId, executionId, actionEventId: event, chatTurnId: turn,
      toolId: "web.fetch", action: "inspect-image", executionState: "complete", arguments: { referenceId: selected.reference.id }, outputPresent: true, output: selected };
    await writeFile(path, JSON.stringify(body));
    expect(await readWebImageActionEvidence(path, owner, evidenceId, executionId, hash(JSON.stringify(body)), async () => undefined)).toEqual(selected);
    const wrong = { ...body, actionEventId: "another-action" }; await writeFile(path, JSON.stringify(wrong));
    await expect(readWebImageActionEvidence(path, owner, evidenceId, executionId, hash(JSON.stringify(wrong)), async () => undefined)).rejects.toThrow(/operands/);
    f.permission.revision = "3".repeat(64);
    await expect(f.service.validateObservation(owner, turn, selected)).rejects.toThrow(/renewed current authority/);
  });

  it("preserves copied exact source provenance while retiring image-reference authority instead of inheriting current grants", async () => {
    const f = await fixture(), selected = await f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined);
    const copy = await directory(); await cp(join(f.root, "attachments"), join(copy, "attachments"), { recursive: true });
    await rekeyTaskAttachmentWorkspace(copy, owner, "child");
    const store = await WebImageReferenceStore.open(copy, "child", true);
    try { store.validate(); expect(() => store.resolve(turn, selected.reference.id)).toThrow(/dormant/); } finally { store.close(); }
    const db = new DatabaseSync(join(copy, "attachments", "registry.sqlite3"), { readOnly: true });
    try { const source = JSON.parse(String(db.prepare("SELECT source_json FROM attachments WHERE id=?").get(selected.attachment.attachmentId)?.source_json)); expect(source.webImage).toEqual(selected.download); }
    finally { db.close(); }
    expect(hash(await readFile(join(f.root, selected.attachment.relativePath)))).toBe(selected.download.sha256);
    expect((await f.service.validateObservation(owner, turn, selected)).selectionSha256).toBe(selected.selectionSha256);
  });

  it("rejects executable schema objects attached to reference custody before any new image selection", async () => {
    const f = await fixture(), db = new DatabaseSync(join(f.root, "attachments", "registry.sqlite3"));
    try { db.exec("CREATE TRIGGER hostile_ref AFTER INSERT ON web_image_references BEGIN DELETE FROM attachments; END"); } finally { db.close(); }
    await expect(WebImageReferenceStore.open(f.root, owner)).rejects.toThrow(/schema objects/);
    expect(f.network.calls).toBe(0);
  });

  it("refuses changed pixels when a reference already binds a previously fetched exact image response", async () => {
    const f = await fixture();
    const refs = await f.service.register(owner, turn, { kind: "image-response", url: "https://images.example/photo.png", sha256: hash("earlier different pixels"), fetchedAt: "2026-10-13T16:00:00Z" },
      [{ url: "https://images.example/photo.png", pageUrl: "https://images.example/photo.png" }], new AbortController().signal);
    await expect(f.service.inspect(owner, turn, event, refs[0]!.id, new AbortController().signal, async () => undefined)).rejects.toThrow(/Previously fetched image bytes changed/);
    expect(await f.attachments.list(owner)).toHaveLength(0);
  });

  it("keeps trusted action/turn context in the completed-tool fallback instead of generated turn operands or execution ID", async () => {
    const f = await fixture(), selected = await f.service.inspect(owner, turn, event, f.refs[0]!.id, new AbortController().signal, async () => undefined);
    const engine = join(f.root, "engine"); await mkdir(engine);
    const store = new CompletedActionEvidenceStore(engine, owner, async () => undefined);
    const execution = { id: "12345678-1234-4234-8234-123456789abc", toolId: "web.fetch", action: "inspect-image", state: "complete" as const,
      startedAt: "2026-10-13T16:00:00Z", finishedAt: "2026-10-13T16:00:01Z", dispatchStarted: true as const, output: selected };
    const job = await store.stageTool({ brainId: owner, toolId: "web.fetch", action: "inspect-image",
      arguments: { referenceId: selected.reference.id, chatTurnId: "untrusted-generated-turn" } }, execution, turn, event);
    expect(job.provenance.chatTurnId).toBe(turn); expect(job.provenance.actionEventId).toBe(event);
    expect(await readWebImageActionEvidence(store.evidencePath(job.evidenceId), owner, job.evidenceId, execution.id, job.evidenceSha256, async () => undefined)).toEqual(selected);
  });
});
