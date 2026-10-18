import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { lstat, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { canonicalPluginPackage } from "../src/main/pluginCatalog";
import { PluginRuntimeStreamParser, type PluginRuntimeTrainingParserHost } from "../src/main/pluginRuntimeStreamParser";
import { runtimeModuleSha256 } from "../src/main/pluginRuntimeValidation";
import {
  PLUGIN_RUNTIME_PARSER_ABI,
  PLUGIN_TRAINING_PARSER_CAPABILITY,
  type OmniPluginPackage
} from "../src/shared/plugins";
import type {
  PluginRuntimeEvent,
  PluginRuntimeParserStep,
  PluginRuntimePendingApproval,
  PluginRuntimeSession,
  PluginRuntimeTrainingParserActivationRequest
} from "../src/shared/pluginRuntime";
import type { PluginTrainingFormatBinding } from "../src/shared/types";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const sha = (value: Buffer | string): string => createHash("sha256").update(value).digest("hex");

function moduleDefinition() {
  return { id: "records", title: "Records", description: "Capability-aware record stream.", script:
    "omniRuntime.registerParser({push: async step => ({state: step.final ? 'done' : 'need-input'}), drain: async () => ({state:'need-input'})});",
    events: [], ui: { kinds: [] as [], slots: [] }, dynamicTools: false };
}

function packageFixture(): OmniPluginPackage {
  const module = moduleDefinition();
  return { format: "omni-plugin-package", formatVersion: 1, manifest: { id: "fixture.runtime-parser", version: "1.0.0",
    name: "Runtime parser", description: "Exact runtime stream parser.", author: "Fixture", license: "MIT", documentation: "Fixture",
    capabilities: [{ toolId: PLUGIN_TRAINING_PARSER_CAPABILITY, actions: ["records"], reason: "Parse selected records." }],
    tools: [], panels: [], interactivePanels: [], runtimeModules: [module], themes: [], trainingFormats: [{ id: "records", title: "Records",
      description: "Runtime records.", extensions: ["rstream"], mediaTypes: ["application/x-runtime-records"],
      parser: { kind: "runtime-stream", moduleId: module.id, output: "jsonl" } }] } };
}

class FakeHost implements PluginRuntimeTrainingParserHost {
  readonly activations: PluginRuntimeTrainingParserActivationRequest[] = [];
  readonly steps: PluginRuntimeParserStep[] = [];
  readonly stopped: string[] = [];
  readonly listeners = new Set<(event: PluginRuntimeEvent) => void>();
  pending: PluginRuntimePendingApproval[] = [];
  block = false;
  splitOutput = false;
  activationError?: Error;
  beforeStep?: (step: PluginRuntimeParserStep) => void | Promise<void>;
  started?: () => void;
  lastSession?: PluginRuntimeSession;
  private readonly outputQueue = new Map<number, string[]>();

  async activateTrainingParser(request: PluginRuntimeTrainingParserActivationRequest) {
    if (this.activationError) throw this.activationError;
    this.activations.push(structuredClone(request));
    const session: PluginRuntimeSession = { format: "omni-plugin-runtime-session", version: 1, brainId: request.brainId,
      pluginId: request.pluginId, contentSha256: request.contentSha256, moduleId: request.moduleId, sessionId: "11111111-1111-4111-8111-111111111111",
      catalogRevision: request.expectedCatalogRevision, instanceRevision: request.expectedInstanceRevision, state: "active",
      startedAt: "2026-10-15T00:00:00.000Z", purpose: "training-parser", parser: { formatId: request.formatId,
        transactionKey: request.transactionKey, sourceSha256: request.sourceSha256, sourceBytes: request.sourceBytes, moduleSha256: request.moduleSha256,
        ...(request.inputMediaType ? { inputMediaType: request.inputMediaType } : {}) } };
    this.lastSession = session; return { session, maxMessageBytes: 32 * 1024 };
  }
  async parserStep(brainId: string, sessionId: string, step: PluginRuntimeParserStep, signal: AbortSignal) {
    this.steps.push(structuredClone(step)); this.started?.(); await this.beforeStep?.(step);
    if (this.block) return new Promise<never>((_resolve, reject) => {
      const abort = () => reject(signal.reason ?? new Error("cancelled")); signal.addEventListener("abort", abort, { once: true }); if (signal.aborted) abort();
    });
    if (step.operation === "drain") {
      const queued = this.outputQueue.get(step.sequence) ?? [], outputBase64 = queued.shift();
      if (queued.length) this.outputQueue.set(step.sequence, queued); else this.outputQueue.delete(step.sequence);
      return outputBase64 ? { state: "output" as const, outputBase64 } : { state: "need-input" as const };
    }
    if (step.final) return { state: "done" as const };
    if (!this.pending.length) {
      this.pending = [{ brainId, sessionId, rpcId: "rpc-review", reviewId: "review-safe", pluginId: "fixture.runtime-parser",
        contentSha256: "a".repeat(64), moduleId: "records", executionId: "execution-safe",
        invocation: { brainId, toolId: "system.files", action: "read", arguments: { attachmentId: "owned" } } }];
      for (const listener of this.listeners) listener({ brainId, kind: "approval", session: this.lastSession });
      await new Promise(resolve => setTimeout(resolve, 0)); this.pending = [];
    }
    const source = Buffer.from(step.inputBase64!, "base64");
    const line = Buffer.from(`${JSON.stringify({ sourceHex: source.toString("hex") })}\n`);
    if (this.splitOutput) {
      const middle = Math.max(1, Math.floor(line.length / 2)); this.outputQueue.set(step.sequence, [line.subarray(middle).toString("base64")]);
      return { state: "output" as const, outputBase64: line.subarray(0, middle).toString("base64") };
    }
    return { state: "output" as const, outputBase64: line.toString("base64") };
  }
  approvals(): Promise<PluginRuntimePendingApproval[]> { return Promise.resolve(structuredClone(this.pending)); }
  onEvent(listener: (event: PluginRuntimeEvent) => void): () => void { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  admitParserRead(): Promise<{ release(): void }> { return Promise.resolve({ release: () => undefined }); }
  async stop(_brainId: string, sessionId: string): Promise<PluginRuntimeSession> {
    this.stopped.push(sessionId); if (!this.lastSession) throw new Error("No fake runtime session."); return { ...this.lastSession, state: "stopped" };
  }
}

async function fixture(host: FakeHost, options?: ConstructorParameters<typeof PluginRuntimeStreamParser>[1]) {
  const root = await mkdtemp(join(tmpdir(), "omni-runtime-stream-")); roots.push(root);
  const source = join(root, "binary.rstream"), bytes = Buffer.from([0, 255, 1, 2, 3, 4]); await writeFile(source, bytes);
  const module = moduleDefinition(), contentSha256 = "a".repeat(64);
  const binding: PluginTrainingFormatBinding = { pluginId: "fixture.runtime-parser", contentSha256, formatId: "records", parserFormat: "jsonl",
    catalogRevision: 3, instanceRevision: 5, inputMediaType: "application/x-runtime-records",
    executable: { kind: "runtime-stream", moduleId: module.id, moduleSha256: runtimeModuleSha256(module), abi: PLUGIN_RUNTIME_PARSER_ABI, outputFormat: "jsonl" } };
  return { root, source, bytes, module, binding, parser: new PluginRuntimeStreamParser(host, options), request: { sourcePath: source, sourceSha256: sha(bytes),
    sourceBytes: bytes.length, inputMediaType: "application/x-runtime-records", transactionKey: "b".repeat(64), workspaceRoot: root } };
}

describe("runtime-stream plugin parser", () => {
  it("accepts only an exact declared runtime parser package", () => {
    const prepared = canonicalPluginPackage(packageFixture()).package;
    expect(prepared.manifest.trainingFormats[0]?.parser).toEqual({ kind: "runtime-stream", moduleId: "records", output: "jsonl" });
    expect(() => canonicalPluginPackage({ ...packageFixture(), manifest: { ...packageFixture().manifest, runtimeModules: [] } }))
      .toThrow(/undeclared runtime module/i);
  });

  it("streams host-owned binary pages to JSONL without exposing a path and publishes a safe Ask wait", async () => {
    const host = new FakeHost(), value = await fixture(host), progress: unknown[] = [];
    const prepared = await value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: { ...value.request, onProgress: (_fraction, _message, detail) => { progress.push(detail); } } });
    if (!("path" in prepared)) throw new Error("JSONL parser returned an empty media result.");
    expect(await readFile(prepared.path, "utf8")).toBe(`${JSON.stringify({ sourceHex: value.bytes.toString("hex") })}\n`);
    expect(prepared.provenance).toMatchObject({ abi: PLUGIN_RUNTIME_PARSER_ABI, sourceSha256: sha(value.bytes), sourceBytes: value.bytes.length,
      outputRecords: 1, complete: true });
    expect(JSON.stringify(host.activations)).not.toContain(value.source);
    expect(JSON.stringify(host.steps)).not.toContain(value.source);
    expect(progress).toContainEqual(expect.objectContaining({ state: "waiting-for-approval", approval: expect.objectContaining({
      reviewId: "review-safe", toolId: "system.files", action: "read" }) }));
    const states = progress.map(value => (value as { state?: string }).state);
    expect(states.indexOf("waiting-for-approval")).toBeLessThan(states.lastIndexOf("streaming"));
    expect(states.at(-1)).toBe("complete");
    expect(JSON.stringify(progress)).not.toContain("approvalToken");
    expect(host.stopped).toHaveLength(1); await prepared.cleanup();
  });

  it("cancels an in-flight page and retires the dedicated parser session", async () => {
    const host = new FakeHost(); host.block = true; let started!: () => void; const entered = new Promise<void>(resolve => { started = resolve; }); host.started = started;
    const value = await fixture(host), controller = new AbortController();
    const pending = value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: { ...value.request, signal: controller.signal } });
    await entered; controller.abort(new Error("fixture cancellation"));
    await expect(pending).rejects.toThrow(/fixture cancellation/i);
    expect(host.stopped).toHaveLength(1);
  });

  it("cleans activation and late receipt failures without publishing a completed output", async () => {
    const activationHost = new FakeHost(); activationHost.activationError = new Error("activation refused");
    const activation = await fixture(activationHost);
    await expect(activation.parser.prepare({ brainId: "brain-a", binding: activation.binding, module: activation.module,
      moduleSha256: activation.binding.executable!.moduleSha256, request: activation.request })).rejects.toThrow(/activation refused/i);
    expect(await readdir(join(activation.root, "plugin-runtime-parser"))).toEqual([]);

    const receiptHost = new FakeHost(), receipt = await fixture(receiptHost);
    receiptHost.beforeStep = async step => { if (step.operation === "push" && step.final) {
      await mkdir(join(receipt.root, "plugin-runtime-parser", `${receipt.request.transactionKey}.receipt.json`));
    } };
    await expect(receipt.parser.prepare({ brainId: "brain-a", binding: receipt.binding, module: receipt.module,
      moduleSha256: receipt.binding.executable!.moduleSha256, request: receipt.request })).rejects.toThrow();
    await expect(lstat(join(receipt.root, "plugin-runtime-parser", `${receipt.request.transactionKey}.jsonl`))).rejects.toMatchObject({ code: "ENOENT" });
    expect(receiptHost.stopped).toHaveLength(1);
  });

  it("refuses a same-path source replacement before sending another page or committing output", async () => {
    const host = new FakeHost(), value = await fixture(host); let replaced = false;
    host.beforeStep = async step => { if (!replaced && step.operation === "push" && !step.final) {
      replaced = true; await rm(value.source); await writeFile(value.source, value.bytes);
    } };
    await expect(value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request })).rejects.toThrow(/path or opened file changed/i);
    expect(host.stopped).toHaveLength(1);
    await expect(lstat(join(value.root, "plugin-runtime-parser", `${value.request.transactionKey}.jsonl`))).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("validates a complete JSONL record that spans multiple admitted output pages", async () => {
    const host = new FakeHost(); host.splitOutput = true; const value = await fixture(host);
    const prepared = await value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request });
    if (!("path" in prepared)) throw new Error("JSONL parser returned an empty media result.");
    expect(await readFile(prepared.path, "utf8")).toBe(`${JSON.stringify({ sourceHex: value.bytes.toString("hex") })}\n`);
    expect(prepared.provenance.outputRecords).toBe(1); expect(host.steps.filter(step => step.operation === "drain").length).toBeGreaterThanOrEqual(2);
    await prepared.cleanup();
  });

  it("captures an early asynchronous output error before the next parser boundary and cleans custody", async () => {
    const host = new FakeHost(); host.splitOutput = true;
    const value = await fixture(host, { createOutput: (path, options) => {
      const stream = createWriteStream(path, options), write = stream.write.bind(stream);
      stream.write = ((chunk: unknown, ...args: unknown[]) => {
        const result = write(chunk as any, ...(args as any[])); queueMicrotask(() => stream.destroy(new Error("fixture async output failure"))); return result || true;
      }) as typeof stream.write;
      return stream;
    } });
    await expect(value.parser.prepare({ brainId: "brain-a", binding: value.binding, module: value.module,
      moduleSha256: value.binding.executable!.moduleSha256, request: value.request })).rejects.toThrow(/fixture async output failure/i);
    expect(host.stopped).toHaveLength(1);
    await expect(lstat(join(value.root, "plugin-runtime-parser", `${value.request.transactionKey}.jsonl`))).rejects.toMatchObject({ code: "ENOENT" });
  });
});
