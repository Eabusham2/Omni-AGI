import { createHash } from "node:crypto";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { PluginRuntimeService, pluginRuntimeHostRequestId } from "../src/main/pluginRuntimeService";
import { runtimeHash, runtimeJson, runtimeModuleSha256 } from "../src/main/pluginRuntimeValidation";
import type { PluginService } from "../src/main/pluginService";
import type { PluginRuntimeModuleDefinition, PluginRuntimeProcessFactory, PluginRuntimeTrainingParserActivationRequest } from "../src/shared/pluginRuntime";
import type { ToolExecutionResult, ToolInvocation } from "../src/shared/types";

const roots: string[] = [], services: PluginRuntimeService[] = [];
afterEach(async () => { for (const service of services.splice(0)) await service.dispose(); await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });

describe("runtime parser effect resumption", () => {
  it("reuses transaction-stable host receipts across fresh sessions and rejects changed ordinal operands", async () => {
    const root = await mkdtemp(join(tmpdir(), "omni-runtime-effects-")); roots.push(root);
    const hash = "a".repeat(64), transactionKey = "b".repeat(64), sourceSha256 = "c".repeat(64);
    const module: PluginRuntimeModuleDefinition = { id: "records", title: "Records", description: "Runtime parser effect fixture.", script:
      "omniRuntime.registerParser({push:async()=>({state:'done'}),drain:async()=>({state:'done'})});", events: [], ui: { kinds: [], slots: [] }, dynamicTools: false };
    const entry = { pluginId: "fixture.runtime-parser", enabled: true, activeContentSha256: hash, revision: 4 };
    const access = { enabled: true, contentSha256: hash, revision: 6,
      capabilities: [{ toolId: "data.training-parser", enabled: true, actions: ["records"] }, { toolId: "system.files", enabled: true, actions: ["read"] }] };
    const plugin = { list: async () => [entry], instance: async () => access, package: async () => ({ manifest: { runtimeModules: [module],
      trainingFormats: [{ id: "records", extensions: ["runtime"], mediaTypes: [], parser: { kind: "runtime-stream", moduleId: "records", output: "jsonl" } }] } }),
      onEvent: () => () => undefined } as unknown as PluginService;
    const retained = new Map<string, { hash: string; result: ToolExecutionResult }>(); let effects = 0, sender = 40;
    const execute = async (invocation: ToolInvocation, _progress?: undefined, requestId?: string): Promise<ToolExecutionResult> => {
      if (!requestId) throw new Error("Missing host request identity."); const digest = runtimeHash(runtimeJson(invocation)), prior = retained.get(requestId);
      if (prior && prior.hash !== digest) return { ...prior.result, state: "failed", error: "Changed operands under a retained effect identity." };
      if (prior) return prior.result;
      effects++; const result: ToolExecutionResult = { id: `effect-${effects}`, toolId: invocation.toolId, action: invocation.action,
        state: "complete", startedAt: "2026-10-15T00:00:00.000Z", finishedAt: "2026-10-15T00:00:00.001Z", output: { effects } };
      retained.set(requestId, { hash: digest, result }); return result;
    };
    let currentLaunch: Parameters<PluginRuntimeProcessFactory["launch"]>[0] | undefined; const senders = new Map<string, number>();
    const service = new PluginRuntimeService(plugin, { brainRoot: () => root,
      tools: { execute, cancelAndWait: async () => 0, invalidateApproval: () => true },
      admit: async () => ({ maxMessageBytes: 8192, check: async () => undefined, release: () => undefined }),
      admitTraining: async () => ({ maxMessageBytes: 8192, check: async () => undefined, release: () => undefined }),
      admitRead: async () => ({ release: () => undefined }), checkDisk: async () => undefined,
      processFactory: { launch: async (launch, callbacks) => { currentLaunch = launch; const ownedSender = sender++; senders.set(launch.session.sessionId, ownedSender);
        setImmediate(() => { void (async () => { await callbacks.receive(ownedSender, launch.url, { format: "omni-plugin-runtime-rpc", version: 1,
          sessionId: launch.session.sessionId, nonce: launch.nonce, id: "parser-registration", operation: "parser-ready",
          arguments: { abi: "omni-runtime-record-stream-1", formatId: "records" } });
          await callbacks.receive(ownedSender, launch.url, { format: "omni-plugin-runtime-rpc", version: 1,
            sessionId: launch.session.sessionId, nonce: launch.nonce, id: "runtime-ready", operation: "ready", arguments: { ok: true } }); })(); });
        return { senderId: ownedSender, processId: 5000 + ownedSender, send: () => undefined, terminate: async () => undefined }; } },
      publishTools: async () => undefined, retireTools: async () => undefined, moduleReadyTimeoutMs: 1000, moduleShutdownTimeoutMs: 1 }); services.push(service);
    const request: PluginRuntimeTrainingParserActivationRequest = { brainId: "brain-a", pluginId: entry.pluginId, contentSha256: hash,
      moduleId: module.id, moduleSha256: runtimeModuleSha256(module), formatId: "records", transactionKey, sourceSha256, sourceBytes: 10,
      expectedCatalogRevision: 4, expectedInstanceRevision: 6, requestId: transactionKey };
    const call = async (arguments_: Record<string, unknown>) => {
      const active = await service.activateTrainingParser(request), launch = currentLaunch!;
      const response = await service.receive(senders.get(active.session.sessionId)!, launch.url, {
        format: "omni-plugin-runtime-rpc", version: 1, sessionId: active.session.sessionId, nonce: launch.nonce,
        id: "parser-call-0", operation: "call", arguments: { toolId: "system.files", action: "read", arguments: arguments_ } });
      const stableId = pluginRuntimeHostRequestId(active.session, "parser-call-0"); await service.stop("brain-a", active.session.sessionId); return { response, stableId };
    };
    const first = await call({ attachmentId: "same" }), changed = await call({ attachmentId: "changed" }), recovered = await call({ attachmentId: "same" });
    expect(first.response).toMatchObject({ state: "complete", value: { effects: 1 } });
    expect(changed.response).toMatchObject({ state: "failed", error: expect.stringMatching(/changed operands/i) });
    expect(recovered.response).toMatchObject({ state: "complete", value: { effects: 1 } });
    expect(changed.stableId).toBe(first.stableId); expect(recovered.stableId).toBe(first.stableId); expect(effects).toBe(1);
    expect(first.stableId).toBe(`plugin-runtime-${createHash("sha256").update(runtimeJson(["training-parser", "brain-a", entry.pluginId, hash,
      module.id, runtimeModuleSha256(module), "records", transactionKey, sourceSha256, "parser-call-0"])).digest("hex")}`);
  });
});
