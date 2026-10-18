import { describe, expect, it } from "vitest";
import { createPluginRuntimeOriginAuthority, type PluginRuntimeOriginSnapshot } from "../src/main/pluginRuntimeService";
import { runtimeHash, runtimeJson } from "../src/main/pluginRuntimeValidation";
import type { PluginRuntimeSession, PluginRuntimeToolExecutor } from "../src/shared/pluginRuntime";
import type { ToolInvocation } from "../src/shared/types";

function fixture() {
  const controller = new AbortController();
  const session: PluginRuntimeSession = { format: "omni-plugin-runtime-session", version: 1, brainId: "brain-a",
    pluginId: "fixture.runtime", contentSha256: "a".repeat(64), moduleId: "worker", sessionId: "11111111-1111-4111-8111-111111111111",
    catalogRevision: 4, instanceRevision: 7, state: "active", startedAt: "2026-10-15T00:00:00.000Z" };
  const invocation: ToolInvocation = { brainId: session.brainId, toolId: "data.learn", action: "start",
    arguments: { sourceId: "owned-source", privateValue: "never-serialize-this" } };
  let currentArgs = structuredClone(invocation.arguments);
  const current: PluginRuntimeOriginSnapshot = { brainId: session.brainId, pluginId: session.pluginId,
    contentSha256: session.contentSha256, moduleId: session.moduleId, sessionId: session.sessionId,
    catalogRevision: session.catalogRevision, instanceRevision: session.instanceRevision, state: "active",
    toolId: invocation.toolId, action: invocation.action, argumentsSha256: runtimeHash(runtimeJson(currentArgs)), capabilityEnabled: true };
  const authority = createPluginRuntimeOriginAuthority({ session, invocation, signal: controller.signal,
    observe: async () => ({ ...current, argumentsSha256: runtimeHash(runtimeJson(currentArgs)) }) });
  return { authority, controller, current, setArgs: (value: Record<string, unknown>) => { currentArgs = value; } };
}

describe("plugin runtime detached origin authority", () => {
  it("retains a main-only exact guard after a queued result and rejects capability, version, operand, Stop, and abort changes", async () => {
    const value = fixture();
    let retained = value.authority;
    const tools: PluginRuntimeToolExecutor = { execute: async (_invocation, _progress, _requestId, _completed, context) => {
      retained = context!.originAuthority;
      return { id: "execution-a", toolId: "data.learn", action: "start", state: "complete",
        startedAt: "2026-10-15T00:00:00.000Z", finishedAt: "2026-10-15T00:00:00.001Z", output: { jobId: "queued-neural-import" } };
    }, cancelAndWait: async () => 0, invalidateApproval: () => false };
    await tools.execute({ brainId: "brain-a", toolId: "data.learn", action: "start", arguments: { sourceId: "owned-source" } },
      undefined, "plugin-runtime-request", undefined, { actionEventId: "plugin-runtime-request", originAuthority: value.authority });
    await expect(retained.assertCurrent()).resolves.toBeUndefined();
    expect(retained.revision).toMatch(/^[a-f0-9]{64}$/);
    expect(JSON.stringify(retained)).not.toContain("never-serialize-this");

    value.current.capabilityEnabled = false;
    await expect(retained.assertCurrent()).rejects.toThrow(/stale/i);
    value.current.capabilityEnabled = true;
    value.current.instanceRevision += 1;
    await expect(retained.assertCurrent()).rejects.toThrow(/stale/i);
    value.current.instanceRevision -= 1;
    value.setArgs({ sourceId: "another-source" });
    await expect(retained.assertCurrent()).rejects.toThrow(/stale/i);
    value.setArgs({ sourceId: "owned-source", privateValue: "never-serialize-this" });
    value.current.state = "stopped";
    await expect(retained.assertCurrent()).rejects.toThrow(/stale/i);
    value.current.state = "active";
    value.controller.abort(new Error("runtime stopped"));
    await expect(retained.assertCurrent()).rejects.toThrow(/runtime stopped/i);
  });

  it("rechecks revocation after an asynchronous observation boundary", async () => {
    const value = fixture();
    let release!: () => void;
    const held = new Promise<void>(resolve => { release = resolve; });
    const authority = createPluginRuntimeOriginAuthority({
      session: { format: "omni-plugin-runtime-session", version: 1, brainId: value.current.brainId,
        pluginId: value.current.pluginId, contentSha256: value.current.contentSha256, moduleId: value.current.moduleId,
        sessionId: value.current.sessionId, catalogRevision: value.current.catalogRevision, instanceRevision: value.current.instanceRevision,
        state: "active", startedAt: "2026-10-15T00:00:00.000Z" },
      invocation: { brainId: value.current.brainId, toolId: value.current.toolId, action: value.current.action,
        arguments: { sourceId: "owned-source", privateValue: "never-serialize-this" } },
      signal: value.controller.signal,
      observe: async () => { await held; return structuredClone(value.current); }
    });
    const pending = authority.assertCurrent();
    value.current.capabilityEnabled = false;
    release();
    await expect(pending).rejects.toThrow(/stale/i);
  });
});
