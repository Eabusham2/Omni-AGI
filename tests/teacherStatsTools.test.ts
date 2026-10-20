/** New typed host routes only: no model, provider, system reader or UI is run. */
import { mkdtemp, mkdir, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import ts from "typescript";
import { afterEach, describe, expect, it, vi } from "vitest";
import catalog from "../tools/catalog.json";
import { BrainService, type RuntimeJobManager } from "../src/main/brainService";
import { DEFAULT_TOOL_PERMISSIONS } from "../src/main/brainRepository";
import { ToolExecutor } from "../src/main/toolExecutor";
import { checkedApiTeacherToolCommand, type ApiTeacherToolService } from "../src/main/apiTeacherTool";
import { declaredTeacherStatsCapabilityPack } from "../src/main/teacherStatsCapabilityPack";
import { BuildInitializationPlanStore } from "../src/main/buildInitializationPlan";
import type { BrainDocument, ToolInvocation, ToolPermissionLevel } from "../src/shared/types";
import type { SystemStatsSnapshot } from "../src/shared/systemStats";
import type { HostActionOriginAuthority } from "../src/shared/hostActionAuthority";

const roots: string[] = [];
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });
const start: ToolInvocation = { brainId: "teacher-brain", toolId: "data.distill-api", action: "start",
  arguments: { provider: "openrouter", model: "selected-model", prompts: ["Compare these supplied examples."], maxOutputTokens: 32, responseModalities: ["text"] } };

async function host(level: ToolPermissionLevel = "ask") {
  const root = await mkdtemp(join(tmpdir(), "omni-teacher-tool-")); roots.push(root);
  const directory = join(root, start.brainId); await mkdir(directory);
  let revision = 1, notify: (value: { brainId: string; toolId: string }) => void = () => undefined;
  const service = { repository: { brainDirectory: () => directory },
    listToolPermissions: vi.fn(async () => ["data.distill-api", "system.stats"].map(toolId => ({ toolId, label: toolId, level, revision, updatedAt: "2026-10-20T00:00:00Z" }))),
    onToolPermissionChanged: (listener: typeof notify) => { notify = listener; },
    authorizeInlineImagination: vi.fn(async () => undefined), queueCompletedToolExecution: vi.fn(async () => undefined)
  } as unknown as BrainService;
  const executor = new ToolExecutor(service, {} as RuntimeJobManager);
  vi.spyOn(executor as unknown as { audit(): Promise<void> }, "audit").mockResolvedValue(undefined);
  const teacher = { status: vi.fn(async () => [{ provider: "openrouter", configured: true, persistence: "encrypted", keyHint: "never-expose" }]),
    start: vi.fn(async () => ({ id: "queued-job", brainId: start.brainId, kind: "teacher", state: "queued" })),
    list: vi.fn(async () => []), details: vi.fn(async () => ({ jobId: "selected-job", state: "paused", items: [] })),
    pause: vi.fn(async () => ({ state: "paused" })), resume: vi.fn(async () => ({ state: "queued" })), stop: vi.fn(async () => ({ state: "cancelled" })),
    resolveUncertain: vi.fn(async () => ({ state: "paused" })), invalidateToolAuthority: vi.fn(),
    discoverModels: vi.fn(), cancelModelDiscovery: vi.fn()
  };
  executor.setApiTeacherService(teacher as unknown as ApiTeacherToolService);
  return { executor, teacher, service, revoke: () => { revision++; notify({ brainId: start.brainId, toolId: start.toolId }); } };
}

describe("new API teacher and read-only system statistics routes", () => {
  it("rejects non-string modality and uncertain-review choices without coercing nested values", () => {
    for (const value of [["text"], {}, null, 1]) {
      expect(() => checkedApiTeacherToolCommand("start", { ...start.arguments, responseModalities: [value] })).toThrow(/valid explicit/);
    }
    for (const decision of [["retry"], ["skip"], {}, null, true]) {
      expect(() => checkedApiTeacherToolCommand("resolve-uncertain", { jobId: "selected-job", itemIndex: 0, decision })).toThrow(/exact uncertain/);
    }
  });

  it("declares credential-free typed schemas, ordinary supplementary lessons and actual native availability", () => {
    const pack = declaredTeacherStatsCapabilityPack();
    expect(pack).toMatchObject({ id: "api-teacher-system-stats-1", executedHostActions: false, preferenceLabels: false, rewardModel: false });
    for (const id of ["data.distill-api", "system.stats"]) {
      const declared = catalog.tools.find(tool => tool.id === id)!;
      const schema = pack.schemas.find(tool => tool.id === id)!;
      expect(declared.defaultGrant).toBe("ask"); expect(schema.grant).toBe("ask");
      expect(schema.actions).toEqual(Object.keys(declared.actions));
      expect(DEFAULT_TOOL_PERMISSIONS.find(value => value.toolId === id)?.level).toBe("ask");
    }
    const native = (records: unknown) => BrainService.prototype.neuralToolSchemas.call(
      { externalToolSchemas: new Map() } as unknown as BrainService, { toolPermissions: records } as BrainDocument);
    const selected = native(DEFAULT_TOOL_PERMISSIONS);
    expect(selected.find(tool => tool.id === "data.distill-api")?.actionInputSchemas?.start).toMatchObject({ additionalProperties: false,
      required: ["provider", "model", "prompts"], properties: { responseModalities: { type: "array", items: { enum: ["text", "image", "audio", "video"] } } } });
    expect(native(DEFAULT_TOOL_PERMISSIONS.map(value => ({ ...value, level: "off" }))).some(tool => ["data.distill-api", "system.stats"].includes(tool.id))).toBe(false);
    const oldFull = native(DEFAULT_TOOL_PERMISSIONS.filter(value => !["data.distill-api", "system.stats"].includes(value.toolId)).map(value => ({ ...value, level: "full" })));
    expect(oldFull.filter(tool => ["data.distill-api", "system.stats"].includes(tool.id)).map(tool => tool.grant)).toEqual(["ask", "ask"]);
    for (const record of pack.records.filter(record => record.toolId === start.toolId))
      expect(() => checkedApiTeacherToolCommand(record.action!, record.arguments)).not.toThrow();
  });

  it("uses the actual Build master-access projection for Off, Ask and explicit Full", async () => {
    const source = await readFile(new URL("../src/renderer/src/App.tsx", import.meta.url), "utf8");
    const ast = ts.createSourceFile("App.tsx", source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    const declarations: string[] = [];
    for (const statement of ast.statements) {
      if (ts.isVariableStatement(statement)) for (const declaration of statement.declarationList.declarations)
        if (ts.isIdentifier(declaration.name) && ["buildToolProtocolIds", "standardBuildAccess"].includes(declaration.name.text)) declarations.push(`const ${declaration.getText(ast)};`);
      if (ts.isFunctionDeclaration(statement) && statement.name?.text === "buildAccessLevels") declarations.push(statement.getText(ast));
    }
    expect(declarations).toHaveLength(3);
    const compiled = ts.transpileModule(declarations.join("\n"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
    const projection = new Function(`${compiled}\nreturn { ids: buildToolProtocolIds, levels: buildAccessLevels };`)() as {
      ids: Record<string, string[]>; levels(on: boolean, full?: boolean): Record<string, ToolPermissionLevel> };
    for (const [enabled, full, expected] of [[false, false, "off"], [true, false, "ask"], [true, true, "full"]] as const) {
      const levels = projection.levels(enabled, full);
      const ids = Object.entries(levels).flatMap(([id, level]) => projection.ids[id]!.map(toolId => ({ toolId, level })));
      for (const toolId of ["data.distill-api", "system.stats"]) expect(ids.find(value => value.toolId === toolId)?.level).toBe(expected);
    }
  });

  it("passes the declared pack through the real owned initial-capability method, without native execution", async () => {
    const plan = vi.spyOn(BuildInitializationPlanStore.prototype, "get").mockResolvedValue({ phase: "initial-learning", tasks: [], foundation: {} } as never);
    const commit = vi.fn(async (_id: string, pack: { id: string; sha256: string }) => ({ packId: pack.id, packSha256: pack.sha256, candidateAccepted: false }));
    const receiver = { repository: { get: async () => ({ readiness: { state: "initializing" } }) },
      resumeFoundation: vi.fn(async () => undefined), commitSupplementaryCapabilities: commit };
    try {
      await BrainService.prototype.teachInitialCapabilities.call(receiver as unknown as BrainService, start.brainId);
      expect(commit.mock.calls.filter(([, pack]) => pack.id === "api-teacher-system-stats-1")).toHaveLength(1);
      expect(commit.mock.calls.find(([, pack]) => pack.id === "api-teacher-system-stats-1")?.[1].sha256).toBe(declaredTeacherStatsCapabilityPack().sha256);
    } finally { plan.mockRestore(); }
  });

  it("refuses credential, owner, endpoint and request-identity fields before any tool journal or dispatch", async () => {
    const value = await host("full");
    for (const field of ["apiKey", "credential", "brainId", "requestId", "endpoint", "authority"])
      await expect(value.executor.execute({ ...start, arguments: { ...start.arguments, [field]: "not-accepted" } })).rejects.toThrow(/unsupported fields/);
    expect(value.teacher.start).not.toHaveBeenCalled(); expect(value.service.queueCompletedToolExecution).not.toHaveBeenCalled();
    expect(() => checkedApiTeacherToolCommand("start", { ...start.arguments, maxOutputTokens: 65_537 })).toThrow(/valid explicit/);
  });

  it("honors Off and Ask/Auto review, then forwards only the exact main-owned start and detached authority", async () => {
    const off = await host("off"); expect(await off.executor.execute(start)).toMatchObject({ state: "failed", error: expect.stringContaining("disabled") });
    expect(off.teacher.start).not.toHaveBeenCalled();
    for (const level of ["ask", "auto"] as const) {
      const value = await host(level), context = { actionEventId: `teacher-${level}-action` };
      const held = await value.executor.execute(start, undefined, `start-${level}`, undefined, context);
      expect(held.state).toBe("approval-required"); expect(value.teacher.start).not.toHaveBeenCalled();
      const result = await value.executor.execute({ ...start, approvalToken: held.approvalToken }, undefined, `start-${level}`, undefined, context);
      expect(result).toMatchObject({ state: "complete", output: { kind: "teacher", state: "queued" } });
      const [request, authority] = value.teacher.start.mock.calls[0] as unknown as [Record<string, unknown>, HostActionOriginAuthority];
      expect(request).toEqual({ ...start.arguments, brainId: start.brainId, requestId: context.actionEventId });
      await expect(authority.assertCurrent()).resolves.toBeUndefined(); value.revoke();
      expect(value.teacher.invalidateToolAuthority).toHaveBeenCalledWith(start.brainId);
      await expect(authority.assertCurrent()).rejects.toThrow(/permission changed/i);
      expect(JSON.stringify(result)).not.toContain("assertCurrent");
    }
  });

  it("requires explicit human review for uncertain Retry/Skip even under Full", async () => {
    const value = await host("full"), invocation = { ...start, action: "resolve-uncertain", arguments: { jobId: "selected-job", itemIndex: 2, decision: "skip" } };
    const held = await value.executor.execute(invocation, undefined, "review-request");
    expect(held.state).toBe("approval-required"); expect(value.teacher.resolveUncertain).not.toHaveBeenCalled();
    expect(await value.executor.execute({ ...invocation, approvalToken: held.approvalToken }, undefined, "review-request")).toMatchObject({ state: "complete", output: { state: "paused" } });
    expect(value.teacher.resolveUncertain).toHaveBeenCalledExactlyOnceWith(start.brainId, "selected-job", 2, "skip");
    expect(value.teacher.resume).not.toHaveBeenCalled();
  });

  it("does not disclose a job read after current permission was revoked during that read", async () => {
    const value = await host("full");
    value.teacher.details.mockImplementation(async () => { value.revoke(); return { jobId: "selected-job", state: "paused", items: [{ response: "held-private-reply" }] } as never; });
    const result = await value.executor.execute({ ...start, action: "status", arguments: { jobId: "selected-job" } });
    expect(result).toMatchObject({ state: "failed", dispatchStarted: true });
    expect(JSON.stringify(result)).not.toContain("held-private-reply"); expect(value.teacher.details).toHaveBeenCalledWith(start.brainId, "selected-job");
  });

  it("cancels exact in-flight model discovery through the existing service without exposing keys", async () => {
    const value = await host("full");
    const providers = await value.executor.execute({ ...start, action: "providers", arguments: {} });
    expect(JSON.stringify(providers)).not.toContain("never-expose");
    let entered!: () => void, reject!: (reason: Error) => void;
    const started = new Promise<void>(resolve => { entered = resolve; });
    value.teacher.discoverModels.mockImplementation(() => { entered(); return new Promise((_resolve, rejectWork) => { reject = rejectWork; }); });
    value.teacher.cancelModelDiscovery.mockImplementation(() => { reject(new Error("Discovery stopped")); });
    const pending = value.executor.execute({ ...start, action: "discover-models", arguments: { provider: "openrouter" } }, undefined, "discovery-request");
    await started; expect(value.executor.cancel(start.brainId, "discovery-request")).toBe(1);
    expect(await pending).toMatchObject({ state: "failed", dispatchStarted: true });
    expect(value.teacher.cancelModelDiscovery).toHaveBeenCalledWith({ provider: "openrouter", requestId: expect.any(String) });
    expect(value.teacher.cancelModelDiscovery.mock.calls[0]).toEqual(value.teacher.discoverModels.mock.calls[0]);
  });

  it("keeps system snapshots behind ordinary Off/Ask/Auto/Full and rechecks the actual host guard", async () => {
    for (const level of ["off", "ask", "auto", "full"] as const) {
      const value = await host(level), snapshot = vi.fn(async (_id: string, signal: AbortSignal, current: () => Promise<void>) => {
        signal.throwIfAborted(); await current(); return { format: "omni-system-stats" } as SystemStatsSnapshot;
      }); value.executor.setSystemStatsService({ snapshot });
      const result = await value.executor.execute({ brainId: start.brainId, toolId: "system.stats", action: "snapshot", arguments: {} });
      expect(result.state).toBe(level === "off" ? "failed" : level === "ask" ? "approval-required" : "complete");
      expect(snapshot).toHaveBeenCalledTimes(level === "full" || level === "auto" ? 1 : 0);
      if (level === "full") {
        expect(await value.executor.execute({ brainId: start.brainId, toolId: "system.stats", action: "snapshot", arguments: { path: "/not-authority" } })).toMatchObject({ state: "failed" });
        expect(snapshot).toHaveBeenCalledOnce();
      }
    }
  });
});
