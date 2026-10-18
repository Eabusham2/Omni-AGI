import { createHash } from "node:crypto";
import { copyFile, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PluginRuntimeStore, PLUGIN_RUNTIME_WORKSPACE } from "../src/main/pluginRuntimeStore";
import { mergePluginRuntimeWorkspace, openPluginRuntimeSource, rekeyPluginRuntimeWorkspace, validatePluginRuntimeWorkspace } from "../src/main/pluginRuntimePortability";
import { snapshotLearningWorkspace, adoptLearningWorkspace } from "../src/main/learningWorkspacePortability";
import { runtimeHash, runtimeJson } from "../src/main/pluginRuntimeValidation";
import type { PluginRuntimeSession } from "../src/shared/pluginRuntime";

const roots: string[] = [], plugin = "a".repeat(64), timestamp = "2026-10-13T12:00:00.000Z";
afterEach(async () => { vi.restoreAllMocks(); for (const path of roots.splice(0)) await rm(path, { recursive: true, force: true }); });
async function root(): Promise<string> { const value = await mkdtemp(join(tmpdir(), "omni-runtime-custody-")); roots.push(value); return value; }
const sha = async (path: string) => createHash("sha256").update(await readFile(path)).digest("hex");
async function fixture(owner: string, value: unknown = { exact: owner }): Promise<string> { const path = await root(), store = await PluginRuntimeStore.open(path, owner, { create: true });
  try { const session: PluginRuntimeSession = { format: "omni-plugin-runtime-session", version: 1, brainId: owner, pluginId: "fixture.feature", contentSha256: plugin, moduleId: "feature", sessionId: `session-${owner}`, catalogRevision: 1, instanceRevision: 2, state: "active", startedAt: timestamp, processId: 9876 };
    await store!.session(session); await store!.put(plugin, "feature", "state", value, 0); await store!.beginRpc(session.sessionId, "effect", "call", { actual: owner }); await store!.markRpc(session.sessionId, "effect", "dispatched");
    const text = runtimeJson({ exact: "public schema" }); await store!.queueSchema({ brainId: owner, pluginId: "fixture.feature", moduleId: "feature", contentSha256: plugin, text, sourceSha256: runtimeHash(text) });
  } finally { store!.close(); } return path; }

describe("lifecycle module complete dormant custody - new small SQLite edges", () => {
  it("round-trips through the existing export envelope with unsanitized history/state and no process/activation inheritance", async () => {
    const source = await fixture("source", { privateText: "complete unsanitized user text🙂", remembered: [1, 2, 3] }), target = await root(), packaged = await root(), before = await sha(join(source, PLUGIN_RUNTIME_WORKSPACE));
    const snapshot = await snapshotLearningWorkspace(source, packaged, "source"); expect(snapshot).toBeDefined(); await adoptLearningWorkspace(snapshot!, target, "target"); await validatePluginRuntimeWorkspace(target, "target");
    const store = (await PluginRuntimeStore.open(target, "target", { readOnly: true }))!; try {
      expect(store.value(plugin, "feature", "state").value).toEqual({ privateText: "complete unsanitized user text🙂", remembered: [1, 2, 3] }); expect(store.sessionsPage().sessions[0]?.state).toBe("dormant"); expect(store.sessionsPage().sessions[0]).not.toHaveProperty("processId");
      expect([...store.pendingSchemas()]).toEqual([]); expect(store.rpc("session-source", "effect")?.state).toBe("dispatched"); expect(store.sources().sources.length).toBeGreaterThan(0);
      const archived = await openPluginRuntimeSource(target, "target", store.sources().sources[0]!.sha256); try { expect(archived.history().events.some(row => row.kind === "state")).toBe(true); } finally { archived.close(); }
    } finally { store.close(); } expect(await sha(join(source, PLUGIN_RUNTIME_WORKSPACE))).toBe(before);
  });

  it("retains conflicting parent states in exact sources and blocks a default winner until reviewed selection", async () => {
    const a = await fixture("a", { distinct: "left" }), b = await fixture("b", { distinct: "right" }), target = await root(), beforeA = await sha(join(a, PLUGIN_RUNTIME_WORKSPACE)), beforeB = await sha(join(b, PLUGIN_RUNTIME_WORKSPACE));
    await rekeyPluginRuntimeWorkspace(a, "a", "merged"); await rekeyPluginRuntimeWorkspace(b, "b", "merged"); const parentA = await sha(join(a, PLUGIN_RUNTIME_WORKSPACE)), parentB = await sha(join(b, PLUGIN_RUNTIME_WORKSPACE));
    await mergePluginRuntimeWorkspace(a, target, "merged", "a"); await mergePluginRuntimeWorkspace(b, target, "merged", "b"); const store = (await PluginRuntimeStore.open(target, "merged"))!;
    try { expect(() => store.value(plugin, "feature", "state")).toThrow(/conflicts/); expect(store.conflictCount(plugin, "feature")).toBe(1); const sources = store.sources().sources, originalA = sources.find(row => row.owner === "a")!;
      const archived = await openPluginRuntimeSource(target, "merged", originalA.sha256); try { expect(archived.value(plugin, "feature", "state").value).toEqual({ distinct: "left" }); } finally { archived.close(); }
      await store.selectValue(plugin, "feature", "state", { distinct: "left" }, 0, originalA.sha256); expect(store.value(plugin, "feature", "state").value).toEqual({ distinct: "left" }); expect(store.conflicts().total).toBe(0);
      expect(sources.some(row => row.owner === "b")).toBe(true); expect(sources.length).toBeGreaterThanOrEqual(4);
    } finally { store.close(); }
    expect(await sha(join(a, PLUGIN_RUNTIME_WORKSPACE))).toBe(parentA); expect(await sha(join(b, PLUGIN_RUNTIME_WORKSPACE))).toBe(parentB); expect(beforeA).not.toBe(parentA); expect(beforeB).not.toBe(parentB);
  });

  it("deduplicates only identical archive bytes on retry and keeps equal state unambiguous", async () => {
    const a = await fixture("a", { same: 7 }), b = await fixture("b", { same: 7 }), target = await root(); await rekeyPluginRuntimeWorkspace(a, "a", "merged"); await rekeyPluginRuntimeWorkspace(b, "b", "merged");
    await mergePluginRuntimeWorkspace(a, target, "merged", "a"); await mergePluginRuntimeWorkspace(b, target, "merged", "b"); const before = await sha(join(target, PLUGIN_RUNTIME_WORKSPACE)); await mergePluginRuntimeWorkspace(b, target, "merged", "b");
    expect(await sha(join(target, PLUGIN_RUNTIME_WORKSPACE))).toBe(before); const store = (await PluginRuntimeStore.open(target, "merged", { readOnly: true }))!; try { expect(store.conflicts().total).toBe(0); expect(store.value(plugin, "feature", "state").value).toEqual({ same: 7 }); expect(store.sources().total).toBeGreaterThanOrEqual(4); } finally { store.close(); }
  });

  it("keeps later real effects in recovery history without overwriting recovered variables", async () => {
    const later = await fixture("owner", { value: "later" }), recovered = await fixture("owner", { value: "snapshot" }), before = await sha(join(later, PLUGIN_RUNTIME_WORKSPACE));
    await mergePluginRuntimeWorkspace(later, recovered, "owner", "recovery-snapshot", { mode: "history-only" }); const store = (await PluginRuntimeStore.open(recovered, "owner", { readOnly: true }))!;
    try { expect(store.value(plugin, "feature", "state").value).toEqual({ value: "snapshot" }); const source = await openPluginRuntimeSource(recovered, "owner", store.sources().sources[0]!.sha256); try { expect(source.rpc("session-owner", "effect")?.state).toBe("dispatched"); expect(source.value(plugin, "feature", "state").value).toEqual({ value: "later" }); } finally { source.close(); } } finally { store.close(); }
    expect(await sha(join(later, PLUGIN_RUNTIME_WORKSPACE))).toBe(before);
  });

  it("keeps oversized inherited values intact/inert and pages before JSON parsing", async () => {
    const source = await fixture("source", { complete: "large🙂".repeat(15000) }), target = await root(); await rekeyPluginRuntimeWorkspace(source, "source", "target"); await mergePluginRuntimeWorkspace(source, target, "target", "source");
    const store = (await PluginRuntimeStore.open(target, "target", { readOnly: true }))!; try { expect(store.conflicts().total).toBe(1); const descriptor = store.sources().sources.find(row => row.owner === "source")!, archived = await openPluginRuntimeSource(target, "target", descriptor.sha256);
      try { const parse = vi.spyOn(JSON, "parse"), value = archived.valuePage(plugin, "feature", "state", 4096); expect(value.receipt!.bytes).toBeGreaterThan(65536); expect(parse).not.toHaveBeenCalled(); const page = archived.valueBytes(plugin, "feature", "state", 0, 1000); expect(page.nextOffset).toBe(1000); expect(page.sha256).toBe(value.receipt!.sha256); expect(parse).not.toHaveBeenCalled(); } finally { archived.close(); }
    } finally { store.close(); }
  });

  it("refuses corrupted exact archives and cancelled rekey without changing the source", async () => {
    const source = await fixture("source"), before = await sha(join(source, PLUGIN_RUNTIME_WORKSPACE)), controller = new AbortController(); controller.abort(); await expect(rekeyPluginRuntimeWorkspace(source, "source", "target", { signal: controller.signal })).rejects.toThrow(); expect(await sha(join(source, PLUGIN_RUNTIME_WORKSPACE))).toBe(before);
    await rekeyPluginRuntimeWorkspace(source, "source", "target"); const store = (await PluginRuntimeStore.open(source, "target", { readOnly: true }))!, descriptor = store.sources().sources[0]!; store.close();
    await writeFile(join(source, "plugin-runtime", "archives", `${descriptor.sha256}.sqlite3`), "tampered"); await expect(validatePluginRuntimeWorkspace(source, "target")).rejects.toThrow(/checksum/);
  });

  it("returns newest sessions first with explicit paging instead of hiding active sessions behind history", async () => {
    const path = await fixture("owner"), store = (await PluginRuntimeStore.open(path, "owner"))!; try { const original = store.sessionsPage().sessions[0]!; for (let i = 0; i < 30; i++) await store.session({ ...original, sessionId: `new-${i}`, state: "stopped" });
      await store.session({ ...original, sessionId: "currently-active", state: "active" }); const page = store.sessionsPage(undefined, 4096); expect(page.sessions[0]?.sessionId).toBe("currently-active"); expect(page.nextRowid).toBeDefined(); expect(page.total).toBe(32); const next = store.sessionsPage(page.nextRowid, 4096); expect(new Set([...page.sessions, ...next.sessions].map(row => row.sessionId)).size).toBe(page.sessions.length + next.sessions.length);
    } finally { store.close(); }
  });

  it("rechecks exact source SHA and inode after an opened inherited read before publishing its value", async () => {
    const path = await fixture("source"); await rekeyPluginRuntimeWorkspace(path, "source", "target"); const store = (await PluginRuntimeStore.open(path, "target", { readOnly: true }))!, descriptor = store.sources().sources[0]!; store.close();
    const source = await openPluginRuntimeSource(path, "target", descriptor.sha256); try { expect(source.value(plugin, "feature", "state").value).toEqual({ exact: "source" }); await writeFile(join(path, "plugin-runtime", "archives", `${descriptor.sha256}.sqlite3`), "replaced while open"); await expect(source.verifyRead()).rejects.toThrow(/changed/); } finally { source.close(); }
  });
});
