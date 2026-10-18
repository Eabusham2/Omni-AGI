import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { ClipboardOperationError, ClipboardService } from "../src/main/clipboardService";
import { ClipboardStore } from "../src/main/clipboardStore";
import { clipboardCanonical, clipboardHash } from "../src/main/clipboardFormat";
import { CLIPBOARD_FORMAT, CLIPBOARD_READ_PAGE_BYTES, type ClipboardCapabilities } from "../src/shared/clipboard";
import type { HostClipboardAdapter, HostClipboardContext, HostClipboardCursor, HostClipboardMutation, HostClipboardPage } from "../src/main/clipboardHostBridge";
import type { ToolPermissionLevel } from "../src/shared/types";

const roots: string[] = [], brainId = "77777777-7777-4777-8777-777777777777";
afterEach(async () => { await Promise.all(roots.splice(0).map(root => rm(root, { recursive: true, force: true }))); });

class FakeHostClipboard implements HostClipboardAdapter {
  text = "host fixture"; writes = 0; failAfterDispatch = false;
  async capabilities(): Promise<ClipboardCapabilities> { return { format: CLIPBOARD_FORMAT, scope: "host", implemented: true, adapter: "fake-host", contentTypes: ["text/plain"],
    maxReadPageBytes: CLIPBOARD_READ_PAGE_BYTES, readIsBounded: true, write: true, clear: true, authority: "current-host-permission-only", detail: "fixture" }; }
  async read(cursor: HostClipboardCursor | undefined, maximumBytes: number, context: HostClipboardContext): Promise<HostClipboardPage> {
    context.signal.throwIfAborted(); context.onSpawn(42001); const bytes = Buffer.from(this.text), after = cursor?.afterBytes ?? 0;
    if (cursor && cursor.prefixSha256 !== clipboardHash(bytes.subarray(0, after))) throw new Error("changed prefix");
    const page = bytes.subarray(after, after + maximumBytes), complete = after + page.length >= bytes.length;
    return { adapter: "fake-host", text: page.toString(), bytes: page.length, chunkSha256: clipboardHash(page), complete, capturedAt: new Date().toISOString(),
      ...(!complete ? { next: { afterBytes: after + page.length, prefixSha256: clipboardHash(bytes.subarray(0, after + page.length)) } } : {}) };
  }
  async write(text: string, context: HostClipboardContext): Promise<HostClipboardMutation> { await context.beforeDispatch?.(); context.signal.throwIfAborted(); context.onSpawn(42002);
    this.writes++; this.text = text; if (this.failAfterDispatch) throw new Error("fixture lost acknowledgement"); return { adapter: "fake-host", completedAt: new Date().toISOString() }; }
  clear(context: HostClipboardContext): Promise<HostClipboardMutation> { return this.write("", context); }
}

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "omni-clipboard-service-")); roots.push(root); const host = new FakeHostClipboard();
  const levels = new Map<string, ToolPermissionLevel>([["studio.clipboard", "auto"], ["system.clipboard", "off"]]), revisions = new Map<string, number>();
  const service = new ClipboardService({ brainDirectory: async () => root, assertBrainOwner: async id => { expect(id).toBe(brainId); }, requireActiveOwner: async () => undefined,
    currentPermission: async (_brain, toolId) => ({ level: levels.get(toolId)!, revision: String(revisions.get(toolId) ?? 0) }),
    checkResources: async () => undefined, reserveWrite: async () => undefined, host });
  return { root, host, service, set(tool: "studio.clipboard" | "system.clipboard", level: ToolPermissionLevel) { levels.set(tool, level); revisions.set(tool, (revisions.get(tool) ?? 0) + 1); service.permissionChanged(brainId, tool); } };
}

describe("permissioned instance and host clipboards", () => {
  it("keeps exact instance text/version history portable and request-idempotent", async () => {
    const f = await fixture(), signal = new AbortController().signal, text = `alpha-${"🙂".repeat(40)}-omega`;
    const written = await f.service.execute(brainId, "studio.clipboard", "write", { scope: "instance", contentType: "text/plain", text }, "instance-write", signal) as any;
    expect(written).toMatchObject({ state: "complete", scope: "instance", contentSha256: clipboardHash(text), writeMayHaveOccurred: true });
    const repeated = await f.service.execute(brainId, "studio.clipboard", "write", { scope: "instance", contentType: "text/plain", text }, "instance-write", signal) as any;
    expect(repeated.repeated).toBe(true); expect(repeated.versionId).toBe(written.versionId);
    let cursor: string | undefined, reconstructed = "";
    do { const page = await f.service.execute(brainId, "studio.clipboard", "read", { scope: "instance", maxBytes: 17, ...(cursor ? { cursor } : {}) }, `instance-read-${reconstructed.length}`, signal) as any;
      reconstructed += page.text; cursor = page.cursor; expect(page.chunkBytes).toBe(Buffer.byteLength(page.text)); } while (cursor);
    expect(reconstructed).toBe(text);
    const capabilities = await f.service.execute(brainId, "studio.clipboard", "capabilities", { scope: "instance" }, "instance-capabilities", signal) as any;
    expect(capabilities).toMatchObject({ scope: "instance", authority: "brain-portable-no-os-grant", currentVersionId: written.versionId });
    const cleared = await f.service.execute(brainId, "studio.clipboard", "clear", { scope: "instance" }, "instance-clear", signal) as any; expect(cleared.contentBytes).toBe(0);
    const empty = await f.service.execute(brainId, "studio.clipboard", "read", { scope: "instance" }, "instance-empty", signal) as any; expect(empty).toMatchObject({ text: "", complete: true });
  });

  it("keeps host scope Off until current policy allows it and never replays completed or uncertain effects", async () => {
    const f = await fixture(), signal = new AbortController().signal;
    await expect(f.service.execute(brainId, "system.clipboard", "read", { scope: "host" }, "host-off", signal)).rejects.toThrow(/authority|Off/i);
    f.set("system.clipboard", "ask");
    const read = await f.service.execute(brainId, "system.clipboard", "read", { scope: "host", maxBytes: 8 }, "host-read", signal) as any;
    expect(read).toMatchObject({ scope: "host", state: "partial", text: "host fix", provenance: { adapter: "fake-host" } });
    const written = await f.service.execute(brainId, "system.clipboard", "write", { scope: "host", contentType: "text/plain", text: "new host" }, "host-write", signal) as any;
    expect(written.state).toBe("complete"); expect(f.host.writes).toBe(1);
    const repeated = await f.service.execute(brainId, "system.clipboard", "write", { scope: "host", contentType: "text/plain", text: "new host" }, "host-write", signal) as any;
    expect(repeated.repeated).toBe(true); expect(f.host.writes).toBe(1);
    f.host.failAfterDispatch = true;
    try { await f.service.execute(brainId, "system.clipboard", "clear", { scope: "host" }, "host-uncertain", signal); throw new Error("expected uncertainty"); }
    catch (error) { expect(error).toBeInstanceOf(ClipboardOperationError); expect((error as ClipboardOperationError).receipt).toMatchObject({ state: "write-uncertain", writeMayHaveOccurred: true, retrySafe: false }); }
    expect(f.host.writes).toBe(2);
    await expect(f.service.execute(brainId, "system.clipboard", "clear", { scope: "host" }, "host-uncertain", signal)).rejects.toBeInstanceOf(ClipboardOperationError);
    expect(f.host.writes).toBe(2);
  });

  it("rejects scope substitution and makes recovered dispatched host effects nonreplayable", async () => {
    const f = await fixture(), signal = new AbortController().signal;
    await expect(f.service.execute(brainId, "studio.clipboard", "read", { scope: "host" }, "scope-confusion", signal)).rejects.toThrow(/accepts only scope=instance/i);
    const store = await ClipboardStore.open(f.root, brainId); try { const contentSha256 = clipboardHash("lost"), unsigned = { scope: "host" as const, action: "write" as const, contentSha256, contentBytes: 4 },
      base = { ...unsigned, requestSha256: clipboardHash(clipboardCanonical(unsigned)) }; store.prepareHostEffect("lost-host-write", base); store.dispatched("lost-host-write"); } finally { store.close(); }
    await f.service.forget(brainId); f.set("system.clipboard", "ask");
    await expect(f.service.execute(brainId, "system.clipboard", "write", { scope: "host", contentType: "text/plain", text: "lost" }, "lost-host-write", signal)).rejects.toThrow(/retained|replay|acknowledgement/i);
    expect(f.host.writes).toBe(0);
  });
});
